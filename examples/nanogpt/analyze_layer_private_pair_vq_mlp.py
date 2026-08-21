#!/usr/bin/env python3
"""Quantize every dense teacher MLP with layer-private Euclidean pair codes."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from examples.nanogpt.analyze_cproj_manifold import load_model
from examples.nanogpt.analyze_layer_private_product_quantized_mlp import (
    InstalledQuantizedDenseMLP,
    QuantizedDenseMLPFamily,
    stack_optional_dense_gains,
)
from examples.nanogpt.analyze_mlp_cproj_bilateral_endpoint_fixed_eval import (
    evaluate_fixed_ce,
)
from examples.nanogpt.analyze_residual_compatibility import fixed_validation_batches
from examples.nanogpt.analyze_shared_mlp_endpoint_function import (
    sha256_file,
    validate_core_configs,
)
from examples.nanogpt.analyze_shared_mlp_exact_family_teacher_fit import (
    atomic_json,
    collect_stratified_inputs,
    git_head,
)
from examples.nanogpt.analyze_shared_trunk_private_ridge_teacher_fit import (
    build_data,
    jvp_metrics,
    output_metrics,
)
from examples.nanogpt.train import (
    TokenBatchSource,
    fixed_eval_indices_digest,
    make_fixed_eval_indices,
)


PLAN_SCHEMA = "mai_layer_private_pair_vq_mlp_plan_v1"
RESULT_SCHEMA = "mai_layer_private_pair_vq_mlp_result_v1"


def _squared_distances(values: Tensor, codebook: Tensor) -> Tensor:
    return (
        values.square().sum(dim=1, keepdim=True)
        + codebook.square().sum(dim=1)[None, :]
        - 2.0 * values @ codebook.T
    ).clamp_min_(0.0)


@torch.no_grad()
def euclidean_pair_quantize(
    weight: Tensor,
    *,
    vector_length: int,
    codebook_size: int,
    sample_vectors: int,
    iterations: int,
    assignment_chunk: int,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor, dict[str, Any]]:
    """Return a BF16 codebook, uint8 codes, decoded FP32 weight, and metrics."""

    if int(vector_length) != 2 or int(codebook_size) != 256:
        raise ValueError("this preregistered oracle requires 2D/uint8 pair codes")
    if weight.numel() % int(vector_length):
        raise ValueError("weight entries are not divisible by vector length")
    vectors = weight.detach().float().reshape(-1, int(vector_length))
    generator = torch.Generator(device=weight.device).manual_seed(int(seed))
    sample_count = min(int(sample_vectors), vectors.shape[0])
    sample_indices = torch.randperm(
        vectors.shape[0], generator=generator, device=weight.device
    )[:sample_count]
    samples = vectors.index_select(0, sample_indices)

    # A deterministic 16 x 16 marginal-quantile grid covers both Gaussian-like
    # bulk and tails much more evenly than 256 random seeds. Lloyd updates then
    # remove the independence assumption using the joint 2D sample.
    bins = int(math.isqrt(int(codebook_size)))
    if bins * bins != int(codebook_size):
        raise ValueError("pair-code codebook must have a square cardinality")
    probabilities = (
        torch.arange(bins, device=weight.device, dtype=torch.float32) + 0.5
    ) / bins
    first = torch.quantile(samples[:, 0], probabilities)
    second = torch.quantile(samples[:, 1], probabilities)
    grid_first, grid_second = torch.meshgrid(first, second, indexing="ij")
    codebook = torch.stack(
        (grid_first.reshape(-1), grid_second.reshape(-1)), dim=1
    )

    dead_counts = []
    for _ in range(int(iterations)):
        sample_codes = _squared_distances(samples, codebook).argmin(dim=1)
        accum = torch.zeros_like(codebook)
        accum.index_add_(0, sample_codes, samples)
        counts = torch.bincount(sample_codes, minlength=int(codebook_size))
        live = counts > 0
        dead_counts.append(int((~live).sum()))
        codebook[live] = accum[live] / counts[live, None]

    code_parts = []
    for start in range(0, vectors.shape[0], int(assignment_chunk)):
        stop = min(start + int(assignment_chunk), vectors.shape[0])
        codes = _squared_distances(vectors[start:stop], codebook).argmin(dim=1)
        code_parts.append(codes.to(torch.uint8).cpu())
    codes_cpu = torch.cat(code_parts)
    codebook_cpu = codebook.to(torch.bfloat16).cpu()
    decoded_vectors = codebook_cpu.to(
        weight.device, dtype=torch.float32
    ).index_select(0, codes_cpu.to(weight.device, dtype=torch.long))
    squared_error = float((decoded_vectors - vectors).square().sum())
    squared_target = float(vectors.square().sum())
    recovery = 1.0 - squared_error / max(squared_target, 1e-30)
    diagnostics = {
        "shape": list(weight.shape),
        "entries": int(weight.numel()),
        "vectors": int(vectors.shape[0]),
        "weight_energy_recovery": recovery,
        "relative_residual_rms": math.sqrt(
            squared_error / max(squared_target, 1e-30)
        ),
        "squared_error": squared_error,
        "squared_target_norm": squared_target,
        "sample_vectors": sample_count,
        "dead_codewords_by_iteration": dead_counts,
        "terminal_used_codewords": int(torch.unique(codes_cpu).numel()),
        "codebook_mean_norm": float(codebook_cpu.float().norm(dim=1).mean()),
        "codebook_maximum_norm": float(codebook_cpu.float().norm(dim=1).max()),
    }
    return codebook_cpu, codes_cpu, decoded_vectors.reshape_as(weight), diagnostics


@torch.no_grad()
def quantize_teacher(
    teacher: nn.Module,
    *,
    quantizer: dict[str, Any],
    matrix_limit: int | None = None,
) -> tuple[QuantizedDenseMLPFamily | None, dict[str, Any], dict[str, Tensor]]:
    decoded_fc, decoded_proj = [], []
    state: dict[str, list[Tensor]] = {
        "c_fc_codebook": [],
        "c_fc_codes": [],
        "c_proj_codebook": [],
        "c_proj_codes": [],
    }
    rows, completed = [], 0
    for layer, block in enumerate(teacher.transformer.h):
        layer_decoded: dict[str, Tensor] = {}
        for side in ("c_fc", "c_proj"):
            if matrix_limit is not None and completed >= int(matrix_limit):
                return None, {"matrices": rows}, {}
            weight = getattr(block.mlp, side).weight
            codebook, codes, decoded, diagnostics = euclidean_pair_quantize(
                weight,
                vector_length=int(quantizer["vector_length"]),
                codebook_size=int(quantizer["codebook_size"]),
                sample_vectors=int(quantizer["sample_vectors_per_matrix"]),
                iterations=int(quantizer["lloyd_iterations"]),
                assignment_chunk=int(quantizer["assignment_chunk_vectors"]),
                seed=int(quantizer["algorithm_seed"]) + layer * 10 + completed,
            )
            state[f"{side}_codebook"].append(codebook)
            state[f"{side}_codes"].append(codes)
            layer_decoded[side] = decoded
            rows.append({"layer": layer, "side": side, **diagnostics})
            completed += 1
            print(
                json.dumps(
                    {
                        "quantized_matrix": completed,
                        "total_matrices": int(teacher.config.n_layer) * 2,
                        "layer": layer,
                        "side": side,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        decoded_fc.append(layer_decoded["c_fc"])
        decoded_proj.append(layer_decoded["c_proj"])
    pre_gain, output_log_gain = stack_optional_dense_gains(
        list(teacher.transformer.h)
    )
    family = QuantizedDenseMLPFamily(
        c_fc=torch.stack(decoded_fc),
        c_proj=torch.stack(decoded_proj),
        pre_gain=pre_gain,
        output_log_gain=output_log_gain,
    ).to(decoded_fc[0].device)
    packed = {key: torch.stack(value) for key, value in state.items()}
    squared_error = sum(row["squared_error"] for row in rows)
    squared_target = sum(row["squared_target_norm"] for row in rows)
    diagnostics = {
        "matrices": rows,
        "aggregate_weight_energy_recovery": 1.0
        - squared_error / max(squared_target, 1e-30),
        "minimum_matrix_weight_energy_recovery": min(
            row["weight_energy_recovery"] for row in rows
        ),
    }
    return family, diagnostics, packed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-matrices", type=int, default=2)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan.get("schema_version") != PLAN_SCHEMA or args.device != "cuda":
        raise ValueError("unexpected plan schema or device")
    causal = plan["causal_basis"]
    if sha256_file(Path(causal["product_quantized_result"])) != causal[
        "product_quantized_result_sha256"
    ]:
        raise ValueError("product-quantized causal result identity mismatch")
    identity = plan["identities"]
    teacher_path = Path(identity["dense_teacher_checkpoint"]["path"])
    candidate_path = Path(identity["state_bank_checkpoint"]["path"])
    data_dir = Path("/mnt/ssd-data/orj/MappingNetworks/data/finewebedu_20b")
    if sha256_file(teacher_path) != identity["dense_teacher_checkpoint"]["sha256"]:
        raise ValueError("dense teacher checkpoint identity mismatch")
    if sha256_file(candidate_path) != identity["state_bank_checkpoint"]["sha256"]:
        raise ValueError("state bank checkpoint identity mismatch")
    if sha256_file(data_dir / "manifest.json") != identity["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest identity mismatch")
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    teacher = load_model(teacher_path, args.device)

    if args.preflight_only:
        matrix_started = time.time()
        quantize_teacher(
            teacher,
            quantizer=plan["quantizer"],
            matrix_limit=int(args.preflight_matrices),
        )
        elapsed = time.time() - matrix_started
        print(
            json.dumps(
                {
                    "preflight": "complete",
                    "matrices": int(args.preflight_matrices),
                    "wall_seconds": elapsed,
                    "conservative_projected_quantization_seconds": elapsed
                    * 24
                    / max(int(args.preflight_matrices), 1),
                    "maximum_cuda_memory_bytes": int(
                        torch.cuda.max_memory_allocated()
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return

    candidate = load_model(candidate_path, args.device)
    validate_core_configs(candidate, teacher)
    family, matrix_metrics, packed = quantize_teacher(
        teacher, quantizer=plan["quantizer"]
    )
    assert family is not None
    accounting = plan["accounting"]
    packed_bytes = sum(tensor.numel() * tensor.element_size() for tensor in packed.values())
    if packed_bytes != int(accounting["persistent_matrix_state_bytes"]):
        raise ValueError("packed pair-VQ state byte accounting mismatch")

    measurement = plan["measurement"]
    batches = fixed_validation_batches(
        data_dir,
        int(measurement["state_bank_token_batch_size"]),
        teacher.config.block_size,
        int(measurement["state_bank_batches"]),
        int(measurement["state_bank_seed"]),
    )
    banks = {
        "teacher": collect_stratified_inputs(
            teacher,
            batches,
            sample_cap=int(measurement["state_bank_sample_cap_per_layer"]),
            seed=int(measurement["state_bank_seed"]),
            device=args.device,
        ),
        "candidate": collect_stratified_inputs(
            candidate,
            batches,
            sample_cap=int(measurement["state_bank_sample_cap_per_layer"]),
            seed=int(measurement["state_bank_seed"]),
            device=args.device,
        ),
    }
    holdout = build_data(
        banks=banks,
        teacher=teacher,
        relative_rms=float(measurement["local_perturbation_relative_rms"]),
        seed=int(measurement["state_bank_seed"]),
        device=args.device,
    )
    output_summary, output_rows = output_metrics(family, holdout)
    jvp_summary, jvp_rows = jvp_metrics(
        family,
        holdout,
        teacher=teacher,
        directions=int(measurement["input_jvp_directions"]),
        seed=int(measurement["input_jvp_seed"]),
        device=args.device,
    )
    fixed = make_fixed_eval_indices(
        data_dir,
        int(measurement["fixed_eval_batch_size"]),
        int(measurement["fixed_eval_block_size"]),
        int(measurement["fixed_eval_batches"]),
        int(measurement["fixed_eval_seed"]),
    )
    digest = fixed_eval_indices_digest(fixed)
    if digest != identity["fixed_eval_indices_sha256"]:
        raise ValueError("fixed evaluation digest mismatch")
    source = TokenBatchSource(data_dir)
    teacher_ce = evaluate_fixed_ce(
        teacher,
        data_dir=data_dir,
        fixed_indices=fixed,
        split="val",
        eval_iters=int(measurement["fixed_eval_batches"]),
        eval_batch_size=int(measurement["fixed_eval_batch_size"]),
        block_size=int(measurement["fixed_eval_block_size"]),
        device=args.device,
        dtype="bfloat16",
        source=source,
    )
    splice = load_model(teacher_path, args.device)
    for layer in range(family.layers):
        splice.transformer.h[layer].mlp = InstalledQuantizedDenseMLP(family, layer)
    candidate_ce = evaluate_fixed_ce(
        splice,
        data_dir=data_dir,
        fixed_indices=fixed,
        split="val",
        eval_iters=int(measurement["fixed_eval_batches"]),
        eval_batch_size=int(measurement["fixed_eval_batch_size"]),
        block_size=int(measurement["fixed_eval_block_size"]),
        device=args.device,
        dtype="bfloat16",
        source=source,
    )
    gap = candidate_ce - teacher_ce
    gates = plan["frozen_gates"]
    passed = bool(
        matrix_metrics["aggregate_weight_energy_recovery"]
        >= gates["minimum_aggregate_matrix_energy_recovery"]
        and matrix_metrics["minimum_matrix_weight_energy_recovery"]
        >= gates["minimum_every_matrix_energy_recovery"]
        and output_summary["mean_explained_target_energy"]
        >= gates["minimum_mean_output_recovery"]
        and output_summary["minimum_explained_target_energy"]
        >= gates["minimum_worst_output_recovery"]
        and jvp_summary["mean_explained_target_energy"]
        >= gates["minimum_mean_input_jvp_recovery"]
        and jvp_summary["minimum_explained_target_energy"]
        >= gates["minimum_worst_input_jvp_recovery"]
        and gap <= gates["maximum_fixed_validation_cross_entropy_gap"]
        and accounting["persistent_bf16_storage_compression"]
        >= gates["minimum_persistent_bf16_storage_compression"]
        and accounting["continuous_coordinate_compression"]
        >= gates["minimum_continuous_coordinate_compression"]
    )
    classification = (
        "LAYER_PRIVATE_PAIR_VQ_MLP_PASS"
        if passed
        else "LAYER_PRIVATE_PAIR_VQ_MLP_FAIL"
    )
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "pair_vq_state.pt"
    torch.save(
        {
            "schema_version": "mai_layer_private_pair_vq_mlp_state_v1",
            "vector_length": int(plan["quantizer"]["vector_length"]),
            **packed,
        },
        state_path,
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "classification": classification,
        "repository_commit": git_head(Path(__file__).resolve().parents[2]),
        "plan": {"path": str(args.plan), "sha256": sha256_file(args.plan)},
        "identities": identity,
        "accounting": {**accounting, "measured_packed_state_bytes": packed_bytes},
        "matrix_metrics": matrix_metrics,
        "functional_metrics": {
            "summary": {"output": output_summary, "input_jvp": jvp_summary},
            "output_rows": output_rows,
            "input_jvp_rows": jvp_rows,
        },
        "teacher_validation_cross_entropy": teacher_ce,
        "candidate_validation_cross_entropy": candidate_ce,
        "fixed_validation_cross_entropy_gap": gap,
        "fixed_eval_indices_sha256": digest,
        "passed": passed,
        "state_artifact": {"path": str(state_path), "sha256": sha256_file(state_path)},
        "maximum_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "wall_seconds": time.time() - started,
    }
    result_path = args.output / "result.json"
    atomic_json(result_path, result)
    print(
        json.dumps(
            {
                "classification": classification,
                "matrix": {
                    "aggregate": matrix_metrics["aggregate_weight_energy_recovery"],
                    "minimum": matrix_metrics["minimum_matrix_weight_energy_recovery"],
                },
                "summary": result["functional_metrics"]["summary"],
                "teacher_ce": teacher_ce,
                "candidate_ce": candidate_ce,
                "ce_gap": gap,
                "passed": passed,
                "result": str(result_path),
                "result_sha256": sha256_file(result_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
