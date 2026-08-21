#!/usr/bin/env python3
"""Quantize every dense teacher MLP with layer-private product codebooks."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from examples.nanogpt.analyze_cproj_manifold import load_model
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


PLAN_SCHEMA = "mai_layer_private_product_quantized_mlp_plan_v1"
RESULT_SCHEMA = "mai_layer_private_product_quantized_mlp_result_v1"


@torch.no_grad()
def signed_spherical_product_quantize(
    weight: Tensor,
    *,
    block_length: int,
    codebook_size: int,
    sample_vectors: int,
    iterations: int,
    assignment_chunk: int,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Any]]:
    """Return BF16 codebook/amplitudes, uint8 codes, and decoded FP32 weight."""

    if weight.numel() % int(block_length):
        raise ValueError("weight entries are not divisible by block length")
    vectors = weight.detach().float().reshape(-1, int(block_length))
    norms = vectors.norm(dim=1).clamp_min(1e-30)
    directions = vectors / norms[:, None]
    generator = torch.Generator(device=weight.device).manual_seed(int(seed))
    sample_count = min(int(sample_vectors), directions.shape[0])
    sample_indices = torch.randperm(
        directions.shape[0], generator=generator, device=weight.device
    )[:sample_count]
    samples = directions.index_select(0, sample_indices)
    if int(codebook_size) > sample_count or int(codebook_size) > 256:
        raise ValueError("codebook size exceeds sample or uint8 capacity")
    initialization = torch.randperm(
        sample_count, generator=generator, device=weight.device
    )[: int(codebook_size)]
    codebook = samples.index_select(0, initialization).clone()
    codebook = F.normalize(codebook, dim=1)
    dead_counts = []
    for _ in range(int(iterations)):
        similarity = samples @ codebook.T
        codes = similarity.abs().argmax(dim=1)
        signed = torch.sign(similarity.gather(1, codes[:, None]).squeeze(1))
        signed = torch.where(signed == 0, torch.ones_like(signed), signed)
        accum = torch.zeros_like(codebook)
        accum.index_add_(0, codes, samples * signed[:, None])
        counts = torch.bincount(codes, minlength=int(codebook_size))
        live = counts > 0
        dead_counts.append(int((~live).sum()))
        updated = F.normalize(accum[live], dim=1)
        codebook[live] = updated

    code_parts, amplitude_parts = [], []
    squared_error, squared_target = 0.0, 0.0
    for start in range(0, vectors.shape[0], int(assignment_chunk)):
        stop = min(start + int(assignment_chunk), vectors.shape[0])
        direction = directions[start:stop]
        similarity = direction @ codebook.T
        codes = similarity.abs().argmax(dim=1)
        selected = codebook.index_select(0, codes)
        amplitudes = (vectors[start:stop] * selected).sum(dim=1)
        code_parts.append(codes.to(torch.uint8).cpu())
        amplitude_parts.append(amplitudes.to(torch.bfloat16).cpu())
    codes_cpu = torch.cat(code_parts)
    amplitudes_cpu = torch.cat(amplitude_parts)
    codebook_cpu = codebook.to(torch.bfloat16).cpu()
    decoded_vectors = (
        amplitudes_cpu.to(weight.device, dtype=torch.float32)[:, None]
        * codebook_cpu.to(weight.device, dtype=torch.float32).index_select(
            0, codes_cpu.to(weight.device, dtype=torch.long)
        )
    )
    squared_error = float((decoded_vectors - vectors).square().sum())
    squared_target = float(vectors.square().sum())
    recovery = 1.0 - squared_error / max(squared_target, 1e-30)
    decoded = decoded_vectors.reshape_as(weight).float()
    diagnostics = {
        "shape": list(weight.shape),
        "entries": int(weight.numel()),
        "blocks": int(vectors.shape[0]),
        "weight_energy_recovery": recovery,
        "relative_residual_rms": math.sqrt(
            squared_error / max(squared_target, 1e-30)
        ),
        "squared_error": squared_error,
        "squared_target_norm": squared_target,
        "sample_vectors": sample_count,
        "dead_codewords_by_iteration": dead_counts,
        "terminal_used_codewords": int(torch.unique(codes_cpu).numel()),
        "amplitude_mean_absolute": float(amplitudes_cpu.float().abs().mean()),
        "amplitude_maximum_absolute": float(amplitudes_cpu.float().abs().max()),
    }
    return codebook_cpu, codes_cpu, amplitudes_cpu, decoded, diagnostics


class QuantizedDenseMLPFamily(nn.Module):
    def __init__(
        self,
        *,
        c_fc: Tensor,
        c_proj: Tensor,
        pre_gain: Tensor,
        output_log_gain: Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("c_fc", c_fc.detach().float().clone())
        self.register_buffer("c_proj", c_proj.detach().float().clone())
        self.register_buffer("pre_gain", pre_gain.detach().float().clone())
        self.register_buffer(
            "output_log_gain", output_log_gain.detach().float().clone()
        )
        layers, hidden, width = self.c_fc.shape
        if self.c_proj.shape != (layers, width, hidden):
            raise ValueError("quantized matrix pairs do not match")
        if self.pre_gain.shape != (layers, hidden):
            raise ValueError("pre-GELU gain shape mismatch")
        if self.output_log_gain.shape != (layers, width):
            raise ValueError("output gain shape mismatch")

    @property
    def layers(self) -> int:
        return int(self.c_fc.shape[0])

    def forward_layer(self, layer: int, values: Tensor) -> Tensor:
        layer = int(layer)
        hidden = F.gelu(F.linear(values, self.c_fc[layer]) * self.pre_gain[layer])
        output = F.linear(hidden, self.c_proj[layer])
        return output * self.output_log_gain[layer].exp()


class InstalledQuantizedDenseMLP(nn.Module):
    def __init__(self, family: QuantizedDenseMLPFamily, layer: int) -> None:
        super().__init__()
        self.family = family
        self.layer = int(layer)
        self.residual_conditioned_output_slope = None
        self.conditioned_output_gate_source = "residual"

    def forward(self, values: Tensor) -> Tensor:
        return self.family.forward_layer(self.layer, values)


def stack_optional_dense_gains(blocks: list[nn.Module]) -> tuple[Tensor, Tensor]:
    """Materialize identity gains for dense MLPs that encode them as ``None``."""

    pre_gain, output_log_gain = [], []
    for block in blocks:
        hidden, width = block.mlp.c_fc.weight.shape
        pre = block.mlp.pregelu_gain
        if pre is None:
            pre = torch.ones(
                hidden,
                device=block.mlp.c_fc.weight.device,
                dtype=block.mlp.c_fc.weight.dtype,
            )
        output = block.mlp.residual_output_log_gain
        if output is None:
            output = torch.zeros(
                width,
                device=block.mlp.c_fc.weight.device,
                dtype=block.mlp.c_fc.weight.dtype,
            )
        else:
            output = output * block.mlp.residual_output_gain_scale
        pre_gain.append(pre)
        output_log_gain.append(output)
    return torch.stack(pre_gain), torch.stack(output_log_gain)


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
        "c_fc_amplitudes": [],
        "c_proj_codebook": [],
        "c_proj_codes": [],
        "c_proj_amplitudes": [],
    }
    rows, completed = [], 0
    for layer, block in enumerate(teacher.transformer.h):
        layer_decoded: dict[str, Tensor] = {}
        for side in ("c_fc", "c_proj"):
            if matrix_limit is not None and completed >= int(matrix_limit):
                return None, {"matrices": rows}, {}
            weight = getattr(block.mlp, side).weight
            codebook, codes, amplitudes, decoded, diagnostics = (
                signed_spherical_product_quantize(
                    weight,
                    block_length=int(quantizer["block_length"]),
                    codebook_size=int(quantizer["codebook_size"]),
                    sample_vectors=int(quantizer["sample_vectors_per_matrix"]),
                    iterations=int(quantizer["lloyd_iterations"]),
                    assignment_chunk=int(quantizer["assignment_chunk_vectors"]),
                    seed=int(quantizer["algorithm_seed"]) + layer * 10 + completed,
                )
            )
            state[f"{side}_codebook"].append(codebook)
            state[f"{side}_codes"].append(codes)
            state[f"{side}_amplitudes"].append(amplitudes)
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
    weighted_recovery = 1.0 - squared_error / max(squared_target, 1e-30)
    diagnostics = {
        "matrices": rows,
        "aggregate_weight_energy_recovery": weighted_recovery,
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
    for key in ("fullrank_coordinate_result", "extreme_residual_vq_result"):
        if sha256_file(Path(causal[key])) != causal[f"{key}_sha256"]:
            raise ValueError(f"causal result identity mismatch: {key}")
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
                    "maximum_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
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
        raise ValueError("packed product-quantized state byte accounting mismatch")

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
        "LAYER_PRIVATE_PRODUCT_QUANTIZED_MLP_PASS"
        if passed
        else "LAYER_PRIVATE_PRODUCT_QUANTIZED_MLP_FAIL"
    )
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "product_quantized_state.pt"
    torch.save(
        {
            "schema_version": "mai_layer_private_product_quantized_mlp_state_v1",
            "block_length": int(plan["quantizer"]["block_length"]),
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
