#!/usr/bin/env python3
"""Add one layer-private residual pair-code stage to c_fc only."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from examples.nanogpt.analyze_cproj_manifold import load_model
from examples.nanogpt.analyze_layer_private_pair_vq_mlp import (
    euclidean_pair_quantize,
)
from examples.nanogpt.analyze_layer_private_product_quantized_mlp import (
    InstalledQuantizedDenseMLP,
    QuantizedDenseMLPFamily,
    stack_optional_dense_gains,
)
from examples.nanogpt.analyze_mlp_cproj_bilateral_endpoint_fixed_eval import (
    evaluate_fixed_ce,
)
from examples.nanogpt.analyze_pair_vq_side_localization import decode_pair_vq_side
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


PLAN_SCHEMA = "mai_cfc_residual_pair_vq_mlp_plan_v1"
RESULT_SCHEMA = "mai_cfc_residual_pair_vq_mlp_result_v1"


def _matrix_metric(target: Tensor, decoded: Tensor) -> dict[str, float]:
    squared_error = float((decoded.float() - target.float()).square().sum())
    squared_target = float(target.float().square().sum())
    return {
        "squared_error": squared_error,
        "squared_target_norm": squared_target,
        "weight_energy_recovery": 1.0
        - squared_error / max(squared_target, 1e-30),
    }


@torch.no_grad()
def build_repaired_family(
    teacher: nn.Module,
    *,
    first_stage: dict[str, Tensor],
    quantizer: dict[str, Any],
    matrix_limit: int | None = None,
) -> tuple[QuantizedDenseMLPFamily | None, dict[str, Any], dict[str, Tensor]]:
    blocks = list(teacher.transformer.h)
    fc_shapes = [block.mlp.c_fc.weight.shape for block in blocks]
    proj_shapes = [block.mlp.c_proj.weight.shape for block in blocks]
    dense_fc = torch.stack([block.mlp.c_fc.weight.detach().float() for block in blocks])
    dense_proj = torch.stack(
        [block.mlp.c_proj.weight.detach().float() for block in blocks]
    )
    first_fc = decode_pair_vq_side(
        codebooks=first_stage["c_fc_codebook"],
        codes=first_stage["c_fc_codes"],
        shapes=fc_shapes,
        device=str(dense_fc.device),
    )
    first_proj = decode_pair_vq_side(
        codebooks=first_stage["c_proj_codebook"],
        codes=first_stage["c_proj_codes"],
        shapes=proj_shapes,
        device=str(dense_proj.device),
    )

    residual_codebooks, residual_codes, final_fc = [], [], []
    residual_rows = []
    limit = len(blocks) if matrix_limit is None else min(int(matrix_limit), len(blocks))
    for layer in range(limit):
        residual = dense_fc[layer] - first_fc[layer]
        codebook, codes, decoded_residual, diagnostics = euclidean_pair_quantize(
            residual,
            vector_length=int(quantizer["vector_length"]),
            codebook_size=int(quantizer["codebook_size"]),
            sample_vectors=int(quantizer["sample_vectors_per_matrix"]),
            iterations=int(quantizer["lloyd_iterations"]),
            assignment_chunk=int(quantizer["assignment_chunk_vectors"]),
            seed=int(quantizer["algorithm_seed"]) + layer,
        )
        repaired = first_fc[layer] + decoded_residual
        final_metric = _matrix_metric(dense_fc[layer], repaired)
        residual_codebooks.append(codebook)
        residual_codes.append(codes)
        final_fc.append(repaired)
        residual_rows.append(
            {
                "layer": layer,
                "side": "c_fc",
                "residual_stage_energy_recovery": diagnostics[
                    "weight_energy_recovery"
                ],
                "residual_stage_relative_residual_rms": diagnostics[
                    "relative_residual_rms"
                ],
                **final_metric,
            }
        )
        print(
            json.dumps(
                {
                    "residual_quantized_matrix": layer + 1,
                    "total_residual_matrices": len(blocks),
                    "layer": layer,
                    "side": "c_fc",
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if matrix_limit is not None:
        return None, {"c_fc_residual_matrices": residual_rows}, {}

    final_fc_tensor = torch.stack(final_fc)
    matrix_rows = residual_rows + [
        {"layer": layer, "side": "c_proj", **_matrix_metric(dense_proj[layer], first_proj[layer])}
        for layer in range(len(blocks))
    ]
    squared_error = sum(row["squared_error"] for row in matrix_rows)
    squared_target = sum(row["squared_target_norm"] for row in matrix_rows)
    matrix_metrics = {
        "matrices": matrix_rows,
        "aggregate_weight_energy_recovery": 1.0
        - squared_error / max(squared_target, 1e-30),
        "minimum_matrix_weight_energy_recovery": min(
            row["weight_energy_recovery"] for row in matrix_rows
        ),
        "minimum_c_fc_weight_energy_recovery": min(
            row["weight_energy_recovery"] for row in residual_rows
        ),
        "minimum_c_proj_weight_energy_recovery": min(
            row["weight_energy_recovery"]
            for row in matrix_rows
            if row["side"] == "c_proj"
        ),
    }
    pre_gain, output_log_gain = stack_optional_dense_gains(blocks)
    family = QuantizedDenseMLPFamily(
        c_fc=final_fc_tensor,
        c_proj=first_proj,
        pre_gain=pre_gain,
        output_log_gain=output_log_gain,
    ).to(dense_fc.device)
    packed = {
        "c_fc_codebook": first_stage["c_fc_codebook"],
        "c_fc_codes": first_stage["c_fc_codes"],
        "c_fc_residual_codebook": torch.stack(residual_codebooks),
        "c_fc_residual_codes": torch.stack(residual_codes),
        "c_proj_codebook": first_stage["c_proj_codebook"],
        "c_proj_codes": first_stage["c_proj_codes"],
    }
    return family, matrix_metrics, packed


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
    if sha256_file(Path(causal["side_localization_result"])) != causal[
        "side_localization_result_sha256"
    ]:
        raise ValueError("side-localization causal result identity mismatch")
    identity = plan["identities"]
    for key in (
        "dense_teacher_checkpoint",
        "state_bank_checkpoint",
        "first_stage_pair_vq_state",
    ):
        if sha256_file(Path(identity[key]["path"])) != identity[key]["sha256"]:
            raise ValueError(f"identity mismatch: {key}")
    data_dir = Path("/mnt/ssd-data/orj/MappingNetworks/data/finewebedu_20b")
    if sha256_file(data_dir / "manifest.json") != identity["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest identity mismatch")
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    teacher_path = Path(identity["dense_teacher_checkpoint"]["path"])
    teacher = load_model(teacher_path, args.device)
    first_stage = torch.load(
        identity["first_stage_pair_vq_state"]["path"],
        map_location="cpu",
        weights_only=False,
    )

    if args.preflight_only:
        matrix_started = time.time()
        build_repaired_family(
            teacher,
            first_stage=first_stage,
            quantizer=plan["residual_quantizer"],
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
                    * 12
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

    candidate = load_model(Path(identity["state_bank_checkpoint"]["path"]), args.device)
    validate_core_configs(candidate, teacher)
    family, matrix_metrics, packed = build_repaired_family(
        teacher,
        first_stage=first_stage,
        quantizer=plan["residual_quantizer"],
    )
    assert family is not None
    accounting = plan["accounting"]
    packed_bytes = sum(tensor.numel() * tensor.element_size() for tensor in packed.values())
    if packed_bytes != int(accounting["persistent_matrix_state_bytes"]):
        raise ValueError("packed c_fc residual pair-VQ byte accounting mismatch")

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
        "CFC_RESIDUAL_PAIR_VQ_MLP_PASS"
        if passed
        else "CFC_RESIDUAL_PAIR_VQ_MLP_FAIL"
    )
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "cfc_residual_pair_vq_state.pt"
    torch.save(
        {
            "schema_version": "mai_cfc_residual_pair_vq_mlp_state_v1",
            "vector_length": int(plan["residual_quantizer"]["vector_length"]),
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
                    key: value
                    for key, value in matrix_metrics.items()
                    if key != "matrices"
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
