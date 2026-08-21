#!/usr/bin/env python3
"""Teacher-fit a shared dense MLP base plus layer-private 2:4 residuals."""
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
from examples.nanogpt.analyze_layer_private_2of4_mlp_teacher_fit import (
    gather_values,
    magnitude_indices,
    random_indices,
    unpack_weight,
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
    passes,
)
from examples.nanogpt.analyze_layer_private_2of4_mlp_teacher_fit import fit_variant
from examples.nanogpt.train import (
    TokenBatchSource,
    fixed_eval_indices_digest,
    make_fixed_eval_indices,
)


PLAN_SCHEMA = "mai_shared_base_private_2of4_mlp_teacher_fit_plan_v1"
RESULT_SCHEMA = "mai_shared_base_private_2of4_mlp_teacher_fit_result_v1"
SPARSE_RESULT = Path(
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_layer_private_2of4_mlp_teacher_fit_result.json"
)
LAYER_AXIS_RESULT = Path(
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_layer_axis_basis7_mlp_teacher_fit_result.json"
)


def support_mask(indices: Tensor) -> Tensor:
    """Expand packed pair indices to a boolean mask over the input axis."""

    if indices.ndim != 4 or indices.shape[-1] != 2:
        raise ValueError("expected [layer,output,group,2] indices")
    mask = torch.zeros(
        *indices.shape[:-1], 4, device=indices.device, dtype=torch.bool
    )
    mask.scatter_(-1, indices.long(), True)
    return mask.reshape(indices.shape[0], indices.shape[1], -1)


def least_squares_base(weights: Tensor, indices: Tensor) -> Tensor:
    """Fit each shared coordinate from layers that omit the sparse residual."""

    if weights.ndim != 3 or weights.shape[-1] % 4:
        raise ValueError("expected [layer,output,input] weights")
    mask = support_mask(indices).to(weights.device)
    omitted = ~mask
    count = omitted.sum(dim=0)
    numerator = (weights.float() * omitted).sum(dim=0)
    fallback = weights.float().mean(dim=0)
    return torch.where(count > 0, numerator / count.clamp_min(1), fallback)


class SharedBasePrivate24MLP(nn.Module):
    """One unrestricted shared MLP pair plus private packed 2:4 residuals."""

    def __init__(
        self,
        *,
        base_fc: Tensor,
        base_proj: Tensor,
        values_fc: Tensor,
        indices_fc: Tensor,
        values_proj: Tensor,
        indices_proj: Tensor,
    ) -> None:
        super().__init__()
        self.base_fc = nn.Parameter(base_fc.detach().float().clone())
        self.base_proj = nn.Parameter(base_proj.detach().float().clone())
        self.values_fc = nn.Parameter(values_fc.detach().float().clone())
        self.values_proj = nn.Parameter(values_proj.detach().float().clone())
        self.register_buffer("indices_fc", indices_fc.detach().to(torch.uint8))
        self.register_buffer("indices_proj", indices_proj.detach().to(torch.uint8))
        layers, hidden, groups, selected = self.values_fc.shape
        if selected != 2 or self.indices_fc.shape != self.values_fc.shape:
            raise ValueError("invalid packed c_fc residual shape")
        if self.base_fc.shape != (hidden, groups * 4):
            raise ValueError("shared c_fc base does not match residual shape")
        if self.values_proj.shape != (layers, groups * 4, hidden // 4, 2):
            raise ValueError("invalid paired packed c_proj residual shape")
        if self.indices_proj.shape != self.values_proj.shape:
            raise ValueError("c_proj residual indices do not match values")
        if self.base_proj.shape != (groups * 4, hidden):
            raise ValueError("shared c_proj base does not match residual shape")

    @property
    def layers(self) -> int:
        return int(self.values_fc.shape[0])

    def weights(self, layer: int) -> tuple[Tensor, Tensor]:
        layer = int(layer)
        return (
            self.base_fc
            + unpack_weight(self.values_fc[layer], self.indices_fc[layer]),
            self.base_proj
            + unpack_weight(self.values_proj[layer], self.indices_proj[layer]),
        )

    def forward_layer(self, layer: int, values: Tensor) -> Tensor:
        c_fc, c_proj = self.weights(layer)
        return F.linear(F.gelu(F.linear(values, c_fc)), c_proj)

    def forward_selected(self, values: Tensor, layer_indices: Tensor) -> Tensor:
        if values.ndim != 5 or values.shape[1] != layer_indices.numel():
            raise ValueError("unexpected selected-layer input shape")
        return torch.stack(
            [
                self.forward_layer(int(layer), values[:, position])
                for position, layer in enumerate(layer_indices)
            ],
            dim=1,
        )


class InstalledSharedBasePrivate24MLP(nn.Module):
    """Materialized frozen layer view used only for fixed-CE measurement."""

    def __init__(self, family: SharedBasePrivate24MLP, layer: int) -> None:
        super().__init__()
        c_fc, c_proj = family.weights(int(layer))
        self.register_buffer("c_fc_weight", c_fc.detach().clone())
        self.register_buffer("c_proj_weight", c_proj.detach().clone())
        self.residual_conditioned_output_slope = None
        self.conditioned_output_gate_source = "residual"

    def forward(self, values: Tensor) -> Tensor:
        return F.linear(
            F.gelu(F.linear(values, self.c_fc_weight)), self.c_proj_weight
        )


def initial_family(
    teacher: nn.Module, *, variant: str, support_seed: int
) -> SharedBasePrivate24MLP:
    dense_fc = torch.stack(
        [block.mlp.c_fc.weight.detach().float() for block in teacher.transformer.h]
    )
    dense_proj = torch.stack(
        [block.mlp.c_proj.weight.detach().float() for block in teacher.transformer.h]
    )
    for block in teacher.transformer.h:
        if block.mlp.c_fc.bias is not None or block.mlp.c_proj.bias is not None:
            raise ValueError("registered shared-base sparse family is bias free")
    mean_fc, mean_proj = dense_fc.mean(dim=0), dense_proj.mean(dim=0)
    if variant == "residual_magnitude_ceiling":
        indices_fc = torch.stack(
            [magnitude_indices(weight - mean_fc) for weight in dense_fc]
        )
        indices_proj = torch.stack(
            [magnitude_indices(weight - mean_proj) for weight in dense_proj]
        )
    elif variant == "fixed_random":
        indices_fc = torch.stack(
            [
                random_indices(tuple(weight.shape), seed=support_seed + layer * 2)
                for layer, weight in enumerate(dense_fc)
            ]
        )
        indices_proj = torch.stack(
            [
                random_indices(
                    tuple(weight.shape), seed=support_seed + layer * 2 + 1
                )
                for layer, weight in enumerate(dense_proj)
            ]
        )
    else:
        raise ValueError(f"unknown shared-base sparse variant: {variant}")
    base_fc = least_squares_base(dense_fc, indices_fc)
    base_proj = least_squares_base(dense_proj, indices_proj)
    values_fc = torch.stack(
        [
            gather_values(weight - base_fc, index)
            for weight, index in zip(dense_fc, indices_fc)
        ]
    )
    values_proj = torch.stack(
        [
            gather_values(weight - base_proj, index)
            for weight, index in zip(dense_proj, indices_proj)
        ]
    )
    return SharedBasePrivate24MLP(
        base_fc=base_fc,
        base_proj=base_proj,
        values_fc=values_fc,
        indices_fc=indices_fc,
        values_proj=values_proj,
        indices_proj=indices_proj,
    )


def family_from_state(
    state: dict[str, Tensor], device: str
) -> SharedBasePrivate24MLP:
    return SharedBasePrivate24MLP(
        base_fc=state["base_fc"],
        base_proj=state["base_proj"],
        values_fc=state["values_fc"],
        indices_fc=state["indices_fc"],
        values_proj=state["values_proj"],
        indices_proj=state["indices_proj"],
    ).to(device)


def function_measurement(
    module: SharedBasePrivate24MLP,
    *,
    holdout: dict[str, Tensor],
    teacher: nn.Module,
    measurement: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    output_summary, output_rows = output_metrics(module, holdout)
    jvp_summary, jvp_rows = jvp_metrics(
        module,
        holdout,
        teacher=teacher,
        directions=int(measurement["input_jvp_directions"]),
        seed=int(measurement["input_jvp_seed"]),
        device=device,
    )
    return {
        "summary": {"output": output_summary, "input_jvp": jvp_summary},
        "holdout_output_rows": output_rows,
        "holdout_input_jvp_rows": jvp_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-steps", type=int, default=10)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan.get("schema_version") != PLAN_SCHEMA or args.device != "cuda":
        raise ValueError("unexpected plan schema or device")
    causal = plan["causal_basis"]
    if sha256_file(SPARSE_RESULT) != causal["sparse_result_sha256"]:
        raise ValueError("sparse causal result identity mismatch")
    if sha256_file(LAYER_AXIS_RESULT) != causal["layer_axis_result_sha256"]:
        raise ValueError("layer-axis causal result identity mismatch")
    identity = plan["identities"]
    teacher_path = Path(identity["dense_teacher_checkpoint"]["path"])
    candidate_path = Path(identity["state_bank_checkpoint"]["path"])
    if sha256_file(teacher_path) != identity["dense_teacher_checkpoint"]["sha256"]:
        raise ValueError("teacher checkpoint identity mismatch")
    if sha256_file(candidate_path) != identity["state_bank_checkpoint"]["sha256"]:
        raise ValueError("state-bank checkpoint identity mismatch")
    data_dir = Path("/mnt/ssd-data/orj/MappingNetworks/data/finewebedu_20b")
    if sha256_file(data_dir / "manifest.json") != identity["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest identity mismatch")
    protocol, measurement = plan["fit_protocol"], plan["measurement"]
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    teacher = load_model(teacher_path, args.device)
    candidate = load_model(candidate_path, args.device)
    validate_core_configs(candidate, teacher)

    def collect(seed: int, count: int) -> dict[str, dict[int, Tensor]]:
        batches = fixed_validation_batches(
            data_dir,
            int(protocol["token_batch_size"]),
            teacher.config.block_size,
            int(count),
            int(seed),
        )
        return {
            "teacher": collect_stratified_inputs(
                teacher,
                batches,
                sample_cap=int(protocol["sample_cap_per_layer_per_bank"]),
                seed=int(seed),
                device=args.device,
            ),
            "candidate": collect_stratified_inputs(
                candidate,
                batches,
                sample_cap=int(protocol["sample_cap_per_layer_per_bank"]),
                seed=int(seed),
                device=args.device,
            ),
        }

    print("collecting fit and holdout banks", flush=True)
    fit = build_data(
        banks=collect(int(protocol["fit_token_seed"]), int(protocol["fit_batches"])),
        teacher=teacher,
        relative_rms=float(protocol["local_perturbation_relative_rms"]),
        seed=int(protocol["fit_token_seed"]),
        device=args.device,
    )
    holdout = build_data(
        banks=collect(
            int(protocol["holdout_token_seed"]), int(protocol["holdout_batches"])
        ),
        teacher=teacher,
        relative_rms=float(protocol["local_perturbation_relative_rms"]),
        seed=int(protocol["holdout_token_seed"]),
        device=args.device,
    )
    variant_names = [row["name"] for row in plan["family"]["variants"]]
    if args.preflight_only:
        module = initial_family(
            teacher,
            variant="fixed_random",
            support_seed=int(protocol["support_seed"]),
        ).to(args.device)
        row, _ = fit_variant(
            module,
            fit=fit,
            holdout=holdout,
            protocol=protocol,
            seed=int(protocol["algorithm_seeds"]["fixed_random"]),
            steps_override=int(args.preflight_steps),
        )
        seconds = row["wall_seconds"] / max(int(args.preflight_steps), 1)
        print(
            json.dumps(
                {
                    "preflight": "complete",
                    "seconds_per_fit_step_including_preflight_evaluations": seconds,
                    "conservative_estimated_total_fit_seconds": seconds
                    * int(protocol["steps_per_variant"])
                    * len(variant_names),
                    "maximum_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return

    variant_results: dict[str, dict[str, Any]] = {}
    variant_states: dict[str, dict[str, Tensor]] = {}
    for variant in variant_names:
        print(json.dumps({"starting_variant": variant}), flush=True)
        module = initial_family(
            teacher,
            variant=variant,
            support_seed=int(protocol["support_seed"]),
        ).to(args.device)
        if sum(parameter.numel() for parameter in module.parameters()) != int(
            plan["family"]["total_trainable_values"]
        ):
            raise ValueError("shared-base sparse parameter accounting mismatch")
        row, state = fit_variant(
            module,
            fit=fit,
            holdout=holdout,
            protocol=protocol,
            seed=int(protocol["algorithm_seeds"][variant]),
        )
        row.update(
            function_measurement(
                module,
                holdout=holdout,
                teacher=teacher,
                measurement=measurement,
                device=args.device,
            )
        )
        variant_results[variant] = row
        variant_states[variant] = state
        del module
        torch.cuda.empty_cache()

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
    gates = plan["frozen_gates"]
    for variant in variant_names:
        family = family_from_state(variant_states[variant], args.device)
        splice = load_model(teacher_path, args.device)
        for layer in range(family.layers):
            splice.transformer.h[layer].mlp = InstalledSharedBasePrivate24MLP(
                family, layer
            )
        value = evaluate_fixed_ce(
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
        row = variant_results[variant]
        reduction = row["objective_reduction_fraction"]
        healthy = bool(
            row["finite"]
            and (
                row["final_fit_objective"]["total"] <= 0.5
                or (reduction is not None and reduction >= 0.5)
            )
        )
        row.update(
            {
                "fixed_validation_cross_entropy": value,
                "gap": value - teacher_ce,
                "optimization_healthy": healthy,
            }
        )
        row["passes"] = passes(row, value - teacher_ce, gates)
        del splice, family
        torch.cuda.empty_cache()

    magnitude = variant_results["residual_magnitude_ceiling"]
    random = variant_results["fixed_random"]
    if not magnitude["optimization_healthy"] or not random["optimization_healthy"]:
        classification = "OPTIMIZATION_INCONCLUSIVE"
    elif not magnitude["passes"]:
        classification = "SHARED_BASE_PRIVATE_TWO_OF_FOUR_CAPACITY_FAIL"
    elif not random["passes"]:
        classification = "SUPPORT_ACQUISITION_REQUIRED"
    else:
        classification = "FIXED_RANDOM_SHARED_BASE_PRIVATE_TWO_OF_FOUR_PASS"
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "fitted_states.pt"
    torch.save(
        {
            "schema_version": "mai_shared_base_private_2of4_mlp_state_v1",
            "variants": variant_states,
        },
        state_path,
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "classification": classification,
        "repository_commit": git_head(Path(__file__).resolve().parents[2]),
        "plan": {"path": str(args.plan), "sha256": sha256_file(args.plan)},
        "identities": identity,
        "variant_results": variant_results,
        "teacher_validation_cross_entropy": teacher_ce,
        "fixed_eval_indices_sha256": digest,
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
                "variants": {
                    name: {
                        "fit": row["final_fit_objective"],
                        "holdout": row["holdout_objective"],
                        "reduction": row["objective_reduction_fraction"],
                        "healthy": row["optimization_healthy"],
                        "summary": row["summary"],
                        "candidate_ce": row["fixed_validation_cross_entropy"],
                        "ce_gap": row["gap"],
                        "passes": row["passes"],
                    }
                    for name, row in variant_results.items()
                },
                "teacher_ce": teacher_ce,
                "result": str(result_path),
                "result_sha256": sha256_file(result_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
