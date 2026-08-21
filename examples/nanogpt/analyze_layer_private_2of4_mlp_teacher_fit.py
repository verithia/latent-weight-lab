#!/usr/bin/env python3
"""Teacher-fit layer-private packed 2:4 sparse MLPs."""
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
from examples.nanogpt.analyze_depth_shared_top1_mlp_teacher_fit import objective_parts
from examples.nanogpt.analyze_layer_axis_basis7_mlp_teacher_fit import full_objective
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
from examples.nanogpt.train import (
    TokenBatchSource,
    fixed_eval_indices_digest,
    make_fixed_eval_indices,
)


PLAN_SCHEMA = "mai_layer_private_2of4_mlp_teacher_fit_plan_v1"
RESULT_SCHEMA = "mai_layer_private_2of4_mlp_teacher_fit_result_v1"
LAYER_AXIS_RESULT = Path(
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_layer_axis_basis7_mlp_teacher_fit_result.json"
)
PAIRS = torch.tensor(
    ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)),
    dtype=torch.uint8,
)
RANDOM_MASK_SEED = 20260934


def magnitude_indices(weight: Tensor) -> Tensor:
    """Stable top-two absolute entries in every consecutive group of four."""

    if weight.ndim != 2 or weight.shape[-1] % 4:
        raise ValueError("2:4 weights require a matrix with input divisible by four")
    grouped = weight.detach().float().reshape(weight.shape[0], -1, 4)
    order = torch.argsort(grouped.abs(), dim=-1, descending=True, stable=True)
    return order[..., :2].sort(dim=-1).values.to(torch.uint8)


def random_indices(shape: tuple[int, int], *, seed: int) -> Tensor:
    out_features, in_features = (int(value) for value in shape)
    if in_features % 4:
        raise ValueError("2:4 weights require input divisible by four")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    choices = torch.randint(
        len(PAIRS),
        (out_features, in_features // 4),
        generator=generator,
    )
    return PAIRS.index_select(0, choices.reshape(-1)).reshape(
        out_features, in_features // 4, 2
    )


def gather_values(weight: Tensor, indices: Tensor) -> Tensor:
    grouped = weight.detach().float().reshape(weight.shape[0], -1, 4)
    return grouped.gather(-1, indices.to(grouped.device).long())


def unpack_weight(values: Tensor, indices: Tensor) -> Tensor:
    if values.shape != indices.shape or values.shape[-1] != 2:
        raise ValueError("packed values and indices must have matching [...,2] shape")
    grouped = torch.zeros(
        *values.shape[:-1], 4, device=values.device, dtype=values.dtype
    ).scatter(-1, indices.to(values.device).long(), values)
    return grouped.reshape(values.shape[0], -1)


class LayerPrivate24MLP(nn.Module):
    """Independent packed 2:4 c_fc/c_proj matrices for every layer."""

    def __init__(
        self,
        *,
        values_fc: Tensor,
        indices_fc: Tensor,
        values_proj: Tensor,
        indices_proj: Tensor,
    ) -> None:
        super().__init__()
        self.values_fc = nn.Parameter(values_fc.detach().float().clone())
        self.values_proj = nn.Parameter(values_proj.detach().float().clone())
        self.register_buffer("indices_fc", indices_fc.detach().to(torch.uint8))
        self.register_buffer("indices_proj", indices_proj.detach().to(torch.uint8))
        layers, hidden, groups, selected = self.values_fc.shape
        if selected != 2 or self.indices_fc.shape != self.values_fc.shape:
            raise ValueError("invalid packed c_fc shape")
        if self.values_proj.shape != (layers, groups * 4, hidden // 4, 2):
            raise ValueError("invalid paired packed c_proj shape")
        if self.indices_proj.shape != self.values_proj.shape:
            raise ValueError("c_proj indices do not match values")

    @property
    def layers(self) -> int:
        return int(self.values_fc.shape[0])

    def weights(self, layer: int) -> tuple[Tensor, Tensor]:
        layer = int(layer)
        return (
            unpack_weight(self.values_fc[layer], self.indices_fc[layer]),
            unpack_weight(self.values_proj[layer], self.indices_proj[layer]),
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


class InstalledPacked24MLP(nn.Module):
    """Materialized frozen layer view used only for fixed-CE measurement."""

    def __init__(self, family: LayerPrivate24MLP, layer: int) -> None:
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
    teacher: nn.Module, *, variant: str, random_seed: int
) -> LayerPrivate24MLP:
    values_fc, indices_fc, values_proj, indices_proj = [], [], [], []
    for layer, block in enumerate(teacher.transformer.h):
        mlp = block.mlp
        if mlp.c_fc.bias is not None or mlp.c_proj.bias is not None:
            raise ValueError("registered sparse family is bias free")
        c_fc, c_proj = mlp.c_fc.weight, mlp.c_proj.weight
        if variant == "magnitude_ceiling":
            fc_indices = magnitude_indices(c_fc)
            proj_indices = magnitude_indices(c_proj)
        elif variant == "fixed_random":
            fc_indices = random_indices(
                tuple(c_fc.shape), seed=int(random_seed) + layer * 2
            )
            proj_indices = random_indices(
                tuple(c_proj.shape), seed=int(random_seed) + layer * 2 + 1
            )
        else:
            raise ValueError(f"unknown sparse variant: {variant}")
        values_fc.append(gather_values(c_fc, fc_indices))
        indices_fc.append(fc_indices)
        values_proj.append(gather_values(c_proj, proj_indices))
        indices_proj.append(proj_indices)
    return LayerPrivate24MLP(
        values_fc=torch.stack(values_fc),
        indices_fc=torch.stack(indices_fc),
        values_proj=torch.stack(values_proj),
        indices_proj=torch.stack(indices_proj),
    )


def fit_variant(
    module: LayerPrivate24MLP,
    *,
    fit: dict[str, Tensor],
    holdout: dict[str, Tensor],
    protocol: dict[str, Any],
    seed: int,
    steps_override: int | None = None,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    optimizer = torch.optim.Adam(
        module.parameters(),
        lr=float(protocol["learning_rate"]),
        weight_decay=float(protocol["weight_decay"]),
    )
    generator = torch.Generator(device=fit["clean"].device).manual_seed(int(seed))
    steps = int(
        protocol["steps_per_variant"] if steps_override is None else steps_override
    )
    initial = full_objective(module, fit)
    best_loss = initial["total"]
    best_state = {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }
    finite, started = True, time.time()
    for step in range(steps):
        layer_indices = torch.randperm(
            module.layers, generator=generator, device=fit["clean"].device
        )[: int(protocol["layers_per_update"])]
        row_indices = torch.randint(
            fit["clean"].shape[2],
            (
                min(
                    int(protocol["row_batch_size_per_layer_per_bank"]),
                    fit["clean"].shape[2],
                ),
            ),
            generator=generator,
            device=fit["clean"].device,
        )
        variants = fit["variants"].index_select(1, layer_indices)
        variants = variants.index_select(-2, row_indices)
        targets = fit["targets"].index_select(1, layer_indices)
        targets = targets.index_select(-2, row_indices)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            parts = objective_parts(
                module.forward_selected(variants, layer_indices), targets
            )
        if not torch.isfinite(parts["total"]):
            finite = False
            break
        parts["total"].backward()
        torch.nn.utils.clip_grad_norm_(
            module.parameters(), float(protocol["gradient_clip_norm"])
        )
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == steps:
            evaluated = full_objective(module, fit)
            if evaluated["total"] < best_loss:
                best_loss = evaluated["total"]
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in module.state_dict().items()
                }
            print(
                json.dumps(
                    {
                        "fit_step": step + 1,
                        "fit_steps": steps,
                        "minibatch_value": float(parts["value"].detach()),
                        "minibatch_direction": float(parts["direction"].detach()),
                        "evaluated_total": evaluated["total"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    module.load_state_dict(best_state)
    final = (
        full_objective(module, fit)
        if finite
        else {"value": math.inf, "direction": math.inf, "total": math.inf}
    )
    holdout_value = (
        full_objective(module, holdout)
        if finite
        else {"value": math.inf, "direction": math.inf, "total": math.inf}
    )
    return {
        "finite": finite,
        "initial_fit_objective": initial,
        "final_fit_objective": final,
        "objective_reduction_fraction": (
            1.0 - final["total"] / max(initial["total"], 1e-30)
            if finite
            else None
        ),
        "holdout_objective": holdout_value,
        "wall_seconds": time.time() - started,
    }, {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def family_from_state(
    state: dict[str, Tensor], device: str
) -> LayerPrivate24MLP:
    return LayerPrivate24MLP(
        values_fc=state["values_fc"],
        indices_fc=state["indices_fc"],
        values_proj=state["values_proj"],
        indices_proj=state["indices_proj"],
    ).to(device)


def function_measurement(
    module: LayerPrivate24MLP,
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
    if sha256_file(LAYER_AXIS_RESULT) != plan["causal_basis"]["layer_axis_result_sha256"]:
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
            random_seed=RANDOM_MASK_SEED,
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
            teacher, variant=variant, random_seed=RANDOM_MASK_SEED
        ).to(args.device)
        if sum(parameter.numel() for parameter in module.parameters()) != int(
            plan["family"]["trainable_values"]
        ):
            raise ValueError("packed trainable parameter accounting mismatch")
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
            splice.transformer.h[layer].mlp = InstalledPacked24MLP(family, layer)
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

    magnitude = variant_results["magnitude_ceiling"]
    random = variant_results["fixed_random"]
    if not magnitude["optimization_healthy"] or not random["optimization_healthy"]:
        classification = "OPTIMIZATION_INCONCLUSIVE"
    elif not magnitude["passes"]:
        classification = "TWO_OF_FOUR_CAPACITY_FAIL"
    elif not random["passes"]:
        classification = "SUPPORT_ACQUISITION_REQUIRED"
    else:
        classification = "FIXED_RANDOM_TWO_OF_FOUR_PASS"
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "fitted_states.pt"
    torch.save(
        {
            "schema_version": "mai_layer_private_2of4_mlp_state_v1",
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
