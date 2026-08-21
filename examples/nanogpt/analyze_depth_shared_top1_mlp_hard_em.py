#!/usr/bin/env python3
"""Hard-assignment alternating-minimization repair for the routed MLP pool."""
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
from examples.nanogpt.analyze_depth_shared_top1_mlp_teacher_fit import (
    DepthSharedTop1MLP,
    InstalledTop1MLP,
    evaluate_stage,
    family_from_state,
    full_objective,
    objective_parts,
)
from examples.nanogpt.analyze_mlp_cproj_bilateral_endpoint_fixed_eval import (
    evaluate_fixed_ce,
)
from examples.nanogpt.analyze_residual_compatibility import fixed_validation_batches
from examples.nanogpt.analyze_shared_mlp_endpoint_function import (
    module_jvp,
    pair_metrics,
    rademacher_tangent,
    sha256_file,
    summarize,
    validate_core_configs,
)
from examples.nanogpt.analyze_shared_mlp_exact_family_teacher_fit import (
    atomic_json,
    collect_stratified_inputs,
    git_head,
)
from examples.nanogpt.analyze_shared_trunk_private_ridge_teacher_fit import build_data
from examples.nanogpt.train import (
    TokenBatchSource,
    fixed_eval_indices_digest,
    make_fixed_eval_indices,
)


PLAN_SCHEMA = "mai_depth_shared_top1_mlp_hard_em_plan_v1"
RESULT_SCHEMA = "mai_depth_shared_top1_mlp_hard_em_result_v1"
PARENT_RESULT = Path(
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_depth_shared_top1_mlp_teacher_fit_result.json"
)


def expert_output(
    module: DepthSharedTop1MLP, layer: int, expert: int, values: Tensor
) -> Tensor:
    hidden = F.linear(values, module.expert_fc[int(expert)])
    hidden = F.gelu(hidden * module.pre_gain[int(layer)])
    output = F.linear(hidden, module.expert_proj[int(expert)])
    return output * module.output_log_gain[int(layer)].exp()


@torch.no_grad()
def assignment_costs(
    module: DepthSharedTop1MLP, data: dict[str, Tensor]
) -> Tensor:
    """Return separately normalized value-plus-direction cost [L,B,N,E]."""

    rows = []
    for layer in range(module.layers):
        values = data["variants"][:, layer]
        target = data["targets"][:, layer].float()
        outputs = torch.stack(
            [expert_output(module, layer, expert, values) for expert in range(module.experts)],
            dim=-2,
        ).float()
        value_target = target[0]
        value_error = (outputs[0] - value_target[..., None, :]).square().mean(-1)
        value_scale = value_target.square().mean(-1).clamp_min(1e-12)
        direction_target = target[1] - target[2]
        direction_output = outputs[1] - outputs[2]
        direction_error = (
            direction_output - direction_target[..., None, :]
        ).square().mean(-1)
        direction_scale = direction_target.square().mean(-1).clamp_min(1e-12)
        rows.append(
            value_error / value_scale[..., None]
            + direction_error / direction_scale[..., None]
        )
    return torch.stack(rows)


@torch.no_grad()
def oracle_assignments(
    module: DepthSharedTop1MLP, data: dict[str, Tensor]
) -> Tensor:
    return assignment_costs(module, data).argmin(dim=-1)


def selected_prediction(
    module: DepthSharedTop1MLP,
    variants: Tensor,
    layer_indices: Tensor,
    assignments: Tensor,
    row_indices: Tensor,
) -> Tensor:
    outputs = []
    for position, layer_tensor in enumerate(layer_indices):
        layer = int(layer_tensor)
        current = variants[:, position]
        all_experts = torch.stack(
            [
                expert_output(module, layer, expert, current)
                for expert in range(module.experts)
            ],
            dim=-2,
        )
        chosen = assignments[layer].index_select(-1, row_indices)
        gather = chosen[None, ..., None, None].expand(
            current.shape[0],
            *chosen.shape,
            1,
            current.shape[-1],
        )
        outputs.append(all_experts.gather(-2, gather).squeeze(-2))
    return torch.stack(outputs, dim=1)


def set_expert_trainable(module: DepthSharedTop1MLP) -> list[nn.Parameter]:
    for parameter in module.parameters():
        parameter.requires_grad_(True)
    module.router_weight.requires_grad_(False)
    module.router_bias.requires_grad_(False)
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def fit_experts(
    module: DepthSharedTop1MLP,
    *,
    data: dict[str, Tensor],
    assignments: Tensor,
    settings: dict[str, Any],
    rows_per_layer: int,
    layers_per_update: int,
    generator: torch.Generator,
    steps_override: int | None = None,
) -> dict[str, float]:
    parameters = set_expert_trainable(module)
    optimizer = torch.optim.Adam(parameters, lr=float(settings["learning_rate"]))
    steps = int(settings["steps_per_cycle"] if steps_override is None else steps_override)
    initial, final = None, None
    for step in range(steps):
        layer_indices = torch.randperm(
            module.layers, generator=generator, device=data["clean"].device
        )[: int(layers_per_update)]
        row_indices = torch.randint(
            data["clean"].shape[2],
            (min(int(rows_per_layer), data["clean"].shape[2]),),
            generator=generator,
            device=data["clean"].device,
        )
        variants = data["variants"].index_select(1, layer_indices)
        variants = variants.index_select(-2, row_indices)
        targets = data["targets"].index_select(1, layer_indices)
        targets = targets.index_select(-2, row_indices)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = selected_prediction(
                module, variants, layer_indices, assignments, row_indices
            )
            parts = objective_parts(prediction, targets)
        if not torch.isfinite(parts["total"]):
            raise RuntimeError("non-finite expert M-step")
        parts["total"].backward()
        torch.nn.utils.clip_grad_norm_(
            parameters, float(settings["gradient_clip_norm"])
        )
        optimizer.step()
        module.clamp_charts()
        value = float(parts["total"].detach())
        initial = value if initial is None else initial
        final = value
    return {"initial_minibatch_loss": float(initial), "final_minibatch_loss": float(final)}


def fit_router(
    module: DepthSharedTop1MLP,
    *,
    data: dict[str, Tensor],
    assignments: Tensor,
    settings: dict[str, Any],
    rows_per_layer: int,
    layers_per_update: int,
    generator: torch.Generator,
    steps_override: int | None = None,
) -> dict[str, float]:
    parameters = module.set_trainable(router_only=True)
    optimizer = torch.optim.Adam(parameters, lr=float(settings["learning_rate"]))
    steps = int(settings["steps_per_cycle"] if steps_override is None else steps_override)
    initial, final = None, None
    for _step in range(steps):
        layer_indices = torch.randperm(
            module.layers, generator=generator, device=data["clean"].device
        )[: int(layers_per_update)]
        row_indices = torch.randint(
            data["clean"].shape[2],
            (min(int(rows_per_layer), data["clean"].shape[2]),),
            generator=generator,
            device=data["clean"].device,
        )
        losses = []
        for layer_tensor in layer_indices:
            layer = int(layer_tensor)
            values = data["clean"][layer].index_select(-2, row_indices)
            target = assignments[layer].index_select(-1, row_indices)
            losses.append(
                F.cross_entropy(
                    module._logits(values, layer).reshape(-1, module.experts),
                    target.reshape(-1),
                )
            )
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite router M-step")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            parameters, float(settings["gradient_clip_norm"])
        )
        optimizer.step()
        value = float(loss.detach())
        initial = value if initial is None else initial
        final = value
    return {"initial_minibatch_loss": float(initial), "final_minibatch_loss": float(final)}


@torch.no_grad()
def router_assignments(
    module: DepthSharedTop1MLP, data: dict[str, Tensor]
) -> Tensor:
    return torch.stack(
        [
            module._logits(data["clean"][layer], layer).argmax(dim=-1)
            for layer in range(module.layers)
        ]
    )


def assignment_summary(
    assignments: Tensor, *, previous: Tensor | None = None
) -> dict[str, Any]:
    counts = torch.bincount(assignments.reshape(-1), minlength=4)
    return {
        "counts": [int(value) for value in counts.cpu()],
        "fractions": [float(value / assignments.numel()) for value in counts.cpu()],
        "change_fraction": (
            None
            if previous is None
            else float((assignments != previous).float().mean())
        ),
    }


@torch.no_grad()
def oracle_objective(
    module: DepthSharedTop1MLP,
    data: dict[str, Tensor],
    assignments: Tensor,
) -> dict[str, float]:
    indices = torch.arange(module.layers, device=data["clean"].device)
    rows = torch.arange(data["clean"].shape[2], device=data["clean"].device)
    prediction = selected_prediction(
        module, data["variants"], indices, assignments, rows
    )
    parts = objective_parts(prediction, data["targets"])
    return {key: float(value) for key, value in parts.items()}


def oracle_function_metrics(
    module: DepthSharedTop1MLP,
    data: dict[str, Tensor],
    assignments: Tensor,
    *,
    teacher: nn.Module,
    directions: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    output_records, jvp_records, output_rows, jvp_rows = [], [], [], []
    for layer in range(module.layers):
        teacher_mlp = teacher.transformer.h[layer].mlp
        for bank_index, bank in enumerate(("teacher", "candidate")):
            values = data["clean"][layer, bank_index]
            chosen = assignments[layer, bank_index]
            predictions = torch.stack(
                [expert_output(module, layer, expert, values) for expert in range(module.experts)],
                dim=-2,
            )
            gather = chosen[..., None, None].expand(
                *chosen.shape, 1, values.shape[-1]
            )
            prediction = predictions.gather(-2, gather).squeeze(-2)
            output_metric = pair_metrics(
                data["targets"][0, layer, bank_index].cpu(), prediction.cpu()
            )
            output_records.append(output_metric)
            output_rows.append({"layer": layer, "bank": bank, **output_metric})
            teacher_parts, candidate_parts = [], []
            for direction in range(int(directions)):
                tangent = rademacher_tangent(
                    values.shape,
                    device=device,
                    seed=int(seed) + layer * 1000 + bank_index * 100_000 + direction,
                )
                teacher_parts.append(module_jvp(teacher_mlp, values, tangent).cpu())
                expert_jvps = torch.stack(
                    [
                        module_jvp(
                            lambda x, expert=expert: expert_output(
                                module, layer, expert, x
                            ),
                            values,
                            tangent,
                        )
                        for expert in range(module.experts)
                    ],
                    dim=-2,
                )
                candidate_parts.append(
                    expert_jvps.gather(-2, gather).squeeze(-2).cpu()
                )
            jvp_metric = pair_metrics(
                torch.stack(teacher_parts), torch.stack(candidate_parts)
            )
            jvp_records.append(jvp_metric)
            jvp_rows.append({"layer": layer, "bank": bank, **jvp_metric})
    return {
        "summary": {
            "output": summarize(output_records),
            "input_jvp": summarize(jvp_records),
        },
        "output_rows": output_rows,
        "input_jvp_rows": jvp_rows,
    }


def function_gates_pass(summary: dict[str, Any], gates: dict[str, Any]) -> bool:
    output, jvp = summary["output"], summary["input_jvp"]
    return bool(
        output["mean_explained_target_energy"] >= gates["minimum_mean_output_recovery"]
        and output["minimum_explained_target_energy"] >= gates["minimum_worst_output_recovery"]
        and jvp["mean_explained_target_energy"] >= gates["minimum_mean_input_jvp_recovery"]
        and jvp["minimum_explained_target_energy"] >= gates["minimum_worst_input_jvp_recovery"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-steps", type=int, default=5)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan.get("schema_version") != PLAN_SCHEMA or args.device != "cuda":
        raise ValueError("unexpected plan schema or device")
    if sha256_file(PARENT_RESULT) != plan["causal_basis"]["parent_result_sha256"]:
        raise ValueError("parent result identity mismatch")
    parent_result = json.loads(PARENT_RESULT.read_text())
    parent_plan_path = Path(parent_result["identity"]["plan_path"])
    parent_plan = json.loads(parent_plan_path.read_text())
    identity = plan["identities"]
    for key in (
        "parent_state",
        "dense_teacher_checkpoint",
        "four_trunk_initialization_checkpoint",
        "state_bank_checkpoint",
    ):
        path = Path(identity[key]["path"])
        if sha256_file(path) != identity[key]["sha256"]:
            raise ValueError(f"identity mismatch: {key}")
    data_dir = Path("/mnt/ssd-data/orj/MappingNetworks/data/finewebedu_20b")
    if sha256_file(data_dir / "manifest.json") != identity["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest identity mismatch")
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    teacher = load_model(Path(identity["dense_teacher_checkpoint"]["path"]), args.device)
    initializer = load_model(
        Path(identity["four_trunk_initialization_checkpoint"]["path"]), args.device
    )
    candidate = load_model(Path(identity["state_bank_checkpoint"]["path"]), args.device)
    validate_core_configs(initializer, teacher)
    validate_core_configs(candidate, teacher)
    protocol = parent_plan["fit_protocol"]
    repair = plan["repair_protocol"]
    measurement = plan["measurement"]

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

    print("collecting unchanged parent fit and holdout banks", flush=True)
    fit = build_data(
        banks=collect(int(repair["fit_token_seed"]), int(protocol["fit_batches"])),
        teacher=teacher,
        relative_rms=float(protocol["local_perturbation_relative_rms"]),
        seed=int(repair["fit_token_seed"]),
        device=args.device,
    )
    holdout = build_data(
        banks=collect(int(repair["holdout_token_seed"]), int(protocol["holdout_batches"])),
        teacher=teacher,
        relative_rms=float(protocol["local_perturbation_relative_rms"]),
        seed=int(repair["holdout_token_seed"]),
        device=args.device,
    )
    parent_state = torch.load(
        identity["parent_state"]["path"], map_location="cpu", weights_only=True
    )
    state = parent_state["stages"][identity["parent_state"]["stage"]]
    module = family_from_state(state, args.device)
    if sum(parameter.numel() for parameter in module.parameters()) != int(
        plan["family"]["compact_parameters"]
    ):
        raise ValueError("compact parameter accounting mismatch")
    generator = torch.Generator(device=args.device).manual_seed(
        int(repair["algorithm_seed"])
    )

    if args.preflight_only:
        assignments = oracle_assignments(module, fit)
        cycle_started = time.time()
        fit_experts(
            module,
            data=fit,
            assignments=assignments,
            settings=repair["expert_m_step"],
            rows_per_layer=int(repair["row_batch_size_per_layer_per_bank"]),
            layers_per_update=int(repair["layers_per_update"]),
            generator=generator,
            steps_override=int(args.preflight_steps),
        )
        fit_router(
            module,
            data=fit,
            assignments=assignments,
            settings=repair["router_m_step"],
            rows_per_layer=int(repair["row_batch_size_per_layer_per_bank"]),
            layers_per_update=int(repair["layers_per_update"]),
            generator=generator,
            steps_override=int(args.preflight_steps),
        )
        seconds = time.time() - cycle_started
        print(
            json.dumps(
                {
                    "preflight": "complete",
                    "seconds_per_preflight_cycle": seconds,
                    "conservative_estimated_fit_seconds": seconds
                    * int(repair["cycles"])
                    * float(repair["expert_m_step"]["steps_per_cycle"])
                    / max(int(args.preflight_steps), 1),
                    "maximum_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return

    initial_routed = full_objective(
        module, fit, mode="hard_top1", temperature=1.0
    )
    cycles = []
    previous, best_state, best_loss = None, None, math.inf
    for cycle in range(int(repair["cycles"])):
        assignments = oracle_assignments(module, fit)
        assignment_row = assignment_summary(assignments, previous=previous)
        expert_row = fit_experts(
            module,
            data=fit,
            assignments=assignments,
            settings=repair["expert_m_step"],
            rows_per_layer=int(repair["row_batch_size_per_layer_per_bank"]),
            layers_per_update=int(repair["layers_per_update"]),
            generator=generator,
        )
        router_row = fit_router(
            module,
            data=fit,
            assignments=assignments,
            settings=repair["router_m_step"],
            rows_per_layer=int(repair["row_batch_size_per_layer_per_bank"]),
            layers_per_update=int(repair["layers_per_update"]),
            generator=generator,
        )
        routed = full_objective(
            module, fit, mode="hard_top1", temperature=1.0
        )
        oracle_value = oracle_objective(module, fit, assignments)
        router_fit = router_assignments(module, fit)
        accuracy = float((router_fit == assignments).float().mean())
        row = {
            "cycle": cycle + 1,
            "assignment": assignment_row,
            "expert_m_step": expert_row,
            "router_m_step": router_row,
            "router_assignment_accuracy": accuracy,
            "oracle_fit_objective": oracle_value,
            "hard_routed_fit_objective": routed,
        }
        cycles.append(row)
        if routed["total"] < best_loss:
            best_loss = routed["total"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in module.state_dict().items()
            }
        print(json.dumps(row, sort_keys=True), flush=True)
        previous = assignments
    if best_state is not None:
        module.load_state_dict(best_state)
    routed_fit = full_objective(module, fit, mode="hard_top1", temperature=1.0)
    routed_holdout = full_objective(
        module, holdout, mode="hard_top1", temperature=1.0
    )
    fit_oracle = oracle_assignments(module, fit)
    holdout_oracle = oracle_assignments(module, holdout)
    fit_router_assignments = router_assignments(module, fit)
    holdout_router_assignments = router_assignments(module, holdout)
    routed_metrics = evaluate_stage(
        module,
        holdout=holdout,
        teacher=teacher,
        measurement=measurement,
        device=args.device,
    )
    oracle_metrics = oracle_function_metrics(
        module,
        holdout,
        holdout_oracle,
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
    splice = load_model(Path(identity["dense_teacher_checkpoint"]["path"]), args.device)
    for layer in range(module.layers):
        splice.transformer.h[layer].mlp = InstalledTop1MLP(module, layer)
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
    reduction = 1.0 - routed_fit["total"] / 3.7679713517427444
    healthy = bool(math.isfinite(routed_fit["total"]) and reduction >= 0.25)
    gates = plan["frozen_gates"]
    routed_pass = bool(
        healthy
        and function_gates_pass(routed_metrics["summary"], gates)
        and candidate_ce - teacher_ce
        <= gates["maximum_fixed_validation_cross_entropy_gap"]
    )
    oracle_pass = function_gates_pass(oracle_metrics["summary"], gates)
    if routed_pass:
        classification = "HARD_EM_PASS"
    elif oracle_pass:
        classification = "ORACLE_GOOD_AFFINE_ROUTER_BAD"
    elif healthy:
        classification = "FOUR_EXPERT_TOP1_FAMILY_FAIL"
    else:
        classification = "OPTIMIZATION_INCONCLUSIVE"
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "fitted_state.pt"
    torch.save(
        {
            "schema_version": "mai_depth_shared_top1_mlp_hard_em_state_v1",
            "state": {
                key: value.detach().cpu().clone()
                for key, value in module.state_dict().items()
            },
        },
        state_path,
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "classification": classification,
        "repository_commit": git_head(Path(__file__).resolve().parents[2]),
        "plan": {"path": str(args.plan), "sha256": sha256_file(args.plan)},
        "identities": identity,
        "initial_hard_routed_fit_objective": initial_routed,
        "cycles": cycles,
        "final": {
            "hard_routed_fit_objective": routed_fit,
            "hard_routed_holdout_objective": routed_holdout,
            "objective_reduction_fraction": reduction,
            "optimization_healthy": healthy,
            "fit_router_accuracy": float(
                (fit_router_assignments == fit_oracle).float().mean()
            ),
            "holdout_router_accuracy": float(
                (holdout_router_assignments == holdout_oracle).float().mean()
            ),
            "fit_oracle_assignment": assignment_summary(fit_oracle),
            "holdout_oracle_assignment": assignment_summary(holdout_oracle),
            "routed_metrics": routed_metrics,
            "oracle_metrics": oracle_metrics,
            "teacher_validation_cross_entropy": teacher_ce,
            "candidate_validation_cross_entropy": candidate_ce,
            "validation_cross_entropy_gap": candidate_ce - teacher_ce,
            "routed_pass": routed_pass,
            "oracle_function_pass": oracle_pass,
        },
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
                "final": {
                    "hard_routed_fit_objective": routed_fit,
                    "hard_routed_holdout_objective": routed_holdout,
                    "objective_reduction_fraction": reduction,
                    "optimization_healthy": healthy,
                    "routed_summary": routed_metrics["summary"],
                    "oracle_summary": oracle_metrics["summary"],
                    "teacher_ce": teacher_ce,
                    "candidate_ce": candidate_ce,
                    "ce_gap": candidate_ce - teacher_ce,
                },
                "result": str(result_path),
                "result_sha256": sha256_file(result_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
