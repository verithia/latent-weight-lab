#!/usr/bin/env python3
"""Test a coupled solve over the deployed c_fc skew + shear topology.

The exact step-120 replay supplies one gradient, persisted Muon momentum,
functional context, and base model.  The deployed 64-stage skew parent and
24-stage symmetric-shear residual pairings are selected exactly as in
production.  Their 135,168 coordinates are then solved together in weight
Frobenius geometry rather than sequentially.  The resulting BF16 family
endpoint is normalized to the actual production c_fc Frobenius radius and
scored alone and beside the production c_proj update on untouched windows.
No checkpoint state is changed or written.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_mlp_cfc_exact_current_matcher import (
    exact_muon_update,
    file_sha256,
    fixed_batches,
    git_commit,
    load_model_and_optimizer,
)
from examples.nanogpt.analyze_mlp_dense_oracle_gap import (
    ExactVariantApplier,
    aggregate_direction_metrics,
    evaluate_candidates,
    family_fro,
    merge_updates,
    scale_family,
)
from examples.nanogpt.analyze_mlp_fixed_radius_capacity import (
    _owner_and_group,
    normalize_family_to_radius,
    production_muon_request,
    quantized_update,
    reconstruct_cproj_update,
)
from examples.nanogpt.analyze_mlp_joint_prospective_step import (
    _autocast,
    _gradient_norm,
    family_weights,
    historical_double_decay_update,
    optimizer_parameters,
)
from examples.nanogpt.analyze_mlp_joint_step_response_surface import (
    paired_comparison,
)
from examples.nanogpt.muon_matched_givens import (
    MuonFunctionalShear,
    MuonMatchedGivens,
    _weight_shear_permutations,
    apply_givens_flow,
    diagonal_metric_angles,
    fast_muon_matched_permutations,
    functional_coordinate_mix_update,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "nanogpt_mlp_cfc_mixed_joint_solve_v1"
CANDIDATE_ORDER = (
    "baseline",
    "production_cfc",
    "production_cproj",
    "production_joint",
    "dense_norm_cfc",
    "hybrid_norm_cfc",
    "mixed_joint_cfc",
    "hybrid_mixed_joint_cfc",
    "cproj64_only",
    "hybrid_cproj64",
)


def pairs_from_permutations(permutations: torch.Tensor) -> torch.Tensor:
    if permutations.ndim != 2 or permutations.shape[1] % 2:
        raise ValueError("permutations must be shaped [stages, even_width]")
    return permutations.reshape(-1, 2).to(dtype=torch.long)


def mixed_jvp(
    source: torch.Tensor,
    skew_pairs: torch.Tensor,
    shear_pairs: torch.Tensor,
    coordinates: torch.Tensor,
) -> torch.Tensor:
    """Apply all selected column-skew and column-shear tangents jointly."""
    skew_count = int(skew_pairs.shape[0])
    if coordinates.numel() != skew_count + int(shear_pairs.shape[0]):
        raise ValueError("coordinate count does not match mixed topology")
    result = torch.zeros_like(source, dtype=torch.float32)
    source_f = source.float()
    for pairs, values, left_sign in (
        (skew_pairs, coordinates[:skew_count], -1.0),
        (shear_pairs, coordinates[skew_count:], 1.0),
    ):
        if not pairs.numel():
            continue
        pairs = pairs.to(source.device)
        values = values.to(source.device, dtype=torch.float32).unsqueeze(0)
        left, right = pairs[:, 0], pairs[:, 1]
        result.index_add_(
            1,
            left,
            float(left_sign) * source_f.index_select(1, right) * values,
        )
        result.index_add_(
            1,
            right,
            source_f.index_select(1, left) * values,
        )
    return result


def mixed_vjp(
    source: torch.Tensor,
    skew_pairs: torch.Tensor,
    shear_pairs: torch.Tensor,
    cotangent: torch.Tensor,
) -> torch.Tensor:
    """Apply the exact transpose of :func:`mixed_jvp`."""
    source_f = source.float()
    cotangent_f = cotangent.float()
    values: list[torch.Tensor] = []
    for pairs, left_sign in ((skew_pairs, -1.0), (shear_pairs, 1.0)):
        if not pairs.numel():
            continue
        pairs = pairs.to(source.device)
        left, right = pairs[:, 0], pairs[:, 1]
        first = (
            source_f.index_select(1, right)
            * cotangent_f.index_select(1, left)
        ).sum(dim=0)
        second = (
            source_f.index_select(1, left)
            * cotangent_f.index_select(1, right)
        ).sum(dim=0)
        values.append(float(left_sign) * first + second)
    if not values:
        return torch.empty(0, device=source.device, dtype=torch.float32)
    return torch.cat(values)


def mixed_diagonal(
    source: torch.Tensor,
    skew_pairs: torch.Tensor,
    shear_pairs: torch.Tensor,
) -> torch.Tensor:
    column_energy = source.float().square().sum(dim=0)
    values = []
    for pairs in (skew_pairs, shear_pairs):
        if pairs.numel():
            pairs = pairs.to(source.device)
            values.append(
                column_energy[pairs[:, 0]] + column_energy[pairs[:, 1]]
            )
    return torch.cat(values).clamp_min(1e-30)


@torch.no_grad()
def solve_mixed_tangent(
    source: torch.Tensor,
    target: torch.Tensor,
    skew_pairs: torch.Tensor,
    shear_pairs: torch.Tensor,
    *,
    iterations: int,
    damping: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Solve damped mixed normal equations with diagonal-preconditioned CG."""
    if iterations <= 0 or not 0.0 < damping < 1.0:
        raise ValueError("invalid mixed tangent solver settings")
    diagonal = mixed_diagonal(source, skew_pairs, shear_pairs)
    rhs = mixed_vjp(source, skew_pairs, shear_pairs, target)

    def normal(values: torch.Tensor) -> torch.Tensor:
        tangent = mixed_jvp(source, skew_pairs, shear_pairs, values)
        return (
            mixed_vjp(source, skew_pairs, shear_pairs, tangent)
            + float(damping) * diagonal * values
        )

    coordinates = torch.zeros_like(rhs)
    residual = rhs.clone()
    preconditioned = residual / ((1.0 + float(damping)) * diagonal)
    direction = preconditioned.clone()
    residual_dot = torch.dot(residual, preconditioned)
    history = []
    rhs_norm = rhs.norm().clamp_min(1e-30)
    for iteration in range(int(iterations)):
        applied = normal(direction)
        alpha = residual_dot / torch.dot(direction, applied).clamp_min(1e-30)
        coordinates.add_(direction, alpha=alpha)
        residual.add_(applied, alpha=-alpha)
        next_preconditioned = residual / (
            (1.0 + float(damping)) * diagonal
        )
        next_dot = torch.dot(residual, next_preconditioned)
        history.append(
            {
                "iteration": iteration + 1,
                "relative_normal_residual": float(residual.norm() / rhs_norm),
            }
        )
        beta = next_dot / residual_dot.clamp_min(1e-30)
        direction.mul_(beta).add_(next_preconditioned)
        residual_dot = next_dot
    tangent = mixed_jvp(source, skew_pairs, shear_pairs, coordinates)
    return tangent, {
        "coordinates": int(coordinates.numel()),
        "skew_coordinates": int(skew_pairs.shape[0]),
        "shear_coordinates": int(shear_pairs.shape[0]),
        "iterations": int(iterations),
        "damping": float(damping),
        "coordinate_rms": float(coordinates.square().mean().sqrt()),
        "coordinate_max_abs": float(coordinates.abs().max()),
        "target_recovery": float(
            1.0
            - (target.float() - tangent).square().sum()
            / target.float().square().sum().clamp_min(1e-30)
        ),
        "history": history,
    }


@torch.no_grad()
def mixed_joint_cfc_update(
    weight: torch.Tensor,
    requested_update: torch.Tensor,
    selection_direction: torch.Tensor,
    *,
    parent_stages: int,
    shear_stages: int,
    neighbors: int,
    seed: int,
    cg_iterations: int,
    cg_damping: float,
    learning_rate: float,
    weight_decay: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Select the deployed topology, then jointly solve all its coordinates."""
    source = weight.float().T.contiguous()
    target = requested_update.float().T.contiguous()
    selection = selection_direction.float().T.contiguous()
    parent_permutations, parent_selection = fast_muon_matched_permutations(
        source,
        selection,
        stages=int(parent_stages),
        neighbors=int(neighbors),
        seed=int(seed),
    )
    parent_permutations = parent_permutations.to(source.device)
    parent_angles = diagonal_metric_angles(source, target, parent_permutations)
    after_parent = apply_givens_flow(source, parent_angles, parent_permutations)
    parent_update = after_parent - source
    shear_permutations, shear_selection = _weight_shear_permutations(
        after_parent,
        target - parent_update,
        stages=int(shear_stages),
        neighbors=int(neighbors),
        seed=int(seed) + 2,
    )
    shear_permutations = shear_permutations.to(source.device)
    tangent, solver = solve_mixed_tangent(
        source,
        target,
        pairs_from_permutations(parent_permutations),
        pairs_from_permutations(shear_permutations),
        iterations=int(cg_iterations),
        damping=float(cg_damping),
    )
    endpoint = source + tangent
    endpoint.mul_(1.0 - float(learning_rate) * float(weight_decay))
    update = endpoint.T.contiguous() - weight.float()
    return update, {
        **solver,
        "parent_stages": int(parent_stages),
        "shear_stages": int(shear_stages),
        "parent_matching": parent_selection,
        "shear_matching": shear_selection,
    }


def classify(
    rows: list[dict[str, Any]],
    *,
    confidence_z: float,
    minimum_fraction: float,
    mean_fraction: float,
) -> dict[str, Any]:
    pairs = {
        "dense_single": ("dense_norm_cfc", "production_cfc"),
        "dense_hybrid": ("hybrid_norm_cfc", "production_joint"),
        "mixed_single": ("mixed_joint_cfc", "production_cfc"),
        "mixed_hybrid": ("hybrid_mixed_joint_cfc", "production_joint"),
        "cproj64_single": ("cproj64_only", "production_cproj"),
        "cproj64_hybrid": ("hybrid_cproj64", "production_joint"),
    }
    comparisons = {
        name: paired_comparison(rows, candidate, reference, confidence_z)
        for name, (candidate, reference) in pairs.items()
    }
    means = {
        point: sum(float(row["ce"]) for row in rows if row["point_id"] == point)
        / sum(1 for row in rows if row["point_id"] == point)
        for point in CANDIDATE_ORDER
    }

    def recovery(candidate: str, production: str, oracle: str) -> float:
        gap = means[production] - means[oracle]
        return (means[production] - means[candidate]) / gap if gap > 0 else math.nan

    fractions = {
        "single": recovery(
            "mixed_joint_cfc", "production_cfc", "dense_norm_cfc"
        ),
        "hybrid": recovery(
            "hybrid_mixed_joint_cfc", "production_joint", "hybrid_norm_cfc"
        ),
    }
    oracle_valid = all(
        comparisons[name]["candidate_reliably_better"]
        for name in ("dense_single", "dense_hybrid")
    )
    mixed_reliable = all(
        comparisons[name]["candidate_reliably_better"]
        for name in ("mixed_single", "mixed_hybrid")
    )
    fraction_pass = (
        min(fractions.values()) >= float(minimum_fraction)
        and sum(fractions.values()) / len(fractions) >= float(mean_fraction)
    )
    if not oracle_valid:
        label = "HELDOUT_DENSE_CFC_ORACLE_NOT_STABLE"
        next_action = "DO_NOT_TRAIN_RESELECT_DISCRIMINATING_WINDOWS"
    elif mixed_reliable and fraction_pass:
        label = "MIXED_JOINT_CFC_SOLVE_PASSES"
        next_action = "IMPLEMENT_PERFORMANCE_PREFLIGHT_ONLY"
    elif mixed_reliable:
        label = "MIXED_JOINT_CFC_GAIN_TOO_SMALL"
        next_action = "DO_NOT_TRAIN_CHANGE_GENERATOR_FAMILY"
    else:
        label = "MIXED_JOINT_CFC_SOLVE_REJECTED"
        next_action = "DO_NOT_TRAIN_CHANGE_GENERATOR_FAMILY"
    return {
        "classification": label,
        "next_action": next_action,
        "candidate_means": means,
        "comparisons": comparisons,
        "oracle_gap_fraction_recovered": fractions,
        "gates": {
            "dense_cfc_oracle_valid": oracle_valid,
            "mixed_candidate_reliable_singleton_and_hybrid": mixed_reliable,
            "oracle_fraction_pass": fraction_pass,
        },
    }


def validate_plan(
    path: Path, checkpoint: Path, config: Path, data_dir: Path
) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    actual = {
        "checkpoint_sha256": file_sha256(checkpoint),
        "config_sha256": file_sha256(config),
        "dataset_manifest_sha256": file_sha256(data_dir / "manifest.json"),
        "entrypoint_sha256": file_sha256(Path(__file__).resolve()),
    }
    for key, value in actual.items():
        if value != plan["identity"][key]:
            raise ValueError(f"registered identity mismatch: {key}")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    plan = validate_plan(args.plan, args.checkpoint, args.config, args.data_dir)
    protocol = plan["protocol"]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[str(config["dtype"])]
    train_batches = fixed_batches(
        args.data_dir,
        "train",
        batch_size=int(config["batch_size"]),
        block_size=int(config["block_size"]) + 1,
        batches=int(protocol["gradient_accumulation_steps"]),
        seed=int(protocol["train_seed"]),
    )
    model, optimizer, checkpoint_payload = load_model_and_optimizer(
        args.checkpoint, config, args.device
    )
    model.train()
    weights = family_weights(model)
    optimizer.zero_grad(set_to_none=True)
    model.prepare_block_fht_cache(dtype=dtype)
    train_losses = []
    try:
        for tokens in train_batches:
            tokens = tokens.to(args.device)
            with _autocast(args.device, dtype):
                _logits, loss = model(
                    tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
                )
            if loss is None:
                raise RuntimeError("model did not return a training loss")
            train_losses.append(float(loss.detach()))
            (loss / len(train_batches)).backward()
    finally:
        model.flush_block_fht_cache()
    all_parameters = optimizer_parameters(optimizer)
    grad_before = _gradient_norm(all_parameters)
    clip_reported = torch.nn.utils.clip_grad_norm_(
        model.product_fht_clip_parameters(), float(config["grad_clip"])
    )
    grad_after = _gradient_norm(all_parameters)
    cfc_owner, cfc_group = _owner_and_group(optimizer, MuonFunctionalShear)
    cproj_owner, cproj_group = _owner_and_group(optimizer, MuonMatchedGivens)
    bases = {
        family: {
            layer: weight.detach().cpu().clone()
            for layer, weight in by_layer.items()
        }
        for family, by_layer in weights.items()
    }
    dense_cfc: dict[int, torch.Tensor] = {}
    control_cfc: dict[int, torch.Tensor] = {}
    mixed_raw: dict[int, torch.Tensor] = {}
    cproj64_raw: dict[int, torch.Tensor] = {}
    solver_rows: dict[int, Any] = {}
    for layer, weight in weights["c_fc"].items():
        momentum = cfc_owner.state[weight].get("momentum_buffer")
        if weight.grad is None or momentum is None:
            raise RuntimeError(f"missing c_fc gradient/state for layer {layer}")
        canonical, _descent, _row = exact_muon_update(
            weight,
            weight.grad,
            momentum,
            learning_rate=float(cfc_group["lr"]),
            momentum=float(cfc_group["momentum"]),
            weight_decay=float(cfc_group["weight_decay"]),
            ns_steps=int(cfc_group["ns_steps"]),
        )
        requested, direction = production_muon_request(
            weight,
            weight.grad,
            momentum,
            learning_rate=float(cfc_group["lr"]),
            momentum=float(cfc_group["momentum"]),
            weight_decay=float(cfc_group["weight_decay"]),
            ns_steps=int(cfc_group["ns_steps"]),
        )
        dense_cfc[layer] = historical_double_decay_update(
            weight,
            canonical,
            learning_rate=float(cfc_group["lr"]),
            weight_decay=float(cfc_group["weight_decay"]),
        ).cpu()
        module = model.transformer.h[layer].mlp.c_fc
        if module._functional_inputs is None or module._functional_pre_gelu is None:
            raise RuntimeError("functional c_fc context is missing")
        control, _control_row = functional_coordinate_mix_update(
            weight,
            requested,
            direction,
            module._functional_inputs,
            module._functional_pre_gelu,
            model.transformer.h[layer].mlp.c_proj.weight,
            parent_stages=int(module.parent_stages),
            shear_stages=int(module.shear_stages),
            neighbors=int(module.neighbors),
            seed=int(module.matching_seed) + int(module.optimizer_step),
            beta=float(module.coordinate_mix_beta),
            project_to_weight_norm=bool(module.project_to_weight_norm),
            max_condition_number=module.max_condition_number,
            learning_rate=float(cfc_group["lr"]),
            weight_decay=float(cfc_group["weight_decay"]),
        )
        control_cfc[layer] = control.cpu()
        mixed, row = mixed_joint_cfc_update(
            weight,
            requested,
            direction,
            parent_stages=int(module.parent_stages),
            shear_stages=int(module.shear_stages),
            neighbors=int(module.neighbors),
            seed=int(module.matching_seed) + int(module.optimizer_step),
            cg_iterations=int(protocol["cg_iterations"]),
            cg_damping=float(protocol["cg_damping"]),
            learning_rate=float(cfc_group["lr"]),
            weight_decay=float(cfc_group["weight_decay"]),
        )
        mixed_raw[layer] = mixed.cpu()
        solver_rows[layer] = row
        print(json.dumps({"mixed_layer_complete": layer}), flush=True)
    for layer, weight in weights["c_proj"].items():
        momentum = cproj_owner.state[weight].get("momentum_buffer")
        if weight.grad is None or momentum is None:
            raise RuntimeError(f"missing c_proj gradient/state for layer {layer}")
        requested, direction = production_muon_request(
            weight,
            weight.grad,
            momentum,
            learning_rate=float(cproj_group["lr"]),
            momentum=float(cproj_group["momentum"]),
            weight_decay=float(cproj_group["weight_decay"]),
            ns_steps=int(cproj_group["ns_steps"]),
        )
        module = model.transformer.h[layer].mlp.c_proj
        raw, _row = reconstruct_cproj_update(
            weight,
            requested,
            direction,
            parent_stages=int(module.stages),
            residual_stages=int(protocol["cproj_control_residual_stages"]),
            neighbors=int(module.neighbors),
            seed=int(module.matching_seed) + int(module.optimizer_step),
            learning_rate=float(cproj_group["lr"]),
            weight_decay=float(cproj_group["weight_decay"]),
        )
        cproj64_raw[layer] = raw.cpu()
    cfc_owner.step()
    cproj_owner.step()
    production = {
        family: {
            layer: weight.detach().float().cpu() - bases[family][layer].float()
            for layer, weight in by_layer.items()
        }
        for family, by_layer in weights.items()
    }
    prod_cfc = production["c_fc"]
    prod_cproj = production["c_proj"]
    reconstruction_error = max(
        float(
            (
                prod_cfc[layer]
                - quantized_update(bases["c_fc"][layer], control_cfc[layer])
            )
            .abs()
            .max()
        )
        for layer in prod_cfc
    )
    if reconstruction_error > float(protocol["control_max_abs_tolerance"]):
        raise RuntimeError(f"production c_fc reconstruction failed: {reconstruction_error}")
    cfc_radius = family_fro(prod_cfc)
    cproj_radius = family_fro(prod_cproj)
    mixed_cfc, mixed_normalization = normalize_family_to_radius(
        bases["c_fc"], mixed_raw, cfc_radius
    )
    cproj64, cproj_normalization = normalize_family_to_radius(
        bases["c_proj"], cproj64_raw, cproj_radius
    )
    for row in (mixed_normalization, cproj_normalization):
        if row["relative_radius_error"] > float(
            protocol["maximum_relative_radius_error"]
        ):
            raise RuntimeError("fixed-radius normalization failed")
    norm_dense_cfc = scale_family(
        dense_cfc, cfc_radius / family_fro(dense_cfc)
    )
    candidates = {
        "baseline": {},
        "production_cfc": {"c_fc": prod_cfc},
        "production_cproj": {"c_proj": prod_cproj},
        "production_joint": merge_updates(prod_cfc, prod_cproj),
        "dense_norm_cfc": {"c_fc": norm_dense_cfc},
        "hybrid_norm_cfc": merge_updates(norm_dense_cfc, prod_cproj),
        "mixed_joint_cfc": {"c_fc": mixed_cfc},
        "hybrid_mixed_joint_cfc": merge_updates(mixed_cfc, prod_cproj),
        "cproj64_only": {"c_proj": cproj64},
        "hybrid_cproj64": merge_updates(prod_cfc, cproj64),
    }
    if tuple(candidates) != CANDIDATE_ORDER:
        raise RuntimeError("candidate order differs from registration")
    del model, optimizer
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    model, _optimizer, checkpoint_reloaded = load_model_and_optimizer(
        args.checkpoint, config, args.device
    )
    applier = ExactVariantApplier(model)
    windows = {
        f"window_{index + 1}": fixed_batches(
            args.data_dir,
            "val",
            batch_size=int(protocol["evaluation_batch_size"]),
            block_size=int(protocol["evaluation_block_size"]) + 1,
            batches=int(protocol["validation_batches_per_window"]),
            seed=int(seed),
        )
        for index, seed in enumerate(protocol["validation_seeds"])
    }
    ce_rows = evaluate_candidates(
        model,
        applier,
        windows,
        candidates,
        device=args.device,
        dtype=dtype,
    )
    rule = plan["decision_rule"]
    decision = classify(
        ce_rows,
        confidence_z=float(rule["confidence_z"]),
        minimum_fraction=float(rule["minimum_oracle_gap_fraction"]),
        mean_fraction=float(rule["mean_oracle_gap_fraction"]),
    )
    args.output.mkdir(parents=True, exist_ok=False)
    paths = {
        "ce": args.output / "heldout_ce.json",
        "solver": args.output / "mixed_joint_solver.json",
        "replay": args.output / "prospective_step_metadata.json",
    }
    paths["ce"].write_text(json.dumps(ce_rows, indent=2, sort_keys=True) + "\n")
    paths["solver"].write_text(
        json.dumps(solver_rows, indent=2, sort_keys=True) + "\n"
    )
    replay = {
        "checkpoint_next_iter": int(checkpoint_payload["next_iter"]),
        "checkpoint_reloaded_next_iter": int(checkpoint_reloaded["next_iter"]),
        "mean_training_ce": sum(train_losses) / len(train_losses),
        "gradient_norm_before_clip": grad_before,
        "clip_norm_reported": float(clip_reported),
        "gradient_norm_after_clip": grad_after,
        "production_cfc_reconstruction_max_abs_error": reconstruction_error,
        "mixed_normalization": mixed_normalization,
        "cproj64_normalization": cproj_normalization,
        "mixed_direction_recovery_vs_norm_dense": aggregate_direction_metrics(
            norm_dense_cfc, mixed_cfc
        ),
    }
    paths["replay"].write_text(json.dumps(replay, indent=2, sort_keys=True) + "\n")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "parameter_updates_to_checkpoint": 0,
        "disposable_optimizer_steps": 2,
        "identity": {
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "config_sha256": file_sha256(args.config),
            "dataset_manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
            "plan_sha256": file_sha256(args.plan),
        },
        "replay": replay,
        "outputs": {
            f"{name}_sha256": file_sha256(path) for name, path in paths.items()
        },
        "execution": {
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": str(Path(__file__).resolve()),
            "entrypoint_sha256": file_sha256(Path(__file__).resolve()),
            "command": sys.argv,
            "device": args.device,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "direct_foreground_polling": True,
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
