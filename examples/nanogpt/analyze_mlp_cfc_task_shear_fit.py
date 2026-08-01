#!/usr/bin/env python3
"""Fit equal-coordinate task-matched shear charts to exact-current c_fc Muon.

This zero-update diagnostic retains a task-selected sparse rotational parent
and reallocates the remaining fresh88 coordinate budget to symmetric shear or
joint skew-plus-shear 2x2 blocks on the 3,072 c_fc output channels.  It is a
fit-only gate: no validation CE, model update, optimizer update, or training
is performed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Literal

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.nanogpt.analyze_mlp_cfc_exact_current_matcher import (
    _optimizer_and_group_for_parameter,
    _weight_decay_after_rotation,
    direction_metrics,
    exact_muon_update,
    file_sha256,
    fixed_batches,
    load_model_and_optimizer,
)
from examples.nanogpt.analyze_mlp_cfc_residual_structure import (
    validate_identity,
    write_csv,
)
from examples.nanogpt.analyze_mlp_cfc_trust_radius import (
    collect_gradient_window,
)
from examples.nanogpt.analyze_mlp_muon_matched_givens import (
    diagonal_metric_causal_givens_update,
)
from examples.nanogpt.fast_task_matching import (
    color_sorted_edges,
    fast_muon_matched_permutations,
)


Family = Literal["shear", "skew_shear"]
SCHEMA_VERSION = "nanogpt_mlp_cfc_task_shear_fit_v1"
CONTROL = "fresh88"
EQUAL_COORDINATE_CANDIDATES = (
    "fresh64_shear24",
    "fresh64_skew_shear12",
    "fresh48_skew_shear20",
)
CANDIDATES = ("fresh64", CONTROL, *EQUAL_COORDINATE_CANDIDATES)


def git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def _pair_normal_equations(
    source: torch.Tensor,
    residual: torch.Tensor,
    pairs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact tangent normal equations for [symmetric, skew]."""
    left, right = pairs.unbind(dim=1)
    u = source[:, left].double()
    v = source[:, right].double()
    residual_left = residual[:, left].double()
    residual_right = residual[:, right].double()
    norm_u = u.square().sum(dim=0)
    norm_v = v.square().sum(dim=0)
    total = norm_u + norm_v
    difference = norm_v - norm_u
    rhs = torch.stack(
        (
            (v * residual_left).sum(dim=0)
            + (u * residual_right).sum(dim=0),
            (v * residual_left).sum(dim=0)
            - (u * residual_right).sum(dim=0),
        ),
        dim=1,
    )
    normal = torch.zeros(
        (pairs.shape[0], 2, 2), dtype=torch.float64, device=source.device
    )
    normal[:, 0, 0] = total
    normal[:, 1, 1] = total
    normal[:, 0, 1] = difference
    normal[:, 1, 0] = difference
    return normal, rhs


def fit_pair_coordinates(
    source: torch.Tensor,
    residual: torch.Tensor,
    pairs: torch.Tensor,
    *,
    family: Family,
) -> torch.Tensor:
    """Fit symmetric-shear and skew coordinates for one disjoint stage."""
    normal, rhs = _pair_normal_equations(source, residual, pairs)
    scale = normal.diagonal(dim1=1, dim2=2).mean(dim=1).clamp_min(1e-30)
    if family == "shear":
        coordinates = torch.zeros_like(rhs)
        coordinates[:, 0] = rhs[:, 0] / scale
        return coordinates
    if family != "skew_shear":
        raise ValueError(f"unknown pair family: {family}")
    ridge = scale.mul(1e-10).reshape(-1, 1, 1)
    return torch.linalg.solve(
        normal
        + ridge
        * torch.eye(2, dtype=normal.dtype, device=normal.device).unsqueeze(0),
        rhs.unsqueeze(-1),
    ).squeeze(-1)


def coordinates_to_generators(coordinates: torch.Tensor) -> torch.Tensor:
    """Map [symmetric shear, skew] coordinates to trace-free 2x2 blocks."""
    symmetric, skew = coordinates.unbind(dim=1)
    generators = torch.zeros(
        (coordinates.shape[0], 2, 2),
        dtype=coordinates.dtype,
        device=coordinates.device,
    )
    generators[:, 0, 1] = symmetric - skew
    generators[:, 1, 0] = symmetric + skew
    return generators


def apply_pair_stage(
    source: torch.Tensor,
    pairs: torch.Tensor,
    coordinates: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply exact determinant-one pair maps to matrix columns."""
    left, right = pairs.unbind(dim=1)
    maps = torch.matrix_exp(coordinates_to_generators(coordinates)).to(
        dtype=source.dtype
    )
    pair_values = torch.stack((source[:, left], source[:, right]), dim=-1)
    transformed = torch.einsum("rpk,pkj->rpj", pair_values, maps)
    result = source.clone()
    result[:, left] = transformed[:, :, 0]
    result[:, right] = transformed[:, :, 1]
    determinants = torch.linalg.det(maps.double())
    conditions = torch.linalg.cond(maps.double())
    return result, {
        "minimum_determinant": float(determinants.min()),
        "maximum_determinant_error": float((determinants - 1.0).abs().max()),
        "maximum_condition_number": float(conditions.max()),
    }


@torch.no_grad()
def fit_pair_flow(
    source: torch.Tensor,
    requested_update: torch.Tensor,
    permutations: torch.Tensor,
    *,
    stages: int,
    family: Family,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Causally fit and apply a task-selected pair flow."""
    if (
        source.ndim != 2
        or source.shape != requested_update.shape
        or permutations.ndim != 2
        or permutations.shape[1] != source.shape[1]
        or stages <= 0
        or stages > permutations.shape[0]
    ):
        raise ValueError("invalid pair-flow inputs")
    current = source.float().clone()
    target = source.float() + requested_update.float()
    coordinate_rows: list[torch.Tensor] = []
    stage_recovery: list[float] = []
    minimum_determinant = float("inf")
    maximum_determinant_error = 0.0
    maximum_condition_number = 0.0
    for stage in range(stages):
        residual_before = target - current
        pairs = (
            permutations[stage]
            .to(device=source.device, dtype=torch.long)
            .reshape(-1, 2)
        )
        coordinates = fit_pair_coordinates(
            current, residual_before, pairs, family=family
        )
        updated, finite = apply_pair_stage(current, pairs, coordinates)
        before_energy = residual_before.double().square().sum().clamp_min(1e-30)
        after_energy = (target - updated).double().square().sum()
        stage_recovery.append(float(1.0 - after_energy / before_energy))
        current = updated
        coordinate_rows.append(coordinates)
        minimum_determinant = min(
            minimum_determinant, finite["minimum_determinant"]
        )
        maximum_determinant_error = max(
            maximum_determinant_error, finite["maximum_determinant_error"]
        )
        maximum_condition_number = max(
            maximum_condition_number, finite["maximum_condition_number"]
        )
    coordinates = torch.cat(coordinate_rows, dim=0)
    target_energy = requested_update.double().square().sum().clamp_min(1e-30)
    residual_energy = (target - current).double().square().sum()
    return current - source.float(), {
        "family": family,
        "stages": stages,
        "coordinates": int(
            stages * (source.shape[1] // 2) * (1 if family == "shear" else 2)
        ),
        "requested_update_recovery": float(1.0 - residual_energy / target_energy),
        "mean_stage_requested_recovery": sum(stage_recovery) / len(stage_recovery),
        "symmetric_shear_rms": float(coordinates[:, 0].square().mean().sqrt()),
        "symmetric_shear_max_abs": float(coordinates[:, 0].abs().max()),
        "skew_rms": float(coordinates[:, 1].square().mean().sqrt()),
        "skew_max_abs": float(coordinates[:, 1].abs().max()),
        "minimum_determinant": minimum_determinant,
        "maximum_determinant_error": maximum_determinant_error,
        "maximum_condition_number": maximum_condition_number,
    }


@torch.no_grad()
def pair_matched_permutations(
    source: torch.Tensor,
    residual: torch.Tensor,
    *,
    stages: int,
    neighbors: int,
    seed: int,
    family: Family,
    native_cache: Path | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Select disjoint pair stages by exact shear/skew projection score."""
    if (
        source.ndim != 2
        or source.shape != residual.shape
        or source.shape[1] <= 0
        or source.shape[1] % 2
        or stages <= 0
        or stages > 64
        or neighbors < stages
        or neighbors >= source.shape[1]
    ):
        raise ValueError("invalid task-shear matching inputs")
    source = source.float()
    residual = residual.float()
    width = source.shape[1]
    cross = source.T @ residual
    norm = source.square().sum(dim=0)
    total = norm[:, None] + norm[None, :]
    q_symmetric = cross + cross.T
    if family == "shear":
        scores = q_symmetric.square() / total.clamp_min(1e-30)
    elif family == "skew_shear":
        q_skew = cross.T - cross
        difference = norm[None, :] - norm[:, None]
        determinant = (total.square() - difference.square()).clamp_min(1e-30)
        scores = (
            total * (q_symmetric.square() + q_skew.square())
            - 2.0 * difference * q_symmetric * q_skew
        ) / determinant
    else:
        raise ValueError(f"unknown pair family: {family}")
    scores.fill_diagonal_(-1.0)
    top_scores, top_indices = torch.topk(scores, k=neighbors, dim=1)
    order = torch.argsort(top_scores.reshape(-1), descending=True)
    left = (
        torch.arange(width, device=source.device)
        .repeat_interleave(neighbors)
        .index_select(0, order)
    )
    right = top_indices.reshape(-1).index_select(0, order)
    edges = torch.stack(
        (torch.minimum(left, right), torch.maximum(left, right)), dim=1
    ).to(device="cpu", dtype=torch.int32)
    permutations, diagnostics = color_sorted_edges(
        edges,
        width=width,
        stages=stages,
        seed=seed,
        cache_dir=native_cache,
    )
    diagnostics.update(
        {
            "candidate_edges": int(edges.shape[0]),
            "preselection_neighbors": neighbors,
            "score_family": f"exact_{family}_2x2_tangent",
            "maximum_projection_score": float(top_scores.max()),
            "mean_top_projection_score": float(top_scores.mean()),
        }
    )
    return permutations, diagnostics


@torch.no_grad()
def _fit_rotational_parent(
    source: torch.Tensor,
    target_update: torch.Tensor,
    selection_direction: torch.Tensor,
    *,
    stages: int,
    neighbors: int,
    seed: int,
    native_cache: Path | None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    permutations, selection = fast_muon_matched_permutations(
        source,
        selection_direction,
        stages=stages,
        neighbors=neighbors,
        seed=seed,
        cache_dir=native_cache,
    )
    update, fit = diagonal_metric_causal_givens_update(
        source,
        target_update,
        stages=stages,
        seed=seed,
        permutations=permutations,
    )
    return source.float() + update, update, {"selection": selection, "fit": fit}


@torch.no_grad()
def build_candidates(
    weight: torch.Tensor,
    dense_update: torch.Tensor,
    polar_descent_per_lr: torch.Tensor,
    *,
    neighbors: int,
    seed: int,
    learning_rate: float,
    weight_decay: float,
    native_cache: Path | None,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    """Build the fixed equal-coordinate c_fc task-shear bracket."""
    source = weight.float().T.contiguous()
    target = dense_update.float().T.contiguous()
    selection_direction = polar_descent_per_lr.float().T.contiguous()
    after64, parent64, parent64_diag = _fit_rotational_parent(
        source,
        target,
        selection_direction,
        stages=64,
        neighbors=neighbors,
        seed=seed,
        native_cache=native_cache,
    )
    residual64 = target - parent64
    control_permutations, control_selection = fast_muon_matched_permutations(
        after64,
        residual64,
        stages=24,
        neighbors=neighbors,
        seed=seed + 1,
        cache_dir=native_cache,
    )
    control_residual, control_fit = diagonal_metric_causal_givens_update(
        after64,
        residual64,
        stages=24,
        seed=seed + 1,
        permutations=control_permutations,
    )
    shear_permutations, shear_selection = pair_matched_permutations(
        after64,
        residual64,
        stages=24,
        neighbors=neighbors,
        seed=seed + 2,
        family="shear",
        native_cache=native_cache,
    )
    shear24, shear_fit = fit_pair_flow(
        after64,
        residual64,
        shear_permutations,
        stages=24,
        family="shear",
    )
    joint_permutations, joint_selection = pair_matched_permutations(
        after64,
        residual64,
        stages=12,
        neighbors=neighbors,
        seed=seed + 3,
        family="skew_shear",
        native_cache=native_cache,
    )
    joint12, joint12_fit = fit_pair_flow(
        after64,
        residual64,
        joint_permutations,
        stages=12,
        family="skew_shear",
    )
    after48, parent48, parent48_diag = _fit_rotational_parent(
        source,
        target,
        selection_direction,
        stages=48,
        neighbors=neighbors,
        seed=seed + 4,
        native_cache=native_cache,
    )
    residual48 = target - parent48
    joint20_permutations, joint20_selection = pair_matched_permutations(
        after48,
        residual48,
        stages=20,
        neighbors=neighbors,
        seed=seed + 5,
        family="skew_shear",
        native_cache=native_cache,
    )
    joint20, joint20_fit = fit_pair_flow(
        after48,
        residual48,
        joint20_permutations,
        stages=20,
        family="skew_shear",
    )
    rotations = {
        "fresh64": parent64,
        CONTROL: parent64 + control_residual,
        "fresh64_shear24": parent64 + shear24,
        "fresh64_skew_shear12": parent64 + joint12,
        "fresh48_skew_shear20": parent48 + joint20,
    }
    candidates = {
        name: _weight_decay_after_rotation(
            source,
            rotation,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        .T.contiguous()
        for name, rotation in rotations.items()
    }
    diagnostics = [
        {"candidate": "fresh64", **parent64_diag},
        {
            "candidate": CONTROL,
            "selection": control_selection,
            "fit": control_fit,
        },
        {
            "candidate": "fresh64_shear24",
            "selection": shear_selection,
            "fit": shear_fit,
        },
        {
            "candidate": "fresh64_skew_shear12",
            "selection": joint_selection,
            "fit": joint12_fit,
        },
        {"candidate": "fresh48_parent", **parent48_diag},
        {
            "candidate": "fresh48_skew_shear20",
            "selection": joint20_selection,
            "fit": joint20_fit,
        },
    ]
    return candidates, diagnostics


def aggregate(
    metric_rows: list[dict[str, Any]],
    fit_rows: list[dict[str, Any]],
    *,
    minimum_layer_delta: float,
    minimum_aggregate_ratio: float,
    maximum_determinant_error: float,
    maximum_condition_number: float,
) -> dict[str, Any]:
    by_candidate: dict[str, dict[str, Any]] = {}
    control_rows = [row for row in metric_rows if row["candidate"] == CONTROL]
    control_by_layer = {int(row["layer"]): row for row in control_rows}
    for candidate in CANDIDATES:
        rows = [row for row in metric_rows if row["candidate"] == candidate]
        target_energy = sum(float(row["target_energy"]) for row in rows)
        residual_energy = sum(float(row["residual_energy"]) for row in rows)
        aggregate_recovery = 1.0 - residual_energy / max(target_energy, 1e-30)
        deltas = [
            float(row["fixed_scale_recovery"])
            - float(control_by_layer[int(row["layer"])]["fixed_scale_recovery"])
            for row in rows
        ]
        by_candidate[candidate] = {
            "aggregate_fixed_scale_recovery": aggregate_recovery,
            "minimum_layer_fixed_scale_recovery": min(
                float(row["fixed_scale_recovery"]) for row in rows
            ),
            "median_layer_fixed_scale_recovery": float(
                torch.tensor(
                    [float(row["fixed_scale_recovery"]) for row in rows],
                    dtype=torch.float64,
                ).median()
            ),
            "minimum_layer_delta_vs_fresh88": min(deltas),
            "maximum_layer_delta_vs_fresh88": max(deltas),
        }
    control_recovery = by_candidate[CONTROL]["aggregate_fixed_scale_recovery"]
    stable = True
    for row in fit_rows:
        fit = row.get("fit")
        if isinstance(fit, dict) and "maximum_determinant_error" in fit:
            stable = stable and (
                float(fit["maximum_determinant_error"])
                <= maximum_determinant_error
                and float(fit["maximum_condition_number"])
                <= maximum_condition_number
            )
    passing: list[str] = []
    for candidate in EQUAL_COORDINATE_CANDIDATES:
        values = by_candidate[candidate]
        values["aggregate_ratio_vs_fresh88"] = (
            values["aggregate_fixed_scale_recovery"]
            / max(control_recovery, 1e-30)
        )
        values["passes_fit_gate"] = bool(
            stable
            and values["minimum_layer_delta_vs_fresh88"] >= minimum_layer_delta
            and values["aggregate_ratio_vs_fresh88"] >= minimum_aggregate_ratio
        )
        if values["passes_fit_gate"]:
            passing.append(candidate)
    if not stable:
        decision = "TASK_SHEAR_NUMERICAL_GATE_FAILED"
    elif passing:
        decision = "PROMOTE_TASK_MATCHED_SHEAR_TO_HELDOUT_CE"
    else:
        decision = "REJECT_EQUAL_COORDINATE_TASK_SHEAR_FIT"
    selected = max(
        passing,
        key=lambda candidate: (
            by_candidate[candidate]["minimum_layer_delta_vs_fresh88"],
            by_candidate[candidate]["aggregate_fixed_scale_recovery"],
        ),
        default=None,
    )
    return {
        "decision": decision,
        "selected_candidate": selected,
        "parameter_updates": 0,
        "candidate_results": by_candidate,
        "gates": {"numerically_stable": stable},
        "thresholds": {
            "minimum_layer_delta": minimum_layer_delta,
            "minimum_aggregate_ratio": minimum_aggregate_ratio,
            "maximum_determinant_error": maximum_determinant_error,
            "maximum_condition_number": maximum_condition_number,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--native-cache", type=Path)
    args = parser.parse_args()
    started = time.time()
    plan = validate_identity(args.checkpoint, args.config, args.data_dir, args.plan)
    protocol = plan["fixed_protocol"]
    rule = plan["decision_rule"]
    layers = [int(layer) for layer in protocol["layers"]]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fit_batches = fixed_batches(
        args.data_dir,
        "train",
        batch_size=int(protocol["batch_size"]),
        block_size=int(protocol["block_size"]),
        batches=int(protocol["fit_batches"]),
        seed=int(protocol["fit_train_seed"]),
    )
    model, optimizer, checkpoint = load_model_and_optimizer(
        args.checkpoint, config, args.device
    )
    fit_loss, gradients = collect_gradient_window(
        model, fit_batches, layers, device=args.device, dtype=torch.bfloat16
    )
    metric_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    optimizer_rows: list[dict[str, Any]] = []
    for layer in layers:
        weight = model.transformer.h[layer].mlp.c_fc.weight
        owner, group = _optimizer_and_group_for_parameter(optimizer, weight)
        buffer = owner.state[weight].get("momentum_buffer")
        if buffer is None:
            raise RuntimeError(f"missing c_fc momentum at layer {layer}")
        dense_update, descent, optimizer_diag = exact_muon_update(
            weight.detach(),
            gradients[layer].to(weight.device),
            buffer,
            learning_rate=float(group["lr"]),
            momentum=float(group["momentum"]),
            weight_decay=float(group["weight_decay"]),
            ns_steps=int(group["ns_steps"]),
        )
        polar_descent = (
            descent + float(group["weight_decay"]) * weight.detach().float()
        )
        candidates, diagnostics = build_candidates(
            weight.detach(),
            dense_update,
            polar_descent,
            neighbors=int(protocol["matching_neighbors"]),
            seed=int(protocol["matching_seed"]) + layer * 1009,
            learning_rate=float(group["lr"]),
            weight_decay=float(group["weight_decay"]),
            native_cache=args.native_cache,
        )
        for candidate, update in candidates.items():
            values = direction_metrics(dense_update, update)
            metric_rows.append(
                {
                    "layer": layer,
                    "candidate": candidate,
                    **values,
                    "residual_energy": float(
                        (dense_update.double() - update.double()).square().sum()
                    ),
                }
            )
        fit_rows.extend(
            {"layer": layer, **diagnostic} for diagnostic in diagnostics
        )
        optimizer_rows.append({"layer": layer, **optimizer_diag})
        print(
            json.dumps(
                {"layer_complete": layer, "layers_total": len(layers)},
                sort_keys=True,
            ),
            flush=True,
        )
    result = aggregate(
        metric_rows,
        fit_rows,
        minimum_layer_delta=float(rule["minimum_layer_delta"]),
        minimum_aggregate_ratio=float(rule["minimum_aggregate_ratio"]),
        maximum_determinant_error=float(rule["maximum_determinant_error"]),
        maximum_condition_number=float(rule["maximum_condition_number"]),
    )
    result["fit_gradient_loss_bfloat16"] = fit_loss
    args.output.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": args.output / "cfc_task_shear_fit_metrics.csv",
        "fits": args.output / "cfc_task_shear_fit_fits.json",
        "optimizer": args.output / "cfc_task_shear_fit_optimizer.csv",
        "aggregate": args.output / "cfc_task_shear_fit_aggregate.json",
    }
    write_csv(paths["metrics"], metric_rows)
    paths["fits"].write_text(
        json.dumps(fit_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(paths["optimizer"], optimizer_rows)
    paths["aggregate"].write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "decision": result["decision"],
        "parameter_updates": 0,
        "checkpoint_next_iter": int(checkpoint["next_iter"]),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "config_sha256": file_sha256(args.config),
        "dataset_manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
        "plan_sha256": file_sha256(args.plan),
        "analysis_execution": {
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": str(Path(__file__).resolve()),
            "entrypoint_sha256": file_sha256(Path(__file__).resolve()),
            "command": sys.argv,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "device": args.device,
            "direct_foreground_polling": True,
            "watchdog": False,
            "callback": False,
        },
        "protocol": protocol,
        "outputs": {
            f"{name}_sha256": file_sha256(path) for name, path in paths.items()
        },
        "limitations": plan["limitations"],
    }
    metadata_path = args.output / "cfc_task_shear_fit_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "selected_candidate": result["selected_candidate"],
                "aggregate": str(paths["aggregate"]),
                "metadata": str(metadata_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
