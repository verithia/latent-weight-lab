#!/usr/bin/env python3
"""Fit equal-coordinate c_fc shear charts in post-GELU/c_proj function space.

The qualified 64-stage task-selected rotational parent is retained.  The
remaining 24 one-coordinate shear stages are selected and/or fitted either
in weight Frobenius geometry or in the exact linearized MLP-output geometry
induced by observed inputs, GELU slopes, and the fixed c_proj.  This is a
zero-update fit/holdout functional discriminator, not finite-CE evaluation or
training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.nanogpt.analyze_mlp_cfc_exact_current_matcher import (
    _optimizer_and_group_for_parameter,
    _weight_decay_after_rotation,
    activation_effect_metrics,
    collect_window,
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
from examples.nanogpt.analyze_mlp_cfc_task_shear_fit import (
    _fit_rotational_parent,
    apply_pair_stage,
    fit_pair_flow,
    pair_matched_permutations,
)
from examples.nanogpt.analyze_mlp_muon_matched_givens import (
    diagonal_metric_causal_givens_update,
)
from examples.nanogpt.fast_task_matching import (
    color_sorted_edges,
    fast_muon_matched_permutations,
)


SCHEMA_VERSION = "nanogpt_mlp_cfc_functional_shear_fit_v1"
CONTROL = "fresh88"
WEIGHT_SHEAR = "fresh64_weight_shear24"
FUNCTIONAL_TOPOLOGY = "fresh64_functional_topology_weight_fit24"
FUNCTIONAL_FIT = "fresh64_weight_topology_functional_fit24"
FUNCTIONAL_BOTH = "fresh64_functional_shear24"
CANDIDATES = (
    CONTROL,
    WEIGHT_SHEAR,
    FUNCTIONAL_TOPOLOGY,
    FUNCTIONAL_FIT,
    FUNCTIONAL_BOTH,
)
WINDOWS = ("fit", "holdout")


def git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def gelu_derivative(values: torch.Tensor) -> torch.Tensor:
    """Derivative of the exact erf GELU used by torch.nn.functional.gelu."""
    values = values.float()
    return 0.5 * (1.0 + torch.erf(values / math.sqrt(2.0))) + (
        values * torch.exp(-0.5 * values.square()) / math.sqrt(2.0 * math.pi)
    )


def sample_aligned(
    inputs: torch.Tensor,
    pre_gelu: torch.Tensor,
    *,
    sample_cap: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Select one deterministic aligned token subset on CPU."""
    if (
        inputs.ndim != 2
        or pre_gelu.ndim != 2
        or inputs.shape[0] != pre_gelu.shape[0]
        or sample_cap <= 0
        or sample_cap > inputs.shape[0]
    ):
        raise ValueError("invalid aligned activation sample")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    indices = torch.randperm(inputs.shape[0], generator=generator)[:sample_cap]
    # Tensor SHA is recorded from the stable integer byte representation.
    sha = hashlib.sha256(
        memoryview(indices.contiguous().numpy())
    ).hexdigest()
    return (
        inputs.index_select(0, indices),
        pre_gelu.index_select(0, indices),
        sha,
    )


@torch.no_grad()
def functional_shear_scores(
    source: torch.Tensor,
    requested_update: torch.Tensor,
    inputs: torch.Tensor,
    pre_gelu: torch.Tensor,
    cproj_weight: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return exact all-pair linearized MLP-output shear projection scores."""
    if (
        source.ndim != 2
        or source.shape != requested_update.shape
        or inputs.ndim != 2
        or inputs.shape[1] != source.shape[0]
        or pre_gelu.shape != (inputs.shape[0], source.shape[1])
        or cproj_weight.ndim != 2
        or cproj_weight.shape[1] != source.shape[1]
    ):
        raise ValueError("invalid functional-shear score inputs")
    source = source.float()
    requested_update = requested_update.float()
    inputs = inputs.to(device=source.device, dtype=torch.float32)
    pre_gelu = pre_gelu.to(device=source.device, dtype=torch.float32)
    cproj = cproj_weight.to(device=source.device, dtype=torch.float32)
    slopes = gelu_derivative(pre_gelu)
    source_pre = inputs @ source
    target_pre = inputs @ requested_update
    target_output = (slopes * target_pre) @ cproj.T
    projected_target = target_output @ cproj
    cross = (slopes * projected_target).T @ source_pre
    numerator = (cross + cross.T).square()
    slope_square = slopes.square()
    source_square = source_pre.square()
    pair_activation_energy = slope_square.T @ source_square
    gated_source = slopes * source_pre
    gated_cross = gated_source.T @ gated_source
    cproj_gram = cproj.T @ cproj
    cproj_norm = cproj_gram.diagonal()
    denominator = (
        cproj_norm[:, None] * pair_activation_energy
        + cproj_norm[None, :] * pair_activation_energy.T
        + 2.0 * cproj_gram * gated_cross
    ).clamp_min(1e-30)
    scores = numerator / denominator
    scores.fill_diagonal_(-1.0)
    return scores, {
        "target_functional_energy": float(target_output.square().sum()),
        "maximum_projection_score": float(scores.max()),
        "mean_positive_projection_score": float(
            scores.clamp_min(0.0).mean()
        ),
    }


@torch.no_grad()
def functional_matched_permutations(
    source: torch.Tensor,
    requested_update: torch.Tensor,
    inputs: torch.Tensor,
    pre_gelu: torch.Tensor,
    cproj_weight: torch.Tensor,
    *,
    stages: int,
    neighbors: int,
    seed: int,
    native_cache: Path | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Select pair stages by exact linearized MLP-output recovery."""
    if (
        stages <= 0
        or stages > 64
        or neighbors < stages
        or neighbors >= source.shape[1]
    ):
        raise ValueError("invalid functional matching sizes")
    scores, score_diagnostics = functional_shear_scores(
        source, requested_update, inputs, pre_gelu, cproj_weight
    )
    top_scores, top_indices = torch.topk(scores, k=neighbors, dim=1)
    order = torch.argsort(top_scores.reshape(-1), descending=True)
    left = (
        torch.arange(source.shape[1], device=source.device)
        .repeat_interleave(neighbors)
        .index_select(0, order)
    )
    right = top_indices.reshape(-1).index_select(0, order)
    edges = torch.stack(
        (torch.minimum(left, right), torch.maximum(left, right)), dim=1
    ).to(device="cpu", dtype=torch.int32)
    permutations, diagnostics = color_sorted_edges(
        edges,
        width=source.shape[1],
        stages=stages,
        seed=seed,
        cache_dir=native_cache,
    )
    diagnostics.update(
        {
            **score_diagnostics,
            "candidate_edges": int(edges.shape[0]),
            "preselection_neighbors": neighbors,
            "score_family": "exact_linearized_postgelu_cproj_shear",
        }
    )
    return permutations, diagnostics


@torch.no_grad()
def fit_functional_shear_recipe(
    source: torch.Tensor,
    requested_update: torch.Tensor,
    inputs: torch.Tensor,
    pre_gelu: torch.Tensor,
    cproj_weight: torch.Tensor,
    permutations: torch.Tensor,
    *,
    stages: int,
) -> tuple[
    torch.Tensor,
    dict[str, Any],
    list[tuple[torch.Tensor, torch.Tensor]],
]:
    """Fit exact shears and return their ordered finite-map recipe."""
    if (
        source.ndim != 2
        or source.shape != requested_update.shape
        or permutations.ndim != 2
        or permutations.shape[1] != source.shape[1]
        or stages <= 0
        or stages > permutations.shape[0]
    ):
        raise ValueError("invalid functional shear flow inputs")
    current = source.float().clone()
    inputs = inputs.to(device=source.device, dtype=torch.float32)
    pre_gelu = pre_gelu.to(device=source.device, dtype=torch.float32)
    cproj = cproj_weight.to(device=source.device, dtype=torch.float32)
    slopes = gelu_derivative(pre_gelu)
    projected = inputs @ current
    target_output = (slopes * (inputs @ requested_update.float())) @ cproj.T
    residual_output = target_output.clone()
    cproj_gram = cproj.T @ cproj
    cproj_norm = cproj_gram.diagonal()
    coordinates_all: list[torch.Tensor] = []
    stage_recovery: list[float] = []
    maximum_determinant_error = 0.0
    maximum_condition_number = 0.0
    recipe: list[tuple[torch.Tensor, torch.Tensor]] = []
    for stage in range(stages):
        pairs = (
            permutations[stage]
            .to(device=source.device, dtype=torch.long)
            .reshape(-1, 2)
        )
        left, right = pairs.unbind(dim=1)
        projected_target = residual_output @ cproj
        left_direction = slopes[:, left] * projected[:, right]
        right_direction = slopes[:, right] * projected[:, left]
        dot = (
            left_direction * projected_target[:, left]
            + right_direction * projected_target[:, right]
        ).sum(dim=0)
        norm = (
            cproj_norm[left] * left_direction.square().sum(dim=0)
            + cproj_norm[right] * right_direction.square().sum(dim=0)
            + 2.0
            * cproj_gram[left, right]
            * (left_direction * right_direction).sum(dim=0)
        ).clamp_min(1e-30)
        coordinates = torch.zeros(
            (pairs.shape[0], 2), dtype=torch.float64, device=source.device
        )
        coordinates[:, 0] = dot.double() / norm.double()
        updated, finite = apply_pair_stage(current, pairs, coordinates)
        projected_updated, _projected_finite = apply_pair_stage(
            projected, pairs, coordinates
        )
        contribution_output = (
            slopes * (projected_updated - projected)
        ) @ cproj.T
        before = residual_output.double().square().sum().clamp_min(1e-30)
        residual_output = residual_output - contribution_output
        after = residual_output.double().square().sum()
        stage_recovery.append(float(1.0 - after / before))
        current = updated
        projected = projected_updated
        coordinates_all.append(coordinates)
        recipe.append((pairs.detach().clone(), coordinates.detach().clone()))
        maximum_determinant_error = max(
            maximum_determinant_error, finite["maximum_determinant_error"]
        )
        maximum_condition_number = max(
            maximum_condition_number, finite["maximum_condition_number"]
        )
    coordinates = torch.cat(coordinates_all, dim=0)
    target_energy = target_output.double().square().sum().clamp_min(1e-30)
    return current - source.float(), {
        "family": "functional_shear",
        "stages": stages,
        "coordinates": int(stages * source.shape[1] // 2),
        "functional_requested_recovery": float(
            1.0 - residual_output.double().square().sum() / target_energy
        ),
        "mean_stage_functional_recovery": sum(stage_recovery) / len(stage_recovery),
        "shear_rms": float(coordinates[:, 0].square().mean().sqrt()),
        "shear_max_abs": float(coordinates[:, 0].abs().max()),
        "maximum_determinant_error": maximum_determinant_error,
        "maximum_condition_number": maximum_condition_number,
    }, recipe


@torch.no_grad()
def replay_functional_shear_recipe(
    source: torch.Tensor,
    recipe: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    coordinate_scale: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Replay one fitted recipe after scaling only its shear coordinates."""
    if coordinate_scale < 0.0 or not math.isfinite(float(coordinate_scale)):
        raise ValueError("coordinate_scale must be finite and non-negative")
    current = source.float().clone()
    maximum_determinant_error = 0.0
    maximum_condition_number = 0.0
    minimum_determinant = float("inf")
    for pairs, coordinates in recipe:
        current, finite = apply_pair_stage(
            current,
            pairs.to(device=source.device),
            coordinates.to(device=source.device) * float(coordinate_scale),
        )
        minimum_determinant = min(
            minimum_determinant, finite["minimum_determinant"]
        )
        maximum_determinant_error = max(
            maximum_determinant_error, finite["maximum_determinant_error"]
        )
        maximum_condition_number = max(
            maximum_condition_number, finite["maximum_condition_number"]
        )
    if not recipe:
        minimum_determinant = 1.0
    return current - source.float(), {
        "coordinate_scale": float(coordinate_scale),
        "minimum_determinant": minimum_determinant,
        "maximum_determinant_error": maximum_determinant_error,
        "maximum_condition_number": maximum_condition_number,
    }


@torch.no_grad()
def fit_functional_shear_flow(
    source: torch.Tensor,
    requested_update: torch.Tensor,
    inputs: torch.Tensor,
    pre_gelu: torch.Tensor,
    cproj_weight: torch.Tensor,
    permutations: torch.Tensor,
    *,
    stages: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Causally fit exact shear maps in linearized MLP-output geometry."""
    update, diagnostics, _recipe = fit_functional_shear_recipe(
        source,
        requested_update,
        inputs,
        pre_gelu,
        cproj_weight,
        permutations,
        stages=stages,
    )
    return update, diagnostics


@torch.no_grad()
def build_candidates(
    weight: torch.Tensor,
    dense_update: torch.Tensor,
    polar_descent_per_lr: torch.Tensor,
    inputs: torch.Tensor,
    pre_gelu: torch.Tensor,
    cproj_weight: torch.Tensor,
    *,
    neighbors: int,
    seed: int,
    learning_rate: float,
    weight_decay: float,
    native_cache: Path | None,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    """Build the equal-coordinate weight/function shear attribution grid."""
    source = weight.float().T.contiguous()
    target = dense_update.float().T.contiguous()
    selection_direction = polar_descent_per_lr.float().T.contiguous()
    after_parent, parent, parent_diagnostics = _fit_rotational_parent(
        source,
        target,
        selection_direction,
        stages=64,
        neighbors=neighbors,
        seed=seed,
        native_cache=native_cache,
    )
    residual = target - parent
    control_permutations, control_selection = fast_muon_matched_permutations(
        after_parent,
        residual,
        stages=24,
        neighbors=neighbors,
        seed=seed + 1,
        cache_dir=native_cache,
    )
    control_residual, control_fit = diagonal_metric_causal_givens_update(
        after_parent,
        residual,
        stages=24,
        seed=seed + 1,
        permutations=control_permutations,
    )
    weight_permutations, weight_selection = pair_matched_permutations(
        after_parent,
        residual,
        stages=24,
        neighbors=neighbors,
        seed=seed + 2,
        family="shear",
        native_cache=native_cache,
    )
    functional_permutations, functional_selection = (
        functional_matched_permutations(
            after_parent,
            residual,
            inputs,
            pre_gelu,
            cproj_weight,
            stages=24,
            neighbors=neighbors,
            seed=seed + 3,
            native_cache=native_cache,
        )
    )
    weight_weight, weight_weight_fit = fit_pair_flow(
        after_parent,
        residual,
        weight_permutations,
        stages=24,
        family="shear",
    )
    functional_weight, functional_weight_fit = fit_pair_flow(
        after_parent,
        residual,
        functional_permutations,
        stages=24,
        family="shear",
    )
    weight_functional, weight_functional_fit = fit_functional_shear_flow(
        after_parent,
        residual,
        inputs,
        pre_gelu,
        cproj_weight,
        weight_permutations,
        stages=24,
    )
    functional_both, functional_both_fit = fit_functional_shear_flow(
        after_parent,
        residual,
        inputs,
        pre_gelu,
        cproj_weight,
        functional_permutations,
        stages=24,
    )
    rotations = {
        CONTROL: parent + control_residual,
        WEIGHT_SHEAR: parent + weight_weight,
        FUNCTIONAL_TOPOLOGY: parent + functional_weight,
        FUNCTIONAL_FIT: parent + weight_functional,
        FUNCTIONAL_BOTH: parent + functional_both,
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
        {"candidate": "fresh64_parent", **parent_diagnostics},
        {
            "candidate": CONTROL,
            "selection": control_selection,
            "fit": control_fit,
        },
        {
            "candidate": WEIGHT_SHEAR,
            "selection": weight_selection,
            "fit": weight_weight_fit,
        },
        {
            "candidate": FUNCTIONAL_TOPOLOGY,
            "selection": functional_selection,
            "fit": functional_weight_fit,
        },
        {
            "candidate": FUNCTIONAL_FIT,
            "selection": weight_selection,
            "fit": weight_functional_fit,
        },
        {
            "candidate": FUNCTIONAL_BOTH,
            "selection": functional_selection,
            "fit": functional_both_fit,
        },
    ]
    return candidates, diagnostics


def weighted(rows: list[dict[str, Any]], value: str, energy: str) -> float:
    weights = torch.tensor(
        [float(row[energy]) for row in rows], dtype=torch.float64
    )
    values = torch.tensor(
        [float(row[value]) for row in rows], dtype=torch.float64
    )
    return float((weights * values).sum() / weights.sum().clamp_min(1e-30))


def safe_ratio(numerator: float, denominator: float) -> float:
    if float(denominator) <= 0.0:
        return -1e30
    return float(numerator) / float(denominator)


def aggregate(
    metric_rows: list[dict[str, Any]],
    fit_rows: list[dict[str, Any]],
    *,
    minimum_functional_ratio: float,
    minimum_ce_descent_ratio: float,
    minimum_weight_ratio: float,
    maximum_determinant_error: float,
    maximum_condition_number: float,
) -> dict[str, Any]:
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    for candidate in CANDIDATES:
        metrics[candidate] = {}
        for window in WINDOWS:
            rows = [
                row
                for row in metric_rows
                if row["candidate"] == candidate and row["window"] == window
            ]
            metrics[candidate][window] = {
                "weight_fixed_scale_recovery": weighted(
                    rows, "weight_fixed_scale_recovery", "weight_target_energy"
                ),
                "post_gelu_fixed_scale_recovery": weighted(
                    rows, "post_gelu_fixed_scale_recovery", "post_gelu_target_energy"
                ),
                "mlp_output_fixed_scale_recovery": weighted(
                    rows, "mlp_output_fixed_scale_recovery", "mlp_output_target_energy"
                ),
                "predicted_ce_decrease": sum(
                    float(row["predicted_ce_decrease"]) for row in rows
                ),
            }
    primary = metrics[FUNCTIONAL_BOTH]
    weight_shear = metrics[WEIGHT_SHEAR]
    control = metrics[CONTROL]
    ratios = {
        window: {
            "mlp_output_vs_weight_shear": safe_ratio(
                primary[window]["mlp_output_fixed_scale_recovery"],
                weight_shear[window]["mlp_output_fixed_scale_recovery"],
            ),
            "mlp_output_vs_fresh88": safe_ratio(
                primary[window]["mlp_output_fixed_scale_recovery"],
                control[window]["mlp_output_fixed_scale_recovery"],
            ),
            "ce_descent_vs_weight_shear": safe_ratio(
                primary[window]["predicted_ce_decrease"],
                weight_shear[window]["predicted_ce_decrease"],
            ),
            "weight_vs_weight_shear": safe_ratio(
                primary[window]["weight_fixed_scale_recovery"],
                weight_shear[window]["weight_fixed_scale_recovery"],
            ),
        }
        for window in WINDOWS
    }
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
    passes = bool(
        stable
        and min(
            ratios[window]["mlp_output_vs_weight_shear"] for window in WINDOWS
        )
        >= minimum_functional_ratio
        and min(ratios[window]["mlp_output_vs_fresh88"] for window in WINDOWS)
        >= minimum_functional_ratio
        and min(
            ratios[window]["ce_descent_vs_weight_shear"] for window in WINDOWS
        )
        >= minimum_ce_descent_ratio
        and min(ratios[window]["weight_vs_weight_shear"] for window in WINDOWS)
        >= minimum_weight_ratio
    )
    if not stable:
        decision = "FUNCTIONAL_SHEAR_NUMERICAL_GATE_FAILED"
    elif passes:
        decision = "PROMOTE_FUNCTIONAL_SHEAR_TO_HELDOUT_CE"
    else:
        decision = "REJECT_ACTIVATION_WEIGHTED_SHEAR"
    return {
        "decision": decision,
        "parameter_updates": 0,
        "metrics": metrics,
        "primary_ratios": ratios,
        "gates": {"numerically_stable": stable, "primary_passes": passes},
        "thresholds": {
            "minimum_functional_ratio": minimum_functional_ratio,
            "minimum_ce_descent_ratio": minimum_ce_descent_ratio,
            "minimum_weight_ratio": minimum_weight_ratio,
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
    batches = {
        "fit": fixed_batches(
            args.data_dir,
            "train",
            batch_size=int(protocol["batch_size"]),
            block_size=int(protocol["block_size"]),
            batches=int(protocol["batches_per_window"]),
            seed=int(protocol["fit_train_seed"]),
        ),
        "holdout": fixed_batches(
            args.data_dir,
            "train",
            batch_size=int(protocol["batch_size"]),
            block_size=int(protocol["block_size"]),
            batches=int(protocol["batches_per_window"]),
            seed=int(protocol["holdout_train_seed"]),
        ),
    }
    model, optimizer, checkpoint = load_model_and_optimizer(
        args.checkpoint, config, args.device
    )
    collected: dict[str, dict[str, Any]] = {}
    for window in WINDOWS:
        loss, gradients, inputs, pre_gelu = collect_window(
            model,
            batches[window],
            layers,
            device=args.device,
            dtype=torch.bfloat16,
        )
        collected[window] = {
            "loss": loss,
            "gradients": gradients,
            "inputs": inputs,
            "pre_gelu": pre_gelu,
        }
    sample_cap = int(protocol["functional_sample_cap"])
    sample_indices_sha: dict[str, str] = {}
    sampled: dict[str, dict[str, dict[int, torch.Tensor]]] = {}
    for window_index, window in enumerate(WINDOWS):
        sampled[window] = {"inputs": {}, "pre_gelu": {}}
        for layer in layers:
            inputs, pre_gelu, sha = sample_aligned(
                collected[window]["inputs"][layer],
                collected[window]["pre_gelu"][layer],
                sample_cap=sample_cap,
                seed=int(protocol["functional_sample_seed"]) + window_index,
            )
            sampled[window]["inputs"][layer] = inputs
            sampled[window]["pre_gelu"][layer] = pre_gelu
            sample_indices_sha[f"{window}_layer{layer}"] = sha
    updates: dict[str, dict[int, torch.Tensor]] = {
        candidate: {} for candidate in CANDIDATES
    }
    dense_updates: dict[int, torch.Tensor] = {}
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
            collected["fit"]["gradients"][layer].to(weight.device),
            buffer,
            learning_rate=float(group["lr"]),
            momentum=float(group["momentum"]),
            weight_decay=float(group["weight_decay"]),
            ns_steps=int(group["ns_steps"]),
        )
        polar_descent = (
            descent + float(group["weight_decay"]) * weight.detach().float()
        )
        fitted, diagnostics = build_candidates(
            weight.detach(),
            dense_update,
            polar_descent,
            sampled["fit"]["inputs"][layer],
            sampled["fit"]["pre_gelu"][layer],
            model.transformer.h[layer].mlp.c_proj.weight.detach(),
            neighbors=int(protocol["matching_neighbors"]),
            seed=int(protocol["matching_seed"]) + layer * 1009,
            learning_rate=float(group["lr"]),
            weight_decay=float(group["weight_decay"]),
            native_cache=args.native_cache,
        )
        dense_updates[layer] = dense_update.cpu()
        for candidate, update in fitted.items():
            updates[candidate][layer] = update.cpu()
        fit_rows.extend({"layer": layer, **row} for row in diagnostics)
        optimizer_rows.append({"layer": layer, **optimizer_diag})
        print(
            json.dumps(
                {"layer_complete": layer, "layers_total": len(layers)},
                sort_keys=True,
            ),
            flush=True,
        )
    metric_rows: list[dict[str, Any]] = []
    for window in WINDOWS:
        for layer in layers:
            target = dense_updates[layer]
            cproj = model.transformer.h[layer].mlp.c_proj.weight.detach().cpu()
            for candidate in CANDIDATES:
                update = updates[candidate][layer]
                weight_metrics = direction_metrics(target, update)
                functional = activation_effect_metrics(
                    sampled[window]["inputs"][layer],
                    sampled[window]["pre_gelu"][layer],
                    cproj,
                    target,
                    update,
                    device=args.device,
                )
                gradient = collected[window]["gradients"][layer]
                metric_rows.append(
                    {
                        "window": window,
                        "layer": layer,
                        "candidate": candidate,
                        "weight_target_energy": weight_metrics["target_energy"],
                        "weight_fixed_scale_recovery": weight_metrics[
                            "fixed_scale_recovery"
                        ],
                        "post_gelu_target_energy": functional["post_gelu"][
                            "target_energy"
                        ],
                        "post_gelu_fixed_scale_recovery": functional[
                            "post_gelu"
                        ]["fixed_scale_recovery"],
                        "mlp_output_target_energy": functional["mlp_output"][
                            "target_energy"
                        ],
                        "mlp_output_fixed_scale_recovery": functional[
                            "mlp_output"
                        ]["fixed_scale_recovery"],
                        "predicted_ce_decrease": float(
                            -(gradient.double() * update.double()).sum()
                        ),
                    }
                )
    result = aggregate(
        metric_rows,
        fit_rows,
        minimum_functional_ratio=float(rule["minimum_functional_ratio"]),
        minimum_ce_descent_ratio=float(rule["minimum_ce_descent_ratio"]),
        minimum_weight_ratio=float(rule["minimum_weight_ratio"]),
        maximum_determinant_error=float(rule["maximum_determinant_error"]),
        maximum_condition_number=float(rule["maximum_condition_number"]),
    )
    result["fit_loss_bfloat16"] = float(collected["fit"]["loss"])
    result["holdout_loss_bfloat16"] = float(collected["holdout"]["loss"])
    args.output.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": args.output / "cfc_functional_shear_fit_metrics.csv",
        "fits": args.output / "cfc_functional_shear_fit_fits.json",
        "optimizer": args.output / "cfc_functional_shear_fit_optimizer.csv",
        "aggregate": args.output / "cfc_functional_shear_fit_aggregate.json",
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
        "sample_indices_sha256": sample_indices_sha,
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
    metadata_path = args.output / "cfc_functional_shear_fit_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "aggregate": str(paths["aggregate"]),
                "metadata": str(metadata_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
