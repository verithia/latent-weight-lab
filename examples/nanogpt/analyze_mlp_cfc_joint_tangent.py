#!/usr/bin/env python3
"""Jointly solve fixed sparse output/input tangents for ``mlp.c_fc``.

The prior equal-coordinate screen used one diagonal output-then-input sweep.
This diagnostic freezes the same task-selected edge family, then solves all
selected output/input tangent coordinates together with preconditioned CG.
It distinguishes sparse-connectivity capacity from diagonal-solver error.
No model or optimizer parameter is updated.
"""

from __future__ import annotations

import argparse
import json
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
    diagonal_metric_causal_givens_update,
    exact_muon_update,
    file_sha256,
    fixed_batches,
    load_model_and_optimizer,
)
from examples.nanogpt.analyze_mlp_cfc_residual_structure import (
    residual_metrics,
    validate_identity,
    write_csv,
)
from examples.nanogpt.analyze_mlp_cfc_trust_radius import (
    collect_gradient_window,
    repeated_losses,
    summarize,
)
from examples.nanogpt.fast_task_matching import (
    fast_muon_matched_permutations,
)


SCHEMA_VERSION = "nanogpt_mlp_cfc_joint_tangent_v1"
DIAGONAL_CONTROL = "diagonal_output88"
DENSE = "dense_exact"
JOINT_OUTPUT = "joint_output88"
JOINT_BILATERAL = (
    "joint_output80_input32",
    "joint_output72_input64",
)
CANDIDATES = (DIAGONAL_CONTROL, DENSE, JOINT_OUTPUT, *JOINT_BILATERAL)


def git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def pairs_from_permutations(permutations: torch.Tensor) -> torch.Tensor:
    if permutations.ndim != 2 or permutations.shape[1] % 2:
        raise ValueError("permutations must be shaped [stages, even_width]")
    return permutations.reshape(-1, 2).to(dtype=torch.long)


def _skew_from_coordinates(
    size: int,
    pairs: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    output_side: bool,
    device: torch.device,
) -> torch.Tensor:
    skew = torch.zeros(size, size, device=device, dtype=torch.float32)
    if not pairs.numel():
        return skew
    pairs = pairs.to(device=device)
    coordinates = coordinates.to(device=device, dtype=torch.float32)
    left, right = pairs[:, 0], pairs[:, 1]
    if output_side:
        skew.index_put_((left, right), -coordinates, accumulate=True)
        skew.index_put_((right, left), coordinates, accumulate=True)
    else:
        skew.index_put_((left, right), coordinates, accumulate=True)
        skew.index_put_((right, left), -coordinates, accumulate=True)
    return skew


def joint_jvp(
    weight: torch.Tensor,
    output_pairs: torch.Tensor,
    input_pairs: torch.Tensor,
    coordinates: torch.Tensor,
) -> torch.Tensor:
    """Apply the linear tangent ``Omega_out W + W Omega_in``."""
    output_count = int(output_pairs.shape[0])
    output_coordinates = coordinates[:output_count]
    input_coordinates = coordinates[output_count:]
    omega_output = _skew_from_coordinates(
        weight.shape[0],
        output_pairs,
        output_coordinates,
        output_side=True,
        device=weight.device,
    )
    omega_input = _skew_from_coordinates(
        weight.shape[1],
        input_pairs,
        input_coordinates,
        output_side=False,
        device=weight.device,
    )
    return omega_output @ weight.float() + weight.float() @ omega_input


def joint_vjp(
    weight: torch.Tensor,
    output_pairs: torch.Tensor,
    input_pairs: torch.Tensor,
    cotangent: torch.Tensor,
) -> torch.Tensor:
    """Apply the exact transpose of :func:`joint_jvp`."""
    values: list[torch.Tensor] = []
    if output_pairs.numel():
        output_pairs = output_pairs.to(weight.device)
        cross = cotangent.float() @ weight.float().T
        left, right = output_pairs[:, 0], output_pairs[:, 1]
        values.append(cross[right, left] - cross[left, right])
    if input_pairs.numel():
        input_pairs = input_pairs.to(weight.device)
        cross = weight.float().T @ cotangent.float()
        left, right = input_pairs[:, 0], input_pairs[:, 1]
        values.append(cross[left, right] - cross[right, left])
    if not values:
        return torch.empty(0, device=weight.device, dtype=torch.float32)
    return torch.cat(values)


def joint_diagonal(
    weight: torch.Tensor,
    output_pairs: torch.Tensor,
    input_pairs: torch.Tensor,
) -> torch.Tensor:
    values: list[torch.Tensor] = []
    if output_pairs.numel():
        pairs = output_pairs.to(weight.device)
        row_energy = weight.float().square().sum(dim=1)
        values.append(row_energy[pairs[:, 0]] + row_energy[pairs[:, 1]])
    if input_pairs.numel():
        pairs = input_pairs.to(weight.device)
        column_energy = weight.float().square().sum(dim=0)
        values.append(
            column_energy[pairs[:, 0]] + column_energy[pairs[:, 1]]
        )
    return torch.cat(values).clamp_min(1e-30)


@torch.no_grad()
def solve_joint_tangent(
    weight: torch.Tensor,
    target: torch.Tensor,
    output_pairs: torch.Tensor,
    input_pairs: torch.Tensor,
    *,
    iterations: int,
    damping: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Solve damped joint normal equations with diagonal-preconditioned CG."""
    if iterations <= 0 or not 0.0 < damping < 1.0:
        raise ValueError("invalid joint tangent solver settings")
    diagonal = joint_diagonal(weight, output_pairs, input_pairs)
    rhs = joint_vjp(weight, output_pairs, input_pairs, target)

    def normal(values: torch.Tensor) -> torch.Tensor:
        tangent = joint_jvp(weight, output_pairs, input_pairs, values)
        return (
            joint_vjp(weight, output_pairs, input_pairs, tangent)
            + float(damping) * diagonal * values
        )

    coordinates = torch.zeros_like(rhs)
    residual = rhs.clone()
    preconditioned = residual / ((1.0 + float(damping)) * diagonal)
    direction = preconditioned.clone()
    residual_dot = torch.dot(residual, preconditioned)
    initial_rhs_norm = float(rhs.norm())
    history: list[dict[str, float | int]] = []
    for iteration in range(int(iterations)):
        applied = normal(direction)
        denominator = torch.dot(direction, applied).clamp_min(1e-30)
        alpha = residual_dot / denominator
        coordinates.add_(direction, alpha=alpha)
        residual.add_(applied, alpha=-alpha)
        next_preconditioned = residual / (
            (1.0 + float(damping)) * diagonal
        )
        next_dot = torch.dot(residual, next_preconditioned)
        history.append(
            {
                "iteration": iteration + 1,
                "relative_normal_residual": float(
                    residual.norm() / rhs.norm().clamp_min(1e-30)
                ),
            }
        )
        beta = next_dot / residual_dot.clamp_min(1e-30)
        direction.mul_(beta).add_(next_preconditioned)
        preconditioned = next_preconditioned
        residual_dot = next_dot
    tangent = joint_jvp(weight, output_pairs, input_pairs, coordinates)
    return tangent, {
        "coordinates": int(coordinates.numel()),
        "output_coordinates": int(output_pairs.shape[0]),
        "input_coordinates": int(input_pairs.shape[0]),
        "iterations": int(iterations),
        "damping": float(damping),
        "rhs_norm": initial_rhs_norm,
        "coordinate_rms": float(coordinates.square().mean().sqrt()),
        "coordinate_max_abs": float(coordinates.abs().max()),
        "history": history,
    }


@torch.no_grad()
def select_shared_connectivity(
    weight: torch.Tensor,
    rotation_target: torch.Tensor,
    polar_direction: torch.Tensor,
    *,
    neighbors: int,
    seed: int,
    native_cache: Path | None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, list[dict[str, Any]]]:
    """Select one shared 88-output/64-input task-matched edge family."""
    source_t = weight.float().T.contiguous()
    target_t = rotation_target.float().T.contiguous()
    polar_t = polar_direction.float().T.contiguous()
    parent_permutations, parent_selection = fast_muon_matched_permutations(
        source_t,
        polar_t,
        stages=64,
        neighbors=neighbors,
        seed=seed,
        cache_dir=native_cache,
    )
    parent_rotation, parent_fit = diagonal_metric_causal_givens_update(
        source_t,
        target_t,
        stages=64,
        seed=seed,
        permutations=parent_permutations,
    )
    after_parent = source_t + parent_rotation
    residual = target_t - parent_rotation
    residual_permutations, residual_selection = (
        fast_muon_matched_permutations(
            after_parent,
            residual,
            stages=24,
            neighbors=neighbors,
            seed=seed + 1,
            cache_dir=native_cache,
        )
    )
    residual_rotation, residual_fit = diagonal_metric_causal_givens_update(
        after_parent,
        residual,
        stages=24,
        seed=seed + 1,
        permutations=residual_permutations,
    )
    after_output = (after_parent + residual_rotation).T.contiguous()
    output_rotation = after_output - weight.float()
    input_residual = rotation_target.float() - output_rotation
    input_permutations, input_selection = fast_muon_matched_permutations(
        after_output,
        input_residual,
        stages=64,
        neighbors=neighbors,
        seed=seed + 101,
        cache_dir=native_cache,
    )
    records = [
        {
            "selection": "output_parent64",
            "axis": "output",
            "stages": 64,
            "fit": parent_fit,
            **parent_selection,
        },
        {
            "selection": "output_residual24",
            "axis": "output",
            "stages": 24,
            "fit": residual_fit,
            **residual_selection,
        },
        {
            "selection": "input_residual64",
            "axis": "input",
            "stages": 64,
            **input_selection,
        },
    ]
    return {
        "output_parent64": parent_permutations,
        "output_residual24": residual_permutations,
        "input_residual64": input_permutations,
    }, output_rotation, records


def median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def aggregate(
    loss_rows: list[dict[str, Any]],
    *,
    windows: list[str],
    maximum_replicate_range: float,
    minimum_recovery: float,
    median_recovery: float,
) -> dict[str, Any]:
    summaries: dict[str, dict[str, dict[str, float]]] = {}
    for window in windows:
        summaries[window] = {}
        for candidate in ("baseline", *CANDIDATES):
            summaries[window][candidate] = summarize(
                [
                    float(row["loss"])
                    for row in loss_rows
                    if row["window"] == window
                    and row["candidate"] == candidate
                ]
            )
    stable = all(
        value["range"] <= maximum_replicate_range
        for window in summaries.values()
        for value in window.values()
    )
    dense_positive = all(
        values[DENSE]["maximum"]
        < values[DIAGONAL_CONTROL]["minimum"]
        for values in summaries.values()
    )
    results: dict[str, dict[str, Any]] = {}
    for candidate in (JOINT_OUTPUT, *JOINT_BILATERAL):
        recoveries = {}
        for window, values in summaries.items():
            gap = (
                values[DIAGONAL_CONTROL]["mean"]
                - values[DENSE]["mean"]
            )
            recoveries[window] = (
                values[DIAGONAL_CONTROL]["mean"]
                - values[candidate]["mean"]
            ) / max(gap, 1e-30)
        minimum = min(recoveries.values())
        med = median(list(recoveries.values()))
        beats_control = all(
            values[candidate]["maximum"]
            < values[DIAGONAL_CONTROL]["minimum"]
            for values in summaries.values()
        )
        results[candidate] = {
            "recovery_by_window": recoveries,
            "minimum_recovery": minimum,
            "median_recovery": med,
            "beats_diagonal_output88_every_window": beats_control,
            "sufficient": all(
                (
                    stable,
                    dense_positive,
                    beats_control,
                    minimum >= minimum_recovery,
                    med >= median_recovery,
                )
            ),
        }
    bilateral_over_output = {
        candidate: all(
            values[candidate]["maximum"]
            < values[JOINT_OUTPUT]["minimum"]
            for values in summaries.values()
        )
        for candidate in JOINT_BILATERAL
    }
    eligible_bilateral = [
        candidate
        for candidate in JOINT_BILATERAL
        if results[candidate]["sufficient"]
        and bilateral_over_output[candidate]
    ]
    selected = (
        max(
            eligible_bilateral,
            key=lambda candidate: (
                results[candidate]["minimum_recovery"],
                results[candidate]["median_recovery"],
            ),
        )
        if eligible_bilateral
        else None
    )
    if not dense_positive:
        decision = "DENSE_EXACT_NOT_POSITIVE_CONTROL"
    elif selected is not None:
        decision = f"SELECT_{selected.upper()}_JOINT_TANGENT"
    elif results[JOINT_OUTPUT]["sufficient"]:
        decision = "JOINT_SOLVER_SUFFICIENT_WITHOUT_INPUT_SIDE"
    else:
        decision = "REJECT_FIXED_SPARSE_JOINT_TANGENT_CAPACITY"
    return {
        "decision": decision,
        "selected_candidate": selected,
        "summaries": summaries,
        "candidate_results": results,
        "bilateral_beats_joint_output_every_window": bilateral_over_output,
        "gates": {
            "numerically_stable": stable,
            "dense_beats_diagonal_output88_every_window": dense_positive,
        },
        "thresholds": {
            "maximum_replicate_range": maximum_replicate_range,
            "minimum_recovery": minimum_recovery,
            "median_recovery": median_recovery,
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
    layers = [int(value) for value in protocol["layers"]]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fit_batches = fixed_batches(
        args.data_dir,
        "train",
        batch_size=int(protocol["batch_size"]),
        block_size=int(protocol["block_size"]),
        batches=int(protocol["fit_batches"]),
        seed=int(protocol["fit_train_seed"]),
    )
    windows = [
        f"validation_{index + 1}"
        for index in range(len(protocol["validation_seeds"]))
    ]
    validation_batches = {
        window: fixed_batches(
            args.data_dir,
            "val",
            batch_size=int(protocol["batch_size"]),
            block_size=int(protocol["block_size"]),
            batches=int(protocol["validation_batches_per_window"]),
            seed=int(seed),
        )
        for window, seed in zip(
            windows, protocol["validation_seeds"], strict=True
        )
    }
    model, optimizer, checkpoint = load_model_and_optimizer(
        args.checkpoint, config, args.device
    )
    fit_loss, gradients = collect_gradient_window(
        model,
        fit_batches,
        layers,
        device=args.device,
        dtype=torch.bfloat16,
    )
    updates: dict[str, dict[int, torch.Tensor]] = {
        candidate: {} for candidate in CANDIDATES
    }
    metric_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    solver_rows: list[dict[str, Any]] = []
    for layer in layers:
        weight = model.transformer.h[layer].mlp.c_fc.weight
        owner, group = _optimizer_and_group_for_parameter(optimizer, weight)
        buffer = owner.state[weight].get("momentum_buffer")
        if buffer is None:
            raise RuntimeError(f"missing c_fc momentum at layer {layer}")
        dense_update, descent, optimizer_diagnostics = exact_muon_update(
            weight.detach(),
            gradients[layer].to(weight.device),
            buffer,
            learning_rate=float(group["lr"]),
            momentum=float(group["momentum"]),
            weight_decay=float(group["weight_decay"]),
            ns_steps=int(group["ns_steps"]),
        )
        polar_direction = (
            descent
            + float(group["weight_decay"]) * weight.detach().float()
        )
        rotation_target = float(group["lr"]) * polar_direction
        connectivity, diagonal_rotation, selections = select_shared_connectivity(
            weight.detach(),
            rotation_target,
            polar_direction,
            neighbors=int(protocol["matching_neighbors"]),
            seed=int(protocol["matching_seed"]) + layer * 1009,
            native_cache=args.native_cache,
        )
        selection_rows.extend(
            {"layer": layer, **selection} for selection in selections
        )
        output_all = torch.cat(
            (
                connectivity["output_parent64"],
                connectivity["output_residual24"],
            ),
            dim=0,
        )
        input_all = connectivity["input_residual64"]
        empty_input = torch.empty(0, 2, dtype=torch.long)
        candidate_pairs = {
            JOINT_OUTPUT: (
                pairs_from_permutations(output_all),
                empty_input,
            ),
            "joint_output80_input32": (
                pairs_from_permutations(output_all[:80]),
                pairs_from_permutations(input_all[:32]),
            ),
            "joint_output72_input64": (
                pairs_from_permutations(output_all[:72]),
                pairs_from_permutations(input_all[:64]),
            ),
        }
        diagonal_update = _weight_decay_after_rotation(
            weight.detach(),
            diagonal_rotation,
            learning_rate=float(group["lr"]),
            weight_decay=float(group["weight_decay"]),
        )
        updates[DIAGONAL_CONTROL][layer] = diagonal_update.cpu()
        updates[DENSE][layer] = dense_update.float().cpu()
        for candidate, (output_pairs, input_pairs) in candidate_pairs.items():
            tangent, solver = solve_joint_tangent(
                weight.detach().float(),
                rotation_target.float(),
                output_pairs,
                input_pairs,
                iterations=int(protocol["cg_iterations"]),
                damping=float(protocol["cg_damping"]),
            )
            update = _weight_decay_after_rotation(
                weight.detach(),
                tangent,
                learning_rate=float(group["lr"]),
                weight_decay=float(group["weight_decay"]),
            )
            updates[candidate][layer] = update.cpu()
            solver_rows.append({"layer": layer, "candidate": candidate, **solver})
        for candidate in CANDIDATES:
            candidate_update = updates[candidate][layer].to(weight.device)
            coordinates = (
                int(weight.numel())
                if candidate == DENSE
                else 135168
            )
            metric_rows.append(
                {
                    "layer": layer,
                    "candidate": candidate,
                    "coordinates_per_layer": coordinates,
                    "coordinate_fraction": coordinates / weight.numel(),
                    **residual_metrics(dense_update, candidate_update),
                    **{
                        f"optimizer_{key}": value
                        for key, value in optimizer_diagnostics.items()
                    },
                }
            )
        print(
            json.dumps(
                {
                    "layer_complete": layer,
                    "layers_total": len(layers),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    repeats = int(protocol["evaluation_repeats"])
    loss_rows: list[dict[str, Any]] = []
    for window, batches in validation_batches.items():
        baseline = repeated_losses(
            model,
            batches,
            None,
            repeats=repeats,
            device=args.device,
            dtype=torch.float32,
        )
        for repeat, loss in enumerate(baseline):
            loss_rows.append(
                {
                    "window": window,
                    "candidate": "baseline",
                    "repeat": repeat,
                    "loss": loss,
                }
            )
        for candidate in CANDIDATES:
            values = repeated_losses(
                model,
                batches,
                updates[candidate],
                repeats=repeats,
                device=args.device,
                dtype=torch.float32,
            )
            for repeat, loss in enumerate(values):
                loss_rows.append(
                    {
                        "window": window,
                        "candidate": candidate,
                        "repeat": repeat,
                        "loss": loss,
                    }
                )
    result = aggregate(
        loss_rows,
        windows=windows,
        maximum_replicate_range=float(rule["maximum_replicate_range"]),
        minimum_recovery=float(rule["minimum_recovery"]),
        median_recovery=float(rule["median_recovery"]),
    )
    result.update(
        {
            "fit_gradient_loss_bfloat16": fit_loss,
            "parameter_updates": 0,
            "equal_coordinates_per_layer": 135168,
        }
    )
    args.output.mkdir(parents=True, exist_ok=True)
    losses_path = args.output / "cfc_joint_tangent_losses.csv"
    metrics_path = args.output / "cfc_joint_tangent_metrics.csv"
    selections_path = args.output / "cfc_joint_tangent_selections.json"
    solvers_path = args.output / "cfc_joint_tangent_solvers.json"
    aggregate_path = args.output / "cfc_joint_tangent_aggregate.json"
    write_csv(losses_path, loss_rows)
    write_csv(metrics_path, metric_rows)
    selections_path.write_text(
        json.dumps(selection_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    solvers_path.write_text(
        json.dumps(solver_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    aggregate_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "decision": result["decision"],
        "parameter_updates": 0,
        "checkpoint_next_iter": int(checkpoint["next_iter"]),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "config_sha256": file_sha256(args.config),
        "dataset_manifest_sha256": file_sha256(
            args.data_dir / "manifest.json"
        ),
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
            "losses_sha256": file_sha256(losses_path),
            "metrics_sha256": file_sha256(metrics_path),
            "selections_sha256": file_sha256(selections_path),
            "solvers_sha256": file_sha256(solvers_path),
            "aggregate_sha256": file_sha256(aggregate_path),
        },
        "limitations": plan["limitations"],
    }
    metadata_path = args.output / "cfc_joint_tangent_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "selected_candidate": result["selected_candidate"],
                "aggregate": str(aggregate_path),
                "metadata": str(metadata_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
