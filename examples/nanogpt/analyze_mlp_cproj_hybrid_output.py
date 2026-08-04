#!/usr/bin/env python3
"""Test equal-budget compositions of task-Givens and directed c_proj maps."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.nanogpt.analyze_mlp_activation_update_alignment import load_snapshot, model_from_snapshot
from examples.nanogpt.analyze_mlp_cproj_activation_weighted_output_selector import (
    all_finite,
    file_sha256,
    fit_frobenius_pass,
    git_commit,
    load_probe,
    output_residual_energy,
    parameter_name,
    shared_hidden_chart,
)
from examples.nanogpt.analyze_mlp_cproj_global_directed_affine_output import (
    fit_global_directed_map,
)
from examples.nanogpt.analyze_mlp_cproj_global_directed_minimax_output import (
    minimax_support_score,
)
from examples.nanogpt.analyze_mlp_cproj_task_gradient_output_selector import (
    fit_task_gradient_hybrid_pass,
)
from examples.nanogpt.analyze_mlp_muon_matched_functional_metric import (
    evaluate_and_collect,
    evaluate_with_updates,
    fixed_scale_recovery,
    output_space_metrics,
    task_descent_metrics,
)
from examples.nanogpt.analyze_mlp_task_gradient_direction import collect_cproj_gradients
from examples.nanogpt.analyze_parameter_trajectory import write_csv
from examples.nanogpt.analyze_residual_compatibility import fixed_validation_batches
from examples.nanogpt.muon_matched_givens import apply_givens_flow


SCHEMA_VERSION = "mai_124m_mlp_cproj_hybrid_output_v1"
EXPECTED_PLAN_SCHEMA = "mai_124m_mlp_cproj_hybrid_output_plan_v1"
ARMS = (
    "frobenius_output32",
    "task_gradient_output32",
    "global_directed_minimax16",
    "task16_then_minimax8",
    "minimax8_then_task16",
)
CANDIDATES = ARMS[-2:]
WINDOWS = ("fit", "holdout")


def validate_plan(plan: dict[str, Any]) -> None:
    analysis = plan.get("analysis", {})
    expected = {
        "schema_version": EXPECTED_PLAN_SCHEMA,
        "layers": [0, 3, 6, 9, 11],
        "phases": [[0, 60], [60, 120], [120, 180], [180, 238]],
        "fit_window": {"split": "validation", "seed": 20260804, "batch_size": 2, "block_size": 256, "batches": 4, "rows_per_layer": 2048},
        "holdout_window": {"split": "validation", "seed": 20260805, "batch_size": 2, "block_size": 256, "batches": 4, "rows_per_layer": 2048},
        "shared_hidden_chart": {
            "parent_stages": 64, "residual_stages": 24, "neighbors": 64,
            "matching_seed": 20260804, "coordinates_per_layer": 135168,
            "feedback": "zero for this one-step prospective diagnostic",
            "weight_decay_application": "identical production ordering in every arm",
        },
        "output_budget": {
            "source_channels": 768, "coordinates_per_layer": 12288,
            "total_coordinates_per_layer": 147456, "control_output_stages": 32,
            "full_task_givens_stages": 32, "full_minimax_incoming_per_target": 16,
            "hybrid_task_givens_stages": 16, "hybrid_minimax_incoming_per_target": 8,
            "task_givens_coordinates": 6144, "directed_coordinates": 6144,
        },
        "arms": list(ARMS),
        "selection_order": list(CANDIDATES),
        "parameter_updates": 0,
    }
    observed = {
        "schema_version": plan.get("schema_version"),
        "layers": analysis.get("layers"), "phases": analysis.get("phases"),
        "fit_window": analysis.get("fit_window"), "holdout_window": analysis.get("holdout_window"),
        "shared_hidden_chart": analysis.get("shared_hidden_chart"),
        "output_budget": analysis.get("output_budget"), "arms": analysis.get("arms"),
        "selection_order": analysis.get("selection_order"),
        "parameter_updates": analysis.get("parameter_updates"),
    }
    if observed != expected:
        raise ValueError("hybrid-output plan does not match the immutable v1 contract")
    if plan.get("authorization", {}).get("implement_and_run_zero_update_analysis") is not True:
        raise ValueError("zero-update hybrid analysis is not authorized")


def task_givens_component(
    source: torch.Tensor,
    residual: torch.Tensor,
    gradient: torch.Tensor,
    *,
    stages: int,
    neighbors: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    updated, raw = fit_task_gradient_hybrid_pass(
        source, residual, gradient, stages=stages, neighbors=neighbors, seed=seed
    )
    permutations = raw["permutations"].to(source.device)
    angles = raw["angles"].to(source.device)
    identity = torch.eye(source.shape[1], device=source.device, dtype=source.dtype)
    transform = apply_givens_flow(
        identity, angles, permutations, torch.argsort(permutations, dim=1)
    )
    torch.testing.assert_close(updated, source @ transform, rtol=2e-5, atol=2e-6)
    diagnostics = {
        key: value
        for key, value in raw.items()
        if key not in {"permutations", "angles"}
    }
    return updated, transform, diagnostics


def bound_composed_transform(
    source: torch.Tensor,
    transform: torch.Tensor,
    *,
    trust_output_energy: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | bool]]:
    identity = torch.eye(transform.shape[0], device=transform.device, dtype=transform.dtype)
    raw_delta = source.float() @ (transform.float() - identity)
    raw_energy = float(raw_delta.double().square().sum())
    scale = min(1.0, math.sqrt(float(trust_output_energy) / max(raw_energy, 1e-30)))
    bounded = identity + scale * (transform.float() - identity)
    updated = source.float() @ bounded
    bounded_energy = float((updated - source.float()).double().square().sum())
    minimum_singular = float(torch.linalg.svdvals(bounded.double()).min())
    diagnostics: dict[str, float | bool] = {
        "raw_output_delta_energy": raw_energy,
        "bounded_output_delta_energy": bounded_energy,
        "trust_output_energy": float(trust_output_energy),
        "trust_scale": scale,
        "trust_energy_obeyed": bounded_energy <= trust_output_energy + max(1e-12, 1e-5 * trust_output_energy),
        "minimum_singular_value": minimum_singular,
    }
    if not all_finite(diagnostics) or not torch.isfinite(updated).all():
        raise ValueError("bounded hybrid transform is nonfinite")
    return updated, bounded, diagnostics


def fit_hybrid(
    source: torch.Tensor,
    residual: torch.Tensor,
    activation: torch.Tensor,
    train_gradient: torch.Tensor,
    fit_gradient: torch.Tensor,
    *,
    order: str,
    task_stages: int,
    incoming: int,
    neighbors: int,
    seed: int,
    trust_output_energy: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    identity = torch.eye(source.shape[1], device=source.device, dtype=source.dtype)
    if order == "task16_then_minimax8":
        first, task_transform, task_diag = task_givens_component(
            source, residual, fit_gradient.T, stages=task_stages,
            neighbors=neighbors, seed=seed,
        )
        remaining = residual - (first - source)
        score, score_diag = minimax_support_score(
            first, remaining, activation, train_gradient, fit_gradient
        )
        second, _supports, directed_diag, mapping = fit_global_directed_map(
            first, remaining, activation, score, incoming=incoming,
            trust_output_energy=trust_output_energy, return_mapping=True,
        )
        transform = task_transform @ (identity + mapping)
    elif order == "minimax8_then_task16":
        score, score_diag = minimax_support_score(
            source, residual, activation, train_gradient, fit_gradient
        )
        first, _supports, directed_diag, mapping = fit_global_directed_map(
            source, residual, activation, score, incoming=incoming,
            trust_output_energy=trust_output_energy, return_mapping=True,
        )
        remaining = residual - (first - source)
        second, task_transform, task_diag = task_givens_component(
            first, remaining, fit_gradient.T, stages=task_stages,
            neighbors=neighbors, seed=seed,
        )
        transform = (identity + mapping) @ task_transform
    else:
        raise ValueError(f"unsupported hybrid order: {order}")
    torch.testing.assert_close(second, source @ transform, rtol=2e-5, atol=2e-6)
    updated, _bounded, trust_diag = bound_composed_transform(
        source, transform, trust_output_energy=trust_output_energy
    )
    return updated, {
        "coordinates": int(task_diag["coordinates"]) + int(directed_diag["coordinates"]),
        "task_maximum_abs_angle": float(task_diag["maximum_abs_angle"]),
        "task_mean_abs_angle": float(task_diag["mean_abs_angle"]),
        "directed_component_trust_scale": float(directed_diag["trust_scale"]),
        "directed_component_minimum_singular_value": float(directed_diag["minimum_singular_value_i_plus_b"]),
        "positive_task_agreement_fraction": float(score_diag["positive_task_agreement_fraction"]),
        **trust_diag,
    }


def aggregate_results(
    rows: list[dict[str, Any]],
    finite_rows: list[dict[str, Any]],
    chart_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_arm: dict[str, Any] = {}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        by_arm[arm] = {
            "coordinates_per_layer": sorted({int(row["coordinates_per_layer"]) for row in selected}),
            "activation_output_residual_energy": {
                window: sum(float(row["activation_output_residual_energy"]) for row in selected if row["window"] == window)
                for window in WINDOWS
            },
            "task_gradient_predicted_ce_decrease": {
                window: sum(float(row["validation_gradient_predicted_ce_decrease"]) for row in selected if row["window"] == window)
                for window in WINDOWS
            },
        }
    index = {
        (int(row["phase_start"]), str(row["window"]), str(row["arm"])): float(row["loss"])
        for row in finite_rows if row["arm"] in ARMS
    }
    control = by_arm[ARMS[0]]
    mean_by_arm = {
        arm: sum(value for (phase, window, name), value in index.items() if name == arm) / 8.0
        for arm in ARMS
    }
    summaries: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    comparisons: dict[str, list[dict[str, Any]]] = {}
    for candidate in CANDIDATES:
        comp: list[dict[str, Any]] = []
        for phase in sorted({key[0] for key in index}):
            for window in WINDOWS:
                candidate_loss = index[(phase, window, candidate)]
                control_loss = index[(phase, window, ARMS[0])]
                comp.append({
                    "phase_start": phase, "window": window,
                    "candidate_minus_control": candidate_loss - control_loss,
                    "candidate_minus_task32": candidate_loss - index[(phase, window, ARMS[1])],
                    "candidate_minus_minimax16": candidate_loss - index[(phase, window, ARMS[2])],
                    "candidate_wins_control": candidate_loss < control_loss,
                })
        comparisons[candidate] = comp
        metrics = by_arm[candidate]
        residual_ratio = float(metrics["activation_output_residual_energy"]["holdout"]) / max(float(control["activation_output_residual_energy"]["holdout"]), 1e-30)
        task = float(metrics["task_gradient_predicted_ce_decrease"]["holdout"])
        control_task = float(control["task_gradient_predicted_ce_decrease"]["holdout"])
        chart = [row for row in chart_rows if row["arm"] == candidate]
        wins = sum(bool(row["candidate_wins_control"]) for row in comp)
        holdout_wins = sum(bool(row["candidate_wins_control"]) for row in comp if row["window"] == "holdout")
        mean_candidate = mean_by_arm[candidate]
        gate = {
            "all_outputs_scores_solves_transformations_and_metrics_finite": all_finite({"rows": rows, "finite": finite_rows, "chart": chart_rows}),
            "equal_coordinate_budget": all(by_arm[arm]["coordinates_per_layer"] == [147456] for arm in ARMS),
            "combined_output_energy_trust_obeyed_every_cell": all(bool(row["trust_energy_obeyed"]) for row in chart),
            "minimum_combined_singular_value_at_least_0p95": min(float(row["minimum_singular_value"]) for row in chart) >= 0.95,
            "heldout_residual_at_most_0p95_control": residual_ratio <= 0.95,
            "heldout_task_descent_gate": task > 0.0 and task >= 1.15 * control_task and task >= 0.004353292856055148,
            "all_4_holdout_ce_wins": holdout_wins == 4,
            "at_least_7_of_8_ce_wins": wins >= 7,
            "mean_ce_at_least_0p0005_better_and_prior_best": mean_candidate <= mean_by_arm[ARMS[0]] - 0.0005 and mean_candidate <= 7.180657014250755,
            "mean_ce_lower_than_both_full_components": mean_candidate < mean_by_arm[ARMS[1]] and mean_candidate < mean_by_arm[ARMS[2]],
        }
        gates[candidate] = gate
        summaries[candidate] = {
            "heldout_residual_ratio_to_control": residual_ratio,
            "heldout_task_descent": task,
            "heldout_task_ratio_to_control": task / max(control_task, 1e-30),
            "wins_vs_control": wins, "holdout_wins_vs_control": holdout_wins,
            "mean_ce": mean_candidate,
            "mean_control_ce": mean_by_arm[ARMS[0]],
            "mean_task32_ce": mean_by_arm[ARMS[1]],
            "mean_minimax16_ce": mean_by_arm[ARMS[2]],
            "minimum_trust_scale": min(float(row["trust_scale"]) for row in chart),
            "minimum_singular_value": min(float(row["minimum_singular_value"]) for row in chart),
            "passed": all(gate.values()),
        }
    selected = next((candidate for candidate in CANDIDATES if summaries[candidate]["passed"]), None)
    return {
        "by_arm": by_arm,
        "mean_finite_step_ce_by_arm": mean_by_arm,
        "comparisons": comparisons,
        "candidate_summaries": summaries,
        "gate": gates,
        "selected": selected,
        "passed": selected is not None,
        "decision": "HYBRID_OUTPUT_PASS" if selected else "REJECT_HYBRID_OUTPUT",
        "authorization": {"production_preregistration_authorized": selected is not None, "language_model_training_authorized": False},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--acquisition-result", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    acquisition = json.loads(args.acquisition_result.read_text())
    identity = plan["identity"]
    for key in ("task_gradient_result", "minimax_result"):
        if file_sha256(Path(identity[key])) != identity[f"{key}_sha256"]:
            raise ValueError(f"hybrid plan {key} identity mismatch")
    if (
        file_sha256(args.acquisition_result) != identity["acquisition_result_sha256"]
        or acquisition["identity"]["run_identity_sha256"] != identity["run_identity_sha256"]
    ):
        raise ValueError("hybrid plan acquisition identity mismatch")
    manifest_path = args.data_dir / "manifest.json"
    if not manifest_path.is_file() or file_sha256(manifest_path) != identity["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest SHA-256 mismatch")

    analysis = plan["analysis"]
    layers = [int(value) for value in analysis["layers"]]
    phases = [[int(value) for value in pair] for pair in analysis["phases"]]
    hidden_chart = analysis["shared_hidden_chart"]
    budget = analysis["output_budget"]
    run_identity = identity["run_identity_sha256"]
    snapshot_paths = {step: args.snapshot_dir / f"step_{step:06d}.pt" for step in sorted({value for pair in phases for value in pair})}
    probe_paths = {start: args.probe_dir / f"step_{start:06d}.pt" for start, _end in phases}
    for step, path in snapshot_paths.items():
        if file_sha256(path) != acquisition["snapshots"]["sha256_by_step"][str(step)]:
            raise ValueError(f"snapshot SHA-256 mismatch at step {step}")
    for step, path in probe_paths.items():
        if file_sha256(path) != acquisition["optimizer_probes"]["sha256_by_step"][str(step)]:
            raise ValueError(f"probe SHA-256 mismatch at step {step}")
    windows = {
        name: fixed_validation_batches(args.data_dir, int(spec["batch_size"]), int(spec["block_size"]) + 1, int(spec["batches"]), int(spec["seed"]))
        for name, spec in (("fit", analysis["fit_window"]), ("holdout", analysis["holdout_window"]))
    }

    metric_rows: list[dict[str, Any]] = []
    finite_rows: list[dict[str, Any]] = []
    chart_rows: list[dict[str, Any]] = []
    phase_summaries: list[dict[str, Any]] = []
    for phase_index, (phase_start, phase_end) in enumerate(phases):
        phase_started = time.perf_counter()
        snapshot = load_snapshot(snapshot_paths[phase_start])
        if snapshot.get("run_identity_sha256") != run_identity:
            raise ValueError("snapshot run identity mismatch")
        model = model_from_snapshot(snapshot, args.device)
        baseline_losses: dict[str, float] = {}
        hidden: dict[str, dict[int, torch.Tensor]] = {}
        gradients: dict[str, dict[int, torch.Tensor]] = {}
        for window in WINDOWS:
            baseline_losses[window], hidden[window] = evaluate_and_collect(model, windows[window], layers, args.device)
            gradients[window], _ = collect_cproj_gradients(model, windows[window], layers, args.device)
        probe = load_probe(probe_paths[phase_start], phase_start, run_identity)
        updates: dict[str, dict[int, torch.Tensor]] = {arm: {} for arm in ARMS}
        for layer in layers:
            name = parameter_name(layer)
            state = probe["parameters"][name]
            hyper = probe["hyperparameters"][name]
            weight = state["weight_before_step"].to(args.device, dtype=torch.float32)
            torch.testing.assert_close(weight.cpu(), snapshot["parameters"][name].float(), rtol=0.0, atol=0.0)
            learning_rate = float(hyper["lr"])
            weight_decay = float(hyper["weight_decay"])
            applied_per_lr = state["applied_direction_per_lr"].to(args.device, dtype=torch.float32)
            requested = learning_rate * applied_per_lr
            matching_direction = applied_per_lr + weight_decay * weight
            seed = int(hidden_chart["matching_seed"]) + layer * 100000 + phase_index * 10
            hidden_weight, residual, hidden_diagnostics = shared_hidden_chart(
                weight, requested, matching_direction,
                parent_stages=int(hidden_chart["parent_stages"]),
                residual_stages=int(hidden_chart["residual_stages"]),
                neighbors=int(hidden_chart["neighbors"]), seed=seed,
            )
            hidden_coordinates = sum(int(item["coordinates"]) for item in hidden_diagnostics)
            source = hidden_weight.T.contiguous()
            residual_t = residual.T.contiguous()
            control_t, control_diag = fit_frobenius_pass(
                source, residual_t, stages=int(budget["control_output_stages"]),
                neighbors=int(hidden_chart["neighbors"]), seed=seed + 2,
            )
            trust_energy = float((control_t - source).double().square().sum())
            train_gradient = state["gradient_after_clip"].to(args.device)
            fit_gradient = gradients["fit"][layer].to(args.device)
            task32_t, _task32_transform, task32_diag = task_givens_component(
                source, residual_t, fit_gradient.T,
                stages=int(budget["full_task_givens_stages"]),
                neighbors=int(hidden_chart["neighbors"]), seed=seed + 2,
            )
            minimax_score, minimax_score_diag = minimax_support_score(
                source, residual_t, hidden["fit"][layer], train_gradient, fit_gradient
            )
            minimax16_t, _support, minimax16_diag = fit_global_directed_map(
                source, residual_t, hidden["fit"][layer], minimax_score,
                incoming=int(budget["full_minimax_incoming_per_target"]),
                trust_output_energy=trust_energy,
            )
            hybrid_outputs: dict[str, torch.Tensor] = {}
            for order in CANDIDATES:
                hybrid_outputs[order], diagnostics = fit_hybrid(
                    source, residual_t, hidden["fit"][layer], train_gradient,
                    fit_gradient, order=order,
                    task_stages=int(budget["hybrid_task_givens_stages"]),
                    incoming=int(budget["hybrid_minimax_incoming_per_target"]),
                    neighbors=int(hidden_chart["neighbors"]), seed=seed + 2,
                    trust_output_energy=trust_energy,
                )
                chart_rows.append({"phase_start": phase_start, "layer": layer, "arm": order, **diagnostics})
            chart_rows.append({
                "phase_start": phase_start, "layer": layer,
                "arm": ARMS[2], "coordinates": int(minimax16_diag["coordinates"]),
                "trust_scale": float(minimax16_diag["trust_scale"]),
                "trust_energy_obeyed": bool(minimax16_diag["trust_energy_obeyed"]),
                "minimum_singular_value": float(minimax16_diag["minimum_singular_value_i_plus_b"]),
                "positive_task_agreement_fraction": float(minimax_score_diag["positive_task_agreement_fraction"]),
            })
            output_coordinates = {
                ARMS[0]: int(control_diag["coordinates"]),
                ARMS[1]: int(task32_diag["coordinates"]),
                ARMS[2]: int(minimax16_diag["coordinates"]),
                **{arm: int(budget["task_givens_coordinates"]) + int(budget["directed_coordinates"]) for arm in CANDIDATES},
            }
            coordinate_counts = {arm: hidden_coordinates + output_coordinates[arm] for arm in ARMS}
            if set(coordinate_counts.values()) != {int(budget["total_coordinates_per_layer"])}:
                raise ValueError("hybrid coordinate budget mismatch")
            decay = 1.0 - learning_rate * weight_decay
            final_outputs = {ARMS[0]: control_t, ARMS[1]: task32_t, ARMS[2]: minimax16_t, **hybrid_outputs}
            for arm, output_weight_t in final_outputs.items():
                final_weight = output_weight_t.T.contiguous() * decay
                update = (final_weight - weight).detach().cpu()
                updates[arm][layer] = update
                error = requested.cpu() - update
                for window in WINDOWS:
                    output_metrics = output_space_metrics(hidden[window][layer], requested.cpu(), update)
                    task_metrics = task_descent_metrics(gradients[window][layer], update)
                    metric_rows.append({
                        "phase_start": phase_start, "phase_end": phase_end,
                        "layer": layer, "window": window, "arm": arm,
                        "coordinates_per_layer": coordinate_counts[arm],
                        "validation_gradient_predicted_ce_decrease": task_metrics["predicted_ce_decrease"],
                        "activation_output_residual_energy": output_residual_energy(hidden[window][layer], requested.cpu(), update),
                        "output_fixed_scale_recovery": output_metrics["fixed_scale_recovery"],
                        "output_positive_step_line_recovery": output_metrics["positive_step_line_recovery"],
                        "update_energy": float(update.double().square().sum()),
                        "weight_error_energy": float(error.double().square().sum()),
                        "weight_fixed_scale_recovery": fixed_scale_recovery(requested.cpu(), update),
                    })
        for window in WINDOWS:
            finite_rows.append({"phase_start": phase_start, "phase_end": phase_end, "window": window, "arm": "baseline", "loss": baseline_losses[window]})
            for arm in ARMS:
                finite_rows.append({
                    "phase_start": phase_start, "phase_end": phase_end,
                    "window": window, "arm": arm,
                    "loss": evaluate_with_updates(model, windows[window], updates[arm], args.device),
                })
        phase_summaries.append({"phase_start": phase_start, "phase_end": phase_end, "baseline_loss": baseline_losses, "elapsed_seconds": time.perf_counter() - phase_started})
        print(json.dumps(phase_summaries[-1], sort_keys=True), flush=True)
        del model, snapshot, probe, hidden, gradients, updates
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    aggregate = aggregate_results(metric_rows, finite_rows, chart_rows)
    args.output.mkdir(parents=True, exist_ok=True)
    cells_path = args.output / "hybrid_output_cells.csv"
    finite_path = args.output / "hybrid_output_finite_ce.csv"
    chart_path = args.output / "hybrid_output_chart.csv"
    write_csv(cells_path, metric_rows)
    write_csv(finite_path, finite_rows)
    write_csv(chart_path, chart_rows)
    result = {
        "schema_version": SCHEMA_VERSION,
        "scientific_question": plan["scientific_question"],
        "source_commit": git_commit(REPO_ROOT),
        "source": {"path": str(Path(__file__).relative_to(REPO_ROOT)), "sha256": file_sha256(Path(__file__))},
        "execution": {"command": [sys.executable, *sys.argv], "started_at_utc": started_at, "host": "PRO6", "parameter_updates": 0, "watchdog": False, "callback": False},
        "plan": {"path": str(args.plan), "sha256": file_sha256(args.plan)},
        "acquisition_result": {"path": str(args.acquisition_result), "sha256": file_sha256(args.acquisition_result)},
        "identity": acquisition["identity"], "protocol": analysis,
        "phase_summaries": phase_summaries,
        "artifacts": {
            "cells": {"path": str(cells_path), "sha256": file_sha256(cells_path)},
            "finite_ce": {"path": str(finite_path), "sha256": file_sha256(finite_path)},
            "chart": {"path": str(chart_path), "sha256": file_sha256(chart_path)},
        },
        "aggregate": aggregate,
        "elapsed_seconds": time.time() - started,
    }
    result_path = args.output / "hybrid_output_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
