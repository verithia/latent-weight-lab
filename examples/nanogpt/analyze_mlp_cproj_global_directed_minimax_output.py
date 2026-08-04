#!/usr/bin/env python3
"""Evaluate a minimax task-agreement selector for global directed c_proj maps."""

from __future__ import annotations

import argparse
import datetime as dt
import json
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
    _normalized_consensus,
    directed_support_scores,
    fit_global_directed_map,
    support_overlap,
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


SCHEMA_VERSION = "mai_124m_mlp_cproj_global_directed_minimax_output_v1"
EXPECTED_PLAN_SCHEMA = "mai_124m_mlp_cproj_global_directed_minimax_output_plan_v1"
ARMS = (
    "frobenius_output32",
    "global_directed16_average_consensus",
    "global_directed16_minimax_consensus",
)
CANDIDATES = ARMS[1:]
WINDOWS = ("fit", "holdout")


def validate_plan(plan: dict[str, Any]) -> None:
    analysis = plan.get("analysis", {})
    directed = analysis.get("directed_map", {})
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
        "control": {"name": "frobenius_output32", "output_stages": 32, "output_coordinates_per_layer": 12288, "total_coordinates_per_layer": 147456},
        "directed_map": {
            "source_channels": 768, "target_channels": 768,
            "incoming_coordinates_per_target": 16,
            "output_coordinates_per_layer": 12288,
            "total_coordinates_per_layer": 147456,
            "diagonal_edges_allowed": True,
        },
        "parameter_updates": 0,
    }
    observed = {
        "schema_version": plan.get("schema_version"),
        "layers": analysis.get("layers"), "phases": analysis.get("phases"),
        "fit_window": analysis.get("fit_window"), "holdout_window": analysis.get("holdout_window"),
        "shared_hidden_chart": analysis.get("shared_hidden_chart"),
        "control": analysis.get("control"),
        "directed_map": {key: directed.get(key) for key in expected["directed_map"]},
        "parameter_updates": analysis.get("parameter_updates"),
    }
    if observed != expected:
        raise ValueError("minimax-directed plan does not match immutable v1 contract")
    if plan.get("authorization", {}).get("implement_and_run_zero_update_analysis") is not True:
        raise ValueError("zero-update minimax analysis is not authorized")


def minimax_support_score(
    source: torch.Tensor,
    residual: torch.Tensor,
    activation: torch.Tensor,
    train_gradient: torch.Tensor,
    fit_gradient: torch.Tensor,
    *,
    relative_ridge: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Score edges by activation fit plus worst-of-two normalized task descent."""
    h = activation.to(source.device, dtype=torch.float32)
    design = h @ source.float()
    target = h @ residual.float()
    energy = design.square().sum(dim=0)
    ridge = float(relative_ridge) * float(energy.mean().clamp_min(1e-30))
    cross = design.T @ target
    beta = cross / (energy[:, None] + ridge)
    residual_score = 2.0 * beta * cross - beta.square() * energy[:, None]
    train = train_gradient.to(source.device, dtype=torch.float32)
    fit = fit_gradient.to(source.device, dtype=torch.float32)
    train = train / train.norm().clamp_min(1e-30)
    fit = fit / fit.norm().clamp_min(1e-30)
    train_task = -beta * (train @ source.float()).T
    fit_task = -beta * (fit @ source.float()).T
    residual_rms = residual_score.square().mean().sqrt().clamp_min(1e-30)
    train_rms = train_task.square().mean().sqrt().clamp_min(1e-30)
    fit_rms = fit_task.square().mean().sqrt().clamp_min(1e-30)
    agreement = torch.minimum(train_task / train_rms, fit_task / fit_rms)
    score = residual_score / residual_rms + agreement
    both_positive = (train_task > 0.0) & (fit_task > 0.0)
    diagnostics = {
        "single_edge_ridge": ridge,
        "residual_score_rms": float(residual_rms),
        "train_task_score_rms": float(train_rms),
        "fit_task_score_rms": float(fit_rms),
        "positive_task_agreement_fraction": float(both_positive.float().mean()),
        "mean_agreement_score": float(agreement.mean()),
        "maximum_agreement_score": float(agreement.max()),
    }
    if not all_finite(diagnostics) or not torch.isfinite(score).all():
        raise ValueError("minimax support score is nonfinite")
    return score, diagnostics


def aggregate_results(rows: list[dict[str, Any]], finite_rows: list[dict[str, Any]], chart_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, Any] = {}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        by_arm[arm] = {
            "coordinates_per_layer": sorted({int(row["coordinates_per_layer"]) for row in selected}),
            "activation_output_residual_energy": {window: sum(float(row["activation_output_residual_energy"]) for row in selected if row["window"] == window) for window in WINDOWS},
            "task_gradient_predicted_ce_decrease": {window: sum(float(row["validation_gradient_predicted_ce_decrease"]) for row in selected if row["window"] == window) for window in WINDOWS},
            "update_energy": sum(float(row["update_energy"]) for row in selected if row["window"] == "fit"),
        }
    index = {(int(row["phase_start"]), str(row["window"]), str(row["arm"])): float(row["loss"]) for row in finite_rows if row["arm"] in ARMS}
    comparisons: list[dict[str, Any]] = []
    for phase in sorted({int(row["phase_start"]) for row in finite_rows}):
        for window in WINDOWS:
            control_loss = index[(phase, window, ARMS[0])]
            average_loss = index[(phase, window, ARMS[1])]
            minimax_loss = index[(phase, window, ARMS[2])]
            comparisons.append({
                "phase_start": phase, "window": window,
                "minimax_minus_control": minimax_loss - control_loss,
                "minimax_minus_average": minimax_loss - average_loss,
                "minimax_wins_control": minimax_loss < control_loss,
                "minimax_wins_average": minimax_loss < average_loss,
            })
    control = by_arm[ARMS[0]]
    candidate = by_arm[ARMS[2]]
    candidate_chart = [row for row in chart_rows if row["arm"] == ARMS[2]]
    residual_ratio = float(candidate["activation_output_residual_energy"]["holdout"]) / max(float(control["activation_output_residual_energy"]["holdout"]), 1e-30)
    task = float(candidate["task_gradient_predicted_ce_decrease"]["holdout"])
    control_task = float(control["task_gradient_predicted_ce_decrease"]["holdout"])
    candidate_losses = [index[(row["phase_start"], row["window"], ARMS[2])] for row in comparisons]
    control_losses = [index[(row["phase_start"], row["window"], ARMS[0])] for row in comparisons]
    average_losses = [index[(row["phase_start"], row["window"], ARMS[1])] for row in comparisons]
    mean_candidate = sum(candidate_losses) / len(candidate_losses)
    mean_control = sum(control_losses) / len(control_losses)
    mean_average = sum(average_losses) / len(average_losses)
    wins = sum(bool(row["minimax_wins_control"]) for row in comparisons)
    holdout_wins = sum(bool(row["minimax_wins_control"]) for row in comparisons if row["window"] == "holdout")
    average_holdout_wins = sum(bool(row["minimax_wins_average"]) for row in comparisons if row["window"] == "holdout")
    gate = {
        "all_outputs_scores_solves_and_metrics_finite": all_finite({"rows": rows, "finite": finite_rows, "chart": chart_rows}),
        "equal_coordinate_budget": all(by_arm[arm]["coordinates_per_layer"] == [147456] for arm in ARMS),
        "output_energy_trust_obeyed_every_cell": all(bool(row["trust_energy_obeyed"]) for row in candidate_chart),
        "minimum_singular_value_at_least_0p95": min(float(row["minimum_singular_value_i_plus_b"]) for row in candidate_chart) >= 0.95,
        "heldout_residual_at_most_0p95_control": residual_ratio <= 0.95,
        "heldout_task_descent_gate": task > 0.0 and task >= 1.15 * control_task and task >= 0.004353292856055148,
        "all_4_holdout_ce_wins": holdout_wins == 4,
        "at_least_7_of_8_ce_wins": wins >= 7,
        "beats_average_on_at_least_3_holdouts": average_holdout_wins >= 3,
        "mean_ce_lower_than_average": mean_candidate < mean_average,
        "mean_ce_at_least_0p0005_better_and_prior_best": mean_candidate <= mean_control - 0.0005 and mean_candidate <= 7.180657014250755,
    }
    passed = all(gate.values())
    return {
        "by_arm": by_arm, "comparisons": comparisons,
        "candidate_summary": {
            "heldout_residual_ratio_to_control": residual_ratio,
            "heldout_task_descent": task,
            "heldout_task_ratio_to_control": task / max(control_task, 1e-30),
            "wins_vs_control": wins, "holdout_wins_vs_control": holdout_wins,
            "holdout_wins_vs_average": average_holdout_wins,
            "mean_ce": mean_candidate, "mean_control_ce": mean_control,
            "mean_average_ce": mean_average,
        },
        "chart": {
            "mean_support_overlap_with_average": sum(float(row["support_overlap_with_other_arm"]) for row in candidate_chart) / len(candidate_chart),
            "mean_train_fit_gradient_cosine": sum(float(row["train_fit_gradient_cosine"]) for row in candidate_chart) / len(candidate_chart),
            "mean_positive_task_agreement_fraction": sum(float(row["positive_task_agreement_fraction"]) for row in candidate_chart) / len(candidate_chart),
            "minimum_trust_scale": min(float(row["trust_scale"]) for row in candidate_chart),
            "minimum_singular_value_i_plus_b": min(float(row["minimum_singular_value_i_plus_b"]) for row in candidate_chart),
        },
        "gate": gate, "passed": passed,
        "decision": "GLOBAL_DIRECTED16_MINIMAX_PASS" if passed else "REJECT_GLOBAL_DIRECTED16_MINIMAX",
        "authorization": {"production_preregistration_authorized": passed, "language_model_training_authorized": False},
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
    if (
        file_sha256(Path(plan["identity"]["global_directed_result"])) != plan["identity"]["global_directed_result_sha256"]
        or file_sha256(args.acquisition_result) != plan["identity"]["acquisition_result_sha256"]
        or acquisition["identity"]["run_identity_sha256"] != plan["identity"]["run_identity_sha256"]
    ):
        raise ValueError("minimax plan input identity mismatch")
    manifest_path = args.data_dir / "manifest.json"
    if not manifest_path.is_file() or file_sha256(manifest_path) != plan["identity"]["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest SHA-256 mismatch")
    analysis = plan["analysis"]
    layers = [int(value) for value in analysis["layers"]]
    phases = [[int(value) for value in pair] for pair in analysis["phases"]]
    hidden_chart = analysis["shared_hidden_chart"]
    control = analysis["control"]
    directed = analysis["directed_map"]
    run_identity = plan["identity"]["run_identity_sha256"]
    snapshot_paths = {step: args.snapshot_dir / f"step_{step:06d}.pt" for step in sorted({value for pair in phases for value in pair})}
    probe_paths = {start: args.probe_dir / f"step_{start:06d}.pt" for start, _end in phases}
    for step, path in snapshot_paths.items():
        if file_sha256(path) != acquisition["snapshots"]["sha256_by_step"][str(step)]:
            raise ValueError(f"snapshot SHA-256 mismatch at step {step}")
    for step, path in probe_paths.items():
        if file_sha256(path) != acquisition["optimizer_probes"]["sha256_by_step"][str(step)]:
            raise ValueError(f"probe SHA-256 mismatch at step {step}")
    windows = {name: fixed_validation_batches(args.data_dir, int(spec["batch_size"]), int(spec["block_size"]) + 1, int(spec["batches"]), int(spec["seed"])) for name, spec in (("fit", analysis["fit_window"]), ("holdout", analysis["holdout_window"]))}

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
                parent_stages=int(hidden_chart["parent_stages"]), residual_stages=int(hidden_chart["residual_stages"]),
                neighbors=int(hidden_chart["neighbors"]), seed=seed,
            )
            hidden_coordinates = sum(int(item["coordinates"]) for item in hidden_diagnostics)
            source = hidden_weight.T.contiguous()
            residual_t = residual.T.contiguous()
            control_t, control_diagnostics = fit_frobenius_pass(source, residual_t, stages=int(control["output_stages"]), neighbors=int(hidden_chart["neighbors"]), seed=seed + 2)
            control_output_energy = float((control_t - source).double().square().sum())
            train_gradient = state["gradient_after_clip"].to(args.device)
            fit_gradient = gradients["fit"][layer].to(args.device)
            average_gradient, gradient_cosine = _normalized_consensus(train_gradient, fit_gradient)
            _activation_score, average_score, average_score_diagnostics = directed_support_scores(source, residual_t, hidden["fit"][layer], average_gradient)
            minimax_score, minimax_score_diagnostics = minimax_support_score(source, residual_t, hidden["fit"][layer], train_gradient, fit_gradient)
            fitted: dict[str, tuple[torch.Tensor, torch.Tensor, dict[str, Any]]] = {}
            for arm, score in ((ARMS[1], average_score), (ARMS[2], minimax_score)):
                fitted[arm] = fit_global_directed_map(source, residual_t, hidden["fit"][layer], score, incoming=int(directed["incoming_coordinates_per_target"]), trust_output_energy=control_output_energy)
            overlap = support_overlap(fitted[ARMS[1]][1], fitted[ARMS[2]][1])
            chart_rows.append({"phase_start": phase_start, "layer": layer, "arm": ARMS[1], "train_fit_gradient_cosine": gradient_cosine, "support_overlap_with_other_arm": overlap, **average_score_diagnostics, **fitted[ARMS[1]][2]})
            chart_rows.append({"phase_start": phase_start, "layer": layer, "arm": ARMS[2], "train_fit_gradient_cosine": gradient_cosine, "support_overlap_with_other_arm": overlap, **minimax_score_diagnostics, **fitted[ARMS[2]][2]})
            coordinate_counts = {
                ARMS[0]: hidden_coordinates + int(control_diagnostics["coordinates"]),
                ARMS[1]: hidden_coordinates + int(fitted[ARMS[1]][2]["coordinates"]),
                ARMS[2]: hidden_coordinates + int(fitted[ARMS[2]][2]["coordinates"]),
            }
            if set(coordinate_counts.values()) != {int(control["total_coordinates_per_layer"])}:
                raise ValueError("minimax coordinate budget mismatch")
            decay = 1.0 - learning_rate * weight_decay
            final_weights = {ARMS[0]: control_t.T.contiguous() * decay, ARMS[1]: fitted[ARMS[1]][0].T.contiguous() * decay, ARMS[2]: fitted[ARMS[2]][0].T.contiguous() * decay}
            for arm, final_weight in final_weights.items():
                update = (final_weight - weight).detach().cpu()
                updates[arm][layer] = update
                error = requested.cpu() - update
                for window in WINDOWS:
                    output = output_space_metrics(hidden[window][layer], requested.cpu(), update)
                    task = task_descent_metrics(gradients[window][layer], update)
                    metric_rows.append({
                        "phase_start": phase_start, "phase_end": phase_end, "layer": layer, "window": window, "arm": arm,
                        "coordinates_per_layer": coordinate_counts[arm],
                        "validation_gradient_predicted_ce_decrease": task["predicted_ce_decrease"],
                        "activation_output_residual_energy": output_residual_energy(hidden[window][layer], requested.cpu(), update),
                        "output_fixed_scale_recovery": output["fixed_scale_recovery"],
                        "output_positive_step_line_recovery": output["positive_step_line_recovery"],
                        "update_energy": float(update.double().square().sum()),
                        "weight_error_energy": float(error.double().square().sum()),
                        "weight_fixed_scale_recovery": fixed_scale_recovery(requested.cpu(), update),
                    })
        for window in WINDOWS:
            finite_rows.append({"phase_start": phase_start, "phase_end": phase_end, "window": window, "arm": "baseline", "loss": baseline_losses[window]})
            for arm in ARMS:
                finite_rows.append({"phase_start": phase_start, "phase_end": phase_end, "window": window, "arm": arm, "loss": evaluate_with_updates(model, windows[window], updates[arm], args.device)})
        phase_summaries.append({"phase_start": phase_start, "phase_end": phase_end, "baseline_loss": baseline_losses, "elapsed_seconds": time.perf_counter() - phase_started})
        print(json.dumps(phase_summaries[-1], sort_keys=True), flush=True)
        del model, snapshot, probe, hidden, gradients, updates
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    aggregate = aggregate_results(metric_rows, finite_rows, chart_rows)
    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "global_directed_minimax_cells.csv"
    finite_path = args.output / "global_directed_minimax_finite_ce.csv"
    chart_path = args.output / "global_directed_minimax_chart.csv"
    write_csv(metrics_path, metric_rows)
    write_csv(finite_path, finite_rows)
    write_csv(chart_path, chart_rows)
    result = {
        "schema_version": SCHEMA_VERSION, "scientific_question": plan["scientific_question"],
        "source_commit": git_commit(REPO_ROOT), "source": {"path": str(Path(__file__).relative_to(REPO_ROOT)), "sha256": file_sha256(Path(__file__))},
        "execution": {"command": [sys.executable, *sys.argv], "started_at_utc": started_at, "host": "PRO6", "parameter_updates": 0, "watchdog": False, "callback": False},
        "plan": {"path": str(args.plan), "sha256": file_sha256(args.plan)},
        "acquisition_result": {"path": str(args.acquisition_result), "sha256": file_sha256(args.acquisition_result)},
        "identity": acquisition["identity"], "protocol": analysis, "phase_summaries": phase_summaries,
        "artifacts": {"cells": {"path": str(metrics_path), "sha256": file_sha256(metrics_path)}, "finite_ce": {"path": str(finite_path), "sha256": file_sha256(finite_path)}, "chart": {"path": str(chart_path), "sha256": file_sha256(chart_path)}},
        "aggregate": aggregate, "elapsed_seconds": time.time() - started,
    }
    result_path = args.output / "global_directed_minimax_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
