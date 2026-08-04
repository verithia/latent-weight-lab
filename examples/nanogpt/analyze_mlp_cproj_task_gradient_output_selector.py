#!/usr/bin/env python3
"""Test task-gradient-aware output pairing with bounded Frobenius angles.

The candidate uses current fit-window task gradients only to alter the
discrete output-pair score.  Its angles remain the ordinary Frobenius
residual-fit angles used by the control.  This is a zero-update diagnostic;
holdout data is scoring-only and no training authority is emitted directly.
"""

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

from examples.nanogpt.analyze_mlp_activation_update_alignment import (
    load_snapshot,
    model_from_snapshot,
)
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
from examples.nanogpt.analyze_mlp_muon_matched_functional_metric import (
    evaluate_and_collect,
    evaluate_with_updates,
    fixed_scale_recovery,
    output_space_metrics,
    task_descent_metrics,
)
from examples.nanogpt.analyze_mlp_task_gradient_direction import (
    collect_cproj_gradients,
)
from examples.nanogpt.analyze_parameter_trajectory import write_csv
from examples.nanogpt.analyze_residual_compatibility import (
    fixed_validation_batches,
)
from examples.nanogpt.fast_task_matching import color_sorted_edges
from examples.nanogpt.muon_matched_givens import (
    apply_givens_flow,
    diagonal_metric_angles,
)


SCHEMA_VERSION = "mai_124m_mlp_cproj_task_gradient_output_selector_v1"
EXPECTED_PLAN_SCHEMA = "mai_124m_mlp_cproj_task_gradient_output_selector_plan_v1"
CANDIDATES = ("frobenius_output32", "task_gradient_hybrid_output32")
WINDOWS = ("fit", "holdout")


def validate_plan(plan: dict[str, Any]) -> None:
    analysis = plan.get("analysis", {})
    chart = analysis.get("shared_chart", {})
    candidate = analysis.get("candidate", {})
    expected = {
        "schema_version": EXPECTED_PLAN_SCHEMA,
        "layers": [0, 3, 6, 9, 11],
        "phases": [[0, 60], [60, 120], [120, 180], [180, 238]],
        "fit_window": {
            "split": "validation",
            "seed": 20260804,
            "batch_size": 2,
            "block_size": 256,
            "batches": 4,
            "rows_per_layer": 2048,
        },
        "holdout_window": {
            "split": "validation",
            "seed": 20260805,
            "batch_size": 2,
            "block_size": 256,
            "batches": 4,
            "rows_per_layer": 2048,
        },
        "chart": {
            "hidden_parent_stages": 64,
            "hidden_residual_stages": 24,
            "output_stages": 32,
            "neighbors": 64,
            "matching_seed": 20260804,
            "coordinate_count_per_layer": 147456,
            "feedback": "zero for this one-step prospective diagnostic",
            "weight_decay_application": (
                "identical production ordering in both arms"
            ),
        },
        "candidate": {
            "name": "task_gradient_hybrid_output32",
            "source": "S = W_after_hidden^T",
            "residual": (
                "R = remaining requested update^T after shared hidden64+24"
            ),
            "fit_task_gradient": (
                "G = current fit-window validation gradient^T; holdout gradient is scoring-only"
            ),
            "per_edge_residual_inner": (
                "r_ij = dot(S_i,R_j)-dot(S_j,R_i)"
            ),
            "per_edge_coordinate_norm": (
                "q_ij = ||S_i||^2+||S_j||^2"
            ),
            "per_edge_angle": "a_ij = r_ij/max(q_ij,1e-30)",
            "per_edge_residual_score": (
                "u_ij = r_ij^2/max(q_ij,1e-30)"
            ),
            "per_edge_task_inner": (
                "g_ij = dot(S_i,G_j)-dot(S_j,G_i)"
            ),
            "per_edge_task_score": "v_ij = -a_ij*g_ij",
            "normalization": (
                "divide u and v independently by their RMS over finite strict-upper-triangle edges, each clamped at 1e-30"
            ),
            "combined_score": (
                "score_ij = normalized(u_ij)+normalized(v_ij)"
            ),
            "candidate_graph": (
                "top 64 combined-score neighbors per output vertex, then the existing deterministic compiled edge coloring"
            ),
            "angle_fit": (
                "After selecting 32 matchings, recompute ordinary Frobenius diagonal_metric_angles(S,R,pairs); do not fit angles in activation or task metric."
            ),
            "application": (
                "Apply selected pairs and Frobenius angles to unprojected S, transpose, then use the same production weight-decay ordering as control."
            ),
        },
        "parameter_updates": 0,
    }
    observed = {
        "schema_version": plan.get("schema_version"),
        "layers": analysis.get("layers"),
        "phases": analysis.get("phases"),
        "fit_window": analysis.get("fit_window"),
        "holdout_window": analysis.get("holdout_window"),
        "chart": chart,
        "candidate": candidate,
        "parameter_updates": analysis.get("parameter_updates"),
    }
    if observed != expected:
        raise ValueError("task-gradient plan does not match the immutable v1 contract")
    if plan.get("authorization", {}).get("implement_and_run_zero_update_analysis") is not True:
        raise ValueError("zero-update task-gradient analysis is not authorized")


def score_matched_permutations(
    scores: torch.Tensor,
    *,
    stages: int,
    neighbors: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply the production top-neighbor graph and compiled coloring to scores."""
    if (
        scores.ndim != 2
        or scores.shape[0] != scores.shape[1]
        or scores.shape[0] % 2
    ):
        raise ValueError("scores must be an even square matrix")
    width = int(scores.shape[0])
    if not (0 < stages <= neighbors < width):
        raise ValueError("require 0 < stages <= neighbors < width")
    if not torch.isfinite(scores[~torch.eye(width, dtype=torch.bool, device=scores.device)]).all():
        raise ValueError("off-diagonal pair scores must be finite")
    prepared = time.perf_counter()
    local = scores.float().clone()
    local.fill_diagonal_(-torch.inf)
    top_scores, top_indices = torch.topk(local, k=neighbors, dim=1)
    order = torch.argsort(top_scores.reshape(-1), descending=True)
    left = (
        torch.arange(width, device=scores.device)
        .repeat_interleave(neighbors)
        .index_select(0, order)
    )
    right = top_indices.reshape(-1).index_select(0, order)
    edges = torch.stack(
        (torch.minimum(left, right), torch.maximum(left, right)), dim=1
    ).to(device="cpu", dtype=torch.int32)
    prepared_seconds = time.perf_counter() - prepared
    permutations, diagnostics = color_sorted_edges(
        edges, width=width, stages=stages, seed=seed
    )
    diagnostics.update(
        {
            "prepared_seconds": prepared_seconds,
            "total_seconds": prepared_seconds + diagnostics["native_seconds"],
            "candidate_edges": int(edges.shape[0]),
        }
    )
    return permutations, diagnostics


def task_gradient_pair_scores(
    source: torch.Tensor,
    residual: torch.Tensor,
    gradient: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return the preregistered unit-RMS residual-plus-task edge score."""
    if source.ndim != 2 or residual.shape != source.shape or gradient.shape != source.shape:
        raise ValueError("source, residual, and gradient must share one matrix shape")
    source = source.float()
    residual = residual.float()
    gradient = gradient.float()
    residual_cross = source.T @ residual
    residual_inner = residual_cross - residual_cross.T
    column_energy = source.square().sum(dim=0)
    coordinate_norm = (
        column_energy[:, None] + column_energy[None, :]
    ).clamp_min(1e-30)
    angle = residual_inner / coordinate_norm
    residual_score = residual_inner.square() / coordinate_norm
    gradient_cross = source.T @ gradient
    task_inner = gradient_cross - gradient_cross.T
    task_score = -angle * task_inner
    mask = torch.triu(
        torch.ones_like(residual_score, dtype=torch.bool), diagonal=1
    )
    residual_rms = residual_score[mask].square().mean().sqrt().clamp_min(1e-30)
    task_rms = task_score[mask].square().mean().sqrt().clamp_min(1e-30)
    combined = residual_score / residual_rms + task_score / task_rms
    return combined, {
        "residual_score_rms": float(residual_rms),
        "task_score_rms": float(task_rms),
        "positive_task_edge_fraction": float((task_score[mask] > 0.0).float().mean()),
        "combined_score_min": float(combined[mask].min()),
        "combined_score_max": float(combined[mask].max()),
    }


def fit_task_gradient_hybrid_pass(
    source: torch.Tensor,
    residual: torch.Tensor,
    gradient: torch.Tensor,
    *,
    stages: int,
    neighbors: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    scores, score_diagnostics = task_gradient_pair_scores(
        source, residual, gradient
    )
    permutations, matching = score_matched_permutations(
        scores, stages=stages, neighbors=neighbors, seed=seed
    )
    permutations = permutations.to(source.device)
    angles = diagonal_metric_angles(source, residual, permutations)
    updated = apply_givens_flow(
        source,
        angles,
        permutations,
        torch.argsort(permutations, dim=1),
    )
    return updated, {
        **matching,
        **score_diagnostics,
        "stages": stages,
        "coordinates": int(stages * source.shape[1] // 2),
        "maximum_abs_angle": float(angles.abs().max()),
        "mean_abs_angle": float(angles.abs().mean()),
        "permutations": permutations.detach().cpu(),
        "angles": angles.detach().cpu(),
    }


def aggregate_results(
    rows: list[dict[str, Any]], finite_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    by_candidate: dict[str, Any] = {}
    for candidate in CANDIDATES:
        selected = [row for row in rows if row["candidate"] == candidate]
        by_candidate[candidate] = {
            "cells_times_windows": len(selected),
            "task_gradient_predicted_ce_decrease": {
                window: sum(
                    float(row["validation_gradient_predicted_ce_decrease"])
                    for row in selected
                    if row["window"] == window
                )
                for window in WINDOWS
            },
            "activation_output_residual_energy": {
                window: sum(
                    float(row["activation_output_residual_energy"])
                    for row in selected
                    if row["window"] == window
                )
                for window in WINDOWS
            },
            "update_energy": sum(
                float(row["update_energy"])
                for row in selected
                if row["window"] == "fit"
            ),
            "weight_error_energy": sum(
                float(row["weight_error_energy"])
                for row in selected
                if row["window"] == "fit"
            ),
            "coordinates_per_layer": sorted(
                {int(row["coordinates_per_layer"]) for row in selected}
            ),
        }
    control = by_candidate["frobenius_output32"]
    candidate = by_candidate["task_gradient_hybrid_output32"]
    task = {
        window: {
            "frobenius": float(control["task_gradient_predicted_ce_decrease"][window]),
            "candidate": float(candidate["task_gradient_predicted_ce_decrease"][window]),
        }
        for window in WINDOWS
    }
    advantages = {
        window: task[window]["candidate"] - task[window]["frobenius"]
        for window in WINDOWS
    }
    retention = (
        advantages["holdout"] / advantages["fit"]
        if abs(advantages["fit"]) > 1e-30
        else float("nan")
    )
    residual_ratio = float(
        candidate["activation_output_residual_energy"]["holdout"]
    ) / max(float(control["activation_output_residual_energy"]["holdout"]), 1e-30)
    update_ratio = float(candidate["update_energy"]) / max(
        float(control["update_energy"]), 1e-30
    )

    comparisons: list[dict[str, Any]] = []
    for phase_start in sorted({int(row["phase_start"]) for row in finite_rows}):
        for window in WINDOWS:
            indexed = {
                str(row["candidate"]): float(row["loss"])
                for row in finite_rows
                if int(row["phase_start"]) == phase_start
                and row["window"] == window
                and row["candidate"] in CANDIDATES
            }
            if set(indexed) != set(CANDIDATES):
                raise ValueError("finite-step comparison inventory is incomplete")
            delta = (
                indexed["task_gradient_hybrid_output32"]
                - indexed["frobenius_output32"]
            )
            comparisons.append(
                {
                    "phase_start": phase_start,
                    "window": window,
                    "frobenius_loss": indexed["frobenius_output32"],
                    "candidate_loss": indexed["task_gradient_hybrid_output32"],
                    "candidate_minus_frobenius": delta,
                    "candidate_wins": delta < 0.0,
                }
            )
    wins = sum(bool(row["candidate_wins"]) for row in comparisons)
    mean_control = sum(float(row["frobenius_loss"]) for row in comparisons) / len(
        comparisons
    )
    mean_candidate = sum(float(row["candidate_loss"]) for row in comparisons) / len(
        comparisons
    )
    finite_summary = all_finite(
        {
            "rows": rows,
            "finite_rows": finite_rows,
            "by_candidate": by_candidate,
            "task": task,
            "advantages": advantages,
            "retention": retention,
            "residual_ratio": residual_ratio,
            "update_ratio": update_ratio,
            "comparisons": comparisons,
        }
    )
    coordinates_exact = all(
        value["coordinates_per_layer"] == [147456]
        for value in by_candidate.values()
    )
    holdout_control = task["holdout"]["frobenius"]
    holdout_candidate = task["holdout"]["candidate"]
    gate = {
        "all_outputs_and_metrics_finite": finite_summary,
        "coordinate_budget_exact": coordinates_exact,
        "holdout_task_descent_at_least_1p25_control_and_positive": (
            holdout_candidate > 0.0
            and holdout_candidate >= 1.25 * holdout_control
        ),
        "fit_task_advantage_positive": advantages["fit"] > 0.0,
        "holdout_task_advantage_positive": advantages["holdout"] > 0.0,
        "holdout_to_fit_task_advantage_retention_at_least_0p50": (
            retention >= 0.50
        ),
        "holdout_activation_residual_at_most_1p25_control": residual_ratio <= 1.25,
        "candidate_update_energy_at_most_1p25_control": update_ratio <= 1.25,
        "finite_step_ce_wins_at_least_6_of_8": wins >= 6,
        "mean_finite_step_ce_not_worse": mean_candidate <= mean_control,
    }
    passed = all(gate.values())
    return {
        "by_candidate": by_candidate,
        "task_gradient_predicted_ce_decrease": task,
        "task_descent_advantage": advantages,
        "holdout_to_fit_task_advantage_retention": retention,
        "holdout_activation_residual_energy_ratio": residual_ratio,
        "candidate_to_control_update_energy_ratio": update_ratio,
        "finite_step": {
            "comparisons": comparisons,
            "candidate_wins": wins,
            "total_comparisons": len(comparisons),
            "mean_frobenius_loss": mean_control,
            "mean_candidate_loss": mean_candidate,
        },
        "gate": gate,
        "passed": passed,
        "decision": (
            "TASK_GRADIENT_OUTPUT_SELECTOR_PASS"
            if passed
            else "REJECT_TASK_GRADIENT_OUTPUT_SELECTOR"
        ),
        "authorization": {
            "production_preregistration_authorized": passed,
            "language_model_training_authorized": False,
        },
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
        file_sha256(Path(plan["identity"]["activation_selector_result"]))
        != plan["identity"]["activation_selector_result_sha256"]
        or file_sha256(args.acquisition_result)
        != plan["identity"]["acquisition_result_sha256"]
        or acquisition["identity"]["run_identity_sha256"]
        != plan["identity"]["run_identity_sha256"]
    ):
        raise ValueError("task-gradient plan input identity mismatch")
    manifest_path = args.data_dir / "manifest.json"
    if (
        not manifest_path.is_file()
        or file_sha256(manifest_path)
        != plan["identity"]["dataset_manifest_sha256"]
    ):
        raise ValueError("dataset manifest SHA-256 mismatch")

    analysis = plan["analysis"]
    layers = [int(value) for value in analysis["layers"]]
    phases = [[int(value) for value in pair] for pair in analysis["phases"]]
    chart = analysis["shared_chart"]
    run_identity = plan["identity"]["run_identity_sha256"]
    snapshot_paths = {
        step: args.snapshot_dir / f"step_{step:06d}.pt"
        for step in sorted({value for pair in phases for value in pair})
    }
    probe_paths = {
        start: args.probe_dir / f"step_{start:06d}.pt"
        for start, _end in phases
    }
    for step, path in snapshot_paths.items():
        if file_sha256(path) != acquisition["snapshots"]["sha256_by_step"][str(step)]:
            raise ValueError(f"snapshot SHA-256 mismatch at step {step}")
    for step, path in probe_paths.items():
        if file_sha256(path) != acquisition["optimizer_probes"]["sha256_by_step"][str(step)]:
            raise ValueError(f"probe SHA-256 mismatch at step {step}")

    windows = {
        name: fixed_validation_batches(
            args.data_dir,
            int(spec["batch_size"]),
            int(spec["block_size"]) + 1,
            int(spec["batches"]),
            int(spec["seed"]),
        )
        for name, spec in (
            ("fit", analysis["fit_window"]),
            ("holdout", analysis["holdout_window"]),
        )
    }
    metric_rows: list[dict[str, Any]] = []
    finite_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
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
            baseline_losses[window], hidden[window] = evaluate_and_collect(
                model, windows[window], layers, args.device
            )
            gradients[window], _gradient_loss = collect_cproj_gradients(
                model, windows[window], layers, args.device
            )

        probe = load_probe(probe_paths[phase_start], phase_start, run_identity)
        updates: dict[str, dict[int, torch.Tensor]] = {
            candidate: {} for candidate in CANDIDATES
        }
        for layer in layers:
            name = parameter_name(layer)
            state = probe["parameters"][name]
            hyper = probe["hyperparameters"][name]
            weight = state["weight_before_step"].to(args.device, dtype=torch.float32)
            torch.testing.assert_close(
                weight.cpu(), snapshot["parameters"][name].float(), rtol=0.0, atol=0.0
            )
            learning_rate = float(hyper["lr"])
            weight_decay = float(hyper["weight_decay"])
            applied_per_lr = state["applied_direction_per_lr"].to(
                args.device, dtype=torch.float32
            )
            requested = learning_rate * applied_per_lr
            matching_direction = applied_per_lr + weight_decay * weight
            seed = int(chart["matching_seed"]) + layer * 100000 + phase_index * 10
            hidden_weight, residual, hidden_diagnostics = shared_hidden_chart(
                weight,
                requested,
                matching_direction,
                parent_stages=int(chart["hidden_parent_stages"]),
                residual_stages=int(chart["hidden_residual_stages"]),
                neighbors=int(chart["neighbors"]),
                seed=seed,
            )
            source = hidden_weight.T.contiguous()
            residual_t = residual.T.contiguous()
            control_t, control_diagnostics = fit_frobenius_pass(
                source,
                residual_t,
                stages=int(chart["output_stages"]),
                neighbors=int(chart["neighbors"]),
                seed=seed + 2,
            )
            candidate_t, candidate_diagnostics = fit_task_gradient_hybrid_pass(
                source,
                residual_t,
                gradients["fit"][layer].T.to(args.device),
                stages=int(chart["output_stages"]),
                neighbors=int(chart["neighbors"]),
                seed=seed + 2,
            )
            decay = 1.0 - learning_rate * weight_decay
            final_weights = {
                "frobenius_output32": control_t.T.contiguous() * decay,
                "task_gradient_hybrid_output32": candidate_t.T.contiguous() * decay,
            }
            coordinate_count = sum(
                int(item["coordinates"]) for item in hidden_diagnostics
            ) + int(control_diagnostics["coordinates"])
            if coordinate_count != int(chart["coordinate_count_per_layer"]):
                raise ValueError("chart coordinate budget mismatch")
            for candidate_name, final_weight in final_weights.items():
                update = (final_weight - weight).detach().cpu()
                updates[candidate_name][layer] = update
                weight_error = requested.cpu() - update
                for window in WINDOWS:
                    output = output_space_metrics(
                        hidden[window][layer], requested.cpu(), update
                    )
                    task = task_descent_metrics(gradients[window][layer], update)
                    metric_rows.append(
                        {
                            "phase_start": phase_start,
                            "phase_end": phase_end,
                            "layer": layer,
                            "window": window,
                            "candidate": candidate_name,
                            "coordinates_per_layer": coordinate_count,
                            "validation_gradient_predicted_ce_decrease": task[
                                "predicted_ce_decrease"
                            ],
                            "validation_gradient_predicted_ce_decrease_per_fro": task[
                                "predicted_ce_decrease_per_fro"
                            ],
                            "activation_output_residual_energy": output_residual_energy(
                                hidden[window][layer], requested.cpu(), update
                            ),
                            "output_fixed_scale_recovery": output[
                                "fixed_scale_recovery"
                            ],
                            "output_positive_step_line_recovery": output[
                                "positive_step_line_recovery"
                            ],
                            "output_cosine": output["cosine"],
                            "target_output_energy": output[
                                "target_output_energy"
                            ],
                            "update_energy": float(update.double().square().sum()),
                            "weight_error_energy": float(
                                weight_error.double().square().sum()
                            ),
                            "weight_fixed_scale_recovery": fixed_scale_recovery(
                                requested.cpu(), update
                            ),
                        }
                    )
            timing_rows.extend(
                {
                    "phase_start": phase_start,
                    "layer": layer,
                    "pass": pass_name,
                    "coordinates": int(diagnostics["coordinates"]),
                    "matching_total_seconds": float(diagnostics["total_seconds"]),
                    "candidate_edge_fraction": float(
                        diagnostics["candidate_edge_fraction"]
                    ),
                    "maximum_abs_angle": float(diagnostics["maximum_abs_angle"]),
                    "mean_abs_angle": float(diagnostics["mean_abs_angle"]),
                    "residual_score_rms": diagnostics.get("residual_score_rms"),
                    "task_score_rms": diagnostics.get("task_score_rms"),
                    "positive_task_edge_fraction": diagnostics.get(
                        "positive_task_edge_fraction"
                    ),
                }
                for pass_name, diagnostics in (
                    ("hidden_parent64", hidden_diagnostics[0]),
                    ("hidden_residual24", hidden_diagnostics[1]),
                    ("frobenius_output32", control_diagnostics),
                    ("task_gradient_hybrid_output32", candidate_diagnostics),
                )
            )

        for window in WINDOWS:
            finite_rows.append(
                {
                    "phase_start": phase_start,
                    "phase_end": phase_end,
                    "window": window,
                    "candidate": "baseline",
                    "loss": baseline_losses[window],
                }
            )
            for candidate in CANDIDATES:
                finite_rows.append(
                    {
                        "phase_start": phase_start,
                        "phase_end": phase_end,
                        "window": window,
                        "candidate": candidate,
                        "loss": evaluate_with_updates(
                            model, windows[window], updates[candidate], args.device
                        ),
                    }
                )
        phase_summaries.append(
            {
                "phase_start": phase_start,
                "phase_end": phase_end,
                "baseline_loss": baseline_losses,
                "elapsed_seconds": time.perf_counter() - phase_started,
            }
        )
        print(json.dumps(phase_summaries[-1], sort_keys=True), flush=True)
        del model, snapshot, probe, hidden, gradients, updates
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    aggregate = aggregate_results(metric_rows, finite_rows)
    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "task_gradient_selector_cells.csv"
    finite_path = args.output / "task_gradient_selector_finite_ce.csv"
    timing_path = args.output / "task_gradient_selector_timing.csv"
    write_csv(metrics_path, metric_rows)
    write_csv(finite_path, finite_rows)
    write_csv(timing_path, timing_rows)
    result = {
        "schema_version": SCHEMA_VERSION,
        "scientific_question": plan["scientific_question"],
        "source_commit": git_commit(REPO_ROOT),
        "source": {
            "path": str(Path(__file__).relative_to(REPO_ROOT)),
            "sha256": file_sha256(Path(__file__)),
        },
        "execution": {
            "command": [sys.executable, *sys.argv],
            "started_at_utc": started_at,
            "host": "PRO6",
            "parameter_updates": 0,
            "watchdog": False,
            "callback": False,
        },
        "plan": {"path": str(args.plan), "sha256": file_sha256(args.plan)},
        "acquisition_result": {
            "path": str(args.acquisition_result),
            "sha256": file_sha256(args.acquisition_result),
        },
        "identity": acquisition["identity"],
        "protocol": analysis,
        "phase_summaries": phase_summaries,
        "artifacts": {
            "cells": {"path": str(metrics_path), "sha256": file_sha256(metrics_path)},
            "finite_ce": {"path": str(finite_path), "sha256": file_sha256(finite_path)},
            "timing": {"path": str(timing_path), "sha256": file_sha256(timing_path)},
        },
        "aggregate": aggregate,
        "elapsed_seconds": time.time() - started,
    }
    result_path = args.output / "task_gradient_selector_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
