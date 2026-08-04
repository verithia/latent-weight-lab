#!/usr/bin/env python3
"""Evaluate sequentially refitted output32 charts under a frozen trust radius."""

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
from examples.nanogpt.analyze_mlp_task_gradient_direction import (
    collect_cproj_gradients,
)
from examples.nanogpt.analyze_parameter_trajectory import write_csv
from examples.nanogpt.analyze_residual_compatibility import (
    fixed_validation_batches,
)
from examples.nanogpt.muon_matched_givens import (
    apply_givens_flow,
    diagonal_metric_angles,
)


SCHEMA_VERSION = "mai_124m_mlp_cproj_task_gradient_sequential_trust_v1"
EXPECTED_PLAN_SCHEMA = "mai_124m_mlp_cproj_task_gradient_sequential_trust_plan_v1"
ARMS = (
    "frobenius_simultaneous",
    "frobenius_sequential_trust",
    "task_gradient_sequential_trust",
)
WINDOWS = ("fit", "holdout")


def validate_plan(plan: dict[str, Any]) -> None:
    analysis = plan.get("analysis", {})
    chart = analysis.get("shared_chart", {})
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
            "weight_decay_application": "identical production ordering in all arms",
        },
        "trust": {
            "definition": "For each layer-phase cell, max_abs(diagonal_metric_angles(S,R,frobenius_control_pairs)) from the existing simultaneous Frobenius output32 control.",
            "application": "Clamp every sequentially refitted scalar angle in both sequential arms to [-radius,+radius].",
            "minimum": 0.0,
            "tunable": False,
            "shared_between_sequential_arms": True,
        },
        "sequential": {
            "connectivity_fixed_before_refit": True,
            "stages": 32,
            "stage_angle": "diagonal_metric_angles(current_source,current_remaining_residual,current_single_matching)",
            "after_stage": "remaining_residual -= next_source-current_source; current_source = next_source",
            "task_gradient_used_for_angles": False,
            "holdout_used_for_selection_or_angles": False,
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
        "trust": analysis.get("trust_radius"),
        "sequential": analysis.get("sequential_refit"),
        "parameter_updates": analysis.get("parameter_updates"),
    }
    if observed != expected:
        raise ValueError("sequential-trust plan does not match the immutable v1 contract")
    if plan.get("authorization", {}).get("implement_and_run_zero_update_analysis") is not True:
        raise ValueError("zero-update sequential-trust analysis is not authorized")


def sequential_refit_pass(
    source: torch.Tensor,
    residual: torch.Tensor,
    permutations: torch.Tensor,
    *,
    trust_radius: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Refit one stage at a time and clamp every scalar by one frozen radius."""
    if trust_radius < 0.0 or not math.isfinite(trust_radius):
        raise ValueError("trust radius must be finite and non-negative")
    if permutations.ndim != 2 or permutations.shape[1] != source.shape[1]:
        raise ValueError("permutations and source shapes disagree")
    current = source.float()
    remaining = residual.float().clone()
    fitted: list[torch.Tensor] = []
    maximum_unclamped = 0.0
    clipped_coordinates = 0
    for stage in range(int(permutations.shape[0])):
        matching = permutations[stage : stage + 1].to(current.device)
        raw = diagonal_metric_angles(current, remaining, matching)
        maximum_unclamped = max(maximum_unclamped, float(raw.abs().max()))
        clipped_coordinates += int((raw.abs() > trust_radius).sum())
        angle = raw.clamp(min=-trust_radius, max=trust_radius)
        next_source = apply_givens_flow(
            current,
            angle,
            matching,
            torch.argsort(matching, dim=1),
        )
        remaining -= next_source - current
        current = next_source
        fitted.append(angle.detach().cpu())
    angles = torch.cat(fitted, dim=0)
    return current, {
        "stages": int(permutations.shape[0]),
        "coordinates": int(permutations.shape[0] * source.shape[1] // 2),
        "trust_radius": trust_radius,
        "maximum_abs_angle": float(angles.abs().max()),
        "mean_abs_angle": float(angles.abs().mean()),
        "maximum_unclamped_abs_angle": maximum_unclamped,
        "clipped_coordinates": clipped_coordinates,
        "trust_obeyed": bool(
            float(angles.abs().max())
            <= trust_radius + max(1e-7, trust_radius * 1e-6)
        ),
        "angles": angles,
        "permutations": permutations.detach().cpu(),
    }


def aggregate_results(
    rows: list[dict[str, Any]],
    finite_rows: list[dict[str, Any]],
    trust_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_arm: dict[str, Any] = {}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        by_arm[arm] = {
            "cells_times_windows": len(selected),
            "coordinates_per_layer": sorted(
                {int(row["coordinates_per_layer"]) for row in selected}
            ),
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
        }
    control = by_arm["frobenius_sequential_trust"]
    candidate = by_arm["task_gradient_sequential_trust"]
    simultaneous = by_arm["frobenius_simultaneous"]
    task = {
        window: {
            "control": float(control["task_gradient_predicted_ce_decrease"][window]),
            "candidate": float(candidate["task_gradient_predicted_ce_decrease"][window]),
        }
        for window in WINDOWS
    }
    advantages = {
        window: task[window]["candidate"] - task[window]["control"]
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
                str(row["arm"]): float(row["loss"])
                for row in finite_rows
                if int(row["phase_start"]) == phase_start
                and row["window"] == window
                and row["arm"] in ARMS
            }
            if set(indexed) != set(ARMS):
                raise ValueError("finite-step arm inventory is incomplete")
            delta = (
                indexed["task_gradient_sequential_trust"]
                - indexed["frobenius_sequential_trust"]
            )
            comparisons.append(
                {
                    "phase_start": phase_start,
                    "window": window,
                    "frobenius_simultaneous_loss": indexed[
                        "frobenius_simultaneous"
                    ],
                    "frobenius_sequential_loss": indexed[
                        "frobenius_sequential_trust"
                    ],
                    "candidate_loss": indexed["task_gradient_sequential_trust"],
                    "candidate_minus_sequential_control": delta,
                    "candidate_wins": delta < 0.0,
                }
            )
    wins = sum(bool(row["candidate_wins"]) for row in comparisons)
    holdout_wins = sum(
        bool(row["candidate_wins"])
        for row in comparisons
        if row["window"] == "holdout"
    )
    mean_sequential = sum(
        float(row["frobenius_sequential_loss"]) for row in comparisons
    ) / len(comparisons)
    mean_candidate = sum(float(row["candidate_loss"]) for row in comparisons) / len(
        comparisons
    )
    mean_simultaneous = sum(
        float(row["frobenius_simultaneous_loss"]) for row in comparisons
    ) / len(comparisons)
    trust_obeyed = all(
        bool(row["trust_obeyed"])
        for row in trust_rows
        if row["arm"] != "frobenius_simultaneous"
    )
    coordinates_exact = all(
        value["coordinates_per_layer"] == [147456] for value in by_arm.values()
    )
    finite_summary = all_finite(
        {
            "rows": rows,
            "finite_rows": finite_rows,
            "trust_rows": trust_rows,
            "by_arm": by_arm,
            "task": task,
            "advantages": advantages,
            "retention": retention,
            "ratios": [residual_ratio, update_ratio],
            "comparisons": comparisons,
        }
    )
    gate = {
        "all_outputs_and_metrics_finite": finite_summary,
        "coordinate_budget_exact": coordinates_exact,
        "all_sequential_angles_obey_shared_trust_radius": trust_obeyed,
        "holdout_task_descent_positive_and_above_control": (
            task["holdout"]["candidate"] > 0.0
            and advantages["holdout"] > 0.0
        ),
        "fit_task_advantage_positive": advantages["fit"] > 0.0,
        "holdout_to_fit_task_advantage_retention_at_least_0p10": (
            retention >= 0.10
        ),
        "holdout_activation_residual_at_most_1p10_control": residual_ratio <= 1.10,
        "candidate_update_energy_at_most_1p10_control": update_ratio <= 1.10,
        "all_4_holdout_finite_ce_wins": holdout_wins == 4,
        "finite_step_ce_wins_at_least_7_of_8": wins >= 7,
        "mean_ce_at_least_0p0005_better_than_sequential_control": (
            mean_candidate <= mean_sequential - 0.0005
        ),
        "mean_ce_no_worse_than_simultaneous_control": (
            mean_candidate <= mean_simultaneous
        ),
    }
    passed = all(gate.values())
    return {
        "by_arm": by_arm,
        "task_gradient_predicted_ce_decrease": task,
        "task_descent_advantage": advantages,
        "holdout_to_fit_task_advantage_retention": retention,
        "holdout_activation_residual_energy_ratio": residual_ratio,
        "candidate_to_control_update_energy_ratio": update_ratio,
        "finite_step": {
            "comparisons": comparisons,
            "candidate_wins": wins,
            "holdout_wins": holdout_wins,
            "mean_frobenius_simultaneous_loss": mean_simultaneous,
            "mean_frobenius_sequential_loss": mean_sequential,
            "mean_candidate_loss": mean_candidate,
        },
        "gate": gate,
        "passed": passed,
        "decision": (
            "TASK_GRADIENT_SEQUENTIAL_TRUST_PASS"
            if passed
            else "REJECT_TASK_GRADIENT_SEQUENTIAL_TRUST"
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
        file_sha256(Path(plan["identity"]["task_gradient_selector_result"]))
        != plan["identity"]["task_gradient_selector_result_sha256"]
        or file_sha256(args.acquisition_result)
        != plan["identity"]["acquisition_result_sha256"]
        or acquisition["identity"]["run_identity_sha256"]
        != plan["identity"]["run_identity_sha256"]
    ):
        raise ValueError("sequential-trust plan input identity mismatch")
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
    trust_rows: list[dict[str, Any]] = []
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
            gradients[window], _ = collect_cproj_gradients(
                model, windows[window], layers, args.device
            )

        probe = load_probe(probe_paths[phase_start], phase_start, run_identity)
        updates: dict[str, dict[int, torch.Tensor]] = {arm: {} for arm in ARMS}
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
            simultaneous, fro_diagnostics = fit_frobenius_pass(
                source,
                residual_t,
                stages=int(chart["output_stages"]),
                neighbors=int(chart["neighbors"]),
                seed=seed + 2,
            )
            _unused_task_simultaneous, task_diagnostics = (
                fit_task_gradient_hybrid_pass(
                    source,
                    residual_t,
                    gradients["fit"][layer].T.to(args.device),
                    stages=int(chart["output_stages"]),
                    neighbors=int(chart["neighbors"]),
                    seed=seed + 2,
                )
            )
            trust_radius = float(fro_diagnostics["maximum_abs_angle"])
            fro_sequential, fro_seq_diagnostics = sequential_refit_pass(
                source,
                residual_t,
                fro_diagnostics["permutations"],
                trust_radius=trust_radius,
            )
            task_sequential, task_seq_diagnostics = sequential_refit_pass(
                source,
                residual_t,
                task_diagnostics["permutations"],
                trust_radius=trust_radius,
            )
            decay = 1.0 - learning_rate * weight_decay
            final_weights = {
                "frobenius_simultaneous": simultaneous.T.contiguous() * decay,
                "frobenius_sequential_trust": fro_sequential.T.contiguous() * decay,
                "task_gradient_sequential_trust": task_sequential.T.contiguous() * decay,
            }
            coordinate_count = sum(
                int(item["coordinates"]) for item in hidden_diagnostics
            ) + int(fro_diagnostics["coordinates"])
            if coordinate_count != int(chart["coordinate_count_per_layer"]):
                raise ValueError("chart coordinate budget mismatch")
            for arm, final_weight in final_weights.items():
                update = (final_weight - weight).detach().cpu()
                updates[arm][layer] = update
                error = requested.cpu() - update
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
                            "arm": arm,
                            "coordinates_per_layer": coordinate_count,
                            "validation_gradient_predicted_ce_decrease": task[
                                "predicted_ce_decrease"
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
                            "update_energy": float(update.double().square().sum()),
                            "weight_error_energy": float(error.double().square().sum()),
                            "weight_fixed_scale_recovery": fixed_scale_recovery(
                                requested.cpu(), update
                            ),
                        }
                    )
            trust_rows.extend(
                [
                    {
                        "phase_start": phase_start,
                        "layer": layer,
                        "arm": "frobenius_simultaneous",
                        "trust_radius": trust_radius,
                        "maximum_abs_angle": float(
                            fro_diagnostics["maximum_abs_angle"]
                        ),
                        "maximum_unclamped_abs_angle": float(
                            fro_diagnostics["maximum_abs_angle"]
                        ),
                        "clipped_coordinates": 0,
                        "trust_obeyed": True,
                    },
                    {
                        "phase_start": phase_start,
                        "layer": layer,
                        "arm": "frobenius_sequential_trust",
                        **{
                            key: fro_seq_diagnostics[key]
                            for key in (
                                "trust_radius",
                                "maximum_abs_angle",
                                "maximum_unclamped_abs_angle",
                                "clipped_coordinates",
                                "trust_obeyed",
                            )
                        },
                    },
                    {
                        "phase_start": phase_start,
                        "layer": layer,
                        "arm": "task_gradient_sequential_trust",
                        **{
                            key: task_seq_diagnostics[key]
                            for key in (
                                "trust_radius",
                                "maximum_abs_angle",
                                "maximum_unclamped_abs_angle",
                                "clipped_coordinates",
                                "trust_obeyed",
                            )
                        },
                    },
                ]
            )

        for window in WINDOWS:
            finite_rows.append(
                {
                    "phase_start": phase_start,
                    "phase_end": phase_end,
                    "window": window,
                    "arm": "baseline",
                    "loss": baseline_losses[window],
                }
            )
            for arm in ARMS:
                finite_rows.append(
                    {
                        "phase_start": phase_start,
                        "phase_end": phase_end,
                        "window": window,
                        "arm": arm,
                        "loss": evaluate_with_updates(
                            model, windows[window], updates[arm], args.device
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

    aggregate = aggregate_results(metric_rows, finite_rows, trust_rows)
    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "sequential_trust_cells.csv"
    finite_path = args.output / "sequential_trust_finite_ce.csv"
    trust_path = args.output / "sequential_trust_angles.csv"
    write_csv(metrics_path, metric_rows)
    write_csv(finite_path, finite_rows)
    write_csv(trust_path, trust_rows)
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
            "angles": {"path": str(trust_path), "sha256": file_sha256(trust_path)},
        },
        "aggregate": aggregate,
        "elapsed_seconds": time.time() - started,
    }
    result_path = args.output / "sequential_trust_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
