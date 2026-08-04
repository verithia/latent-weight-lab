#!/usr/bin/env python3
"""Test a fixed-FHT-conjugated block-Cayley c_proj output chart.

This is a zero-update diagnostic.  It keeps the accepted hidden64+24 chart,
replaces the data-dependent output-pair graph with one fixed-basis block chart,
and fits every skew coordinate in each 32-wide block jointly.
"""

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
from examples.nanogpt.model import LearnedFHTBlockOrthogonalOutputMix


SCHEMA_VERSION = "mai_124m_mlp_cproj_fht_block_cayley_output_v1"
EXPECTED_PLAN_SCHEMA = "mai_124m_mlp_cproj_fht_block_cayley_output_plan_v1"
ARMS = ("frobenius_output32", "fht_block32_cayley1")
WINDOWS = ("fit", "holdout")


def validate_plan(plan: dict[str, Any]) -> None:
    analysis = plan.get("analysis", {})
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
        "shared_hidden_chart": {
            "parent_stages": 64,
            "residual_stages": 24,
            "neighbors": 64,
            "matching_seed": 20260804,
            "coordinates_per_layer": 135168,
            "feedback": "zero for this one-step prospective diagnostic",
            "weight_decay_application": "identical production ordering in both arms",
        },
        "control": {
            "name": "frobenius_output32",
            "output_stages": 32,
            "output_coordinates_per_layer": 12288,
            "total_coordinates_per_layer": 147456,
            "definition": "Existing top-neighbor Frobenius pair selector with simultaneous Frobenius residual-fit angles.",
        },
        "candidate": {
            "name": "fht_block32_cayley1",
            "basis": "one exact fixed signed/permuted normalized block-FHT basis and its inverse",
            "basis_block_size": 256,
            "basis_seed": 20260804,
            "rotation_block_size": 32,
            "stages": 1,
            "rotation_blocks": 24,
            "coordinates_per_block": 496,
            "output_coordinates_per_layer": 11904,
            "total_coordinates_per_layer": 147072,
            "fit": {
                "source": "X = basis(W_after_hidden^T)",
                "residual": "R = basis(remaining requested update^T)",
                "per_block_equation": "C B + B C = D - D^T, where C=X^T X and D=X^T R",
                "solver": "symmetric eigendecomposition of C in float64; denominator lambda_i+lambda_j+1e-6*mean(diag(C))",
                "cayley_coordinates": "A=B/2 because Cayley(A)=I+2A+O(A^2)",
                "trust_radius": "multiply all A blocks in a layer-phase cell by min(1, max_abs(control_output32_angle)/max_abs(A))",
                "application": "apply the exact block Cayley transform in the fixed basis, invert the basis, transpose, and use the same decoupled-weight-decay ordering as control",
            },
        },
        "parameter_updates": 0,
    }
    observed = {
        "schema_version": plan.get("schema_version"),
        "layers": analysis.get("layers"),
        "phases": analysis.get("phases"),
        "fit_window": analysis.get("fit_window"),
        "holdout_window": analysis.get("holdout_window"),
        "shared_hidden_chart": analysis.get("shared_hidden_chart"),
        "control": analysis.get("control"),
        "candidate": analysis.get("candidate"),
        "parameter_updates": analysis.get("parameter_updates"),
    }
    if observed != expected:
        raise ValueError("block-Cayley plan does not match the immutable v1 contract")
    if plan.get("authorization", {}).get("implement_and_run_zero_update_analysis") is not True:
        raise ValueError("zero-update block-Cayley analysis is not authorized")


def solve_block_cayley_coordinates(
    source: torch.Tensor,
    residual: torch.Tensor,
    *,
    relative_ridge: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Solve the tangent-optimal Cayley coordinates for one feature block."""
    if source.ndim != 2 or residual.shape != source.shape:
        raise ValueError("source and residual must be equal two-dimensional blocks")
    x = source.double()
    r = residual.double()
    gram = x.T @ x
    cross = x.T @ r
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    ridge = float(relative_ridge) * float(gram.diag().mean().clamp_min(1e-30))
    rhs = cross - cross.T
    rhs_eigen = eigenvectors.T @ rhs @ eigenvectors
    denominator = eigenvalues[:, None] + eigenvalues[None, :] + ridge
    tangent_eigen = rhs_eigen / denominator.clamp_min(1e-30)
    tangent = eigenvectors @ tangent_eigen @ eigenvectors.T
    tangent = 0.5 * (tangent - tangent.T)
    coordinates = 0.5 * tangent
    linear_prediction = 2.0 * x @ coordinates
    return coordinates, {
        "ridge": ridge,
        "minimum_eigenvalue": float(eigenvalues.min()),
        "maximum_eigenvalue": float(eigenvalues.max()),
        "linear_residual_energy": float((r - linear_prediction).square().sum()),
        "target_residual_energy": float(r.square().sum()),
    }


def fit_fht_block_cayley_pass(
    source: torch.Tensor,
    residual: torch.Tensor,
    *,
    rotation_block_size: int,
    basis_block_size: int,
    seed: int,
    trust_radius: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Fit one fixed-basis block-Cayley stage and apply it exactly."""
    if source.ndim != 2 or residual.shape != source.shape:
        raise ValueError("source and residual must share one matrix shape")
    if not torch.isfinite(source).all() or not torch.isfinite(residual).all():
        raise ValueError("source and residual must be finite")
    if trust_radius < 0.0:
        raise ValueError("trust radius must be nonnegative")
    mixer = LearnedFHTBlockOrthogonalOutputMix(
        features=int(source.shape[1]),
        stages=1,
        rotation_block_size=int(rotation_block_size),
        basis_block_size=int(basis_block_size),
        seed=int(seed),
    ).to(device=source.device, dtype=torch.float32)
    with torch.no_grad():
        basis_source = mixer._basis(source.float(), 0, inverse=False)
        basis_residual = mixer._basis(residual.float(), 0, inverse=False)
        source_blocks = basis_source.reshape(
            source.shape[0], mixer.rotation_blocks, mixer.rotation_block_size
        )
        residual_blocks = basis_residual.reshape_as(source_blocks)
        solved: list[torch.Tensor] = []
        block_records: list[dict[str, float]] = []
        for block in range(mixer.rotation_blocks):
            coordinates, record = solve_block_cayley_coordinates(
                source_blocks[:, block], residual_blocks[:, block]
            )
            solved.append(
                coordinates[
                    mixer.upper_rows.to(coordinates.device),
                    mixer.upper_columns.to(coordinates.device),
                ]
            )
            block_records.append(record)
        raw = torch.stack(solved).float()
        raw_max = float(raw.abs().max())
        scale = min(1.0, float(trust_radius) / max(raw_max, 1e-30))
        bounded = raw * scale
        mixer.coordinates.copy_(bounded.reshape(-1))
        updated = mixer(source.float())
        exact_delta = updated - source.float()
    diagnostics: dict[str, Any] = {
        "coordinates": int(mixer.coordinates.numel()),
        "rotation_blocks": int(mixer.rotation_blocks),
        "coordinates_per_block": int(mixer.coordinates_per_block),
        "raw_maximum_abs_coordinate": raw_max,
        "maximum_abs_coordinate": float(bounded.abs().max()),
        "trust_radius": float(trust_radius),
        "trust_scale": scale,
        "trust_radius_obeyed": float(bounded.abs().max())
        <= float(trust_radius)
        + max(1e-12, 1e-6 * float(trust_radius)),
        "exact_update_energy": float(exact_delta.double().square().sum()),
        "target_residual_energy": float(residual.double().square().sum()),
        "linear_residual_energy": sum(row["linear_residual_energy"] for row in block_records),
        "ridge_minimum": min(row["ridge"] for row in block_records),
        "ridge_maximum": max(row["ridge"] for row in block_records),
        "gram_minimum_eigenvalue": min(row["minimum_eigenvalue"] for row in block_records),
        "gram_maximum_eigenvalue": max(row["maximum_eigenvalue"] for row in block_records),
    }
    if not all_finite(diagnostics) or not torch.isfinite(updated).all():
        raise ValueError("block-Cayley solve produced a nonfinite result")
    return updated, diagnostics


def aggregate_results(
    rows: list[dict[str, Any]],
    finite_rows: list[dict[str, Any]],
    chart_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_arm: dict[str, Any] = {}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        by_arm[arm] = {
            "cells_times_windows": len(selected),
            "coordinates_per_layer": sorted(
                {int(row["coordinates_per_layer"]) for row in selected}
            ),
            "activation_output_residual_energy": {
                window: sum(
                    float(row["activation_output_residual_energy"])
                    for row in selected
                    if row["window"] == window
                )
                for window in WINDOWS
            },
            "task_gradient_predicted_ce_decrease": {
                window: sum(
                    float(row["validation_gradient_predicted_ce_decrease"])
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
    control = by_arm["frobenius_output32"]
    candidate = by_arm["fht_block32_cayley1"]
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
                raise ValueError("finite-step comparison inventory is incomplete")
            delta = indexed["fht_block32_cayley1"] - indexed["frobenius_output32"]
            comparisons.append(
                {
                    "phase_start": phase_start,
                    "window": window,
                    "frobenius_loss": indexed["frobenius_output32"],
                    "candidate_loss": indexed["fht_block32_cayley1"],
                    "candidate_minus_frobenius": delta,
                    "candidate_wins": delta < 0.0,
                }
            )
    wins = sum(bool(row["candidate_wins"]) for row in comparisons)
    holdout_wins = sum(
        bool(row["candidate_wins"])
        for row in comparisons
        if row["window"] == "holdout"
    )
    mean_control = sum(float(row["frobenius_loss"]) for row in comparisons) / len(comparisons)
    mean_candidate = sum(float(row["candidate_loss"]) for row in comparisons) / len(comparisons)
    holdout_task = float(candidate["task_gradient_predicted_ce_decrease"]["holdout"])
    trust_obeyed = all(bool(row["trust_radius_obeyed"]) for row in chart_rows)
    finite_summary = all_finite(
        {
            "rows": rows,
            "finite_rows": finite_rows,
            "chart_rows": chart_rows,
            "by_arm": by_arm,
            "ratios": [residual_ratio, update_ratio],
            "comparisons": comparisons,
        }
    )
    gate = {
        "all_outputs_and_metrics_finite": finite_summary,
        "coordinate_budget_exact_and_no_larger_than_control": (
            control["coordinates_per_layer"] == [147456]
            and candidate["coordinates_per_layer"] == [147072]
        ),
        "all_coordinates_obey_trust_radius": trust_obeyed,
        "holdout_activation_residual_at_most_0p95_control": residual_ratio <= 0.95,
        "candidate_update_energy_at_most_1p10_control": update_ratio <= 1.10,
        "holdout_task_descent_positive_above_control_and_prior_best": (
            holdout_task > 0.0
            and holdout_task
            > float(control["task_gradient_predicted_ce_decrease"]["holdout"])
            and holdout_task >= 0.004353292856055148
        ),
        "all_4_holdout_ce_wins": holdout_wins == 4,
        "at_least_7_of_8_ce_wins": wins >= 7,
        "mean_ce_at_least_0p0005_better_than_control": mean_candidate <= mean_control - 0.0005,
        "mean_ce_no_worse_than_prior_best": mean_candidate <= 7.180657014250755,
    }
    passed = all(gate.values())
    return {
        "by_arm": by_arm,
        "holdout_activation_residual_energy_ratio": residual_ratio,
        "candidate_to_control_update_energy_ratio": update_ratio,
        "finite_step": {
            "comparisons": comparisons,
            "wins": wins,
            "holdout_wins": holdout_wins,
            "mean_frobenius_loss": mean_control,
            "mean_candidate_loss": mean_candidate,
        },
        "chart": {
            "cells": len(chart_rows),
            "trust_radius_obeyed": trust_obeyed,
            "minimum_trust_scale": min(float(row["trust_scale"]) for row in chart_rows),
            "maximum_raw_coordinate": max(float(row["raw_maximum_abs_coordinate"]) for row in chart_rows),
            "maximum_bounded_coordinate": max(float(row["maximum_abs_coordinate"]) for row in chart_rows),
        },
        "gate": gate,
        "passed": passed,
        "decision": (
            "FHT_BLOCK_CAYLEY_OUTPUT_PASS"
            if passed
            else "REJECT_FHT_BLOCK_CAYLEY_OUTPUT"
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
        file_sha256(Path(plan["identity"]["cross_batch_consensus_result"]))
        != plan["identity"]["cross_batch_consensus_result_sha256"]
        or file_sha256(args.acquisition_result)
        != plan["identity"]["acquisition_result_sha256"]
        or acquisition["identity"]["run_identity_sha256"]
        != plan["identity"]["run_identity_sha256"]
    ):
        raise ValueError("block-Cayley plan input identity mismatch")
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
    hidden_chart = analysis["shared_hidden_chart"]
    control_spec = analysis["control"]
    candidate_spec = analysis["candidate"]
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
            seed = int(hidden_chart["matching_seed"]) + layer * 100000 + phase_index * 10
            hidden_weight, residual, hidden_diagnostics = shared_hidden_chart(
                weight,
                requested,
                matching_direction,
                parent_stages=int(hidden_chart["parent_stages"]),
                residual_stages=int(hidden_chart["residual_stages"]),
                neighbors=int(hidden_chart["neighbors"]),
                seed=seed,
            )
            hidden_coordinates = sum(int(item["coordinates"]) for item in hidden_diagnostics)
            if hidden_coordinates != int(hidden_chart["coordinates_per_layer"]):
                raise ValueError("shared hidden coordinate budget mismatch")
            source = hidden_weight.T.contiguous()
            residual_t = residual.T.contiguous()
            control_t, control_diagnostics = fit_frobenius_pass(
                source,
                residual_t,
                stages=int(control_spec["output_stages"]),
                neighbors=int(hidden_chart["neighbors"]),
                seed=seed + 2,
            )
            candidate_t, candidate_diagnostics = fit_fht_block_cayley_pass(
                source,
                residual_t,
                rotation_block_size=int(candidate_spec["rotation_block_size"]),
                basis_block_size=int(candidate_spec["basis_block_size"]),
                seed=int(candidate_spec["basis_seed"]),
                trust_radius=float(control_diagnostics["maximum_abs_angle"]),
            )
            chart_rows.append(
                {
                    "phase_start": phase_start,
                    "layer": layer,
                    "control_maximum_abs_angle": float(control_diagnostics["maximum_abs_angle"]),
                    **candidate_diagnostics,
                }
            )
            coordinate_counts = {
                "frobenius_output32": hidden_coordinates + int(control_diagnostics["coordinates"]),
                "fht_block32_cayley1": hidden_coordinates + int(candidate_diagnostics["coordinates"]),
            }
            expected_counts = {
                "frobenius_output32": int(control_spec["total_coordinates_per_layer"]),
                "fht_block32_cayley1": int(candidate_spec["total_coordinates_per_layer"]),
            }
            if coordinate_counts != expected_counts:
                raise ValueError("output chart coordinate budget mismatch")
            decay = 1.0 - learning_rate * weight_decay
            final_weights = {
                "frobenius_output32": control_t.T.contiguous() * decay,
                "fht_block32_cayley1": candidate_t.T.contiguous() * decay,
            }
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
                            "coordinates_per_layer": coordinate_counts[arm],
                            "validation_gradient_predicted_ce_decrease": task["predicted_ce_decrease"],
                            "activation_output_residual_energy": output_residual_energy(
                                hidden[window][layer], requested.cpu(), update
                            ),
                            "output_fixed_scale_recovery": output["fixed_scale_recovery"],
                            "output_positive_step_line_recovery": output["positive_step_line_recovery"],
                            "update_energy": float(update.double().square().sum()),
                            "weight_error_energy": float(error.double().square().sum()),
                            "weight_fixed_scale_recovery": fixed_scale_recovery(
                                requested.cpu(), update
                            ),
                        }
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

    aggregate = aggregate_results(metric_rows, finite_rows, chart_rows)
    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "fht_block_cayley_cells.csv"
    finite_path = args.output / "fht_block_cayley_finite_ce.csv"
    chart_path = args.output / "fht_block_cayley_chart.csv"
    write_csv(metrics_path, metric_rows)
    write_csv(finite_path, finite_rows)
    write_csv(chart_path, chart_rows)
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
            "chart": {"path": str(chart_path), "sha256": file_sha256(chart_path)},
        },
        "aggregate": aggregate,
        "elapsed_seconds": time.time() - started,
    }
    result_path = args.output / "fht_block_cayley_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
