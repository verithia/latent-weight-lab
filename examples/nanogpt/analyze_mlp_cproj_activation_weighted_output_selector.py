#!/usr/bin/env python3
"""Compare Frobenius and post-GELU-weighted c_proj output selectors.

This is the preregistered zero-update diagnostic for the matched
BlockFHT-attention/dense-MLP parent.  The two arms share the exact same
64+24 hidden-side chart and the same 32 output-side coordinate budget.  They
differ only in how output pairs and angles are selected:

* the control works in ordinary transposed-weight Frobenius geometry;
* the candidate works after multiplying source and residual by fit-window
  post-GELU activation rows, then applies the selected pairs and angles to the
  unprojected source.

No optimizer step is taken, no basis is learned, and holdout activations never
participate in selection.
"""

from __future__ import annotations

import argparse
import datetime as dt
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

from examples.nanogpt.analyze_mlp_activation_update_alignment import (
    load_snapshot,
    model_from_snapshot,
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
from examples.nanogpt.fast_task_matching import (
    fast_muon_matched_permutations,
)
from examples.nanogpt.muon_matched_givens import (
    apply_givens_flow,
    diagonal_metric_angles,
)
from examples.nanogpt.parameter_trajectory import (
    OPTIMIZER_PROBE_SCHEMA_VERSION,
)


SCHEMA_VERSION = "mai_124m_mlp_cproj_activation_weighted_output_selector_v1"
CANDIDATES = ("frobenius_output32", "activation_output32")
WINDOWS = ("fit", "holdout")
EXPECTED_PLAN_SCHEMA = (
    "mai_124m_mlp_cproj_activation_weighted_output_selector_plan_v1"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def parameter_name(layer: int) -> str:
    return f"transformer.h.{layer}.mlp.c_proj.weight"


def validate_plan(plan: dict[str, Any]) -> None:
    """Fail closed if the immutable selector contract changed."""
    analysis = plan.get("selector_analysis", {})
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
            "weight_decay_application": (
                "identical production ordering in both arms"
            ),
        },
        "parameter_updates": 0,
        "prohibited": [
            "learned basis",
            "inverse JtJ or conjugate-gradient pullback",
            "dense residual",
            "extra chart coordinates",
            "selection on holdout activations",
        ],
    }
    observed = {
        "schema_version": plan.get("schema_version"),
        "layers": analysis.get("layers"),
        "phases": analysis.get("phases"),
        "fit_window": analysis.get("fit_window"),
        "holdout_window": analysis.get("holdout_window"),
        "chart": chart,
        "parameter_updates": analysis.get("parameter_updates"),
        "prohibited": analysis.get("prohibited"),
    }
    if observed != expected:
        raise ValueError(
            "selector plan does not match the preregistered v1 contract"
        )
    if plan.get("authorization", {}).get("zero_update_selector_analysis") is not True:
        raise ValueError("zero-update selector analysis is not authorized")


def fit_frobenius_pass(
    source: torch.Tensor,
    residual: torch.Tensor,
    *,
    stages: int,
    neighbors: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Select and fit a right Givens pass in ordinary Frobenius geometry."""
    permutations, diagnostics = fast_muon_matched_permutations(
        source,
        residual,
        stages=stages,
        neighbors=neighbors,
        seed=seed,
    )
    permutations = permutations.to(device=source.device)
    angles = diagonal_metric_angles(source, residual, permutations)
    updated = apply_givens_flow(
        source,
        angles,
        permutations,
        torch.argsort(permutations, dim=1),
    )
    return updated, {
        **diagnostics,
        "stages": stages,
        "coordinates": int(stages * source.shape[1] // 2),
        "maximum_abs_angle": float(angles.abs().max()),
        "mean_abs_angle": float(angles.abs().mean()),
        "permutations": permutations.detach().cpu(),
        "angles": angles.detach().cpu(),
    }


def fit_activation_weighted_pass(
    source: torch.Tensor,
    residual: torch.Tensor,
    hidden: torch.Tensor,
    *,
    stages: int,
    neighbors: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Select in output space and apply the resulting chart to ``source``."""
    if hidden.ndim != 2 or hidden.shape[1] != source.shape[0]:
        raise ValueError("hidden/source shapes disagree")
    projected_source = hidden.to(source.device, dtype=torch.float32) @ source.float()
    projected_residual = hidden.to(source.device, dtype=torch.float32) @ residual.float()
    permutations, diagnostics = fast_muon_matched_permutations(
        projected_source,
        projected_residual,
        stages=stages,
        neighbors=neighbors,
        seed=seed,
    )
    permutations = permutations.to(device=source.device)
    angles = diagonal_metric_angles(
        projected_source, projected_residual, permutations
    )
    updated = apply_givens_flow(
        source,
        angles,
        permutations,
        torch.argsort(permutations, dim=1),
    )
    return updated, {
        **diagnostics,
        "stages": stages,
        "coordinates": int(stages * source.shape[1] // 2),
        "fit_rows": int(hidden.shape[0]),
        "maximum_abs_angle": float(angles.abs().max()),
        "mean_abs_angle": float(angles.abs().mean()),
        "permutations": permutations.detach().cpu(),
        "angles": angles.detach().cpu(),
    }


def output_residual_energy(
    hidden: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
) -> float:
    error = hidden.float() @ (target.float() - prediction.float()).T
    return float(error.double().square().sum())


def all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite(item) for item in value)
    return True


def aggregate_results(
    rows: list[dict[str, Any]],
    finite_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply the immutable all-of gate from the preregistration."""
    by_candidate: dict[str, Any] = {}
    for candidate in CANDIDATES:
        selected = [row for row in rows if row["candidate"] == candidate]
        by_candidate[candidate] = {
            "cells_times_windows": len(selected),
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
            "recorded_train_gradient_predicted_ce_decrease": sum(
                float(row["recorded_train_gradient_predicted_ce_decrease"])
                for row in selected
                if row["window"] == "fit"
            ),
            "target_output_energy": {
                window: sum(
                    float(row["target_output_energy"])
                    for row in selected
                    if row["window"] == window
                )
                for window in WINDOWS
            },
            "weight_error_energy": sum(
                float(row["weight_error_energy"])
                for row in selected
                if row["window"] == "fit"
            ),
            "target_weight_energy": sum(
                float(row["target_weight_energy"])
                for row in selected
                if row["window"] == "fit"
            ),
        }

    control = by_candidate["frobenius_output32"]
    candidate = by_candidate["activation_output32"]
    energies = {
        window: {
            "frobenius": float(
                control["activation_output_residual_energy"][window]
            ),
            "activation": float(
                candidate["activation_output_residual_energy"][window]
            ),
        }
        for window in WINDOWS
    }
    advantages = {
        window: 1.0
        - energies[window]["activation"]
        / max(energies[window]["frobenius"], 1e-30)
        for window in WINDOWS
    }
    retention = (
        advantages["holdout"] / advantages["fit"]
        if abs(advantages["fit"]) > 1e-30
        else float("nan")
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
            comparisons.append(
                {
                    "phase_start": phase_start,
                    "window": window,
                    "frobenius_loss": indexed["frobenius_output32"],
                    "activation_loss": indexed["activation_output32"],
                    "activation_minus_frobenius": (
                        indexed["activation_output32"]
                        - indexed["frobenius_output32"]
                    ),
                    "activation_wins": (
                        indexed["activation_output32"]
                        < indexed["frobenius_output32"]
                    ),
                }
            )
    wins = sum(bool(row["activation_wins"]) for row in comparisons)
    mean_control = sum(float(row["frobenius_loss"]) for row in comparisons) / len(
        comparisons
    )
    mean_candidate = sum(float(row["activation_loss"]) for row in comparisons) / len(
        comparisons
    )
    holdout_task_control = float(
        control["task_gradient_predicted_ce_decrease"]["holdout"]
    )
    holdout_task_candidate = float(
        candidate["task_gradient_predicted_ce_decrease"]["holdout"]
    )

    finite_summary = all_finite(
        {
            "rows": rows,
            "finite_rows": finite_rows,
            "energies": energies,
            "advantages": advantages,
            "retention": retention,
            "holdout_task_control": holdout_task_control,
            "holdout_task_candidate": holdout_task_candidate,
            "comparisons": comparisons,
            "mean_control": mean_control,
            "mean_candidate": mean_candidate,
        }
    )
    gate = {
        "all_outputs_and_metrics_finite": finite_summary,
        "holdout_activation_residual_at_most_0p95_control": (
            energies["holdout"]["activation"]
            <= 0.95 * energies["holdout"]["frobenius"]
        ),
        "fit_advantage_positive": advantages["fit"] > 0.0,
        "holdout_to_fit_advantage_retention_at_least_0p80": retention >= 0.80,
        "holdout_predicted_ce_descent_not_worse": (
            holdout_task_candidate >= holdout_task_control
        ),
        "finite_step_ce_wins_at_least_6_of_8": wins >= 6,
        "mean_finite_step_ce_not_worse": mean_candidate <= mean_control,
    }
    passed = all(gate.values())
    return {
        "by_candidate": by_candidate,
        "activation_output_residual_energy": energies,
        "activation_residual_energy_advantage": advantages,
        "holdout_to_fit_advantage_retention": retention,
        "holdout_task_gradient_predicted_ce_decrease": {
            "frobenius": holdout_task_control,
            "activation": holdout_task_candidate,
        },
        "finite_step": {
            "comparisons": comparisons,
            "activation_wins": wins,
            "total_comparisons": len(comparisons),
            "mean_frobenius_loss": mean_control,
            "mean_activation_loss": mean_candidate,
        },
        "gate": gate,
        "passed": passed,
        "decision": (
            "ACTIVATION_WEIGHTED_OUTPUT_SELECTOR_PASS"
            if passed
            else "REJECT_ACTIVATION_WEIGHTED_OUTPUT_SELECTOR"
        ),
        "authorization": {
            "production_preregistration_authorized": passed,
            "language_model_training_authorized": False,
        },
    }


def load_probe(path: Path, expected_step: int, expected_identity: str) -> dict[str, Any]:
    probe = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(probe, dict)
        or probe.get("schema_version") != OPTIMIZER_PROBE_SCHEMA_VERSION
        or int(probe.get("step", -1)) != expected_step
        or probe.get("run_identity_sha256") != expected_identity
    ):
        raise ValueError(f"optimizer probe identity mismatch: {path}")
    return probe


def shared_hidden_chart(
    weight: torch.Tensor,
    requested_update: torch.Tensor,
    matching_direction: torch.Tensor,
    *,
    parent_stages: int,
    residual_stages: int,
    neighbors: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    """Reproduce production's common parent and residual hidden passes."""
    current = weight.float()
    # Connectivity of the parent pass is selected from non-decay Muon motion;
    # refit only the permutations while retaining exact requested angles.
    parent_permutations, parent_matching = fast_muon_matched_permutations(
        current,
        matching_direction,
        stages=parent_stages,
        neighbors=neighbors,
        seed=seed,
    )
    parent_permutations = parent_permutations.to(current.device)
    parent_angles = diagonal_metric_angles(
        current, requested_update, parent_permutations
    )
    updated = apply_givens_flow(
        current,
        parent_angles,
        parent_permutations,
        torch.argsort(parent_permutations, dim=1),
    )
    first = {
        **parent_matching,
        "stages": parent_stages,
        "coordinates": int(parent_stages * current.shape[1] // 2),
        "maximum_abs_angle": float(parent_angles.abs().max()),
        "mean_abs_angle": float(parent_angles.abs().mean()),
    }
    residual = requested_update.float() - (updated - current)
    second, second_diagnostics = fit_frobenius_pass(
        updated,
        residual,
        stages=residual_stages,
        neighbors=neighbors,
        seed=seed + 1,
    )
    residual = residual - (second - updated)
    return second, residual, [first, second_diagnostics]


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
        acquisition.get("plan", {}).get("sha256") != file_sha256(args.plan)
        or acquisition.get("decision", {}).get(
            "zero_update_selector_analysis_authorized"
        )
        is not True
    ):
        raise ValueError("accepted acquisition does not authorize this plan")
    analysis = plan["selector_analysis"]
    manifest_path = args.data_dir / "manifest.json"
    if (
        not manifest_path.is_file()
        or file_sha256(manifest_path)
        != plan["identity"]["dataset_manifest_sha256"]
    ):
        raise ValueError("dataset manifest SHA-256 mismatch")
    layers = [int(value) for value in analysis["layers"]]
    phases = [[int(value) for value in pair] for pair in analysis["phases"]]
    chart = analysis["shared_chart"]
    run_identity = acquisition["identity"]["run_identity_sha256"]

    snapshot_paths = {
        step: args.snapshot_dir / f"step_{step:06d}.pt"
        for step in sorted({value for pair in phases for value in pair})
    }
    probe_paths = {
        start: args.probe_dir / f"step_{start:06d}.pt"
        for start, _end in phases
    }
    for step, path in snapshot_paths.items():
        expected = acquisition["snapshots"]["sha256_by_step"][str(step)]
        if file_sha256(path) != expected:
            raise ValueError(f"snapshot SHA-256 mismatch at step {step}")
    for step, path in probe_paths.items():
        expected = acquisition["optimizer_probes"]["sha256_by_step"][str(step)]
        if file_sha256(path) != expected:
            raise ValueError(f"probe SHA-256 mismatch at step {step}")

    fit_spec = analysis["fit_window"]
    holdout_spec = analysis["holdout_window"]
    windows = {
        "fit": fixed_validation_batches(
            args.data_dir,
            int(fit_spec["batch_size"]),
            int(fit_spec["block_size"]) + 1,
            int(fit_spec["batches"]),
            int(fit_spec["seed"]),
        ),
        "holdout": fixed_validation_batches(
            args.data_dir,
            int(holdout_spec["batch_size"]),
            int(holdout_spec["block_size"]) + 1,
            int(holdout_spec["batches"]),
            int(holdout_spec["seed"]),
        ),
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
        gradient_losses: dict[str, float] = {}
        for window in WINDOWS:
            baseline_losses[window], hidden[window] = evaluate_and_collect(
                model, windows[window], layers, args.device
            )
            gradients[window], gradient_losses[window] = collect_cproj_gradients(
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
            weight = state["weight_before_step"].to(
                args.device, dtype=torch.float32
            )
            torch.testing.assert_close(
                weight.cpu(),
                snapshot["parameters"][name].float(),
                rtol=0.0,
                atol=0.0,
            )
            learning_rate = float(hyper["lr"])
            weight_decay = float(hyper["weight_decay"])
            applied_per_lr = state["applied_direction_per_lr"].to(
                args.device, dtype=torch.float32
            )
            requested = learning_rate * applied_per_lr
            matching_direction = applied_per_lr + weight_decay * weight
            seed = (
                int(chart["matching_seed"])
                + layer * 100000
                + phase_index * 10
            )
            hidden_weight, residual, hidden_diagnostics = shared_hidden_chart(
                weight,
                requested,
                matching_direction,
                parent_stages=int(chart["hidden_parent_stages"]),
                residual_stages=int(chart["hidden_residual_stages"]),
                neighbors=int(chart["neighbors"]),
                seed=seed,
            )
            output_source = hidden_weight.T.contiguous()
            output_residual = residual.T.contiguous()
            candidate_weight_t, activation_diagnostics = (
                fit_activation_weighted_pass(
                    output_source,
                    output_residual,
                    hidden["fit"][layer],
                    stages=int(chart["output_stages"]),
                    neighbors=int(chart["neighbors"]),
                    seed=seed + 2,
                )
            )
            control_weight_t, frobenius_diagnostics = fit_frobenius_pass(
                output_source,
                output_residual,
                stages=int(chart["output_stages"]),
                neighbors=int(chart["neighbors"]),
                seed=seed + 2,
            )
            decay = 1.0 - learning_rate * weight_decay
            candidates = {
                "frobenius_output32": control_weight_t.T.contiguous() * decay,
                "activation_output32": candidate_weight_t.T.contiguous() * decay,
            }
            target_weight_energy = float(requested.double().square().sum())
            recorded_gradient = state["gradient_after_clip"].float()
            for candidate_name, final_weight in candidates.items():
                update = (final_weight - weight).detach().cpu()
                updates[candidate_name][layer] = update
                weight_error = requested.cpu() - update
                weight_metrics = {
                    "weight_fixed_scale_recovery": fixed_scale_recovery(
                        requested.cpu(), update
                    ),
                    "weight_error_energy": float(
                        weight_error.double().square().sum()
                    ),
                    "target_weight_energy": target_weight_energy,
                    "update_fro": float(update.double().norm()),
                }
                train_task = task_descent_metrics(recorded_gradient, update)
                for window in WINDOWS:
                    output = output_space_metrics(
                        hidden[window][layer], requested.cpu(), update
                    )
                    validation_task = task_descent_metrics(
                        gradients[window][layer], update
                    )
                    metric_rows.append(
                        {
                            "phase_start": phase_start,
                            "phase_end": phase_end,
                            "layer": layer,
                            "window": window,
                            "candidate": candidate_name,
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
                            "prediction_output_energy": output[
                                "prediction_output_energy"
                            ],
                            "validation_gradient_predicted_ce_decrease": validation_task[
                                "predicted_ce_decrease"
                            ],
                            "validation_gradient_predicted_ce_decrease_per_fro": validation_task[
                                "predicted_ce_decrease_per_fro"
                            ],
                            "recorded_train_gradient_predicted_ce_decrease": train_task[
                                "predicted_ce_decrease"
                            ],
                            **weight_metrics,
                        }
                    )
            coordinate_count = sum(
                int(item["coordinates"]) for item in hidden_diagnostics
            ) + int(frobenius_diagnostics["coordinates"])
            if coordinate_count != int(chart["coordinate_count_per_layer"]):
                raise ValueError("chart coordinate budget mismatch")
            timing_rows.extend(
                {
                    "phase_start": phase_start,
                    "layer": layer,
                    "pass": pass_name,
                    "coordinates": int(diagnostics["coordinates"]),
                    "matching_total_seconds": float(
                        diagnostics["total_seconds"]
                    ),
                    "candidate_edge_fraction": float(
                        diagnostics["candidate_edge_fraction"]
                    ),
                    "maximum_abs_angle": float(
                        diagnostics["maximum_abs_angle"]
                    ),
                    "mean_abs_angle": float(diagnostics["mean_abs_angle"]),
                }
                for pass_name, diagnostics in (
                    ("hidden_parent64", hidden_diagnostics[0]),
                    ("hidden_residual24", hidden_diagnostics[1]),
                    ("frobenius_output32", frobenius_diagnostics),
                    ("activation_output32", activation_diagnostics),
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
                            model,
                            windows[window],
                            updates[candidate],
                            args.device,
                        ),
                    }
                )
        phase_summaries.append(
            {
                "phase_start": phase_start,
                "phase_end": phase_end,
                "baseline_loss": baseline_losses,
                "gradient_collection_loss": gradient_losses,
                "elapsed_seconds": time.perf_counter() - phase_started,
            }
        )
        print(json.dumps(phase_summaries[-1], sort_keys=True), flush=True)
        del model, snapshot, probe, hidden, gradients, updates
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    aggregate = aggregate_results(metric_rows, finite_rows)
    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "activation_weighted_selector_cells.csv"
    finite_path = args.output / "activation_weighted_selector_finite_ce.csv"
    timing_path = args.output / "activation_weighted_selector_timing.csv"
    write_csv(metrics_path, metric_rows)
    write_csv(finite_path, finite_rows)
    write_csv(timing_path, timing_rows)
    result = {
        "schema_version": SCHEMA_VERSION,
        "scientific_question": plan["scientific_question"],
        "execution": {
            "command": [sys.executable, *sys.argv],
            "started_at_utc": started_at,
            "host": "PRO6",
            "parameter_updates": 0,
            "watchdog": False,
            "callback": False,
        },
        "source_commit": git_commit(REPO_ROOT),
        "source": {
            "path": str(Path(__file__).relative_to(REPO_ROOT)),
            "sha256": file_sha256(Path(__file__)),
        },
        "plan": {"path": str(args.plan), "sha256": file_sha256(args.plan)},
        "acquisition_result": {
            "path": str(args.acquisition_result),
            "sha256": file_sha256(args.acquisition_result),
        },
        "identity": acquisition["identity"],
        "inputs": {
            "snapshot_paths": [
                {
                    "step": step,
                    "path": str(path),
                    "sha256": file_sha256(path),
                }
                for step, path in snapshot_paths.items()
            ],
            "probe_paths": [
                {
                    "step": step,
                    "path": str(path),
                    "sha256": file_sha256(path),
                }
                for step, path in probe_paths.items()
            ],
            "data_dir": str(args.data_dir),
            "dataset_manifest_path": str(manifest_path),
            "dataset_manifest_sha256": plan["identity"][
                "dataset_manifest_sha256"
            ],
            "fixed_eval_indices_sha256": plan["identity"][
                "fixed_eval_indices_sha256"
            ],
        },
        "protocol": {
            "layers": layers,
            "phases": phases,
            "windows": {"fit": fit_spec, "holdout": holdout_spec},
            "chart": chart,
            "parameter_updates": 0,
            "holdout_used_for_selection": False,
            "learned_basis": False,
            "inverse_metric_solve": False,
        },
        "phase_summaries": phase_summaries,
        "artifacts": {
            "cells": {"path": str(metrics_path), "sha256": file_sha256(metrics_path)},
            "finite_ce": {"path": str(finite_path), "sha256": file_sha256(finite_path)},
            "timing": {"path": str(timing_path), "sha256": file_sha256(timing_path)},
        },
        "aggregate": aggregate,
        "elapsed_seconds": time.time() - started,
    }
    result_path = args.output / "activation_weighted_selector_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
