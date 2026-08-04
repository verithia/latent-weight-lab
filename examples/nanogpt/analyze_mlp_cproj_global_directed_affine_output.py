#!/usr/bin/env python3
"""Evaluate globally supported directed c_proj output maps without training."""

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
from examples.nanogpt.analyze_mlp_cproj_fht_block_affine_output import _parts_energy
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


SCHEMA_VERSION = "mai_124m_mlp_cproj_global_directed_affine_output_v1"
EXPECTED_PLAN_SCHEMA = "mai_124m_mlp_cproj_global_directed_affine_output_plan_v1"
ARMS = (
    "frobenius_output32",
    "global_directed16_activation",
    "global_directed16_activation_task_consensus",
)
CANDIDATES = ARMS[1:]
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
        "control": {
            "name": "frobenius_output32", "output_stages": 32,
            "output_coordinates_per_layer": 12288, "total_coordinates_per_layer": 147456,
        },
        "directed_map": {
            "source_channels": 768, "target_channels": 768,
            "incoming_coordinates_per_target": 16,
            "output_coordinates_per_layer": 12288,
            "total_coordinates_per_layer": 147456,
            "diagonal_edges_allowed": True,
        },
        "parameter_updates": 0,
    }
    directed = analysis.get("directed_map", {})
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
        raise ValueError("global-directed plan does not match immutable v1 contract")
    if plan.get("authorization", {}).get("implement_and_run_zero_update_analysis") is not True:
        raise ValueError("zero-update global-directed analysis is not authorized")


def _normalized_consensus(train_gradient: torch.Tensor, fit_gradient: torch.Tensor) -> tuple[torch.Tensor, float]:
    train = train_gradient.float()
    fit = fit_gradient.to(train.device, dtype=torch.float32)
    train_norm = train.norm().clamp_min(1e-30)
    fit_norm = fit.norm().clamp_min(1e-30)
    cosine = float((train.double() * fit.double()).sum() / (train.double().norm() * fit.double().norm()).clamp_min(1e-30))
    return 0.5 * (train / train_norm + fit / fit_norm), cosine


def directed_support_scores(
    source: torch.Tensor,
    residual: torch.Tensor,
    activation: torch.Tensor,
    consensus_gradient: torch.Tensor,
    *,
    relative_ridge: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Return residual-only and residual-plus-task scores for every i->j edge."""
    if source.ndim != 2 or residual.shape != source.shape:
        raise ValueError("source and residual must have equal two-dimensional shapes")
    if activation.ndim != 2 or activation.shape[1] != source.shape[0]:
        raise ValueError("activation/source shapes disagree")
    if consensus_gradient.shape != source.T.shape:
        raise ValueError("consensus gradient/source shapes disagree")
    h = activation.to(source.device, dtype=torch.float32)
    design = h @ source.float()
    target = h @ residual.float()
    column_energy = design.square().sum(dim=0)
    ridge = float(relative_ridge) * float(column_energy.mean().clamp_min(1e-30))
    cross = design.T @ target
    beta = cross / (column_energy[:, None] + ridge)
    residual_score = 2.0 * beta * cross - beta.square() * column_energy[:, None]
    gradient_dot = (consensus_gradient.to(source.device, dtype=torch.float32) @ source.float()).T
    task_score = -beta * gradient_dot
    residual_rms = residual_score.square().mean().sqrt().clamp_min(1e-30)
    task_rms = task_score.square().mean().sqrt().clamp_min(1e-30)
    combined = residual_score / residual_rms + task_score / task_rms
    diagnostics = {
        "single_edge_ridge": ridge,
        "residual_score_rms": float(residual_rms),
        "task_score_rms": float(task_rms),
        "maximum_residual_score": float(residual_score.max()),
        "maximum_task_score": float(task_score.max()),
    }
    if not all_finite(diagnostics) or not torch.isfinite(combined).all():
        raise ValueError("directed support score is nonfinite")
    return residual_score, combined, diagnostics


def fit_global_directed_map(
    source: torch.Tensor,
    residual: torch.Tensor,
    activation: torch.Tensor,
    score: torch.Tensor,
    *,
    incoming: int,
    trust_output_energy: float,
    relative_ridge: float = 1e-6,
    return_mapping: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]] | tuple[
    torch.Tensor, torch.Tensor, dict[str, Any], torch.Tensor
]:
    """Select global directed supports, jointly refit, and apply sparse ``B``."""
    outputs = int(source.shape[1])
    if score.shape != (outputs, outputs):
        raise ValueError("score matrix has the wrong shape")
    if incoming <= 0 or incoming > outputs:
        raise ValueError("incoming support size is invalid")
    h = activation.to(source.device, dtype=torch.float32)
    design = h @ source.float()
    target = h @ residual.float()
    supports = torch.topk(score, k=incoming, dim=0, largest=True, sorted=True).indices
    selected_design = design[:, supports.T].permute(1, 0, 2).contiguous()
    target_by_output = target.T.unsqueeze(-1)
    gram = selected_design.transpose(1, 2).double() @ selected_design.double()
    cross = selected_design.transpose(1, 2).double() @ target_by_output.double()
    ridge = float(relative_ridge) * torch.diagonal(gram, dim1=-2, dim2=-1).mean(dim=1).clamp_min(1e-30)
    eye = torch.eye(incoming, device=gram.device, dtype=gram.dtype).unsqueeze(0)
    coefficients = torch.linalg.solve(gram + ridge[:, None, None] * eye, cross).squeeze(-1).float()
    mapping = torch.zeros(outputs, outputs, device=source.device, dtype=torch.float32)
    targets = torch.arange(outputs, device=source.device).unsqueeze(1).expand(-1, incoming)
    mapping[supports.T, targets] = coefficients
    raw_delta = source.float() @ mapping
    raw_energy = float(raw_delta.double().square().sum())
    scale = min(1.0, math.sqrt(float(trust_output_energy) / max(raw_energy, 1e-30)))
    bounded = mapping * scale
    delta = source.float() @ bounded
    bounded_energy = float(delta.double().square().sum())
    identity = torch.eye(outputs, device=source.device, dtype=torch.float32)
    minimum_singular = float(torch.linalg.svdvals((identity + bounded).double()).min())
    updated = source.float() + delta
    diagnostics: dict[str, Any] = {
        "coordinates": int(outputs * incoming),
        "incoming_per_target": int(incoming),
        "raw_output_delta_energy": raw_energy,
        "bounded_output_delta_energy": bounded_energy,
        "trust_output_energy": float(trust_output_energy),
        "trust_scale": scale,
        "trust_energy_obeyed": bounded_energy <= trust_output_energy + max(1e-12, 1e-5 * trust_output_energy),
        "minimum_singular_value_i_plus_b": minimum_singular,
        "maximum_abs_coordinate": float(bounded.abs().max()),
        "minimum_joint_ridge": float(ridge.min()),
        "maximum_joint_ridge": float(ridge.max()),
        **_parts_energy(bounded.unsqueeze(0)),
    }
    if not all_finite(diagnostics) or not torch.isfinite(updated).all():
        raise ValueError("global-directed map produced a nonfinite result")
    result = (updated, supports.detach().cpu(), diagnostics)
    if return_mapping:
        return (*result, bounded)
    return result


def support_overlap(first: torch.Tensor, second: torch.Tensor) -> float:
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("support tensors must have equal [incoming,target] shapes")
    overlap = []
    for target in range(first.shape[1]):
        overlap.append(len(set(first[:, target].tolist()) & set(second[:, target].tolist())) / first.shape[0])
    return sum(overlap) / len(overlap)


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
    control = by_arm["frobenius_output32"]
    summaries: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    comparisons: dict[str, list[dict[str, Any]]] = {}
    for candidate in CANDIDATES:
        comp: list[dict[str, Any]] = []
        for phase in sorted({int(row["phase_start"]) for row in finite_rows}):
            for window in WINDOWS:
                candidate_loss = index[(phase, window, candidate)]
                control_loss = index[(phase, window, "frobenius_output32")]
                activation_loss = index[(phase, window, "global_directed16_activation")]
                comp.append({
                    "phase_start": phase, "window": window,
                    "candidate_minus_control": candidate_loss - control_loss,
                    "candidate_minus_activation_only": candidate_loss - activation_loss,
                    "candidate_wins_control": candidate_loss < control_loss,
                    "candidate_wins_activation_only": candidate_loss < activation_loss,
                })
        comparisons[candidate] = comp
        metrics = by_arm[candidate]
        residual_ratio = float(metrics["activation_output_residual_energy"]["holdout"]) / max(float(control["activation_output_residual_energy"]["holdout"]), 1e-30)
        task = float(metrics["task_gradient_predicted_ce_decrease"]["holdout"])
        control_task = float(control["task_gradient_predicted_ce_decrease"]["holdout"])
        candidate_losses = [index[(row["phase_start"], row["window"], candidate)] for row in comp]
        control_losses = [index[(row["phase_start"], row["window"], "frobenius_output32")] for row in comp]
        activation_losses = [index[(row["phase_start"], row["window"], "global_directed16_activation")] for row in comp]
        mean_candidate = sum(candidate_losses) / len(candidate_losses)
        mean_control = sum(control_losses) / len(control_losses)
        candidate_chart = [row for row in chart_rows if row["arm"] == candidate]
        wins = sum(bool(row["candidate_wins_control"]) for row in comp)
        holdout_wins = sum(bool(row["candidate_wins_control"]) for row in comp if row["window"] == "holdout")
        gate = {
            "all_outputs_scores_solves_and_metrics_finite": all_finite({"rows": rows, "finite": finite_rows, "chart": chart_rows}),
            "equal_coordinate_budget": metrics["coordinates_per_layer"] == [147456] and control["coordinates_per_layer"] == [147456],
            "output_energy_trust_obeyed_every_cell": all(bool(row["trust_energy_obeyed"]) for row in candidate_chart),
            "minimum_singular_value_at_least_0p95": min(float(row["minimum_singular_value_i_plus_b"]) for row in candidate_chart) >= 0.95,
            "heldout_residual_at_most_0p95_control": residual_ratio <= 0.95,
            "heldout_task_descent_gate": task > 0.0 and task >= 1.15 * control_task and task >= 0.004353292856055148,
            "all_4_holdout_ce_wins": holdout_wins == 4,
            "at_least_7_of_8_ce_wins": wins >= 7,
            "mean_ce_at_least_0p0005_better_and_prior_best": mean_candidate <= mean_control - 0.0005 and mean_candidate <= 7.180657014250755,
        }
        if candidate.endswith("task_consensus"):
            gate.update({
                "beats_activation_only_on_at_least_3_holdouts": sum(bool(row["candidate_wins_activation_only"]) for row in comp if row["window"] == "holdout") >= 3,
                "mean_ce_lower_than_activation_only": mean_candidate < sum(activation_losses) / len(activation_losses),
            })
        gates[candidate] = gate
        summaries[candidate] = {
            "heldout_residual_ratio_to_control": residual_ratio,
            "heldout_task_descent": task,
            "heldout_task_ratio_to_control": task / max(control_task, 1e-30),
            "wins_vs_control": wins, "holdout_wins_vs_control": holdout_wins,
            "mean_ce": mean_candidate, "mean_control_ce": mean_control,
            "passed": all(gate.values()),
        }
    if summaries["global_directed16_activation"]["passed"]:
        selected = "global_directed16_activation"
    elif summaries["global_directed16_activation_task_consensus"]["passed"]:
        selected = "global_directed16_activation_task_consensus"
    else:
        selected = None
    return {
        "by_arm": by_arm, "comparisons": comparisons,
        "candidate_summaries": summaries, "gate": gates,
        "chart": {
            "mean_train_fit_gradient_cosine": sum(float(row["train_fit_gradient_cosine"]) for row in chart_rows if row["arm"] == CANDIDATES[0]) / len([row for row in chart_rows if row["arm"] == CANDIDATES[0]]),
            "mean_support_overlap": sum(float(row["support_overlap_with_other_arm"]) for row in chart_rows if row["arm"] == CANDIDATES[0]) / len([row for row in chart_rows if row["arm"] == CANDIDATES[0]]),
            **{
                candidate: {
                    "minimum_trust_scale": min(float(row["trust_scale"]) for row in chart_rows if row["arm"] == candidate),
                    "minimum_singular_value_i_plus_b": min(float(row["minimum_singular_value_i_plus_b"]) for row in chart_rows if row["arm"] == candidate),
                    "coordinate_energy": sum(float(row["coordinate_energy"]) for row in chart_rows if row["arm"] == candidate),
                    "skew_coordinate_energy": sum(float(row["skew_coordinate_energy"]) for row in chart_rows if row["arm"] == candidate),
                    "symmetric_offdiag_coordinate_energy": sum(float(row["symmetric_offdiag_coordinate_energy"]) for row in chart_rows if row["arm"] == candidate),
                    "diagonal_coordinate_energy": sum(float(row["diagonal_coordinate_energy"]) for row in chart_rows if row["arm"] == candidate),
                }
                for candidate in CANDIDATES
            },
        },
        "selected": selected, "passed": selected is not None,
        "decision": "GLOBAL_DIRECTED16_OUTPUT_PASS" if selected else "REJECT_GLOBAL_DIRECTED16_OUTPUT",
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
    if (
        file_sha256(Path(plan["identity"]["block_affine_result"])) != plan["identity"]["block_affine_result_sha256"]
        or file_sha256(args.acquisition_result) != plan["identity"]["acquisition_result_sha256"]
        or acquisition["identity"]["run_identity_sha256"] != plan["identity"]["run_identity_sha256"]
    ):
        raise ValueError("global-directed plan input identity mismatch")
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
            control_t, control_diagnostics = fit_frobenius_pass(
                source, residual_t, stages=int(control["output_stages"]),
                neighbors=int(hidden_chart["neighbors"]), seed=seed + 2,
            )
            control_output_energy = float((control_t - source).double().square().sum())
            consensus_gradient, gradient_cosine = _normalized_consensus(
                state["gradient_after_clip"].to(args.device), gradients["fit"][layer].to(args.device)
            )
            activation_score, consensus_score, score_diagnostics = directed_support_scores(
                source, residual_t, hidden["fit"][layer], consensus_gradient
            )
            fitted: dict[str, tuple[torch.Tensor, torch.Tensor, dict[str, Any]]] = {}
            for arm, score in (
                ("global_directed16_activation", activation_score),
                ("global_directed16_activation_task_consensus", consensus_score),
            ):
                fitted[arm] = fit_global_directed_map(
                    source, residual_t, hidden["fit"][layer], score,
                    incoming=int(directed["incoming_coordinates_per_target"]),
                    trust_output_energy=control_output_energy,
                )
            overlap = support_overlap(fitted[CANDIDATES[0]][1], fitted[CANDIDATES[1]][1])
            for arm in CANDIDATES:
                diagnostics = fitted[arm][2]
                chart_rows.append({
                    "phase_start": phase_start, "layer": layer, "arm": arm,
                    "train_fit_gradient_cosine": gradient_cosine,
                    "support_overlap_with_other_arm": overlap,
                    **score_diagnostics, **diagnostics,
                })
            coordinate_counts = {
                "frobenius_output32": hidden_coordinates + int(control_diagnostics["coordinates"]),
                CANDIDATES[0]: hidden_coordinates + int(fitted[CANDIDATES[0]][2]["coordinates"]),
                CANDIDATES[1]: hidden_coordinates + int(fitted[CANDIDATES[1]][2]["coordinates"]),
            }
            if set(coordinate_counts.values()) != {int(control["total_coordinates_per_layer"])}:
                raise ValueError("global-directed coordinate budget mismatch")
            decay = 1.0 - learning_rate * weight_decay
            final_weights = {
                "frobenius_output32": control_t.T.contiguous() * decay,
                CANDIDATES[0]: fitted[CANDIDATES[0]][0].T.contiguous() * decay,
                CANDIDATES[1]: fitted[CANDIDATES[1]][0].T.contiguous() * decay,
            }
            for arm, final_weight in final_weights.items():
                update = (final_weight - weight).detach().cpu()
                updates[arm][layer] = update
                error = requested.cpu() - update
                for window in WINDOWS:
                    output = output_space_metrics(hidden[window][layer], requested.cpu(), update)
                    task = task_descent_metrics(gradients[window][layer], update)
                    metric_rows.append({
                        "phase_start": phase_start, "phase_end": phase_end,
                        "layer": layer, "window": window, "arm": arm,
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
    metrics_path = args.output / "global_directed_affine_cells.csv"
    finite_path = args.output / "global_directed_affine_finite_ce.csv"
    chart_path = args.output / "global_directed_affine_chart.csv"
    write_csv(metrics_path, metric_rows)
    write_csv(finite_path, finite_rows)
    write_csv(chart_path, chart_rows)
    result = {
        "schema_version": SCHEMA_VERSION, "scientific_question": plan["scientific_question"],
        "source_commit": git_commit(REPO_ROOT),
        "source": {"path": str(Path(__file__).relative_to(REPO_ROOT)), "sha256": file_sha256(Path(__file__))},
        "execution": {"command": [sys.executable, *sys.argv], "started_at_utc": started_at, "host": "PRO6", "parameter_updates": 0, "watchdog": False, "callback": False},
        "plan": {"path": str(args.plan), "sha256": file_sha256(args.plan)},
        "acquisition_result": {"path": str(args.acquisition_result), "sha256": file_sha256(args.acquisition_result)},
        "identity": acquisition["identity"], "protocol": analysis,
        "phase_summaries": phase_summaries,
        "artifacts": {
            "cells": {"path": str(metrics_path), "sha256": file_sha256(metrics_path)},
            "finite_ce": {"path": str(finite_path), "sha256": file_sha256(finite_path)},
            "chart": {"path": str(chart_path), "sha256": file_sha256(chart_path)},
        },
        "aggregate": aggregate, "elapsed_seconds": time.time() - started,
    }
    result_path = args.output / "global_directed_affine_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
