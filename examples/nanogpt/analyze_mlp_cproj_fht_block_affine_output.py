#!/usr/bin/env python3
"""Evaluate fixed-FHT block-affine c_proj output maps without training."""

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
from examples.nanogpt.analyze_residual_compatibility import fixed_validation_batches
from examples.nanogpt.model import LearnedFHTBlockOrthogonalOutputMix


SCHEMA_VERSION = "mai_124m_mlp_cproj_fht_block_affine_output_v1"
EXPECTED_PLAN_SCHEMA = "mai_124m_mlp_cproj_fht_block_affine_output_plan_v1"
ARMS = (
    "frobenius_output32",
    "frobenius_output64",
    "fht_block32_affine_fro",
    "fht_block32_affine_activation",
)
CANDIDATES = ARMS[2:]
WINDOWS = ("fit", "holdout")


def validate_plan(plan: dict[str, Any]) -> None:
    """Fail closed on any field that defines the immutable experiment."""
    analysis = plan.get("analysis", {})
    expected = {
        "schema_version": EXPECTED_PLAN_SCHEMA,
        "layers": [0, 3, 6, 9, 11],
        "phases": [[0, 60], [60, 120], [120, 180], [180, 238]],
        "fit_window": {
            "split": "validation", "seed": 20260804, "batch_size": 2,
            "block_size": 256, "batches": 4, "rows_per_layer": 2048,
        },
        "holdout_window": {
            "split": "validation", "seed": 20260805, "batch_size": 2,
            "block_size": 256, "batches": 4, "rows_per_layer": 2048,
        },
        "shared_hidden_chart": {
            "parent_stages": 64,
            "residual_stages": 24,
            "neighbors": 64,
            "matching_seed": 20260804,
            "coordinates_per_layer": 135168,
            "feedback": "zero for this one-step prospective diagnostic",
            "weight_decay_application": "identical production ordering in every arm",
        },
        "controls": {
            "frobenius_output32": {
                "output_stages": 32,
                "output_coordinates_per_layer": 12288,
                "total_coordinates_per_layer": 147456,
            },
            "frobenius_output64": {
                "output_stages": 64,
                "output_coordinates_per_layer": 24576,
                "total_coordinates_per_layer": 159744,
                "role": "primary equal-coordinate control",
            },
        },
        "fixed_block_basis": {
            "definition": "one exact fixed signed/permuted normalized block-FHT basis and its inverse",
            "basis_block_size": 256,
            "basis_seed": 20260804,
            "affine_block_size": 32,
            "blocks": 24,
            "coordinates_per_block": 1024,
            "output_coordinates_per_layer": 24576,
            "total_coordinates_per_layer": 159744,
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
        "controls": analysis.get("controls"),
        "fixed_block_basis": analysis.get("fixed_block_basis"),
        "parameter_updates": analysis.get("parameter_updates"),
    }
    if observed != expected:
        raise ValueError("block-affine plan does not match the immutable v1 contract")
    if plan.get("authorization", {}).get("implement_and_run_zero_update_analysis") is not True:
        raise ValueError("zero-update block-affine analysis is not authorized")


def solve_block_affine(
    design: torch.Tensor,
    target: torch.Tensor,
    *,
    relative_ridge: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return the ridge-optimal local right map ``design @ B ~= target``."""
    if design.ndim != 2 or target.shape != design.shape:
        raise ValueError("design and target must have equal two-dimensional shapes")
    x = design.double()
    y = target.double()
    gram = x.T @ x
    cross = x.T @ y
    ridge = float(relative_ridge) * float(gram.diag().mean().clamp_min(1e-30))
    regularized = gram + ridge * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    coordinates = torch.linalg.solve(regularized, cross)
    prediction = x @ coordinates
    eigenvalues = torch.linalg.eigvalsh(regularized)
    return coordinates, {
        "ridge": ridge,
        "minimum_regularized_eigenvalue": float(eigenvalues.min()),
        "maximum_regularized_eigenvalue": float(eigenvalues.max()),
        "condition_number": float(eigenvalues.max() / eigenvalues.min().clamp_min(1e-30)),
        "fit_residual_energy": float((y - prediction).square().sum()),
        "target_energy": float(y.square().sum()),
    }


def _parts_energy(blocks: torch.Tensor) -> dict[str, float]:
    skew = 0.5 * (blocks - blocks.transpose(-1, -2))
    symmetric = 0.5 * (blocks + blocks.transpose(-1, -2))
    diagonal = torch.diag_embed(torch.diagonal(symmetric, dim1=-2, dim2=-1))
    symmetric_offdiag = symmetric - diagonal
    total = float(blocks.double().square().sum())
    return {
        "coordinate_energy": total,
        "skew_coordinate_energy": float(skew.double().square().sum()),
        "symmetric_offdiag_coordinate_energy": float(symmetric_offdiag.double().square().sum()),
        "diagonal_coordinate_energy": float(diagonal.double().square().sum()),
    }


def fit_fht_block_affine_pass(
    source: torch.Tensor,
    residual: torch.Tensor,
    *,
    activation: torch.Tensor | None,
    affine_block_size: int,
    basis_block_size: int,
    seed: int,
    trust_output_energy: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Fit and apply one fixed-basis block-affine output map."""
    if source.ndim != 2 or residual.shape != source.shape:
        raise ValueError("source and residual must share one matrix shape")
    if trust_output_energy < 0.0:
        raise ValueError("trust output energy must be nonnegative")
    mixer = LearnedFHTBlockOrthogonalOutputMix(
        features=int(source.shape[1]),
        stages=1,
        rotation_block_size=int(affine_block_size),
        basis_block_size=int(basis_block_size),
        seed=int(seed),
    ).to(device=source.device, dtype=torch.float32)
    with torch.no_grad():
        basis_source = mixer._basis(source.float(), 0, inverse=False)
        basis_residual = mixer._basis(residual.float(), 0, inverse=False)
        if activation is None:
            design_source = basis_source
            design_target = basis_residual
            fit_rows = int(source.shape[0])
        else:
            if activation.ndim != 2 or activation.shape[1] != source.shape[0]:
                raise ValueError("activation/source shapes disagree")
            h = activation.to(source.device, dtype=torch.float32)
            design_source = h @ basis_source
            design_target = h @ basis_residual
            fit_rows = int(h.shape[0])
        source_blocks = basis_source.reshape(
            source.shape[0], mixer.rotation_blocks, mixer.rotation_block_size
        )
        design_blocks = design_source.reshape(
            design_source.shape[0], mixer.rotation_blocks, mixer.rotation_block_size
        )
        target_blocks = design_target.reshape_as(design_blocks)
        solved: list[torch.Tensor] = []
        records: list[dict[str, float]] = []
        for block in range(mixer.rotation_blocks):
            coordinates, record = solve_block_affine(
                design_blocks[:, block], target_blocks[:, block]
            )
            solved.append(coordinates)
            records.append(record)
        raw = torch.stack(solved).float()
        raw_delta_blocks = torch.einsum("nbi,bij->nbj", source_blocks, raw)
        raw_output_energy = float(raw_delta_blocks.double().square().sum())
        scale = min(1.0, math.sqrt(trust_output_energy / max(raw_output_energy, 1e-30)))
        bounded = raw * scale
        delta_blocks = torch.einsum("nbi,bij->nbj", source_blocks, bounded)
        bounded_output_energy = float(delta_blocks.double().square().sum())
        updated_basis = (source_blocks + delta_blocks).reshape_as(basis_source)
        updated = mixer._basis(updated_basis, 0, inverse=True)
        identities = torch.eye(
            mixer.rotation_block_size, device=bounded.device, dtype=bounded.dtype
        ).expand(mixer.rotation_blocks, -1, -1)
        min_singular_value = float(torch.linalg.svdvals((identities + bounded).double()).min())
    diagnostics: dict[str, Any] = {
        "coordinates": int(raw.numel()),
        "blocks": int(mixer.rotation_blocks),
        "coordinates_per_block": int(raw.shape[-1] * raw.shape[-2]),
        "fit_rows": fit_rows,
        "raw_output_delta_energy": raw_output_energy,
        "bounded_output_delta_energy": bounded_output_energy,
        "trust_output_energy": float(trust_output_energy),
        "trust_scale": scale,
        "trust_energy_obeyed": bounded_output_energy <= trust_output_energy + max(1e-12, 1e-5 * trust_output_energy),
        "minimum_singular_value_i_plus_b": min_singular_value,
        "maximum_abs_coordinate": float(bounded.abs().max()),
        "maximum_condition_number": max(row["condition_number"] for row in records),
        "minimum_regularized_eigenvalue": min(row["minimum_regularized_eigenvalue"] for row in records),
        "maximum_regularized_eigenvalue": max(row["maximum_regularized_eigenvalue"] for row in records),
        "fit_residual_energy": sum(row["fit_residual_energy"] for row in records),
        "fit_target_energy": sum(row["target_energy"] for row in records),
        **_parts_energy(bounded),
    }
    if not all_finite(diagnostics) or not torch.isfinite(updated).all():
        raise ValueError("block-affine solve produced a nonfinite result")
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
            "coordinates_per_layer": sorted({int(row["coordinates_per_layer"]) for row in selected}),
            "activation_output_residual_energy": {
                window: sum(float(row["activation_output_residual_energy"]) for row in selected if row["window"] == window)
                for window in WINDOWS
            },
            "task_gradient_predicted_ce_decrease": {
                window: sum(float(row["validation_gradient_predicted_ce_decrease"]) for row in selected if row["window"] == window)
                for window in WINDOWS
            },
            "update_energy": sum(float(row["update_energy"]) for row in selected if row["window"] == "fit"),
        }
    finite_index = {
        (int(row["phase_start"]), str(row["window"]), str(row["arm"])): float(row["loss"])
        for row in finite_rows if row["arm"] in ARMS
    }
    comparisons: dict[str, list[dict[str, Any]]] = {candidate: [] for candidate in CANDIDATES}
    for candidate in CANDIDATES:
        for phase in sorted({int(row["phase_start"]) for row in finite_rows}):
            for window in WINDOWS:
                control_loss = finite_index[(phase, window, "frobenius_output64")]
                candidate_loss = finite_index[(phase, window, candidate)]
                fro_affine_loss = finite_index[(phase, window, "fht_block32_affine_fro")]
                comparisons[candidate].append({
                    "phase_start": phase,
                    "window": window,
                    "candidate_minus_output64": candidate_loss - control_loss,
                    "candidate_minus_fro_affine": candidate_loss - fro_affine_loss,
                    "candidate_wins_output64": candidate_loss < control_loss,
                    "candidate_wins_fro_affine": candidate_loss < fro_affine_loss,
                })
    control = by_arm["frobenius_output64"]
    gates: dict[str, dict[str, bool]] = {}
    summaries: dict[str, Any] = {}
    all_chart_finite = all_finite(chart_rows)
    for candidate in CANDIDATES:
        candidate_metrics = by_arm[candidate]
        comp = comparisons[candidate]
        residual_ratio = float(candidate_metrics["activation_output_residual_energy"]["holdout"]) / max(float(control["activation_output_residual_energy"]["holdout"]), 1e-30)
        task = float(candidate_metrics["task_gradient_predicted_ce_decrease"]["holdout"])
        control_task = float(control["task_gradient_predicted_ce_decrease"]["holdout"])
        candidate_losses = [finite_index[(row["phase_start"], row["window"], candidate)] for row in comp]
        control_losses = [finite_index[(row["phase_start"], row["window"], "frobenius_output64")] for row in comp]
        mean_candidate = sum(candidate_losses) / len(candidate_losses)
        mean_control = sum(control_losses) / len(control_losses)
        wins = sum(bool(row["candidate_wins_output64"]) for row in comp)
        holdout_wins = sum(bool(row["candidate_wins_output64"]) for row in comp if row["window"] == "holdout")
        candidate_chart = [row for row in chart_rows if row["arm"] == candidate]
        common = {
            "all_outputs_solves_and_metrics_finite": all_chart_finite and all_finite({"rows": rows, "finite": finite_rows}),
            "equal_coordinate_budget": candidate_metrics["coordinates_per_layer"] == [159744] and control["coordinates_per_layer"] == [159744],
            "output_energy_trust_obeyed_every_cell": all(bool(row["trust_energy_obeyed"]) for row in candidate_chart),
            "minimum_singular_value_at_least_0p95": min(float(row["minimum_singular_value_i_plus_b"]) for row in candidate_chart) >= 0.95,
            "heldout_residual_at_most_0p95_output64": residual_ratio <= 0.95,
            "heldout_task_descent_gate": task > 0.0 and task >= 1.10 * control_task and task >= 0.004353292856055148,
            "all_4_holdout_ce_wins": holdout_wins == 4,
            "at_least_7_of_8_ce_wins": wins >= 7,
            "mean_ce_at_least_0p0005_better_and_prior_best": mean_candidate <= mean_control - 0.0005 and mean_candidate <= 7.180657014250755,
        }
        if candidate == "fht_block32_affine_activation":
            activation_holdout_wins = sum(bool(row["candidate_wins_fro_affine"]) for row in comp if row["window"] == "holdout")
            fro_mean = sum(finite_index[(row["phase_start"], row["window"], "fht_block32_affine_fro")] for row in comp) / len(comp)
            common.update({
                "beats_fro_affine_on_at_least_3_holdouts": activation_holdout_wins >= 3,
                "mean_ce_lower_than_fro_affine": mean_candidate < fro_mean,
            })
        gates[candidate] = common
        summaries[candidate] = {
            "heldout_residual_ratio_to_output64": residual_ratio,
            "heldout_task_descent": task,
            "heldout_task_ratio_to_output64": task / max(control_task, 1e-30),
            "wins_vs_output64": wins,
            "holdout_wins_vs_output64": holdout_wins,
            "mean_ce": mean_candidate,
            "mean_output64_ce": mean_control,
            "passed": all(common.values()),
        }
    if summaries["fht_block32_affine_fro"]["passed"]:
        selected = "fht_block32_affine_fro"
    elif summaries["fht_block32_affine_activation"]["passed"]:
        selected = "fht_block32_affine_activation"
    else:
        selected = None
    return {
        "by_arm": by_arm,
        "comparisons": comparisons,
        "candidate_summaries": summaries,
        "gate": gates,
        "selected": selected,
        "passed": selected is not None,
        "decision": "FHT_BLOCK_AFFINE_OUTPUT_PASS" if selected else "REJECT_FHT_BLOCK_AFFINE_OUTPUT",
        "chart_decomposition": {
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
        "authorization": {
            "production_preregistration_authorized": selected is not None,
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
        file_sha256(Path(plan["identity"]["block_cayley_result"])) != plan["identity"]["block_cayley_result_sha256"]
        or file_sha256(args.acquisition_result) != plan["identity"]["acquisition_result_sha256"]
        or acquisition["identity"]["run_identity_sha256"] != plan["identity"]["run_identity_sha256"]
    ):
        raise ValueError("block-affine plan input identity mismatch")
    manifest_path = args.data_dir / "manifest.json"
    if not manifest_path.is_file() or file_sha256(manifest_path) != plan["identity"]["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest SHA-256 mismatch")

    analysis = plan["analysis"]
    layers = [int(value) for value in analysis["layers"]]
    phases = [[int(value) for value in pair] for pair in analysis["phases"]]
    hidden_chart = analysis["shared_hidden_chart"]
    controls = analysis["controls"]
    basis = analysis["fixed_block_basis"]
    run_identity = plan["identity"]["run_identity_sha256"]
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
            if hidden_coordinates != int(hidden_chart["coordinates_per_layer"]):
                raise ValueError("shared hidden coordinate budget mismatch")
            source = hidden_weight.T.contiguous()
            residual_t = residual.T.contiguous()
            controls_t: dict[str, torch.Tensor] = {}
            control_diagnostics: dict[str, dict[str, Any]] = {}
            for arm in ARMS[:2]:
                spec = controls[arm]
                controls_t[arm], control_diagnostics[arm] = fit_frobenius_pass(
                    source, residual_t, stages=int(spec["output_stages"]),
                    neighbors=int(hidden_chart["neighbors"]), seed=seed + 2,
                )
            output64_energy = float((controls_t["frobenius_output64"] - source).double().square().sum())
            candidates_t: dict[str, torch.Tensor] = {}
            for arm, activation in (
                ("fht_block32_affine_fro", None),
                ("fht_block32_affine_activation", hidden["fit"][layer]),
            ):
                candidates_t[arm], diagnostics = fit_fht_block_affine_pass(
                    source, residual_t, activation=activation,
                    affine_block_size=int(basis["affine_block_size"]),
                    basis_block_size=int(basis["basis_block_size"]),
                    seed=int(basis["basis_seed"]), trust_output_energy=output64_energy,
                )
                chart_rows.append({"phase_start": phase_start, "layer": layer, "arm": arm, **diagnostics})
            coordinate_counts = {
                "frobenius_output32": hidden_coordinates + int(control_diagnostics["frobenius_output32"]["coordinates"]),
                "frobenius_output64": hidden_coordinates + int(control_diagnostics["frobenius_output64"]["coordinates"]),
                "fht_block32_affine_fro": hidden_coordinates + int(basis["output_coordinates_per_layer"]),
                "fht_block32_affine_activation": hidden_coordinates + int(basis["output_coordinates_per_layer"]),
            }
            expected_counts = {
                "frobenius_output32": int(controls["frobenius_output32"]["total_coordinates_per_layer"]),
                "frobenius_output64": int(controls["frobenius_output64"]["total_coordinates_per_layer"]),
                "fht_block32_affine_fro": int(basis["total_coordinates_per_layer"]),
                "fht_block32_affine_activation": int(basis["total_coordinates_per_layer"]),
            }
            if coordinate_counts != expected_counts:
                raise ValueError("output chart coordinate budget mismatch")
            decay = 1.0 - learning_rate * weight_decay
            final_weights = {
                **{arm: value.T.contiguous() * decay for arm, value in controls_t.items()},
                **{arm: value.T.contiguous() * decay for arm, value in candidates_t.items()},
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
                finite_rows.append({
                    "phase_start": phase_start, "phase_end": phase_end,
                    "window": window, "arm": arm,
                    "loss": evaluate_with_updates(model, windows[window], updates[arm], args.device),
                })
        phase_summaries.append({
            "phase_start": phase_start, "phase_end": phase_end,
            "baseline_loss": baseline_losses,
            "elapsed_seconds": time.perf_counter() - phase_started,
        })
        print(json.dumps(phase_summaries[-1], sort_keys=True), flush=True)
        del model, snapshot, probe, hidden, gradients, updates
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    aggregate = aggregate_results(metric_rows, finite_rows, chart_rows)
    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "fht_block_affine_cells.csv"
    finite_path = args.output / "fht_block_affine_finite_ce.csv"
    chart_path = args.output / "fht_block_affine_chart.csv"
    write_csv(metrics_path, metric_rows)
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
    result_path = args.output / "fht_block_affine_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
