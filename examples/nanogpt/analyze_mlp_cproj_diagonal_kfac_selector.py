#!/usr/bin/env python3
"""Compare raw and diagonal activation-error c_proj output selectors.

This zero-update diagnostic reuses the accepted matched-attention trajectory.
Every arm has the same hidden64+24 parent, output32 coordinate budget, raw
Frobenius angle fit, and weight-decay ordering.  The only intervention is the
geometry used to select output-channel pairs.  The diagonal-KFAC arm uses both
post-GELU activations and per-output RMS backpropagated task error from the fit
window.  Holdout data is scoring-only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
from collections import defaultdict
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
    evaluate_with_updates,
    task_descent_metrics,
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


SCHEMA_VERSION = "mai_124m_mlp_cproj_diagonal_kfac_selector_result_v1"
EXPECTED_PLAN_SCHEMA = "mai_124m_mlp_cproj_diagonal_kfac_selector_plan_v1"
CALIBRATION_PLAN_SCHEMA = (
    "mai_124m_mlp_cproj_5tpp_functional_metric_calibration_plan_v1"
)
CALIBRATION_RESULT_SCHEMA = (
    "mai_124m_mlp_cproj_5tpp_functional_metric_calibration_result_v1"
)
VARIANTS = (
    "frobenius_output32",
    "activation_selector_output32",
    "error_selector_output32",
    "diagonal_kfac_selector_output32",
)
SMALLEST_PASS_ORDER = VARIANTS[1:]
WINDOWS = ("fit", "holdout")


def validate_plan(plan: dict[str, Any]) -> None:
    analysis = plan.get("analysis", {})
    schema = plan.get("schema_version")
    if schema == EXPECTED_PLAN_SCHEMA:
        layers = [0, 3, 6, 9, 11]
        phases = [[0, 60], [60, 120], [120, 180], [180, 238]]
        fit_seed = 20260804
        holdout_seed = 20260805
        matching_seed = 20260804
    elif schema == CALIBRATION_PLAN_SCHEMA:
        layers = list(range(8))
        phases = [
            [0, 594],
            [594, 1188],
            [1188, 1782],
            [1782, 2373],
        ]
        fit_seed = 20260806
        holdout_seed = 20260807
        matching_seed = 20260806
    else:
        raise ValueError("unknown diagonal-KFAC/calibration plan schema")
    expected = {
        "schema_version": schema,
        "parameter_updates": 0,
        "layers": layers,
        "phases": phases,
        "fit_window": {
            "split": "validation",
            "seed": fit_seed,
            "batch_size": 2,
            "block_size": 256,
            "batches": 4,
            "rows_per_layer": 2048,
            "participates_in_selection": True,
        },
        "holdout_window": {
            "split": "validation",
            "seed": holdout_seed,
            "batch_size": 2,
            "block_size": 256,
            "batches": 4,
            "rows_per_layer": 2048,
            "participates_in_selection": False,
        },
        "shared_chart": {
            "hidden_parent_stages": 64,
            "hidden_residual_stages": 24,
            "output_stages": 32,
            "neighbors": 64,
            "matching_seed": matching_seed,
            "coordinate_count_per_layer": 147456,
            "feedback": "zero for this one-step prospective diagnostic",
            "weight_decay_application": (
                "identical production ordering in every arm"
            ),
        },
        "scale_bounds": [0.25, 4.0],
        "smallest_pass_order": list(SMALLEST_PASS_ORDER),
    }
    observed = {
        "schema_version": plan.get("schema_version"),
        "parameter_updates": analysis.get("parameter_updates"),
        "layers": analysis.get("layers"),
        "phases": analysis.get("phases"),
        "fit_window": analysis.get("fit_window"),
        "holdout_window": analysis.get("holdout_window"),
        "shared_chart": analysis.get("shared_chart"),
        "scale_bounds": analysis.get("output_error_scale", {}).get("bounds"),
        "smallest_pass_order": analysis.get("smallest_pass_order"),
    }
    if observed != expected:
        raise ValueError("diagonal-KFAC plan does not match the v1 contract")
    authorization = plan.get("authorization", {})
    if schema == EXPECTED_PLAN_SCHEMA:
        if authorization.get("implement_and_run_zero_update_analysis") is not True:
            raise ValueError("zero-update diagonal-KFAC analysis is not authorized")
    else:
        if authorization.get("run_zero_update_metric_calibration") is not True:
            raise ValueError("zero-update 5TPP metric calibration is not authorized")
        if authorization.get("implement_candidate_structure") is not False:
            raise ValueError("calibration must not authorize candidate implementation")
    if authorization.get("run_language_model_training") is not False:
        raise ValueError("the plan must not authorize language-model training")


def apply_plan_authorization(
    aggregate: dict[str, Any], plan_schema: str
) -> dict[str, Any]:
    """Keep the 5TPP audit diagnostic even if its metric clears the old gate."""
    if plan_schema != CALIBRATION_PLAN_SCHEMA:
        return aggregate
    passed = aggregate.get("selected_variant") is not None
    aggregate["classification"] = (
        "5TPP_STATIC_METRIC_SIGNAL_PRESENT_NEEDS_TEMPORAL_VALIDATION"
        if passed
        else "REJECT_5TPP_STATIC_FUNCTIONAL_METRIC_SELECTOR"
    )
    aggregate["authorization"] = {
        "temporal_residual_decomposition_authorized": True,
        "short_shadow_rollout_authorized": passed,
        "production_implementation_authorized": False,
        "exact_config_mfu_preflight_authorized": False,
        "language_model_training_authorized": False,
    }
    return aggregate


class CProjGeometryCollector:
    """Collect c_proj inputs and output derivatives for selected layers."""

    def __init__(self, model: torch.nn.Module, layers: list[int]) -> None:
        self.layers = set(layers)
        self.hidden: dict[int, list[torch.Tensor]] = defaultdict(list)
        self.errors: dict[int, list[torch.Tensor]] = defaultdict(list)
        self.handles = []
        for layer, block in enumerate(model.transformer.h):
            if layer not in self.layers:
                continue
            self.handles.append(
                block.mlp.c_proj.register_forward_hook(self._hook(layer))
            )

    def _hook(self, layer: int):
        def hook(_module, inputs, output):
            hidden = inputs[0]
            if not torch.is_tensor(hidden) or not torch.is_tensor(output):
                raise TypeError("c_proj hook expected tensor input and output")
            self.hidden[layer].append(
                hidden.detach().float().reshape(-1, hidden.shape[-1]).cpu()
            )

            def save_error(gradient: torch.Tensor) -> None:
                self.errors[layer].append(
                    gradient.detach()
                    .float()
                    .reshape(-1, gradient.shape[-1])
                    .cpu()
                )

            output.register_hook(save_error)

        return hook

    def tensors(self) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        if set(self.hidden) != self.layers or set(self.errors) != self.layers:
            raise RuntimeError("c_proj activation/error collection is incomplete")
        hidden = {
            layer: torch.cat(self.hidden[layer], dim=0)
            for layer in sorted(self.layers)
        }
        errors = {
            layer: torch.cat(self.errors[layer], dim=0)
            for layer in sorted(self.layers)
        }
        for layer in self.layers:
            if hidden[layer].shape[0] != errors[layer].shape[0]:
                raise RuntimeError("activation and output-error rows disagree")
        return hidden, errors

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def collect_geometry(
    model: torch.nn.Module,
    batches: list[torch.Tensor],
    layers: list[int],
    device: str,
) -> tuple[
    float,
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
]:
    """Return mean CE, post-GELU rows, c_proj output errors, and gradients."""
    model.eval()
    model.zero_grad(set_to_none=True)
    collector = CProjGeometryCollector(model, layers)
    losses: list[float] = []
    try:
        for tokens in batches:
            tokens = tokens.to(device)
            inputs = tokens[:, :-1].contiguous()
            targets = tokens[:, 1:].contiguous()
            _logits, loss = model(inputs, targets)
            if loss is None:
                raise RuntimeError("model did not return a loss")
            losses.append(float(loss.detach()))
            (loss / len(batches)).backward()
        hidden, errors = collector.tensors()
        gradients = {}
        for layer in layers:
            parameter = model.transformer.h[layer].mlp.c_proj.weight
            if parameter.grad is None:
                raise RuntimeError(f"missing c_proj gradient for layer {layer}")
            gradients[layer] = parameter.grad.detach().float().cpu().clone()
        return sum(losses) / len(losses), hidden, errors, gradients
    finally:
        collector.close()
        model.zero_grad(set_to_none=True)


def normalized_output_error_scale(
    errors: torch.Tensor,
    *,
    minimum: float,
    maximum: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return bounded unit-RMS per-output error scale."""
    if errors.ndim != 2 or errors.shape[0] == 0:
        raise ValueError("output errors must be a nonempty matrix")
    raw = errors.float().square().mean(dim=0).sqrt()
    denominator = raw.square().mean().sqrt().clamp_min(1e-30)
    normalized = raw / denominator
    bounded = normalized.clamp(min=minimum, max=maximum)
    clamped = (bounded != normalized).float()
    return bounded, {
        "raw_minimum": float(raw.min()),
        "raw_maximum": float(raw.max()),
        "normalized_minimum": float(normalized.min()),
        "normalized_maximum": float(normalized.max()),
        "bounded_minimum": float(bounded.min()),
        "bounded_maximum": float(bounded.max()),
        "clamp_fraction": float(clamped.mean()),
    }


def fit_metric_selected_raw_angle_pass(
    source: torch.Tensor,
    residual: torch.Tensor,
    *,
    hidden: torch.Tensor | None,
    output_scale: torch.Tensor | None,
    stages: int,
    neighbors: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Select pairs in a functional metric, then fit raw Frobenius angles."""
    if source.ndim != 2 or residual.shape != source.shape:
        raise ValueError("source and residual must be equal-shape matrices")
    projected_source = source.float()
    projected_residual = residual.float()
    if hidden is not None:
        if hidden.ndim != 2 or hidden.shape[1] != source.shape[0]:
            raise ValueError("hidden/source shapes disagree")
        projected_source = hidden.to(source.device).float() @ projected_source
        projected_residual = hidden.to(source.device).float() @ projected_residual
    if output_scale is not None:
        if output_scale.ndim != 1 or output_scale.numel() != source.shape[1]:
            raise ValueError("output scale width disagrees with source")
        scale = output_scale.to(source.device).float().reshape(1, -1)
        projected_source = projected_source * scale
        projected_residual = projected_residual * scale
    permutations, matching = fast_muon_matched_permutations(
        projected_source,
        projected_residual,
        stages=stages,
        neighbors=neighbors,
        seed=seed,
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
        "stages": stages,
        "coordinates": int(stages * source.shape[1] // 2),
        "metric_rows": None if hidden is None else int(hidden.shape[0]),
        "output_error_scaled": output_scale is not None,
        "maximum_abs_angle": float(angles.abs().max()),
        "mean_abs_angle": float(angles.abs().mean()),
        "permutations": permutations.detach().cpu(),
        "angles": angles.detach().cpu(),
    }


def aggregate_results(
    rows: list[dict[str, Any]],
    finite_rows: list[dict[str, Any]],
    scale_rows: list[dict[str, Any]],
    decision_rule: dict[str, Any],
) -> dict[str, Any]:
    control = "frobenius_output32"
    by_variant: dict[str, Any] = {}
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        by_variant[variant] = {
            "task_predicted_ce_decrease": {
                window: sum(
                    float(row["task_predicted_ce_decrease"])
                    for row in selected
                    if row["window"] == window
                )
                for window in WINDOWS
            },
            "activation_residual_energy": {
                window: sum(
                    float(row["activation_residual_energy"])
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

    comparisons: dict[str, list[dict[str, Any]]] = {
        variant: [] for variant in SMALLEST_PASS_ORDER
    }
    phases = sorted({int(row["phase_start"]) for row in finite_rows})
    for phase in phases:
        for window in WINDOWS:
            indexed = {
                row["variant"]: float(row["loss"])
                for row in finite_rows
                if int(row["phase_start"]) == phase and row["window"] == window
                and row["variant"] in VARIANTS
            }
            if set(indexed) != set(VARIANTS):
                raise ValueError("finite-step inventory is incomplete")
            for variant in SMALLEST_PASS_ORDER:
                comparisons[variant].append(
                    {
                        "phase_start": phase,
                        "window": window,
                        "control_loss": indexed[control],
                        "candidate_loss": indexed[variant],
                        "gain": indexed[control] - indexed[variant],
                    }
                )

    requirements = decision_rule["candidate_requirements"]
    scale_clamp_fraction = (
        sum(float(row["clamp_fraction"]) for row in scale_rows)
        / max(len(scale_rows), 1)
    )
    candidates: dict[str, Any] = {}
    selected_variant: str | None = None
    for variant in SMALLEST_PASS_ORDER:
        cells = comparisons[variant]
        gains = [float(cell["gain"]) for cell in cells]
        holdout_gains = [
            float(cell["gain"]) for cell in cells if cell["window"] == "holdout"
        ]
        fit_task_advantage = (
            by_variant[variant]["task_predicted_ce_decrease"]["fit"]
            - by_variant[control]["task_predicted_ce_decrease"]["fit"]
        )
        holdout_task_advantage = (
            by_variant[variant]["task_predicted_ce_decrease"]["holdout"]
            - by_variant[control]["task_predicted_ce_decrease"]["holdout"]
        )
        retention = (
            holdout_task_advantage / fit_task_advantage
            if fit_task_advantage > 0.0
            else None
        )
        residual_ratio = (
            by_variant[variant]["activation_residual_energy"]["holdout"]
            / max(
                by_variant[control]["activation_residual_energy"]["holdout"],
                1e-30,
            )
        )
        update_ratio = by_variant[variant]["update_energy"] / max(
            by_variant[control]["update_energy"], 1e-30
        )
        candidate_scale_clamp = (
            scale_clamp_fraction
            if variant in (
                "error_selector_output32",
                "diagonal_kfac_selector_output32",
            )
            else 0.0
        )
        gate = {
            "mean_finite_step_ce_gain": (
                sum(gains) / len(gains)
                >= float(
                    requirements[
                        "mean_finite_step_ce_gain_over_frobenius_minimum"
                    ]
                )
            ),
            "finite_step_wins": sum(gain > 0.0 for gain in gains)
            >= int(requirements["finite_step_wins_minimum"]),
            "holdout_wins": sum(gain > 0.0 for gain in holdout_gains)
            >= int(requirements["holdout_wins_minimum"]),
            "minimum_holdout_gain": min(holdout_gains)
            >= float(requirements["minimum_holdout_finite_step_ce_gain"]),
            "holdout_predicted_descent": (
                by_variant[variant]["task_predicted_ce_decrease"]["holdout"]
                >= by_variant[control]["task_predicted_ce_decrease"]["holdout"]
            ),
            "holdout_activation_residual": residual_ratio
            <= float(requirements["holdout_activation_residual_energy_ratio_maximum"]),
            "update_energy": update_ratio
            <= float(requirements["candidate_update_energy_ratio_maximum"]),
            "positive_fit_and_holdout_task_advantage": (
                fit_task_advantage > 0.0 and holdout_task_advantage > 0.0
            ),
            "task_advantage_retention": (
                retention is not None
                and retention
                >= float(
                    requirements[
                        "holdout_to_fit_task_advantage_retention_minimum"
                    ]
                )
            ),
            "output_error_scale_clamp_fraction": candidate_scale_clamp
            <= float(requirements["output_error_scale_clamp_fraction_maximum"]),
        }
        finite = all_finite(
            {
                "gains": gains,
                "fit_task_advantage": fit_task_advantage,
                "holdout_task_advantage": holdout_task_advantage,
                "retention": retention,
                "residual_ratio": residual_ratio,
                "update_ratio": update_ratio,
                "candidate_scale_clamp": candidate_scale_clamp,
            }
        )
        passed = finite and all(gate.values())
        candidates[variant] = {
            "finite": finite,
            "mean_finite_step_ce_gain": sum(gains) / len(gains),
            "finite_step_wins": sum(gain > 0.0 for gain in gains),
            "holdout_wins": sum(gain > 0.0 for gain in holdout_gains),
            "minimum_holdout_gain": min(holdout_gains),
            "fit_task_advantage": fit_task_advantage,
            "holdout_task_advantage": holdout_task_advantage,
            "holdout_to_fit_task_advantage_retention": retention,
            "holdout_activation_residual_energy_ratio": residual_ratio,
            "candidate_to_control_update_energy_ratio": update_ratio,
            "output_error_scale_clamp_fraction": candidate_scale_clamp,
            "gate": gate,
            "passed": passed,
            "comparisons": cells,
        }
        if selected_variant is None and passed:
            selected_variant = variant

    return {
        "all_outputs_and_metrics_finite": all_finite(
            {"rows": rows, "finite_rows": finite_rows, "scale_rows": scale_rows}
        ),
        "by_variant": by_variant,
        "output_error_scale_mean_clamp_fraction": scale_clamp_fraction,
        "candidates": candidates,
        "selected_variant": selected_variant,
        "passed": selected_variant is not None,
        "classification": (
            "PASS_DIAGONAL_KFAC_SELECTOR"
            if selected_variant is not None
            else "REJECT_DIAGONAL_KFAC_SELECTOR"
        ),
        "authorization": {
            "production_implementation_authorized": selected_variant is not None,
            "exact_config_mfu_preflight_authorized": selected_variant is not None,
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
    if file_sha256(args.acquisition_result) != plan["identity"][
        "acquisition_result_sha256"
    ]:
        raise ValueError("acquisition result SHA-256 mismatch")
    if acquisition.get("identity", {}).get("run_identity_sha256") != plan[
        "identity"
    ]["run_identity_sha256"]:
        raise ValueError("acquisition run identity mismatch")
    manifest = args.data_dir / "manifest.json"
    if file_sha256(manifest) != plan["identity"]["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest SHA-256 mismatch")
    if args.output.exists():
        raise FileExistsError(f"output directory already exists: {args.output}")
    args.output.mkdir(parents=True)

    analysis = plan["analysis"]
    layers = [int(value) for value in analysis["layers"]]
    phases = [[int(value) for value in pair] for pair in analysis["phases"]]
    chart = analysis["shared_chart"]
    run_identity = plan["identity"]["run_identity_sha256"]
    snapshot_paths = {
        step: args.snapshot_dir / f"step_{step:06d}.pt"
        for step in sorted({value for phase in phases for value in phase})
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

    windows = {}
    for name in WINDOWS:
        spec = analysis[f"{name}_window"]
        windows[name] = fixed_validation_batches(
            args.data_dir,
            int(spec["batch_size"]),
            int(spec["block_size"]) + 1,
            int(spec["batches"]),
            int(spec["seed"]),
        )

    rows: list[dict[str, Any]] = []
    finite_rows: list[dict[str, Any]] = []
    scale_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    scale_min, scale_max = [
        float(value) for value in analysis["output_error_scale"]["bounds"]
    ]
    for phase_index, (phase_start, phase_end) in enumerate(phases):
        phase_started = time.perf_counter()
        snapshot = load_snapshot(snapshot_paths[phase_start])
        if snapshot.get("run_identity_sha256") != run_identity:
            raise ValueError("snapshot run identity mismatch")
        model = model_from_snapshot(snapshot, args.device)
        baseline_losses = {}
        hidden = {}
        errors = {}
        gradients = {}
        for window in WINDOWS:
            (
                baseline_losses[window],
                hidden[window],
                errors[window],
                gradients[window],
            ) = collect_geometry(model, windows[window], layers, args.device)

        probe = load_probe(probe_paths[phase_start], phase_start, run_identity)
        updates: dict[str, dict[int, torch.Tensor]] = {
            variant: {} for variant in VARIANTS
        }
        for layer in layers:
            name = parameter_name(layer)
            state = probe["parameters"][name]
            hyper = probe["hyperparameters"][name]
            weight = state["weight_before_step"].to(args.device).float()
            torch.testing.assert_close(
                weight.cpu(), snapshot["parameters"][name].float(), rtol=0.0, atol=0.0
            )
            learning_rate = float(hyper["lr"])
            weight_decay = float(hyper["weight_decay"])
            applied_per_lr = state["applied_direction_per_lr"].to(args.device).float()
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
            output_residual = residual.T.contiguous()
            scale, scale_diagnostics = normalized_output_error_scale(
                errors["fit"][layer], minimum=scale_min, maximum=scale_max
            )
            scale_rows.append(
                {
                    "phase_start": phase_start,
                    "layer": layer,
                    **scale_diagnostics,
                }
            )
            fitted = {}
            diagnostics = {}
            fitted["frobenius_output32"], diagnostics["frobenius_output32"] = (
                fit_frobenius_pass(
                    source,
                    output_residual,
                    stages=int(chart["output_stages"]),
                    neighbors=int(chart["neighbors"]),
                    seed=seed + 2,
                )
            )
            for variant, use_hidden, use_error in (
                ("activation_selector_output32", True, False),
                ("error_selector_output32", False, True),
                ("diagonal_kfac_selector_output32", True, True),
            ):
                fitted[variant], diagnostics[variant] = (
                    fit_metric_selected_raw_angle_pass(
                        source,
                        output_residual,
                        hidden=hidden["fit"][layer] if use_hidden else None,
                        output_scale=scale if use_error else None,
                        stages=int(chart["output_stages"]),
                        neighbors=int(chart["neighbors"]),
                        seed=seed + 2,
                    )
                )
            decay = 1.0 - learning_rate * weight_decay
            for variant in VARIANTS:
                final_weight = fitted[variant].T.contiguous() * decay
                update = (final_weight - weight).detach().cpu()
                updates[variant][layer] = update
                weight_error = requested.cpu() - update
                for window in WINDOWS:
                    task = task_descent_metrics(gradients[window][layer], update)
                    rows.append(
                        {
                            "phase_start": phase_start,
                            "phase_end": phase_end,
                            "layer": layer,
                            "window": window,
                            "variant": variant,
                            "task_predicted_ce_decrease": task["predicted_ce_decrease"],
                            "activation_residual_energy": output_residual_energy(
                                hidden[window][layer], requested.cpu(), update
                            ),
                            "update_energy": float(update.double().square().sum()),
                            "weight_error_energy": float(
                                weight_error.double().square().sum()
                            ),
                            "coordinates_per_layer": int(
                                chart["coordinate_count_per_layer"]
                            ),
                        }
                    )
                total_coordinates = sum(
                    int(item["coordinates"]) for item in hidden_diagnostics
                ) + int(diagnostics[variant]["coordinates"])
                if total_coordinates != int(chart["coordinate_count_per_layer"]):
                    raise ValueError("chart coordinate budget mismatch")

        for window in WINDOWS:
            finite_rows.append(
                {
                    "phase_start": phase_start,
                    "phase_end": phase_end,
                    "window": window,
                    "variant": "baseline",
                    "loss": baseline_losses[window],
                }
            )
            for variant in VARIANTS:
                finite_rows.append(
                    {
                        "phase_start": phase_start,
                        "phase_end": phase_end,
                        "window": window,
                        "variant": variant,
                        "loss": evaluate_with_updates(
                            model, windows[window], updates[variant], args.device
                        ),
                    }
                )
        phase_rows.append(
            {
                "phase_start": phase_start,
                "phase_end": phase_end,
                "seconds": time.perf_counter() - phase_started,
            }
        )
        del model
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

    aggregate = aggregate_results(
        rows, finite_rows, scale_rows, plan["decision_rule"]
    )
    aggregate = apply_plan_authorization(
        aggregate, str(plan.get("schema_version"))
    )
    result = {
        "schema_version": (
            CALIBRATION_RESULT_SCHEMA
            if plan.get("schema_version") == CALIBRATION_PLAN_SCHEMA
            else SCHEMA_VERSION
        ),
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "classification": aggregate["classification"],
        "execution": {
            "host": "PRO6",
            "device": args.device,
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": "examples.nanogpt.analyze_mlp_cproj_diagonal_kfac_selector",
            "parameter_updates": 0,
            "direct_foreground_polling": True,
            "watchdog": False,
            "callback": False,
            "started_at": started_at,
            "elapsed_seconds": time.time() - started,
        },
        "identity": {
            "plan_path": str(args.plan),
            "plan_sha256": file_sha256(args.plan),
            "acquisition_result_path": str(args.acquisition_result),
            "acquisition_result_sha256": file_sha256(args.acquisition_result),
            "dataset_manifest_sha256": file_sha256(manifest),
            "run_identity_sha256": run_identity,
        },
        "aggregate": aggregate,
        "phase_timing": phase_rows,
    }
    write_csv(args.output / "diagonal_kfac_selector_cells.csv", rows)
    write_csv(args.output / "diagonal_kfac_selector_finite_ce.csv", finite_rows)
    write_csv(args.output / "diagonal_kfac_selector_scales.csv", scale_rows)
    result_path = args.output / "diagonal_kfac_selector_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
