#!/usr/bin/env python3
"""Measure whether structured-Muon state transports the critical c_proj residual.

This is a zero-update, same-gauge analysis.  It reconstructs the dense Muon
request, decayed error-feedback target, realized structured-chart update, and
unrepresented residual from phase-aligned probes.  Each component is scored
against the terminal exact-minus-scheduled residual and the next phase chord
in raw weight space and under the frozen terminal post-GELU metric.
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

from examples.nanogpt.analyze_mlp_cproj_activation_weighted_output_selector import (
    file_sha256,
    git_commit,
)
from examples.nanogpt.analyze_mlp_cproj_coadapted_orbit_geometry import (
    buffer_name,
    load_snapshot,
)
from examples.nanogpt.analyze_mlp_cproj_modulation_residual_oracle import (
    fit_output_additive,
)
from examples.nanogpt.analyze_mlp_cproj_multiscale_path import (
    cumulative_lr_coordinate,
    polynomial_predict,
)
from examples.nanogpt.analyze_mlp_cproj_polynomial_oracle_ce import restore_radius
from examples.nanogpt.analyze_mlp_cproj_predictive_manifold import (
    fit_through_origin_basis,
    terminal_post_gelu_activations,
)
from examples.nanogpt.analyze_parameter_trajectory import write_csv
from examples.nanogpt.muon import zeropower_via_newtonschulz5


PLAN_SCHEMA = "mai_124m_mlp_cproj_optimizer_state_transport_plan_v1"
RESULT_SCHEMA = "mai_124m_mlp_cproj_optimizer_state_transport_result_v1"
LAYERS = tuple(range(8, 12))
PROBE_STEPS = (98, 296, 593, 890, 1187, 1484, 1781, 2078, 2372)
REFERENCE_POST_STEPS = (99, 297, 594, 891, 1188, 1485, 1782, 2079, 2373)
DISCOVERY_STEPS = (
    0, 99, 198, 297, 396, 495, 594, 693, 792, 891,
    990, 1089, 1188, 1287, 1386, 1485, 1584, 1683, 1782,
)
HELDOUT_PROBE_STEPS = (1781, 2078, 2372)
COMPONENTS = ("requested", "feedback", "corrected", "realized", "unrepresented")
TERMINAL_STEP = 2373


def metric(target: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    target = target.double().reshape(-1)
    candidate = candidate.double().reshape(-1)
    target_energy = target.square().sum().clamp_min(1e-30)
    candidate_energy = candidate.square().sum().clamp_min(1e-30)
    dot = (target * candidate).sum()
    cosine = dot / (target_energy.sqrt() * candidate_energy.sqrt())
    return {
        "target_energy": float(target_energy),
        "candidate_energy": float(candidate_energy),
        "cosine": float(cosine),
        "positive_line_recovery": float(
            dot.clamp_min(0).square() / (target_energy * candidate_energy)
        ),
        "fixed_scale_recovery": float(
            1.0 - (target - candidate).square().sum() / target_energy
        ),
    }


def functional_metric(
    target: torch.Tensor, candidate: torch.Tensor, hidden: torch.Tensor
) -> dict[str, float]:
    return metric(hidden.float() @ target.float().T, hidden.float() @ candidate.float().T)


def output_additive_projection(
    component: torch.Tensor, hidden: torch.Tensor
) -> torch.Tensor:
    target_output = hidden.float() @ component.float().T
    return fit_output_additive(
        hidden.float(), target_output, component.float()
    )


def reconstruct_components(
    state: dict[str, torch.Tensor], hyper: dict[str, Any]
) -> tuple[dict[str, torch.Tensor], float]:
    weight = state["weight_before_step"].float()
    combined = state["combined_momentum_update"].float()
    polar = zeropower_via_newtonschulz5(
        combined, steps=int(hyper["ns_steps"])
    ).float()
    direction = -float(hyper["polar_scale"]) * polar
    learning_rate = float(hyper["lr"])
    requested = learning_rate * (
        direction - float(hyper["weight_decay"]) * weight
    )
    feedback = (
        float(hyper["error_feedback_decay"])
        * state["compression_residual_before_step"].float()
    )
    corrected = requested + feedback
    realized = (
        state["weight_after_step"].float()
        - state["weight_before_step"].float()
    )
    unrepresented = corrected - realized
    observed = state["compression_residual_after_step"].float()
    relative_error = float(
        (unrepresented - observed).double().norm()
        / observed.double().norm().clamp_min(1e-30)
    )
    return {
        "requested": requested,
        "feedback": feedback,
        "corrected": corrected,
        "realized": realized,
        "unrepresented": unrepresented,
    }, relative_error


def weighted(rows: list[dict[str, Any]], field: str, energy: str) -> float:
    denominator = sum(float(row[energy]) for row in rows)
    return sum(float(row[field]) * float(row[energy]) for row in rows) / max(
        denominator, 1e-30
    )


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("optimizer-state transport plan schema mismatch")
    expected = {
        "parameter_updates": 0,
        "layers": list(LAYERS),
        "probe_steps": list(PROBE_STEPS),
        "reference_post_steps": list(REFERENCE_POST_STEPS),
        "discovery_steps": list(DISCOVERY_STEPS),
        "terminal_step": TERMINAL_STEP,
        "polynomial_rank": 4,
        "polynomial_degree": 2,
        "activation_rows": 2048,
        "components": list(COMPONENTS),
        "heldout_probe_steps": list(HELDOUT_PROBE_STEPS),
        "output_additive_projection": True,
    }
    for key, value in expected.items():
        if plan.get("analysis", {}).get(key) != value:
            raise ValueError(f"optimizer-state analysis field changed: {key}")
    thresholds = plan.get("decision_rule", {}).get("thresholds", {})
    if thresholds != {
        "compression_reconstruction_max_relative_error": 1e-4,
        "causal_heldout_functional_line_recovery_minimum": 0.80,
    }:
        raise ValueError("optimizer-state thresholds changed")
    authorization = plan.get("authorization", {})
    if authorization.get("run_zero_update_state_transport_analysis") is not True:
        raise ValueError("state-transport analysis is not authorized")
    for key in (
        "implement_candidate_structure",
        "run_exact_config_mfu",
        "run_language_model_training",
        "larger_rung",
    ):
        if authorization.get(key) is not False:
            raise ValueError(f"plan must keep {key} false")


def scheduled_terminal_residual(
    snapshots: dict[int, dict[str, Any]],
    config: dict[str, Any],
    layer: int,
    device: str,
) -> torch.Tensor:
    steps = (*DISCOVERY_STEPS, TERMINAL_STEP)
    weights = {
        step: snapshots[step]["buffers"][buffer_name(layer, "weight")]
        .float()
        .to(device)
        for step in steps
    }
    initial_norm = weights[0].norm().clamp_min(1e-30)
    normalized = {
        step: weight * (initial_norm / weight.norm().clamp_min(1e-30))
        for step, weight in weights.items()
    }
    discovery = torch.stack(
        [
            (normalized[step] - normalized[0]).reshape(-1)
            for step in DISCOVERY_STEPS
        ]
    )
    basis = fit_through_origin_basis(discovery[1:], 4)
    coordinates = discovery @ basis.T
    discovery_progress = torch.tensor(
        [cumulative_lr_coordinate(step, config) for step in DISCOVERY_STEPS],
        device=device,
    )
    terminal_progress = torch.tensor(
        [cumulative_lr_coordinate(TERMINAL_STEP, config)], device=device
    )
    prediction = polynomial_predict(
        coordinates, discovery_progress, terminal_progress, 2
    )
    predicted = normalized[0] + (prediction @ basis).reshape_as(normalized[0])
    predicted = restore_radius(predicted, weights[TERMINAL_STEP].norm())
    return weights[TERMINAL_STEP] - predicted


def classify(
    aggregate: dict[str, dict[str, float]], reconstruction_valid: bool, threshold: float
) -> tuple[str, str | None]:
    if not reconstruction_valid:
        return "INVALID_OPTIMIZER_STATE_RECONSTRUCTION", None
    best = max(
        COMPONENTS,
        key=lambda name: aggregate[name][
            "heldout_terminal_functional_positive_line_recovery"
        ],
    )
    if (
        aggregate[best]["heldout_terminal_functional_positive_line_recovery"]
        >= threshold
    ):
        return "CAUSAL_OPTIMIZER_STATE_TRANSPORT_SUFFICIENT", best
    return "OPTIMIZER_STATE_TRANSPORT_INSUFFICIENT", best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--acquisition-result", type=Path, required=True)
    parser.add_argument("--trajectory-verification", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"output already exists: {args.output_dir}")
    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    identity = plan["identity"]
    pinned = {
        Path(__file__): identity["analyzer_sha256"],
        args.acquisition_result: identity["acquisition_result_sha256"],
        args.trajectory_verification: identity["trajectory_verification_sha256"],
        args.config: identity["config_sha256"],
    }
    for path, expected in pinned.items():
        if file_sha256(path) != expected:
            raise ValueError(f"pinned artifact SHA-256 mismatch: {path}")
    for relative, expected in identity["supporting_source_sha256"].items():
        if file_sha256(REPO_ROOT / relative) != expected:
            raise ValueError(f"supporting source changed: {relative}")
    acquisition = json.loads(args.acquisition_result.read_text())
    if acquisition.get("classification") != (
        "ACCEPTED_COADAPTED_LATE_CPROJ_OPTIMIZER_STATE_TRAJECTORY"
    ):
        raise ValueError("optimizer-state acquisition is not accepted")
    verification = json.loads(args.trajectory_verification.read_text())
    if verification.get("classification") != (
        "ACCEPTED_COADAPTED_LATE_CPROJ_FULL_STATE_TRAJECTORY"
    ):
        raise ValueError("full-state trajectory is not accepted")
    config = json.loads(args.config.read_text())
    old_hashes = verification["inventory"]["snapshot_sha256_by_step"]
    old_identity = verification["identity"]["run_identity_sha256"]
    required_snapshot_steps = sorted(
        set(DISCOVERY_STEPS) | set(REFERENCE_POST_STEPS) | {TERMINAL_STEP}
    )
    snapshots = {
        step: load_snapshot(
            args.snapshot_dir / f"step_{step:06d}.pt",
            old_hashes[str(step)],
            old_identity,
        )
        for step in required_snapshot_steps
    }
    from examples.nanogpt.train import fixed_eval_indices_digest, make_fixed_eval_indices

    fixed = make_fixed_eval_indices(
        Path(config["data_dir"]), int(config["eval_batch_size"]),
        int(config["block_size"]), int(config["eval_iters"]),
        int(config["eval_seed"]),
    )
    if fixed_eval_indices_digest(fixed) != identity["fixed_eval_indices_sha256"]:
        raise ValueError("fixed evaluation indices changed")
    activations = terminal_post_gelu_activations(
        snapshots[TERMINAL_STEP], config, fixed,
        int(plan["analysis"]["activation_rows"]), args.device,
    )
    terminal_residuals = {
        layer: scheduled_terminal_residual(snapshots, config, layer, args.device)
        for layer in LAYERS
    }
    probe_hashes = acquisition["inventory"]["probe_sha256_by_step"]
    probe_identity = acquisition["identity"]["run_identity_sha256"]
    rows: list[dict[str, Any]] = []
    max_reconstruction_error = 0.0
    started = time.time()
    for index, probe_step in enumerate(PROBE_STEPS):
        path = args.probe_dir / f"step_{probe_step:06d}.pt"
        if file_sha256(path) != probe_hashes[str(probe_step)]:
            raise ValueError(f"probe SHA-256 mismatch at step {probe_step}")
        probe = torch.load(path, map_location="cpu", weights_only=False)
        if probe["run_identity_sha256"] != probe_identity:
            raise ValueError("optimizer probe run identity changed")
        for layer in LAYERS:
            name = buffer_name(layer, "weight")
            state = {key: value.to(args.device) for key, value in probe["parameters"][name].items()}
            hyper = probe["hyperparameters"][name]
            components, reconstruction_error = reconstruct_components(state, hyper)
            max_reconstruction_error = max(max_reconstruction_error, reconstruction_error)
            hidden = activations[layer].to(args.device)
            terminal_target = terminal_residuals[layer]
            future_target = None
            if index + 1 < len(REFERENCE_POST_STEPS):
                next_step = REFERENCE_POST_STEPS[index + 1]
                future_target = (
                    snapshots[next_step]["buffers"][name].float().to(args.device)
                    - state["weight_before_step"].float()
                )
            for component_name, component in components.items():
                raw_terminal = metric(terminal_target, component)
                functional_terminal = functional_metric(terminal_target, component, hidden)
                projected = output_additive_projection(component, hidden)
                component_function = functional_metric(component, projected, hidden)
                row: dict[str, Any] = {
                    "probe_step": probe_step,
                    "layer": layer,
                    "component": component_name,
                    "heldout": probe_step in HELDOUT_PROBE_STEPS,
                    "compression_reconstruction_relative_error": reconstruction_error,
                    "terminal_raw_target_energy": raw_terminal["target_energy"],
                    "terminal_raw_cosine": raw_terminal["cosine"],
                    "terminal_raw_positive_line_recovery": raw_terminal["positive_line_recovery"],
                    "terminal_raw_fixed_scale_recovery": raw_terminal["fixed_scale_recovery"],
                    "terminal_functional_target_energy": functional_terminal["target_energy"],
                    "terminal_functional_cosine": functional_terminal["cosine"],
                    "terminal_functional_positive_line_recovery": functional_terminal["positive_line_recovery"],
                    "terminal_functional_fixed_scale_recovery": functional_terminal["fixed_scale_recovery"],
                    "component_output_additive_functional_recovery": component_function["fixed_scale_recovery"],
                }
                if future_target is not None:
                    raw_future = metric(future_target, component)
                    functional_future = functional_metric(future_target, component, hidden)
                    row.update(
                        {
                            "future_reference_step": REFERENCE_POST_STEPS[index + 1],
                            "future_raw_target_energy": raw_future["target_energy"],
                            "future_raw_positive_line_recovery": raw_future["positive_line_recovery"],
                            "future_functional_target_energy": functional_future["target_energy"],
                            "future_functional_positive_line_recovery": functional_future["positive_line_recovery"],
                        }
                    )
                rows.append(row)
        del probe
    aggregate: dict[str, dict[str, float]] = {}
    for name in COMPONENTS:
        component_rows = [row for row in rows if row["component"] == name]
        heldout = [row for row in component_rows if row["heldout"]]
        heldout_future = [row for row in heldout if "future_reference_step" in row]
        aggregate[name] = {
            "heldout_terminal_raw_positive_line_recovery": weighted(
                heldout, "terminal_raw_positive_line_recovery", "terminal_raw_target_energy"
            ),
            "heldout_terminal_functional_positive_line_recovery": weighted(
                heldout, "terminal_functional_positive_line_recovery", "terminal_functional_target_energy"
            ),
            "heldout_future_raw_positive_line_recovery": weighted(
                heldout_future, "future_raw_positive_line_recovery", "future_raw_target_energy"
            ),
            "heldout_future_functional_positive_line_recovery": weighted(
                heldout_future, "future_functional_positive_line_recovery", "future_functional_target_energy"
            ),
            "mean_component_output_additive_functional_recovery": sum(
                float(row["component_output_additive_functional_recovery"])
                for row in component_rows
            ) / len(component_rows),
        }
    threshold = float(plan["decision_rule"]["thresholds"]["causal_heldout_functional_line_recovery_minimum"])
    reconstruction_valid = max_reconstruction_error <= float(
        plan["decision_rule"]["thresholds"]["compression_reconstruction_max_relative_error"]
    )
    classification, best = classify(aggregate, reconstruction_valid, threshold)
    args.output_dir.mkdir(parents=True)
    rows_path = args.output_dir / "optimizer_state_transport_rows.csv"
    result_path = args.output_dir / "optimizer_state_transport_result.json"
    write_csv(rows_path, rows)
    result = {
        "schema_version": RESULT_SCHEMA,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "classification": classification,
        "execution": {
            "host": "PRO6", "device": args.device,
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": "examples.nanogpt.analyze_mlp_cproj_optimizer_state_transport",
            "parameter_updates": 0,
            "elapsed_seconds": time.time() - started,
        },
        "identity": {
            "plan_sha256": file_sha256(args.plan),
            "analyzer_sha256": file_sha256(Path(__file__)),
            "acquisition_result_sha256": file_sha256(args.acquisition_result),
            "trajectory_verification_sha256": file_sha256(args.trajectory_verification),
            "config_sha256": file_sha256(args.config),
            "optimizer_probe_run_identity_sha256": probe_identity,
            "full_state_run_identity_sha256": old_identity,
        },
        "mechanical_reconstruction": {
            "passed": reconstruction_valid,
            "maximum_relative_error": max_reconstruction_error,
        },
        "aggregate": aggregate,
        "best_heldout_component": best,
        "authorization": {
            "compact_state_conditioned_mapper": classification == "CAUSAL_OPTIMIZER_STATE_TRANSPORT_SUFFICIENT",
            "implement_candidate_structure": False,
            "run_exact_config_mfu": False,
            "run_language_model_training": False,
        },
        "artifacts": {
            "rows": str(rows_path),
            "rows_sha256": file_sha256(rows_path),
        },
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, sort_keys=True))
    if not reconstruction_valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
