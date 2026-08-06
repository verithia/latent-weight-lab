#!/usr/bin/env python3
"""Gauge-invariant analysis of the accepted co-adapted late-c_proj path.

The v1 comparison correctly failed because the procedural and dense-parent
runs did not share bitwise c_proj initialization.  This repair compares only
within-run quantities: scaled-right-orbit recovery, normalized output-Gram
deformation, task-selected edge transport, and descriptive temporal PCA.
No raw cross-run parameter displacement is computed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_mlp_cproj_activation_weighted_output_selector import (
    file_sha256,
)
from examples.nanogpt.analyze_mlp_cproj_coadapted_orbit_geometry import (
    buffer_name,
    git_commit,
    load_snapshot,
    mean,
    parameter_name,
    scaled_right_orbit_metrics,
    support_metrics,
    weighted_mean,
)
from examples.nanogpt.analyze_parameter_trajectory import (
    summarize_parameter,
    write_csv,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = (
    REPO_ROOT
    / "examples/nanogpt/analyze_mlp_cproj_coadapted_orbit_geometry.py"
)
PLAN_SCHEMA = "mai_124m_mlp_cproj_coadapted_orbit_geometry_v2_plan_v1"
RESULT_SCHEMA = "mai_124m_mlp_cproj_coadapted_orbit_geometry_v2_result_v1"
LATE_LAYERS = tuple(range(8, 12))
EARLY_LAYERS = tuple(range(8))
ALL_LAYERS = tuple(range(12))
PHASE_STEPS = (0, 594, 1188, 1782, 2373)


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unexpected gauge-invariant orbit plan schema")
    expected = {
        "procedural_minimum_orbit_recovery": 0.995,
        "late_minus_early_orbit_recovery": 0.05,
        "late_to_early_gram_drift_ratio": 0.8,
        "reusable_support_retention_fraction": 0.10,
        "reusable_support_enrichment": 2.0,
    }
    if plan.get("decision_rule", {}).get("thresholds") != expected:
        raise ValueError("gauge-invariant orbit thresholds changed")
    analysis = plan.get("analysis", {})
    if analysis.get("parameter_updates") != 0:
        raise ValueError("orbit geometry must perform zero parameter updates")
    if analysis.get("late_layers") != list(LATE_LAYERS):
        raise ValueError("late-layer set changed")
    if analysis.get("phase_steps") != list(PHASE_STEPS):
        raise ValueError("phase steps changed")
    if analysis.get("cross_run_parameter_displacements") is not False:
        raise ValueError("v2 must prohibit raw cross-run displacements")
    authorization = plan.get("authorization", {})
    if authorization.get("run_gauge_invariant_zero_update_v2") is not True:
        raise ValueError("gauge-invariant zero-update analysis is not authorized")
    for key in (
        "implement_candidate_structure",
        "run_exact_config_mfu",
        "run_language_model_training",
        "larger_rung",
    ):
        if authorization.get(key) is not False:
            raise ValueError(f"plan must keep {key} false")


def classify(gates: dict[str, bool]) -> str:
    if not gates["accepted_path_is_scaled_right_orbit"]:
        return "ACCEPTED_PATH_EXCEEDS_SCALED_RIGHT_ORBIT"
    if gates["right_orbit_localizes_late_band"] and gates[
        "support_is_reusable"
    ]:
        return "RIGHT_ORBIT_LOCALIZES_LWT_WITH_REUSABLE_SUPPORT"
    if gates["right_orbit_localizes_late_band"]:
        return "RIGHT_ORBIT_LOCALIZES_LWT_BUT_SUPPORT_IS_MOVING"
    if gates["support_is_reusable"]:
        return "RIGHT_ORBIT_NOT_LAYER_LOCALIZING_BUT_SUPPORT_IS_REUSABLE"
    return "ADAPTIVE_RIGHT_ORBIT_WITH_MOVING_SUPPORT_NOT_LAYER_LOCALIZED"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--trajectory-verification", type=Path, required=True)
    parser.add_argument("--trajectory-snapshot-dir", type=Path, required=True)
    parser.add_argument("--parent-acquisition", type=Path, required=True)
    parser.add_argument("--parent-snapshot-dir", type=Path, required=True)
    parser.add_argument("--invalid-v1-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists() or args.rows.exists():
        raise FileExistsError("gauge-invariant orbit output already exists")

    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    identity = plan["identity"]
    pinned = {
        args.trajectory_verification: identity[
            "trajectory_verification_sha256"
        ],
        args.parent_acquisition: identity["parent_acquisition_sha256"],
        args.invalid_v1_audit: identity["invalid_v1_audit_sha256"],
        Path(__file__): identity["analyzer_sha256"],
        HELPER_PATH: identity["helper_sha256"],
    }
    for path, expected_hash in pinned.items():
        if file_sha256(path) != expected_hash:
            raise ValueError(f"pinned artifact SHA-256 mismatch: {path}")

    trajectory = json.loads(args.trajectory_verification.read_text())
    parent = json.loads(args.parent_acquisition.read_text())
    invalid_v1 = json.loads(args.invalid_v1_audit.read_text())
    if trajectory.get("classification") != (
        "ACCEPTED_COADAPTED_LATE_CPROJ_FULL_STATE_TRAJECTORY"
    ):
        raise ValueError("co-adapted trajectory is not accepted")
    if parent.get("classification") != (
        "ACCEPTED_PARENT_EQUIVALENT_EXACT_FUNCTIONAL_REPLAY"
    ):
        raise ValueError("dense parent acquisition is not accepted")
    if invalid_v1.get("classification") != "INVALID_CROSS_RUN_GAUGE":
        raise ValueError("v1 cross-run gauge failure is not sealed")

    trajectory_identity = trajectory["identity"]["run_identity_sha256"]
    parent_identity = parent["identity"]["run_identity_sha256"]
    trajectory_hashes = trajectory["inventory"]["snapshot_sha256_by_step"]
    parent_hashes = parent["artifacts"]["snapshot_sha256_by_step"]
    trajectory_steps = sorted(int(value) for value in trajectory_hashes)
    if trajectory_steps != plan["analysis"]["trajectory_steps"]:
        raise ValueError("trajectory step inventory changed")
    if sorted(int(value) for value in parent_hashes) != list(PHASE_STEPS):
        raise ValueError("parent phase inventory changed")

    trajectory_weights: dict[int, dict[int, torch.Tensor]] = {}
    trajectory_supports: dict[int, dict[int, torch.Tensor]] = {}
    trajectory_refresh: dict[int, dict[int, tuple[int, int, bool]]] = {}
    for step in trajectory_steps:
        snapshot = load_snapshot(
            args.trajectory_snapshot_dir / f"step_{step:06d}.pt",
            trajectory_hashes[str(step)],
            trajectory_identity,
        )
        trajectory_weights[step] = {
            layer: snapshot["buffers"][buffer_name(layer, "weight")]
            .float()
            .clone()
            for layer in LATE_LAYERS
        }
        trajectory_supports[step] = {}
        trajectory_refresh[step] = {}
        for layer in LATE_LAYERS:
            primary = snapshot["buffers"][
                buffer_name(layer, "selected_permutations")
            ]
            residual_key = buffer_name(
                layer, "residual_selected_permutations"
            )
            support = primary
            if residual_key in snapshot["buffers"]:
                support = torch.cat(
                    (support, snapshot["buffers"][residual_key]), dim=0
                )
            trajectory_supports[step][layer] = support.clone()
            trajectory_refresh[step][layer] = (
                int(snapshot["buffers"][buffer_name(layer, "refresh_count")]),
                int(snapshot["buffers"][buffer_name(layer, "last_refresh_step")]),
                bool(snapshot["buffers"][buffer_name(layer, "matching_valid")]),
            )
        del snapshot

    parent_weights: dict[int, dict[int, torch.Tensor]] = {}
    for step in PHASE_STEPS:
        snapshot = load_snapshot(
            args.parent_snapshot_dir / f"step_{step:06d}.pt",
            parent_hashes[str(step)],
            parent_identity,
        )
        parent_weights[step] = {
            layer: snapshot["parameters"][parameter_name(layer)]
            .float()
            .clone()
            for layer in ALL_LAYERS
        }
        del snapshot

    started = time.time()
    rows: list[dict[str, Any]] = []
    procedural_rows: list[dict[str, Any]] = []
    for layer in LATE_LAYERS:
        for start, end in zip(
            trajectory_steps[:-1], trajectory_steps[1:], strict=True
        ):
            procedural_rows.append(
                {
                    "row_type": "procedural_consecutive_orbit",
                    "layer": layer,
                    "start_step": start,
                    "end_step": end,
                    **scaled_right_orbit_metrics(
                        trajectory_weights[start][layer].to(args.device),
                        trajectory_weights[end][layer].to(args.device),
                    ),
                }
            )
    rows.extend(procedural_rows)

    parent_rows: list[dict[str, Any]] = []
    for layer in ALL_LAYERS:
        for start, end in zip(PHASE_STEPS[:-1], PHASE_STEPS[1:], strict=True):
            parent_rows.append(
                {
                    "row_type": "dense_parent_phase_orbit",
                    "band": "early" if layer in EARLY_LAYERS else "late",
                    "layer": layer,
                    "start_step": start,
                    "end_step": end,
                    **scaled_right_orbit_metrics(
                        parent_weights[start][layer].to(args.device),
                        parent_weights[end][layer].to(args.device),
                    ),
                }
            )
    rows.extend(parent_rows)

    support_rows: list[dict[str, Any]] = []
    for layer in LATE_LAYERS:
        for start, end in zip(
            trajectory_steps[1:-1], trajectory_steps[2:], strict=True
        ):
            left_refresh = trajectory_refresh[start][layer]
            right_refresh = trajectory_refresh[end][layer]
            if not left_refresh[2] or not right_refresh[2]:
                continue
            support_rows.append(
                {
                    "row_type": "consecutive_support",
                    "layer": layer,
                    "start_step": start,
                    "end_step": end,
                    "start_refresh_count": left_refresh[0],
                    "end_refresh_count": right_refresh[0],
                    "start_last_refresh_step": left_refresh[1],
                    "end_last_refresh_step": right_refresh[1],
                    **support_metrics(
                        trajectory_supports[start][layer],
                        trajectory_supports[end][layer],
                    ),
                }
            )
    rows.extend(support_rows)

    pca_rows: list[dict[str, Any]] = []
    for layer in LATE_LAYERS:
        name = parameter_name(layer)
        for row_type, steps, tensors in (
            (
                "procedural_fine_temporal_pca",
                trajectory_steps,
                [trajectory_weights[step][layer] for step in trajectory_steps],
            ),
            (
                "procedural_phase_temporal_pca",
                list(PHASE_STEPS),
                [trajectory_weights[step][layer] for step in PHASE_STEPS],
            ),
            (
                "dense_parent_phase_temporal_pca",
                list(PHASE_STEPS),
                [parent_weights[step][layer] for step in PHASE_STEPS],
            ),
        ):
            summary, _coordinates, _polynomials = summarize_parameter(
                name=name,
                steps=steps,
                tensors=tensors,
                device=args.device,
            )
            pca_rows.append({"row_type": row_type, **summary})
    rows.extend(pca_rows)

    thresholds = plan["decision_rule"]["thresholds"]
    procedural_minimum_recovery = min(
        float(row["orbit_recovery"]) for row in procedural_rows
    )
    procedural_maximum_gram_drift = max(
        float(row["normalized_output_gram_drift"])
        for row in procedural_rows
    )
    parent_early = [row for row in parent_rows if row["band"] == "early"]
    parent_late = [row for row in parent_rows if row["band"] == "late"]
    early_recovery = weighted_mean(parent_early, "orbit_recovery")
    late_recovery = weighted_mean(parent_late, "orbit_recovery")
    early_gram_drift = weighted_mean(
        parent_early, "normalized_output_gram_drift"
    )
    late_gram_drift = weighted_mean(
        parent_late, "normalized_output_gram_drift"
    )
    support_retention = mean(support_rows, "retention_fraction")
    support_enrichment = mean(support_rows, "retention_enrichment")
    gates = {
        "accepted_path_is_scaled_right_orbit": procedural_minimum_recovery
        >= thresholds["procedural_minimum_orbit_recovery"],
        "right_orbit_localizes_late_band": (
            late_recovery - early_recovery
            >= thresholds["late_minus_early_orbit_recovery"]
            and late_gram_drift
            <= thresholds["late_to_early_gram_drift_ratio"]
            * max(early_gram_drift, 1e-30)
        ),
        "support_is_reusable": (
            support_retention
            >= thresholds["reusable_support_retention_fraction"]
            and support_enrichment
            >= thresholds["reusable_support_enrichment"]
        ),
    }
    classification = classify(gates)
    output = {
        "schema_version": RESULT_SCHEMA,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "classification": classification,
        "execution": {
            "host": "PRO6",
            "device": args.device,
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": (
                "examples.nanogpt.analyze_mlp_cproj_coadapted_orbit_geometry_v2"
            ),
            "parameter_updates": 0,
            "elapsed_seconds": time.time() - started,
        },
        "identity": {
            "plan_sha256": file_sha256(args.plan),
            "trajectory_verification_sha256": file_sha256(
                args.trajectory_verification
            ),
            "parent_acquisition_sha256": file_sha256(
                args.parent_acquisition
            ),
            "invalid_v1_audit_sha256": file_sha256(args.invalid_v1_audit),
            "trajectory_run_identity_sha256": trajectory_identity,
            "parent_run_identity_sha256": parent_identity,
        },
        "aggregate": {
            "procedural_minimum_orbit_recovery": procedural_minimum_recovery,
            "procedural_maximum_normalized_output_gram_drift": (
                procedural_maximum_gram_drift
            ),
            "dense_parent_early_weighted_orbit_recovery": early_recovery,
            "dense_parent_late_weighted_orbit_recovery": late_recovery,
            "dense_parent_late_minus_early_orbit_recovery": (
                late_recovery - early_recovery
            ),
            "dense_parent_early_weighted_gram_drift": early_gram_drift,
            "dense_parent_late_weighted_gram_drift": late_gram_drift,
            "dense_parent_late_to_early_gram_drift_ratio": late_gram_drift
            / max(early_gram_drift, 1e-30),
            "mean_consecutive_support_retention": support_retention,
            "mean_consecutive_support_retention_enrichment": support_enrichment,
        },
        "gates": gates,
        "thresholds": thresholds,
        "inventory": {
            "trajectory_steps": trajectory_steps,
            "parent_steps": list(PHASE_STEPS),
            "row_count": len(rows),
            "procedural_orbit_rows": len(procedural_rows),
            "parent_orbit_rows": len(parent_rows),
            "support_rows": len(support_rows),
            "pca_rows": len(pca_rows),
            "cross_run_parameter_displacement_rows": 0,
        },
        "authorization": {
            "fixed_support_compression_analysis": classification
            == "RIGHT_ORBIT_LOCALIZES_LWT_WITH_REUSABLE_SUPPORT",
            "phase_atlas_compression_analysis": classification
            == "RIGHT_ORBIT_LOCALIZES_LWT_BUT_SUPPORT_IS_MOVING",
            "functional_metric_path_analysis": classification in (
                "RIGHT_ORBIT_NOT_LAYER_LOCALIZING_BUT_SUPPORT_IS_REUSABLE",
                "ADAPTIVE_RIGHT_ORBIT_WITH_MOVING_SUPPORT_NOT_LAYER_LOCALIZED",
            ),
            "implement_candidate_structure": False,
            "run_exact_config_mfu": False,
            "run_language_model_training": False,
            "larger_rung": False,
        },
    }
    if not all(
        math.isfinite(float(value))
        for value in output["aggregate"].values()
    ):
        raise ValueError("non-finite aggregate metric")
    write_csv(args.rows, rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
