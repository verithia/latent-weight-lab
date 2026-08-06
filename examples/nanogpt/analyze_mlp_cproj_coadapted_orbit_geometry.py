#!/usr/bin/env python3
"""Measure the accepted late-c_proj path in its actual right-orthogonal orbit.

MuonMatchedGivensLinear owns a folded dense weight buffer.  Each optimizer
step applies task-selected rotations on the hidden/input side plus scalar
weight decay.  The resulting family is a scaled right-orthogonal orbit: it
preserves the shape of ``W W^T`` and the normalized singular spectrum.

This zero-update analysis compares the accepted co-adapted layers 8--11 path
with the same-seed dense-parent path.  It also measures whether task-selected
Givens edge supports persist.  Temporal PCA is descriptive only; it is never
treated as proof that an arbitrary compact chart fits the path.
"""

from __future__ import annotations

import argparse
import datetime as dt
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

from examples.nanogpt.analyze_mlp_cproj_activation_weighted_output_selector import (
    file_sha256,
)
from examples.nanogpt.analyze_mlp_cproj_diagonal_kfac_selector import (
    require_full_state_snapshot,
)
from examples.nanogpt.analyze_parameter_trajectory import (
    summarize_parameter,
    write_csv,
)


PLAN_SCHEMA = "mai_124m_mlp_cproj_coadapted_orbit_geometry_plan_v1"
RESULT_SCHEMA = "mai_124m_mlp_cproj_coadapted_orbit_geometry_result_v1"
LATE_LAYERS = tuple(range(8, 12))
EARLY_LAYERS = tuple(range(8))
ALL_LAYERS = tuple(range(12))
PHASE_STEPS = (0, 594, 1188, 1782, 2373)


def git_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def parameter_name(layer: int) -> str:
    return f"transformer.h.{layer}.mlp.c_proj.weight"


def buffer_name(layer: int, suffix: str) -> str:
    return f"transformer.h.{layer}.mlp.c_proj.{suffix}"


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.double().reshape(-1)
    right = right.double().reshape(-1)
    denominator = left.norm() * right.norm()
    if float(denominator) <= 0.0:
        return 0.0
    return float((left @ right) / denominator)


def scaled_right_orbit_metrics(
    source: torch.Tensor, target: torch.Tensor
) -> dict[str, float]:
    """Return the best ``scale * source @ Q`` endpoint fit.

    The orthogonal Procrustes nuclear term is computed from the two compact
    output Gram matrices.  This avoids materializing a 3072-by-3072 matrix.
    """
    if source.ndim != 2 or source.shape != target.shape:
        raise ValueError("right-orbit endpoints must be same-shaped matrices")
    source = source.double()
    target = target.double()
    source_gram = source @ source.T
    target_gram = target @ target.T
    source_energy = source_gram.trace().clamp_min(1e-30)
    target_energy = target_gram.trace().clamp_min(1e-30)
    source_eigenvalues, source_eigenvectors = torch.linalg.eigh(source_gram)
    source_root = (
        source_eigenvectors
        * source_eigenvalues.clamp_min(0).sqrt().unsqueeze(0)
    ) @ source_eigenvectors.T
    middle = source_root @ target_gram @ source_root
    nuclear = torch.linalg.eigvalsh(
        (middle + middle.T) * 0.5
    ).clamp_min(0).sqrt().sum()
    optimal_scale = nuclear / source_energy
    residual_energy = (
        target_energy - nuclear.square() / source_energy
    ).clamp_min(0)
    delta_energy = (target - source).square().sum().clamp_min(1e-30)
    normalized_source_gram = source_gram / source_energy
    normalized_target_gram = target_gram / target_energy
    gram_shape_drift = (
        (normalized_target_gram - normalized_source_gram).norm()
        / normalized_source_gram.norm().clamp_min(1e-30)
    )
    return {
        "delta_energy": float(delta_energy),
        "optimal_scale": float(optimal_scale),
        "orbit_residual_energy": float(residual_energy),
        "orbit_recovery": float(1.0 - residual_energy / delta_energy),
        "normalized_output_gram_drift": float(gram_shape_drift),
    }


def permutation_edges(permutations: torch.Tensor) -> tuple[set[int], int]:
    if permutations.ndim != 2 or permutations.shape[1] % 2:
        raise ValueError("Givens permutations must be [stages, even_width]")
    width = int(permutations.shape[1])
    pairs = permutations.long().reshape(-1, 2)
    left = torch.minimum(pairs[:, 0], pairs[:, 1])
    right = torch.maximum(pairs[:, 0], pairs[:, 1])
    if bool((left == right).any()):
        raise ValueError("Givens support contains a self edge")
    encoded = (left * width + right).tolist()
    return set(int(value) for value in encoded), width


def support_metrics(
    left: torch.Tensor, right: torch.Tensor
) -> dict[str, float | int]:
    left_edges, left_width = permutation_edges(left)
    right_edges, right_width = permutation_edges(right)
    if left_width != right_width:
        raise ValueError("Givens support widths disagree")
    universe = left_width * (left_width - 1) // 2
    smaller = min(len(left_edges), len(right_edges))
    intersection = len(left_edges & right_edges)
    retention = intersection / max(smaller, 1)
    random_retention = max(len(left_edges), len(right_edges)) / max(
        universe, 1
    )
    return {
        "left_edges": len(left_edges),
        "right_edges": len(right_edges),
        "intersection_edges": intersection,
        "retention_fraction": retention,
        "random_retention_fraction": random_retention,
        "retention_enrichment": retention / max(random_retention, 1e-30),
    }


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unexpected orbit-geometry plan schema")
    expected = {
        "same_initialization_max_absolute_error": 0.0,
        "procedural_minimum_orbit_recovery": 0.995,
        "late_minus_early_orbit_recovery": 0.05,
        "late_to_early_gram_drift_ratio": 0.8,
        "reusable_support_retention_fraction": 0.10,
        "reusable_support_enrichment": 2.0,
        "path_alignment_cosine": 0.5,
    }
    if plan.get("decision_rule", {}).get("thresholds") != expected:
        raise ValueError("orbit-geometry thresholds changed")
    analysis = plan.get("analysis", {})
    if analysis.get("parameter_updates") != 0:
        raise ValueError("orbit geometry must perform zero parameter updates")
    if analysis.get("late_layers") != list(LATE_LAYERS):
        raise ValueError("late-layer set changed")
    if analysis.get("phase_steps") != list(PHASE_STEPS):
        raise ValueError("phase steps changed")
    authorization = plan.get("authorization", {})
    if authorization.get("run_zero_update_orbit_geometry") is not True:
        raise ValueError("zero-update orbit analysis is not authorized")
    for key in (
        "implement_candidate_structure",
        "run_exact_config_mfu",
        "run_language_model_training",
        "larger_rung",
    ):
        if authorization.get(key) is not False:
            raise ValueError(f"plan must keep {key} false")


def load_snapshot(path: Path, expected_hash: str, run_identity: str) -> dict[str, Any]:
    if file_sha256(path) != expected_hash:
        raise ValueError(f"snapshot SHA-256 mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    require_full_state_snapshot(payload)
    if payload.get("run_identity_sha256") != run_identity:
        raise ValueError("snapshot run identity mismatch")
    return payload


def weighted_mean(rows: list[dict[str, Any]], field: str) -> float:
    denominator = sum(float(row["delta_energy"]) for row in rows)
    return sum(
        float(row[field]) * float(row["delta_energy"]) for row in rows
    ) / max(denominator, 1e-30)


def mean(rows: list[dict[str, Any]], field: str) -> float:
    return sum(float(row[field]) for row in rows) / max(len(rows), 1)


def classify(gates: dict[str, bool]) -> str:
    """Apply the frozen fail-closed interpretation to measured gates."""
    if not gates["same_initialization"]:
        return "INVALID_CROSS_RUN_GAUGE"
    if not gates["accepted_path_is_scaled_right_orbit"]:
        return "ACCEPTED_PATH_EXCEEDS_SCALED_RIGHT_ORBIT"
    if gates["right_orbit_localizes_late_band"] and gates[
        "support_is_reusable"
    ]:
        return "RIGHT_ORBIT_LOCALIZES_LWT_WITH_REUSABLE_SUPPORT"
    if gates["right_orbit_localizes_late_band"]:
        return "RIGHT_ORBIT_LOCALIZES_LWT_BUT_SUPPORT_IS_MOVING"
    if not gates["support_is_reusable"]:
        return "TASK_METRIC_OR_SUPPORT_TRANSPORT_EXPLAINS_LWT_NOT_ORBIT_CAPACITY"
    return "RIGHT_ORBIT_CAPACITY_NOT_LAYER_LOCALIZING"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--trajectory-verification", type=Path, required=True)
    parser.add_argument("--trajectory-snapshot-dir", type=Path, required=True)
    parser.add_argument("--parent-acquisition", type=Path, required=True)
    parser.add_argument("--parent-snapshot-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists() or args.rows.exists():
        raise FileExistsError("orbit-geometry output already exists")

    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    identity = plan["identity"]
    if file_sha256(args.trajectory_verification) != identity[
        "trajectory_verification_sha256"
    ]:
        raise ValueError("trajectory verification SHA-256 mismatch")
    if file_sha256(args.parent_acquisition) != identity[
        "parent_acquisition_sha256"
    ]:
        raise ValueError("parent acquisition SHA-256 mismatch")
    if file_sha256(Path(__file__)) != identity["analyzer_sha256"]:
        raise ValueError("analyzer SHA-256 mismatch")
    trajectory = json.loads(args.trajectory_verification.read_text())
    parent = json.loads(args.parent_acquisition.read_text())
    if trajectory.get("classification") != (
        "ACCEPTED_COADAPTED_LATE_CPROJ_FULL_STATE_TRAJECTORY"
    ):
        raise ValueError("co-adapted trajectory is not accepted")
    if parent.get("classification") != (
        "ACCEPTED_PARENT_EQUIVALENT_EXACT_FUNCTIONAL_REPLAY"
    ):
        raise ValueError("dense parent acquisition is not accepted")

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
    initialization_rows: list[dict[str, Any]] = []
    for layer in LATE_LAYERS:
        delta = trajectory_weights[0][layer] - parent_weights[0][layer]
        initialization_rows.append(
            {
                "row_type": "same_initialization",
                "layer": layer,
                "maximum_absolute_error": float(delta.abs().max()),
                "bitwise_equal": bool(
                    torch.equal(
                        trajectory_weights[0][layer], parent_weights[0][layer]
                    )
                ),
            }
        )
    rows.extend(initialization_rows)

    procedural_orbit_rows: list[dict[str, Any]] = []
    for layer in LATE_LAYERS:
        for start, end in zip(
            trajectory_steps[:-1], trajectory_steps[1:], strict=True
        ):
            metrics = scaled_right_orbit_metrics(
                trajectory_weights[start][layer].to(args.device),
                trajectory_weights[end][layer].to(args.device),
            )
            procedural_orbit_rows.append(
                {
                    "row_type": "procedural_consecutive_orbit",
                    "layer": layer,
                    "start_step": start,
                    "end_step": end,
                    **metrics,
                }
            )
    rows.extend(procedural_orbit_rows)

    parent_orbit_rows: list[dict[str, Any]] = []
    for layer in ALL_LAYERS:
        for start, end in zip(PHASE_STEPS[:-1], PHASE_STEPS[1:], strict=True):
            metrics = scaled_right_orbit_metrics(
                parent_weights[start][layer].to(args.device),
                parent_weights[end][layer].to(args.device),
            )
            parent_orbit_rows.append(
                {
                    "row_type": "dense_parent_phase_orbit",
                    "band": "early" if layer in EARLY_LAYERS else "late",
                    "layer": layer,
                    "start_step": start,
                    "end_step": end,
                    **metrics,
                }
            )
    rows.extend(parent_orbit_rows)

    support_rows: list[dict[str, Any]] = []
    for layer in LATE_LAYERS:
        for start, end in zip(
            trajectory_steps[1:-1], trajectory_steps[2:], strict=True
        ):
            left_refresh = trajectory_refresh[start][layer]
            right_refresh = trajectory_refresh[end][layer]
            if not left_refresh[2] or not right_refresh[2]:
                continue
            metrics = support_metrics(
                trajectory_supports[start][layer],
                trajectory_supports[end][layer],
            )
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
                    **metrics,
                }
            )
    rows.extend(support_rows)

    alignment_rows: list[dict[str, Any]] = []
    for layer in LATE_LAYERS:
        common = parent_weights[0][layer]
        for step in PHASE_STEPS[1:]:
            procedural_delta = trajectory_weights[step][layer] - common
            parent_delta = parent_weights[step][layer] - common
            alignment_rows.append(
                {
                    "row_type": "coadapted_dense_path_alignment",
                    "layer": layer,
                    "step": step,
                    "displacement_cosine": cosine(
                        procedural_delta, parent_delta
                    ),
                    "procedural_displacement_fro": float(
                        procedural_delta.norm()
                    ),
                    "parent_displacement_fro": float(parent_delta.norm()),
                }
            )
    rows.extend(alignment_rows)

    pca_rows: list[dict[str, Any]] = []
    for layer in LATE_LAYERS:
        name = parameter_name(layer)
        fine_summary, _coordinates, _polynomials = summarize_parameter(
            name=name,
            steps=trajectory_steps,
            tensors=[trajectory_weights[step][layer] for step in trajectory_steps],
            device=args.device,
        )
        pca_rows.append(
            {"row_type": "procedural_fine_temporal_pca", **fine_summary}
        )
        phase_summary, _coordinates, _polynomials = summarize_parameter(
            name=name,
            steps=list(PHASE_STEPS),
            tensors=[trajectory_weights[step][layer] for step in PHASE_STEPS],
            device=args.device,
        )
        pca_rows.append(
            {"row_type": "procedural_phase_temporal_pca", **phase_summary}
        )
        parent_summary, _coordinates, _polynomials = summarize_parameter(
            name=name,
            steps=list(PHASE_STEPS),
            tensors=[parent_weights[step][layer] for step in PHASE_STEPS],
            device=args.device,
        )
        pca_rows.append(
            {"row_type": "dense_parent_phase_temporal_pca", **parent_summary}
        )
    rows.extend(pca_rows)

    thresholds = plan["decision_rule"]["thresholds"]
    same_initialization_error = max(
        float(row["maximum_absolute_error"]) for row in initialization_rows
    )
    procedural_minimum_orbit_recovery = min(
        float(row["orbit_recovery"]) for row in procedural_orbit_rows
    )
    procedural_maximum_gram_drift = max(
        float(row["normalized_output_gram_drift"])
        for row in procedural_orbit_rows
    )
    parent_early = [
        row for row in parent_orbit_rows if row["band"] == "early"
    ]
    parent_late = [
        row for row in parent_orbit_rows if row["band"] == "late"
    ]
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
    path_alignment = mean(alignment_rows, "displacement_cosine")
    gates = {
        "same_initialization": same_initialization_error
        <= thresholds["same_initialization_max_absolute_error"],
        "accepted_path_is_scaled_right_orbit": procedural_minimum_orbit_recovery
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
        "coadapted_path_tracks_dense_parent": path_alignment
        >= thresholds["path_alignment_cosine"],
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
                "examples.nanogpt.analyze_mlp_cproj_coadapted_orbit_geometry"
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
            "trajectory_run_identity_sha256": trajectory_identity,
            "parent_run_identity_sha256": parent_identity,
        },
        "aggregate": {
            "same_initialization_max_absolute_error": same_initialization_error,
            "procedural_minimum_orbit_recovery": procedural_minimum_orbit_recovery,
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
            "mean_coadapted_dense_displacement_cosine": path_alignment,
        },
        "gates": gates,
        "thresholds": thresholds,
        "inventory": {
            "trajectory_steps": trajectory_steps,
            "parent_steps": list(PHASE_STEPS),
            "row_count": len(rows),
            "initialization_rows": len(initialization_rows),
            "procedural_orbit_rows": len(procedural_orbit_rows),
            "parent_orbit_rows": len(parent_orbit_rows),
            "support_rows": len(support_rows),
            "alignment_rows": len(alignment_rows),
            "pca_rows": len(pca_rows),
        },
        "authorization": {
            "phase_atlas_compression_analysis": classification
            == "RIGHT_ORBIT_LOCALIZES_LWT_BUT_SUPPORT_IS_MOVING",
            "functional_metric_path_analysis": classification
            == (
                "TASK_METRIC_OR_SUPPORT_TRANSPORT_EXPLAINS_LWT_NOT_ORBIT_CAPACITY"
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
