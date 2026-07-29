#!/usr/bin/env python3
"""Test causal task-selected hidden/output Givens updates for MLP c_proj.

The already validated hidden-side chart follows ``W -> W R_hidden``.  This
diagnostic asks whether its remaining exact-Muon residual contains useful
output-side motion ``W -> R_output^T W``.  The output matching and angles are
selected only after applying the causal hidden update and use only the
remaining current requested update.  Future dense-trajectory chords remain
hidden until scoring.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.nanogpt.analyze_mlp_bilateral_phase_capture import (
    file_sha256,
    git_commit,
)
from examples.nanogpt.analyze_mlp_optimizer_state_direction import (
    reconstruct_directions,
)
from examples.nanogpt.analyze_mlp_task_gradient_direction import (
    direction_metrics,
)
from examples.nanogpt.analyze_parameter_trajectory import (
    load_snapshots,
    parse_int_list,
    write_csv,
)
from examples.nanogpt.muon_matched_givens import (
    apply_givens_flow,
    diagonal_metric_angles,
    muon_matched_permutations,
    random_unique_matchings,
)
from examples.nanogpt.parameter_trajectory import (
    OPTIMIZER_PROBE_SCHEMA_VERSION,
)


def one_sided_update(
    source: torch.Tensor,
    requested_update: torch.Tensor,
    permutations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply one closed-form right-side Givens correction."""
    permutations = permutations.to(
        device=source.device, dtype=torch.long
    )
    angles = diagonal_metric_angles(
        source, requested_update, permutations
    )
    updated = apply_givens_flow(
        source,
        angles,
        permutations,
        torch.argsort(permutations, dim=1),
    )
    return updated, updated - source, angles


def causal_bilateral_update(
    source: torch.Tensor,
    requested_update: torch.Tensor,
    *,
    right_permutations: torch.Tensor,
    left_permutations: torch.Tensor | None,
    left_stages: int,
    left_neighbors: int,
    left_seed: int,
) -> tuple[torch.Tensor, dict[str, float | int], torch.Tensor]:
    """Fit one causal diagonal Gauss-Seidel hidden/output sweep.

    Hidden- and output-side tangent columns are not mutually orthogonal.
    Therefore this right-then-left residual correction is a deployable
    closed-form approximation, not the exact joint least-squares solution.
    """
    after_right, right_update, right_angles = one_sided_update(
        source, requested_update, right_permutations
    )
    residual = requested_update - right_update
    transposed = after_right.T.contiguous()
    if left_permutations is None:
        left_permutations, _diagnostics = muon_matched_permutations(
            transposed,
            residual.T.contiguous(),
            stages=left_stages,
            neighbors=left_neighbors,
            seed=left_seed,
        )
    after_left_t, _left_update_t, left_angles = one_sided_update(
        transposed,
        residual.T.contiguous(),
        left_permutations,
    )
    final = after_left_t.T.contiguous()
    predicted_update = final - source
    requested_energy = requested_update.float().square().sum().clamp_min(
        1e-30
    )
    right_residual_energy = (
        requested_update.float() - right_update.float()
    ).square().sum()
    final_residual_energy = (
        requested_update.float() - predicted_update.float()
    ).square().sum()
    coordinates = int(
        right_angles.numel() + left_angles.numel()
    )
    return predicted_update, {
        "coordinates": coordinates,
        "coordinate_fraction": coordinates / source.numel(),
        "right_requested_update_recovery": float(
            1.0 - right_residual_energy / requested_energy
        ),
        "requested_update_recovery": float(
            1.0 - final_residual_energy / requested_energy
        ),
        "left_incremental_recovery": float(
            (right_residual_energy - final_residual_energy)
            / requested_energy
        ),
        "right_angle_rms": float(
            right_angles.square().mean().sqrt()
        ),
        "left_angle_rms": float(
            left_angles.square().mean().sqrt()
        ),
    }, final


def one_sided_record(
    source: torch.Tensor,
    requested_update: torch.Tensor,
    permutations: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    updated, predicted, angles = one_sided_update(
        source, requested_update, permutations
    )
    requested_energy = requested_update.float().square().sum().clamp_min(
        1e-30
    )
    residual_energy = (
        requested_update.float() - predicted.float()
    ).square().sum()
    coordinates = int(angles.numel())
    return predicted, {
        "coordinates": coordinates,
        "coordinate_fraction": coordinates / source.numel(),
        "requested_update_recovery": float(
            1.0 - residual_energy / requested_energy
        ),
        "angle_rms": float(angles.square().mean().sqrt()),
        "updated_weight_fro": float(updated.float().norm()),
    }


def output_only_record(
    source: torch.Tensor,
    requested_update: torch.Tensor,
    permutations: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    predicted_t, record = one_sided_record(
        source.T.contiguous(),
        requested_update.T.contiguous(),
        permutations,
    )
    return predicted_t.T.contiguous(), record


def aggregate_rows(
    rows: list[dict[str, Any]],
    *,
    minimum_current_ratio: float,
    minimum_future_ratio: float,
    minimum_task_over_random: float,
    minimum_cell_future_cosine: float,
) -> tuple[dict[str, Any], str]:
    kinds = sorted({str(row["chart"]) for row in rows})
    by_kind: dict[str, dict[str, float | int]] = {}
    for kind in kinds:
        selected = [row for row in rows if row["chart"] == kind]
        energy = torch.tensor(
            [float(row["future_chord_fro"]) ** 2 for row in selected],
            dtype=torch.float64,
        )

        def weighted(key: str) -> float:
            values = torch.tensor(
                [float(row[key]) for row in selected],
                dtype=torch.float64,
            )
            return float((energy * values).sum() / energy.sum())

        by_kind[kind] = {
            "cells": len(selected),
            "coordinates_per_layer": int(selected[0]["coordinates"]),
            "coordinate_fraction": float(
                selected[0]["coordinate_fraction"]
            ),
            "requested_update_recovery": weighted(
                "requested_update_recovery"
            ),
            "future_recovery": weighted("future_recovery"),
            "future_cosine": weighted("future_cosine"),
            "minimum_future_cosine": min(
                float(row["future_cosine"]) for row in selected
            ),
            "positive_future_cells": sum(
                float(row["future_cosine"]) > 0.0
                for row in selected
            ),
        }
        if "left_incremental_recovery" in selected[0]:
            by_kind[kind]["left_incremental_recovery"] = weighted(
                "left_incremental_recovery"
            )

    bilateral = by_kind["task_bilateral_32x32"]
    same_size = by_kind["task_hidden_40"]
    random = by_kind["random_bilateral_32x32"]
    aggregate = {
        "chart_results": by_kind,
        "bilateral_over_same_coordinate_hidden_current": (
            float(bilateral["requested_update_recovery"])
            / max(
                float(same_size["requested_update_recovery"]), 1e-30
            )
        ),
        "bilateral_over_same_coordinate_hidden_future": (
            float(bilateral["future_recovery"])
            / max(float(same_size["future_recovery"]), 1e-30)
        ),
        "task_bilateral_over_random_bilateral_future": (
            float(bilateral["future_recovery"])
            / max(float(random["future_recovery"]), 1e-30)
        ),
    }
    passed = (
        aggregate[
            "bilateral_over_same_coordinate_hidden_current"
        ]
        >= minimum_current_ratio
        and aggregate[
            "bilateral_over_same_coordinate_hidden_future"
        ]
        >= minimum_future_ratio
        and aggregate[
            "task_bilateral_over_random_bilateral_future"
        ]
        >= minimum_task_over_random
        and float(bilateral["minimum_future_cosine"])
        >= minimum_cell_future_cosine
        and int(bilateral["positive_future_cells"]) == len(
            [row for row in rows if row["chart"] == "task_bilateral_32x32"]
        )
    )
    decision = (
        "PROMOTE_TASK_SELECTED_BILATERAL_TO_PRODUCTION_DESIGN"
        if passed
        else "REJECT_TASK_SELECTED_OUTPUT_CORRECTION"
    )
    return aggregate, decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--probe-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", default="0,3,6,9,11")
    parser.add_argument("--phase-boundaries", default="0,60,120,180,238")
    parser.add_argument("--right-stages", type=int, default=32)
    parser.add_argument("--right-control-stages", type=int, default=40)
    parser.add_argument("--left-stages", type=int, default=32)
    parser.add_argument("--right-neighbors", type=int, default=64)
    parser.add_argument("--left-neighbors", type=int, default=64)
    parser.add_argument("--matching-seed", type=int, default=161803)
    parser.add_argument("--random-seed", type=int, default=271828)
    parser.add_argument(
        "--minimum-current-ratio", type=float, default=1.1
    )
    parser.add_argument(
        "--minimum-future-ratio", type=float, default=1.2
    )
    parser.add_argument(
        "--minimum-task-over-random", type=float, default=3.0
    )
    parser.add_argument(
        "--minimum-cell-future-cosine", type=float, default=0.05
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    layers = parse_int_list(args.layers)
    boundaries = parse_int_list(args.phase_boundaries)
    if (
        not layers
        or len(boundaries) < 2
        or boundaries != sorted(set(boundaries))
        or not (
            0
            < args.right_stages
            < args.right_control_stages
            <= args.right_neighbors
        )
        or not 0 < args.left_stages <= args.left_neighbors
        or args.minimum_current_ratio <= 1.0
        or args.minimum_future_ratio <= 1.0
        or args.minimum_task_over_random <= 1.0
        or args.minimum_cell_future_cosine < 0.0
    ):
        raise ValueError("invalid bilateral diagnostic arguments")
    phase_starts = boundaries[:-1]
    phase_end = dict(
        zip(boundaries[:-1], boundaries[1:], strict=True)
    )
    snapshot_paths = [
        args.snapshot_dir / f"step_{step:06d}.pt"
        for step in boundaries
    ]
    probe_paths = [
        args.probe_dir / f"step_{step:06d}.pt"
        for step in phase_starts
    ]
    missing = [
        str(path)
        for path in (*snapshot_paths, *probe_paths)
        if not path.is_file()
    ]
    if missing:
        raise ValueError(f"required inputs are absent: {missing}")
    steps, values, snapshot_metadata = load_snapshots(
        snapshot_paths,
        layers=set(layers),
        targets={"mlp.c_proj"},
    )
    step_index = {step: index for index, step in enumerate(steps)}

    rows: list[dict[str, Any]] = []
    input_run_identity: str | None = None
    for layer in layers:
        parameter = f"transformer.h.{layer}.mlp.c_proj.weight"
        for start in phase_starts:
            end = phase_end[start]
            source = values[parameter][step_index[start]].to(args.device)
            target = values[parameter][step_index[end]].to(args.device)
            future_chord = target - source
            probe_path = args.probe_dir / f"step_{start:06d}.pt"
            probe = torch.load(
                probe_path, map_location="cpu", weights_only=False
            )
            if (
                probe.get("schema_version")
                != OPTIMIZER_PROBE_SCHEMA_VERSION
            ):
                raise ValueError(
                    f"unexpected optimizer probe: {probe_path}"
                )
            observed_identity = str(probe["run_identity_sha256"])
            if input_run_identity is None:
                input_run_identity = observed_identity
            elif input_run_identity != observed_identity:
                raise ValueError(
                    "optimizer probes have inconsistent identities"
                )
            state = {
                name: tensor.to(args.device)
                for name, tensor in probe["parameters"][
                    parameter
                ].items()
            }
            directions = reconstruct_directions(
                state, probe["hyperparameters"][parameter]
            )
            scheduled_lr = float(
                probe["hyperparameters"][parameter]["lr"]
            )
            requested_update = (
                scheduled_lr * directions["exact_applied_direction"]
            )
            exact_future = direction_metrics(
                future_chord, requested_update
            )
            task_right, _ = muon_matched_permutations(
                source,
                requested_update,
                stages=args.right_control_stages,
                neighbors=args.right_neighbors,
                seed=args.matching_seed + 1009 * layer + start,
            )
            task_output, _ = muon_matched_permutations(
                source.T.contiguous(),
                requested_update.T.contiguous(),
                stages=args.left_stages,
                neighbors=args.left_neighbors,
                seed=args.matching_seed + 2003 * layer + start + 1,
            )
            random_right = random_unique_matchings(
                width=source.shape[1],
                stages=args.right_stages,
                seed=args.random_seed + 1009 * layer + start,
            )
            random_left = random_unique_matchings(
                width=source.shape[0],
                stages=args.left_stages,
                seed=args.random_seed + 2003 * layer + start + 1,
            )

            candidates: list[
                tuple[str, torch.Tensor, dict[str, float | int]]
            ] = []
            hidden32, hidden32_record = one_sided_record(
                source,
                requested_update,
                task_right[: args.right_stages],
            )
            candidates.append(
                ("task_hidden_32", hidden32, hidden32_record)
            )
            hidden40, hidden40_record = one_sided_record(
                source, requested_update, task_right
            )
            candidates.append(
                ("task_hidden_40", hidden40, hidden40_record)
            )
            output32, output32_record = output_only_record(
                source, requested_update, task_output
            )
            candidates.append(
                ("task_output_32", output32, output32_record)
            )
            bilateral, bilateral_record, _ = causal_bilateral_update(
                source,
                requested_update,
                right_permutations=task_right[
                    : args.right_stages
                ],
                left_permutations=None,
                left_stages=args.left_stages,
                left_neighbors=args.left_neighbors,
                left_seed=(
                    args.matching_seed + 2003 * layer + start + 2
                ),
            )
            candidates.append(
                (
                    "task_bilateral_32x32",
                    bilateral,
                    bilateral_record,
                )
            )
            random_bilateral, random_record, _ = (
                causal_bilateral_update(
                    source,
                    requested_update,
                    right_permutations=random_right,
                    left_permutations=random_left,
                    left_stages=args.left_stages,
                    left_neighbors=args.left_neighbors,
                    left_seed=0,
                )
            )
            candidates.append(
                (
                    "random_bilateral_32x32",
                    random_bilateral,
                    random_record,
                )
            )
            for chart, predicted, record in candidates:
                current = direction_metrics(
                    requested_update, predicted
                )
                future = direction_metrics(future_chord, predicted)
                row = {
                    "parameter": parameter,
                    "layer": layer,
                    "phase_start": start,
                    "phase_end": end,
                    "chart": chart,
                    "scheduled_learning_rate": scheduled_lr,
                    **record,
                    "requested_update_cosine": current["cosine"],
                    "future_recovery": future[
                        "positive_step_line_recovery"
                    ],
                    "future_cosine": future["cosine"],
                    "future_chord_fro": future["target_chord_fro"],
                    "exact_direction_future_recovery": exact_future[
                        "positive_step_line_recovery"
                    ],
                    "exact_direction_future_cosine": exact_future[
                        "cosine"
                    ],
                }
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
            del (
                source,
                target,
                future_chord,
                state,
                directions,
                requested_update,
            )
            if "cuda" in args.device:
                torch.cuda.empty_cache()

    aggregate, decision = aggregate_rows(
        rows,
        minimum_current_ratio=args.minimum_current_ratio,
        minimum_future_ratio=args.minimum_future_ratio,
        minimum_task_over_random=args.minimum_task_over_random,
        minimum_cell_future_cosine=args.minimum_cell_future_cosine,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    detail_path = args.output / "muon_matched_bilateral.csv"
    aggregate_path = args.output / "muon_matched_bilateral_aggregate.json"
    write_csv(detail_path, rows)
    aggregate_path.write_text(
        json.dumps(
            {"decision": decision, **aggregate},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_mlp_muon_matched_bilateral_v1",
        "decision": decision,
        "causal_protocol": (
            "task hidden connectivity/angles use the current exact Muon "
            "update; task output connectivity/angles use only its residual "
            "after the hidden correction; future phase chords are scoring "
            "only"
        ),
        "solver": (
            "one closed-form diagonal Gauss-Seidel sweep, hidden then "
            "output; this is not an exact joint normal-equation solve"
        ),
        "controls": (
            "40 hidden stages have exactly the same continuous coordinate "
            "count as 32 hidden plus 32 output stages; random bilateral uses "
            "edge-disjoint one-factorized matchings"
        ),
        "singular_value_control": (
            "not added: prior FP64 decomposition attributes only 0.6754 "
            "percent of c_proj phase energy to singular-value drift, almost "
            "all global scale"
        ),
        "layers": layers,
        "phase_boundaries": boundaries,
        "right_stages": args.right_stages,
        "right_control_stages": args.right_control_stages,
        "left_stages": args.left_stages,
        "right_neighbors": args.right_neighbors,
        "left_neighbors": args.left_neighbors,
        "decision_rule": {
            "minimum_current_ratio": args.minimum_current_ratio,
            "minimum_future_ratio": args.minimum_future_ratio,
            "minimum_task_over_random": (
                args.minimum_task_over_random
            ),
            "minimum_cell_future_cosine": (
                args.minimum_cell_future_cosine
            ),
        },
        "input_run_identity_sha256": input_run_identity,
        "snapshot_metadata": snapshot_metadata,
        "analysis_execution": {
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": str(script),
            "entrypoint_sha256": file_sha256(script),
            "command": sys.argv,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "device": args.device,
        },
        "inputs": [
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in (*snapshot_paths, *probe_paths)
        ],
        "outputs": {
            "detail": {
                "path": str(detail_path),
                "sha256": file_sha256(detail_path),
            },
            "aggregate": {
                "path": str(aggregate_path),
                "sha256": file_sha256(aggregate_path),
            },
        },
        "limitations": [
            "This is a no-update causal tangent diagnostic, not training.",
            "The dense Muon trajectory is one optimization path, not the full solution manifold.",
            "Only five representative layers are analyzed.",
        ],
    }
    metadata_path = args.output / "muon_matched_bilateral_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "decision": decision,
                "aggregate": aggregate,
                "detail": str(detail_path),
                "metadata": str(metadata_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
