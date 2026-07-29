#!/usr/bin/env python3
"""Test sparse hidden Givens charts selected by coherent Muon state."""

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
from examples.nanogpt.analyze_mlp_global_givens_transport_fit import (
    parse_cells,
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
from examples.nanogpt.model import LearnedGivensOutputMix
from examples.nanogpt.parameter_trajectory import (
    OPTIMIZER_PROBE_SCHEMA_VERSION,
)


def _complete_unique_matchings(
    edge_scores: dict[tuple[int, int], float],
    *,
    width: int,
    stages: int,
    seed: int,
) -> tuple[torch.Tensor, list[dict[str, float | int]]]:
    """Greedily edge-color scores into unique perfect matchings."""
    ordered_edges = sorted(
        (
            (score, left, right)
            for (left, right), score in edge_scores.items()
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    used_edges: set[tuple[int, int]] = set()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    permutations: list[torch.Tensor] = []
    diagnostics: list[dict[str, float | int]] = []
    for stage in range(stages):
        occupied = [False] * width
        pairs: list[tuple[int, int, float, bool]] = []
        for score, left, right in ordered_edges:
            edge = (left, right)
            if edge in used_edges or occupied[left] or occupied[right]:
                continue
            occupied[left] = occupied[right] = True
            used_edges.add(edge)
            pairs.append((left, right, score, True))
            if len(pairs) == width // 2:
                break

        remaining = [
            index for index, is_occupied in enumerate(occupied)
            if not is_occupied
        ]
        if remaining:
            order = torch.randperm(
                len(remaining), generator=generator
            ).tolist()
            remaining = [remaining[index] for index in order]
        while remaining:
            left = remaining.pop()
            partner_index = next(
                (
                    index
                    for index, right in enumerate(remaining)
                    if (min(left, right), max(left, right))
                    not in used_edges
                ),
                None,
            )
            if partner_index is None:
                repaired = False
                for right_index, right in enumerate(remaining):
                    for pair_index, (
                        prior_left,
                        prior_right,
                        _prior_score,
                        _prior_candidate,
                    ) in enumerate(pairs):
                        prior_edge = (
                            min(prior_left, prior_right),
                            max(prior_left, prior_right),
                        )
                        for first, second in (
                            (prior_left, prior_right),
                            (prior_right, prior_left),
                        ):
                            first_edge = (
                                min(left, first), max(left, first)
                            )
                            second_edge = (
                                min(right, second), max(right, second)
                            )
                            if (
                                first_edge in used_edges
                                or second_edge in used_edges
                                or first_edge == second_edge
                            ):
                                continue
                            used_edges.remove(prior_edge)
                            used_edges.add(first_edge)
                            used_edges.add(second_edge)
                            pairs[pair_index] = (
                                left,
                                first,
                                edge_scores.get(first_edge, 0.0),
                                first_edge in edge_scores,
                            )
                            pairs.append(
                                (
                                    right,
                                    second,
                                    edge_scores.get(second_edge, 0.0),
                                    second_edge in edge_scores,
                                )
                            )
                            remaining.pop(right_index)
                            repaired = True
                            break
                        if repaired:
                            break
                    if repaired:
                        break
                if not repaired:
                    raise RuntimeError(
                        "could not complete a unique task matching"
                    )
                continue
            right = remaining.pop(partner_index)
            edge = (min(left, right), max(left, right))
            used_edges.add(edge)
            pairs.append(
                (
                    left,
                    right,
                    edge_scores.get(edge, 0.0),
                    edge in edge_scores,
                )
            )

        if len(pairs) != width // 2:
            raise RuntimeError("task matching is incomplete")
        permutation = torch.tensor(
            [
                index
                for left, right, _score, _candidate in pairs
                for index in (left, right)
            ],
            dtype=torch.long,
        )
        if not torch.equal(
            torch.sort(permutation).values, torch.arange(width)
        ):
            raise RuntimeError("task matching is not a permutation")
        permutations.append(permutation)
        diagnostics.append(
            {
                "stage": stage,
                "pairs": len(pairs),
                "candidate_edge_fraction": (
                    sum(candidate for *_rest, candidate in pairs)
                    / len(pairs)
                ),
                "mean_abs_coordinate_gradient": (
                    sum(
                        score
                        for _left, _right, score, _candidate in pairs
                    )
                    / len(pairs)
                ),
            }
        )
    return torch.stack(permutations), diagnostics


def muon_matched_permutations(
    weight: torch.Tensor,
    direction: torch.Tensor,
    *,
    stages: int,
    neighbors: int,
    seed: int,
) -> tuple[torch.Tensor, list[dict[str, float | int]]]:
    """Select hidden-channel pairs from the exact identity-angle gradient.

    For a pair ``(i, j)``, the identity-angle derivative has columns
    ``(-W[:, j], W[:, i])``.  Its inner product with the requested
    materialized direction is
    ``<W[:, i], D[:, j]> - <W[:, j], D[:, i]>``.
    """
    if (
        weight.ndim != 2
        or weight.shape != direction.shape
        or weight.shape[1] <= 0
        or weight.shape[1] % 2
    ):
        raise ValueError(
            "weight and direction must be same-shaped matrices with even width"
        )
    width = int(weight.shape[1])
    if stages <= 0 or neighbors < stages or neighbors >= width:
        raise ValueError("require 0 < stages <= neighbors < width")
    weight = weight.float()
    direction = direction.float()
    cross = weight.T @ direction
    scores = (cross - cross.T).abs()
    scores.fill_diagonal_(-1.0)
    top_scores, top_indices = torch.topk(scores, k=neighbors, dim=1)
    top_scores = top_scores.cpu()
    top_indices = top_indices.cpu()
    del cross, scores

    edge_scores: dict[tuple[int, int], float] = {}
    for left in range(width):
        for raw_score, raw_right in zip(
            top_scores[left].tolist(),
            top_indices[left].tolist(),
            strict=True,
        ):
            right = int(raw_right)
            edge = (left, right) if left < right else (right, left)
            edge_scores[edge] = max(
                float(raw_score), edge_scores.get(edge, -1.0)
            )
    return _complete_unique_matchings(
        edge_scores, width=width, stages=stages, seed=seed
    )


def fit_causal_givens_update(
    source: torch.Tensor,
    requested_update: torch.Tensor,
    *,
    stages: int,
    seed: int,
    steps: int,
    learning_rate: float,
    permutations: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Fit angles only to the current requested update, never a future chord."""
    if source.ndim != 2 or source.shape != requested_update.shape:
        raise ValueError(
            "source and requested_update must be same-shaped matrices"
        )
    if stages <= 0 or steps <= 0 or learning_rate <= 0.0:
        raise ValueError("stages, steps, and learning_rate must be positive")
    source = source.float()
    requested_update = requested_update.float()
    update_energy = requested_update.square().sum().clamp_min(1e-30)
    flow = LearnedGivensOutputMix(
        source.shape[-1], int(stages), int(seed)
    ).to(device=source.device, dtype=torch.float32)
    if permutations is not None:
        expected = (int(stages), source.shape[-1])
        if tuple(permutations.shape) != expected:
            raise ValueError(
                f"permutations must have shape {expected}, got "
                f"{tuple(permutations.shape)}"
            )
        with torch.no_grad():
            flow.permutations.copy_(
                permutations.to(flow.permutations.device)
            )
            flow.inverse_permutations.copy_(
                torch.argsort(flow.permutations, dim=1)
            )

    optimizer = torch.optim.Adam(
        [flow.angles], lr=float(learning_rate)
    )
    best_loss = float("inf")
    best_angles = flow.angles.detach().clone()
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        predicted_update = flow(source) - source
        loss = (
            requested_update - predicted_update
        ).square().sum() / update_energy
        loss.backward()
        optimizer.step()
        observed = float(loss.detach())
        if observed < best_loss:
            best_loss = observed
            best_angles.copy_(flow.angles.detach())
    with torch.no_grad():
        flow.angles.copy_(best_angles)
        predicted_update = flow(source) - source
        residual = requested_update - predicted_update
        recovery = 1.0 - residual.square().sum() / update_energy
    return predicted_update.detach(), {
        "stages": int(stages),
        "coordinates": int(flow.angles.numel()),
        "coordinate_fraction": float(
            flow.angles.numel() / source.numel()
        ),
        "requested_update_recovery": float(recovery),
        "angle_rms": float(
            flow.angles.detach().square().mean().sqrt()
        ),
        "angle_max_abs": float(flow.angles.detach().abs().max()),
    }


def aggregate_rows(
    rows: list[dict[str, Any]],
    stage_counts: list[int],
    *,
    minimum_future_recovery: float = 0.02,
    minimum_future_over_random: float = 2.0,
    minimum_update_over_coordinate: float = 4.0,
) -> tuple[list[dict[str, Any]], str]:
    aggregates: list[dict[str, Any]] = []
    for stages in stage_counts:
        by_kind: dict[str, dict[str, float | int]] = {}
        for kind in ("task_matched", "random"):
            selected = [
                row
                for row in rows
                if int(row["stages"]) == stages
                and row["connectivity"] == kind
            ]
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
        task = by_kind["task_matched"]
        random = by_kind["random"]
        aggregates.append(
            {
                "stages": stages,
                "task_matched": task,
                "random": random,
                "task_over_random_future_recovery": (
                    float(task["future_recovery"])
                    / max(float(random["future_recovery"]), 1e-30)
                ),
                "task_update_recovery_over_coordinate_fraction": (
                    float(task["requested_update_recovery"])
                    / float(task["coordinate_fraction"])
                ),
            }
        )

    selected = next(
        row for row in aggregates
        if int(row["stages"]) == max(stage_counts)
    )
    task = selected["task_matched"]
    if (
        float(task["future_recovery"]) >= minimum_future_recovery
        and float(selected["task_over_random_future_recovery"])
        >= minimum_future_over_random
        and float(
            selected["task_update_recovery_over_coordinate_fraction"]
        )
        >= minimum_update_over_coordinate
        and int(task["positive_future_cells"]) == int(task["cells"])
    ):
        decision = "PROMOTE_MUON_MATCHED_GIVENS_TO_ALL_CELL_ORACLE"
    else:
        decision = "REJECT_MUON_MATCHED_SPARSE_GIVENS_PILOT"
    return aggregates, decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--probe-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cells", default="0:0,0:180,6:60,11:120")
    parser.add_argument(
        "--phase-boundaries", default="0,60,120,180,238"
    )
    parser.add_argument("--stages", default="1,4,8")
    parser.add_argument("--neighbors", type=int, default=64)
    parser.add_argument("--fit-steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--matching-seed", type=int, default=161803)
    parser.add_argument("--random-seed", type=int, default=271828)
    parser.add_argument(
        "--minimum-future-recovery", type=float, default=0.02
    )
    parser.add_argument(
        "--minimum-future-over-random", type=float, default=2.0
    )
    parser.add_argument(
        "--minimum-update-over-coordinate", type=float, default=4.0
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    cells = parse_cells(args.cells)
    boundaries = parse_int_list(args.phase_boundaries)
    stage_counts = parse_int_list(args.stages)
    if (
        not stage_counts
        or min(stage_counts) <= 0
        or max(stage_counts) > args.neighbors
        or args.minimum_future_recovery <= 0.0
        or args.minimum_future_over_random <= 0.0
        or args.minimum_update_over_coordinate <= 0.0
    ):
        raise ValueError("invalid stage counts or decision thresholds")
    end_by_start = dict(
        zip(boundaries[:-1], boundaries[1:], strict=True)
    )
    if any(phase not in end_by_start for _layer, phase in cells):
        raise ValueError("each cell must use a registered phase start")
    required_steps = sorted(
        {
            step
            for _layer, phase in cells
            for step in (phase, end_by_start[phase])
        }
    )
    snapshot_paths = [
        args.snapshot_dir / f"step_{step:06d}.pt"
        for step in required_steps
    ]
    probe_paths = [
        args.probe_dir / f"step_{phase:06d}.pt"
        for phase in sorted({phase for _layer, phase in cells})
    ]
    missing = [
        str(path)
        for path in (*snapshot_paths, *probe_paths)
        if not path.is_file()
    ]
    if missing:
        raise ValueError(f"required inputs are absent: {missing}")
    layers = {layer for layer, _phase in cells}
    steps, values, snapshot_metadata = load_snapshots(
        snapshot_paths,
        layers=layers,
        targets={"mlp.c_proj"},
    )
    step_index = {step: index for index, step in enumerate(steps)}

    rows: list[dict[str, Any]] = []
    matching_rows: list[dict[str, Any]] = []
    input_run_identity: str | None = None
    for layer, phase_start in cells:
        phase_end = end_by_start[phase_start]
        parameter = f"transformer.h.{layer}.mlp.c_proj.weight"
        source = values[parameter][step_index[phase_start]].to(args.device)
        target = values[parameter][step_index[phase_end]].to(args.device)
        future_chord = target - source
        probe_path = args.probe_dir / f"step_{phase_start:06d}.pt"
        probe = torch.load(
            probe_path, map_location="cpu", weights_only=False
        )
        if probe.get("schema_version") != OPTIMIZER_PROBE_SCHEMA_VERSION:
            raise ValueError(f"unexpected optimizer probe: {probe_path}")
        observed_identity = str(probe["run_identity_sha256"])
        if input_run_identity is None:
            input_run_identity = observed_identity
        elif input_run_identity != observed_identity:
            raise ValueError("optimizer probes have inconsistent identities")
        state = {
            name: tensor.to(args.device)
            for name, tensor in probe["parameters"][parameter].items()
        }
        hyperparameters = probe["hyperparameters"][parameter]
        directions = reconstruct_directions(state, hyperparameters)
        exact_direction = directions["exact_applied_direction"]
        scheduled_lr = float(hyperparameters["lr"])
        requested_update = scheduled_lr * exact_direction
        exact_future = direction_metrics(future_chord, requested_update)
        permutations, matching_diagnostics = muon_matched_permutations(
            source,
            exact_direction,
            stages=max(stage_counts),
            neighbors=args.neighbors,
            seed=args.matching_seed + 1009 * layer + phase_start,
        )
        for diagnostic in matching_diagnostics:
            matching_rows.append(
                {
                    "layer": layer,
                    "phase_start": phase_start,
                    **diagnostic,
                }
            )

        for stages in stage_counts:
            for connectivity, selected_permutations in (
                ("task_matched", permutations[:stages]),
                ("random", None),
            ):
                predicted_update, fit = fit_causal_givens_update(
                    source,
                    requested_update,
                    stages=stages,
                    seed=args.random_seed + 1009 * layer + phase_start,
                    steps=args.fit_steps,
                    learning_rate=args.learning_rate,
                    permutations=selected_permutations,
                )
                update_metrics = direction_metrics(
                    requested_update, predicted_update
                )
                future_metrics = direction_metrics(
                    future_chord, predicted_update
                )
                row = {
                    "parameter": parameter,
                    "layer": layer,
                    "phase_start": phase_start,
                    "phase_end": phase_end,
                    "scheduled_learning_rate": scheduled_lr,
                    "connectivity": connectivity,
                    **fit,
                    "requested_update_cosine": update_metrics["cosine"],
                    "future_recovery": future_metrics[
                        "positive_step_line_recovery"
                    ],
                    "future_cosine": future_metrics["cosine"],
                    "future_chord_fro": future_metrics[
                        "target_chord_fro"
                    ],
                    "exact_direction_future_recovery": exact_future[
                        "positive_step_line_recovery"
                    ],
                    "exact_direction_future_cosine": exact_future[
                        "cosine"
                    ],
                }
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
                del predicted_update
        del source, target, future_chord, state, directions, requested_update
        if "cuda" in args.device:
            torch.cuda.empty_cache()

    aggregates, decision = aggregate_rows(
        rows,
        stage_counts,
        minimum_future_recovery=args.minimum_future_recovery,
        minimum_future_over_random=args.minimum_future_over_random,
        minimum_update_over_coordinate=(
            args.minimum_update_over_coordinate
        ),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    detail_path = args.output / "muon_matched_givens_pilot.csv"
    matching_path = args.output / "muon_matched_givens_matchings.csv"
    aggregate_path = (
        args.output / "muon_matched_givens_pilot_aggregate.json"
    )
    write_csv(detail_path, rows)
    write_csv(matching_path, matching_rows)
    aggregate_path.write_text(
        json.dumps(
            {"decision": decision, "stage_results": aggregates},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_mlp_muon_matched_givens_v1",
        "decision": decision,
        "decision_rule": (
            "at the largest registered stage count require future recovery "
            f">={args.minimum_future_recovery}, recovery over random "
            f">={args.minimum_future_over_random}x, requested-update "
            "recovery over coordinate fraction "
            f">={args.minimum_update_over_coordinate}x, and positive future "
            "cosine in every pilot cell"
        ),
        "causal_protocol": (
            "connectivity and angles use only the exact current coherent "
            "Muon update; future phase chords are used only for scoring"
        ),
        "cells": [
            {"layer": layer, "phase_start": phase}
            for layer, phase in cells
        ],
        "phase_boundaries": boundaries,
        "stage_counts": stage_counts,
        "neighbors": args.neighbors,
        "fit_steps": args.fit_steps,
        "learning_rate": args.learning_rate,
        "matching_seed": args.matching_seed,
        "random_seed": args.random_seed,
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
        "inputs": {
            "snapshots": [
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
                for path in snapshot_paths
            ],
            "optimizer_probes": [
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
                for path in probe_paths
            ],
        },
        "outputs": {
            "detail_sha256": file_sha256(detail_path),
            "matchings_sha256": file_sha256(matching_path),
            "aggregate_sha256": file_sha256(aggregate_path),
        },
        "limitations": [
            "This pilot tests four preregistered layer/phase cells before an all-cell oracle.",
            "Integer matching selection is discrete and is not yet implemented in language-model training.",
            "The fit is a finite-step causal approximation to one exact dense-Muon update, not an endpoint fit.",
        ],
    }
    metadata_path = (
        args.output / "muon_matched_givens_pilot_metadata.json"
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": decision,
                "stage_results": aggregates,
                "metadata": str(metadata_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
