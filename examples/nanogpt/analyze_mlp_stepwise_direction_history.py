#!/usr/bin/env python3
"""Resolve the causal 1-12 update direction horizon of dense Muon c_proj."""

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
from examples.nanogpt.analyze_mlp_phase_leading_update import (
    leading_direction_metrics,
)
from examples.nanogpt.analyze_parameter_trajectory import (
    load_snapshots,
    parse_int_list,
    write_csv,
)


def history_span_metrics(
    history: torch.Tensor,
    target: torch.Tensor,
    *,
    relative_rank_tolerance: float = 1e-10,
) -> dict[str, float | int]:
    """Return optimal target capture by the row-span of realized updates."""
    if history.ndim != 2 or target.ndim != 1:
        raise ValueError("history must be rank two and target rank one")
    if history.shape[1] != target.numel() or not history.shape[0]:
        raise ValueError("history and target dimensions are incompatible")
    target_energy = target.square().sum()
    if float(target_energy) <= 0.0:
        raise ValueError("target must be nonzero")
    gram = history @ history.T
    rhs = history @ target
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    maximum = eigenvalues.max().clamp_min(0.0)
    threshold = maximum * relative_rank_tolerance
    retained = eigenvalues > threshold
    rank = int(retained.sum())
    if rank:
        projected_rhs = eigenvectors[:, retained].T @ rhs
        captured_energy = (
            projected_rhs.square() / eigenvalues[retained]
        ).sum()
    else:
        captured_energy = target_energy.new_zeros(())
    capture = (captured_energy / target_energy).clamp(0.0, 1.0)
    return {
        "span_capture": float(capture),
        "history_rank": rank,
        "history_condition_number": (
            float(eigenvalues[retained].max() / eigenvalues[retained].min())
            if rank
            else float("inf")
        ),
    }


def vector_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        raise ValueError("cosine inputs must have matching shapes")
    denominator = left.norm() * right.norm()
    if float(denominator) <= 0.0:
        raise ValueError("cosine inputs must be nonzero")
    return float(torch.dot(left, right) / denominator)


def direction_history_metrics(
    history: torch.Tensor,
    full_chord: torch.Tensor,
    future_chord: torch.Tensor,
) -> dict[str, float | int]:
    if history.ndim != 2:
        raise ValueError("history must be a matrix of flattened updates")
    cumulative = history.sum(dim=0)
    full = leading_direction_metrics(
        cumulative.reshape(1, -1),
        full_chord.reshape(1, -1),
    )
    future = leading_direction_metrics(
        cumulative.reshape(1, -1),
        future_chord.reshape(1, -1),
    )
    return {
        "full_chord_fro": full["chord_fro"],
        "future_chord_fro": future["chord_fro"],
        "cumulative_full_cosine": full["cosine"],
        "cumulative_full_capture": full["one_direction_energy_capture"],
        "cumulative_future_cosine": future["cosine"],
        "cumulative_future_capture": future[
            "one_direction_energy_capture"
        ],
        "full_span_capture": history_span_metrics(
            history, full_chord
        )["span_capture"],
        **{
            f"future_{key}": value
            for key, value in history_span_metrics(
                history, future_chord
            ).items()
        },
        "first_to_current_update_cosine": vector_cosine(
            history[0], history[-1]
        ),
        "previous_to_current_update_cosine": (
            1.0
            if history.shape[0] == 1
            else vector_cosine(history[-2], history[-1])
        ),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    for horizon in sorted({int(row["horizon"]) for row in rows}):
        selected = [
            row for row in rows if int(row["horizon"]) == horizon
        ]
        full_energy = torch.tensor(
            [float(row["full_chord_fro"]) ** 2 for row in selected],
            dtype=torch.float64,
        )
        future_energy = torch.tensor(
            [float(row["future_chord_fro"]) ** 2 for row in selected],
            dtype=torch.float64,
        )

        def weighted(key: str, energy: torch.Tensor) -> float:
            values = torch.tensor(
                [float(row[key]) for row in selected],
                dtype=torch.float64,
            )
            return float((values * energy).sum() / energy.sum())

        future_cosines = torch.tensor(
            [
                float(row["cumulative_future_cosine"])
                for row in selected
            ],
            dtype=torch.float64,
        )
        future_span = torch.tensor(
            [float(row["future_span_capture"]) for row in selected],
            dtype=torch.float64,
        )
        aggregates.append(
            {
                "horizon": horizon,
                "cells": len(selected),
                "energy_weighted_cumulative_full_capture": weighted(
                    "cumulative_full_capture", full_energy
                ),
                "energy_weighted_cumulative_full_cosine": weighted(
                    "cumulative_full_cosine", full_energy
                ),
                "energy_weighted_full_span_capture": weighted(
                    "full_span_capture", full_energy
                ),
                "energy_weighted_cumulative_future_capture": weighted(
                    "cumulative_future_capture", future_energy
                ),
                "energy_weighted_cumulative_future_cosine": weighted(
                    "cumulative_future_cosine", future_energy
                ),
                "energy_weighted_future_span_capture": weighted(
                    "future_span_capture", future_energy
                ),
                "minimum_future_span_capture": float(future_span.min()),
                "minimum_cumulative_future_cosine": float(
                    future_cosines.min()
                ),
                "positive_future_cosine_fraction": float(
                    (future_cosines > 0).double().mean()
                ),
                "energy_weighted_first_to_current_update_cosine": weighted(
                    "first_to_current_update_cosine", future_energy
                ),
                "energy_weighted_previous_to_current_update_cosine": weighted(
                    "previous_to_current_update_cosine", future_energy
                ),
            }
        )
    return aggregates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", default="0,3,6,9,11")
    parser.add_argument(
        "--phase-boundaries", default="0,60,120,180,238"
    )
    parser.add_argument("--maximum-horizon", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    layers = parse_int_list(args.layers)
    boundaries = parse_int_list(args.phase_boundaries)
    if (
        not layers
        or len(boundaries) < 2
        or boundaries != sorted(set(boundaries))
        or args.maximum_horizon <= 0
        or any(
            end - start <= args.maximum_horizon
            for start, end in zip(
                boundaries[:-1], boundaries[1:], strict=True
            )
        )
    ):
        raise ValueError("invalid layers, boundaries, or maximum horizon")

    required_steps = set(boundaries)
    for start in boundaries[:-1]:
        required_steps.update(
            range(start + 1, start + args.maximum_horizon + 1)
        )
    ordered_steps = sorted(required_steps)
    paths = [
        args.snapshot_dir / f"step_{step:06d}.pt"
        for step in ordered_steps
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"required snapshots are absent: {missing}")
    steps, values, snapshot_metadata = load_snapshots(
        paths,
        layers=set(layers),
        targets={"mlp.c_proj"},
    )
    if steps != ordered_steps:
        raise ValueError("loaded steps do not match requested steps")
    step_index = {step: index for index, step in enumerate(steps)}

    rows: list[dict[str, Any]] = []
    for name, tensors in sorted(values.items()):
        layer = int(name.split(".")[2])
        for start, end in zip(
            boundaries[:-1], boundaries[1:], strict=True
        ):
            sequence = torch.stack(
                [
                    tensors[step_index[step]].to(
                        device=args.device, dtype=torch.float64
                    )
                    for step in range(
                        start, start + args.maximum_horizon + 1
                    )
                ]
            )
            updates = (sequence[1:] - sequence[:-1]).reshape(
                args.maximum_horizon, -1
            )
            terminal = tensors[step_index[end]].to(
                device=args.device, dtype=torch.float64
            )
            full_chord = (terminal - sequence[0]).reshape(-1)
            for horizon in range(1, args.maximum_horizon + 1):
                future_chord = (
                    terminal - sequence[horizon]
                ).reshape(-1)
                row = {
                    "parameter": name,
                    "layer": layer,
                    "phase_start": start,
                    "phase_end": end,
                    "horizon": horizon,
                    **direction_history_metrics(
                        updates[:horizon],
                        full_chord,
                        future_chord,
                    ),
                }
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
            del sequence, updates, terminal, full_chord

    aggregates = aggregate_rows(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    detail_path = args.output / "stepwise_direction_history.csv"
    aggregate_path = (
        args.output / "stepwise_direction_history_aggregate.csv"
    )
    write_csv(detail_path, rows)
    write_csv(aggregate_path, aggregates)

    script = Path(__file__).resolve()
    repo = script.parents[2]
    metadata = {
        "schema_version": "nanogpt_mlp_stepwise_direction_history_v1",
        "snapshot_metadata": snapshot_metadata,
        "layers": layers,
        "phase_boundaries": boundaries,
        "maximum_horizon": args.maximum_horizon,
        "target": "mlp.c_proj",
        "analysis_dtype": "float64",
        "analysis_execution": {
            "git_commit": git_commit(repo),
            "entrypoint": str(script),
            "entrypoint_sha256": file_sha256(script),
            "command": sys.argv,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "device": args.device,
        },
        "snapshot_files": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in paths
        ],
        "output": {
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
            "The directions are realized Muon updates, not raw gradients.",
            "History-span capture is an in-sample local direction diagnostic, not a trainable-parameter count claim.",
            "Only five preregistered representative layers are analyzed.",
        ],
    }
    metadata_path = (
        args.output / "stepwise_direction_history_metadata.json"
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "rows": len(rows),
                "aggregates": aggregates,
                "metadata": str(metadata_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
