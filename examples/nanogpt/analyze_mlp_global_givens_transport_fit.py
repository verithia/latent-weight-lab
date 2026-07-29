#!/usr/bin/env python3
"""Fit a compact global Givens flow to dense c_proj phase endpoints."""

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
from examples.nanogpt.analyze_mlp_orthogonal_transport_oracle import (
    orthogonal_transport_metrics,
)
from examples.nanogpt.analyze_parameter_trajectory import (
    load_snapshots,
    parse_int_list,
    write_csv,
)
from examples.nanogpt.model import LearnedGivensOutputMix


def fit_global_givens_transport(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    stages: int,
    seed: int,
    steps: int,
    learning_rate: float,
) -> dict[str, float | int]:
    if source.ndim != 2 or source.shape != target.shape:
        raise ValueError("source and target must be same-shaped matrices")
    if stages <= 0 or steps <= 0 or learning_rate <= 0.0:
        raise ValueError("stages, steps, and learning rate must be positive")
    source = source.float()
    target = target.float()
    chord_energy = (target - source).square().sum().clamp_min(1e-30)
    flow = LearnedGivensOutputMix(
        source.shape[-1], int(stages), int(seed)
    ).to(device=source.device, dtype=torch.float32)
    optimizer = torch.optim.Adam(
        [flow.angles], lr=float(learning_rate)
    )
    best_residual = float("inf")
    best_angles = flow.angles.detach().clone()
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        prediction = flow(source)
        residual = (target - prediction).square().sum()
        loss = residual / chord_energy
        loss.backward()
        optimizer.step()
        observed = float(residual.detach())
        if observed < best_residual:
            best_residual = observed
            best_angles.copy_(flow.angles.detach())
    with torch.no_grad():
        flow.angles.copy_(best_angles)
        prediction = flow(source)
        residual = (target - prediction).square().sum()
        recovery = 1.0 - residual / chord_energy
    return {
        "stages": int(stages),
        "coordinates": int(flow.angles.numel()),
        "coordinate_fraction": float(
            flow.angles.numel() / source.numel()
        ),
        "endpoint_recovery": float(recovery),
        "residual_fro": float(residual.sqrt()),
        "chord_fro": float(chord_energy.sqrt()),
        "angle_rms": float(
            flow.angles.detach().square().mean().sqrt()
        ),
        "angle_max_abs": float(flow.angles.detach().abs().max()),
    }


def parse_cells(value: str) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for raw in value.split(","):
        layer, phase = raw.split(":", maxsplit=1)
        cell = (int(layer), int(phase))
        if min(cell) < 0 or cell in cells:
            raise ValueError("cells must be unique non-negative layer:phase pairs")
        cells.append(cell)
    if not cells:
        raise ValueError("at least one cell is required")
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--cells", default="0:0,0:180,6:60,11:120"
    )
    parser.add_argument(
        "--phase-boundaries", default="0,60,120,180,238"
    )
    parser.add_argument("--stages", default="8,16,32")
    parser.add_argument("--fit-steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    cells = parse_cells(args.cells)
    boundaries = parse_int_list(args.phase_boundaries)
    stage_counts = parse_int_list(args.stages)
    if (
        len(boundaries) < 2
        or boundaries != sorted(set(boundaries))
        or not stage_counts
        or min(stage_counts) <= 0
    ):
        raise ValueError("invalid phase boundaries or stage counts")
    end_by_start = dict(
        zip(boundaries[:-1], boundaries[1:], strict=True)
    )
    if any(phase not in end_by_start for _, phase in cells):
        raise ValueError("each cell phase must be a registered phase start")
    layers = sorted({layer for layer, _ in cells})
    required_steps = sorted(
        {step for _, phase in cells for step in (phase, end_by_start[phase])}
    )
    paths = [
        args.snapshot_dir / f"step_{step:06d}.pt"
        for step in required_steps
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"required snapshots are absent: {missing}")
    steps, values, snapshot_metadata = load_snapshots(
        paths,
        layers=set(layers),
        targets={"mlp.c_proj"},
    )
    if steps != required_steps:
        raise ValueError("loaded steps do not match requested steps")
    step_index = {step: index for index, step in enumerate(steps)}

    rows: list[dict[str, Any]] = []
    for layer, phase_start in cells:
        name = f"transformer.h.{layer}.mlp.c_proj.weight"
        tensors = values[name]
        phase_end = end_by_start[phase_start]
        source = tensors[step_index[phase_start]].to(args.device)
        target = tensors[step_index[phase_end]].to(args.device)
        oracle = orthogonal_transport_metrics(source, target)
        for stages in stage_counts:
            fit = fit_global_givens_transport(
                source,
                target,
                stages=stages,
                seed=args.seed + layer * 64,
                steps=args.fit_steps,
                learning_rate=args.learning_rate,
            )
            row = {
                "parameter": name,
                "layer": layer,
                "phase_start": phase_start,
                "phase_end": phase_end,
                "right_oracle_recovery": oracle[
                    "right_endpoint_recovery"
                ],
                **fit,
            }
            row["fraction_of_right_oracle"] = (
                float(row["endpoint_recovery"])
                / float(row["right_oracle_recovery"])
            )
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
        del source, target

    aggregates: list[dict[str, Any]] = []
    for stages in stage_counts:
        selected = [
            row for row in rows if int(row["stages"]) == stages
        ]
        energy = torch.tensor(
            [float(row["chord_fro"]) ** 2 for row in selected],
            dtype=torch.float64,
        )

        def weighted(key: str) -> float:
            values = torch.tensor(
                [float(row[key]) for row in selected],
                dtype=torch.float64,
            )
            return float((energy * values).sum() / energy.sum())

        recoveries = [
            float(row["endpoint_recovery"]) for row in selected
        ]
        aggregates.append(
            {
                "stages": stages,
                "cells": len(selected),
                "coordinates_per_layer": int(
                    selected[0]["coordinates"]
                ),
                "coordinate_fraction": float(
                    selected[0]["coordinate_fraction"]
                ),
                "energy_weighted_endpoint_recovery": weighted(
                    "endpoint_recovery"
                ),
                "energy_weighted_fraction_of_right_oracle": weighted(
                    "fraction_of_right_oracle"
                ),
                "minimum_endpoint_recovery": min(recoveries),
                "maximum_endpoint_recovery": max(recoveries),
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    detail_path = args.output / "global_givens_transport_fit.csv"
    aggregate_path = (
        args.output / "global_givens_transport_fit_aggregate.csv"
    )
    write_csv(detail_path, rows)
    write_csv(aggregate_path, aggregates)
    script = Path(__file__).resolve()
    repo = script.parents[2]
    metadata = {
        "schema_version": "nanogpt_mlp_global_givens_transport_fit_v1",
        "snapshot_metadata": snapshot_metadata,
        "cells": [
            {"layer": layer, "phase_start": phase}
            for layer, phase in cells
        ],
        "phase_boundaries": boundaries,
        "stage_counts": stage_counts,
        "fit_steps": args.fit_steps,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "analysis_dtype": "float32_fit_fp64_oracle",
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
            "Angles are fitted independently to each dense endpoint; this is an orbit-capacity test, not a language-model training result.",
            "The pilot uses four preregistered phase/layer cells before any all-cell validation.",
            "The fit is finite-step Adam and is a lower bound on each fixed-connectivity flow's best recovery.",
        ],
    }
    metadata_path = (
        args.output / "global_givens_transport_fit_metadata.json"
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
