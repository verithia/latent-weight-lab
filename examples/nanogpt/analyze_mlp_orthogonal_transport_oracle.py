#!/usr/bin/env python3
"""Measure exact one- and two-sided orthogonal transport oracles for c_proj."""

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
from examples.nanogpt.analyze_parameter_trajectory import (
    load_snapshots,
    parse_int_list,
    write_csv,
)


def orthogonal_transport_metrics(
    source: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    """Return exact Procrustes-orbit endpoint recovery.

    ``left`` fits ``L @ source``, ``right`` fits ``source @ R``, and
    ``bilateral`` fits ``L @ source @ R`` for orthogonal matrices.  The
    bilateral residual is exactly the squared distance between singular-value
    vectors; no dense right rotation is materialized.
    """
    if source.ndim != 2 or source.shape != target.shape:
        raise ValueError("source and target must be same-shaped matrices")
    source = source.to(dtype=torch.float64)
    target = target.to(dtype=torch.float64)
    chord_energy = (target - source).square().sum()
    if float(chord_energy) <= 0.0:
        raise ValueError("source and target must differ")
    source_energy = source.square().sum()
    target_energy = target.square().sum()

    left_cross = target @ source.T
    left_nuclear = torch.linalg.svdvals(left_cross).sum()
    left_residual = (
        source_energy + target_energy - 2.0 * left_nuclear
    ).clamp_min(0.0)

    source_left, source_singular, _ = torch.linalg.svd(
        source, full_matrices=False
    )
    target_left, target_singular, _ = torch.linalg.svd(
        target, full_matrices=False
    )
    compressed_right_cross = (
        source_singular[:, None]
        * (source_left.T @ target_left)
        * target_singular[None, :]
    )
    right_nuclear = torch.linalg.svdvals(
        compressed_right_cross
    ).sum()
    right_residual = (
        source_energy + target_energy - 2.0 * right_nuclear
    ).clamp_min(0.0)

    bilateral_residual = (
        target_singular - source_singular
    ).square().sum()

    def recovery(residual: torch.Tensor) -> float:
        return float((1.0 - residual / chord_energy).clamp(max=1.0))

    return {
        "chord_fro": float(chord_energy.sqrt()),
        "left_residual_fro": float(left_residual.sqrt()),
        "right_residual_fro": float(right_residual.sqrt()),
        "bilateral_residual_fro": float(bilateral_residual.sqrt()),
        "left_endpoint_recovery": recovery(left_residual),
        "right_endpoint_recovery": recovery(right_residual),
        "bilateral_endpoint_recovery": recovery(bilateral_residual),
        "singular_value_drift_fraction": float(
            bilateral_residual / chord_energy
        ),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    energy = torch.tensor(
        [float(row["chord_fro"]) ** 2 for row in rows],
        dtype=torch.float64,
    )

    def weighted(key: str) -> float:
        values = torch.tensor(
            [float(row[key]) for row in rows], dtype=torch.float64
        )
        return float((energy * values).sum() / energy.sum())

    result: dict[str, Any] = {"cells": len(rows)}
    for side in ("left", "right", "bilateral"):
        key = f"{side}_endpoint_recovery"
        values = [float(row[key]) for row in rows]
        result[f"energy_weighted_{key}"] = weighted(key)
        result[f"minimum_{key}"] = min(values)
        result[f"maximum_{key}"] = max(values)
    result["energy_weighted_singular_value_drift_fraction"] = weighted(
        "singular_value_drift_fraction"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", default="0,3,6,9,11")
    parser.add_argument(
        "--phase-boundaries", default="0,60,120,180,238"
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
    ):
        raise ValueError("invalid layers or phase boundaries")
    paths = [
        args.snapshot_dir / f"step_{step:06d}.pt"
        for step in boundaries
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"required snapshots are absent: {missing}")
    steps, values, snapshot_metadata = load_snapshots(
        paths,
        layers=set(layers),
        targets={"mlp.c_proj"},
    )
    if steps != boundaries:
        raise ValueError("loaded steps do not match phase boundaries")

    rows: list[dict[str, Any]] = []
    for name, tensors in sorted(values.items()):
        layer = int(name.split(".")[2])
        for index, (start, end) in enumerate(
            zip(boundaries[:-1], boundaries[1:], strict=True)
        ):
            row = {
                "parameter": name,
                "layer": layer,
                "phase_start": start,
                "phase_end": end,
                **orthogonal_transport_metrics(
                    tensors[index].to(args.device),
                    tensors[index + 1].to(args.device),
                ),
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    aggregate = aggregate_rows(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    detail_path = args.output / "orthogonal_transport_oracle.csv"
    aggregate_path = (
        args.output / "orthogonal_transport_oracle_aggregate.json"
    )
    write_csv(detail_path, rows)
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    script = Path(__file__).resolve()
    repo = script.parents[2]
    metadata = {
        "schema_version": "nanogpt_mlp_orthogonal_transport_oracle_v1",
        "snapshot_metadata": snapshot_metadata,
        "layers": layers,
        "phase_boundaries": boundaries,
        "target": "mlp.c_proj",
        "analysis_dtype": "float64",
        "oracles": {
            "left": "minimum ||target-L@source|| over orthogonal L",
            "right": "minimum ||target-source@R|| over orthogonal R",
            "bilateral": "minimum ||target-L@source@R|| over orthogonal L,R",
        },
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
            "These are dense orthogonal-orbit upper bounds, not compact implementations.",
            "Endpoint recovery does not prove that gradient training can follow the orbit.",
            "Only five preregistered representative layers are analyzed.",
        ],
    }
    metadata_path = (
        args.output / "orthogonal_transport_oracle_metadata.json"
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "rows": len(rows),
                "aggregate": aggregate,
                "metadata": str(metadata_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
