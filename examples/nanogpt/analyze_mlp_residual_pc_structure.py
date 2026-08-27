#!/usr/bin/env python3
"""Measure matrix structure of high-cadence MLP residual PCs."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_mlp_highcadence_basis import file_sha256
from examples.nanogpt.analyze_mlp_residual_qtt_basis import residual_temporal_basis
from examples.nanogpt.analyze_parameter_trajectory import (
    PARAMETER_PATTERN,
    load_snapshots,
    write_csv,
)


RANKS = (1, 2, 4, 6, 8, 12, 16, 32, 64)
FRACTIONS = (0.001, 0.0025, 0.005, 0.01)
ENERGY_THRESHOLDS = (0.5, 0.9, 0.95, 0.99)


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def top_fraction_capture(energy: torch.Tensor, fraction: float) -> float:
    flat = energy.reshape(-1).double()
    count = max(1, math.ceil(fraction * flat.numel()))
    total = flat.sum().clamp_min(1e-30)
    return float(torch.topk(flat, count, sorted=False).values.sum() / total)


def matrix_structure_metrics(matrix: torch.Tensor) -> dict[str, float | int]:
    matrix = matrix.float()
    singular_values = torch.linalg.svdvals(matrix)
    singular_energy = singular_values.double().square()
    total = singular_energy.sum().clamp_min(1e-30)
    probabilities = singular_energy / total
    cumulative = probabilities.cumsum(0)
    metrics: dict[str, float | int] = {
        "stable_rank": float(total / singular_energy[0].clamp_min(1e-30)),
        "entropy_rank": float(
            torch.exp(
                -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
            )
        ),
    }
    for threshold in ENERGY_THRESHOLDS:
        rank = int(
            torch.searchsorted(
                cumulative,
                torch.tensor(threshold, device=cumulative.device),
            ).item()
            + 1
        )
        metrics[f"rank_{int(threshold * 100)}"] = rank
    for rank in RANKS:
        kept = min(rank, singular_energy.numel())
        metrics[f"rank{rank}_capture"] = float(singular_energy[:kept].sum() / total)

    entry_energy = matrix.double().square()
    row_energy = entry_energy.sum(dim=1)
    column_energy = entry_energy.sum(dim=0)
    for fraction in FRACTIONS:
        suffix = str(fraction).replace("0.", "p")
        metrics[f"entry_{suffix}_capture"] = top_fraction_capture(
            entry_energy, fraction
        )
        metrics[f"row_{suffix}_capture"] = top_fraction_capture(
            row_energy, fraction
        )
        metrics[f"column_{suffix}_capture"] = top_fraction_capture(
            column_energy, fraction
        )
    return metrics


def aggregate_metrics(
    rows: list[dict[str, Any]], probabilities: torch.Tensor
) -> list[dict[str, Any]]:
    excluded = {"parameter", "target", "pc", "variance_weight"}
    metric_names = [name for name in rows[0] if name not in excluded]
    weights = probabilities.double().cpu()
    summaries: list[dict[str, Any]] = []
    for name in metric_names:
        values = torch.tensor(
            [float(row[name]) for row in rows], dtype=torch.float64
        )
        summaries.append(
            {
                "parameter": rows[0]["parameter"],
                "target": rows[0]["target"],
                "metric": name,
                "weighted": float((weights * values).sum()),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--targets", default="mlp.c_fc,mlp.c_proj")
    parser.add_argument("--basis-rank", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    targets = {item for item in args.targets.split(",") if item}
    paths = sorted(args.snapshot_dir.glob("step_*.pt"))
    steps, values, snapshot_metadata = load_snapshots(
        paths, layers={args.layer}, targets=targets
    )
    all_rows: list[dict[str, Any]] = []
    all_summaries: list[dict[str, Any]] = []
    retained_fractions: dict[str, float] = {}
    for parameter, tensors in sorted(values.items()):
        match = PARAMETER_PATTERN.match(parameter)
        if match is None:
            raise ValueError(f"unsupported parameter {parameter}")
        target = match.group("target")
        positions = torch.stack(tensors).to(args.device, dtype=torch.float32)
        _residuals, eigenvalues, basis = residual_temporal_basis(
            positions, maximum_rank=args.basis_rank
        )
        retained = eigenvalues[: basis.shape[1]]
        probabilities = retained / retained.sum().clamp_min(1e-30)
        retained_fractions[parameter] = float(
            retained.sum() / eigenvalues.sum().clamp_min(1e-30)
        )
        basis_matrices = basis.T.reshape(basis.shape[1], *positions.shape[1:])
        parameter_rows: list[dict[str, Any]] = []
        for index, matrix in enumerate(basis_matrices):
            row = {
                "parameter": parameter,
                "target": target,
                "pc": index + 1,
                "variance_weight": float(probabilities[index]),
                **matrix_structure_metrics(matrix),
            }
            parameter_rows.append(row)
            all_rows.append(row)
        all_summaries.extend(aggregate_metrics(parameter_rows, probabilities))

    args.output.mkdir(parents=True, exist_ok=False)
    pc_path = args.output / "pc_structure.csv"
    summary_path = args.output / "summary.csv"
    write_csv(pc_path, all_rows)
    write_csv(summary_path, all_summaries)
    metadata = {
        "schema_version": "nanogpt_mlp_residual_pc_structure_v1",
        "analysis_execution": {
            "command": [str(Path(sys.argv[0]).resolve()), *sys.argv[1:]],
            "git_commit": git_commit(Path(__file__).resolve().parents[2]),
            "started_at_unix": started,
            "finished_at_unix": time.time(),
        },
        "snapshot_metadata": snapshot_metadata,
        "steps": steps,
        "layer": args.layer,
        "targets": sorted(targets),
        "basis_rank": args.basis_rank,
        "retained_residual_energy_fraction": retained_fractions,
        "ranks": list(RANKS),
        "fractions": list(FRACTIONS),
        "energy_thresholds": list(ENERGY_THRESHOLDS),
        "interpretation": {
            "per_pc_upper_bound": True,
            "does_not_store_or_authorize_basis": True,
            "rank6_state_fraction": 23040 / (3072 * 768),
        },
        "outputs": {
            pc_path.name: file_sha256(pc_path),
            summary_path.name: file_sha256(summary_path),
        },
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"metadata": str(metadata_path), "rows": len(all_rows)}))


if __name__ == "__main__":
    main()
