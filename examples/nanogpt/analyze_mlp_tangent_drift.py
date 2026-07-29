#!/usr/bin/env python3
"""Measure phase-to-phase drift of finite-window MLP trajectory tangents.

This analysis treats the leading PCA directions of one short trajectory
window as a finite-window tangent *proxy*.  It then asks whether that proxy
predicts the next window's centered positions, increments, and endpoint
chord.  The calculation is exact in the sampled checkpoint coordinates; it
does not claim to recover a solution manifold or a differential tangent
space.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_parameter_trajectory import (
    PARAMETER_PATTERN,
    load_snapshots,
    parse_int_list,
    write_csv,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def energy_capture(rows: torch.Tensor, basis: torch.Tensor) -> float:
    """Return the fraction of row energy captured by an orthonormal basis."""
    if rows.ndim != 2 or basis.ndim != 2 or rows.shape[1] != basis.shape[0]:
        raise ValueError("rows and basis have incompatible shapes")
    denominator = rows.square().sum()
    if float(denominator) <= 0.0:
        return 1.0
    projection = rows @ basis
    return float(projection.square().sum() / denominator)


def temporal_basis(
    rows: torch.Tensor,
    *,
    maximum_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recover exact spatial PCs from a small temporal Gram matrix."""
    if rows.ndim != 2 or rows.shape[0] < 3:
        raise ValueError("temporal basis requires at least three matrix rows")
    if maximum_rank <= 0:
        raise ValueError("maximum_rank must be positive")
    centered = rows - rows.mean(dim=0, keepdim=True)
    gram = centered @ centered.T
    gram = ((gram + gram.T) * 0.5).double()
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0.0)
    eigenvectors = eigenvectors[:, order]
    threshold = eigenvalues[0].clamp_min(1e-30) * 1e-10
    numerical_rank = int((eigenvalues > threshold).sum())
    usable = min(maximum_rank, numerical_rank)
    if usable == 0:
        basis = torch.empty(
            (rows.shape[1], 0),
            device=rows.device,
            dtype=rows.dtype,
        )
    else:
        basis = (
            centered.T @ eigenvectors[:, :usable].to(centered.dtype)
        ) / eigenvalues[:usable].sqrt().to(centered.dtype).unsqueeze(0)
        # Numerical drift is small, but QR makes the canonical-cosine
        # interpretation exact even for nearly degenerate temporal PCs.
        basis = torch.linalg.qr(basis, mode="reduced").Q
    return centered, eigenvalues, basis


def tangent_pair_metrics(
    *,
    left_centered: torch.Tensor,
    left_basis: torch.Tensor,
    right_rows: torch.Tensor,
    right_centered: torch.Tensor,
    right_basis: torch.Tensor,
    rank: int,
) -> dict[str, float]:
    """Compare equal-rank tangent proxies from adjacent trajectory windows."""
    if rank <= 0 or left_basis.shape[1] < rank or right_basis.shape[1] < rank:
        raise ValueError("requested rank exceeds an available tangent basis")
    left = left_basis[:, :rank]
    right = right_basis[:, :rank]
    canonical_cosines = torch.linalg.svdvals(left.T @ right).clamp(0.0, 1.0)
    angles = torch.rad2deg(torch.acos(canonical_cosines))
    right_increments = right_rows[1:] - right_rows[:-1]
    right_chord = right_rows[-1:] - right_rows[:1]
    left_increments = left_centered[1:] - left_centered[:-1]
    return {
        "mean_squared_canonical_cosine": float(canonical_cosines.square().mean()),
        "mean_canonical_cosine": float(canonical_cosines.mean()),
        "minimum_canonical_cosine": float(canonical_cosines.min()),
        "maximum_principal_angle_degrees": float(angles.max()),
        "mean_principal_angle_degrees": float(angles.mean()),
        "left_self_centered_capture": energy_capture(left_centered, left),
        "left_self_increment_capture": energy_capture(left_increments, left),
        "right_self_centered_capture": energy_capture(right_centered, right),
        "right_prior_centered_capture": energy_capture(right_centered, left),
        "right_prior_increment_capture": energy_capture(right_increments, left),
        "right_prior_chord_capture": energy_capture(right_chord, left),
        "left_future_centered_capture": energy_capture(left_centered, right),
    }


def _window_records(
    *,
    steps: list[int],
    rows: torch.Tensor,
    boundaries: list[int],
    maximum_rank: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        indices = [index for index, step in enumerate(steps) if start <= step <= end]
        if len(indices) < 3:
            raise ValueError(f"window {start}:{end} has fewer than three snapshots")
        window_rows = rows[indices]
        centered, eigenvalues, basis = temporal_basis(
            window_rows,
            maximum_rank=maximum_rank,
        )
        records.append(
            {
                "start": start,
                "end": end,
                "steps": [steps[index] for index in indices],
                "rows": window_rows,
                "centered": centered,
                "increments": window_rows[1:] - window_rows[:-1],
                "eigenvalues": eigenvalues,
                "basis": basis,
            }
        )
    return records


def analyze_parameter(
    *,
    name: str,
    steps: list[int],
    tensors: list[torch.Tensor],
    boundaries: list[int],
    ranks: list[int],
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    match = PARAMETER_PATTERN.match(name)
    if match is None:
        raise ValueError(f"unsupported parameter name: {name}")
    rows = torch.stack(tensors).to(device=device, dtype=torch.float32).flatten(1)
    maximum_rank = max(ranks)
    global_centered, global_eigenvalues, global_basis = temporal_basis(
        rows,
        maximum_rank=maximum_rank,
    )
    records = _window_records(
        steps=steps,
        rows=rows,
        boundaries=boundaries,
        maximum_rank=maximum_rank,
    )
    common = {
        "parameter": name,
        "layer": int(match.group("layer")),
        "target": match.group("target"),
    }
    window_rows: list[dict[str, Any]] = []
    total_global_energy = global_eigenvalues.sum().clamp_min(1e-30)
    for record in records:
        local_energy = record["eigenvalues"].sum().clamp_min(1e-30)
        for rank in ranks:
            if record["basis"].shape[1] < rank or global_basis.shape[1] < rank:
                raise ValueError(f"rank {rank} is unavailable for {name}")
            local = record["basis"][:, :rank]
            global_rank_basis = global_basis[:, :rank]
            window_rows.append(
                {
                    **common,
                    "window_start": record["start"],
                    "window_end": record["end"],
                    "window_snapshot_count": len(record["steps"]),
                    "rank": rank,
                    "local_eigen_energy": float(
                        record["eigenvalues"][:rank].sum() / local_energy
                    ),
                    "global_eigen_energy": float(
                        global_eigenvalues[:rank].sum() / total_global_energy
                    ),
                    "local_centered_capture": energy_capture(record["centered"], local),
                    "local_increment_capture": energy_capture(record["increments"], local),
                    "global_centered_capture": energy_capture(
                        record["centered"],
                        global_rank_basis,
                    ),
                    "global_increment_capture": energy_capture(
                        record["increments"],
                        global_rank_basis,
                    ),
                    "global_chord_capture": energy_capture(
                        record["rows"][-1:] - record["rows"][:1],
                        global_rank_basis,
                    ),
                }
            )

    pair_rows: list[dict[str, Any]] = []
    for left, right in zip(records[:-1], records[1:], strict=True):
        for rank in ranks:
            pair_rows.append(
                {
                    **common,
                    "left_window_start": left["start"],
                    "left_window_end": left["end"],
                    "right_window_start": right["start"],
                    "right_window_end": right["end"],
                    "rank": rank,
                    **tangent_pair_metrics(
                        left_centered=left["centered"],
                        left_basis=left["basis"],
                        right_rows=right["rows"],
                        right_centered=right["centered"],
                        right_basis=right["basis"],
                        rank=rank,
                    ),
                }
            )
    del rows, global_centered, global_basis, records
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return window_rows, pair_rows


def aggregate_rows(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_fields = [
        "mean_squared_canonical_cosine",
        "mean_canonical_cosine",
        "minimum_canonical_cosine",
        "maximum_principal_angle_degrees",
        "right_prior_centered_capture",
        "right_prior_increment_capture",
        "right_prior_chord_capture",
    ]
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in pair_rows:
        transition = f"{row['left_window_start']}-{row['left_window_end']}->{row['right_window_end']}"
        groups.setdefault((str(row["target"]), int(row["rank"]), transition), []).append(row)
        groups.setdefault((str(row["target"]), int(row["rank"]), "all_adjacent"), []).append(row)
    result: list[dict[str, Any]] = []
    for (target, rank, transition), selected in sorted(groups.items()):
        record: dict[str, Any] = {
            "target": target,
            "rank": rank,
            "transition": transition,
            "parameter_count": len(selected),
        }
        for field in metric_fields:
            values = torch.tensor([float(row[field]) for row in selected], dtype=torch.float64)
            record[f"{field}_mean"] = float(values.mean())
            record[f"{field}_median"] = float(values.median())
            record[f"{field}_minimum"] = float(values.min())
            record[f"{field}_maximum"] = float(values.max())
        result.append(record)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", default="")
    parser.add_argument("--targets", default="mlp.c_fc,mlp.c_proj")
    parser.add_argument("--window-boundaries", default="0,60,120,180,238")
    parser.add_argument("--ranks", default="1,2,3")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    ranks = parse_int_list(args.ranks)
    if not ranks or any(rank <= 0 for rank in ranks) or ranks != sorted(set(ranks)):
        raise ValueError("--ranks must be ordered unique positive integers")
    boundaries = parse_int_list(args.window_boundaries)
    paths = sorted(args.snapshot_dir.glob("step_*.pt"))
    layers = parse_int_list(args.layers)
    targets = {item for item in args.targets.split(",") if item}
    steps, values, snapshot_metadata = load_snapshots(
        paths,
        layers=set(layers) if layers else None,
        targets=targets if targets else None,
    )
    if (
        len(boundaries) < 3
        or boundaries != sorted(set(boundaries))
        or boundaries[0] < steps[0]
        or boundaries[-1] > steps[-1]
    ):
        raise ValueError("--window-boundaries must define ordered adjacent windows")
    args.output.mkdir(parents=True, exist_ok=True)

    window_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for name, tensors in sorted(values.items()):
        window, pair = analyze_parameter(
            name=name,
            steps=steps,
            tensors=tensors,
            boundaries=boundaries,
            ranks=ranks,
            device=args.device,
        )
        window_rows.extend(window)
        pair_rows.extend(pair)
    aggregates = aggregate_rows(pair_rows)
    write_csv(args.output / "window_tangent_capture.csv", window_rows)
    write_csv(args.output / "adjacent_tangent_drift.csv", pair_rows)
    write_csv(args.output / "tangent_drift_aggregate.csv", aggregates)

    script = Path(__file__).resolve()
    repo = script.parents[2]
    metadata = {
        "schema_version": "nanogpt_mlp_tangent_drift_v1",
        "snapshot_metadata": snapshot_metadata,
        "steps": steps,
        "parameters": sorted(values),
        "window_boundaries": boundaries,
        "ranks": ranks,
        "method": {
            "tangent_proxy": (
                "top-r exact spatial PCA directions of mean-centered parameter "
                "positions within each finite checkpoint window"
            ),
            "subspace_drift": (
                "canonical cosines/principal angles between equal-rank tangent "
                "proxies from adjacent windows"
            ),
            "predictive_capture": (
                "fraction of next-window centered-position, increment, and chord "
                "energy projected onto the preceding window basis"
            ),
            "global_static_capture": (
                "fraction of each local window's motion captured by a full-trajectory "
                "static PCA basis of equal rank"
            ),
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
            {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in paths
        ],
        "interpretation_limits": [
            "Finite-window PCA is a trajectory-tangent proxy, not a differential manifold tangent.",
            "One optimizer trajectory does not identify the global set of good solutions.",
            "Prior-window capture is retrospective path predictability, not optimization causality.",
            "Raw-coordinate overlap is meaningful only within the same layer and parameter tensor.",
            "Each local PCA rank is bounded by its checkpoint count minus one.",
            "Token Geometry's token-embedding rays are not assumed to transfer to MLP matrices.",
        ],
    }
    metadata_path = args.output / "tangent_drift_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rank2_aggregate = [
        row
        for row in aggregates
        if int(row["rank"]) == 2 and row["transition"] == "all_adjacent"
    ]
    print(
        json.dumps(
            {
                "snapshots": len(steps),
                "parameters": len(values),
                "window_rows": len(window_rows),
                "pair_rows": len(pair_rows),
                "rank2_all_adjacent": rank2_aggregate,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
