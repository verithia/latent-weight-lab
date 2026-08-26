#!/usr/bin/env python3
"""Analyze a high-cadence MLP update span and compact tangent controls.

The input is a same-run sequence of selected dense MLP weight snapshots.  The
script treats consecutive weight differences as *realized optimizer updates*;
it never calls them raw gradients.  It reports centered/uncentered temporal
spectra, chronological basis transfer, phase-mean drift, and exact Euclidean
capture by the production-style fixed BlockFHT tangent at explicit coordinate
budgets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_mlp_chart_fit import (
    TARGET_SEED_OFFSETS,
    exact_block_fht_projection,
    resolved_latent_dim,
)
from examples.nanogpt.analyze_parameter_trajectory import (
    PARAMETER_PATTERN,
    energy_dimension,
    load_snapshots,
    parse_int_list,
    write_csv,
)
from examples.nanogpt.analyze_mlp_tangent_drift import temporal_basis


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


def parse_float_list(value: str) -> list[float]:
    result = [float(item) for item in value.split(",") if item]
    if (
        not result
        or any(not math.isfinite(item) or item <= 0.0 or item > 1.0 for item in result)
        or result != sorted(set(result))
    ):
        raise ValueError("ratios must be ordered unique finite values in (0,1]")
    return result


def gram_spectrum(rows: torch.Tensor, *, centered: bool) -> torch.Tensor:
    if rows.ndim != 2 or rows.shape[0] < 2:
        raise ValueError("spectrum requires at least two row vectors")
    work = rows - rows.mean(dim=0, keepdim=True) if centered else rows
    gram = work @ work.T
    gram = ((gram + gram.T) * 0.5).double()
    values = torch.linalg.eigvalsh(gram).flip(0).clamp_min(0.0)
    return values


def spectrum_record(rows: torch.Tensor, *, centered: bool) -> dict[str, Any]:
    values = gram_spectrum(rows, centered=centered)
    total = values.sum().clamp_min(1e-30)
    positive = values[values > values[0].clamp_min(1e-30) * 1e-10]
    participation = total.square() / values.square().sum().clamp_min(1e-30)
    return {
        "centered": centered,
        "sample_count": rows.shape[0],
        "sampled_rank_upper_bound": rows.shape[0] - (1 if centered else 0),
        "numerical_rank": int(positive.numel()),
        "pc1_energy": float(values[0] / total),
        "dimension_90pct": energy_dimension(values, 0.90),
        "dimension_95pct": energy_dimension(values, 0.95),
        "dimension_99pct": energy_dimension(values, 0.99),
        "participation_dimension": float(participation),
        "total_energy": float(total),
    }


def energy_capture(rows: torch.Tensor, basis: torch.Tensor) -> float:
    denominator = rows.double().square().sum().clamp_min(1e-30)
    coefficients = rows @ basis
    return float(coefficients.double().square().sum() / denominator)


def chronological_splits(
    start_steps: list[int],
    *,
    discovery_stop: int,
    validation_stop: int,
) -> dict[str, list[int]]:
    result = {
        "discovery": [i for i, step in enumerate(start_steps) if step < discovery_stop],
        "validation": [
            i
            for i, step in enumerate(start_steps)
            if discovery_stop <= step < validation_stop
        ],
        "test": [i for i, step in enumerate(start_steps) if step >= validation_stop],
    }
    if any(len(indices) < 2 for indices in result.values()):
        raise ValueError(f"chronological split is empty or degenerate: {result}")
    return result


def phase_mean_rows(
    rows: torch.Tensor,
    splits: dict[str, list[int]],
) -> list[dict[str, Any]]:
    means = {
        name: rows[indices].mean(dim=0)
        for name, indices in splits.items()
    }
    result: list[dict[str, Any]] = []
    order = ["discovery", "validation", "test"]
    for left_name, right_name in zip(order[:-1], order[1:], strict=True):
        left = means[left_name]
        right = means[right_name]
        cosine = torch.nn.functional.cosine_similarity(
            left.unsqueeze(0), right.unsqueeze(0), dim=1, eps=1e-30
        )[0]
        result.append(
            {
                "left_split": left_name,
                "right_split": right_name,
                "left_mean_norm": float(left.norm()),
                "right_mean_norm": float(right.norm()),
                "mean_cosine": float(cosine),
                "mean_shift_over_left_norm": float(
                    (right - left).norm() / left.norm().clamp_min(1e-30)
                ),
            }
        )
    return result


def blockfht_capture(
    rows: torch.Tensor,
    *,
    ratio: float,
    latent_rows: int,
    layers: int,
    seed: int,
) -> dict[str, Any]:
    latent_dim, latent_shape = resolved_latent_dim(rows.shape[1], ratio, latent_rows)
    projection_energy = 0.0
    total_energy = 0.0
    for row in rows:
        metrics = exact_block_fht_projection(
            row,
            latent_dim=latent_dim,
            latent_shape=latent_shape,
            layers=layers,
            seed=seed,
        )
        projection_energy += float(metrics["projection_energy"])
        total_energy += float(metrics["delta_energy"])
    fraction = projection_energy / max(total_energy, 1e-30)
    resolved_ratio = latent_dim / rows.shape[1]
    return {
        "latent_ratio_requested": ratio,
        "latent_ratio_resolved": resolved_ratio,
        "latent_dim": latent_dim,
        "projection_energy_fraction": fraction,
        "haar_expected_fraction": resolved_ratio,
        "enrichment_over_coordinate_fraction": fraction / max(resolved_ratio, 1e-30),
    }


def analyze_parameter(
    *,
    name: str,
    steps: list[int],
    tensors: list[torch.Tensor],
    discovery_stop: int,
    validation_stop: int,
    ranks: list[int],
    ratios: list[float],
    latent_rows: int,
    block_fht_layers: int,
    block_fht_seed: int,
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    match = PARAMETER_PATTERN.match(name)
    if match is None:
        raise ValueError(f"unsupported parameter: {name}")
    positions = torch.stack(tensors).to(device=device, dtype=torch.float32).flatten(1)
    updates = positions[1:] - positions[:-1]
    start_steps = steps[:-1]
    splits = chronological_splits(
        start_steps,
        discovery_stop=discovery_stop,
        validation_stop=validation_stop,
    )
    common = {
        "parameter": name,
        "layer": int(match.group("layer")),
        "target": match.group("target"),
    }

    total_energy = updates.double().square().sum().clamp_min(1e-30)
    mean = updates.mean(dim=0)
    mean_energy_fraction = float(updates.shape[0] * mean.double().square().sum() / total_energy)
    spectra = [
        {**common, "trajectory": "realized_updates", **spectrum_record(updates, centered=False), "mean_update_energy_fraction": mean_energy_fraction},
        {**common, "trajectory": "realized_updates", **spectrum_record(updates, centered=True), "mean_update_energy_fraction": mean_energy_fraction},
    ]

    discovery = updates[splits["discovery"]]
    _centered, eigenvalues, basis = temporal_basis(
        discovery,
        maximum_rank=max(ranks),
    )
    available = basis.shape[1]
    basis_rows: list[dict[str, Any]] = []
    for rank in ranks:
        if rank > available:
            continue
        centered_selected = basis[:, :rank]
        mean_column = discovery.mean(dim=0).unsqueeze(1)
        if rank == 1:
            affine_selected = mean_column / mean_column.norm().clamp_min(1e-30)
        else:
            affine_selected = torch.linalg.qr(
                torch.cat((mean_column, basis[:, : rank - 1]), dim=1),
                mode="reduced",
            ).Q
        for basis_source, selected in (
            ("discovery_centered_realized_updates", centered_selected),
            ("discovery_mean_plus_centered_updates", affine_selected),
        ):
            for split_name, indices in splits.items():
                basis_rows.append(
                    {
                        **common,
                        "basis_source": basis_source,
                        "rank": rank,
                        "split": split_name,
                        "sample_count": len(indices),
                        "energy_capture": energy_capture(updates[indices], selected),
                        "discovery_centered_eigen_energy": float(
                            eigenvalues[:rank].sum()
                            / eigenvalues.sum().clamp_min(1e-30)
                        ),
                    }
                )

    mean_rows = [{**common, **row} for row in phase_mean_rows(updates, splits)]
    block_rows: list[dict[str, Any]] = []
    seed = block_fht_seed + int(match.group("layer")) * 4 + TARGET_SEED_OFFSETS[match.group("target")]
    for split_name, indices in splits.items():
        raw = updates[indices]
        centered = raw - raw.mean(dim=0, keepdim=True)
        for ratio in ratios:
            for centered_flag, selected_rows in ((False, raw), (True, centered)):
                block_rows.append(
                    {
                        **common,
                        "split": split_name,
                        "centered": centered_flag,
                        "seed": seed,
                        **blockfht_capture(
                            selected_rows,
                            ratio=ratio,
                            latent_rows=latent_rows,
                            layers=block_fht_layers,
                            seed=seed,
                        ),
                    }
                )

    del positions, updates, basis
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return spectra, basis_rows, mean_rows, block_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", default="")
    parser.add_argument("--targets", default="mlp.c_fc,mlp.c_proj")
    parser.add_argument("--discovery-stop", type=int, default=119)
    parser.add_argument("--validation-stop", type=int, default=179)
    parser.add_argument("--ranks", default="1,2,4,8,16,32,64,96,119")
    parser.add_argument("--latent-ratios", default="0.001,0.0025,0.005,0.01")
    parser.add_argument("--latent-rows", type=int, default=0)
    parser.add_argument("--block-fht-layers", type=int, default=2)
    parser.add_argument("--block-fht-seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()

    ranks = parse_int_list(args.ranks)
    if not ranks or any(rank <= 0 for rank in ranks) or ranks != sorted(set(ranks)):
        raise ValueError("ranks must be ordered unique positive integers")
    ratios = parse_float_list(args.latent_ratios)
    paths = sorted(args.snapshot_dir.glob("step_*.pt"))
    layers = parse_int_list(args.layers)
    targets = {item for item in args.targets.split(",") if item}
    steps, values, snapshot_metadata = load_snapshots(
        paths,
        layers=set(layers) if layers else None,
        targets=targets if targets else None,
    )
    if not (steps[0] < args.discovery_stop < args.validation_stop <= steps[-1]):
        raise ValueError("invalid chronological split boundaries")

    spectra: list[dict[str, Any]] = []
    basis_rows: list[dict[str, Any]] = []
    mean_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    for name, tensors in sorted(values.items()):
        result = analyze_parameter(
            name=name,
            steps=steps,
            tensors=tensors,
            discovery_stop=args.discovery_stop,
            validation_stop=args.validation_stop,
            ranks=ranks,
            ratios=ratios,
            latent_rows=args.latent_rows,
            block_fht_layers=args.block_fht_layers,
            block_fht_seed=args.block_fht_seed,
            device=args.device,
        )
        spectra.extend(result[0])
        basis_rows.extend(result[1])
        mean_rows.extend(result[2])
        block_rows.extend(result[3])

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "realized_update_spectrum.csv", spectra)
    write_csv(args.output / "discovery_basis_transfer.csv", basis_rows)
    write_csv(args.output / "phase_mean_drift.csv", mean_rows)
    write_csv(args.output / "fixed_blockfht_capture.csv", block_rows)

    script = Path(__file__).resolve()
    repo = script.parents[2]
    metadata = {
        "schema_version": "nanogpt_mlp_highcadence_basis_v1",
        "snapshot_metadata": snapshot_metadata,
        "steps": steps,
        "parameters": sorted(values),
        "split": {
            "discovery_update_start_steps": [steps[0], args.discovery_stop - 1],
            "validation_update_start_steps": [args.discovery_stop, args.validation_stop - 1],
            "test_update_start_steps": [args.validation_stop, steps[-1] - 1],
        },
        "ranks": ranks,
        "latent_ratios": ratios,
        "block_fht": {
            "layers": args.block_fht_layers,
            "base_seed": args.block_fht_seed,
            "latent_rows": args.latent_rows,
        },
        "method": {
            "samples": "exact consecutive dense weight differences; realized optimizer updates, not raw gradients",
            "spectra": "exact temporal Gram eigenspectra in float64 after float32 tensor loading",
            "basis_transfer": "discovery-only centered spatial PCs frozen before validation/test projection",
            "fixed_blockfht": "exact Euclidean projection energy through the repeated orthogonal BlockFHT adjoint",
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
        "limitations": [
            "One optimizer trajectory is not the global manifold of good solutions.",
            "The observed rank remains bounded by the number of chronological samples.",
            "Realized updates mix gradient, momentum, Muon polar normalization, weight decay, LR, and stochastic batches.",
            "Euclidean tangent capture is necessary but not sufficient for functional or loss parity.",
        ],
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "snapshots": len(steps),
                "parameters": len(values),
                "spectra": len(spectra),
                "basis_rows": len(basis_rows),
                "blockfht_rows": len(block_rows),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
