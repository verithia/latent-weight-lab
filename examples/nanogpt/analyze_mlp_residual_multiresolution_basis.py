#!/usr/bin/env python3
"""Measure dense MLP residual PCs in fast deterministic non-FHT bases."""
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

from examples.nanogpt.analyze_mlp_highcadence_basis import (
    file_sha256,
    parse_float_list,
)
from examples.nanogpt.analyze_mlp_residual_qtt_basis import residual_temporal_basis
from examples.nanogpt.analyze_parameter_trajectory import (
    PARAMETER_PATTERN,
    load_snapshots,
    write_csv,
)


CANONICAL_SHAPE = (3072, 768)
HAAR_ROW_MAJOR = 3
HAAR_ROW_POWER = 1024
HAAR_COLUMN_MAJOR = 3
HAAR_COLUMN_POWER = 256


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def dct_ii(values: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    """Orthonormal DCT-II using one complex FFT and procedural phases."""
    moved = values.movedim(dim, -1)
    length = moved.shape[-1]
    reordered = torch.cat(
        (moved[..., ::2], moved[..., 1::2].flip(-1)), dim=-1
    )
    spectrum = torch.fft.fft(reordered, dim=-1)
    index = torch.arange(length, device=values.device, dtype=values.dtype)
    angle = -math.pi * index / (2.0 * length)
    result = spectrum.real * angle.cos() - spectrum.imag * angle.sin()
    result[..., 0] /= math.sqrt(length)
    if length > 1:
        result[..., 1:] *= math.sqrt(2.0 / length)
    return result.movedim(-1, dim)


def dct_2d(values: torch.Tensor) -> torch.Tensor:
    return dct_ii(dct_ii(values, dim=-1), dim=-2)


def haar_1d(values: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    """Orthonormal Haar analysis ordered coarse-to-fine along one axis."""
    moved = values.movedim(dim, -1)
    length = moved.shape[-1]
    if length < 1 or length & (length - 1):
        raise ValueError("Haar axis length must be a positive power of two")
    current = moved
    details: list[torch.Tensor] = []
    inverse_sqrt_two = 1.0 / math.sqrt(2.0)
    while current.shape[-1] > 1:
        even = current[..., ::2]
        odd = current[..., 1::2]
        details.append((even - odd) * inverse_sqrt_two)
        current = (even + odd) * inverse_sqrt_two
    result = torch.cat((current, *reversed(details)), dim=-1)
    return result.movedim(-1, dim)


def haar_2d_canonical(values: torch.Tensor) -> torch.Tensor:
    if tuple(values.shape[-2:]) != CANONICAL_SHAPE:
        raise ValueError(f"expected canonical shape {CANONICAL_SHAPE}")
    shaped = values.reshape(
        *values.shape[:-2],
        HAAR_ROW_MAJOR,
        HAAR_ROW_POWER,
        HAAR_COLUMN_MAJOR,
        HAAR_COLUMN_POWER,
    )
    shaped = haar_1d(shaped, dim=-3)
    return haar_1d(shaped, dim=-1)


def canonicalize(values: torch.Tensor) -> tuple[torch.Tensor, bool]:
    shape = tuple(values.shape[-2:])
    if shape == CANONICAL_SHAPE:
        return values, False
    if shape == tuple(reversed(CANONICAL_SHAPE)):
        return values.transpose(-2, -1).contiguous(), True
    raise ValueError(f"unsupported MLP matrix shape {shape}")


def dct_support_order(*, device: torch.device) -> torch.Tensor:
    row = torch.arange(CANONICAL_SHAPE[0], device=device, dtype=torch.float64)
    column = torch.arange(
        CANONICAL_SHAPE[1], device=device, dtype=torch.float64
    )
    score = (row[:, None] / CANONICAL_SHAPE[0]).square() + (
        column[None, :] / CANONICAL_SHAPE[1]
    ).square()
    return torch.argsort(score.reshape(-1), stable=True)


def haar_support_order(*, device: torch.device) -> torch.Tensor:
    row = torch.arange(HAAR_ROW_POWER, device=device, dtype=torch.float64)
    column = torch.arange(
        HAAR_COLUMN_POWER, device=device, dtype=torch.float64
    )
    score = (row / HAAR_ROW_POWER).square().view(1, -1, 1, 1) + (
        column / HAAR_COLUMN_POWER
    ).square().view(1, 1, 1, -1)
    score = score.expand(
        HAAR_ROW_MAJOR,
        HAAR_ROW_POWER,
        HAAR_COLUMN_MAJOR,
        HAAR_COLUMN_POWER,
    ).clone()
    major_row = torch.arange(
        HAAR_ROW_MAJOR, device=device, dtype=torch.float64
    ).view(-1, 1, 1, 1)
    major_column = torch.arange(
        HAAR_COLUMN_MAJOR, device=device, dtype=torch.float64
    ).view(1, 1, -1, 1)
    score += (major_row * HAAR_COLUMN_MAJOR + major_column) * 1e-15
    return torch.argsort(score.reshape(-1), stable=True)


def evaluate_coefficients(
    coefficients: torch.Tensor,
    probabilities: torch.Tensor,
    order: torch.Tensor,
    *,
    family: str,
    budgets: list[float],
    retained_fraction: float,
    expansion_ops: int,
) -> list[dict[str, Any]]:
    flattened = coefficients.reshape(coefficients.shape[0], -1).double()
    total = flattened.square().sum(dim=1).clamp_min(1e-30)
    dense_scalars = flattened.shape[1]
    rows = []
    for budget in budgets:
        stored = max(1, math.floor(budget * dense_scalars))
        selected = order[:stored]
        captures = flattened.index_select(1, selected).square().sum(dim=1) / total
        weighted = torch.sum(probabilities.double() * captures)
        rows.append(
            {
                "family": family,
                "requested_budget_fraction": budget,
                "stored_scalars": stored,
                "stored_scalar_fraction": stored / dense_scalars,
                "weighted_pc_capture": float(weighted),
                "minimum_pc_capture": float(captures.min()),
                "maximum_pc_capture": float(captures.max()),
                "full_residual_recovery": float(weighted) * retained_fraction,
                "expansion_scalar_ops": expansion_ops,
                "expansion_ops_to_dense_scalars": expansion_ops / dense_scalars,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--targets", default="mlp.c_fc,mlp.c_proj")
    parser.add_argument("--basis-rank", type=int, default=16)
    parser.add_argument("--budgets", default="0.001,0.0025,0.005,0.01")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    targets = {item for item in args.targets.split(",") if item}
    budgets = parse_float_list(args.budgets)
    paths = sorted(args.snapshot_dir.glob("step_*.pt"))
    steps, values, snapshot_metadata = load_snapshots(
        paths, layers={args.layer}, targets=targets
    )
    rows: list[dict[str, Any]] = []
    energy_checks: dict[str, dict[str, float]] = {}
    retained_fractions: dict[str, float] = {}
    device = torch.device(args.device)
    dct_order = dct_support_order(device=device)
    haar_order = haar_support_order(device=device)
    dense_scalars = math.prod(CANONICAL_SHAPE)
    dct_ops = 5 * dense_scalars * (
        math.ceil(math.log2(2 * CANONICAL_SHAPE[0]))
        + math.ceil(math.log2(2 * CANONICAL_SHAPE[1]))
    )
    haar_ops = math.ceil(
        4
        * dense_scalars
        * (
            (HAAR_ROW_POWER - 1) / HAAR_ROW_POWER
            + (HAAR_COLUMN_POWER - 1) / HAAR_COLUMN_POWER
        )
    )
    for parameter, tensors in sorted(values.items()):
        match = PARAMETER_PATTERN.match(parameter)
        if match is None:
            raise ValueError(f"unsupported parameter {parameter}")
        positions = torch.stack(tensors).to(device, dtype=torch.float32)
        _residuals, eigenvalues, basis = residual_temporal_basis(
            positions, maximum_rank=args.basis_rank
        )
        retained = eigenvalues[: basis.shape[1]]
        probabilities = retained / retained.sum().clamp_min(1e-30)
        matrices = basis.T.reshape(basis.shape[1], *positions.shape[1:])
        matrices, transposed = canonicalize(matrices)
        retained_fraction = float(
            retained.sum() / eigenvalues.sum().clamp_min(1e-30)
        )
        retained_fractions[parameter] = retained_fraction
        original_energy = matrices.double().square().sum(dim=(-2, -1))
        families = {
            "dct2_low_radial": (dct_2d(matrices), dct_order, dct_ops),
            "haar2_low_sequency": (
                haar_2d_canonical(matrices),
                haar_order,
                haar_ops,
            ),
        }
        energy_checks[parameter] = {}
        for family, (coefficients, order, expansion_ops) in families.items():
            transformed_energy = coefficients.double().square().sum(
                dim=tuple(range(1, coefficients.ndim))
            )
            relative_error = (
                (transformed_energy - original_energy).abs()
                / original_energy.clamp_min(1e-30)
            ).max()
            energy_checks[parameter][family] = float(relative_error)
            if float(relative_error) > 2e-5:
                raise RuntimeError(
                    f"{family} is not numerically orthonormal: {relative_error}"
                )
            family_rows = evaluate_coefficients(
                coefficients,
                probabilities,
                order,
                family=family,
                budgets=budgets,
                retained_fraction=retained_fraction,
                expansion_ops=expansion_ops,
            )
            for row in family_rows:
                row.update(
                    {
                        "parameter": parameter,
                        "target": match.group("target"),
                        "canonical_transpose": transposed,
                    }
                )
            rows.extend(family_rows)
        del positions, matrices
        torch.cuda.empty_cache()

    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / "multiresolution_recovery.csv"
    write_csv(result_path, rows)
    script = Path(__file__).resolve()
    metadata: dict[str, Any] = {
        "schema_version": "nanogpt_mlp_residual_multiresolution_basis_v1",
        "steps": steps,
        "snapshot_metadata": snapshot_metadata,
        "layer": args.layer,
        "targets": sorted(targets),
        "basis_rank": args.basis_rank,
        "budgets": budgets,
        "retained_residual_energy_fraction": retained_fractions,
        "orthonormal_energy_relative_error": energy_checks,
        "candidate_contract": {
            "stored": "only the current selected real transform coefficients",
            "procedural": "DCT phases or Haar pairings and deterministic low-frequency/sequency support",
            "forbidden": "PCA atom, ambient shadow, learned basis, stored support table, or seed search",
        },
        "analysis_execution": {
            "git_commit": git_commit(script.parents[2]),
            "entrypoint": str(script),
            "entrypoint_sha256": file_sha256(script),
            "command": sys.argv,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "device": args.device,
        },
        "outputs": {result_path.name: file_sha256(result_path)},
        "limitations": [
            "This is an optimistic full-horizon representation test, not an online predictor.",
            "Transform expansion cost does not include subsequent dense target-network matmuls.",
        ],
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": rows, "metadata": str(metadata_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
