#!/usr/bin/env python3
"""Characterize matrix structure of temporal PCA bases for dense MLP states.

The temporal PCA basis is descriptive and uses the complete registered state
trajectory.  This script does not use it as a causal training basis.  It asks
whether several fast decoder tangent families can represent those empirical
basis matrices at explicit 0.1--1% coordinate budgets.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_mlp_applied_basis_structure import (
    bilateral_diagonal_projection,
    blockwise_fht_2d,
    largest_power_of_two_factor,
    low_rank_for_budget,
)
from examples.nanogpt.analyze_mlp_chart_fit import TARGET_SEED_OFFSETS
from examples.nanogpt.analyze_mlp_highcadence_basis import (
    blockfht_capture,
    file_sha256,
    parse_float_list,
)
from examples.nanogpt.analyze_mlp_tangent_drift import temporal_basis
from examples.nanogpt.analyze_parameter_trajectory import (
    PARAMETER_PATTERN,
    load_snapshots,
    parse_int_list,
    write_csv,
)


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def weighted_support_capture(
    coefficients: torch.Tensor,
    eigenvalues: torch.Tensor,
    coordinates: int,
) -> float:
    flat = coefficients.flatten(1).double()
    weights = eigenvalues.double() / eigenvalues.double().sum().clamp_min(1e-30)
    coordinate_energy = (weights.unsqueeze(1) * flat.square()).sum(dim=0)
    selected = torch.topk(coordinate_energy, coordinates, sorted=False).values
    return float(selected.sum())


def weighted_adaptive_svd_capture(
    basis_matrices: torch.Tensor,
    eigenvalues: torch.Tensor,
    rank: int,
) -> tuple[float, float, float]:
    captures = []
    for matrix in basis_matrices:
        if rank == 0:
            captures.append(0.0)
            continue
        singular_values = torch.linalg.svdvals(matrix.float())
        captures.append(
            float(
                singular_values[:rank].double().square().sum()
                / singular_values.double().square().sum().clamp_min(1e-30)
            )
        )
    values = torch.tensor(
        captures, dtype=torch.float64, device=eigenvalues.device
    )
    weights = eigenvalues.double() / eigenvalues.double().sum().clamp_min(1e-30)
    return float((weights * values).sum()), float(values.min()), float(values.max())


def analyze_parameter(
    positions: torch.Tensor,
    *,
    parameter: str,
    ratios: list[float],
    basis_rank: int,
    block_fht_layers: int,
    block_fht_seed: int,
) -> list[dict[str, Any]]:
    match = PARAMETER_PATTERN.match(parameter)
    if match is None:
        raise ValueError(f"unsupported parameter {parameter}")
    _centered, all_eigenvalues, basis = temporal_basis(
        positions.flatten(1), maximum_rank=basis_rank
    )
    available = min(basis_rank, basis.shape[1])
    eigenvalues = all_eigenvalues[:available]
    basis_matrices = basis[:, :available].T.reshape(
        available, *positions.shape[1:]
    )
    matrix_rows, matrix_columns = positions.shape[1:]
    size = matrix_rows * matrix_columns
    row_block = largest_power_of_two_factor(matrix_rows)
    column_block = largest_power_of_two_factor(matrix_columns)
    transformed = blockwise_fht_2d(
        basis_matrices,
        row_block=row_block,
        column_block=column_block,
    )
    common = {
        "parameter": parameter,
        "layer": int(match.group("layer")),
        "target": match.group("target"),
        "state_basis_rank": available,
        "state_basis_energy_fraction": float(
            eigenvalues.sum()
            / all_eigenvalues.sum().clamp_min(1e-30)
        ),
    }
    rows: list[dict[str, Any]] = []
    base = positions[0]
    bilateral_captures = torch.tensor(
        [
            bilateral_diagonal_projection(base, matrix)
            for matrix in basis_matrices
        ],
        dtype=torch.float64,
        device=eigenvalues.device,
    )
    normalized_eigenvalues = eigenvalues.double() / eigenvalues.double().sum()
    bilateral_weighted = float(
        (normalized_eigenvalues * bilateral_captures).sum()
    )
    bilateral_coordinates = matrix_rows + matrix_columns
    rows.append(
        {
            **common,
            "family": "w0_bilateral_diagonal_tangent",
            "coordinate_ratio_requested": None,
            "coordinates": bilateral_coordinates,
            "coordinate_ratio_resolved": bilateral_coordinates / size,
            "weighted_basis_energy_capture": bilateral_weighted,
            "minimum_basis_capture": float(bilateral_captures.min()),
            "maximum_basis_capture": float(bilateral_captures.max()),
            "total_stored_scalar_fraction_for_all_basis_vectors": (
                bilateral_coordinates / size
            ),
        }
    )

    seed = (
        block_fht_seed
        + int(match.group("layer")) * 4
        + TARGET_SEED_OFFSETS[match.group("target")]
    )
    for ratio in ratios:
        coordinates = max(1, round(size * ratio))
        for family, coefficients in (
            ("learned_ambient_support", basis_matrices),
            ("learned_blockwise_2d_fht_support", transformed),
        ):
            rows.append(
                {
                    **common,
                    "family": family,
                    "coordinate_ratio_requested": ratio,
                    "coordinates": coordinates,
                    "coordinate_ratio_resolved": coordinates / size,
                    "weighted_basis_energy_capture": weighted_support_capture(
                        coefficients, eigenvalues, coordinates
                    ),
                    "minimum_basis_capture": None,
                    "maximum_basis_capture": None,
                    "total_stored_scalar_fraction_for_all_basis_vectors": (
                        coordinates / size
                    ),
                }
            )
        # Scale every orthonormal PC by sqrt(lambda_i) so the aggregate
        # projector energy is the variance-weighted state-basis capture.
        weighted_basis_matrices = basis_matrices * normalized_eigenvalues.sqrt().to(
            basis_matrices.dtype
        ).view(-1, 1, 1)
        fixed = blockfht_capture(
            weighted_basis_matrices.flatten(1),
            ratio=ratio,
            latent_rows=0,
            layers=block_fht_layers,
            seed=seed,
        )
        rows.append(
            {
                **common,
                "family": "fixed_production_blockfht",
                "coordinate_ratio_requested": ratio,
                "coordinates": fixed["latent_dim"],
                "coordinate_ratio_resolved": fixed["latent_ratio_resolved"],
                "weighted_basis_energy_capture": fixed[
                    "projection_energy_fraction"
                ],
                "minimum_basis_capture": None,
                "maximum_basis_capture": None,
                "total_stored_scalar_fraction_for_all_basis_vectors": fixed[
                    "latent_ratio_resolved"
                ],
            }
        )
        rank = low_rank_for_budget(matrix_rows, matrix_columns, ratio)
        weighted, minimum, maximum = weighted_adaptive_svd_capture(
            basis_matrices, eigenvalues, rank
        )
        per_basis_scalars = rank * (matrix_rows + matrix_columns)
        rows.append(
            {
                **common,
                "family": "independent_low_rank_basis_oracle",
                "coordinate_ratio_requested": ratio,
                "coordinates": per_basis_scalars,
                "coordinate_ratio_resolved": per_basis_scalars / size,
                "matrix_rank": rank,
                "weighted_basis_energy_capture": weighted,
                "minimum_basis_capture": minimum,
                "maximum_basis_capture": maximum,
                "total_stored_scalar_fraction_for_all_basis_vectors": (
                    available * per_basis_scalars / size
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", default="6")
    parser.add_argument("--targets", default="mlp.c_fc,mlp.c_proj")
    parser.add_argument("--basis-rank", type=int, default=16)
    parser.add_argument("--ratios", default="0.001,0.0025,0.005,0.01")
    parser.add_argument("--block-fht-layers", type=int, default=2)
    parser.add_argument("--block-fht-seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    layers = set(parse_int_list(args.layers))
    targets = {item for item in args.targets.split(",") if item}
    ratios = parse_float_list(args.ratios)
    if not layers or not targets or args.basis_rank < 1:
        raise ValueError("layers, targets, and positive basis rank are required")
    paths = sorted(args.snapshot_dir.glob("step_*.pt"))
    steps, values, snapshot_metadata = load_snapshots(
        paths, layers=layers, targets=targets
    )
    rows: list[dict[str, Any]] = []
    for parameter, tensors in sorted(values.items()):
        positions = torch.stack(tensors).to(
            device=args.device, dtype=torch.float32
        )
        rows.extend(
            analyze_parameter(
                positions,
                parameter=parameter,
                ratios=ratios,
                basis_rank=args.basis_rank,
                block_fht_layers=args.block_fht_layers,
                block_fht_seed=args.block_fht_seed,
            )
        )
        del positions
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / "state_basis_structure.csv"
    write_csv(result_path, rows)
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_mlp_state_basis_structure_v1",
        "steps": steps,
        "snapshot_metadata": snapshot_metadata,
        "parameters": sorted(values),
        "basis_rank": args.basis_rank,
        "ratios": ratios,
        "method": {
            "temporal_basis": "full-horizon centered PCA; descriptive, never used as causal training evidence",
            "learned_support": "one shared coordinate support selected by eigenvalue-weighted energy across temporal PCs",
            "independent_low_rank": "best SVD truncation of every PC; total all-PC storage is reported separately",
            "bilateral": "projection of each PC onto diag(a)W0 + W0diag(b)",
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
        "output": {
            "path": str(result_path),
            "sha256": file_sha256(result_path),
        },
        "limitations": [
            "The complete-horizon PCA basis is noncausal and optimistic.",
            "Independent low-rank PC fits do not share factors and their summed storage can exceed the nominal per-vector budget.",
            "Euclidean basis energy is not fixed-evaluation CE.",
        ],
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "rows": len(rows),
                "parameters": len(values),
                "metadata": str(metadata_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
