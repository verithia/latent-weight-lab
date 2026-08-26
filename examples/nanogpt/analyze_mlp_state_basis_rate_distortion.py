#!/usr/bin/env python3
"""Measure storage needed to reconstruct empirical dense-MLP temporal PCs.

This deliberately noncausal oracle computes a centered temporal PCA from all
registered states, then asks how much explicit description is needed for its
leading basis matrices under common sparse supports, grouped-Hadamard sparse
supports, independent sparse supports, and independent matrix SVDs.  It does
not train or update a language model.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_mlp_applied_basis_structure import (
    blockwise_fht_2d,
    largest_power_of_two_factor,
)
from examples.nanogpt.analyze_mlp_highcadence_basis import (
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


def minimum_prefix(cumulative: torch.Tensor, threshold: float) -> int:
    if cumulative.ndim != 1 or cumulative.numel() == 0:
        raise ValueError("cumulative energy must be a nonempty vector")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0,1]")
    target = cumulative[-1] * threshold
    return int(torch.searchsorted(cumulative, target, right=False)) + 1


def common_support_frontier(
    basis: torch.Tensor,
    weights: torch.Tensor,
    thresholds: list[float],
) -> dict[float, int]:
    flat = basis.flatten(1).double()
    coordinate_energy = (weights.double().unsqueeze(1) * flat.square()).sum(dim=0)
    cumulative = coordinate_energy.sort(descending=True).values.cumsum(0)
    return {threshold: minimum_prefix(cumulative, threshold) for threshold in thresholds}


def independent_support_frontier(
    basis: torch.Tensor, thresholds: list[float]
) -> dict[float, list[int]]:
    result = {threshold: [] for threshold in thresholds}
    for matrix in basis:
        cumulative = matrix.double().square().flatten().sort(descending=True).values.cumsum(0)
        for threshold in thresholds:
            result[threshold].append(minimum_prefix(cumulative, threshold))
    return result


def independent_svd_frontier(
    basis: torch.Tensor, thresholds: list[float]
) -> dict[float, list[int]]:
    result = {threshold: [] for threshold in thresholds}
    for matrix in basis:
        singular_energy = torch.linalg.svdvals(matrix.float()).double().square()
        cumulative = singular_energy.cumsum(0)
        for threshold in thresholds:
            result[threshold].append(minimum_prefix(cumulative, threshold))
    return result


def analyze_parameter(
    positions: torch.Tensor,
    *,
    parameter: str,
    basis_rank: int,
    thresholds: list[float],
) -> list[dict[str, Any]]:
    match = PARAMETER_PATTERN.match(parameter)
    if match is None:
        raise ValueError(f"unsupported parameter: {parameter}")
    _centered, all_eigenvalues, basis = temporal_basis(
        positions.flatten(1), maximum_rank=basis_rank
    )
    available = min(basis_rank, basis.shape[1])
    eigenvalues = all_eigenvalues[:available].double()
    weights = eigenvalues / eigenvalues.sum().clamp_min(1e-30)
    matrices = basis[:, :available].T.reshape(available, *positions.shape[1:])
    rows, columns = positions.shape[1:]
    size = rows * columns
    index_bits = math.ceil(math.log2(size))
    transformed = blockwise_fht_2d(
        matrices,
        row_block=largest_power_of_two_factor(rows),
        column_block=largest_power_of_two_factor(columns),
    )
    common_ambient = common_support_frontier(matrices, weights, thresholds)
    common_hadamard = common_support_frontier(transformed, weights, thresholds)
    independent_ambient = independent_support_frontier(matrices, thresholds)
    independent_svd = independent_svd_frontier(matrices, thresholds)
    common = {
        "parameter": parameter,
        "layer": int(match.group("layer")),
        "target": match.group("target"),
        "matrix_rows": rows,
        "matrix_columns": columns,
        "matrix_scalars": size,
        "basis_rank": available,
        "basis_variance_fraction": float(
            eigenvalues.sum() / all_eigenvalues.double().sum().clamp_min(1e-30)
        ),
    }
    result: list[dict[str, Any]] = []
    for threshold in thresholds:
        for family, count in (
            ("common_ambient_support", common_ambient[threshold]),
            ("common_grouped_hadamard_support", common_hadamard[threshold]),
        ):
            values = available * count
            packed_bits = values * 16 + count * index_bits
            result.append(
                {
                    **common,
                    "family": family,
                    "target_basis_energy": threshold,
                    "support_coordinates": count,
                    "support_fraction": count / size,
                    "stored_value_scalars": values,
                    "stored_scalar_fraction_of_one_dense_matrix": values / size,
                    "packed_fp16_value_and_index_fraction_of_dense_bf16_bytes": (
                        packed_bits / (size * 16)
                    ),
                    "minimum_matrix_rank": None,
                    "maximum_matrix_rank": None,
                }
            )
        counts = independent_ambient[threshold]
        values = sum(counts)
        packed_bits = values * (16 + index_bits)
        result.append(
            {
                **common,
                "family": "independent_ambient_support_per_pc",
                "target_basis_energy": threshold,
                "support_coordinates": values,
                "support_fraction": values / (available * size),
                "stored_value_scalars": values,
                "stored_scalar_fraction_of_one_dense_matrix": values / size,
                "packed_fp16_value_and_index_fraction_of_dense_bf16_bytes": (
                    packed_bits / (size * 16)
                ),
                "minimum_matrix_rank": None,
                "maximum_matrix_rank": None,
            }
        )
        ranks = independent_svd[threshold]
        factor_values = sum(rank * (rows + columns) for rank in ranks)
        result.append(
            {
                **common,
                "family": "independent_matrix_svd_per_pc",
                "target_basis_energy": threshold,
                "support_coordinates": None,
                "support_fraction": None,
                "stored_value_scalars": factor_values,
                "stored_scalar_fraction_of_one_dense_matrix": factor_values / size,
                "packed_fp16_value_and_index_fraction_of_dense_bf16_bytes": (
                    factor_values / size
                ),
                "minimum_matrix_rank": min(ranks),
                "maximum_matrix_rank": max(ranks),
                "mean_matrix_rank": sum(ranks) / len(ranks),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", default="6")
    parser.add_argument("--targets", default="mlp.c_fc,mlp.c_proj")
    parser.add_argument("--basis-rank", type=int, default=16)
    parser.add_argument("--thresholds", default="0.5,0.75,0.9,0.95,0.99")
    parser.add_argument("--maximum-snapshots", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.basis_rank < 1:
        raise ValueError("basis-rank must be positive")
    thresholds = parse_float_list(args.thresholds)
    paths = sorted(args.snapshot_dir.glob("step_*.pt"))
    if args.maximum_snapshots:
        paths = paths[: args.maximum_snapshots]
    started = time.time()
    steps, values, input_metadata = load_snapshots(
        paths,
        layers=set(parse_int_list(args.layers)),
        targets={item for item in args.targets.split(",") if item},
    )
    rows: list[dict[str, Any]] = []
    for parameter, tensors in sorted(values.items()):
        positions = torch.stack(tensors).to(args.device, dtype=torch.float32)
        rows.extend(
            analyze_parameter(
                positions,
                parameter=parameter,
                basis_rank=args.basis_rank,
                thresholds=thresholds,
            )
        )
        del positions
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / "rate_distortion.csv"
    write_csv(result_path, rows)
    metadata = {
        "schema_version": "nanogpt_mlp_state_basis_rate_distortion_v1",
        "method": "noncausal empirical temporal-PC description-length oracle",
        "steps": steps,
        "basis_rank": args.basis_rank,
        "thresholds": thresholds,
        "input": input_metadata,
        "runtime_seconds": time.time() - started,
        "source_commit": git_commit(Path(__file__).resolve().parents[2]),
        "result_sha256": file_sha256(result_path),
        "limitations": [
            "Full-horizon PCA and all supports are noncausal optimistic oracles.",
            "FP16 value accounting is optimistic and excludes optimizer state.",
            "Independent per-PC supports/factors include their full explicit descriptions.",
        ],
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"metadata_sha256={file_sha256(metadata_path)}")


if __name__ == "__main__":
    main()
