#!/usr/bin/env python3
"""Fit basis-free QTT/MPO decoders to dense MLP weight residual PCs.

The dense residual PCA is a diagnostic target, never part of the candidate
state.  Each candidate stores only small tensor-train cores and one current
temporal-coordinate vector.  Fixed reshape/permutation layouts are procedural
and cost no persistent scalars.  No ambient-length basis vector is retained.
"""
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
from examples.nanogpt.analyze_parameter_trajectory import (
    PARAMETER_PATTERN,
    load_snapshots,
    parse_int_list,
    write_csv,
)


ROW_FACTORS = (3,) + (2,) * 10
COLUMN_FACTORS = (3,) + (2,) * 8
SUPPORTED_SHAPE = (3072, 768)


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def residual_temporal_basis(
    positions: torch.Tensor,
    *,
    maximum_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return uncentered PCs of D_t = W_t - W_0 using a temporal Gram."""
    if positions.ndim != 3 or positions.shape[0] < 2:
        raise ValueError("positions must have shape [time, rows, columns]")
    residuals = positions - positions[:1]
    rows = residuals.flatten(1)
    gram = rows @ rows.T
    gram = ((gram + gram.T) * 0.5).double()
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0.0)
    eigenvectors = eigenvectors[:, order]
    threshold = eigenvalues[0].clamp_min(1e-30) * 1e-10
    usable = min(maximum_rank, int((eigenvalues > threshold).sum()))
    basis = (
        rows.T @ eigenvectors[:, :usable].to(rows.dtype)
    ) / eigenvalues[:usable].sqrt().to(rows.dtype).unsqueeze(0)
    return residuals, eigenvalues, basis


def tensorize_basis(
    weighted_basis: torch.Tensor,
    *,
    layout: str,
) -> torch.Tensor:
    """Tensorize [pc, 3072, 768] while keeping the PC axis first."""
    if tuple(weighted_basis.shape[-2:]) != SUPPORTED_SHAPE:
        raise ValueError(
            f"expected matrix shape {SUPPORTED_SHAPE}, got "
            f"{tuple(weighted_basis.shape[-2:])}"
        )
    k = weighted_basis.shape[0]
    base = weighted_basis.reshape(k, *ROW_FACTORS, *COLUMN_FACTORS)
    if layout == "row_then_column_binary":
        return base

    row_major = 1
    row_bits = list(range(2, 12))
    column_major = 12
    column_bits = list(range(13, 21))
    if layout == "morton_binary":
        paired_column_bits = column_bits
    elif layout == "morton_reverse_column_bits":
        paired_column_bits = list(reversed(column_bits))
    elif layout == "morton_nibble":
        paired_column_bits = column_bits
    else:
        raise ValueError(f"unknown tensor layout {layout}")

    axes = [0, row_major, column_major]
    for row_axis, column_axis in zip(row_bits[:8], paired_column_bits):
        axes.extend((row_axis, column_axis))
    axes.extend(row_bits[8:])
    morton = base.permute(*axes).contiguous().reshape(
        k, 9, *((4,) * 8), 2, 2
    )
    if layout == "morton_nibble":
        return morton.reshape(k, 9, 16, 16, 16, 16, 4)
    return morton


def canonicalize_basis_matrices(
    basis_matrices: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    """Use the c_fc orientation; c_proj is represented by its transpose."""
    shape = tuple(basis_matrices.shape[-2:])
    if shape == SUPPORTED_SHAPE:
        return basis_matrices, False
    if shape == tuple(reversed(SUPPORTED_SHAPE)):
        return basis_matrices.transpose(-2, -1).contiguous(), True
    raise ValueError(f"unsupported MLP matrix shape {shape}")


def tt_rank_profile(mode_sizes: tuple[int, ...], rank_cap: int) -> list[int]:
    if rank_cap < 1:
        raise ValueError("rank_cap must be positive")
    ranks = [1]
    prefix = 1
    total = math.prod(mode_sizes)
    for mode in mode_sizes[:-1]:
        prefix *= mode
        suffix = total // prefix
        ranks.append(min(rank_cap, prefix, suffix))
    ranks.append(1)
    return ranks


def tt_parameter_count(mode_sizes: tuple[int, ...], rank_cap: int) -> int:
    ranks = tt_rank_profile(mode_sizes, rank_cap)
    return sum(
        ranks[index] * mode * ranks[index + 1]
        for index, mode in enumerate(mode_sizes)
    )


def choose_rank_cap(
    mode_sizes: tuple[int, ...],
    *,
    scalar_budget: int,
    current_coordinate_scalars: int,
) -> int:
    available = scalar_budget - current_coordinate_scalars
    if available < tt_parameter_count(mode_sizes, 1):
        raise ValueError("budget is too small for even a rank-one TT decoder")
    cap = 1
    while tt_parameter_count(mode_sizes, cap + 1) <= available:
        cap += 1
    return cap


def tt_svd(
    tensor: torch.Tensor,
    *,
    rank_cap: int,
) -> list[torch.Tensor]:
    """Deterministic TT-SVD with a uniform maximum internal bond rank."""
    mode_sizes = tuple(int(size) for size in tensor.shape)
    target_ranks = tt_rank_profile(mode_sizes, rank_cap)
    cores: list[torch.Tensor] = []
    unfolding = tensor
    left_rank = 1
    for index, mode in enumerate(mode_sizes[:-1]):
        matrix = unfolding.reshape(left_rank * mode, -1)
        right_rank = target_ranks[index + 1]
        u, singular_values, vh = torch.linalg.svd(
            matrix, full_matrices=False
        )
        u = u[:, :right_rank]
        singular_values = singular_values[:right_rank]
        vh = vh[:right_rank]
        cores.append(u.reshape(left_rank, mode, right_rank).contiguous())
        unfolding = (singular_values.unsqueeze(1) * vh).reshape(
            right_rank, *mode_sizes[index + 1 :]
        )
        left_rank = right_rank
    cores.append(unfolding.reshape(left_rank, mode_sizes[-1], 1).contiguous())
    return cores


def tt_reconstruct(cores: list[torch.Tensor]) -> torch.Tensor:
    result = cores[0].squeeze(0)
    for core in cores[1:]:
        result = torch.einsum("...r,rns->...ns", result, core)
    return result.squeeze(-1)


def materialization_madds(
    mode_sizes: tuple[int, ...], ranks: list[int]
) -> int:
    # Runtime begins by contracting the current PC-coordinate vector with the
    # first core, then expands the spatial modes in sequence.
    operations = mode_sizes[0] * ranks[1]
    prefix = 1
    for index, mode in enumerate(mode_sizes[1:], start=1):
        operations += prefix * ranks[index] * mode * ranks[index + 1]
        prefix *= mode
    return operations


def fit_budget(
    target: torch.Tensor,
    *,
    layout: str,
    ratio: float,
    matrix_scalars: int,
) -> tuple[dict[str, Any], list[torch.Tensor]]:
    mode_sizes = tuple(int(size) for size in target.shape)
    scalar_budget = math.floor(matrix_scalars * ratio)
    current_coordinates = target.shape[0]
    rank_cap = choose_rank_cap(
        mode_sizes,
        scalar_budget=scalar_budget,
        current_coordinate_scalars=current_coordinates,
    )
    cores = tt_svd(target, rank_cap=rank_cap)
    reconstructed = tt_reconstruct(cores)
    core_scalars = sum(core.numel() for core in cores)
    stored_scalars = core_scalars + current_coordinates
    if stored_scalars > scalar_budget:
        raise AssertionError("realized TT state exceeds its scalar budget")
    target_flat = target.reshape(target.shape[0], -1).double()
    reconstructed_flat = reconstructed.reshape(target.shape[0], -1).double()
    target_energy = target_flat.square().sum(dim=1)
    error_energy = (target_flat - reconstructed_flat).square().sum(dim=1)
    per_pc_capture = 1.0 - error_energy / target_energy.clamp_min(1e-30)
    weighted_capture = float(
        1.0
        - error_energy.sum()
        / target_energy.sum().clamp_min(1e-30)
    )
    ranks = [cores[0].shape[0]] + [core.shape[-1] for core in cores]
    madds = materialization_madds(mode_sizes, ranks)
    result = {
        "family": "joint_qtt_mpo",
        "layout": layout,
        "coordinate_ratio_requested": ratio,
        "scalar_budget": scalar_budget,
        "rank_cap": rank_cap,
        "mode_sizes": "x".join(str(size) for size in mode_sizes),
        "bond_ranks": "x".join(str(rank) for rank in ranks),
        "decoder_core_scalars": core_scalars,
        "current_coordinate_scalars": current_coordinates,
        "stored_scalars": stored_scalars,
        "stored_scalar_fraction": stored_scalars / matrix_scalars,
        "largest_core_scalars": max(core.numel() for core in cores),
        "weighted_retained_basis_capture": weighted_capture,
        "minimum_pc_capture": float(per_pc_capture.min()),
        "maximum_pc_capture": float(per_pc_capture.max()),
        "materialization_madds": madds,
        "materialization_madds_per_weight": madds / matrix_scalars,
    }
    del reconstructed
    return result, [core.detach().cpu() for core in cores]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", default="6")
    parser.add_argument("--targets", default="mlp.c_fc,mlp.c_proj")
    parser.add_argument("--basis-rank", type=int, default=16)
    parser.add_argument("--ratios", default="0.001,0.0025,0.005,0.01")
    parser.add_argument(
        "--layouts",
        default=(
            "row_then_column_binary,morton_binary,"
            "morton_reverse_column_bits,morton_nibble"
        ),
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    layers = set(parse_int_list(args.layers))
    targets = {item for item in args.targets.split(",") if item}
    ratios = parse_float_list(args.ratios)
    layouts = [item for item in args.layouts.split(",") if item]
    if not layers or not targets or not layouts or args.basis_rank < 1:
        raise ValueError("layers, targets, layouts, and positive rank are required")
    if any(ratio < 0.001 or ratio > 0.01 for ratio in ratios):
        raise ValueError("all budgets must remain within 0.1--1.0%")

    paths = sorted(args.snapshot_dir.glob("step_*.pt"))
    steps, values, snapshot_metadata = load_snapshots(
        paths, layers=layers, targets=targets
    )
    rows: list[dict[str, Any]] = []
    saved_cores: dict[str, Any] = {}
    for parameter, tensors in sorted(values.items()):
        match = PARAMETER_PATTERN.match(parameter)
        if match is None:
            raise ValueError(f"unsupported parameter {parameter}")
        positions = torch.stack(tensors).to(args.device, dtype=torch.float32)
        residuals, all_values, basis = residual_temporal_basis(
            positions, maximum_rank=args.basis_rank
        )
        retained_rank = basis.shape[1]
        retained_values = all_values[:retained_rank]
        total_value = all_values.sum().clamp_min(1e-30)
        retained_fraction = float(retained_values.sum() / total_value)
        normalized = retained_values / retained_values.sum().clamp_min(1e-30)
        basis_matrices = basis.T.reshape(
            retained_rank, *positions.shape[1:]
        )
        basis_matrices, canonical_transpose = canonicalize_basis_matrices(
            basis_matrices
        )
        weighted_basis = basis_matrices * normalized.sqrt().to(
            basis_matrices.dtype
        ).view(-1, 1, 1)
        matrix_scalars = positions[0].numel()
        parameter_cores: dict[str, Any] = {}
        for layout in layouts:
            tensor = tensorize_basis(weighted_basis, layout=layout)
            layout_cores: dict[float, Any] = {}
            for ratio in ratios:
                result, cores = fit_budget(
                    tensor,
                    layout=layout,
                    ratio=ratio,
                    matrix_scalars=matrix_scalars,
                )
                result.update(
                    {
                        "parameter": parameter,
                        "layer": int(match.group("layer")),
                        "target": match.group("target"),
                        "canonical_transpose": canonical_transpose,
                        "snapshot_count": len(steps),
                        "residual_count": len(steps),
                        "residual_basis_rank": retained_rank,
                        "retained_residual_energy_fraction": retained_fraction,
                        "full_residual_trajectory_capture": (
                            retained_fraction
                            * result["weighted_retained_basis_capture"]
                        ),
                    }
                )
                rows.append(result)
                layout_cores[ratio] = cores
            parameter_cores[layout] = layout_cores
            del tensor
        saved_cores[parameter] = {
            "residual_eigenvalues": retained_values.detach().cpu(),
            "layouts": parameter_cores,
        }
        del positions, residuals, basis, basis_matrices, weighted_basis
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

    args.output.mkdir(parents=True, exist_ok=True)
    results_path = args.output / "qtt_residual_basis_fit.csv"
    cores_path = args.output / "qtt_decoder_cores.pt"
    write_csv(results_path, rows)
    torch.save(saved_cores, cores_path)
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_mlp_residual_qtt_basis_v1",
        "steps": steps,
        "snapshot_metadata": snapshot_metadata,
        "layers": sorted(layers),
        "targets": sorted(targets),
        "basis_rank": args.basis_rank,
        "ratios": ratios,
        "layouts": layouts,
        "state_contract": {
            "reference": "D_t = W_t - W_0",
            "decoder": "joint tensor train over residual-PC and spatial modes",
            "stored": "small TT cores plus one current residual-coordinate vector",
            "not_stored": "no dense PCA vector, ambient atom, dense shadow, or index table",
            "free_procedural_state": "fixed reshape and axis permutations",
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
        "outputs": {
            results_path.name: file_sha256(results_path),
            cores_path.name: file_sha256(cores_path),
        },
        "limitations": [
            "Full-horizon residual PCA fitting is a noncausal representation ceiling.",
            "Euclidean residual recovery is necessary but not a fixed-evaluation CE result.",
            "TT-SVD is locally quasi-optimal for a fixed mode order, not globally optimal.",
            "Materialization operation counts are theoretical and exclude kernel overhead.",
        ],
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "rows": len(rows),
                "metadata": str(metadata_path),
                "best_full_residual_capture": max(
                    row["full_residual_trajectory_capture"] for row in rows
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
