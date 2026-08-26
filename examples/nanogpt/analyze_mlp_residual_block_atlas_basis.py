#!/usr/bin/env python3
"""Fit basis-free block-atlas tangents to dense MLP residual PCs.

The current residual is represented after a fixed block unfolding as
``X = U V^T``.  Persistent state is only the two thin factors.  No dense PCA
direction, ambient atom, dense shadow, or per-weight code is retained.

This analysis is an optimistic, noncausal representational ceiling: it uses
all saved residual PCs to orient the factor tangent and reports exact
variance-weighted projection onto that tangent.
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

from examples.nanogpt.analyze_mlp_block_unfolding_atlas import unfold_blocks
from examples.nanogpt.analyze_mlp_highcadence_basis import (
    file_sha256,
    parse_float_list,
)
from examples.nanogpt.analyze_mlp_residual_qtt_basis import (
    residual_temporal_basis,
)
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


def top_left_space(matrices: torch.Tensor, rank: int) -> torch.Tensor:
    rows = matrices.permute(1, 0, 2).reshape(
        matrices.shape[1], matrices.shape[0] * matrices.shape[2]
    )
    q = min(rank + 4, min(rows.shape))
    left, _values, _right = torch.pca_lowrank(
        rows, q=q, center=False, niter=4
    )
    return left[:, :rank]


def top_right_space(matrices: torch.Tensor, rank: int) -> torch.Tensor:
    rows = matrices.reshape(
        matrices.shape[0] * matrices.shape[1], matrices.shape[2]
    )
    q = min(rank + 4, min(rows.shape))
    _left, _values, right = torch.pca_lowrank(
        rows, q=q, center=False, niter=4
    )
    return right[:, :rank]


def tangent_projection(
    matrices: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    left_projection = torch.einsum(
        "nr,krm->knm", left, torch.einsum("nr,knm->krm", left, matrices)
    )
    right_projection = torch.einsum(
        "knr,mr->knm", torch.einsum("knm,mr->knr", matrices, right), right
    )
    intersection = torch.einsum(
        "nr,krq,mq->knm",
        left,
        torch.einsum("nr,knm,mq->krq", left, matrices, right),
        right,
    )
    return left_projection + right_projection - intersection


def fit_tangent(
    matrices: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    rank: int,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, float]]]:
    weighted = matrices * probabilities.sqrt().to(matrices.dtype).view(-1, 1, 1)
    left = top_left_space(weighted, rank)
    right = top_right_space(weighted, rank)
    history: list[dict[str, float]] = []
    for iteration in range(iterations):
        right_residual = weighted - torch.einsum(
            "knr,mr->knm",
            torch.einsum("knm,mr->knr", weighted, right),
            right,
        )
        left = top_left_space(right_residual, rank)
        left_residual = weighted - torch.einsum(
            "nr,krm->knm",
            left,
            torch.einsum("nr,knm->krm", left, weighted),
        )
        right = top_right_space(left_residual, rank)
        projected = tangent_projection(weighted, left, right)
        history.append(
            {
                "iteration": iteration + 1,
                "weighted_tangent_capture": float(
                    projected.double().square().sum()
                    / weighted.double().square().sum().clamp_min(1e-30)
                ),
            }
        )
    return left, right, history


def best_rank_capture(matrix: torch.Tensor, rank: int) -> float:
    values = torch.linalg.svdvals(matrix)
    return float(
        values[:rank].double().square().sum()
        / values.double().square().sum().clamp_min(1e-30)
    )


def evaluate(
    matrices: torch.Tensor,
    probabilities: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
) -> tuple[float, float, float, list[float]]:
    projected = tangent_projection(matrices, left, right)
    captures = (
        projected.double().flatten(1).square().sum(dim=1)
        / matrices.double().flatten(1).square().sum(dim=1).clamp_min(1e-30)
    )
    weighted = float((probabilities.double() * captures).sum())
    return weighted, float(captures.min()), float(captures.max()), captures.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--targets", default="mlp.c_fc,mlp.c_proj")
    parser.add_argument("--basis-rank", type=int, default=16)
    parser.add_argument("--ratios", default="0.001,0.0025,0.005,0.01")
    parser.add_argument("--block-sizes", default="16,32,64,128")
    parser.add_argument("--fit-iterations", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    torch.manual_seed(20260826)
    targets = {item for item in args.targets.split(",") if item}
    ratios = parse_float_list(args.ratios)
    block_sizes = parse_int_list(args.block_sizes)
    paths = sorted(args.snapshot_dir.glob("step_*.pt"))
    steps, values, snapshot_metadata = load_snapshots(
        paths, layers={args.layer}, targets=targets
    )
    rows: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    saved_factors: dict[str, Any] = {}
    retained_fractions: dict[str, float] = {}
    for parameter, tensors in sorted(values.items()):
        match = PARAMETER_PATTERN.match(parameter)
        if match is None:
            raise ValueError(f"unsupported parameter {parameter}")
        positions = torch.stack(tensors).to(args.device, dtype=torch.float32)
        residuals, eigenvalues, basis = residual_temporal_basis(
            positions, maximum_rank=args.basis_rank
        )
        retained = eigenvalues[: basis.shape[1]]
        probabilities = retained / retained.sum().clamp_min(1e-30)
        retained_fractions[parameter] = float(
            retained.sum() / eigenvalues.sum().clamp_min(1e-30)
        )
        basis_matrices = basis.T.reshape(basis.shape[1], *positions.shape[1:])
        terminal = residuals[-1]
        dense_scalars = terminal.numel()
        parameter_factors: dict[str, Any] = {}
        for block_size in block_sizes:
            unfolded_basis = torch.stack(
                [unfold_blocks(matrix, block_size) for matrix in basis_matrices]
            )
            unfolded_terminal = unfold_blocks(terminal, block_size)
            tile_count, atom_size = unfolded_terminal.shape
            for ratio in ratios:
                budget = int(dense_scalars * ratio)
                rank = budget // (tile_count + atom_size)
                if rank < 1:
                    rows.append(
                        {
                            "parameter": parameter,
                            "target": match.group("target"),
                            "block_size": block_size,
                            "budget_requested": ratio,
                            "feasible": False,
                            "rank": 0,
                            "stored_scalars": 0,
                            "stored_scalar_fraction": 0.0,
                        }
                    )
                    continue
                rank = min(rank, min(tile_count, atom_size))
                left, right, history = fit_tangent(
                    unfolded_basis,
                    probabilities,
                    rank=rank,
                    iterations=args.fit_iterations,
                )
                weighted, minimum, maximum, captures = evaluate(
                    unfolded_basis, probabilities, left, right
                )
                stored = rank * (tile_count + atom_size)
                core = left.T @ unfolded_terminal @ right
                fitted_terminal = left @ core @ right.T
                terminal_energy = unfolded_terminal.double().square().sum().clamp_min(1e-30)
                rows.append(
                    {
                        "parameter": parameter,
                        "target": match.group("target"),
                        "block_size": block_size,
                        "budget_requested": ratio,
                        "feasible": True,
                        "rank": rank,
                        "tile_count": tile_count,
                        "atom_size": atom_size,
                        "stored_scalars": stored,
                        "stored_scalar_fraction": stored / dense_scalars,
                        "weighted_residual_pc_tangent_capture": weighted,
                        "minimum_pc_tangent_capture": minimum,
                        "maximum_pc_tangent_capture": maximum,
                        "full_residual_trajectory_tangent_capture": (
                            retained_fractions[parameter] * weighted
                        ),
                        "fitted_frame_terminal_value_capture": float(
                            fitted_terminal.double().square().sum() / terminal_energy
                        ),
                        "best_rank_terminal_value_ceiling": best_rank_capture(
                            unfolded_terminal, rank
                        ),
                        "materialization_madd_per_generated_weight": rank,
                    }
                )
                key = f"block{block_size}_ratio{ratio:g}"
                parameter_factors[key] = {
                    "left": left.detach().cpu(),
                    "right": right.detach().cpu(),
                    "rank": rank,
                    "pc_captures": captures,
                }
                histories.extend(
                    {
                        "parameter": parameter,
                        "block_size": block_size,
                        "budget_requested": ratio,
                        **item,
                    }
                    for item in history
                )
            del unfolded_basis, unfolded_terminal
        saved_factors[parameter] = parameter_factors
        del positions, residuals, basis, basis_matrices, terminal
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

    args.output.mkdir(parents=True, exist_ok=True)
    results_path = args.output / "block_atlas_residual_basis.csv"
    history_path = args.output / "fit_history.csv"
    factors_path = args.output / "block_atlas_factors.pt"
    write_csv(results_path, rows)
    write_csv(history_path, histories)
    torch.save(saved_factors, factors_path)
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_mlp_residual_block_atlas_basis_v1",
        "steps": steps,
        "snapshot_metadata": snapshot_metadata,
        "layer": args.layer,
        "targets": sorted(targets),
        "basis_rank": args.basis_rank,
        "retained_residual_energy_fraction": retained_fractions,
        "ratios": ratios,
        "block_sizes": block_sizes,
        "fit_iterations": args.fit_iterations,
        "state_contract": {
            "reference": "D_t = W_t - W_0",
            "decoder": "fixed block-unfolding rank manifold X = U V^T",
            "stored": "only current thin U/V factors",
            "not_stored": "no residual PC, ambient atom, dense shadow, or per-weight code",
            "procedural": "block reshape/fold",
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
            history_path.name: file_sha256(history_path),
            factors_path.name: file_sha256(factors_path),
        },
        "limitations": [
            "Full-horizon residual PCA fitting is a noncausal tangent ceiling.",
            "The tangent-optimal factors need not be the value-optimal current factors.",
            "Euclidean PCA recovery is necessary but not fixed-evaluation CE.",
            "No block-atlas inference kernel is benchmarked unless recovery passes.",
        ],
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(rows), "metadata": str(metadata_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
