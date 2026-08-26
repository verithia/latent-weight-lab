#!/usr/bin/env python3
"""Test a compact learned block-atom atlas on held-out MLP gradients.

For a matrix W, partition it into b-by-b tiles and stack the vectorized tiles
as rows of X.  A rank-k factorization X = C B^T stores

    k * (number_of_tiles + b^2)

scalars while generating a generally full-matrix-rank W.  Each coefficient
is local to one tile, while the learned atoms B are shared across tiles.  This
is a different matricization from ordinary LoRA and is intended to test
whether many cheap local coordinates can recover a rotating dense MLP field.

The audit fits factor spaces from only the preceding probe window and scores
the immediately following raw clipped gradient in the exact rank-k-manifold
tangent.  It also reports the per-gradient best-rank ceiling.  It performs no
language-model updates.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_mlp_highcadence_basis import file_sha256
from examples.nanogpt.analyze_mlp_optimizer_probe_span import load_probe_inventory
from examples.nanogpt.analyze_mlp_raw_gradient_factor_transport import (
    canonical_overlap,
    fit_shared_factors,
    tangent_capture,
)
from examples.nanogpt.analyze_mlp_raw_gradient_rolling_prediction import (
    phase_for_step,
)
from examples.nanogpt.analyze_parameter_trajectory import parse_int_list, write_csv


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def unfold_blocks(matrix: torch.Tensor, block_size: int) -> torch.Tensor:
    rows, columns = matrix.shape
    if rows % block_size or columns % block_size:
        raise ValueError(
            f"matrix shape {tuple(matrix.shape)} is not divisible by block {block_size}"
        )
    return (
        matrix.reshape(rows // block_size, block_size, columns // block_size, block_size)
        .permute(0, 2, 1, 3)
        .reshape((rows // block_size) * (columns // block_size), block_size * block_size)
    )


def fold_blocks(unfolded: torch.Tensor, rows: int, columns: int, block_size: int) -> torch.Tensor:
    return (
        unfolded.reshape(rows // block_size, columns // block_size, block_size, block_size)
        .permute(0, 2, 1, 3)
        .reshape(rows, columns)
    )


def maximum_rank_for_budget(
    rows: int, columns: int, block_size: int, scalar_fraction: float
) -> int:
    tile_count = (rows // block_size) * (columns // block_size)
    atom_size = block_size * block_size
    return max(1, int(scalar_fraction * rows * columns // (tile_count + atom_size)))


def truncated_svd(
    matrix: torch.Tensor, component_rank: int, niter: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = min(component_rank, min(matrix.shape))
    # The seed is fixed per matrix/probe by the caller, making this randomized
    # range finder reproducible without storing dense singular frames.
    u, singular_values, v = torch.svd_lowrank(matrix, q=q, niter=niter)
    order = singular_values.argsort(descending=True)
    return u[:, order], singular_values[order], v[:, order]


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        for phase in (str(row["eval_phase"]), "all"):
            groups.setdefault((int(row["block_size"]), str(row["target"]), phase), []).append(row)
    result: list[dict[str, Any]] = []
    metrics = (
        "best_rank_capture",
        "left_capture",
        "right_capture",
        "rank_manifold_tangent_capture",
        "left_current_overlap_mean_squared_cosine",
        "right_current_overlap_mean_squared_cosine",
    )
    for (block_size, target, phase), members in sorted(groups.items()):
        energies = torch.tensor([float(row["direction_energy"]) for row in members], dtype=torch.float64)
        total = energies.sum().clamp_min(1e-30)
        item: dict[str, Any] = {
            "block_size": block_size,
            "target": target,
            "eval_phase": phase,
            "rank": int(members[0]["rank"]),
            "stored_scalar_fraction": float(members[0]["stored_scalar_fraction"]),
            "sample_count": len(members),
        }
        for metric in metrics:
            values = torch.tensor([float(row[metric]) for row in members], dtype=torch.float64)
            item[f"{metric}_energy_weighted_mean"] = float((values * energies).sum() / total)
            item[f"{metric}_median"] = float(values.median())
            item[f"{metric}_minimum"] = float(values.min())
        result.append(item)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", default="6")
    parser.add_argument("--targets", default="mlp.c_fc,mlp.c_proj")
    parser.add_argument("--block-sizes", default="32,64")
    parser.add_argument("--scalar-fraction", type=float, default=0.01)
    parser.add_argument("--history-probes", type=int, default=10)
    parser.add_argument("--fit-component-rank", type=int, default=24)
    parser.add_argument("--svd-niter", type=int, default=3)
    parser.add_argument("--discovery-stop", type=int, default=119)
    parser.add_argument("--validation-stop", type=int, default=179)
    parser.add_argument("--maximum-probes", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0 < args.scalar_fraction <= 0.01:
        raise ValueError("scalar-fraction must be in (0, 0.01]")
    if args.history_probes < 2:
        raise ValueError("history-probes must be at least two")
    started = time.time()
    paths = sorted(args.probe_dir.glob("step_*.pt"))
    if args.maximum_probes:
        paths = paths[: args.maximum_probes]
    steps, inventory, input_metadata = load_probe_inventory(
        paths,
        layers=set(parse_int_list(args.layers)),
        targets={item for item in args.targets.split(",") if item},
    )
    if len(steps) <= args.history_probes:
        raise ValueError("not enough probes for requested history")

    rows: list[dict[str, Any]] = []
    for parameter_index, (parameter, fields) in enumerate(sorted(inventory.items())):
        target = "mlp.c_fc" if ".mlp.c_fc." in parameter else "mlp.c_proj"
        original = torch.stack(fields["raw_gradient_descent"]).to(args.device, dtype=torch.float32)
        matrix_rows, matrix_columns = original.shape[1:]
        for block_size in parse_int_list(args.block_sizes):
            rank = maximum_rank_for_budget(
                matrix_rows, matrix_columns, block_size, args.scalar_fraction
            )
            component_rank = max(rank, args.fit_component_rank)
            unfolded = torch.stack([unfold_blocks(direction, block_size) for direction in original])
            left: list[torch.Tensor] = []
            values: list[torch.Tensor] = []
            right: list[torch.Tensor] = []
            best_capture: list[float] = []
            for probe_index, direction in enumerate(unfolded):
                torch.manual_seed(20260826 + parameter_index * 10000 + block_size * 100 + probe_index)
                u, s, v = truncated_svd(direction, component_rank, args.svd_niter)
                left.append(u)
                values.append(s)
                right.append(v)
                total = direction.double().square().sum().clamp_min(1e-30)
                best_capture.append(float(s[:rank].double().square().sum() / total))
            tile_count, atom_size = unfolded.shape[1:]
            stored = rank * (tile_count + atom_size)
            for eval_index in range(args.history_probes, len(steps)):
                fit_indices = list(range(eval_index - args.history_probes, eval_index))
                fitted_left, fitted_right = fit_shared_factors(
                    left,
                    values,
                    right,
                    fit_indices,
                    rank,
                    component_rank,
                )
                direction = unfolded[eval_index]
                capture = tangent_capture(direction, fitted_left, fitted_right)
                left_overlap = canonical_overlap(fitted_left, left[eval_index][:, :rank])
                right_overlap = canonical_overlap(fitted_right, right[eval_index][:, :rank])
                rows.append(
                    {
                        "parameter": parameter,
                        "target": target,
                        "block_size": block_size,
                        "rank": rank,
                        "tile_count": tile_count,
                        "atom_size": atom_size,
                        "stored_scalars": stored,
                        "stored_scalar_fraction": stored / (matrix_rows * matrix_columns),
                        "history_probes": args.history_probes,
                        "fit_step_start": steps[eval_index - args.history_probes],
                        "fit_step_stop": steps[eval_index - 1],
                        "eval_step": steps[eval_index],
                        "eval_phase": phase_for_step(
                            steps[eval_index], args.discovery_stop, args.validation_stop
                        ),
                        "direction_energy": float(direction.double().square().sum()),
                        "best_rank_capture": best_capture[eval_index],
                        **capture,
                        "left_current_overlap_mean_squared_cosine": left_overlap[0],
                        "right_current_overlap_mean_squared_cosine": right_overlap[0],
                    }
                )
            del unfolded, left, values, right
            torch.cuda.empty_cache()
        del original
        torch.cuda.empty_cache()

    summary = summarize(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    rows_path = args.output / "rolling_capture.csv"
    summary_path = args.output / "summary.csv"
    write_csv(rows_path, rows)
    write_csv(summary_path, summary)
    metadata = {
        "schema_version": "nanogpt_mlp_block_unfolding_atlas_v1",
        "method": "rank-k learned block-atom factorization; preceding ten probes fit, next probe scored",
        "sample_count": len(steps),
        "steps": steps,
        "block_sizes": parse_int_list(args.block_sizes),
        "scalar_fraction_ceiling": args.scalar_fraction,
        "history_probes": args.history_probes,
        "fit_component_rank": args.fit_component_rank,
        "svd_niter": args.svd_niter,
        "input": input_metadata,
        "summary": summary,
        "runtime_seconds": time.time() - started,
        "source_commit": git_commit(Path(__file__).resolve().parents[2]),
        "rows_sha256": file_sha256(rows_path),
        "summary_sha256": file_sha256(summary_path),
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"metadata_sha256={file_sha256(metadata_path)}")


if __name__ == "__main__":
    main()
