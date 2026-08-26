#!/usr/bin/env python3
"""Fit a basis-free learned Givens sparse chart to MLP residual PCs.

The chart stores row/column Givens angles plus one live sparse coordinate
vector.  Pairings and support locations are regenerated from integer seeds;
no ambient basis vector, permutation table, support-index table, dense shadow,
or per-weight code is persistent state.
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
from torch import nn

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


def seeded_permutations(
    size: int, depth: int, *, seed: int, device: str
) -> torch.Tensor:
    if size % 2:
        raise ValueError("random-matching Givens transform requires an even axis")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.stack(
        [torch.randperm(size, generator=generator) for _ in range(depth)]
    ).to(device)


def apply_matching_rows(
    matrices: torch.Tensor,
    permutation: torch.Tensor,
    angles: torch.Tensor,
) -> torch.Tensor:
    gathered = matrices[:, permutation, :]
    pairs = gathered.reshape(
        matrices.shape[0], permutation.numel() // 2, 2, matrices.shape[2]
    )
    cosine = angles.cos().view(1, -1, 1)
    sine = angles.sin().view(1, -1, 1)
    first = cosine * pairs[:, :, 0] + sine * pairs[:, :, 1]
    second = -sine * pairs[:, :, 0] + cosine * pairs[:, :, 1]
    rotated = torch.stack((first, second), dim=2).reshape_as(gathered)
    inverse = torch.argsort(permutation)
    return rotated[:, inverse, :]


class RandomMatchingGivens2D(nn.Module):
    def __init__(self, rows: int, columns: int, depth: int, *, seed: int, device: str) -> None:
        super().__init__()
        self.rows = rows
        self.columns = columns
        self.depth = depth
        self.seed = seed
        self.row_angles = nn.Parameter(torch.zeros(depth, rows // 2, device=device))
        self.column_angles = nn.Parameter(
            torch.zeros(depth, columns // 2, device=device)
        )
        self.register_buffer(
            "row_permutations",
            seeded_permutations(rows, depth, seed=seed, device=device),
            persistent=False,
        )
        self.register_buffer(
            "column_permutations",
            seeded_permutations(columns, depth, seed=seed + 1, device=device),
            persistent=False,
        )

    @property
    def angle_count(self) -> int:
        return self.row_angles.numel() + self.column_angles.numel()

    def forward(self, matrices: torch.Tensor) -> torch.Tensor:
        current = matrices
        for stage in range(self.depth):
            current = apply_matching_rows(
                current, self.row_permutations[stage], self.row_angles[stage]
            )
            current = apply_matching_rows(
                current.transpose(1, 2),
                self.column_permutations[stage],
                self.column_angles[stage],
            ).transpose(1, 2)
        return current


def procedural_support(
    size: int, coordinates: int, *, seed: int, device: str
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randperm(size, generator=generator)[:coordinates].to(device)


def captures(
    transformed: torch.Tensor,
    probabilities: torch.Tensor,
    support: torch.Tensor,
) -> tuple[float, float, float, list[float]]:
    selected = transformed.flatten(1)[:, support]
    per_pc = (
        selected.double().square().sum(dim=1)
        / transformed.double().flatten(1).square().sum(dim=1).clamp_min(1e-30)
    )
    weighted = float((probabilities.double() * per_pc).sum())
    return weighted, float(per_pc.min()), float(per_pc.max()), per_pc.tolist()


def fit_angles(
    module: RandomMatchingGivens2D,
    basis: torch.Tensor,
    probabilities: torch.Tensor,
    support: torch.Tensor,
    *,
    updates: int,
    learning_rate: float,
) -> list[dict[str, float]]:
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate)
    weighted_basis = basis * probabilities.sqrt().to(basis.dtype).view(-1, 1, 1)
    total = weighted_basis.double().square().sum().clamp_min(1e-30)
    history: list[dict[str, float]] = []
    for update in range(updates):
        transformed = module(weighted_basis)
        selected = transformed.flatten(1)[:, support]
        capture = selected.double().square().sum() / total
        optimizer.zero_grad(set_to_none=True)
        (-capture).backward()
        optimizer.step()
        with torch.no_grad():
            module.row_angles.clamp_(-math.pi, math.pi)
            module.column_angles.clamp_(-math.pi, math.pi)
        if update == 0 or (update + 1) % 16 == 0 or update + 1 == updates:
            history.append(
                {
                    "update": update + 1,
                    "weighted_support_capture": float(capture.detach()),
                    "angle_rms": float(
                        torch.cat(
                            (
                                module.row_angles.detach().flatten(),
                                module.column_angles.detach().flatten(),
                            )
                        ).square().mean().sqrt()
                    ),
                    "angle_max_abs": max(
                        float(module.row_angles.detach().abs().max()),
                        float(module.column_angles.detach().abs().max()),
                    ),
                }
            )
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--targets", default="mlp.c_fc,mlp.c_proj")
    parser.add_argument("--basis-rank", type=int, default=16)
    parser.add_argument("--ratios", default="0.001,0.0025,0.005,0.01")
    parser.add_argument("--depths", default="1,2,4,8")
    parser.add_argument("--fit-updates", type=int, default=128)
    parser.add_argument("--fit-lr", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    torch.manual_seed(args.seed)
    targets = {item for item in args.targets.split(",") if item}
    ratios = parse_float_list(args.ratios)
    depths = parse_int_list(args.depths)
    if len(ratios) != len(depths):
        raise ValueError("ratios and preregistered depths must have equal length")
    paths = sorted(args.snapshot_dir.glob("step_*.pt"))
    steps, values, snapshot_metadata = load_snapshots(
        paths, layers={args.layer}, targets=targets
    )
    rows_out: list[dict[str, Any]] = []
    history_out: list[dict[str, Any]] = []
    saved_angles: dict[str, Any] = {}
    retained_fractions: dict[str, float] = {}
    for parameter_index, (parameter, tensors) in enumerate(sorted(values.items())):
        match = PARAMETER_PATTERN.match(parameter)
        if match is None:
            raise ValueError(f"unsupported parameter {parameter}")
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
        matrix_rows, matrix_columns = positions.shape[1:]
        dense_scalars = matrix_rows * matrix_columns
        parameter_angles: dict[str, Any] = {}
        for ratio_index, (ratio, depth) in enumerate(zip(ratios, depths, strict=True)):
            budget = int(dense_scalars * ratio)
            module = RandomMatchingGivens2D(
                matrix_rows,
                matrix_columns,
                depth,
                seed=args.seed + parameter_index * 100 + ratio_index * 2,
                device=args.device,
            )
            live_coordinates = budget - module.angle_count
            if live_coordinates < 1:
                raise ValueError(
                    f"depth {depth} angle state exceeds ratio {ratio}: "
                    f"{module.angle_count} >= {budget}"
                )
            support_seed = args.seed + parameter_index * 100 + ratio_index * 2 + 1
            support = procedural_support(
                dense_scalars,
                live_coordinates,
                seed=support_seed,
                device=args.device,
            )
            with torch.no_grad():
                initial = captures(
                    module(basis_matrices), probabilities, support
                )
            history = fit_angles(
                module,
                basis_matrices,
                probabilities,
                support,
                updates=args.fit_updates,
                learning_rate=args.fit_lr,
            )
            with torch.no_grad():
                fitted = captures(module(basis_matrices), probabilities, support)
            stored = module.angle_count + live_coordinates
            rows_out.append(
                {
                    "parameter": parameter,
                    "target": match.group("target"),
                    "budget_requested": ratio,
                    "depth": depth,
                    "angle_scalars": module.angle_count,
                    "live_sparse_coordinates": live_coordinates,
                    "stored_scalars": stored,
                    "stored_scalar_fraction": stored / dense_scalars,
                    "identity_weighted_capture": initial[0],
                    "fitted_weighted_capture": fitted[0],
                    "fitted_minimum_pc_capture": fitted[1],
                    "fitted_maximum_pc_capture": fitted[2],
                    "full_residual_trajectory_capture": (
                        retained_fractions[parameter] * fitted[0]
                    ),
                    "materialization_madd_per_generated_weight": 4 * depth,
                    "ideal_compressed_forward_madd_ratio_vs_dense": (
                        2 * depth * (matrix_rows + matrix_columns)
                        + live_coordinates
                    )
                    / dense_scalars,
                    "matching_seed": module.seed,
                    "support_seed": support_seed,
                }
            )
            history_out.extend(
                {
                    "parameter": parameter,
                    "budget_requested": ratio,
                    "depth": depth,
                    **item,
                }
                for item in history
            )
            parameter_angles[f"ratio{ratio:g}"] = {
                "row_angles": module.row_angles.detach().cpu(),
                "column_angles": module.column_angles.detach().cpu(),
                "depth": depth,
                "matching_seed": module.seed,
                "support_seed": support_seed,
                "live_sparse_coordinates": live_coordinates,
                "per_pc_capture": fitted[3],
            }
            del module, support
        saved_angles[parameter] = parameter_angles
        del positions, basis, basis_matrices
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

    args.output.mkdir(parents=True, exist_ok=True)
    results_path = args.output / "givens_sparse_residual_basis.csv"
    history_path = args.output / "fit_history.csv"
    angles_path = args.output / "givens_sparse_angles.pt"
    write_csv(results_path, rows_out)
    write_csv(history_path, history_out)
    torch.save(saved_angles, angles_path)
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_mlp_residual_givens_sparse_basis_v1",
        "steps": steps,
        "snapshot_metadata": snapshot_metadata,
        "layer": args.layer,
        "targets": sorted(targets),
        "basis_rank": args.basis_rank,
        "retained_residual_energy_fraction": retained_fractions,
        "ratios": ratios,
        "depths": depths,
        "fit_updates": args.fit_updates,
        "fit_lr": args.fit_lr,
        "seed": args.seed,
        "state_contract": {
            "stored": "learned Givens angles plus one current sparse coordinate vector",
            "not_stored": "no PCA vector, matching permutation, support index table, ambient atom, dense shadow, or per-weight code",
            "procedural": "random perfect matchings and sparse support regenerated from seeds",
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
            angles_path.name: file_sha256(angles_path),
        },
        "limitations": [
            "Full-horizon residual PCA fitting is a noncausal representation ceiling.",
            "A single preregistered depth/state split is tested per budget.",
            "The compressed-forward arithmetic is ideal and still requires a fused-kernel gate.",
            "Euclidean PCA recovery is necessary but not fixed-evaluation CE.",
        ],
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(rows_out), "metadata": str(metadata_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
