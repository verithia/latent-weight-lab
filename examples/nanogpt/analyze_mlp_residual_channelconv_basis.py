#!/usr/bin/env python3
"""Project dense MLP residual PCs onto a paired channel-convolution chart."""
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
from examples.nanogpt.analyze_mlp_residual_multiresolution_basis import (
    CANONICAL_SHAPE,
    canonicalize,
)
from examples.nanogpt.analyze_mlp_residual_qtt_basis import residual_temporal_basis
from examples.nanogpt.analyze_parameter_trajectory import (
    PARAMETER_PATTERN,
    load_snapshots,
    write_csv,
)


WIDTH = 768
BRANCHES = 4
CHANNELS = 32
POSITIONS = 24
KERNEL_SIZE = 5
AFFINE_MULTIPLIERS = (1, 5, 7, 11)
HIDDEN_OFFSETS = (0, 193, 389, 577)
RESIDUAL_OFFSETS = (0, 149, 313, 521)


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def affine_permutation(width: int, multiplier: int, offset: int, *, device) -> torch.Tensor:
    if math.gcd(multiplier, width) != 1:
        raise ValueError("affine multiplier must be coprime to width")
    index = torch.arange(width, device=device, dtype=torch.long)
    return (multiplier * index + offset) % width


def branch_permutations(branch: int, *, device) -> tuple[torch.Tensor, torch.Tensor]:
    multiplier = AFFINE_MULTIPLIERS[branch]
    hidden = affine_permutation(
        WIDTH, multiplier, HIDDEN_OFFSETS[branch], device=device
    )
    residual = affine_permutation(
        WIDTH,
        AFFINE_MULTIPLIERS[(branch + 1) % BRANCHES],
        RESIDUAL_OFFSETS[branch],
        device=device,
    )
    return hidden, residual


def circular_channel_projection_energy(
    matrices: torch.Tensor,
    *,
    target: str,
) -> torch.Tensor:
    """Return exact projection energy for each canonical PC matrix."""
    if tuple(matrices.shape[-2:]) != CANONICAL_SHAPE:
        raise ValueError(f"expected canonical matrices shaped {CANONICAL_SHAPE}")
    energies = torch.zeros(
        matrices.shape[0], device=matrices.device, dtype=torch.float64
    )
    radius = KERNEL_SIZE // 2
    for branch in range(BRANCHES):
        block = matrices[
            :, branch * WIDTH : (branch + 1) * WIDTH, :
        ]
        hidden, residual = branch_permutations(
            branch, device=matrices.device
        )
        if target == "mlp.c_fc":
            row_permutation, column_permutation = hidden, residual
        elif target == "mlp.c_proj":
            # Canonical c_proj is the transpose: hidden rows, residual columns.
            row_permutation, column_permutation = hidden, residual
        else:
            raise ValueError(f"unsupported target {target!r}")
        block = block.index_select(-2, row_permutation).index_select(
            -1, column_permutation
        )
        shaped = block.reshape(
            block.shape[0], CHANNELS, POSITIONS, CHANNELS, POSITIONS
        )
        positions = range(POSITIONS)
        for delta in range(-radius, radius + 1):
            tied = torch.stack(
                [
                    shaped[:, :, position, :, (position + delta) % POSITIONS]
                    for position in positions
                ],
                dim=-1,
            )
            coefficient = tied.double().mean(dim=-1)
            energies += POSITIONS * coefficient.square().sum(dim=(-2, -1))
    return energies


def materialize_permuted_channel_convolution(
    kernels: torch.Tensor,
) -> torch.Tensor:
    """Materialize canonical matrices for a synthetic exact-family check."""
    if tuple(kernels.shape) != (
        BRANCHES,
        CHANNELS,
        CHANNELS,
        KERNEL_SIZE,
    ):
        raise ValueError("unexpected kernel shape")
    device = kernels.device
    matrix = torch.zeros(CANONICAL_SHAPE, device=device, dtype=kernels.dtype)
    radius = KERNEL_SIZE // 2
    for branch in range(BRANCHES):
        permuted = torch.zeros((WIDTH, WIDTH), device=device, dtype=kernels.dtype)
        for output_channel in range(CHANNELS):
            for input_channel in range(CHANNELS):
                for kernel_index, delta in enumerate(range(-radius, radius + 1)):
                    positions = torch.arange(POSITIONS, device=device)
                    rows = output_channel * POSITIONS + positions
                    columns = input_channel * POSITIONS + (positions + delta) % POSITIONS
                    permuted[rows, columns] = kernels[
                        branch, output_channel, input_channel, kernel_index
                    ]
        hidden, residual = branch_permutations(branch, device=device)
        inverse_hidden = torch.argsort(hidden)
        inverse_residual = torch.argsort(residual)
        block = permuted.index_select(0, inverse_hidden).index_select(
            1, inverse_residual
        )
        matrix[branch * WIDTH : (branch + 1) * WIDTH] = block
    return matrix


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
    device = torch.device(args.device)
    rows: list[dict[str, Any]] = []
    retained_fractions: dict[str, float] = {}
    for parameter, tensors in sorted(values.items()):
        match = PARAMETER_PATTERN.match(parameter)
        if match is None:
            raise ValueError(f"unsupported parameter {parameter}")
        target = match.group("target")
        positions = torch.stack(tensors).to(device, dtype=torch.float32)
        _residuals, eigenvalues, basis = residual_temporal_basis(
            positions, maximum_rank=args.basis_rank
        )
        retained = eigenvalues[: basis.shape[1]]
        probabilities = retained / retained.sum().clamp_min(1e-30)
        matrices = basis.T.reshape(basis.shape[1], *positions.shape[1:])
        matrices, transposed = canonicalize(matrices)
        projection_energy = circular_channel_projection_energy(
            matrices, target=target
        )
        total_energy = matrices.double().square().sum(dim=(-2, -1)).clamp_min(1e-30)
        captures = projection_energy / total_energy
        retained_fraction = float(
            retained.sum() / eigenvalues.sum().clamp_min(1e-30)
        )
        retained_fractions[parameter] = retained_fraction
        stored = BRANCHES * CHANNELS * CHANNELS * KERNEL_SIZE
        dense = math.prod(CANONICAL_SHAPE)
        direct_madds = BRANCHES * WIDTH * CHANNELS * KERNEL_SIZE
        rows.append(
            {
                "parameter": parameter,
                "target": target,
                "canonical_transpose": transposed,
                "family": "paired_channelconv32_circular_k5",
                "stored_scalars": stored,
                "stored_scalar_fraction": stored / dense,
                "weighted_pc_capture": float(
                    torch.sum(probabilities.double() * captures)
                ),
                "minimum_pc_capture": float(captures.min()),
                "maximum_pc_capture": float(captures.max()),
                "full_residual_recovery": float(
                    retained_fraction
                    * torch.sum(probabilities.double() * captures)
                ),
                "direct_madds": direct_madds,
                "direct_madds_to_dense": direct_madds / dense,
            }
        )
        del positions, matrices
        if device.type == "cuda":
            torch.cuda.empty_cache()

    generator = torch.Generator(device=device)
    generator.manual_seed(20260827)
    kernels = torch.randn(
        BRANCHES,
        CHANNELS,
        CHANNELS,
        KERNEL_SIZE,
        generator=generator,
        device=device,
    )
    synthetic = materialize_permuted_channel_convolution(kernels).unsqueeze(0)
    synthetic_energy = circular_channel_projection_energy(
        synthetic, target="mlp.c_fc"
    )
    own_recovery = float(
        synthetic_energy / synthetic.double().square().sum().clamp_min(1e-30)
    )
    if own_recovery < 0.999:
        raise RuntimeError(f"synthetic own-family recovery failed: {own_recovery}")

    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / "channelconv_recovery.csv"
    write_csv(result_path, rows)
    script = Path(__file__).resolve()
    metadata: dict[str, Any] = {
        "schema_version": "nanogpt_mlp_residual_channelconv_basis_v1",
        "steps": steps,
        "snapshot_metadata": snapshot_metadata,
        "layer": args.layer,
        "targets": sorted(targets),
        "basis_rank": args.basis_rank,
        "topology": {
            "branches": BRANCHES,
            "channels": CHANNELS,
            "positions": POSITIONS,
            "kernel_size": KERNEL_SIZE,
            "affine_multipliers": AFFINE_MULTIPLIERS,
            "hidden_offsets": HIDDEN_OFFSETS,
            "residual_offsets": RESIDUAL_OFFSETS,
            "paired_hidden_gauge": True,
        },
        "retained_residual_energy_fraction": retained_fractions,
        "synthetic_own_family_recovery": own_recovery,
        "candidate_contract": {
            "stored": "four learned 32x32x5 kernels per MLP matrix",
            "procedural": "affine permutations, circular positions, and branch partition",
            "forbidden": "PCA atom, ambient shadow, fitted index table, BlockFHT, or topology sweep",
        },
        "gates": {
            "reject_below_weighted_capture": 0.20,
            "performance_gate_at_weighted_capture": 0.50,
            "training_gate_at_weighted_capture": 0.80,
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
            "The chart is linear in kernels; passing would still require a fused direct-apply throughput gate.",
        ],
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": rows, "metadata": str(metadata_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
