#!/usr/bin/env python3
"""Fit a hash-free implicit decoder to dense MLP residual PCs.

The decoder maps procedural row/column features to a small bank of basis
channels.  Persistent state is only the coordinate-MLP weights plus one
current channel-mixing vector.  It stores no per-index embedding, hash table,
dense PCA vector, ambient atom, or dense shadow.
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


def axis_features(length: int, *, maximum_frequencies: int, device: str) -> torch.Tensor:
    indices = torch.arange(length, device=device, dtype=torch.long)
    unit = indices.float() / max(length - 1, 1)
    normalized = unit.mul(2.0).sub(1.0).unsqueeze(1)
    bits = max(1, math.ceil(math.log2(length)))
    binary = torch.stack(
        [((indices >> shift) & 1).float().mul(2.0).sub(1.0) for shift in range(bits)],
        dim=1,
    )
    frequencies = min(bits, maximum_frequencies)
    angles = unit.unsqueeze(1) * math.pi * (
        2.0 ** torch.arange(frequencies, device=device, dtype=torch.float32)
    ).unsqueeze(0)
    return torch.cat((normalized, binary, angles.sin(), angles.cos()), dim=1)


def decoder_scalar_count(input_features: int, width: int, channels: int) -> int:
    # Two hidden affine layers, one channel head, and one current mixing vector.
    return (
        input_features * width
        + width
        + width * width
        + width
        + width * channels
        + channels
        + channels
    )


def maximum_width(
    input_features: int, channels: int, budget: int
) -> tuple[int, int]:
    width = 0
    while decoder_scalar_count(input_features, width + 1, channels) <= budget:
        width += 1
    if width < 1:
        raise ValueError("budget cannot hold even a width-one coordinate decoder")
    return width, decoder_scalar_count(input_features, width, channels)


class CoordinateDecoder(nn.Module):
    def __init__(self, input_features: int, width: int, channels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def coordinate_batch(
    flat_indices: torch.Tensor,
    *,
    columns: int,
    row_features: torch.Tensor,
    column_features: torch.Tensor,
    extra_features: torch.Tensor | None = None,
) -> torch.Tensor:
    rows = torch.div(flat_indices, columns, rounding_mode="floor")
    cols = flat_indices.remainder(columns)
    pieces = [row_features[rows], column_features[cols]]
    if extra_features is not None:
        pieces.append(extra_features[flat_indices])
    return torch.cat(pieces, dim=1)


def initialization_features(
    initial_weight: torch.Tensor, *, frequencies: int = 4
) -> torch.Tensor:
    normalized = initial_weight.float() / initial_weight.float().square().mean().sqrt().clamp_min(1e-12)
    clipped = normalized.clamp(-4.0, 4.0) / 4.0
    polynomial = torch.stack(
        (
            clipped,
            clipped.square(),
            clipped.pow(3),
            clipped.sign(),
            clipped.abs(),
        ),
        dim=1,
    )
    multipliers = 2.0 ** torch.arange(
        frequencies, device=initial_weight.device, dtype=torch.float32
    )
    angles = math.pi * clipped.unsqueeze(1) * multipliers.unsqueeze(0)
    return torch.cat((polynomial, angles.sin(), angles.cos()), dim=1)


def exact_subspace_capture(
    generated: torch.Tensor,
    targets: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    ridge_ratio: float = 1e-8,
) -> tuple[float, float, float, list[float]]:
    gram = generated.double().T @ generated.double()
    cross = generated.double().T @ targets.double()
    ridge = ridge_ratio * float(torch.trace(gram) / max(gram.shape[0], 1))
    solution = torch.linalg.solve(
        gram + ridge * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype),
        cross,
    )
    captured_energy = (cross * solution).sum(dim=0)
    target_energy = targets.double().square().sum(dim=0).clamp_min(1e-30)
    captures = (captured_energy / target_energy).clamp(0.0, 1.0)
    weighted = float((probabilities.double() * captures).sum())
    return weighted, float(captures.min()), float(captures.max()), captures.tolist()


def train_decoder(
    decoder: CoordinateDecoder,
    *,
    target: torch.Tensor,
    rows: int,
    columns: int,
    row_features: torch.Tensor,
    column_features: torch.Tensor,
    updates: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    extra_features: torch.Tensor | None = None,
) -> list[dict[str, float]]:
    generator = torch.Generator(device=target.device).manual_seed(seed)
    optimizer = torch.optim.AdamW(
        decoder.parameters(), lr=learning_rate, weight_decay=1e-6
    )
    history: list[dict[str, float]] = []
    parameter_count = rows * columns
    for update in range(updates):
        indices = torch.randint(
            parameter_count,
            (batch_size,),
            generator=generator,
            device=target.device,
        )
        features = coordinate_batch(
            indices,
            columns=columns,
            row_features=row_features,
            column_features=column_features,
            extra_features=extra_features,
        )
        prediction = decoder(features)
        loss = (prediction - target[indices]).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if update == 0 or (update + 1) % 64 == 0 or update + 1 == updates:
            history.append({"update": update + 1, "sampled_mse": float(loss.detach())})
    return history


def evaluate_streaming(
    decoder: CoordinateDecoder,
    *,
    targets: torch.Tensor,
    probabilities: torch.Tensor,
    rows: int,
    columns: int,
    row_features: torch.Tensor,
    column_features: torch.Tensor,
    chunk_size: int,
    extra_features: torch.Tensor | None = None,
) -> dict[str, Any]:
    generated_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, rows * columns, chunk_size):
            indices = torch.arange(
                start,
                min(start + chunk_size, rows * columns),
                device=targets.device,
            )
            features = coordinate_batch(
                indices,
                columns=columns,
                row_features=row_features,
                column_features=column_features,
                extra_features=extra_features,
            )
            generated_parts.append(decoder(features))
    generated = torch.cat(generated_parts)
    weighted, minimum, maximum, captures = exact_subspace_capture(
        generated, targets, probabilities
    )
    weighted_targets = targets * probabilities.sqrt().to(targets.dtype).view(1, -1)
    assigned_error = (generated - weighted_targets).double().square().sum()
    assigned_capture = 1.0 - assigned_error / weighted_targets.double().square().sum().clamp_min(1e-30)
    return {
        "weighted_best_remixed_basis_capture": weighted,
        "minimum_pc_capture": minimum,
        "maximum_pc_capture": maximum,
        "per_pc_capture": captures,
        "direct_assigned_basis_capture": float(assigned_capture),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--targets", default="mlp.c_fc,mlp.c_proj")
    parser.add_argument("--basis-rank", type=int, default=16)
    parser.add_argument("--ratios", default="0.001,0.0025,0.005,0.01")
    parser.add_argument("--maximum-frequencies", type=int, default=10)
    parser.add_argument("--include-initial-weight-features", action="store_true")
    parser.add_argument("--initial-weight-frequencies", type=int, default=4)
    parser.add_argument("--updates", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--evaluation-chunk", type=int, default=131072)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    targets = {item for item in args.targets.split(",") if item}
    ratios = parse_float_list(args.ratios)
    paths = sorted(args.snapshot_dir.glob("step_*.pt"))
    steps, values, snapshot_metadata = load_snapshots(
        paths, layers={args.layer}, targets=targets
    )
    rows_out: list[dict[str, Any]] = []
    history_out: list[dict[str, Any]] = []
    saved_decoders: dict[str, Any] = {}
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
        matrix_rows, matrix_columns = positions.shape[1:]
        dense_scalars = matrix_rows * matrix_columns
        # Scale unit-norm PC columns by sqrt(P), yielding O(1) targets.
        basis_targets = basis.T.reshape(basis.shape[1], dense_scalars).T.float()
        basis_targets = basis_targets * math.sqrt(dense_scalars)
        weighted_targets = basis_targets * probabilities.sqrt().to(
            basis_targets.dtype
        ).view(1, -1)
        row_features = axis_features(
            matrix_rows,
            maximum_frequencies=args.maximum_frequencies,
            device=args.device,
        )
        column_features = axis_features(
            matrix_columns,
            maximum_frequencies=args.maximum_frequencies,
            device=args.device,
        )
        extra_features = (
            initialization_features(
                positions[0].flatten(), frequencies=args.initial_weight_frequencies
            )
            if args.include_initial_weight_features
            else None
        )
        input_features = row_features.shape[1] + column_features.shape[1]
        if extra_features is not None:
            input_features += extra_features.shape[1]
        parameter_decoders: dict[str, Any] = {}
        for ratio_index, ratio in enumerate(ratios):
            budget = int(dense_scalars * ratio)
            width, stored = maximum_width(input_features, basis.shape[1], budget)
            decoder = CoordinateDecoder(input_features, width, basis.shape[1]).to(
                args.device
            )
            history = train_decoder(
                decoder,
                target=weighted_targets,
                rows=matrix_rows,
                columns=matrix_columns,
                row_features=row_features,
                column_features=column_features,
                updates=args.updates,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed + parameter_index * 100 + ratio_index,
                extra_features=extra_features,
            )
            evaluation = evaluate_streaming(
                decoder,
                targets=basis_targets,
                probabilities=probabilities,
                rows=matrix_rows,
                columns=matrix_columns,
                row_features=row_features,
                column_features=column_features,
                chunk_size=args.evaluation_chunk,
                extra_features=extra_features,
            )
            rows_out.append(
                {
                    "parameter": parameter,
                    "target": match.group("target"),
                    "budget_requested": ratio,
                    "width": width,
                    "input_features": input_features,
                    "channels": basis.shape[1],
                    "stored_scalars_including_current_mix": stored,
                    "stored_scalar_fraction": stored / dense_scalars,
                    "full_residual_trajectory_capture": (
                        retained_fractions[parameter]
                        * evaluation["weighted_best_remixed_basis_capture"]
                    ),
                    "materialization_madd_per_generated_weight": (
                        input_features * width
                        + width * width
                        + width * basis.shape[1]
                        + basis.shape[1]
                    ),
                    **{key: value for key, value in evaluation.items() if key != "per_pc_capture"},
                }
            )
            history_out.extend(
                {
                    "parameter": parameter,
                    "budget_requested": ratio,
                    **item,
                }
                for item in history
            )
            parameter_decoders[f"ratio{ratio:g}"] = {
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in decoder.state_dict().items()
                },
                "width": width,
                "per_pc_capture": evaluation["per_pc_capture"],
            }
            del decoder
        saved_decoders[parameter] = parameter_decoders
        del positions, basis, basis_targets, weighted_targets, extra_features
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

    args.output.mkdir(parents=True, exist_ok=True)
    results_path = args.output / "implicit_coordinate_residual_basis.csv"
    history_path = args.output / "fit_history.csv"
    decoders_path = args.output / "implicit_coordinate_decoders.pt"
    write_csv(results_path, rows_out)
    write_csv(history_path, history_out)
    torch.save(saved_decoders, decoders_path)
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": (
            "nanogpt_mlp_residual_w0_conditioned_basis_v1"
            if args.include_initial_weight_features
            else "nanogpt_mlp_residual_implicit_coordinate_basis_v1"
        ),
        "steps": steps,
        "snapshot_metadata": snapshot_metadata,
        "layer": args.layer,
        "targets": sorted(targets),
        "basis_rank": args.basis_rank,
        "retained_residual_energy_fraction": retained_fractions,
        "ratios": ratios,
        "features": (
            "normalized coordinate + binary digits + dyadic sine/cosine"
            + (
                " + procedural W0 value/polynomial/sign/dyadic features"
                if args.include_initial_weight_features
                else ""
            )
        ),
        "maximum_frequencies": args.maximum_frequencies,
        "include_initial_weight_features": args.include_initial_weight_features,
        "initial_weight_frequencies": args.initial_weight_frequencies,
        "training": {
            "updates": args.updates,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
        },
        "state_contract": {
            "stored": "coordinate-MLP parameters plus one current channel-mixing vector",
            "not_stored": "no residual PC, per-index embedding, hash table, ambient atom, dense shadow, or per-weight code",
            "procedural": (
                "row/column normalization, bits, dyadic Fourier features, and exact seed-regenerated W0"
                if args.include_initial_weight_features
                else "row/column normalization, bits, and dyadic Fourier features"
            ),
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
            decoders_path.name: file_sha256(decoders_path),
        },
        "limitations": [
            "Full-horizon residual PCA fitting is a noncausal representation ceiling.",
            "Finite optimization may underestimate the best coordinate MLP of each width.",
            "Naive per-weight materialization cost is reported and requires a kernel gate if recovery passes.",
            "Euclidean PCA recovery is necessary but not fixed-evaluation CE.",
        ],
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(rows_out), "metadata": str(metadata_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
