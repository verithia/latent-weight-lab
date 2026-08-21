#!/usr/bin/env python3
"""Measure layer-axis task-gradient conflict in a shared dense MLP checkpoint.

The checkpoint is loaded exactly, then only the tied dense MLP matrices are
cloned into layer-private Parameters.  All other parameters are frozen.  A
fixed effective training batch yields the gradient each layer would request if
the sharing constraint were locally released.  No optimizer update is taken.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor, nn

from examples.nanogpt.model import GPT, GPTConfig, MLP
from examples.nanogpt.train import TokenBatchSource, get_batch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _dot(left: Iterable[Tensor], right: Iterable[Tensor]) -> float:
    paired = list(zip(left, right, strict=True))
    if not paired:
        raise ValueError("at least one tensor pair is required")
    total = torch.zeros((), device=paired[0][0].device, dtype=torch.float64)
    for lhs, rhs in paired:
        total += torch.sum(lhs.to(torch.float64) * rhs.to(torch.float64))
    return float(total.item())


def gradient_cosine(left: tuple[Tensor, ...], right: tuple[Tensor, ...]) -> float:
    numerator = _dot(left, right)
    denominator = math.sqrt(max(_dot(left, left) * _dot(right, right), 0.0))
    return numerator / denominator if denominator > 0.0 else 0.0


def common_update_energy_fraction(gradients: list[tuple[Tensor, ...]]) -> float:
    """Return ||sum g||^2 / (n sum ||g||^2), bounded by [0, 1]."""

    if not gradients:
        raise ValueError("at least one gradient is required")
    components = len(gradients[0])
    if any(len(gradient) != components for gradient in gradients):
        raise ValueError("gradient tuples must have equal arity")
    denominator = len(gradients) * sum(_dot(g, g) for g in gradients)
    if denominator <= 0.0:
        return 0.0
    summed = tuple(
        torch.stack([gradient[index] for gradient in gradients], dim=0).sum(dim=0)
        for index in range(components)
    )
    return min(1.0, max(0.0, _dot(summed, summed) / denominator))


def contiguous_group_recoveries(
    gradients: list[tuple[Tensor, ...]], group_count: int
) -> list[float]:
    if group_count <= 0 or len(gradients) % group_count:
        raise ValueError("group_count must positively divide the layer count")
    size = len(gradients) // group_count
    return [
        common_update_energy_fraction(gradients[start : start + size])
        for start in range(0, len(gradients), size)
    ]


def boundary_group_recoveries(
    gradients: list[tuple[Tensor, ...]], boundaries: tuple[int, ...]
) -> list[float]:
    if (
        not boundaries
        or boundaries[-1] != len(gradients)
        or any(value <= 0 or value > len(gradients) for value in boundaries)
        or any(later <= earlier for earlier, later in zip(boundaries, boundaries[1:]))
    ):
        raise ValueError("boundaries must increase strictly and end at the layer count")
    return [
        common_update_energy_fraction(gradients[start:stop])
        for start, stop in zip((0, *boundaries[:-1]), boundaries, strict=True)
    ]


def gradient_gram(gradients: list[tuple[Tensor, ...]]) -> list[list[float]]:
    """Materialize the small layer-axis Gram matrix once."""

    return [[_dot(left, right) for right in gradients] for left in gradients]


def boundary_group_recoveries_from_gram(
    gram: list[list[float]], boundaries: tuple[int, ...]
) -> list[float]:
    layer_count = len(gram)
    if any(len(row) != layer_count for row in gram):
        raise ValueError("gradient Gram matrix must be square")
    if (
        not boundaries
        or boundaries[-1] != layer_count
        or any(value <= 0 or value > layer_count for value in boundaries)
        or any(later <= earlier for earlier, later in zip(boundaries, boundaries[1:]))
    ):
        raise ValueError("boundaries must increase strictly and end at the layer count")
    recoveries: list[float] = []
    for start, stop in zip((0, *boundaries[:-1]), boundaries, strict=True):
        group_size = stop - start
        numerator = sum(
            gram[left][right]
            for left in range(start, stop)
            for right in range(start, stop)
        )
        denominator = group_size * sum(gram[index][index] for index in range(start, stop))
        recovery = numerator / denominator if denominator > 0.0 else 0.0
        recoveries.append(min(1.0, max(0.0, recovery)))
    return recoveries


def contiguous_partition_frontier(
    gram: list[list[float]],
    *,
    minimum_group_count: int,
    maximum_group_count: int,
    minimum_recovery: float,
    minimum_mean_recovery: float,
) -> dict[str, Any]:
    """Enumerate layer-contiguous partitions without revisiting ambient tensors."""

    layer_count = len(gram)
    if not (1 <= minimum_group_count <= maximum_group_count <= layer_count):
        raise ValueError("group-count range must lie within the layer count")
    per_group_count: dict[str, Any] = {}
    selected: dict[str, Any] | None = None
    for group_count in range(minimum_group_count, maximum_group_count + 1):
        candidates: list[dict[str, Any]] = []
        passing = 0
        for cuts in combinations(range(1, layer_count), group_count - 1):
            boundaries = (*cuts, layer_count)
            recoveries = boundary_group_recoveries_from_gram(gram, boundaries)
            mean = sum(recoveries) / len(recoveries)
            minimum = min(recoveries)
            passes = minimum >= minimum_recovery and mean >= minimum_mean_recovery
            passing += int(passes)
            candidates.append(
                {
                    "boundaries": list(boundaries),
                    "per_group": recoveries,
                    "mean": mean,
                    "minimum": minimum,
                    "passes": passes,
                }
            )
        candidates.sort(
            key=lambda candidate: (
                -candidate["minimum"],
                -candidate["mean"],
                candidate["boundaries"],
            )
        )
        best = candidates[0]
        per_group_count[str(group_count)] = {
            "evaluated_partitions": len(candidates),
            "passing_partitions": passing,
            "best_by_minimum_then_mean": best,
        }
        if selected is None and best["passes"]:
            selected = {"group_count": group_count, **best}
    return {
        "minimum_group_count": minimum_group_count,
        "maximum_group_count": maximum_group_count,
        "minimum_recovery_threshold": minimum_recovery,
        "minimum_mean_recovery_threshold": minimum_mean_recovery,
        "selection_rule": "smallest passing group count; within count maximize minimum then mean recovery",
        "per_group_count": per_group_count,
        "selected": selected,
    }


def untie_mlp_weights(model: GPT) -> list[MLP]:
    mlps: list[MLP] = []
    for block in model.transformer.h:
        if not isinstance(block.mlp, MLP):
            raise TypeError("layer-gradient conflict audit requires dense MLP layers")
        block.mlp.c_fc.weight = nn.Parameter(block.mlp.c_fc.weight.detach().clone())
        block.mlp.c_proj.weight = nn.Parameter(block.mlp.c_proj.weight.detach().clone())
        mlps.append(block.mlp)
    return mlps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--data-manifest-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--micro-batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    if args.micro_batch_size <= 0 or args.micro_batches <= 0:
        raise ValueError("batch sizes must be positive")
    manifest = args.data_dir / "manifest.json"
    if sha256_file(manifest) != args.data_manifest_sha256:
        raise ValueError("dataset manifest identity mismatch")

    started = time.time()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = GPTConfig(**checkpoint["model_config"])
    if not config.mlp_shared_dense_trunk:
        raise ValueError("audit requires a shared-dense-trunk checkpoint")
    model = GPT(config)
    model.load_state_dict(checkpoint["model"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    mlps = untie_mlp_weights(model)
    model.to("cuda")
    model.train()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    source = TokenBatchSource(args.data_dir)
    losses: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    model.prepare_block_fht_cache(dtype=torch.bfloat16)
    try:
        for _ in range(args.micro_batches):
            x, y = get_batch(
                args.data_dir,
                "train",
                args.micro_batch_size,
                config.block_size,
                "cuda",
                generator=generator,
                source=source,
            )
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
                assert loss is not None
                scaled = loss / args.micro_batches
            scaled.backward()
            losses.append(float(loss.detach()))
    finally:
        model.flush_block_fht_cache()

    combined: list[tuple[Tensor, ...]] = []
    c_fc: list[tuple[Tensor, ...]] = []
    c_proj: list[tuple[Tensor, ...]] = []
    for layer, mlp in enumerate(mlps):
        if mlp.c_fc.weight.grad is None or mlp.c_proj.weight.grad is None:
            raise RuntimeError(f"layer {layer} did not receive both MLP gradients")
        fc_gradient = mlp.c_fc.weight.grad.detach()
        proj_gradient = mlp.c_proj.weight.grad.detach()
        if not torch.isfinite(fc_gradient).all() or not torch.isfinite(proj_gradient).all():
            raise RuntimeError(f"layer {layer} gradient is nonfinite")
        c_fc.append((fc_gradient,))
        c_proj.append((proj_gradient,))
        combined.append((fc_gradient, proj_gradient))

    combined_gram = gradient_gram(combined)
    pairwise = []
    for i in range(config.n_layer):
        row: list[float] = []
        for j in range(config.n_layer):
            denominator = math.sqrt(
                max(combined_gram[i][i] * combined_gram[j][j], 0.0)
            )
            row.append(combined_gram[i][j] / denominator if denominator > 0.0 else 0.0)
        pairwise.append(row)
    partitions: dict[str, Any] = {}
    for group_count in (1, 2, 3, 4, 6, 12):
        entry: dict[str, Any] = {}
        for name, gradients in (("combined", combined), ("c_fc", c_fc), ("c_proj", c_proj)):
            values = contiguous_group_recoveries(gradients, group_count)
            entry[name] = {
                "per_group": values,
                "mean": sum(values) / len(values),
                "minimum": min(values),
            }
        partitions[str(group_count)] = entry

    configured_boundaries = config.mlp_shared_dense_trunk_boundaries
    if configured_boundaries is None:
        group_size = config.n_layer // config.mlp_shared_dense_trunk_groups
        measured_boundaries = tuple(
            range(group_size, config.n_layer + 1, group_size)
        )
    else:
        measured_boundaries = tuple(configured_boundaries)
    nonuniform: dict[str, Any] = {}
    for name, gradients in (("combined", combined), ("c_fc", c_fc), ("c_proj", c_proj)):
        values = boundary_group_recoveries(gradients, measured_boundaries)
        nonuniform[name] = {
            "per_group": values,
            "mean": sum(values) / len(values),
            "minimum": min(values),
        }

    result = {
        "schema_version": "mai_shared_mlp_layer_gradient_conflict_v1",
        "classification": "MEASUREMENT",
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": sha256_file(args.checkpoint),
            "next_iter": checkpoint["next_iter"],
            "best_val_loss": checkpoint["best_val_loss"],
            "model_config_sha256": hashlib.sha256(
                json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        },
        "dataset_manifest": {
            "path": str(manifest.resolve()),
            "sha256": args.data_manifest_sha256,
        },
        "measurement": {
            "split": "train",
            "seed": args.seed,
            "micro_batch_size": args.micro_batch_size,
            "micro_batches": args.micro_batches,
            "tokens": args.micro_batch_size * args.micro_batches * config.block_size,
            "mean_loss": sum(losses) / len(losses),
            "losses": losses,
            "definition": "locally untied per-layer task gradients; no optimizer update",
        },
        "pairwise_combined_cosine": pairwise,
        "contiguous_partitions": partitions,
        "measured_current_partition": {
            "boundaries": list(measured_boundaries),
            "layers": [
                list(range(start, stop))
                for start, stop in zip(
                    (0, *measured_boundaries[:-1]), measured_boundaries, strict=True
                )
            ],
            "metrics": nonuniform,
        },
        "combined_contiguous_partition_frontier": contiguous_partition_frontier(
            combined_gram,
            minimum_group_count=2,
            maximum_group_count=8,
            minimum_recovery=0.8,
            minimum_mean_recovery=0.9,
        ),
        "parameter_accounting": {
            "dense_twelve_layer_mlp_parameters": 8 * config.n_embd * config.n_embd * config.n_layer,
            "parameters_per_full_rank_trunk_pair": 8 * config.n_embd * config.n_embd,
            "layer_private_diagonal_parameters": 5 * config.n_embd * config.n_layer,
            "definition": "K paired dense trunks plus one 4d pre-GELU and one d residual-write gain per layer",
        },
        "wall_seconds": time.time() - started,
        "maximum_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
    }
    atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
