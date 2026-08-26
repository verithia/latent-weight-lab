#!/usr/bin/env python3
"""Audit terminal gradient coverage of a gradient-seeded low-rank MLP.

The training candidate uses ``W = W_seed + scale * A B^T``.  This audit
computes a fresh dense validation gradient at the terminal checkpoint and
separates three questions:

1. How much of that gradient lies in the *current* rank-r matrix-manifold
   tangent ``{U X^T + Y V^T}``?
2. How far are the learned factor spaces from the best rank-r singular spaces
   of the fresh gradient?
3. Even inside the tangent, how well does ordinary Euclidean optimization of
   ``A`` and ``B`` align with the orthogonal ambient tangent projection?

Only selected layers are audited so no additional trajectory checkpoints are
needed.  The procedural seed is reconstructed from ``model_seed`` exactly as
in training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.model import GPT, GPTConfig, GradientSeededLowRankLinear
from examples.nanogpt.train import (
    TokenBatchSource,
    fixed_eval_indices_digest,
    get_batch,
    make_fixed_eval_indices,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = left.double().square().sum().sqrt() * right.double().square().sum().sqrt()
    if float(denominator) == 0.0:
        return 0.0
    return float((left.double() * right.double()).sum() / denominator)


def canonical_overlap(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    squared = torch.linalg.svdvals(left.T @ right).double().square()
    return {
        "mean_squared_cosine": float(squared.mean()),
        "minimum_squared_cosine": float(squared.min()),
        "maximum_squared_cosine": float(squared.max()),
    }


def terminal_gradient_metrics(
    gradient: torch.Tensor,
    left_factor: torch.Tensor,
    right_factor: torch.Tensor,
    left_factor_gradient: torch.Tensor,
    right_factor_gradient: torch.Tensor,
    *,
    scale: float,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    """Return exact tangent and factor-pullback metrics for one matrix."""
    gradient = gradient.float()
    left_factor = left_factor.float()
    right_factor = right_factor.float()
    left_basis = torch.linalg.qr(left_factor, mode="reduced").Q
    right_basis = torch.linalg.qr(right_factor, mode="reduced").Q

    left_projection = left_basis @ (left_basis.T @ gradient)
    right_projection = (gradient @ right_basis) @ right_basis.T
    intersection = left_basis @ ((left_basis.T @ gradient) @ right_basis) @ right_basis.T
    tangent_projection = left_projection + right_projection - intersection
    total_energy = gradient.double().square().sum().clamp_min(1e-30)

    u, singular_values, vh = torch.linalg.svd(gradient, full_matrices=False)
    rank = left_factor.shape[1]
    best_left = u[:, :rank]
    best_right = vh[:rank].T
    best_rank_energy = singular_values[:rank].double().square().sum()

    # If factor-space SGD uses -dL/dA and -dL/dB, its first-order ambient
    # descent is the negative of this pullback vector.  We compare positive
    # gradient-space quantities, so one means perfect descent alignment.
    factor_pullback = float(scale) * (
        left_factor_gradient.float() @ right_factor.T
        + left_factor @ right_factor_gradient.float().T
    )
    tangent_energy = tangent_projection.double().square().sum().clamp_min(1e-30)

    metrics = {
        "gradient_energy": float(total_energy),
        "left_tangent_capture": float(left_projection.double().square().sum() / total_energy),
        "right_tangent_capture": float(right_projection.double().square().sum() / total_energy),
        "intersection_capture": float(intersection.double().square().sum() / total_energy),
        "tangent_capture": float(tangent_energy / total_energy),
        "residual_gradient_fraction": float(1.0 - tangent_energy / total_energy),
        "best_rank_capture": float(best_rank_energy / total_energy),
        "factor_pullback_gradient_cosine": cosine(factor_pullback, gradient),
        "factor_pullback_tangent_cosine": cosine(factor_pullback, tangent_projection),
        "factor_pullback_norm_over_tangent": float(
            factor_pullback.double().square().sum().sqrt() / tangent_energy.sqrt()
        ),
        "left_factor_condition": float(torch.linalg.cond(left_factor)),
        "right_factor_condition": float(torch.linalg.cond(right_factor)),
        "delta_rank": int(torch.linalg.matrix_rank(left_factor @ right_factor.T)),
        **{
            f"left_current_vs_gradient_top{rank}_{key}": value
            for key, value in canonical_overlap(left_basis, best_left).items()
        },
        **{
            f"right_current_vs_gradient_top{rank}_{key}": value
            for key, value in canonical_overlap(right_basis, best_right).items()
        },
    }
    tensors = {
        "gradient": gradient,
        "tangent_projection": tangent_projection,
        "factor_pullback": factor_pullback,
    }
    return metrics, tensors


def aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    metric_names = (
        "left_tangent_capture",
        "right_tangent_capture",
        "tangent_capture",
        "residual_gradient_fraction",
        "best_rank_capture",
        "factor_pullback_gradient_cosine",
        "factor_pullback_tangent_cosine",
        "factor_pullback_norm_over_tangent",
    )
    result: dict[str, dict[str, float]] = {}
    for target in ("all", "mlp.c_fc", "mlp.c_proj"):
        selected = rows if target == "all" else [row for row in rows if row["target"] == target]
        total_energy = sum(float(row["gradient_energy"]) for row in selected)
        result[target] = {}
        for metric in metric_names:
            result[target][f"{metric}_energy_weighted_mean"] = sum(
                float(row["gradient_energy"]) * float(row[metric]) for row in selected
            ) / max(total_energy, 1e-30)
            result[target][f"{metric}_mean"] = sum(float(row[metric]) for row in selected) / len(selected)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", default="0,6,11")
    parser.add_argument("--model-seed", type=int, default=1337)
    parser.add_argument("--eval-seed", type=int, default=20260715)
    parser.add_argument("--eval-iters", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    args = parser.parse_args()
    started = time.time()
    layers = {int(value) for value in args.layers.split(",") if value}
    checkpoint_sha256 = file_sha256(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = GPTConfig(**checkpoint["model_config"])

    torch.manual_seed(args.model_seed)
    model = GPT(config)
    incompatible = model.load_state_dict(checkpoint["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"checkpoint incompatibility: {incompatible}")
    model.to(args.device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    selected: list[tuple[int, str, GradientSeededLowRankLinear]] = []
    for layer in sorted(layers):
        for target, module in (
            ("mlp.c_fc", model.transformer.h[layer].mlp.c_fc),
            ("mlp.c_proj", model.transformer.h[layer].mlp.c_proj),
        ):
            if not isinstance(module, GradientSeededLowRankLinear):
                raise TypeError(f"layer {layer} {target} is {type(module).__name__}")
            module.weight.requires_grad_(True)
            module.gradient_seeded_left.requires_grad_(True)
            module.gradient_seeded_right.requires_grad_(True)
            selected.append((layer, target, module))

    fixed_indices = make_fixed_eval_indices(
        args.data_dir,
        args.batch_size,
        config.block_size,
        args.eval_iters,
        args.eval_seed,
    )
    source = TokenBatchSource(args.data_dir)
    x, y = get_batch(
        args.data_dir,
        "val",
        args.batch_size,
        config.block_size,
        args.device,
        indices=fixed_indices["val"][0],
        source=source,
    )
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    device_type = "cuda" if "cuda" in args.device else "cpu"
    context = torch.autocast(device_type=device_type, dtype=dtype, enabled=device_type == "cuda")
    with context:
        _, loss = model(x, y)
    if loss is None:
        raise RuntimeError("validation loss was not produced")
    loss.backward()

    rows: list[dict[str, Any]] = []
    for layer, target, module in selected:
        if module.weight.grad is None:
            raise RuntimeError(f"missing dense gradient for layer {layer} {target}")
        if module.gradient_seeded_left.grad is None or module.gradient_seeded_right.grad is None:
            raise RuntimeError(f"missing factor gradient for layer {layer} {target}")
        metrics, _ = terminal_gradient_metrics(
            module.weight.grad,
            module.gradient_seeded_left,
            module.gradient_seeded_right,
            module.gradient_seeded_left.grad,
            module.gradient_seeded_right.grad,
            scale=module.scale,
        )
        rows.append(
            {
                "layer": layer,
                "target": target,
                "rank": module.rank,
                "factor_scalar_fraction": module.rank
                * (module.in_features + module.out_features)
                / (module.in_features * module.out_features),
                **metrics,
            }
        )

    output = {
        "schema_version": "gradient_seeded_lowrank_terminal_audit_v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_next_iter": int(checkpoint["next_iter"]),
        "checkpoint_best_val_loss": float(checkpoint["best_val_loss"]),
        "model_seed": args.model_seed,
        "eval_seed": args.eval_seed,
        "fixed_eval_indices_sha256": fixed_eval_indices_digest(fixed_indices),
        "fixed_val_batch_index": 0,
        "batch_size": args.batch_size,
        "validation_loss_on_audit_batch": float(loss.detach()),
        "layers": sorted(layers),
        "rows": rows,
        "aggregate": aggregate(rows),
        "runtime_seconds": time.time() - started,
        "source_commit": git_commit(Path(__file__).resolve().parents[2]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    print(f"output_sha256={file_sha256(args.output)}")


if __name__ == "__main__":
    main()
