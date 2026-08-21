#!/usr/bin/env python3
"""Measure spectral concentration of private task-gradient residuals.

The exact shared-trunk checkpoint is locally untied, but no optimizer update is
taken.  For each shared depth group we subtract the group-mean c_fc/c_proj
gradient and measure whether the remaining private direction is low-rank and
stable across two independent token banks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from examples.nanogpt.analyze_shared_mlp_layer_gradient_conflict import (
    atomic_json,
    sha256_file,
    untie_mlp_weights,
)
from examples.nanogpt.model import GPT, GPTConfig
from examples.nanogpt.train import TokenBatchSource, get_batch


GROUPS = ((5, 6, 7), (8, 9, 10, 11))
RANKS = (8, 16, 32, 64, 128, 256)


def frobenius_dot(left: Tensor, right: Tensor) -> float:
    return float(torch.sum(left.to(torch.float64) * right.to(torch.float64)).item())


def cosine(left: Tensor, right: Tensor) -> float:
    denominator = math.sqrt(max(frobenius_dot(left, left) * frobenius_dot(right, right), 0.0))
    return frobenius_dot(left, right) / denominator if denominator else 0.0


def private_residuals(
    gradients: list[dict[str, Tensor]], groups: tuple[tuple[int, ...], ...] = GROUPS
) -> dict[str, dict[str, Tensor]]:
    residuals: dict[str, dict[str, Tensor]] = {}
    for group in groups:
        label = "_".join(str(layer) for layer in group)
        residuals[label] = {}
        for matrix in ("c_fc", "c_proj"):
            mean = torch.stack([gradients[layer][matrix] for layer in group]).mean(dim=0)
            for layer in group:
                residuals[label][f"layer_{layer}_{matrix}"] = gradients[layer][matrix] - mean
    return residuals


def spectrum(matrix: Tensor, ranks: tuple[int, ...] = RANKS) -> dict[str, Any]:
    values = torch.linalg.svdvals(matrix.to(torch.float32))
    energy = values.square()
    total = float(energy.sum().item())
    cumulative = energy.cumsum(0)
    return {
        "frobenius_energy": total,
        "effective_rank_entropy": float(
            torch.exp(-(energy / energy.sum()).mul((energy / energy.sum()).clamp_min(1e-30).log()).sum()).item()
        ) if total > 0.0 else 0.0,
        "rank_recovery": {
            str(rank): float(cumulative[min(rank, cumulative.numel()) - 1].item() / total)
            if total > 0.0 else 0.0
            for rank in ranks
        },
    }


def tangent_transfer(source: Tensor, target: Tensor, rank: int) -> dict[str, float]:
    """Energy of target in the tangent of rank-r factors fitted to source."""

    u, _, vh = torch.linalg.svd(source.to(torch.float32), full_matrices=False)
    rank = min(rank, u.shape[1], vh.shape[0])
    u = u[:, :rank]
    v = vh[:rank].transpose(0, 1)
    target = target.to(torch.float32)
    left = u @ (u.transpose(0, 1) @ target)
    right = (target @ v) @ v.transpose(0, 1)
    joint = u @ ((u.transpose(0, 1) @ target) @ v) @ v.transpose(0, 1)
    projection = left + right - joint
    denominator = float(target.square().sum().item())
    return {
        "recovery": float(projection.square().sum().item() / denominator) if denominator else 0.0,
        "source_rank": rank,
    }


def subspace_overlap(left: Tensor, right: Tensor, rank: int) -> dict[str, float]:
    ul, _, vhl = torch.linalg.svd(left.to(torch.float32), full_matrices=False)
    ur, _, vhr = torch.linalg.svd(right.to(torch.float32), full_matrices=False)
    rank = min(rank, ul.shape[1], ur.shape[1], vhl.shape[0], vhr.shape[0])
    left_overlap = (ul[:, :rank].transpose(0, 1) @ ur[:, :rank]).square().sum() / rank
    right_overlap = (vhl[:rank] @ vhr[:rank].transpose(0, 1)).square().sum() / rank
    return {"left": float(left_overlap.item()), "right": float(right_overlap.item())}


def collect_gradients(
    *, model: GPT, mlps: list[Any], data_dir: Path, seed: int,
    micro_batch_size: int, micro_batches: int, source: TokenBatchSource,
) -> tuple[list[dict[str, Tensor]], list[float]]:
    model.zero_grad(set_to_none=True)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    losses: list[float] = []
    model.prepare_block_fht_cache(dtype=torch.bfloat16)
    try:
        for _ in range(micro_batches):
            x, y = get_batch(
                data_dir, "train", micro_batch_size, model.config.block_size,
                "cuda", generator=generator, source=source,
            )
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
                assert loss is not None
                scaled = loss / micro_batches
            scaled.backward()
            losses.append(float(loss.detach()))
    finally:
        model.flush_block_fht_cache()
    gradients: list[dict[str, Tensor]] = []
    for layer, mlp in enumerate(mlps):
        if mlp.c_fc.weight.grad is None or mlp.c_proj.weight.grad is None:
            raise RuntimeError(f"missing MLP gradient at layer {layer}")
        row = {
            "c_fc": mlp.c_fc.weight.grad.detach().to(torch.float32).clone(),
            "c_proj": mlp.c_proj.weight.grad.detach().to(torch.float32).clone(),
        }
        if not all(torch.isfinite(value).all() for value in row.values()):
            raise RuntimeError(f"nonfinite MLP gradient at layer {layer}")
        gradients.append(row)
    return gradients, losses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--data-manifest-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--micro-batches", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs=2, default=(20260960, 20260961))
    args = parser.parse_args()

    manifest = args.data_dir / "manifest.json"
    if sha256_file(manifest) != args.data_manifest_sha256:
        raise ValueError("dataset manifest identity mismatch")
    started = time.time()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = GPTConfig(**checkpoint["model_config"])
    if not config.mlp_shared_dense_trunk or tuple(config.mlp_shared_dense_trunk_boundaries) != (1, 2, 3, 4, 5, 8, 12):
        raise ValueError("expected exact seven-trunk 1,1,1,1,1,3,4 checkpoint")
    model = GPT(config)
    model.load_state_dict(checkpoint["model"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    mlps = untie_mlp_weights(model)
    model.to("cuda")
    model.train()
    source = TokenBatchSource(args.data_dir)
    torch.cuda.reset_peak_memory_stats()

    banks: dict[str, Any] = {}
    residual_banks: dict[str, dict[str, dict[str, Tensor]]] = {}
    for index, seed in enumerate(args.seeds):
        gradients, losses = collect_gradients(
            model=model, mlps=mlps, data_dir=args.data_dir, seed=seed,
            micro_batch_size=args.micro_batch_size, micro_batches=args.micro_batches,
            source=source,
        )
        label = f"bank_{index}"
        residual_banks[label] = private_residuals(gradients)
        banks[label] = {
            "seed": seed,
            "tokens": args.micro_batch_size * args.micro_batches * config.block_size,
            "mean_cross_entropy": sum(losses) / len(losses),
            "losses": losses,
            "groups": {},
        }
        for group, matrices in residual_banks[label].items():
            rows = {name: spectrum(value) for name, value in matrices.items()}
            total = sum(row["frobenius_energy"] for row in rows.values())
            aggregate = {}
            for rank in RANKS:
                recovered = sum(
                    row["frobenius_energy"] * row["rank_recovery"][str(rank)]
                    for row in rows.values()
                )
                aggregate[str(rank)] = recovered / total if total else 0.0
            banks[label]["groups"][group] = {
                "aggregate_rank_recovery": aggregate,
                "frobenius_energy": total,
                "matrices": rows,
            }

    cross_bank: dict[str, Any] = {}
    for group in residual_banks["bank_0"]:
        rows: dict[str, Any] = {}
        matrices_a = residual_banks["bank_0"][group]
        matrices_b = residual_banks["bank_1"][group]
        total_a = sum(frobenius_dot(value, value) for value in matrices_a.values())
        total_b = sum(frobenius_dot(value, value) for value in matrices_b.values())
        dot = sum(frobenius_dot(matrices_a[name], matrices_b[name]) for name in matrices_a)
        rank_aggregate: dict[str, Any] = {}
        for rank in RANKS:
            a_to_b_num = b_to_a_num = 0.0
            left_overlap = right_overlap = 0.0
            for name, value_a in matrices_a.items():
                value_b = matrices_b[name]
                transfer_ab = tangent_transfer(value_a, value_b, rank)["recovery"]
                transfer_ba = tangent_transfer(value_b, value_a, rank)["recovery"]
                a_to_b_num += transfer_ab * frobenius_dot(value_b, value_b)
                b_to_a_num += transfer_ba * frobenius_dot(value_a, value_a)
                overlap = subspace_overlap(value_a, value_b, rank)
                left_overlap += overlap["left"]
                right_overlap += overlap["right"]
            count = len(matrices_a)
            rank_aggregate[str(rank)] = {
                "a_to_b_tangent_recovery": a_to_b_num / total_b if total_b else 0.0,
                "b_to_a_tangent_recovery": b_to_a_num / total_a if total_a else 0.0,
                "mean_left_subspace_overlap": left_overlap / count,
                "mean_right_subspace_overlap": right_overlap / count,
            }
        for name, value_a in matrices_a.items():
            rows[name] = {"cosine": cosine(value_a, matrices_b[name])}
        cross_bank[group] = {
            "aggregate_residual_cosine": dot / math.sqrt(total_a * total_b) if total_a and total_b else 0.0,
            "matrices": rows,
            "rank_aggregate": rank_aggregate,
        }

    rank64_within = min(
        banks[bank]["groups"][group]["aggregate_rank_recovery"]["64"]
        for bank in banks for group in banks[bank]["groups"]
    )
    rank64_transfer = min(
        min(entry["rank_aggregate"]["64"]["a_to_b_tangent_recovery"], entry["rank_aggregate"]["64"]["b_to_a_tangent_recovery"])
        for entry in cross_bank.values()
    )
    residual_cosine = min(entry["aggregate_residual_cosine"] for entry in cross_bank.values())
    passed = rank64_within >= 0.50 and rank64_transfer >= 0.35 and residual_cosine >= 0.25
    result = {
        "schema_version": "mai_seven_trunk_private_residual_spectrum_result_v1",
        "classification": "RANK64_PRIVATE_RESIDUAL_SUPPORTED" if passed else "RANK64_PRIVATE_RESIDUAL_REJECTED",
        "passed": passed,
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": sha256_file(args.checkpoint),
            "next_iter": checkpoint["next_iter"],
            "best_val_loss": checkpoint["best_val_loss"],
            "model_config_sha256": hashlib.sha256(json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        },
        "dataset_manifest": {"path": str(manifest.resolve()), "sha256": args.data_manifest_sha256},
        "measurement": {"split": "train", "micro_batch_size": args.micro_batch_size, "micro_batches": args.micro_batches, "language_model_updates": 0},
        "banks": banks,
        "cross_bank": cross_bank,
        "gate_summary": {
            "minimum_rank64_within_bank_recovery": rank64_within,
            "minimum_rank64_cross_bank_tangent_recovery": rank64_transfer,
            "minimum_shared_group_cross_bank_residual_cosine": residual_cosine,
            "thresholds": {"within": 0.50, "transfer": 0.35, "cosine": 0.25},
        },
        "parameter_accounting": {
            "rank64_additional_private_parameters": 3440640,
            "rank64_total_mlp_parameters": 36516864,
            "rank64_mlp_parameter_compression_ratio": 56623104 / 36516864,
            "rank64_mlp_parameter_saving_fraction": 1.0 - 36516864 / 56623104,
            "cached_inference_extra_flops": 0,
        },
        "wall_seconds": time.time() - started,
        "maximum_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
    }
    atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
