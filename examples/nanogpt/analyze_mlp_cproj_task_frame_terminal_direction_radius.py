#!/usr/bin/env python3
"""Separate task-frame direction from radius at a fixed terminal checkpoint.

This is a zero-update inference diagnostic. It changes only the three saved
frame-coordinate groups, evaluates preregistered variants on shared token
windows, restores the native state, and verifies every non-frame tensor is
bitwise unchanged.
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

import numpy as np
import torch

from examples.nanogpt.analyze_mlp_activation_chart_oracle import tensor_sha256
from examples.nanogpt.model import GPT, GPTConfig
from examples.nanogpt.optimize_mlp_bilateral_endpoint_ce import (
    clear_frozen_base_cache,
    evaluate_ce,
    prepare_frozen_base_cache,
    sha256,
)
from examples.nanogpt.optimize_mlp_cproj_errorfeedback_task_frame_endpoint import (
    capture_frame_state,
    frame_parameters,
    restore_frame_state,
)


PLAN_NAME = "124m_mlp_cproj_task_frame_terminal_direction_radius_plan.json"
SCHEMA = "mai_124m_mlp_cproj_task_frame_terminal_direction_radius_result_v1"
VARIANTS = (
    "native_delayed",
    "identity",
    "endpoint_full_radius",
    "endpoint_direction_native_radius",
    "native_direction_endpoint_radius",
)
GROUP_SUFFIXES = (
    "pregelu_rotation",
    "hidden_rotation",
    "output_rotation",
)


def git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_sha(path: Path, expected: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual} != {expected}")
    return actual


def fixed_token_windows(
    data_dir: Path,
    *,
    batch_size: int,
    block_size: int,
    batches: int,
    seed: int,
) -> list[torch.Tensor]:
    values = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")
    if len(values) <= block_size + 1:
        raise ValueError("validation data is shorter than the requested window")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    indices = torch.randint(
        len(values) - block_size - 1,
        (int(batches), int(batch_size)),
        generator=generator,
    )
    return [
        torch.stack(
            [
                torch.from_numpy(
                    np.asarray(
                        values[
                            int(index) : int(index) + block_size + 1
                        ],
                        dtype=np.int64,
                    )
                )
                for index in row
            ]
        )
        for row in indices
    ]


def group_keys(state: dict[str, torch.Tensor], suffix: str) -> list[str]:
    keys = sorted(key for key in state if key.endswith(f".{suffix}"))
    if len(keys) != 12:
        raise ValueError(f"expected 12 {suffix} tensors, got {len(keys)}")
    return keys


def state_group_norm(state: dict[str, torch.Tensor], suffix: str) -> float:
    squared = sum(
        float(state[key].detach().float().square().sum())
        for key in group_keys(state, suffix)
    )
    return math.sqrt(squared)


def scaled_state_to_reference_norm(
    direction: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    if set(direction) != set(reference):
        raise ValueError("direction/reference frame keys differ")
    output: dict[str, torch.Tensor] = {}
    scales: dict[str, float] = {}
    for suffix in GROUP_SUFFIXES:
        direction_norm = state_group_norm(direction, suffix)
        reference_norm = state_group_norm(reference, suffix)
        if direction_norm <= 0.0:
            raise ValueError(f"zero direction norm for {suffix}")
        scale = reference_norm / direction_norm
        scales[suffix] = scale
        for key in group_keys(direction, suffix):
            output[key] = direction[key].detach().float().clone() * scale
    return output, scales


def make_variants(
    native: dict[str, torch.Tensor],
    endpoint: dict[str, torch.Tensor],
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, dict[str, float]]]:
    if set(native) != set(endpoint):
        raise ValueError("native/endpoint frame keys differ")
    identity = {key: torch.zeros_like(value) for key, value in native.items()}
    endpoint_at_native, endpoint_to_native = scaled_state_to_reference_norm(
        endpoint, native
    )
    native_at_endpoint, native_to_endpoint = scaled_state_to_reference_norm(
        native, endpoint
    )
    variants = {
        "native_delayed": {
            key: value.detach().float().clone() for key, value in native.items()
        },
        "identity": identity,
        "endpoint_full_radius": {
            key: value.detach().float().clone() for key, value in endpoint.items()
        },
        "endpoint_direction_native_radius": endpoint_at_native,
        "native_direction_endpoint_radius": native_at_endpoint,
    }
    if tuple(variants) != VARIANTS:
        raise RuntimeError("variant order drifted from the registered protocol")
    return variants, {
        "endpoint_direction_to_native_radius": endpoint_to_native,
        "native_direction_to_endpoint_radius": native_to_endpoint,
    }


def nonframe_state_digest(model: GPT) -> str:
    frame_ids = {id(parameter) for parameter in frame_parameters(model).values()}
    frame_names = {
        name
        for name, parameter in model.named_parameters()
        if id(parameter) in frame_ids
    }
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if name in frame_names:
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def variant_gains(
    ce_by_variant: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    splits = ("primary", "confirmation")
    native = ce_by_variant["native_delayed"]
    return {
        variant: {
            split: native[split] - ce_by_variant[variant][split]
            for split in splits
        }
        for variant in VARIANTS
        if variant != "native_delayed"
    }


def select_decision(
    ce_by_variant: dict[str, dict[str, float]],
    *,
    minimum_gain: float,
    minimum_fraction: float,
) -> dict[str, Any]:
    gains = variant_gains(ce_by_variant)
    means = {
        variant: sum(values.values()) / len(values)
        for variant, values in gains.items()
    }

    def nonnegative(variant: str) -> bool:
        return all(value >= 0.0 for value in gains[variant].values())

    full = "endpoint_full_radius"
    endpoint_small = "endpoint_direction_native_radius"
    native_large = "native_direction_endpoint_radius"
    portable = means[full] >= minimum_gain and nonnegative(full)
    denominator = means[full]

    def sufficient(variant: str) -> bool:
        fraction = (
            means[variant] / denominator if denominator > 0.0 else float("-inf")
        )
        return (
            portable
            and means[variant] >= minimum_gain
            and nonnegative(variant)
            and fraction >= minimum_fraction
        )

    endpoint_small_ok = sufficient(endpoint_small)
    native_large_ok = sufficient(native_large)
    if not portable:
        decision = "ENDPOINT_NOT_PORTABLE"
    elif native_large_ok and not endpoint_small_ok:
        decision = "AMPLITUDE_DOMINATES"
    elif endpoint_small_ok and not native_large_ok:
        decision = "DIRECTION_DOMINATES"
    elif endpoint_small_ok and native_large_ok:
        decision = "BOTH_CONTROLLED_VARIANTS_SUFFICIENT"
    else:
        decision = "BOTH_DIRECTION_AND_RADIUS_MATTER"
    return {
        "decision": decision,
        "gain_by_window": gains,
        "mean_gain": means,
        "gain_fraction_of_endpoint_full": {
            variant: (
                means[variant] / denominator
                if denominator > 0.0
                else None
            )
            for variant in (endpoint_small, native_large)
        },
        "thresholds": {
            "minimum_mean_gain": minimum_gain,
            "both_windows_nonnegative_required": True,
            "sufficiency_fraction_of_endpoint_full_gain": minimum_fraction,
        },
        "automatic_training_run_authorized": False,
        "automatic_rerun_authorized": False,
        "larger_rung_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--endpoint-state", required=True, type=Path)
    parser.add_argument("--endpoint-state-sha256", required=True)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--primary-seed", type=int, default=20260883)
    parser.add_argument("--confirmation-seed", type=int, default=20260884)
    parser.add_argument("--minimum-gain", type=float, default=0.005)
    parser.add_argument("--minimum-fraction", type=float, default=0.5)
    args = parser.parse_args()
    if not args.device.startswith("cuda"):
        parser.error("the registered diagnostic requires CUDA")
    if args.batch_size != 16 or args.block_size != 1024 or args.eval_batches != 32:
        parser.error("evaluation shape differs from the registered protocol")
    if args.primary_seed != 20260883 or args.confirmation_seed != 20260884:
        parser.error("evaluation seeds differ from the registered protocol")
    if args.output.exists():
        parser.error("output already exists")

    root = Path(__file__).resolve().parents[2]
    plan = root / "examples/nanogpt/configs/selection_artifacts" / PLAN_NAME
    manifest = args.data_dir / "manifest.json"
    input_hashes = {
        "checkpoint": validate_sha(args.checkpoint, args.checkpoint_sha256),
        "endpoint_state": validate_sha(
            args.endpoint_state, args.endpoint_state_sha256
        ),
        "dataset_manifest": validate_sha(manifest, args.manifest_sha256),
        "plan": sha256(plan),
    }
    plan_payload = load_json(plan)
    if plan_payload["protocol"]["training_updates"] != 0:
        raise ValueError("plan no longer describes a zero-update diagnostic")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != "nanogpt_exact_resume_v2":
        raise ValueError("checkpoint is not exact-resume-v2")
    if int(checkpoint.get("next_iter", -1)) != 238:
        raise ValueError("checkpoint is not the registered terminal update")
    with torch.device(args.device):
        model = GPT(GPTConfig(**checkpoint["model_config"]))
    incompatible = model.load_state_dict(checkpoint["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict checkpoint restoration reported incompatibility")
    model.to(args.device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    cached_modules = prepare_frozen_base_cache(model, torch.bfloat16)
    native = capture_frame_state(model)
    endpoint_payload = torch.load(
        args.endpoint_state, map_location="cpu", weights_only=False
    )
    endpoint = endpoint_payload.get("state")
    if not isinstance(endpoint, dict):
        raise ValueError("endpoint state payload is invalid")
    variants, scales = make_variants(native, endpoint)
    before_digest = nonframe_state_digest(model)

    split_seeds = {
        "primary": args.primary_seed,
        "confirmation": args.confirmation_seed,
    }
    windows = {
        split: fixed_token_windows(
            args.data_dir,
            batch_size=args.batch_size,
            block_size=args.block_size,
            batches=args.eval_batches,
            seed=seed,
        )
        for split, seed in split_seeds.items()
    }
    token_hashes = {
        split: tensor_sha256(torch.cat(batches))
        for split, batches in windows.items()
    }
    ce_by_variant: dict[str, dict[str, float]] = {}
    performance: dict[str, dict[str, float]] = {}
    try:
        torch.cuda.reset_peak_memory_stats()
        for variant, state in variants.items():
            restore_frame_state(model, state)
            ce_by_variant[variant] = {}
            performance[variant] = {}
            for split, batches in windows.items():
                torch.cuda.synchronize()
                started = time.perf_counter()
                ce = evaluate_ce(model, batches, args.device)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                if not math.isfinite(ce):
                    raise RuntimeError("non-finite validation CE")
                ce_by_variant[variant][split] = ce
                performance[variant][f"{split}_elapsed_seconds"] = elapsed
                performance[variant][f"{split}_tokens_per_second"] = (
                    args.eval_batches
                    * args.batch_size
                    * args.block_size
                    / elapsed
                )
                print(
                    f"eval variant={variant} split={split} ce={ce:.8f} "
                    f"elapsed_s={elapsed:.3f}",
                    flush=True,
                )
    finally:
        restore_frame_state(model, native)
        clear_frozen_base_cache(model)
    after_digest = nonframe_state_digest(model)
    if before_digest != after_digest:
        raise RuntimeError("non-frame state changed during zero-update diagnostic")

    selection = select_decision(
        ce_by_variant,
        minimum_gain=args.minimum_gain,
        minimum_fraction=args.minimum_fraction,
    )
    args.output.mkdir(parents=True)
    summary = {
        "schema_version": SCHEMA,
        "created_at": "2026-08-05",
        "repository_commit": git_head(root),
        "entrypoint": "examples.nanogpt.analyze_mlp_cproj_task_frame_terminal_direction_radius",
        "command": [str(value) for value in __import__("sys").argv],
        "source_sha256": sha256(Path(__file__).resolve()),
        "input_hashes": input_hashes,
        "protocol": {
            "training_updates": 0,
            "variant_order": list(VARIANTS),
            "batch_size": args.batch_size,
            "block_size": args.block_size,
            "eval_batches": args.eval_batches,
            "split_seeds": split_seeds,
            "token_sha256": token_hashes,
            "cached_block_fht_modules": cached_modules,
        },
        "scale_factors": scales,
        "ce_by_variant": ce_by_variant,
        "performance": performance,
        "peak_mib": torch.cuda.max_memory_allocated() / (1024 * 1024),
        "nonframe_state_sha256_before": before_digest,
        "nonframe_state_sha256_after": after_digest,
        "nonframe_state_bitwise_preserved": True,
        "selection": selection,
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(selection, indent=2, sort_keys=True), flush=True)
    print(f"summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
