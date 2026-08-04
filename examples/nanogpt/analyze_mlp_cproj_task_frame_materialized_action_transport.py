#!/usr/bin/env python3
"""Test additive transport of a frozen endpoint's materialized MLP action.

The diagnostic performs no language-model update. It derives endpoint-induced
materialized c_fc/c_proj deltas at the parent base, adds those exact deltas to
the delayed terminal bases, evaluates fixed variants, then discards all
temporary weight overrides.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_mlp_activation_chart_oracle import tensor_sha256
from examples.nanogpt.analyze_mlp_cproj_task_frame_terminal_direction_radius import (
    fixed_token_windows,
    git_head,
    nonframe_state_digest,
    validate_sha,
)
from examples.nanogpt.model import GPT, GPTConfig
from examples.nanogpt.optimize_mlp_bilateral_endpoint_ce import (
    autocast_context,
    clear_frozen_base_cache,
    evaluate_ce,
    prepare_frozen_base_cache,
    sha256,
)
from examples.nanogpt.optimize_mlp_cproj_errorfeedback_task_frame_endpoint import (
    capture_frame_state,
    load_frame_model,
    restore_frame_state,
)


PLAN_NAME = "124m_mlp_cproj_task_frame_materialized_action_transport_plan.json"
SCHEMA = "mai_124m_mlp_cproj_task_frame_materialized_action_transport_result_v1"
VARIANTS = (
    "native_delayed",
    "identity",
    "raw_endpoint_coordinates",
    "additive_materialized_full",
    "additive_materialized_cfc_only",
    "additive_materialized_cproj_only",
)


def load_delayed_model(checkpoint: dict[str, Any], device: str) -> GPT:
    with torch.device(device):
        model = GPT(GPTConfig(**checkpoint["model_config"]))
    incompatible = model.load_state_dict(checkpoint["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict delayed checkpoint restoration failed")
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def identity_state(native: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: torch.zeros_like(value) for key, value in native.items()}


@torch.no_grad()
def materialized_parent_actions(
    parent: GPT,
    endpoint: dict[str, torch.Tensor],
) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, float]]]:
    restore_frame_state(parent, identity_state(capture_frame_state(parent)))
    bases: list[tuple[torch.Tensor, torch.Tensor]] = []
    for block in parent.transformer.h:
        bases.append(
            (
                block.mlp._cfc_base_weight().detach().clone(),
                block.mlp._cproj_base_weight().detach().clone(),
            )
        )
    restore_frame_state(parent, endpoint)
    actions: list[dict[str, torch.Tensor]] = []
    metrics: list[dict[str, float]] = []
    for layer, (block, (cfc_base, cproj_base)) in enumerate(
        zip(parent.transformer.h, bases, strict=True)
    ):
        cfc_endpoint = block.mlp._materialize_charted_cfc_weight(cfc_base)
        cproj_endpoint = block.mlp._materialize_charted_cproj_weight(cproj_base)
        cfc_delta = cfc_endpoint.float() - cfc_base.float()
        cproj_delta = cproj_endpoint.float() - cproj_base.float()
        actions.append(
            {
                "c_fc": cfc_delta.cpu(),
                "c_proj": cproj_delta.cpu(),
            }
        )
        metrics.append(
            {
                "layer": float(layer),
                "cfc_delta_frobenius": float(torch.linalg.vector_norm(cfc_delta)),
                "cfc_delta_relative": float(
                    torch.linalg.vector_norm(cfc_delta)
                    / torch.linalg.vector_norm(cfc_base.float())
                ),
                "cproj_delta_frobenius": float(
                    torch.linalg.vector_norm(cproj_delta)
                ),
                "cproj_delta_relative": float(
                    torch.linalg.vector_norm(cproj_delta)
                    / torch.linalg.vector_norm(cproj_base.float())
                ),
            }
        )
    return actions, metrics


@torch.no_grad()
def delayed_bases(model: GPT) -> list[dict[str, torch.Tensor]]:
    output: list[dict[str, torch.Tensor]] = []
    for block in model.transformer.h:
        output.append(
            {
                "c_fc": block.mlp._cfc_base_weight().detach().clone(),
                "c_proj": block.mlp._cproj_base_weight().detach().clone(),
            }
        )
    return output


@torch.no_grad()
def install_materialized_override(
    model: GPT,
    bases: list[dict[str, torch.Tensor]],
    actions: list[dict[str, torch.Tensor]],
    *,
    cfc: bool,
    cproj: bool,
) -> None:
    if len(bases) != len(actions) or len(bases) != len(model.transformer.h):
        raise ValueError("materialized action layer count mismatch")
    for block, base, action in zip(
        model.transformer.h, bases, actions, strict=True
    ):
        mlp = block.mlp
        if (
            mlp._cached_charted_cfc_weight is not None
            or mlp._cached_charted_cproj_weight is not None
        ):
            raise RuntimeError("cannot install action over a live chart cache")
        cfc_weight = base["c_fc"]
        cproj_weight = base["c_proj"]
        if cfc:
            cfc_weight = cfc_weight + action["c_fc"].to(
                device=cfc_weight.device, dtype=cfc_weight.dtype
            )
        if cproj:
            cproj_weight = cproj_weight + action["c_proj"].to(
                device=cproj_weight.device, dtype=cproj_weight.dtype
            )
        mlp._cached_charted_cfc_weight = cfc_weight.detach()
        mlp._cached_charted_cproj_weight = cproj_weight.detach()


@torch.no_grad()
def discard_materialized_override(model: GPT) -> None:
    for block in model.transformer.h:
        block.mlp._cached_charted_cfc_weight = None
        block.mlp._cached_charted_cfc_graph_weight = None
        block.mlp._cached_charted_cproj_weight = None


@torch.no_grad()
def evaluate_override_ce(
    model: GPT,
    batches: list[torch.Tensor],
    device: str,
) -> float:
    losses: list[float] = []
    for tokens in batches:
        tokens = tokens.to(device)
        inputs = tokens[:, :-1].contiguous()
        targets = tokens[:, 1:].contiguous()
        with autocast_context(device):
            _, loss = model(inputs, targets)
        if loss is None:
            raise RuntimeError("model returned no CE loss")
        losses.append(float(loss))
    return float(sum(losses) / len(losses))


def select_decision(
    ce_by_variant: dict[str, dict[str, float]],
    *,
    minimum_gain: float,
    component_fraction: float,
) -> dict[str, Any]:
    splits = ("primary", "confirmation")
    native = ce_by_variant["native_delayed"]
    gains = {
        variant: {
            split: native[split] - ce_by_variant[variant][split]
            for split in splits
        }
        for variant in VARIANTS
        if variant != "native_delayed"
    }
    means = {
        variant: sum(values.values()) / len(values)
        for variant, values in gains.items()
    }
    raw_control_passed = all(
        gains["raw_endpoint_coordinates"][split] < 0.0 for split in splits
    )
    full = "additive_materialized_full"
    portable = (
        means[full] >= minimum_gain
        and all(gains[full][split] >= 0.0 for split in splits)
        and raw_control_passed
    )
    components: dict[str, bool] = {}
    for variant in (
        "additive_materialized_cfc_only",
        "additive_materialized_cproj_only",
    ):
        components[variant] = (
            portable
            and all(gains[variant][split] >= 0.0 for split in splits)
            and means[variant] >= component_fraction * means[full]
        )
    return {
        "decision": (
            "MATERIALIZED_ACTION_PORTABLE"
            if portable
            else "MATERIALIZED_ACTION_NONPORTABLE"
        ),
        "gain_by_window": gains,
        "mean_gain": means,
        "raw_coordinate_control_passed": raw_control_passed,
        "component_supported": components,
        "thresholds": {
            "minimum_mean_gain_over_native": minimum_gain,
            "both_windows_nonnegative_required": True,
            "component_support_fraction": component_fraction,
        },
        "automatic_training_run_authorized": False,
        "automatic_rerun_authorized": False,
        "larger_rung_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", required=True, type=Path)
    parser.add_argument("--parent-checkpoint-sha256", required=True)
    parser.add_argument("--delayed-checkpoint", required=True, type=Path)
    parser.add_argument("--delayed-checkpoint-sha256", required=True)
    parser.add_argument("--endpoint-state", required=True, type=Path)
    parser.add_argument("--endpoint-state-sha256", required=True)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--primary-seed", type=int, default=20260885)
    parser.add_argument("--confirmation-seed", type=int, default=20260886)
    parser.add_argument("--minimum-gain", type=float, default=0.005)
    parser.add_argument("--component-fraction", type=float, default=0.5)
    args = parser.parse_args()
    if not args.device.startswith("cuda"):
        parser.error("the registered diagnostic requires CUDA")
    if args.batch_size != 16 or args.block_size != 1024 or args.eval_batches != 32:
        parser.error("evaluation shape differs from the registered protocol")
    if args.primary_seed != 20260885 or args.confirmation_seed != 20260886:
        parser.error("evaluation seeds differ from the registered protocol")
    if args.output.exists():
        parser.error("output already exists")

    root = Path(__file__).resolve().parents[2]
    plan = root / "examples/nanogpt/configs/selection_artifacts" / PLAN_NAME
    input_hashes = {
        "plan": sha256(plan),
        "parent_checkpoint": validate_sha(
            args.parent_checkpoint, args.parent_checkpoint_sha256
        ),
        "delayed_checkpoint": validate_sha(
            args.delayed_checkpoint, args.delayed_checkpoint_sha256
        ),
        "endpoint_state": validate_sha(
            args.endpoint_state, args.endpoint_state_sha256
        ),
        "dataset_manifest": validate_sha(
            args.data_dir / "manifest.json", args.manifest_sha256
        ),
    }
    parent = load_frame_model(args.parent_checkpoint, args.device)
    delayed_checkpoint = torch.load(
        args.delayed_checkpoint, map_location="cpu", weights_only=False
    )
    if delayed_checkpoint.get("schema_version") != "nanogpt_exact_resume_v2":
        raise ValueError("delayed checkpoint is not exact-resume-v2")
    if int(delayed_checkpoint.get("next_iter", -1)) != 238:
        raise ValueError("delayed checkpoint is not terminal")
    delayed = load_delayed_model(delayed_checkpoint, args.device)
    parent_cached = prepare_frozen_base_cache(parent, torch.bfloat16)
    delayed_cached = prepare_frozen_base_cache(delayed, torch.bfloat16)
    native = capture_frame_state(delayed)
    identity = identity_state(native)
    endpoint_payload = torch.load(
        args.endpoint_state, map_location="cpu", weights_only=False
    )
    endpoint = endpoint_payload.get("state")
    if not isinstance(endpoint, dict) or set(endpoint) != set(native):
        raise ValueError("endpoint frame state identity mismatch")
    actions, action_metrics = materialized_parent_actions(parent, endpoint)
    restore_frame_state(delayed, identity)
    bases = delayed_bases(delayed)
    restore_frame_state(delayed, native)
    delayed_before = nonframe_state_digest(delayed)
    parent_before = nonframe_state_digest(parent)

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
    coordinate_states = {
        "native_delayed": native,
        "identity": identity,
        "raw_endpoint_coordinates": endpoint,
    }
    override_flags = {
        "additive_materialized_full": (True, True),
        "additive_materialized_cfc_only": (True, False),
        "additive_materialized_cproj_only": (False, True),
    }
    ce_by_variant: dict[str, dict[str, float]] = {}
    performance: dict[str, dict[str, float]] = {}
    try:
        torch.cuda.reset_peak_memory_stats()
        for variant in VARIANTS:
            ce_by_variant[variant] = {}
            performance[variant] = {}
            if variant in coordinate_states:
                restore_frame_state(delayed, coordinate_states[variant])
            else:
                restore_frame_state(delayed, identity)
                cfc, cproj = override_flags[variant]
                install_materialized_override(
                    delayed, bases, actions, cfc=cfc, cproj=cproj
                )
            try:
                for split, batches in windows.items():
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    ce = (
                        evaluate_ce(delayed, batches, args.device)
                        if variant in coordinate_states
                        else evaluate_override_ce(delayed, batches, args.device)
                    )
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
                if variant in override_flags:
                    discard_materialized_override(delayed)
    finally:
        restore_frame_state(delayed, native)
        discard_materialized_override(delayed)
        clear_frozen_base_cache(delayed)
        restore_frame_state(parent, identity_state(capture_frame_state(parent)))
        clear_frozen_base_cache(parent)
    delayed_after = nonframe_state_digest(delayed)
    parent_after = nonframe_state_digest(parent)
    if delayed_before != delayed_after or parent_before != parent_after:
        raise RuntimeError("non-frame state changed during transport diagnostic")

    selection = select_decision(
        ce_by_variant,
        minimum_gain=args.minimum_gain,
        component_fraction=args.component_fraction,
    )
    args.output.mkdir(parents=True)
    summary = {
        "schema_version": SCHEMA,
        "created_at": "2026-08-05",
        "repository_commit": git_head(root),
        "entrypoint": "examples.nanogpt.analyze_mlp_cproj_task_frame_materialized_action_transport",
        "command": list(sys.argv),
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
            "parent_cached_block_fht_modules": parent_cached,
            "delayed_cached_block_fht_modules": delayed_cached,
        },
        "action_metrics_by_layer": action_metrics,
        "ce_by_variant": ce_by_variant,
        "performance": performance,
        "peak_mib": torch.cuda.max_memory_allocated() / (1024 * 1024),
        "delayed_nonframe_state_sha256_before": delayed_before,
        "delayed_nonframe_state_sha256_after": delayed_after,
        "parent_nonframe_state_sha256_before": parent_before,
        "parent_nonframe_state_sha256_after": parent_after,
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
