#!/usr/bin/env python3
"""Separate c_fc directed-product radius error from residual direction error."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_mlp_cfc_exact_current_matcher import (
    file_sha256,
    fixed_batches,
    git_commit,
    load_model_and_optimizer,
)
from examples.nanogpt.analyze_mlp_dense_oracle_gap import (
    ExactVariantApplier,
    aggregate_direction_metrics,
    evaluate_candidates,
    family_fro,
)
from examples.nanogpt.analyze_mlp_joint_step_response_surface import (
    paired_comparison,
)
from examples.nanogpt.model import MultiOptimizer
from examples.nanogpt.muon import zeropower_via_newtonschulz5
from examples.nanogpt.muon_matched_givens import (
    MuonDirectedProduct,
    MuonDirectedProductLinear,
    batched_multistage_directed_sparse_update,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "nanogpt_mlp_cfc_directed_product_terminal_v1"


def _autocast(device: str, dtype: torch.dtype):
    if not device.startswith("cuda"):
        return nullcontext()
    return torch.amp.autocast("cuda", dtype=dtype)


def directed_optimizer(optimizer: MultiOptimizer) -> MuonDirectedProduct:
    matches = [
        child
        for child in optimizer.optimizers
        if isinstance(child, MuonDirectedProduct)
    ]
    if len(matches) != 1:
        raise ValueError("expected exactly one MuonDirectedProduct child")
    return matches[0]


def cfc_modules(model) -> list[MuonDirectedProductLinear]:
    modules = [
        block.mlp.c_fc
        for block in model.transformer.h
        if isinstance(block.mlp.c_fc, MuonDirectedProductLinear)
    ]
    if len(modules) != len(model.transformer.h):
        raise ValueError("not every c_fc is a directed-product module")
    return modules


def collect_cfc_gradients(
    model,
    modules: list[MuonDirectedProductLinear],
    batches: list[torch.Tensor],
    *,
    device: str,
    dtype: torch.dtype,
) -> float:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in modules:
        module.weight.requires_grad_(True)
        module.weight.grad = None
    model.eval()
    losses: list[float] = []
    model.prepare_block_fht_cache(dtype=dtype)
    try:
        for tokens in batches:
            tokens = tokens.to(device)
            with _autocast(device, dtype):
                _logits, loss = model(
                    tokens[:, :-1].contiguous(),
                    tokens[:, 1:].contiguous(),
                )
            if loss is None:
                raise RuntimeError("model did not return CE")
            losses.append(float(loss.detach()))
            (loss / len(batches)).backward()
    finally:
        model.flush_block_fht_cache()
    if any(module.weight.grad is None for module in modules):
        raise RuntimeError("c_fc gradient collection is incomplete")
    return sum(losses) / len(losses)


@torch.no_grad()
def prospective_updates(
    optimizer: MuonDirectedProduct,
    modules: list[MuonDirectedProductLinear],
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[str, Any]]:
    if len(optimizer.param_groups) != 1:
        raise ValueError("directed-product optimizer must have one group")
    group = optimizer.param_groups[0]
    lr = float(group["lr"])
    momentum = float(group["momentum"])
    weight_decay = float(group["weight_decay"])
    ns_steps = int(group["ns_steps"])
    requested = []
    for module in modules:
        weight = module.weight
        gradient = weight.grad
        if gradient is None:
            raise RuntimeError("missing c_fc gradient")
        state = optimizer.state[weight]
        buffer = state.get("momentum_buffer")
        if buffer is None:
            buffer = torch.zeros_like(weight)
        next_buffer = momentum * buffer.float() + gradient.float()
        combined = gradient.float() + momentum * next_buffer
        polar = zeropower_via_newtonschulz5(combined, steps=ns_steps).float()
        scale = max(
            1.0,
            polar.shape[0] / max(1, polar.numel() / polar.shape[0]),
        ) ** 0.5
        requested.append(lr * (-scale * polar - weight_decay * weight.float()))
    source = torch.stack(
        [module.weight.float().T for module in modules], dim=0
    ).contiguous()
    target = torch.stack([update.T for update in requested], dim=0).contiguous()
    reference = modules[0]
    raw, stages = batched_multistage_directed_sparse_update(
        source,
        target,
        incoming_schedule=reference.incoming_schedule,
        ridge_ratio=reference.ridge_ratio,
        chunk_size=reference.chunk_size,
    )
    dense = {index: value.T.contiguous().cpu() for index, value in enumerate(target)}
    raw_product = {
        index: value.T.contiguous().cpu() for index, value in enumerate(raw)
    }
    return dense, raw_product, {
        "lr": lr,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "ns_steps": ns_steps,
        "incoming_schedule": list(reference.incoming_schedule),
        "registered_family_radius_ratio": reference.family_radius_ratio,
        "dense_family_fro": family_fro(dense),
        "raw_product_family_fro": family_fro(raw_product),
        "stage_rows": stages,
    }


def scaled_to_dense_ratio(
    raw: dict[int, torch.Tensor],
    dense: dict[int, torch.Tensor],
    ratio: float,
) -> dict[int, torch.Tensor]:
    scale = float(ratio) * family_fro(dense) / max(family_fro(raw), 1e-30)
    return {layer: update * scale for layer, update in raw.items()}


def interpolate(
    left: dict[int, torch.Tensor],
    right: dict[int, torch.Tensor],
    fraction: float,
) -> dict[int, torch.Tensor]:
    return {
        layer: (1.0 - float(fraction)) * left[layer] + float(fraction) * right[layer]
        for layer in left
    }


def build_candidates(
    dense: dict[int, torch.Tensor],
    raw_product: dict[int, torch.Tensor],
    *,
    radius_ratios: list[float],
    residual_fractions: list[float],
    registered_ratio: float,
) -> tuple[dict[str, dict[str, dict[int, torch.Tensor]]], dict[str, Any]]:
    product = {
        ratio: scaled_to_dense_ratio(raw_product, dense, ratio)
        for ratio in radius_ratios
    }
    current = scaled_to_dense_ratio(raw_product, dense, registered_ratio)
    dense_same_radius = {
        layer: update * float(registered_ratio)
        for layer, update in dense.items()
    }
    candidates: dict[str, dict[str, dict[int, torch.Tensor]]] = {
        "baseline": {},
        "dense_same_radius": {"c_fc": dense_same_radius},
        "dense_full_radius": {"c_fc": dense},
        "product_registered": {"c_fc": current},
    }
    for ratio in radius_ratios:
        candidates[f"product_radius_{ratio:.6f}"] = {"c_fc": product[ratio]}
    for fraction in residual_fractions:
        candidates[f"product_plus_residual_{fraction:.6f}"] = {
            "c_fc": interpolate(current, dense, fraction)
        }
    return candidates, {
        "registered_product_metrics": aggregate_direction_metrics(dense, current),
        "dense_same_radius_metrics": aggregate_direction_metrics(dense, dense_same_radius),
        "radius_ratios": radius_ratios,
        "residual_fractions": residual_fractions,
    }


def classify(
    rows: list[dict[str, Any]],
    *,
    radius_ratios: list[float],
    registered_ratio: float,
    confidence_z: float,
) -> dict[str, Any]:
    means = {
        point: sum(float(row["ce"]) for row in rows if row["point_id"] == point)
        / sum(1 for row in rows if row["point_id"] == point)
        for point in sorted({str(row["point_id"]) for row in rows})
    }
    current = "product_registered"
    dense_direction = paired_comparison(
        rows, "dense_same_radius", current, confidence_z
    )
    radius = {
        f"{ratio:.6f}": paired_comparison(
            rows, f"product_radius_{ratio:.6f}", current, confidence_z
        )
        for ratio in radius_ratios
        if not math.isclose(
            ratio, registered_ratio, rel_tol=0.0, abs_tol=1e-12
        )
    }
    best_ratio = min(radius_ratios, key=lambda value: means[f"product_radius_{value:.6f}"])
    radius_reliable = any(row["candidate_reliably_better"] for row in radius.values())
    direction_reliable = dense_direction["candidate_reliably_better"]
    if direction_reliable and not radius_reliable:
        classification = "RESIDUAL_DIRECTION_LIMITED"
    elif radius_reliable and not direction_reliable:
        classification = "TRUST_RADIUS_LIMITED"
    elif radius_reliable and direction_reliable:
        classification = "MIXED_RADIUS_AND_DIRECTION_LIMITED"
    else:
        classification = "TERMINAL_LOCAL_STEP_NOT_DISCRIMINATING"
    return {
        "classification": classification,
        "candidate_mean_ce": means,
        "dense_same_radius_vs_product": dense_direction,
        "radius_vs_registered": radius,
        "best_product_radius_ratio": best_ratio,
        "radius_reliably_improves": radius_reliable,
        "dense_direction_reliably_improves_at_same_radius": direction_reliable,
    }


def validate_plan(
    plan_path: Path, checkpoint: Path, config: Path, data_dir: Path
) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text())
    actual = {
        "checkpoint_sha256": file_sha256(checkpoint),
        "config_sha256": file_sha256(config),
        "dataset_manifest_sha256": file_sha256(data_dir / "manifest.json"),
        "entrypoint_sha256": file_sha256(Path(__file__).resolve()),
    }
    for key, value in actual.items():
        if plan["identity"][key] != value:
            raise ValueError(f"registered identity mismatch: {key}")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    plan = validate_plan(args.plan, args.checkpoint, args.config, args.data_dir)
    protocol = plan["protocol"]
    config = json.loads(args.config.read_text())
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[config["dtype"]]
    model, optimizer, checkpoint = load_model_and_optimizer(
        args.checkpoint, config, args.device
    )
    modules = cfc_modules(model)
    train_batches = fixed_batches(
        args.data_dir,
        "train",
        batch_size=int(protocol["gradient_batch_size"]),
        block_size=int(config["block_size"]) + 1,
        batches=int(protocol["gradient_accumulation_steps"]),
        seed=int(protocol["gradient_seed"]),
    )
    gradient_ce = collect_cfc_gradients(
        model, modules, train_batches, device=args.device, dtype=dtype
    )
    dense, raw_product, prospective = prospective_updates(
        directed_optimizer(optimizer), modules
    )
    ratios = [float(value) for value in protocol["radius_ratios"]]
    residuals = [float(value) for value in protocol["residual_fractions"]]
    candidates, geometry = build_candidates(
        dense,
        raw_product,
        radius_ratios=ratios,
        residual_fractions=residuals,
        registered_ratio=float(protocol["registered_radius_ratio"]),
    )
    windows = {
        f"window_{index + 1}": fixed_batches(
            args.data_dir,
            "val",
            batch_size=int(protocol["evaluation_batch_size"]),
            block_size=int(config["block_size"]) + 1,
            batches=int(protocol["validation_batches_per_window"]),
            seed=int(seed),
        )
        for index, seed in enumerate(protocol["validation_seeds"])
    }
    rows = evaluate_candidates(
        model,
        ExactVariantApplier(model),
        windows,
        candidates,
        device=args.device,
        dtype=dtype,
    )
    decision = classify(
        rows,
        radius_ratios=ratios,
        registered_ratio=float(protocol["registered_radius_ratio"]),
        confidence_z=float(plan["decision_rule"]["confidence_z"]),
    )
    args.output.mkdir(parents=True, exist_ok=False)
    rows_path = args.output / "heldout_ce.json"
    rows_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "parameter_updates_to_checkpoint": 0,
        "gradient_window_ce": gradient_ce,
        "checkpoint_next_iter": int(checkpoint["next_iter"]),
        "prospective_update": prospective,
        "geometry": geometry,
        "identity": {
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "config_sha256": file_sha256(args.config),
            "dataset_manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
            "plan_sha256": file_sha256(args.plan),
            "heldout_ce_sha256": file_sha256(rows_path),
        },
        "execution": {
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": str(Path(__file__).resolve()),
            "entrypoint_sha256": file_sha256(Path(__file__).resolve()),
            "command": sys.argv,
            "device": args.device,
            "direct_foreground_polling": True,
            "watchdog": False,
            "callback": False,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
        },
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
