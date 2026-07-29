#!/usr/bin/env python3
"""Gate an exact-current expansion-side matcher for dense ``mlp.c_fc``.

The diagnostic restores the terminal two-pass-hidden88 checkpoint and its
Muon state.  A fixed train window supplies a new task gradient.  The exact
Muon formula combines that gradient with the persisted momentum, but no
optimizer or model update is performed.

Sparse Givens candidates operate on ``c_fc.T`` so the selected pairs act on
the 3,072 expansion/output channels.  They are scored in weight space, after
GELU, after the fixed hidden88 ``c_proj``, by independent validation
gradients, and by finite fixed-window CE.  Static pre-GELU frames, gains,
learned bases, dense residual adapters, and LoRA are outside this gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.nanogpt.analyze_mlp_muon_matched_givens import (
    diagonal_metric_causal_givens_update,
)
from examples.nanogpt.fast_task_matching import (
    build_task_edge_coloring,
    fast_muon_matched_permutations,
)
from examples.nanogpt.model import GPT, GPTConfig, MultiOptimizer
from examples.nanogpt.muon import Muon, zeropower_via_newtonschulz5
from examples.nanogpt.muon_matched_givens import (
    MuonMatchedGivensLinear,
    random_unique_matchings,
)


CANDIDATES = (
    "dense_exact",
    "fresh_expansion64",
    "fresh_expansion88",
    "random_expansion88",
)
WINDOWS = ("fit", "validation_a", "validation_b")
SCHEMA_VERSION = "nanogpt_mlp_cfc_exact_current_matcher_v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fixed_batches(
    data_dir: Path,
    split: str,
    *,
    batch_size: int,
    block_size: int,
    batches: int,
    seed: int,
) -> list[torch.Tensor]:
    if split not in {"train", "val"}:
        raise ValueError("split must be train or val")
    values = np.memmap(
        data_dir / f"{split}.bin", dtype=np.uint16, mode="r"
    )
    if len(values) <= block_size:
        raise ValueError("dataset split is shorter than block_size")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    indices = torch.randint(
        len(values) - block_size,
        (int(batches), int(batch_size)),
        generator=generator,
    )
    return [
        torch.stack(
            [
                torch.from_numpy(
                    np.asarray(
                        values[
                            int(index) : int(index) + block_size
                        ],
                        dtype=np.int64,
                    )
                )
                for index in row
            ]
        )
        for row in indices
    ]


def exact_muon_update(
    weight: torch.Tensor,
    gradient: torch.Tensor,
    momentum_buffer: torch.Tensor,
    *,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    ns_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Return exact Muon descent per LR and the corresponding finite update."""
    if (
        weight.ndim != 2
        or gradient.shape != weight.shape
        or momentum_buffer.shape != weight.shape
    ):
        raise ValueError("Muon tensors must be same-shaped matrices")
    weight_f = weight.float()
    gradient_f = gradient.float()
    buffer_f = momentum_buffer.float()
    combined = gradient_f + float(momentum) * (
        float(momentum) * buffer_f + gradient_f
    )
    polar = zeropower_via_newtonschulz5(
        combined, steps=int(ns_steps)
    ).float()
    scale = max(
        1.0,
        polar.shape[0]
        / max(1, polar.numel() / polar.shape[0]),
    ) ** 0.5
    descent_per_lr = (
        -float(weight_decay) * weight_f - scale * polar
    )
    update = float(learning_rate) * descent_per_lr
    return update, descent_per_lr, {
        "learning_rate": float(learning_rate),
        "momentum": float(momentum),
        "weight_decay": float(weight_decay),
        "ns_steps": float(ns_steps),
        "polar_scale": float(scale),
        "gradient_fro": float(gradient_f.norm()),
        "momentum_buffer_fro": float(buffer_f.norm()),
        "combined_fro": float(combined.norm()),
        "update_fro": float(update.norm()),
    }


def _weight_decay_after_rotation(
    source: torch.Tensor,
    rotation_update: torch.Tensor,
    *,
    learning_rate: float,
    weight_decay: float,
) -> torch.Tensor:
    rotated = source.float() + rotation_update.float()
    decayed = rotated * (
        1.0 - float(learning_rate) * float(weight_decay)
    )
    return decayed - source.float()


def build_candidates(
    weight: torch.Tensor,
    dense_update: torch.Tensor,
    polar_descent_per_lr: torch.Tensor,
    *,
    parent_stages: int,
    residual_stages: int,
    neighbors: int,
    seed: int,
    learning_rate: float,
    weight_decay: float,
    native_cache: Path | None,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    """Fit task-selected and equal-coordinate random expansion-side charts."""
    if weight.shape != dense_update.shape:
        raise ValueError("weight and update shapes disagree")
    source_t = weight.float().T.contiguous()
    dense_t = dense_update.float().T.contiguous()
    polar_t = polar_descent_per_lr.float().T.contiguous()

    parent_permutations, parent_selection = (
        fast_muon_matched_permutations(
            source_t,
            polar_t,
            stages=parent_stages,
            neighbors=neighbors,
            seed=seed,
            cache_dir=native_cache,
        )
    )
    parent_rotation, parent_fit = diagonal_metric_causal_givens_update(
        source_t,
        dense_t,
        stages=parent_stages,
        seed=seed,
        permutations=parent_permutations,
    )
    after_parent = source_t + parent_rotation
    residual = dense_t - parent_rotation
    residual_permutations, residual_selection = (
        fast_muon_matched_permutations(
            after_parent,
            residual,
            stages=residual_stages,
            neighbors=neighbors,
            seed=seed + 1,
            cache_dir=native_cache,
        )
    )
    residual_rotation, residual_fit = (
        diagonal_metric_causal_givens_update(
            after_parent,
            residual,
            stages=residual_stages,
            seed=seed + 1,
            permutations=residual_permutations,
        )
    )

    random_stages = parent_stages + residual_stages
    random_permutations = random_unique_matchings(
        width=source_t.shape[1],
        stages=random_stages,
        seed=seed + 2,
    ).to(source_t.device)
    random_rotation, random_fit = diagonal_metric_causal_givens_update(
        source_t,
        dense_t,
        stages=random_stages,
        seed=seed + 2,
        permutations=random_permutations,
    )

    candidates_t = {
        "fresh_expansion64": _weight_decay_after_rotation(
            source_t,
            parent_rotation,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        ),
        "fresh_expansion88": _weight_decay_after_rotation(
            source_t,
            parent_rotation + residual_rotation,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        ),
        "random_expansion88": _weight_decay_after_rotation(
            source_t,
            random_rotation,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        ),
    }
    candidates = {
        "dense_exact": dense_update.float(),
        **{
            name: update.T.contiguous()
            for name, update in candidates_t.items()
        },
    }
    diagnostics = [
        {
            "selection": "fresh_expansion64",
            **parent_selection,
            "fit": parent_fit,
        },
        {
            "selection": "fresh_expansion_residual24",
            **residual_selection,
            "fit": residual_fit,
        },
        {
            "selection": "random_expansion88",
            "fit": random_fit,
        },
    ]
    return candidates, diagnostics


def direction_metrics(
    target: torch.Tensor, prediction: torch.Tensor
) -> dict[str, float]:
    target_f = target.double().reshape(-1)
    prediction_f = prediction.double().reshape(-1)
    target_energy = target_f.square().sum().clamp_min(1e-30)
    prediction_energy = prediction_f.square().sum().clamp_min(1e-30)
    dot = (target_f * prediction_f).sum()
    return {
        "target_energy": float(target_energy),
        "prediction_energy": float(prediction_energy),
        "cosine": float(dot / (target_energy * prediction_energy).sqrt()),
        "fixed_scale_recovery": float(
            1.0
            - (target_f - prediction_f).square().sum()
            / target_energy
        ),
        "positive_line_recovery": float(
            dot.clamp_min(0.0).square()
            / (target_energy * prediction_energy)
        ),
    }


def _finish_accumulator(values: dict[str, float]) -> dict[str, float]:
    target_energy = max(values["target_energy"], 1e-30)
    prediction_energy = max(values["prediction_energy"], 1e-30)
    dot = values["dot"]
    return {
        "target_energy": target_energy,
        "prediction_energy": prediction_energy,
        "cosine": dot / math.sqrt(target_energy * prediction_energy),
        "fixed_scale_recovery": 1.0
        - values["residual_energy"] / target_energy,
        "positive_line_recovery": max(dot, 0.0) ** 2
        / (target_energy * prediction_energy),
    }


def activation_effect_metrics(
    mlp_input: torch.Tensor,
    pre_gelu: torch.Tensor,
    cproj_weight: torch.Tensor,
    target_update: torch.Tensor,
    candidate_update: torch.Tensor,
    *,
    device: str,
    chunk_size: int = 256,
) -> dict[str, dict[str, float]]:
    """Compare candidate effects after GELU and the fixed c_proj."""
    accumulators = {
        point: defaultdict(float) for point in ("post_gelu", "mlp_output")
    }
    target = target_update.to(device=device, dtype=torch.float32)
    candidate = candidate_update.to(device=device, dtype=torch.float32)
    cproj = cproj_weight.to(device=device, dtype=torch.float32)
    for start in range(0, mlp_input.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), mlp_input.shape[0])
        inputs = mlp_input[start:stop].to(
            device=device, dtype=torch.float32
        )
        base_pre = pre_gelu[start:stop].to(
            device=device, dtype=torch.float32
        )
        target_post = F.gelu(
            base_pre + F.linear(inputs, target)
        ) - F.gelu(base_pre)
        candidate_post = F.gelu(
            base_pre + F.linear(inputs, candidate)
        ) - F.gelu(base_pre)
        pairs = {
            "post_gelu": (target_post, candidate_post),
            "mlp_output": (
                F.linear(target_post, cproj),
                F.linear(candidate_post, cproj),
            ),
        }
        for point, (expected, predicted) in pairs.items():
            expected_f = expected.float()
            predicted_f = predicted.float()
            accumulator = accumulators[point]
            accumulator["target_energy"] += float(
                expected_f.square().sum()
            )
            accumulator["prediction_energy"] += float(
                predicted_f.square().sum()
            )
            accumulator["dot"] += float(
                (expected_f * predicted_f).sum()
            )
            accumulator["residual_energy"] += float(
                (expected_f - predicted_f).square().sum()
            )
    return {
        point: _finish_accumulator(dict(accumulator))
        for point, accumulator in accumulators.items()
    }


class MLPActivationCollector:
    def __init__(self, model: GPT, layers: list[int]) -> None:
        self.layers = set(layers)
        self.inputs: dict[int, list[torch.Tensor]] = defaultdict(list)
        self.pre_gelu: dict[int, list[torch.Tensor]] = defaultdict(list)
        self.handles = []
        for layer, block in enumerate(model.transformer.h):
            if layer not in self.layers:
                continue
            self.handles.append(
                block.mlp.c_fc.register_forward_pre_hook(
                    self._input_hook(layer)
                )
            )
            self.handles.append(
                block.mlp.c_fc.register_forward_hook(
                    self._output_hook(layer)
                )
            )

    def _input_hook(self, layer: int):
        def hook(_module, inputs):
            values = inputs[0].detach().float()
            self.inputs[layer].append(
                values.reshape(-1, values.shape[-1]).cpu()
            )

        return hook

    def _output_hook(self, layer: int):
        def hook(_module, _inputs, output):
            values = output.detach().float()
            self.pre_gelu[layer].append(
                values.reshape(-1, values.shape[-1]).cpu()
            )

        return hook

    def tensors(
        self,
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        if set(self.inputs) != self.layers or set(self.pre_gelu) != self.layers:
            raise RuntimeError("activation collection is incomplete")
        return (
            {
                layer: torch.cat(self.inputs[layer], dim=0)
                for layer in sorted(self.layers)
            },
            {
                layer: torch.cat(self.pre_gelu[layer], dim=0)
                for layer in sorted(self.layers)
            },
        )

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _autocast(device: str, dtype: torch.dtype):
    if not device.startswith("cuda"):
        return nullcontext()
    return torch.amp.autocast("cuda", dtype=dtype)


def collect_window(
    model: GPT,
    batches: list[torch.Tensor],
    layers: list[int],
    *,
    device: str,
    dtype: torch.dtype,
) -> tuple[
    float,
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
]:
    """Collect averaged c_fc gradients and aligned nonlinear activations."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, MuonMatchedGivensLinear):
            module.weight.requires_grad_(False)
    selected = {
        layer: model.transformer.h[layer].mlp.c_fc.weight
        for layer in layers
    }
    for parameter in selected.values():
        parameter.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    collector = MLPActivationCollector(model, layers)
    losses: list[float] = []
    model.prepare_block_fht_cache(dtype=dtype)
    try:
        for tokens in batches:
            tokens = tokens.to(device)
            inputs = tokens[:, :-1].contiguous()
            targets = tokens[:, 1:].contiguous()
            with _autocast(device, dtype):
                _logits, loss = model(inputs, targets)
            if loss is None:
                raise RuntimeError("model did not return task loss")
            losses.append(float(loss.detach()))
            (loss / len(batches)).backward()
    finally:
        model.flush_block_fht_cache()
    gradients = {}
    for layer, parameter in selected.items():
        if parameter.grad is None:
            raise RuntimeError(f"missing c_fc gradient for layer {layer}")
        gradients[layer] = parameter.grad.detach().float().cpu()
    inputs, pre_gelu = collector.tensors()
    collector.close()
    return sum(losses) / len(losses), gradients, inputs, pre_gelu


@torch.no_grad()
def evaluate_loss(
    model: GPT,
    batches: list[torch.Tensor],
    *,
    device: str,
    dtype: torch.dtype,
) -> float:
    losses = []
    model.prepare_block_fht_cache(dtype=dtype)
    try:
        for tokens in batches:
            tokens = tokens.to(device)
            inputs = tokens[:, :-1].contiguous()
            targets = tokens[:, 1:].contiguous()
            with _autocast(device, dtype):
                _logits, loss = model(inputs, targets)
            if loss is None:
                raise RuntimeError("model did not return task loss")
            losses.append(float(loss))
    finally:
        model.flush_block_fht_cache()
    return sum(losses) / len(losses)


@torch.no_grad()
def evaluate_with_updates(
    model: GPT,
    batches: list[torch.Tensor],
    updates: dict[int, torch.Tensor],
    *,
    device: str,
    dtype: torch.dtype,
) -> float:
    originals = {
        layer: model.transformer.h[layer].mlp.c_fc.weight.detach().clone()
        for layer in updates
    }
    try:
        for layer, update in updates.items():
            parameter = model.transformer.h[layer].mlp.c_fc.weight
            parameter.add_(
                update.to(device=parameter.device, dtype=parameter.dtype)
            )
        return evaluate_loss(
            model, batches, device=device, dtype=dtype
        )
    finally:
        for layer, original in originals.items():
            model.transformer.h[layer].mlp.c_fc.weight.copy_(original)


def _optimizer_and_group_for_parameter(
    optimizer: MultiOptimizer,
    parameter: torch.Tensor,
) -> tuple[Muon, dict[str, Any]]:
    for child in optimizer.optimizers:
        for group in child.param_groups:
            if any(candidate is parameter for candidate in group["params"]):
                if not isinstance(child, Muon):
                    raise ValueError("c_fc is not owned by Muon")
                return child, group
    raise ValueError("c_fc optimizer owner is missing")


def load_model_and_optimizer(
    checkpoint_path: Path,
    config: dict[str, Any],
    device: str,
) -> tuple[GPT, MultiOptimizer, dict[str, Any]]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    with torch.device(device):
        model = GPT(GPTConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    optimizer = model.configure_optimizers(
        float(config["weight_decay"]),
        float(config["learning_rate"]),
        (float(config["beta1"]), float(config["beta2"])),
        "cuda" if device.startswith("cuda") else "cpu",
        optimizer=str(config["optimizer"]),
        muon_momentum=float(config["muon_momentum"]),
        muon_ns_steps=int(config["muon_ns_steps"]),
        muon_adamw_lr_scale=float(config["muon_adamw_lr_scale"]),
        block_fht_mlp_chart_lr_scale=float(
            config.get("block_fht_mlp_chart_lr_scale") or 1.0
        ),
        block_fht_mlp_pregelu_chart_lr_scale=float(
            config.get("block_fht_mlp_pregelu_chart_lr_scale")
            or 1.0
        ),
    )
    if not isinstance(optimizer, MultiOptimizer):
        raise ValueError("expected the registered multi-optimizer")
    optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer, checkpoint


def weighted(
    rows: list[dict[str, Any]], key: str, energy_key: str
) -> float:
    weights = torch.tensor(
        [float(row[energy_key]) for row in rows],
        dtype=torch.float64,
    )
    values = torch.tensor(
        [float(row[key]) for row in rows],
        dtype=torch.float64,
    )
    return float(
        (weights * values).sum() / weights.sum().clamp_min(1e-30)
    )


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / max(float(denominator), 1e-30)


def aggregate_results(
    rows: list[dict[str, Any]],
    finite_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for candidate in CANDIDATES:
        metrics[candidate] = {}
        for window in WINDOWS:
            selected = [
                row
                for row in rows
                if row["candidate"] == candidate
                and row["window"] == window
            ]
            metrics[candidate][window] = {
                "cells": len(selected),
                "weight_positive_line_recovery": weighted(
                    selected,
                    "weight_positive_line_recovery",
                    "weight_target_energy",
                ),
                "post_gelu_positive_line_recovery": weighted(
                    selected,
                    "post_gelu_positive_line_recovery",
                    "post_gelu_target_energy",
                ),
                "mlp_output_positive_line_recovery": weighted(
                    selected,
                    "mlp_output_positive_line_recovery",
                    "mlp_output_target_energy",
                ),
                "predicted_ce_decrease": sum(
                    float(row["predicted_ce_decrease"])
                    for row in selected
                ),
                "minimum_predicted_ce_decrease": min(
                    float(row["predicted_ce_decrease"])
                    for row in selected
                ),
            }

    def ratios(candidate: str, control: str) -> dict[str, Any]:
        return {
            window: {
                metric: safe_ratio(
                    metrics[candidate][window][metric],
                    metrics[control][window][metric],
                )
                for metric in (
                    "post_gelu_positive_line_recovery",
                    "mlp_output_positive_line_recovery",
                    "predicted_ce_decrease",
                )
            }
            for window in WINDOWS
        }

    finite = {
        candidate: {
            str(row["window"]): float(row["loss"])
            for row in finite_rows
            if row["candidate"] == candidate
        }
        for candidate in ("baseline", *CANDIDATES)
    }
    validation_windows = ("validation_a", "validation_b")
    dense_fraction = ratios("fresh_expansion88", "dense_exact")
    over_parent = ratios("fresh_expansion88", "fresh_expansion64")
    over_random = ratios("fresh_expansion88", "random_expansion88")
    all_numbers = [
        value
        for row in rows
        for value in row.values()
        if isinstance(value, (int, float))
    ]
    absolute_pass = all(
        dense_fraction[window][metric] >= 0.15
        for window in validation_windows
        for metric in (
            "post_gelu_positive_line_recovery",
            "mlp_output_positive_line_recovery",
            "predicted_ce_decrease",
        )
    )
    parent_pass = all(
        over_parent[window][metric] >= 1.10
        for window in validation_windows
        for metric in (
            "post_gelu_positive_line_recovery",
            "mlp_output_positive_line_recovery",
            "predicted_ce_decrease",
        )
    )
    random_pass = all(
        over_random[window][metric] >= (
            1.25 if metric == "predicted_ce_decrease" else 1.50
        )
        for window in validation_windows
        for metric in (
            "post_gelu_positive_line_recovery",
            "mlp_output_positive_line_recovery",
            "predicted_ce_decrease",
        )
    )
    finite_pass = all(
        finite["fresh_expansion88"][window]
        < finite["baseline"][window]
        and finite["fresh_expansion88"][window]
        < finite["random_expansion88"][window]
        for window in validation_windows
    )
    fit_layer_pass = (
        metrics["fresh_expansion88"]["fit"][
            "minimum_predicted_ce_decrease"
        ]
        > 0.0
    )
    all_finite = all(math.isfinite(float(value)) for value in all_numbers)
    passed = all(
        (
            absolute_pass,
            parent_pass,
            random_pass,
            finite_pass,
            fit_layer_pass,
            all_finite,
        )
    )
    return {
        "candidate_metrics": metrics,
        "comparisons": {
            "fresh88_dense_fraction": dense_fraction,
            "fresh88_over_fresh64": over_parent,
            "fresh88_over_random88": over_random,
        },
        "finite_losses": finite,
        "gates": {
            "absolute_dense_fraction": absolute_pass,
            "fresh88_over_fresh64": parent_pass,
            "fresh88_over_random88": random_pass,
            "finite_ce": finite_pass,
            "positive_fit_effect_every_layer": fit_layer_pass,
            "all_metrics_finite": all_finite,
        },
        "decision": (
            "SELECT_EXACT_CURRENT_CFC_MATCHER_FOR_PRODUCTION_IMPLEMENTATION_AND_SEPARATE_MFU_GATE"
            if passed
            else "REJECT_EXACT_CURRENT_SPARSE_CFC_MATCHER"
        ),
    }


def validate_identity(
    checkpoint: Path,
    config_path: Path,
    data_dir: Path,
    plan_path: Path,
) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    identity = plan["identity"]
    actual = {
        "checkpoint_sha256": file_sha256(checkpoint),
        "config_sha256": file_sha256(config_path),
        "dataset_manifest_sha256": file_sha256(
            data_dir / "manifest.json"
        ),
    }
    for key, value in actual.items():
        if value != identity[key]:
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
    parser.add_argument("--native-cache", type=Path)
    args = parser.parse_args()
    started = time.time()
    plan = validate_identity(
        args.checkpoint, args.config, args.data_dir, args.plan
    )
    protocol = plan["fixed_protocol"]
    layers = [int(layer) for layer in protocol["layers"]]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[str(config["dtype"])]
    batches = {
        "fit": fixed_batches(
            args.data_dir,
            "train",
            batch_size=int(protocol["batch_size"]),
            block_size=int(protocol["block_size"]),
            batches=int(protocol["batches_per_window"]),
            seed=int(protocol["fit_train_seed"]),
        ),
        "validation_a": fixed_batches(
            args.data_dir,
            "val",
            batch_size=int(protocol["batch_size"]),
            block_size=int(protocol["block_size"]),
            batches=int(protocol["batches_per_window"]),
            seed=int(protocol["validation_seeds"][0]),
        ),
        "validation_b": fixed_batches(
            args.data_dir,
            "val",
            batch_size=int(protocol["batch_size"]),
            block_size=int(protocol["block_size"]),
            batches=int(protocol["batches_per_window"]),
            seed=int(protocol["validation_seeds"][1]),
        ),
    }
    model, optimizer, checkpoint = load_model_and_optimizer(
        args.checkpoint, config, args.device
    )
    fit_loss, fit_gradients, fit_inputs, fit_pre = collect_window(
        model,
        batches["fit"],
        layers,
        device=args.device,
        dtype=dtype,
    )
    candidates_by_layer = {candidate: {} for candidate in CANDIDATES}
    optimizer_rows = []
    selection_rows = []
    cproj_weights = {}
    for layer in layers:
        weight = model.transformer.h[layer].mlp.c_fc.weight
        owner, group = _optimizer_and_group_for_parameter(
            optimizer, weight
        )
        state = owner.state[weight]
        buffer = state.get("momentum_buffer")
        if buffer is None:
            raise RuntimeError(f"missing c_fc momentum for layer {layer}")
        dense, descent, diagnostics = exact_muon_update(
            weight.detach(),
            fit_gradients[layer].to(weight.device),
            buffer,
            learning_rate=float(group["lr"]),
            momentum=float(group["momentum"]),
            weight_decay=float(group["weight_decay"]),
            ns_steps=int(group["ns_steps"]),
        )
        polar_descent = (
            descent + float(group["weight_decay"]) * weight.detach().float()
        )
        candidates, selections = build_candidates(
            weight.detach(),
            dense,
            polar_descent,
            parent_stages=64,
            residual_stages=24,
            neighbors=int(protocol["matching_neighbors"]),
            seed=int(protocol["matching_seed"]) + layer * 1009,
            learning_rate=float(group["lr"]),
            weight_decay=float(group["weight_decay"]),
            native_cache=args.native_cache,
        )
        for candidate, update in candidates.items():
            candidates_by_layer[candidate][layer] = update.detach().cpu()
        optimizer_rows.append({"layer": layer, **diagnostics})
        selection_rows.extend(
            {"layer": layer, **selection} for selection in selections
        )
        cproj = model.transformer.h[layer].mlp.c_proj
        if not isinstance(cproj, MuonMatchedGivensLinear):
            raise ValueError("hidden88 c_proj reference is not exact")
        cproj_weights[layer] = cproj.weight.detach().float().cpu()

    rows: list[dict[str, Any]] = []
    finite_rows: list[dict[str, Any]] = []
    windows = {
        "fit": (fit_loss, fit_gradients, fit_inputs, fit_pre)
    }
    for window in ("validation_a", "validation_b"):
        windows[window] = collect_window(
            model,
            batches[window],
            layers,
            device=args.device,
            dtype=dtype,
        )

    for window, (
        base_loss,
        gradients,
        mlp_inputs,
        pre_gelu,
    ) in windows.items():
        if window != "fit":
            finite_rows.append(
                {
                    "window": window,
                    "candidate": "baseline",
                    "loss": base_loss,
                    "loss_change_from_baseline": 0.0,
                }
            )
            for candidate in CANDIDATES:
                loss = evaluate_with_updates(
                    model,
                    batches[window],
                    candidates_by_layer[candidate],
                    device=args.device,
                    dtype=dtype,
                )
                finite_rows.append(
                    {
                        "window": window,
                        "candidate": candidate,
                        "loss": loss,
                        "loss_change_from_baseline": loss - base_loss,
                    }
                )
        for layer in layers:
            target = candidates_by_layer["dense_exact"][layer]
            for candidate in CANDIDATES:
                update = candidates_by_layer[candidate][layer]
                weight_metrics = direction_metrics(target, update)
                functional = activation_effect_metrics(
                    mlp_inputs[layer],
                    pre_gelu[layer],
                    cproj_weights[layer],
                    target,
                    update,
                    device=args.device,
                )
                predicted = -float(
                    (
                        gradients[layer].double()
                        * update.double()
                    ).sum()
                )
                row = {
                    "window": window,
                    "layer": layer,
                    "candidate": candidate,
                    "coordinates_per_layer": (
                        768 * 3072
                        if candidate == "dense_exact"
                        else (
                            (64 if candidate == "fresh_expansion64" else 88)
                            * (3072 // 2)
                        )
                    ),
                    "weight_target_energy": weight_metrics[
                        "target_energy"
                    ],
                    "weight_fixed_scale_recovery": weight_metrics[
                        "fixed_scale_recovery"
                    ],
                    "weight_positive_line_recovery": weight_metrics[
                        "positive_line_recovery"
                    ],
                    "weight_cosine": weight_metrics["cosine"],
                    "post_gelu_target_energy": functional["post_gelu"][
                        "target_energy"
                    ],
                    "post_gelu_fixed_scale_recovery": functional[
                        "post_gelu"
                    ]["fixed_scale_recovery"],
                    "post_gelu_positive_line_recovery": functional[
                        "post_gelu"
                    ]["positive_line_recovery"],
                    "post_gelu_cosine": functional["post_gelu"][
                        "cosine"
                    ],
                    "mlp_output_target_energy": functional["mlp_output"][
                        "target_energy"
                    ],
                    "mlp_output_fixed_scale_recovery": functional[
                        "mlp_output"
                    ]["fixed_scale_recovery"],
                    "mlp_output_positive_line_recovery": functional[
                        "mlp_output"
                    ]["positive_line_recovery"],
                    "mlp_output_cosine": functional["mlp_output"][
                        "cosine"
                    ],
                    "predicted_ce_decrease": predicted,
                }
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)

    aggregate = aggregate_results(rows, finite_rows)
    args.output.mkdir(parents=True, exist_ok=True)
    detail_path = args.output / "cfc_exact_current_matcher.csv"
    finite_path = args.output / "cfc_exact_current_matcher_finite_ce.csv"
    optimizer_path = args.output / "cfc_exact_current_matcher_optimizer.csv"
    selection_path = args.output / "cfc_exact_current_matcher_selections.json"
    aggregate_path = args.output / "cfc_exact_current_matcher_aggregate.json"
    write_csv(detail_path, rows)
    write_csv(finite_path, finite_rows)
    write_csv(optimizer_path, optimizer_rows)
    selection_path.write_text(
        json.dumps(selection_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    native_library, native_path = build_task_edge_coloring(
        args.native_cache
    )
    del native_library
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "decision": aggregate["decision"],
        "parameter_updates": 0,
        "checkpoint_next_iter": int(checkpoint["next_iter"]),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "config_sha256": file_sha256(args.config),
        "dataset_manifest_sha256": file_sha256(
            args.data_dir / "manifest.json"
        ),
        "plan_sha256": file_sha256(args.plan),
        "layers": layers,
        "windows": protocol,
        "learned_dense_basis": False,
        "dense_residual_adapter": False,
        "lora_adapter": False,
        "static_pregelu_frame": False,
        "analysis_execution": {
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": str(script),
            "entrypoint_sha256": file_sha256(script),
            "command": sys.argv,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "device": args.device,
        },
        "native_matcher": {
            "path": str(native_path),
            "sha256": file_sha256(native_path),
        },
        "outputs": {
            "detail_sha256": file_sha256(detail_path),
            "finite_ce_sha256": file_sha256(finite_path),
            "optimizer_sha256": file_sha256(optimizer_path),
            "selections_sha256": file_sha256(selection_path),
            "aggregate_sha256": file_sha256(aggregate_path),
        },
        "limitations": plan["limitations"],
        "gradient_protocol": (
            "Each fixed-window loss is averaged before backward. The exact "
            "Muon momentum/polar/scale/weight-decay formula is then applied "
            "to the resulting c_fc gradient and persisted momentum. This "
            "zero-update diagnostic does not claim to reconstruct the "
            "original training microbatch draw or its all-parameter global "
            "gradient-clipping factor."
        ),
    }
    metadata_path = args.output / "cfc_exact_current_matcher_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": aggregate["decision"],
                "aggregate": str(aggregate_path),
                "metadata": str(metadata_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
