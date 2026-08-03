#!/usr/bin/env python3
"""Measure interaction between prospective production c_fc/c_proj steps.

The accepted 124M full-attention MLP candidate is restored together with its
optimizer state.  One real logical training batch supplies the same averaged,
globally clipped gradients used by training.  The production custom optimizers
are then run on disposable model copies for c_fc only, c_proj only, and both.
No checkpoint is resumed or written.

The extracted materialized-weight deltas are scored in the original
checkpoint's residual frame on two fixed validation windows.  Finite CE and
paired MLP/block-output measurements distinguish destructive interaction from
individually harmful directions and from approximately additive updates.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator

import torch

from examples.nanogpt.analyze_mlp_cfc_exact_current_matcher import (
    file_sha256,
    fixed_batches,
    git_commit,
    load_model_and_optimizer,
)
from examples.nanogpt.model import GPT, MultiOptimizer
from examples.nanogpt.muon_matched_givens import (
    MuonFunctionalShear,
    MuonFunctionalShearLinear,
    MuonMatchedGivens,
    MuonMatchedGivensLinear,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "nanogpt_mlp_joint_prospective_step_v1"
VARIANTS = ("cfc_only", "cproj_only", "joint")


def _autocast(device: str, dtype: torch.dtype):
    if not device.startswith("cuda"):
        return nullcontext()
    return torch.amp.autocast("cuda", dtype=dtype)


def optimizer_parameters(optimizer: MultiOptimizer) -> list[torch.Tensor]:
    parameters: list[torch.Tensor] = []
    seen: set[int] = set()
    for child in optimizer.optimizers:
        for group in child.param_groups:
            for parameter in group["params"]:
                if id(parameter) not in seen:
                    parameters.append(parameter)
                    seen.add(id(parameter))
    return parameters


def family_weights(model: GPT) -> dict[str, dict[int, torch.Tensor]]:
    cfc: dict[int, torch.Tensor] = {}
    cproj: dict[int, torch.Tensor] = {}
    for layer, block in enumerate(model.transformer.h):
        if not isinstance(block.mlp.c_fc, MuonFunctionalShearLinear):
            raise ValueError(f"layer {layer} c_fc is not functional-shear")
        if not isinstance(block.mlp.c_proj, MuonMatchedGivensLinear):
            raise ValueError(f"layer {layer} c_proj is not matched-Givens")
        cfc[layer] = block.mlp.c_fc.weight
        cproj[layer] = block.mlp.c_proj.weight
    return {"c_fc": cfc, "c_proj": cproj}


def _gradient_norm(parameters: list[torch.Tensor]) -> float:
    squares = torch.zeros((), device=parameters[0].device)
    for parameter in parameters:
        if parameter.grad is not None:
            squares = squares + parameter.grad.detach().float().square().sum()
    return float(squares.sqrt())


def _diagnostics(optimizer: MultiOptimizer) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for child in optimizer.optimizers:
        if isinstance(child, MuonFunctionalShear):
            result["c_fc"] = list(child.last_step_diagnostics)
        elif isinstance(child, MuonMatchedGivens):
            result["c_proj"] = list(child.last_step_diagnostics)
    return result


def extract_production_updates(
    checkpoint: Path,
    config: dict[str, Any],
    train_batches: list[torch.Tensor],
    families: set[str],
    *,
    device: str,
    dtype: torch.dtype,
) -> tuple[dict[str, dict[int, torch.Tensor]], dict[str, Any]]:
    """Run one disposable production step and return materialized deltas."""
    model, optimizer, checkpoint_payload = load_model_and_optimizer(
        checkpoint, config, device
    )
    model.train()
    weights = family_weights(model)
    before = {
        family: {
            layer: weight.detach().float().cpu().clone()
            for layer, weight in by_layer.items()
        }
        for family, by_layer in weights.items()
    }
    optimizer.zero_grad(set_to_none=True)
    model.prepare_block_fht_cache(dtype=dtype)
    losses: list[float] = []
    try:
        for tokens in train_batches:
            tokens = tokens.to(device)
            inputs = tokens[:, :-1].contiguous()
            targets = tokens[:, 1:].contiguous()
            with _autocast(device, dtype):
                _logits, loss = model(inputs, targets)
            if loss is None:
                raise RuntimeError("model did not return a training loss")
            losses.append(float(loss.detach()))
            (loss / len(train_batches)).backward()
    finally:
        model.flush_block_fht_cache()

    all_parameters = optimizer_parameters(optimizer)
    gradient_norm_before_clip = _gradient_norm(all_parameters)
    clip_threshold = float(config["grad_clip"])
    reported_norm = torch.nn.utils.clip_grad_norm_(
        model.product_fht_clip_parameters(), clip_threshold
    )
    gradient_norm_after_clip = _gradient_norm(all_parameters)
    allowed = {
        id(weight)
        for family in families
        for weight in weights[family].values()
    }
    for parameter in all_parameters:
        if id(parameter) not in allowed:
            parameter.grad = None
    selected_gradient_norm = _gradient_norm(
        [parameter for parameter in all_parameters if id(parameter) in allowed]
    )
    optimizer.step()

    updates = {
        family: {
            layer: weight.detach().float().cpu() - before[family][layer]
            for layer, weight in weights[family].items()
        }
        for family in families
    }
    group_lrs = {
        type(child).__name__: [float(group["lr"]) for group in child.param_groups]
        for child in optimizer.optimizers
    }
    metadata = {
        "families": sorted(families),
        "next_iter": int(checkpoint_payload["next_iter"]),
        "mean_training_ce": sum(losses) / len(losses),
        "gradient_accumulation_steps": len(train_batches),
        "gradient_norm_before_clip": gradient_norm_before_clip,
        "clip_norm_reported": float(reported_norm),
        "gradient_norm_after_clip": gradient_norm_after_clip,
        "selected_gradient_norm_after_clip": selected_gradient_norm,
        "clip_threshold": clip_threshold,
        "optimizer_group_lrs": group_lrs,
        "update_fro": {
            family: math.sqrt(
                sum(float(update.double().square().sum()) for update in by_layer.values())
            )
            for family, by_layer in updates.items()
        },
        "optimizer_diagnostics": _diagnostics(optimizer),
    }
    del model, optimizer
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return updates, metadata


def assert_joint_matches_singletons(
    single_cfc: dict[int, torch.Tensor],
    single_cproj: dict[int, torch.Tensor],
    joint: dict[str, dict[int, torch.Tensor]],
) -> dict[str, float]:
    maxima = {"c_fc": 0.0, "c_proj": 0.0}
    for family, expected in (("c_fc", single_cfc), ("c_proj", single_cproj)):
        if set(expected) != set(joint[family]):
            raise ValueError(f"joint {family} layers do not match singleton")
        for layer, tensor in expected.items():
            maximum = float((tensor - joint[family][layer]).abs().max())
            maxima[family] = max(maxima[family], maximum)
            torch.testing.assert_close(
                tensor, joint[family][layer], rtol=0.0, atol=1e-7
            )
    return maxima


@contextmanager
def applied_updates(
    model: GPT,
    updates: dict[str, dict[int, torch.Tensor]],
) -> Iterator[None]:
    weights = family_weights(model)
    try:
        with torch.no_grad():
            for family, by_layer in updates.items():
                for layer, update in by_layer.items():
                    weight = weights[family][layer]
                    weight.add_(update.to(device=weight.device, dtype=weight.dtype))
        yield
    finally:
        with torch.no_grad():
            for family, by_layer in updates.items():
                for layer, update in by_layer.items():
                    weight = weights[family][layer]
                    weight.sub_(update.to(device=weight.device, dtype=weight.dtype))


class OutputCollector:
    def __init__(self, model: GPT, layers: list[int]) -> None:
        self.values: dict[tuple[int, str], torch.Tensor] = {}
        self.handles = []
        for layer in layers:
            block = model.transformer.h[layer]
            self.handles.append(
                block.mlp.register_forward_hook(self._hook(layer, "mlp"))
            )
            self.handles.append(
                block.register_forward_hook(self._hook(layer, "block"))
            )

    def _hook(self, layer: int, kind: str):
        def capture(_module, _inputs, output):
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"{kind} output is not a tensor")
            self.values[(layer, kind)] = output.detach().float()

        return capture

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


@torch.no_grad()
def forward_capture(
    model: GPT,
    tokens: torch.Tensor,
    layers: list[int],
    *,
    device: str,
    dtype: torch.dtype,
) -> tuple[float, dict[tuple[int, str], torch.Tensor]]:
    collector = OutputCollector(model, layers)
    try:
        tokens = tokens.to(device)
        with _autocast(device, dtype):
            _logits, loss = model(
                tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
            )
        if loss is None:
            raise RuntimeError("model did not return an evaluation loss")
        return float(loss), dict(collector.values)
    finally:
        collector.close()


def update_variants(
    cfc: dict[int, torch.Tensor], cproj: dict[int, torch.Tensor]
) -> dict[str, dict[str, dict[int, torch.Tensor]]]:
    return {
        "cfc_only": {"c_fc": cfc},
        "cproj_only": {"c_proj": cproj},
        "joint": {"c_fc": cfc, "c_proj": cproj},
    }


def evaluate_windows(
    model: GPT,
    batches_by_window: dict[str, list[torch.Tensor]],
    variants: dict[str, dict[str, dict[int, torch.Tensor]]],
    probe_layers: list[int],
    metric_batches: int,
    *,
    device: str,
    dtype: torch.dtype,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ce_rows: list[dict[str, Any]] = []
    output_sums: dict[tuple[str, int, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    model.eval()
    model.prepare_block_fht_cache(dtype=dtype)
    try:
        for window, batches in batches_by_window.items():
            baseline_losses: list[float] = []
            variant_losses = {variant: [] for variant in VARIANTS}
            for batch_index, tokens in enumerate(batches):
                base_loss, base_values = forward_capture(
                    model,
                    tokens,
                    probe_layers,
                    device=device,
                    dtype=dtype,
                )
                baseline_losses.append(base_loss)
                captured: dict[str, dict[tuple[int, str], torch.Tensor]] = {}
                for variant in VARIANTS:
                    with applied_updates(model, variants[variant]):
                        loss, values = forward_capture(
                            model,
                            tokens,
                            probe_layers,
                            device=device,
                            dtype=dtype,
                        )
                    variant_losses[variant].append(loss)
                    if batch_index < metric_batches:
                        captured[variant] = values
                if batch_index >= metric_batches:
                    continue
                for layer in probe_layers:
                    for kind in ("mlp", "block"):
                        key = (layer, kind)
                        base = base_values[key]
                        delta_fc = captured["cfc_only"][key] - base
                        delta_proj = captured["cproj_only"][key] - base
                        delta_joint = captured["joint"][key] - base
                        additive = delta_fc + delta_proj
                        interaction = delta_joint - additive
                        sums = output_sums[(window, layer, kind)]
                        sums["base_energy"] += float(base.double().square().sum())
                        sums["cfc_energy"] += float(delta_fc.double().square().sum())
                        sums["cproj_energy"] += float(delta_proj.double().square().sum())
                        sums["joint_energy"] += float(delta_joint.double().square().sum())
                        sums["additive_energy"] += float(additive.double().square().sum())
                        sums["interaction_energy"] += float(
                            interaction.double().square().sum()
                        )
                        sums["cfc_cproj_dot"] += float(
                            (delta_fc.double() * delta_proj.double()).sum()
                        )
                        sums["joint_additive_dot"] += float(
                            (delta_joint.double() * additive.double()).sum()
                        )
            baseline = sum(baseline_losses) / len(baseline_losses)
            ce_rows.append(
                {
                    "window": window,
                    "variant": "baseline",
                    "ce": baseline,
                    "loss_change": 0.0,
                }
            )
            for variant in VARIANTS:
                value = sum(variant_losses[variant]) / len(variant_losses[variant])
                ce_rows.append(
                    {
                        "window": window,
                        "variant": variant,
                        "ce": value,
                        "loss_change": value - baseline,
                    }
                )
    finally:
        model.flush_block_fht_cache()

    output_rows: list[dict[str, Any]] = []
    for (window, layer, kind), sums in sorted(output_sums.items()):
        cfc_norm = math.sqrt(sums["cfc_energy"])
        cproj_norm = math.sqrt(sums["cproj_energy"])
        joint_norm = math.sqrt(sums["joint_energy"])
        additive_norm = math.sqrt(sums["additive_energy"])
        interaction_norm = math.sqrt(sums["interaction_energy"])
        output_rows.append(
            {
                "window": window,
                "layer": layer,
                "kind": kind,
                **dict(sums),
                "cfc_cproj_cosine": sums["cfc_cproj_dot"]
                / max(cfc_norm * cproj_norm, 1e-30),
                "joint_additive_cosine": sums["joint_additive_dot"]
                / max(joint_norm * additive_norm, 1e-30),
                "interaction_to_additive_norm": interaction_norm
                / max(additive_norm, 1e-30),
                "joint_to_base_norm": joint_norm
                / max(math.sqrt(sums["base_energy"]), 1e-30),
            }
        )
    return ce_rows, output_rows


def interaction_decision(
    ce_rows: list[dict[str, Any]],
    *,
    additive_tolerance: float,
    destructive_threshold: float,
    cooperative_threshold: float,
) -> dict[str, Any]:
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for row in ce_rows:
        grouped[str(row["window"])][str(row["variant"])] = float(row["ce"])
    metrics: dict[str, dict[str, float | bool]] = {}
    for window, values in grouped.items():
        required = {"baseline", *VARIANTS}
        if set(values) != required:
            raise ValueError(f"incomplete CE variants for {window}")
        base = values["baseline"]
        cfc_delta = values["cfc_only"] - base
        cproj_delta = values["cproj_only"] - base
        joint_delta = values["joint"] - base
        additive_delta = cfc_delta + cproj_delta
        interaction = joint_delta - additive_delta
        scale = max(abs(cfc_delta) + abs(cproj_delta), 1e-12)
        ratio = interaction / scale
        metrics[window] = {
            "cfc_loss_change": cfc_delta,
            "cproj_loss_change": cproj_delta,
            "joint_loss_change": joint_delta,
            "finite_additive_prediction": additive_delta,
            "finite_interaction": interaction,
            "interaction_over_singleton_absolute_change": ratio,
            "joint_is_task_helpful": joint_delta < 0.0,
        }
    ratios = [
        float(row["interaction_over_singleton_absolute_change"])
        for row in metrics.values()
    ]
    if all(ratio >= destructive_threshold for ratio in ratios):
        classification = "DESTRUCTIVE_CFC_CPROJ_UPDATE_INTERACTION"
    elif all(ratio <= -cooperative_threshold for ratio in ratios):
        classification = "COOPERATIVE_CFC_CPROJ_UPDATE_INTERACTION"
    elif all(abs(ratio) <= additive_tolerance for ratio in ratios):
        classification = "CFC_CPROJ_UPDATES_ARE_FINITE_CE_ADDITIVE"
    else:
        classification = "MIXED_CFC_CPROJ_UPDATE_INTERACTION"
    helpful = all(bool(row["joint_is_task_helpful"]) for row in metrics.values())
    if classification == "CFC_CPROJ_UPDATES_ARE_FINITE_CE_ADDITIVE" and not helpful:
        next_action = "FIX_INDIVIDUAL_PROSPECTIVE_DIRECTIONS_NOT_JOINT_MLP_CHART"
    elif classification == "DESTRUCTIVE_CFC_CPROJ_UPDATE_INTERACTION":
        next_action = "TEST_JOINT_BLOCK_OUTPUT_METRIC_CHART"
    elif classification == "COOPERATIVE_CFC_CPROJ_UPDATE_INTERACTION" and helpful:
        next_action = "PRESERVE_JOINT_UPDATE_COUPLING_IN_NEXT_MLP_STRUCTURE"
    else:
        next_action = "LOCALIZE_INTERACTION_BY_DEPTH_BEFORE_ARCHITECTURE_CHANGE"
    return {
        "classification": classification,
        "joint_helpful_on_all_windows": helpful,
        "metrics_by_window": metrics,
        "next_action": next_action,
    }


def validate_plan(
    plan_path: Path,
    checkpoint: Path,
    config_path: Path,
    data_dir: Path,
) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    identity = plan["identity"]
    actual = {
        "checkpoint_sha256": file_sha256(checkpoint),
        "config_sha256": file_sha256(config_path),
        "dataset_manifest_sha256": file_sha256(data_dir / "manifest.json"),
        "entrypoint_sha256": file_sha256(Path(__file__).resolve()),
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
    args = parser.parse_args()
    started = time.time()
    plan = validate_plan(
        args.plan, args.checkpoint, args.config, args.data_dir
    )
    protocol = plan["protocol"]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[str(config["dtype"])]
    if int(protocol["gradient_accumulation_steps"]) != int(
        config["gradient_accumulation_steps"]
    ):
        raise ValueError("plan does not match production accumulation")
    train_batches = fixed_batches(
        args.data_dir,
        "train",
        batch_size=int(config["batch_size"]),
        block_size=int(config["block_size"]) + 1,
        batches=int(protocol["gradient_accumulation_steps"]),
        seed=int(protocol["train_seed"]),
    )
    extracted: dict[str, Any] = {}
    cfc_payload, extracted["cfc_only"] = extract_production_updates(
        args.checkpoint,
        config,
        train_batches,
        {"c_fc"},
        device=args.device,
        dtype=dtype,
    )
    cproj_payload, extracted["cproj_only"] = extract_production_updates(
        args.checkpoint,
        config,
        train_batches,
        {"c_proj"},
        device=args.device,
        dtype=dtype,
    )
    joint_payload, extracted["joint"] = extract_production_updates(
        args.checkpoint,
        config,
        train_batches,
        {"c_fc", "c_proj"},
        device=args.device,
        dtype=dtype,
    )
    exactness = assert_joint_matches_singletons(
        cfc_payload["c_fc"], cproj_payload["c_proj"], joint_payload
    )
    variants = update_variants(
        cfc_payload["c_fc"], cproj_payload["c_proj"]
    )
    batches_by_window = {
        f"validation_{index + 1}": fixed_batches(
            args.data_dir,
            "val",
            batch_size=int(protocol["evaluation_batch_size"]),
            block_size=int(protocol["evaluation_block_size"]) + 1,
            batches=int(protocol["evaluation_batches_per_window"]),
            seed=int(seed),
        )
        for index, seed in enumerate(protocol["validation_seeds"])
    }
    model, _optimizer, checkpoint_payload = load_model_and_optimizer(
        args.checkpoint, config, args.device
    )
    ce_rows, output_rows = evaluate_windows(
        model,
        batches_by_window,
        variants,
        [int(layer) for layer in protocol["probe_layers"]],
        int(protocol["output_metric_batches_per_window"]),
        device=args.device,
        dtype=dtype,
    )
    rule = plan["decision_rule"]
    decision = interaction_decision(
        ce_rows,
        additive_tolerance=float(rule["additive_tolerance"]),
        destructive_threshold=float(rule["destructive_threshold"]),
        cooperative_threshold=float(rule["cooperative_threshold"]),
    )
    args.output.mkdir(parents=True, exist_ok=False)
    ce_path = args.output / "finite_ce.json"
    output_path = args.output / "output_interaction.json"
    extraction_path = args.output / "prospective_step_metadata.json"
    ce_path.write_text(json.dumps(ce_rows, indent=2, sort_keys=True) + "\n")
    output_path.write_text(
        json.dumps(output_rows, indent=2, sort_keys=True) + "\n"
    )
    extraction_path.write_text(
        json.dumps(extracted, indent=2, sort_keys=True) + "\n"
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "parameter_updates_to_checkpoint": 0,
        "disposable_optimizer_steps": 3,
        "checkpoint_next_iter": int(checkpoint_payload["next_iter"]),
        "identity": {
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "config_sha256": file_sha256(args.config),
            "dataset_manifest_sha256": file_sha256(
                args.data_dir / "manifest.json"
            ),
            "plan_sha256": file_sha256(args.plan),
        },
        "protocol": protocol,
        "joint_singleton_update_max_abs_error": exactness,
        "outputs": {
            "finite_ce_sha256": file_sha256(ce_path),
            "output_interaction_sha256": file_sha256(output_path),
            "prospective_step_metadata_sha256": file_sha256(extraction_path),
        },
        "execution": {
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": str(Path(__file__).resolve()),
            "entrypoint_sha256": file_sha256(Path(__file__).resolve()),
            "command": sys.argv,
            "device": args.device,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "direct_foreground_polling": True,
        },
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
