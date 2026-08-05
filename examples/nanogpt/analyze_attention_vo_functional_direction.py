#!/usr/bin/env python3
"""Gate a joint attention value/output direction on held-out task gradients.

At a generated-QK/dense-V/dense-O endpoint, each head contributes through the
gauge-invariant operator ``K_h = O_h @ V_h``.  This zero-update diagnostic
compares a direct Muon direction for K_h with the K_h differential induced by
the actual separate full-matrix Muon transforms for dense V and O.  Directions
are fitted on one deterministic validation window and scored against a
disjoint window.  No model parameter is updated or optimizer state retained.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from examples.nanogpt.model import GPT, GPTConfig
from examples.nanogpt.muon import muon_update
from latent_weight_lab.block_fht import BlockFHTLinear


SCHEMA_VERSION = "mai_124m_attention_vo_functional_direction_v1"
PLAN_SCHEMA = "mai_124m_attention_vo_functional_direction_plan_v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.contiguous().cpu().numpy().tobytes()).hexdigest()


def git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite(item) for item in value)
    return True


def deterministic_window(
    data_dir: Path,
    *,
    batch_size: int,
    block_size: int,
    batches: int,
    seed: int,
) -> tuple[list[torch.Tensor], set[int]]:
    values = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")
    if len(values) <= block_size:
        raise ValueError("validation data is shorter than the requested block")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    indices = torch.randint(
        len(values) - block_size,
        (int(batches), int(batch_size)),
        generator=generator,
    )
    batches_out = [
        torch.stack(
            [
                torch.from_numpy(
                    np.asarray(
                        values[int(index) : int(index) + block_size],
                        dtype=np.int64,
                    )
                )
                for index in row
            ]
        )
        for row in indices
    ]
    return batches_out, {int(value) for value in indices.flatten()}


def load_endpoint_model(checkpoint_path: Path, device: str) -> GPT:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model = GPT(GPTConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()
    return model


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    numerator = (left.double() * right.double()).sum()
    denominator = (
        left.double().square().sum().sqrt()
        * right.double().square().sum().sqrt()
    ).clamp_min(1e-30)
    return float(numerator / denominator)


def factor_induced_direction(
    value_weight: torch.Tensor,
    output_weight: torch.Tensor,
    value_direction: torch.Tensor,
    output_direction: torch.Tensor,
) -> torch.Tensor:
    """Return the first-order positive-descent direction of ``O @ V``."""
    if value_weight.ndim != 2 or output_weight.ndim != 2:
        raise ValueError("V and O must be matrices")
    if value_direction.shape != value_weight.shape:
        raise ValueError("V direction shape mismatch")
    if output_direction.shape != output_weight.shape:
        raise ValueError("O direction shape mismatch")
    if output_weight.shape[1] != value_weight.shape[0]:
        raise ValueError("O and V inner dimensions do not match")
    return output_direction @ value_weight + output_weight @ value_direction


def attention_sources(attn: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return ``A_h @ X_value`` as ``[batch, head, time, channels]``."""
    if not attn.qk_headwise_c_attn:
        raise ValueError("diagnostic requires the registered QK-headwise endpoint")
    qk_input = attn._apply_cayley_atlas(
        x,
        attn.qk_input_cayley,
        attn.qk_input_cayley_atlas,
        attn.active_cayley_atlas_stage,
    )
    value_input = attn._apply_cayley_atlas(
        x,
        attn.v_input_cayley,
        attn.v_input_cayley_atlas,
        attn.active_cayley_atlas_stage,
    )
    qk = attn.c_attn_qk_headwise(qk_input)
    qk = attn._apply_cayley_atlas(
        qk,
        attn.qk_output_cayley,
        attn.qk_output_cayley_atlas,
        attn.active_cayley_atlas_stage,
    )
    q, key = qk.split(attn.n_embd, dim=2)
    batch, sequence, channels = q.shape
    head_dim = channels // attn.n_head

    def heads(value: torch.Tensor) -> torch.Tensor:
        return value.view(batch, sequence, attn.n_head, head_dim).transpose(1, 2)

    q_heads, k_heads = heads(q), heads(key)
    scores = q_heads @ k_heads.transpose(-2, -1) / math.sqrt(head_dim)
    mask = torch.ones(
        (sequence, sequence), dtype=torch.bool, device=scores.device
    ).tril()
    probabilities = F.softmax(
        scores.masked_fill(~mask, -torch.inf), dim=-1
    )
    return probabilities @ value_input.unsqueeze(1)


class FunctionalCollector:
    def __init__(self, model: GPT, layers: list[int]) -> None:
        self.layers = set(layers)
        self.inputs: dict[int, torch.Tensor] = {}
        self.output_gradients: dict[int, torch.Tensor] = {}
        self.handles: list[Any] = []
        for layer, block in enumerate(model.transformer.h):
            if layer not in self.layers:
                continue
            self.handles.append(
                block.ln_1.register_forward_hook(self._input_hook(layer))
            )
            self.handles.append(
                block.attn.c_proj.register_forward_hook(
                    self._output_hook(layer)
                )
            )

    def _input_hook(self, layer: int):
        def hook(_module, _inputs, output):
            self.inputs[layer] = output.detach().float()

        return hook

    def _output_hook(self, layer: int):
        def hook(_module, _inputs, output):
            def capture(gradient: torch.Tensor) -> None:
                self.output_gradients[layer] = gradient.detach().float()

            output.register_hook(capture)

        return hook

    def clear(self) -> None:
        self.inputs.clear()
        self.output_gradients.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def validate_endpoint(
    plan: dict[str, Any],
    result: dict[str, Any],
    config: dict[str, Any],
    *,
    result_path: Path,
    config_path: Path,
    checkpoint_path: Path,
    data_dir: Path,
) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unexpected V/O functional-direction plan schema")
    identity = plan["identity"]
    actual = {
        "required_dataset_manifest_sha256": file_sha256(
            data_dir / "manifest.json"
        ),
        "required_qk_only_config_sha256": file_sha256(config_path),
        "required_qk_only_checkpoint_sha256": file_sha256(checkpoint_path),
        "required_qk_only_result_sha256": file_sha256(result_path),
    }
    for field, value in actual.items():
        if value != identity[field]:
            raise ValueError(f"endpoint identity mismatch: {field}")
    run = result.get("run", {})
    if run.get("classification") != "clean" or run.get("exit_code") != 0:
        raise ValueError("QK-only endpoint is not clean")
    if run.get("checkpoint_sha256") != identity[
        "required_qk_only_checkpoint_sha256"
    ]:
        raise ValueError("QK-only result checkpoint identity mismatch")
    if set(config.get("block_fht_targets", [])) != {
        "attn.c_attn.qk_headwise"
    }:
        raise ValueError("config is not the registered QK-only endpoint")


def ensure_unmixed_vo(attn: torch.nn.Module) -> None:
    if isinstance(attn.c_attn_v, BlockFHTLinear) or isinstance(
        attn.c_proj, BlockFHTLinear
    ):
        raise ValueError("V and O must be dense at the diagnostic endpoint")
    for name in (
        "v_input_cayley",
        "v_output_cayley",
        "cproj_input_cayley",
        "cproj_output_cayley",
    ):
        if getattr(attn, name, None) is not None:
            raise ValueError(f"unsupported V/O mixing transform: {name}")
    for name in (
        "v_input_cayley_atlas",
        "v_output_cayley_atlas",
        "cproj_input_cayley_atlas",
        "cproj_output_cayley_atlas",
    ):
        atlas = getattr(attn, name, None)
        if atlas is not None and len(atlas) != 0:
            raise ValueError(f"unsupported V/O atlas transform: {name}")


def collect_window(
    model: GPT,
    batches: list[torch.Tensor],
    layers: list[int],
    *,
    device: str,
    muon_steps: int,
) -> dict[str, Any]:
    collector = FunctionalCollector(model, layers)
    direct_gradients: dict[tuple[int, int], torch.Tensor] = {}
    losses: list[float] = []
    try:
        model.zero_grad(set_to_none=True)
        for batch_index, tokens in enumerate(batches):
            collector.clear()
            tokens = tokens.to(device)
            inputs = tokens[:, :-1].contiguous()
            targets = tokens[:, 1:].contiguous()
            _logits, loss = model(inputs, targets)
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("non-finite diagnostic loss")
            loss.backward()
            losses.append(float(loss))
            if set(collector.inputs) != set(layers) or set(
                collector.output_gradients
            ) != set(layers):
                raise RuntimeError("incomplete V/O functional capture")
            for layer in layers:
                attn = model.transformer.h[layer].attn
                sources = attention_sources(attn, collector.inputs[layer])
                residual_gradient = collector.output_gradients[layer].reshape(
                    -1, attn.n_embd
                )
                for head in range(attn.n_head):
                    source = sources[:, head].reshape(-1, attn.n_embd)
                    key = (layer, head)
                    contribution = residual_gradient.T @ source
                    if key not in direct_gradients:
                        direct_gradients[key] = contribution
                    else:
                        direct_gradients[key].add_(contribution)
            print(
                f"  batch {batch_index + 1}/{len(batches)} loss={float(loss):.6f}",
                flush=True,
            )

        direct_directions: dict[tuple[int, int], torch.Tensor] = {}
        factor_directions: dict[tuple[int, int], torch.Tensor] = {}
        max_v_discrepancy = 0.0
        max_o_discrepancy = 0.0
        for layer in layers:
            attn = model.transformer.h[layer].attn
            v_weight = attn.c_attn_v.weight.detach().float()
            o_weight = attn.c_proj.weight.detach().float()
            v_gradient = attn.c_attn_v.weight.grad.detach().float()
            o_gradient = attn.c_proj.weight.grad.detach().float()
            derived_v = torch.zeros_like(v_gradient)
            derived_o = torch.zeros_like(o_gradient)
            head_dim = attn.n_embd // attn.n_head
            for head in range(attn.n_head):
                start, stop = head * head_dim, (head + 1) * head_dim
                direct = direct_gradients[(layer, head)]
                derived_v[start:stop].copy_(o_weight[:, start:stop].T @ direct)
                derived_o[:, start:stop].copy_(
                    direct @ v_weight[start:stop].T
                )
            v_denominator = v_gradient.double().norm().clamp_min(1e-30)
            o_denominator = o_gradient.double().norm().clamp_min(1e-30)
            max_v_discrepancy = max(
                max_v_discrepancy,
                float((v_gradient - derived_v).double().norm() / v_denominator),
            )
            max_o_discrepancy = max(
                max_o_discrepancy,
                float((o_gradient - derived_o).double().norm() / o_denominator),
            )
            v_direction = muon_update(v_gradient, steps=muon_steps)
            o_direction = muon_update(o_gradient, steps=muon_steps)
            for head in range(attn.n_head):
                start, stop = head * head_dim, (head + 1) * head_dim
                key = (layer, head)
                direct_directions[key] = muon_update(
                    direct_gradients[key], steps=muon_steps
                ).detach().cpu()
                factor_directions[key] = factor_induced_direction(
                    v_weight[start:stop],
                    o_weight[:, start:stop],
                    v_direction[start:stop],
                    o_direction[:, start:stop],
                ).detach().cpu()
                direct_gradients[key] = direct_gradients[key].detach().cpu()
        return {
            "losses": losses,
            "direct_gradients": direct_gradients,
            "direct_directions": direct_directions,
            "factor_directions": factor_directions,
            "maximum_relative_autograd_v_gradient_discrepancy": max_v_discrepancy,
            "maximum_relative_autograd_o_gradient_discrepancy": max_o_discrepancy,
        }
    finally:
        collector.close()
        model.zero_grad(set_to_none=True)


def global_cosine(
    left: dict[tuple[int, int], torch.Tensor],
    right: dict[tuple[int, int], torch.Tensor],
    keys: list[tuple[int, int]],
) -> float:
    dot = sum(float((left[key].double() * right[key].double()).sum()) for key in keys)
    left_energy = sum(float(left[key].double().square().sum()) for key in keys)
    right_energy = sum(float(right[key].double().square().sum()) for key in keys)
    return dot / max(math.sqrt(left_energy * right_energy), 1e-30)


def summarize(
    fit: dict[str, Any],
    holdout: dict[str, Any],
    layers: list[int],
    plan: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    keys = sorted(fit["direct_gradients"])
    rows = []
    for layer, head in keys:
        key = (layer, head)
        direct_task = cosine(
            holdout["direct_gradients"][key], fit["direct_directions"][key]
        )
        factor_task = cosine(
            holdout["direct_gradients"][key], fit["factor_directions"][key]
        )
        factor_direct = cosine(
            fit["direct_directions"][key], fit["factor_directions"][key]
        )
        rows.append(
            {
                "layer": layer,
                "head": head,
                "fit_direct_heldout_task_cosine": direct_task,
                "fit_factor_heldout_task_cosine": factor_task,
                "direct_minus_factor_heldout_task_cosine": direct_task
                - factor_task,
                "direct_fit_holdout_direction_cosine": cosine(
                    fit["direct_directions"][key],
                    holdout["direct_directions"][key],
                ),
                "factor_fit_holdout_direction_cosine": cosine(
                    fit["factor_directions"][key],
                    holdout["factor_directions"][key],
                ),
                "factor_fit_direct_fit_cosine": factor_direct,
                "factor_line_recovery_of_direct": max(factor_direct, 0.0) ** 2,
                "holdout_gradient_energy": float(
                    holdout["direct_gradients"][key].double().square().sum()
                ),
            }
        )

    direct_task = global_cosine(
        holdout["direct_gradients"], fit["direct_directions"], keys
    )
    factor_task = global_cosine(
        holdout["direct_gradients"], fit["factor_directions"], keys
    )
    direct_stability = global_cosine(
        fit["direct_directions"], holdout["direct_directions"], keys
    )
    factor_stability = global_cosine(
        fit["factor_directions"], holdout["factor_directions"], keys
    )
    factor_direct = global_cosine(
        fit["direct_directions"], fit["factor_directions"], keys
    )
    layer_summary: dict[str, Any] = {}
    for layer in layers:
        layer_keys = [key for key in keys if key[0] == layer]
        layer_direct = global_cosine(
            holdout["direct_gradients"], fit["direct_directions"], layer_keys
        )
        layer_factor = global_cosine(
            holdout["direct_gradients"], fit["factor_directions"], layer_keys
        )
        layer_summary[str(layer)] = {
            "direct_heldout_task_cosine": layer_direct,
            "factor_heldout_task_cosine": layer_factor,
            "direct_advantage": layer_direct - layer_factor,
        }
    threshold = plan["preregistered_gate"]
    positive_factor_multiplier = direct_task / max(factor_task, 1e-12)
    gate = {
        "all_metrics_finite": all_finite(rows),
        "minimum_valid_cells": len(rows)
        >= int(plan["protocol"]["guards"]["minimum_valid_cells"]),
        "aggregate_direct_heldout_task_cosine": direct_task
        >= float(threshold["minimum_aggregate_direct_heldout_task_cosine"]),
        "aggregate_direct_minus_factor_heldout_task_cosine": (
            direct_task - factor_task
        )
        >= float(
            threshold[
                "minimum_aggregate_direct_minus_factor_heldout_task_cosine"
            ]
        ),
        "direct_over_positive_factor_heldout_alignment_multiplier": (
            factor_task <= 0.0
            or positive_factor_multiplier
            >= float(
                threshold[
                    "minimum_direct_over_positive_factor_heldout_alignment_multiplier"
                ]
            )
        ),
        "aggregate_direct_fit_holdout_direction_cosine": direct_stability
        >= float(
            threshold[
                "minimum_aggregate_direct_fit_holdout_direction_cosine"
            ]
        ),
        "aggregate_factor_line_recovery_of_direct": max(factor_direct, 0.0)
        ** 2
        <= float(threshold["maximum_aggregate_factor_line_recovery_of_direct"]),
        "layers_with_positive_direct_advantage": sum(
            value["direct_advantage"] > 0.0 for value in layer_summary.values()
        )
        >= int(threshold["minimum_layers_with_positive_direct_advantage"]),
        "cells_with_positive_direct_heldout_alignment": sum(
            row["fit_direct_heldout_task_cosine"] > 0.0 for row in rows
        )
        >= int(threshold["minimum_cells_with_positive_direct_heldout_alignment"]),
    }
    passed = all(gate.values())
    summary = {
        "valid_cells": len(rows),
        "aggregate": {
            "direct_heldout_task_cosine": direct_task,
            "factor_heldout_task_cosine": factor_task,
            "direct_minus_factor_heldout_task_cosine": direct_task - factor_task,
            "direct_over_positive_factor_heldout_alignment_multiplier": positive_factor_multiplier,
            "direct_fit_holdout_direction_cosine": direct_stability,
            "factor_fit_holdout_direction_cosine": factor_stability,
            "factor_fit_direct_fit_cosine": factor_direct,
            "factor_line_recovery_of_direct": max(factor_direct, 0.0) ** 2,
        },
        "by_layer": layer_summary,
        "positive_direct_heldout_cells": sum(
            row["fit_direct_heldout_task_cosine"] > 0.0 for row in rows
        ),
        "gate": gate,
        "passed": passed,
        "decision": (
            threshold["pass_classification"]
            if passed
            else threshold["fail_classification"]
        ),
        "next_action": (
            threshold["pass_action"] if passed else threshold["fail_action"]
        ),
        "language_model_training_authorized": False,
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--run-result", required=True, type=Path)
    parser.add_argument("--production-config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    started = time.time()
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    plan = json.loads(args.plan.read_text())
    result = json.loads(args.run_result.read_text())
    config = json.loads(args.production_config.read_text())
    validate_endpoint(
        plan,
        result,
        config,
        result_path=args.run_result,
        config_path=args.production_config,
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
    )
    protocol = plan["protocol"]
    layers = [int(value) for value in protocol["layers"]]
    windows: dict[str, list[torch.Tensor]] = {}
    starts: dict[str, set[int]] = {}
    for name in ("fit_window", "holdout_window"):
        spec = protocol[name]
        windows[name], starts[name] = deterministic_window(
            args.data_dir,
            batch_size=int(spec["batch_size"]),
            block_size=int(protocol["block_size"]) + 1,
            batches=int(spec["batches"]),
            seed=int(spec["seed"]),
        )
    overlap = starts["fit_window"] & starts["holdout_window"]
    if overlap:
        raise ValueError("fit and holdout windows share sequence starts")

    model = load_endpoint_model(args.checkpoint, args.device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    # Q/K is a frozen feature map for this endpoint diagnostic.  Decode its
    # cache only after freezing so the cached tensors retain no latent graph.
    model.prepare_block_fht_cache(dtype=torch.float32)
    for layer in layers:
        attn = model.transformer.h[layer].attn
        ensure_unmixed_vo(attn)
        attn.c_attn_v.weight.requires_grad_(True)
        attn.c_proj.weight.requires_grad_(True)

    measured: dict[str, Any] = {}
    for name in ("fit_window", "holdout_window"):
        print(f"collecting {name}", flush=True)
        measured[name] = collect_window(
            model,
            windows[name],
            layers,
            device=args.device,
            muon_steps=int(protocol["muon_newton_schulz_steps"]),
        )
    rows, summary = summarize(
        measured["fit_window"], measured["holdout_window"], layers, plan
    )
    summary["maximum_relative_autograd_v_gradient_discrepancy"] = max(
        float(measured[name]["maximum_relative_autograd_v_gradient_discrepancy"])
        for name in measured
    )
    summary["maximum_relative_autograd_o_gradient_discrepancy"] = max(
        float(measured[name]["maximum_relative_autograd_o_gradient_discrepancy"])
        for name in measured
    )

    args.output.mkdir(parents=True, exist_ok=True)
    cells_path = args.output / "attention_vo_functional_direction_cells.csv"
    write_csv(cells_path, rows)
    repo = Path(__file__).resolve().parents[2]
    sealed = {
        "schema_version": SCHEMA_VERSION,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "started_at": started_at,
        "elapsed_seconds": time.time() - started,
        "source_commit": git_commit(repo),
        "source_sha256": file_sha256(Path(__file__)),
        "plan": {"path": str(args.plan), "sha256": file_sha256(args.plan)},
        "run_result": {
            "path": str(args.run_result),
            "sha256": file_sha256(args.run_result),
        },
        "production_config": {
            "path": str(args.production_config),
            "sha256": file_sha256(args.production_config),
        },
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": file_sha256(args.checkpoint),
        },
        "dataset_manifest": {
            "path": str(args.data_dir / "manifest.json"),
            "sha256": file_sha256(args.data_dir / "manifest.json"),
        },
        "windows": {
            name: {
                "token_sha256": tensor_sha256(torch.cat(windows[name])),
                "sequence_start_count": len(starts[name]),
                "mean_loss": sum(measured[name]["losses"])
                / len(measured[name]["losses"]),
                "losses": measured[name]["losses"],
            }
            for name in windows
        },
        "sequence_start_overlap": len(overlap),
        "summary": summary,
        "cells_csv": {"path": str(cells_path), "sha256": file_sha256(cells_path)},
        "parameter_updates": 0,
        "dense_optimizer_state_retained": False,
    }
    result_path = args.output / "attention_vo_functional_direction_result.json"
    result_path.write_text(json.dumps(sealed, indent=2, sort_keys=True) + "\n")
    print(json.dumps(sealed, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
