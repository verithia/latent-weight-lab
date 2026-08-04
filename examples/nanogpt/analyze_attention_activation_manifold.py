#!/usr/bin/env python3
"""Measure dense attention trajectories in the data-dependent softmax metric.

The exact Q/K and V/O quotient can remain broad even when only a small part of
that motion matters on the activation distribution.  This script freezes the
layer-normalized inputs produced by the terminal dense checkpoint, replays the
registered attention-weight snapshots, and measures three per-head paths:

* row-centered causal score logits;
* causal softmax probabilities;
* projected head contribution to the residual stream.

Kernel paths are measured on the same cells as controls.  The analysis is
diagnostic only and never authorizes training by itself.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from examples.nanogpt.analyze_attention_fht_block_skew_tangent import (
    file_sha256,
    write_csv,
)
from examples.nanogpt.analyze_attention_functional_manifold import (
    product_kernel,
    trajectory_metrics,
)
from examples.nanogpt.analyze_parameter_trajectory import load_snapshots
from examples.nanogpt.analyze_residual_compatibility import (
    fixed_validation_batches,
    load_model,
)


def git_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


class LayerInputCollector:
    def __init__(self, model: torch.nn.Module, layers: list[int]) -> None:
        self.layers = set(layers)
        self.inputs: dict[int, list[torch.Tensor]] = {
            layer: [] for layer in layers
        }
        self.handles = []
        for index, block in enumerate(model.transformer.h):
            if index in self.layers:
                self.handles.append(
                    block.ln_1.register_forward_hook(self._hook(index))
                )

    def _hook(self, layer: int):
        def hook(_module, _inputs, output):
            self.inputs[layer].append(output.detach().float().cpu())

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def collect_terminal_inputs(
    checkpoint: Path,
    batches: list[torch.Tensor],
    layers: list[int],
    device: str,
) -> dict[int, torch.Tensor]:
    model = load_model(checkpoint, device)
    collector = LayerInputCollector(model, layers)
    try:
        with torch.no_grad():
            for batch in batches:
                model(batch.to(device), None)
        return {
            layer: torch.cat(collector.inputs[layer], dim=0)
            for layer in layers
        }
    finally:
        collector.close()
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


def causal_head_function(
    inputs: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    output_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return centered valid logits, valid probabilities, and head output."""
    if inputs.ndim != 3:
        raise ValueError("inputs must be [batch, time, channels]")
    if not (q_weight.shape == k_weight.shape == v_weight.shape):
        raise ValueError("Q/K/V head weights must share one shape")
    head_dim, channels = q_weight.shape
    if inputs.shape[-1] != channels:
        raise ValueError("input channels do not match Q/K/V weights")
    if output_weight.shape != (channels, head_dim):
        raise ValueError("output head slice must be [channels, head_dim]")
    q = F.linear(inputs, q_weight)
    k = F.linear(inputs, k_weight)
    v = F.linear(inputs, v_weight)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
    seq_len = inputs.shape[1]
    mask = torch.ones(
        (seq_len, seq_len), dtype=torch.bool, device=inputs.device
    ).tril()
    counts = torch.arange(
        1, seq_len + 1, device=inputs.device, dtype=scores.dtype
    )
    valid_sum = scores.masked_fill(~mask, 0.0).sum(dim=-1)
    centered = scores - (valid_sum / counts).unsqueeze(-1)
    probabilities = torch.softmax(scores.masked_fill(~mask, -torch.inf), dim=-1)
    head_state = probabilities @ v
    contribution = F.linear(head_state, output_weight)
    return centered[:, mask], probabilities[:, mask], contribution


def weighted_summary(
    rows: list[dict[str, Any]], prefix: str
) -> dict[str, float | int]:
    if not rows:
        raise ValueError("cannot summarize zero rows")
    weights = torch.tensor(
        [float(row[f"{prefix}_terminal_displacement_fro"]) ** 2 for row in rows],
        dtype=torch.float64,
    )
    keys = (
        "pc1_energy",
        "pc1_pc2_energy",
        "dimension_90pct",
        "dimension_95pct",
        "dimension_99pct",
        "participation_dimension",
        "path_length_over_chord",
        "median_relative_terminal_ray_residual",
        "mean_terminal_ray_recovery",
        "minimum_terminal_ray_recovery",
        "mean_consecutive_increment_cosine",
        "median_turn_degrees",
        "maximum_turn_degrees",
        "monotone_terminal_progress_fraction",
    )

    def weighted(key: str) -> float:
        values = torch.tensor(
            [float(row[f"{prefix}_{key}"]) for row in rows],
            dtype=torch.float64,
        )
        return float((weights * values).sum() / weights.sum().clamp_min(1e-30))

    return {"cells": len(rows), **{key: weighted(key) for key in keys}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--terminal-checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    plan = json.loads(args.plan.read_text())
    if plan.get("schema_version") != "mai_124m_attention_activation_manifold_plan_v1":
        raise ValueError("unexpected plan schema")
    protocol = plan["protocol"]
    layers = [int(value) for value in protocol["layers"]]
    steps = [int(value) for value in protocol["steps"]]
    snapshot_paths = [args.snapshot_dir / f"step_{step:06d}.pt" for step in steps]
    missing = [str(path) for path in snapshot_paths if not path.is_file()]
    if missing:
        raise ValueError("missing snapshots: " + ", ".join(missing))
    loaded_steps, values, metadata = load_snapshots(
        snapshot_paths,
        layers=set(layers),
        targets={"attn.c_attn", "attn.c_proj"},
    )
    if loaded_steps != steps:
        raise ValueError("loaded snapshot steps do not match plan")
    if metadata["run_identity_sha256"] != protocol["snapshot_run_identity_sha256"]:
        raise ValueError("snapshot identity does not match plan")
    batches = fixed_validation_batches(
        args.data_dir,
        int(protocol["batch_size"]),
        int(protocol["block_size"]),
        int(protocol["batches"]),
        int(protocol["sample_seed"]),
    )
    print("collecting fixed terminal-dense layer inputs", flush=True)
    fixed_inputs = collect_terminal_inputs(
        args.terminal_checkpoint, batches, layers, args.device
    )
    config = metadata["model_config"]
    n_embd = int(config["n_embd"])
    n_head = int(config["n_head"])
    head_dim = n_embd // n_head
    rows: list[dict[str, Any]] = []
    for layer in layers:
        print(f"analyzing layer {layer}", flush=True)
        inputs = fixed_inputs[layer].to(args.device, dtype=torch.float32)
        c_attn_steps = [
            value.to(args.device, dtype=torch.float32)
            for value in values[f"transformer.h.{layer}.attn.c_attn.weight"]
        ]
        c_proj_steps = [
            value.to(args.device, dtype=torch.float32)
            for value in values[f"transformer.h.{layer}.attn.c_proj.weight"]
        ]
        for head in range(n_head):
            start = head * head_dim
            stop = start + head_dim
            q_steps = [value[start:stop] for value in c_attn_steps]
            k_steps = [value[n_embd + start : n_embd + stop] for value in c_attn_steps]
            v_steps = [value[2 * n_embd + start : 2 * n_embd + stop] for value in c_attn_steps]
            o_steps = [value[:, start:stop] for value in c_proj_steps]
            centered_logits: list[torch.Tensor] = []
            probabilities: list[torch.Tensor] = []
            contributions: list[torch.Tensor] = []
            for q, k, v, output in zip(
                q_steps, k_steps, v_steps, o_steps, strict=True
            ):
                logits, probs, contribution = causal_head_function(
                    inputs, q, k, v, output
                )
                centered_logits.append(logits.flatten())
                probabilities.append(probs.flatten())
                contributions.append(contribution.flatten())
            representations = {
                "score_kernel": torch.stack(
                    [product_kernel(q, k).flatten() for q, k in zip(q_steps, k_steps, strict=True)]
                ),
                "value_output_kernel": torch.stack(
                    [product_kernel(v, output.transpose(0, 1)).flatten() for v, output in zip(v_steps, o_steps, strict=True)]
                ),
                "centered_logits": torch.stack(centered_logits),
                "probabilities": torch.stack(probabilities),
                "head_contribution": torch.stack(contributions),
            }
            metrics = {
                name: trajectory_metrics(sequence)
                for name, sequence in representations.items()
            }
            row: dict[str, Any] = {
                "layer": layer,
                "head": head,
                "snapshots": len(steps),
                "input_sequences": int(inputs.shape[0]),
                "input_tokens": int(inputs.shape[0] * inputs.shape[1]),
            }
            for name, measured in metrics.items():
                row.update({f"{name}_{key}": value for key, value in measured.items()})
            rows.append(row)
            del representations, metrics, centered_logits, probabilities, contributions
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        del inputs, c_attn_steps, c_proj_steps

    prefixes = (
        "score_kernel",
        "value_output_kernel",
        "centered_logits",
        "probabilities",
        "head_contribution",
    )
    aggregate = {prefix: weighted_summary(rows, prefix) for prefix in prefixes}
    comparisons = {
        "centered_logits_over_score_kernel_path": float(
            aggregate["centered_logits"]["path_length_over_chord"]
            / aggregate["score_kernel"]["path_length_over_chord"]
        ),
        "centered_logits_minus_score_kernel_pc1": float(
            aggregate["centered_logits"]["pc1_energy"]
            - aggregate["score_kernel"]["pc1_energy"]
        ),
        "probabilities_over_score_kernel_path": float(
            aggregate["probabilities"]["path_length_over_chord"]
            / aggregate["score_kernel"]["path_length_over_chord"]
        ),
        "probabilities_minus_score_kernel_pc1": float(
            aggregate["probabilities"]["pc1_energy"]
            - aggregate["score_kernel"]["pc1_energy"]
        ),
        "head_contribution_over_value_output_kernel_path": float(
            aggregate["head_contribution"]["path_length_over_chord"]
            / aggregate["value_output_kernel"]["path_length_over_chord"]
        ),
        "head_contribution_minus_value_output_kernel_pc1": float(
            aggregate["head_contribution"]["pc1_energy"]
            - aggregate["value_output_kernel"]["pc1_energy"]
        ),
    }
    thresholds = plan["interpretation_rule"]
    score_simpler = (
        comparisons["probabilities_over_score_kernel_path"]
        <= float(thresholds["maximum_path_length_ratio"])
        and comparisons["probabilities_minus_score_kernel_pc1"]
        >= float(thresholds["minimum_pc1_energy_gain"])
    )
    output_simpler = (
        comparisons["head_contribution_over_value_output_kernel_path"]
        <= float(thresholds["maximum_path_length_ratio"])
        and comparisons["head_contribution_minus_value_output_kernel_pc1"]
        >= float(thresholds["minimum_pc1_energy_gain"])
    )
    classification = (
        "ACTIVATION_SOFTMAX_METRIC_MATERIALLY_SIMPLER"
        if score_simpler and output_simpler
        else "ACTIVATION_SOFTMAX_METRIC_NOT_UNIFORMLY_SIMPLER"
    )
    args.output.mkdir(parents=True, exist_ok=True)
    cells_path = args.output / "attention_activation_manifold_cells.csv"
    write_csv(cells_path, rows)
    repo_root = Path(__file__).resolve().parents[2]
    result = {
        "schema_version": "mai_124m_attention_activation_manifold_v1",
        "source_commit": git_commit(repo_root),
        "source_sha256": file_sha256(Path(__file__)),
        "plan": {"path": str(args.plan), "sha256": file_sha256(args.plan)},
        "snapshot_run_identity_sha256": metadata["run_identity_sha256"],
        "snapshot_paths": [
            {"path": str(path), "sha256": file_sha256(path)}
            for path in snapshot_paths
        ],
        "terminal_checkpoint": {
            "path": str(args.terminal_checkpoint),
            "sha256": file_sha256(args.terminal_checkpoint),
        },
        "data_dir": str(args.data_dir),
        "layers": layers,
        "steps": steps,
        "aggregate": aggregate,
        "comparisons": comparisons,
        "decision": {
            "classification": classification,
            "score_softmax_materially_simpler": score_simpler,
            "residual_head_contribution_materially_simpler": output_simpler,
            "thresholds": thresholds,
            "automatic_training_authorized": False,
        },
        "cells_csv": {"path": str(cells_path), "sha256": file_sha256(cells_path)},
        "limitations": [
            "Inputs are frozen terminal-dense activations, so the metric isolates parameter motion rather than the co-evolving residual stream.",
            "Temporal PCA rank is bounded by the 17 registered snapshots.",
            "A simpler functional path would motivate a separate causal structure gate, not a learned activation basis or automatic training run.",
        ],
        "elapsed_seconds": time.time() - started,
    }
    result_path = args.output / "attention_activation_manifold_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
