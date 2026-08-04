#!/usr/bin/env python3
"""Compare AdamW and Muon geometry for attention Cayley factors.

The selected attention repair learns flattened Cayley factors with AdamW even
though the matched dense trajectory is produced by matrix-space Muon.  This
oracle keeps the exact fixed Cayley frames and computes their identity-chart
task gradients from the dense probes.  It then applies either the exact first
AdamW direction or a Muon polar direction to each thin factor and scores the
resulting effective weight direction against dense Muon's applied direction.

This is an optimizer-geometry gate, not a training result.  A training screen
is admitted only if Muon raises aggregate positive-line recovery by at least
1.25x and does not reduce recovery for any attention target family.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_parameter_trajectory import parse_int_list
from examples.nanogpt.model import LearnedLowRankCayleyMix
from examples.nanogpt.muon import zeropower_via_newtonschulz5
from examples.nanogpt.parameter_trajectory import (
    OPTIMIZER_PROBE_SCHEMA_VERSION,
)


TARGETS = {
    "qk_shared": {
        "rank_key": "attn.c_attn.qk_headwise",
        "input_seed_offset": 0,
        "output_seed_offset": 3,
    },
    "v": {
        "rank_key": "attn.c_attn.v",
        "input_seed_offset": 1,
        "output_seed_offset": 4,
    },
    "cproj": {
        "rank_key": "attn.c_proj",
        "input_seed_offset": None,
        "output_seed_offset": 5,
    },
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def direction_metrics(
    target: torch.Tensor,
    direction: torch.Tensor,
) -> dict[str, float]:
    target = target.double()
    direction = direction.double()
    target_norm = target.norm().clamp_min(1e-30)
    direction_norm = direction.norm().clamp_min(1e-30)
    dot = (target * direction).sum()
    cosine = dot / (target_norm * direction_norm)
    positive = dot.clamp_min(0.0)
    return {
        "cosine": float(cosine),
        "positive_line_recovery": float(
            positive.square()
            / (target_norm.square() * direction_norm.square())
        ),
        "target_fro": float(target_norm),
        "direction_fro": float(direction_norm),
    }


def make_charts(
    *,
    weight: torch.Tensor,
    rank: int,
    base_seed: int,
    layer: int,
    target: str,
) -> tuple[LearnedLowRankCayleyMix | None, LearnedLowRankCayleyMix | None]:
    metadata = TARGETS[target]
    input_offset = metadata["input_seed_offset"]
    output_offset = metadata["output_seed_offset"]
    device = weight.device
    dtype = weight.dtype
    input_chart = (
        LearnedLowRankCayleyMix(
            int(weight.shape[1]),
            rank,
            base_seed + layer * 64 + int(input_offset),
        ).to(device=device, dtype=dtype)
        if input_offset is not None
        else None
    )
    output_chart = (
        LearnedLowRankCayleyMix(
            int(weight.shape[0]),
            rank,
            base_seed + layer * 64 + int(output_offset),
        ).to(device=device, dtype=dtype)
        if output_offset is not None
        else None
    )
    return input_chart, output_chart


def effective_weight(
    weight: torch.Tensor,
    input_chart: LearnedLowRankCayleyMix | None,
    output_chart: LearnedLowRankCayleyMix | None,
) -> torch.Tensor:
    result = weight
    if input_chart is not None:
        identity = torch.eye(
            weight.shape[1], device=weight.device, dtype=weight.dtype
        )
        input_rotation = input_chart(identity)
        result = result @ input_rotation.transpose(0, 1)
    if output_chart is not None:
        identity = torch.eye(
            weight.shape[0], device=weight.device, dtype=weight.dtype
        )
        output_rotation = output_chart(identity)
        result = output_rotation.transpose(0, 1) @ result
    return result


def chart_gradients(
    *,
    weight: torch.Tensor,
    task_gradient: torch.Tensor,
    rank: int,
    base_seed: int,
    layer: int,
    target: str,
) -> list[tuple[str, torch.Tensor]]:
    input_chart, output_chart = make_charts(
        weight=weight,
        rank=rank,
        base_seed=base_seed,
        layer=layer,
        target=target,
    )
    loss = (
        effective_weight(weight, input_chart, output_chart)
        * task_gradient
    ).sum()
    loss.backward()
    gradients: list[tuple[str, torch.Tensor]] = []
    for side, chart in (("input", input_chart), ("output", output_chart)):
        if chart is None:
            continue
        for factor_name in ("left", "right"):
            parameter = getattr(chart, factor_name)
            gradient = parameter.grad
            if gradient is None:
                gradient = torch.zeros_like(parameter)
            gradients.append((f"{side}.{factor_name}", gradient.detach().clone()))
    return gradients


@torch.no_grad()
def apply_coordinate_direction(
    *,
    weight: torch.Tensor,
    rank: int,
    base_seed: int,
    layer: int,
    target: str,
    gradients: list[tuple[str, torch.Tensor]],
    optimizer_kind: str,
    epsilon: float,
    ns_steps: int,
) -> torch.Tensor:
    input_chart, output_chart = make_charts(
        weight=weight,
        rank=rank,
        base_seed=base_seed,
        layer=layer,
        target=target,
    )
    chart_by_side = {"input": input_chart, "output": output_chart}
    for name, gradient in gradients:
        side, factor_name = name.split(".")
        chart = chart_by_side[side]
        assert chart is not None
        parameter = getattr(chart, factor_name)
        if optimizer_kind == "adamw":
            update = gradient / (gradient.abs() + 1e-8)
        elif optimizer_kind == "muon":
            matrix_gradient = gradient.view(chart.features, chart.rank)
            update = zeropower_via_newtonschulz5(
                matrix_gradient,
                steps=ns_steps,
            )
            aspect_scale = math.sqrt(
                max(1.0, chart.features / max(1, chart.rank))
            )
            # Match the first AdamW factor-update Frobenius norm.  This
            # isolates direction geometry from a trivial step-length change.
            update = update * aspect_scale * math.sqrt(chart.rank)
            update = update.reshape_as(parameter)
        else:
            raise ValueError(f"unknown optimizer kind {optimizer_kind}")
        parameter.add_(update, alpha=-epsilon)
    moved = effective_weight(weight, input_chart, output_chart)
    return (moved - weight) / float(epsilon)


def weighted_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    weights = torch.tensor(
        [float(row["target_fro"]) ** 2 for row in rows],
        dtype=torch.float64,
    )

    def weighted(key: str) -> float:
        values = torch.tensor(
            [float(row[key]) for row in rows], dtype=torch.float64
        )
        return float((weights * values).sum() / weights.sum())

    return {
        "cells": len(rows),
        "adamw_cosine": weighted("adamw_cosine"),
        "adamw_positive_line_recovery": weighted(
            "adamw_positive_line_recovery"
        ),
        "muon_cosine": weighted("muon_cosine"),
        "muon_positive_line_recovery": weighted(
            "muon_positive_line_recovery"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--production-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", default="0,3,6,9,11")
    parser.add_argument("--steps", default="0,594,1188,1782,2372")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    args = parser.parse_args()
    started = time.time()
    layers = parse_int_list(args.layers)
    steps = parse_int_list(args.steps)
    config = json.loads(args.production_config.read_text())
    ranks = {
        str(key): int(value)
        for key, value in config["block_fht_attn_cayley_ranks"].items()
    }
    base_seed = int(config["block_fht_attn_cayley_seed"])
    probe_paths = [
        args.probe_dir / f"step_{step:06d}.pt" for step in steps
    ]
    if any(not path.is_file() for path in probe_paths):
        raise ValueError("required optimizer probe is absent")

    rows: list[dict[str, Any]] = []
    run_identity_sha256: str | None = None
    for path in probe_paths:
        probe = torch.load(path, map_location="cpu", weights_only=False)
        if probe.get("schema_version") != OPTIMIZER_PROBE_SCHEMA_VERSION:
            raise ValueError("unexpected optimizer probe schema")
        if run_identity_sha256 is None:
            run_identity_sha256 = probe["run_identity_sha256"]
        elif probe["run_identity_sha256"] != run_identity_sha256:
            raise ValueError("optimizer probes do not share one identity")
        n_embd = int(probe["model_config"]["n_embd"])
        step = int(probe["step"])
        for layer in layers:
            entries = (
                (
                    "qk_shared",
                    f"transformer.h.{layer}.attn.c_attn.weight",
                    slice(0, 2 * n_embd),
                ),
                (
                    "v",
                    f"transformer.h.{layer}.attn.c_attn.weight",
                    slice(2 * n_embd, None),
                ),
                (
                    "cproj",
                    f"transformer.h.{layer}.attn.c_proj.weight",
                    slice(None),
                ),
            )
            for target, name, selection in entries:
                record = probe["parameters"][name]
                weight = record["weight_before_step"][selection].to(
                    args.device, dtype=torch.float64
                )
                task_gradient = record["gradient_after_clip"][selection].to(
                    args.device, dtype=torch.float64
                )
                dense_muon = record["applied_direction_per_lr"][selection].to(
                    args.device, dtype=torch.float64
                )
                rank = ranks[str(TARGETS[target]["rank_key"])]
                gradients = chart_gradients(
                    weight=weight,
                    task_gradient=task_gradient,
                    rank=rank,
                    base_seed=base_seed,
                    layer=layer,
                    target=target,
                )
                candidate_directions = {
                    kind: apply_coordinate_direction(
                        weight=weight,
                        rank=rank,
                        base_seed=base_seed,
                        layer=layer,
                        target=target,
                        gradients=gradients,
                        optimizer_kind=kind,
                        epsilon=float(args.epsilon),
                        ns_steps=int(args.muon_ns_steps),
                    )
                    for kind in ("adamw", "muon")
                }
                row: dict[str, Any] = {
                    "step": step,
                    "layer": layer,
                    "target": target,
                    "rank": rank,
                }
                for kind, direction in candidate_directions.items():
                    metrics = direction_metrics(dense_muon, direction)
                    row.update(
                        {f"{kind}_{key}": value for key, value in metrics.items()}
                    )
                row["target_fro"] = float(dense_muon.norm())
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
                del weight, task_gradient, dense_muon, gradients
        del probe
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    targets = tuple(TARGETS)
    aggregate = weighted_summary(rows)
    by_target = {
        target: weighted_summary(
            [row for row in rows if row["target"] == target]
        )
        for target in targets
    }
    by_step = {
        str(step): weighted_summary(
            [row for row in rows if int(row["step"]) == step]
        )
        for step in steps
    }
    baseline = float(aggregate["adamw_positive_line_recovery"])
    candidate = float(aggregate["muon_positive_line_recovery"])
    ratio = candidate / max(baseline, 1e-30)
    no_target_regression = all(
        float(by_target[target]["muon_positive_line_recovery"])
        >= float(by_target[target]["adamw_positive_line_recovery"])
        for target in targets
    )
    admitted = ratio >= 1.25 and no_target_regression

    args.output.mkdir(parents=True, exist_ok=True)
    cells_path = args.output / "attention_cayley_factor_optimizer_cells.csv"
    write_csv(cells_path, rows)
    repo_root = Path(__file__).resolve().parents[2]
    result = {
        "schema_version": "mai_124m_attention_cayley_factor_optimizer_v1",
        "scientific_question": (
            "Does Muon geometry on thin Cayley factors better recover the "
            "dense Muon-applied attention direction than current AdamW?"
        ),
        "source_commit": git_commit(repo_root),
        "source_sha256": file_sha256(Path(__file__)),
        "production_config": {
            "path": str(args.production_config),
            "sha256": file_sha256(args.production_config),
        },
        "optimizer_probe_run_identity_sha256": run_identity_sha256,
        "optimizer_probe_paths": [
            {"path": str(path), "sha256": file_sha256(path)}
            for path in probe_paths
        ],
        "layers": layers,
        "steps": steps,
        "ranks": ranks,
        "epsilon": float(args.epsilon),
        "muon_ns_steps": int(args.muon_ns_steps),
        "aggregate": aggregate,
        "by_target": by_target,
        "by_step": by_step,
        "decision": {
            "minimum_relative_recovery": 1.25,
            "relative_recovery": ratio,
            "no_target_regression": no_target_regression,
            "admitted": admitted,
            "classification": (
                "ADMIT_CAYLEY_FACTOR_MUON_124M_SCREEN"
                if admitted
                else "REJECT_CAYLEY_FACTOR_MUON_SCREEN"
            ),
        },
        "interpretation": {
            "adamw": "exact bias-corrected first-step AdamW coordinate direction at full Cayley LR",
            "muon": "five-step Newton-Schulz polar factor direction, Frobenius norm matched to AdamW per thin factor",
            "score": "positive line recovery against the exact dense Muon applied_direction_per_lr",
        },
        "limitations": [
            "This identity-chart oracle uses dense weights and gradients; a trained generated model may present different local gradients.",
            "The comparison isolates first-step optimizer geometry and omits accumulated AdamW/Muon momentum.",
            "A positive oracle still requires implementation tests, an exact-config >=20% MFU gate, and a directly polled 124M screen."
        ],
        "outputs": {"cells_sha256": file_sha256(cells_path)},
        "elapsed_seconds": time.time() - started,
    }
    result_path = args.output / "attention_cayley_factor_optimizer_result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["decision"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
