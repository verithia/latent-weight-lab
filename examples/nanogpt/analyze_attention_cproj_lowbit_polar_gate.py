#!/usr/bin/env python3
"""Gate full-support low-bit codecs for attention c_proj Muon requests."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_attention_cproj_fresh_residual_gate import (
    PROBE_SCHEMA,
    aggregate,
    file_sha256,
    git_commit,
    metrics,
    parameter_name,
)


PLAN_SCHEMA = "mai_124m_attention_cproj_lowbit_polar_gate_plan_v1"
RESULT_SCHEMA = "mai_124m_attention_cproj_lowbit_polar_gate_result_v1"


def theoretical_storage(
    *, elements: int, bits: int, block_size: int
) -> dict[str, int | float]:
    blocks = math.ceil(int(elements) / int(block_size))
    code_bytes = math.ceil(int(elements) * int(bits) / 8)
    scale_bytes = blocks * 2
    dense_fp32_bytes = int(elements) * 4
    compact_bytes = code_bytes + scale_bytes
    return {
        "elements": int(elements),
        "bits_per_code": int(bits),
        "blocks": blocks,
        "code_bytes": code_bytes,
        "fp16_scale_bytes": scale_bytes,
        "compact_bytes": compact_bytes,
        "dense_fp32_bytes": dense_fp32_bytes,
        "storage_ratio": compact_bytes / dense_fp32_bytes,
        "storage_reduction_factor": dense_fp32_bytes / compact_bytes,
    }


def quantize_blocks(
    values: torch.Tensor,
    *,
    codec: str,
    block_size: int,
    ternary_threshold_rms: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if values.ndim != 3:
        raise ValueError("values must be [members, rows, columns]")
    flattened = values.float().reshape(values.shape[0], -1)
    if flattened.shape[1] % int(block_size) != 0:
        raise ValueError("registered block size must divide each matrix")
    blocks = flattened.reshape(flattened.shape[0], -1, int(block_size))
    if codec == "binary":
        codes = torch.where(blocks >= 0, 1.0, -1.0)
        scales = blocks.abs().mean(dim=-1, keepdim=True)
    elif codec == "ternary":
        rms = blocks.square().mean(dim=-1, keepdim=True).sqrt()
        active = blocks.abs() >= (float(ternary_threshold_rms) * rms)
        codes = torch.sign(blocks) * active
        scales = (
            (blocks.abs() * active).sum(dim=-1, keepdim=True)
            / active.sum(dim=-1, keepdim=True).clamp_min(1)
        )
    elif codec == "int4":
        scales = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30) / 7.0
        codes = torch.round(blocks / scales).clamp(-7, 7)
    else:
        raise ValueError(f"unsupported codec: {codec}")
    reconstructed = (codes * scales).reshape_as(values)
    return reconstructed, {
        "zero_fraction": float((codes == 0).float().mean()),
        "mean_abs_code": float(codes.abs().mean()),
        "mean_scale": float(scales.mean()),
    }


def normalize_family(
    target: torch.Tensor, prediction: torch.Tensor, ratio: float
) -> tuple[torch.Tensor, float]:
    target_norm = target.double().square().sum().sqrt()
    prediction_norm = prediction.double().square().sum().sqrt()
    scale = float(ratio) * target_norm / prediction_norm.clamp_min(1e-30)
    return prediction * scale, float(scale)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--probe-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    plan = json.loads(args.plan.read_text())
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unexpected plan schema")
    protocol = plan["protocol"]
    steps = [int(value) for value in protocol["probe_steps"]]
    layers = [int(value) for value in protocol["layers"]]
    expected_hashes = plan["identity"]["optimizer_probe_sha256"]
    expected_identity = str(plan["identity"]["trajectory_run_identity_sha256"])
    input_hashes: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    codec_rows: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []

    for step in steps:
        path = args.probe_dir / f"step_{step:06d}.pt"
        digest = file_sha256(path)
        if digest != str(expected_hashes[path.name]):
            raise ValueError(f"probe hash mismatch: {path}")
        probe = torch.load(path, map_location="cpu", weights_only=False)
        if probe.get("schema_version") != PROBE_SCHEMA:
            raise ValueError(f"unexpected probe schema: {path}")
        if str(probe.get("run_identity_sha256")) != expected_identity:
            raise ValueError(f"probe identity mismatch: {path}")
        input_hashes[path.name] = digest
        targets: list[torch.Tensor] = []
        for layer in layers:
            name = parameter_name(layer)
            record = probe["parameters"][name]
            learning_rate = float(probe["hyperparameters"][name]["lr"])
            targets.append(learning_rate * record["applied_direction_per_lr"])
        target = torch.stack(targets).to(args.device, dtype=torch.float32)
        for candidate, spec in protocol["candidates"].items():
            if target.is_cuda:
                torch.cuda.synchronize(target.device)
            candidate_started = time.perf_counter()
            raw, codec_stats = quantize_blocks(
                target,
                codec=str(spec["codec"]),
                block_size=int(spec["block_size"]),
                ternary_threshold_rms=float(protocol["ternary_threshold_rms"]),
            )
            prediction, family_scale = normalize_family(
                target, raw, float(protocol["family_radius_ratio"])
            )
            if target.is_cuda:
                torch.cuda.synchronize(target.device)
            elapsed = time.perf_counter() - candidate_started
            timings.append(
                {"step": step, "candidate": candidate, "seconds": elapsed}
            )
            codec_rows.append(
                {
                    "step": step,
                    "candidate": candidate,
                    "family_scale": family_scale,
                    **codec_stats,
                }
            )
            for index, layer in enumerate(layers):
                row = {
                    "step": step,
                    "layer": layer,
                    "candidate": candidate,
                    **metrics(target[index], prediction[index]),
                }
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)

    elements = layers.__len__() * int(protocol["matrix_elements_per_layer"])
    summaries: dict[str, Any] = {}
    for candidate, spec in protocol["candidates"].items():
        selected = [row for row in rows if row["candidate"] == candidate]
        by_step = {
            str(step): aggregate([row for row in selected if row["step"] == step])
            for step in steps
        }
        by_layer = {
            str(layer): aggregate(
                [row for row in selected if row["layer"] == layer]
            )
            for layer in layers
        }
        timing = [
            float(row["seconds"])
            for row in timings
            if row["candidate"] == candidate
        ]
        summary = aggregate(selected)
        summary.update(
            {
                "codec": str(spec["codec"]),
                "block_size": int(spec["block_size"]),
                "minimum_phase_fixed_scale_recovery": min(
                    float(value["fixed_scale_recovery"]) for value in by_step.values()
                ),
                "minimum_layer_fixed_scale_recovery": min(
                    float(value["fixed_scale_recovery"]) for value in by_layer.values()
                ),
                "minimum_cell_fixed_scale_recovery": min(
                    float(row["fixed_scale_recovery"]) for row in selected
                ),
                "minimum_cell_descent_fraction": min(
                    float(row["descent_fraction"]) for row in selected
                ),
                "mean_codec_seconds_per_phase": sum(timing) / len(timing),
                "maximum_codec_seconds_per_phase": max(timing),
                "storage": theoretical_storage(
                    elements=elements,
                    bits=int(spec["bits"]),
                    block_size=int(spec["block_size"]),
                ),
                "by_step": by_step,
                "by_layer": by_layer,
            }
        )
        summaries[candidate] = summary

    thresholds = plan["decision_rule"]["thresholds"]
    decisions: dict[str, Any] = {}
    passing: list[str] = []
    for candidate, summary in summaries.items():
        checks = {
            "aggregate_recovery": float(summary["fixed_scale_recovery"])
            >= float(thresholds["aggregate_recovery_minimum"]),
            "minimum_phase_recovery": float(
                summary["minimum_phase_fixed_scale_recovery"]
            )
            >= float(thresholds["minimum_phase_recovery"]),
            "minimum_layer_recovery": float(
                summary["minimum_layer_fixed_scale_recovery"]
            )
            >= float(thresholds["minimum_layer_recovery"]),
            "minimum_cell_recovery": float(
                summary["minimum_cell_fixed_scale_recovery"]
            )
            >= float(thresholds["minimum_cell_recovery"]),
            "aggregate_cosine": float(summary["cosine"])
            >= float(thresholds["aggregate_cosine_minimum"]),
            "every_cell_positive_descent": float(
                summary["minimum_cell_descent_fraction"]
            )
            > 0.0,
            "storage_ratio": float(summary["storage"]["storage_ratio"])
            <= float(thresholds["maximum_storage_ratio"]),
        }
        passed = all(checks.values())
        decisions[candidate] = {"checks": checks, "passed": passed}
        if passed:
            passing.append(candidate)
    selected = (
        min(
            passing,
            key=lambda name: (
                float(summaries[name]["storage"]["storage_ratio"]),
                -float(summaries[name]["fixed_scale_recovery"]),
                int(summaries[name]["block_size"]),
            ),
        )
        if passing
        else None
    )

    args.output.mkdir(parents=True, exist_ok=False)
    cells_path = args.output / "cells.pt"
    codecs_path = args.output / "codec_rows.pt"
    torch.save(rows, cells_path)
    torch.save(codec_rows, codecs_path)
    repo_root = Path(__file__).resolve().parents[2]
    result = {
        "schema_version": RESULT_SCHEMA,
        "source_commit": git_commit(repo_root),
        "source_sha256": file_sha256(Path(__file__)),
        "plan": {"path": str(args.plan), "sha256": file_sha256(args.plan)},
        "inputs": {
            "probe_dir": str(args.probe_dir),
            "run_identity_sha256": expected_identity,
            "optimizer_probe_sha256": input_hashes,
        },
        "summaries": summaries,
        "decision": {
            "classification": (
                "PROMOTE_ATTENTION_CPROJ_LOWBIT_TRAJECTORY_ORACLE"
                if selected is not None
                else "REJECT_ATTENTION_CPROJ_LOWBIT_POLAR_CODEC"
            ),
            "selected_candidate": selected,
            "candidate_checks": decisions,
            "thresholds": thresholds,
            "language_model_training_authorized": False,
        },
        "artifacts": {
            "cells": {"path": str(cells_path), "sha256": file_sha256(cells_path)},
            "codec_rows": {
                "path": str(codecs_path),
                "sha256": file_sha256(codecs_path),
            },
        },
        "elapsed_seconds": time.time() - started,
        "parameter_updates": 0,
    }
    result_path = args.output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
