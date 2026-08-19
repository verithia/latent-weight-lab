#!/usr/bin/env python3
"""Gate equal-budget directed sparse transports for attention c_proj."""

from __future__ import annotations

import argparse
import json
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
from examples.nanogpt.muon_matched_givens import (
    batched_multistage_directed_sparse_update,
)


PLAN_SCHEMA = "mai_124m_attention_cproj_directed_transport_gate_plan_v1"
RESULT_SCHEMA = "mai_124m_attention_cproj_directed_transport_gate_result_v1"


def directed_pass(
    source: torch.Tensor,
    residual: torch.Tensor,
    *,
    side: str,
    schedule: list[int],
    ridge_ratio: float,
    chunk_size: int,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    if side not in {"input", "output"}:
        raise ValueError(f"unsupported directed side: {side}")
    values = source if side == "input" else source.transpose(1, 2).contiguous()
    target = residual if side == "input" else residual.transpose(1, 2).contiguous()
    prediction, stages = batched_multistage_directed_sparse_update(
        values,
        target,
        incoming_schedule=schedule,
        ridge_ratio=ridge_ratio,
        chunk_size=chunk_size,
    )
    if side == "output":
        prediction = prediction.transpose(1, 2).contiguous()
    return prediction, stages


def build_candidate(
    source: torch.Tensor,
    target: torch.Tensor,
    passes: list[dict[str, Any]],
    *,
    ridge_ratio: float,
    chunk_size: int,
    family_radius_ratio: float,
) -> tuple[torch.Tensor, list[dict[str, Any]], float]:
    transformed = source.float().clone()
    raw_prediction = torch.zeros_like(target, dtype=torch.float32)
    diagnostics: list[dict[str, Any]] = []
    for pass_index, spec in enumerate(passes):
        residual = target.float() - raw_prediction
        prediction, stage_rows = directed_pass(
            transformed,
            residual,
            side=str(spec["side"]),
            schedule=[int(value) for value in spec["schedule"]],
            ridge_ratio=ridge_ratio,
            chunk_size=chunk_size,
        )
        transformed.add_(prediction)
        raw_prediction.add_(prediction)
        diagnostics.append(
            {
                "pass_index": pass_index,
                "side": str(spec["side"]),
                "schedule": [int(value) for value in spec["schedule"]],
                "stage_rows": stage_rows,
            }
        )
    raw_norm = raw_prediction.double().square().sum().sqrt()
    target_norm = target.double().square().sum().sqrt()
    scale = float(family_radius_ratio) * target_norm / raw_norm.clamp_min(1e-30)
    return raw_prediction * scale, diagnostics, float(scale)


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
    candidates = protocol["candidates"]
    ridge_ratio = float(protocol["ridge_ratio"])
    chunk_size = int(protocol["chunk_size"])
    family_radius_ratio = float(protocol["family_radius_ratio"])
    expected_identity = str(plan["identity"]["trajectory_run_identity_sha256"])
    expected_hashes = plan["identity"]["optimizer_probe_sha256"]
    input_hashes: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []

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
        source_by_layer: list[torch.Tensor] = []
        target_by_layer: list[torch.Tensor] = []
        for layer in layers:
            name = parameter_name(layer)
            record = probe["parameters"][name]
            learning_rate = float(probe["hyperparameters"][name]["lr"])
            source_by_layer.append(record["weight_before_step"])
            target_by_layer.append(
                learning_rate * record["applied_direction_per_lr"]
            )
        source = torch.stack(source_by_layer).to(args.device, dtype=torch.float32)
        target = torch.stack(target_by_layer).to(args.device, dtype=torch.float32)
        for candidate, passes in candidates.items():
            if source.is_cuda:
                torch.cuda.synchronize(source.device)
            candidate_started = time.perf_counter()
            prediction, diagnostics, family_scale = build_candidate(
                source,
                target,
                list(passes),
                ridge_ratio=ridge_ratio,
                chunk_size=chunk_size,
                family_radius_ratio=family_radius_ratio,
            )
            if source.is_cuda:
                torch.cuda.synchronize(source.device)
            elapsed = time.perf_counter() - candidate_started
            coordinate_count = sum(
                sum(int(value) for value in item["schedule"]) * source.shape[1]
                for item in passes
            )
            timing_rows.append(
                {"step": step, "candidate": candidate, "seconds": elapsed}
            )
            for index, layer in enumerate(layers):
                row = {
                    "step": step,
                    "layer": layer,
                    "candidate": candidate,
                    "coordinate_count": coordinate_count,
                    "ambient_count": source[index].numel(),
                    "coordinate_fraction": coordinate_count / source[index].numel(),
                    "family_scale": family_scale,
                    **metrics(target[index], prediction[index]),
                }
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
            diagnostic_rows.append(
                {
                    "step": step,
                    "candidate": candidate,
                    "family_scale": family_scale,
                    "passes": diagnostics,
                }
            )

    summaries: dict[str, Any] = {}
    for candidate in candidates:
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
            for row in timing_rows
            if row["candidate"] == candidate
        ]
        summary = aggregate(selected)
        summary.update(
            {
                "coordinate_count": int(selected[0]["coordinate_count"]),
                "coordinate_fraction": float(selected[0]["coordinate_fraction"]),
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
                "mean_fit_seconds_per_phase": sum(timing) / len(timing),
                "maximum_fit_seconds_per_phase": max(timing),
                "by_step": by_step,
                "by_layer": by_layer,
            }
        )
        summaries[candidate] = summary

    thresholds = plan["decision_rule"]["thresholds"]
    orthogonal_control = float(
        plan["prior_evidence"]["best_equal_budget_orthogonal_recovery"]
    )
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
            "improvement_over_orthogonal": (
                float(summary["fixed_scale_recovery"]) - orthogonal_control
            )
            >= float(thresholds["orthogonal_absolute_improvement"]),
            "coordinate_fraction": float(summary["coordinate_fraction"])
            <= float(thresholds["maximum_coordinate_fraction"]),
        }
        passed = all(checks.values())
        decisions[candidate] = {
            "checks": checks,
            "passed": passed,
            "absolute_improvement_over_best_orthogonal": (
                float(summary["fixed_scale_recovery"]) - orthogonal_control
            ),
        }
        if passed:
            passing.append(candidate)
    selected = (
        max(
            passing,
            key=lambda name: (
                float(summaries[name]["fixed_scale_recovery"]),
                float(summaries[name]["minimum_cell_fixed_scale_recovery"]),
                -float(summaries[name]["mean_fit_seconds_per_phase"]),
            ),
        )
        if passing
        else None
    )

    args.output.mkdir(parents=True, exist_ok=False)
    cells_path = args.output / "cells.pt"
    diagnostics_path = args.output / "diagnostics.pt"
    torch.save(rows, cells_path)
    torch.save(diagnostic_rows, diagnostics_path)
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
                "PROMOTE_ATTENTION_CPROJ_DIRECTED_TRANSPORT"
                if selected is not None
                else "REJECT_ATTENTION_CPROJ_DIRECTED_TRANSPORT"
            ),
            "selected_candidate": selected,
            "candidate_checks": decisions,
            "thresholds": thresholds,
        },
        "artifacts": {
            "cells": {"path": str(cells_path), "sha256": file_sha256(cells_path)},
            "diagnostics": {
                "path": str(diagnostics_path),
                "sha256": file_sha256(diagnostics_path),
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
