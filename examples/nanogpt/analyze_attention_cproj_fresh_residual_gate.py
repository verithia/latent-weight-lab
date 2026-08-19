#!/usr/bin/env python3
"""Gate fresh multi-pass sparse-Givens updates for attention c_proj.

This is a zero-update, same-state capacity diagnostic. At each sealed dense
Muon probe it fits candidate sparse rotation sequences to the exact applied
``attn.c_proj`` update. Every pass is selected afresh from the residual left
by previous passes; no future weight, learned dense basis, or task update is
used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.fast_task_matching import fast_muon_matched_permutations
from examples.nanogpt.muon_matched_givens import (
    apply_givens_flow,
    diagonal_metric_angles,
)


PLAN_SCHEMA = "mai_124m_attention_cproj_fresh_residual_gate_plan_v1"
RESULT_SCHEMA = "mai_124m_attention_cproj_fresh_residual_gate_result_v1"
PROBE_SCHEMA = "nanogpt_optimizer_probe_v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def parameter_name(layer: int) -> str:
    return f"transformer.h.{layer}.attn.c_proj.weight"


def apply_pass(
    source: torch.Tensor,
    residual: torch.Tensor,
    *,
    side: str,
    stages: int,
    neighbors: int,
    seed: int,
    native_cache: Path | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if side not in {"input", "output"}:
        raise ValueError(f"unsupported side: {side}")
    values = source if side == "input" else source.T.contiguous()
    requested = residual if side == "input" else residual.T.contiguous()
    permutations, matching = fast_muon_matched_permutations(
        values,
        requested,
        stages=stages,
        neighbors=neighbors,
        seed=seed,
        cache_dir=native_cache,
    )
    permutations = permutations.to(device=values.device, dtype=torch.long)
    angles = diagonal_metric_angles(values, requested, permutations)
    updated = apply_givens_flow(
        values,
        angles,
        permutations,
        torch.argsort(permutations, dim=1),
    )
    if side == "output":
        updated = updated.T.contiguous()
    return updated, {
        "side": side,
        "stages": stages,
        "coordinates": stages * (values.shape[1] // 2),
        "mean_abs_angle": float(angles.abs().mean()),
        "maximum_abs_angle": float(angles.abs().max()),
        "candidate_edge_fraction": float(matching["candidate_edge_fraction"]),
        "minimum_stage_candidate_edge_fraction": float(
            matching["minimum_stage_candidate_edge_fraction"]
        ),
        "matcher_seconds": float(matching["total_seconds"]),
        "native_output_validated": bool(matching["native_output_validated"]),
        "native_library_sha256": str(matching["native_library_sha256"]),
        "native_source_sha256": str(matching["source_sha256"]),
    }


def build_candidate(
    source: torch.Tensor,
    target: torch.Tensor,
    passes: list[dict[str, Any]],
    *,
    neighbors: int,
    seed: int,
    native_cache: Path | None,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    transformed = source.float()
    rows: list[dict[str, Any]] = []
    for pass_index, spec in enumerate(passes):
        prediction = transformed - source.float()
        residual = target.float() - prediction
        transformed, diagnostics = apply_pass(
            transformed,
            residual,
            side=str(spec["side"]),
            stages=int(spec["stages"]),
            neighbors=neighbors,
            seed=seed + pass_index,
            native_cache=native_cache,
        )
        post_prediction = transformed - source.float()
        target_energy = target.double().square().sum().clamp_min(1e-30)
        post_residual = target.float() - post_prediction.float()
        rows.append(
            {
                "pass_index": pass_index,
                **diagnostics,
                "cumulative_fixed_scale_recovery": float(
                    1.0
                    - post_residual.double().square().sum() / target_energy
                ),
            }
        )
    return transformed - source.float(), rows


def metrics(target: torch.Tensor, prediction: torch.Tensor) -> dict[str, float]:
    target_d = target.double()
    prediction_d = prediction.double()
    target_energy = target_d.square().sum().clamp_min(1e-30)
    prediction_energy = prediction_d.square().sum()
    residual_energy = (target_d - prediction_d).square().sum()
    inner = (target_d * prediction_d).sum()
    cosine = inner / (
        target_energy.sqrt() * prediction_energy.sqrt().clamp_min(1e-30)
    )
    positive_line_recovery = torch.where(
        inner > 0,
        inner.square() / (target_energy * prediction_energy.clamp_min(1e-30)),
        torch.zeros_like(inner),
    )
    return {
        "target_energy": float(target_energy),
        "prediction_energy": float(prediction_energy),
        "residual_energy": float(residual_energy),
        "fixed_scale_recovery": float(1.0 - residual_energy / target_energy),
        "positive_line_recovery": float(positive_line_recovery),
        "cosine": float(cosine),
        "descent_fraction": float(inner / target_energy),
        "prediction_to_target_norm": float(
            prediction_energy.sqrt() / target_energy.sqrt()
        ),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    target_energy = sum(float(row["target_energy"]) for row in rows)
    residual_energy = sum(float(row["residual_energy"]) for row in rows)
    prediction_energy = sum(float(row["prediction_energy"]) for row in rows)
    inner = sum(
        float(row["descent_fraction"]) * float(row["target_energy"])
        for row in rows
    )
    positive_line = 0.0
    if inner > 0.0 and prediction_energy > 0.0:
        positive_line = inner * inner / (target_energy * prediction_energy)
    return {
        "cells": len(rows),
        "target_energy": target_energy,
        "fixed_scale_recovery": 1.0 - residual_energy / target_energy,
        "positive_line_recovery": positive_line,
        "cosine": inner / max((target_energy * prediction_energy) ** 0.5, 1e-30),
        "descent_fraction": inner / target_energy,
        "prediction_to_target_norm": (prediction_energy / target_energy) ** 0.5,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--probe-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--native-cache", type=Path)
    args = parser.parse_args()
    started = time.time()
    plan = json.loads(args.plan.read_text())
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unexpected plan schema")
    protocol = plan["protocol"]
    steps = [int(value) for value in protocol["probe_steps"]]
    layers = [int(value) for value in protocol["layers"]]
    neighbors = int(protocol["neighbors"])
    base_seed = int(protocol["matching_seed"])
    candidate_specs = protocol["candidates"]
    expected_hashes = plan["identity"]["optimizer_probe_sha256"]
    expected_identity = str(plan["identity"]["trajectory_run_identity_sha256"])
    probes: dict[int, dict[str, Any]] = {}
    input_hashes: dict[str, str] = {}
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
        probes[step] = probe
        input_hashes[path.name] = digest

    rows: list[dict[str, Any]] = []
    pass_rows: list[dict[str, Any]] = []
    for step in steps:
        probe = probes[step]
        for layer in layers:
            name = parameter_name(layer)
            record = probe["parameters"][name]
            learning_rate = float(probe["hyperparameters"][name]["lr"])
            source = record["weight_before_step"].to(args.device, dtype=torch.float32)
            target = learning_rate * record["applied_direction_per_lr"].to(
                args.device, dtype=torch.float32
            )
            for candidate_index, (candidate, passes) in enumerate(
                candidate_specs.items()
            ):
                prediction, diagnostics = build_candidate(
                    source,
                    target,
                    list(passes),
                    neighbors=neighbors,
                    seed=(
                        base_seed
                        + step * int(protocol["seed_step_stride"])
                        + layer * int(protocol["seed_layer_stride"])
                        + candidate_index * int(protocol["seed_candidate_stride"])
                    ),
                    native_cache=args.native_cache,
                )
                coordinate_count = sum(
                    int(item["stages"]) * (source.shape[0] // 2)
                    for item in passes
                )
                row = {
                    "step": step,
                    "layer": layer,
                    "candidate": candidate,
                    "coordinate_count": coordinate_count,
                    "ambient_count": source.numel(),
                    "coordinate_fraction": coordinate_count / source.numel(),
                    **metrics(target, prediction),
                }
                rows.append(row)
                pass_rows.extend(
                    {"step": step, "layer": layer, "candidate": candidate, **item}
                    for item in diagnostics
                )
                print(json.dumps(row, sort_keys=True), flush=True)

    summaries: dict[str, Any] = {}
    for candidate in candidate_specs:
        selected = [row for row in rows if row["candidate"] == candidate]
        by_step = {
            str(step): aggregate([row for row in selected if int(row["step"]) == step])
            for step in steps
        }
        by_layer = {
            str(layer): aggregate(
                [row for row in selected if int(row["layer"]) == layer]
            )
            for layer in layers
        }
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
                "minimum_cell_descent_fraction": min(
                    float(row["descent_fraction"]) for row in selected
                ),
                "by_step": by_step,
                "by_layer": by_layer,
            }
        )
        summaries[candidate] = summary

    decision_rule = plan["decision_rule"]
    thresholds = decision_rule["thresholds"]
    control = summaries[decision_rule["control"]]
    equal_budget_control_name = str(decision_rule["equal_budget_control"])
    equal_budget_control = summaries[equal_budget_control_name]
    decisions: dict[str, Any] = {}
    passing: list[str] = []
    for candidate in decision_rule["promotion_candidates"]:
        summary = summaries[candidate]
        improvement = float(summary["fixed_scale_recovery"]) - float(
            control["fixed_scale_recovery"]
        )
        equal_budget_ratio = float(summary["fixed_scale_recovery"]) / max(
            float(equal_budget_control["fixed_scale_recovery"]), 1e-30
        )
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
            "aggregate_cosine": float(summary["cosine"])
            >= float(thresholds["aggregate_cosine_minimum"]),
            "minimum_cell_descent_positive": float(
                summary["minimum_cell_descent_fraction"]
            )
            > float(thresholds["minimum_cell_descent_fraction"]),
            "improvement_over_output64": improvement
            >= float(thresholds["output64_absolute_improvement"]),
            "coordinate_fraction": float(summary["coordinate_fraction"])
            <= float(thresholds["maximum_coordinate_fraction"]),
        }
        if candidate != equal_budget_control_name:
            checks["beats_equal_budget_monolithic"] = equal_budget_ratio >= float(
                thresholds["structured_over_monolithic_ratio"]
            )
        passed = all(checks.values())
        decisions[candidate] = {
            "checks": checks,
            "passed": passed,
            "output64_absolute_improvement": improvement,
            "equal_budget_monolithic_ratio": equal_budget_ratio,
        }
        if passed:
            passing.append(candidate)
    selected = (
        max(
            passing,
            key=lambda name: (
                float(summaries[name]["fixed_scale_recovery"])
                - float(summaries[name]["coordinate_fraction"]),
                -int(summaries[name]["coordinate_count"]),
            ),
        )
        if passing
        else None
    )

    args.output.mkdir(parents=True, exist_ok=False)
    cells_path = args.output / "cells.pt"
    passes_path = args.output / "passes.pt"
    torch.save(rows, cells_path)
    torch.save(pass_rows, passes_path)
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
                "PROMOTE_ATTENTION_CPROJ_FRESH_RESIDUAL"
                if selected is not None
                else "REJECT_ATTENTION_CPROJ_FRESH_RESIDUAL"
            ),
            "selected_candidate": selected,
            "candidate_checks": decisions,
            "thresholds": thresholds,
        },
        "artifacts": {
            "cells": {"path": str(cells_path), "sha256": file_sha256(cells_path)},
            "passes": {"path": str(passes_path), "sha256": file_sha256(passes_path)},
        },
        "elapsed_seconds": time.time() - started,
        "parameter_updates": 0,
    }
    result_path = args.output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
