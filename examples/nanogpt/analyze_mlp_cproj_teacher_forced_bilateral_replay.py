#!/usr/bin/env python3
"""Teacher-force compact c_proj charts through a saved dense trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.nanogpt.analyze_parameter_trajectory import write_csv
from examples.nanogpt.fast_task_matching import fast_muon_matched_permutations
from examples.nanogpt.muon_matched_givens import (
    apply_givens_flow,
    diagonal_metric_angles,
)
from examples.nanogpt.parameter_trajectory import SCHEMA_VERSION


@dataclass(frozen=True)
class Arm:
    name: str
    right_extra_stages: int
    output_stages: int
    feedback_decay: float


ARMS = (
    Arm("hidden88_decay0p5", 0, 0, 0.5),
    Arm("hidden88_decay1p0", 0, 0, 1.0),
    Arm("hidden96_decay0p5", 8, 0, 0.5),
    Arm("hidden104_decay0p5", 16, 0, 0.5),
    Arm("hidden88_output32_decay0p5", 0, 32, 0.5),
    Arm("hidden88_output64_decay0p5", 0, 64, 0.5),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


def cosine_lr(
    step: int,
    *,
    learning_rate: float,
    min_lr: float,
    warmup_iters: int,
    decay_iters: int,
) -> float:
    if step < warmup_iters:
        return learning_rate * (step + 1) / (warmup_iters + 1)
    if step > decay_iters:
        return min_lr
    ratio = (step - warmup_iters) / (decay_iters - warmup_iters)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coefficient * (learning_rate - min_lr)


def fit_right_pass(
    weight: torch.Tensor,
    target_update: torch.Tensor,
    *,
    stages: int,
    neighbors: int,
    seed: int,
) -> torch.Tensor:
    if stages == 0:
        return weight
    permutations, _ = fast_muon_matched_permutations(
        weight,
        target_update,
        stages=stages,
        neighbors=neighbors,
        seed=seed,
    )
    permutations = permutations.to(device=weight.device)
    inverse = torch.argsort(permutations, dim=1)
    angles = diagonal_metric_angles(weight, target_update, permutations)
    return apply_givens_flow(weight, angles, permutations, inverse)


def structured_step(
    weight: torch.Tensor,
    requested_update: torch.Tensor,
    feedback: torch.Tensor,
    *,
    arm: Arm,
    learning_rate: float,
    weight_decay: float,
    neighbors: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    corrected = requested_update.float() + arm.feedback_decay * feedback.float()
    current = weight.float()
    residual = corrected
    for pass_index, stages in enumerate((64, 24, arm.right_extra_stages)):
        if not stages:
            continue
        updated = fit_right_pass(
            current,
            residual,
            stages=stages,
            neighbors=neighbors,
            seed=seed + pass_index,
        )
        residual = residual - (updated - current)
        current = updated
    if arm.output_stages:
        transposed = current.T.contiguous()
        updated_t = fit_right_pass(
            transposed,
            residual.T.contiguous(),
            stages=arm.output_stages,
            neighbors=min(neighbors, transposed.shape[1] - 1),
            seed=seed + 3,
        )
        updated = updated_t.T.contiguous()
        residual = residual - (updated - current)
        current = updated
    if weight_decay:
        current = current * (1.0 - learning_rate * weight_decay)
    actual = current - weight.float()
    new_feedback = corrected - actual
    energy = requested_update.float().square().sum().clamp_min(1e-30)
    recovery = float(
        1.0 - (requested_update.float() - actual).square().sum() / energy
    )
    return current, new_feedback.contiguous(), recovery


def cell_metrics(
    start: torch.Tensor,
    dense_end: torch.Tensor,
    candidate_end: torch.Tensor,
    update_recoveries: list[float],
    feedback: torch.Tensor,
) -> dict[str, float]:
    chord = dense_end.float() - start.float()
    candidate = candidate_end.float() - start.float()
    chord_energy = chord.square().sum().clamp_min(1e-30)
    endpoint_error = (dense_end.float() - candidate_end.float()).square().sum()
    cosine = float(
        (chord * candidate).sum()
        / (chord.norm() * candidate.norm()).clamp_min(1e-30)
    )
    start_gram = start.float() @ start.float().T
    dense_gram = dense_end.float() @ dense_end.float().T
    candidate_gram = candidate_end.float() @ candidate_end.float().T
    gram_chord = dense_gram - start_gram
    gram_energy = gram_chord.square().sum().clamp_min(1e-30)
    gram_error = (dense_gram - candidate_gram).square().sum()
    return {
        "chord_energy": float(chord_energy),
        "endpoint_error_energy": float(endpoint_error),
        "endpoint_recovery": float(1.0 - endpoint_error / chord_energy),
        "endpoint_cosine": cosine,
        "row_gram_chord_energy": float(gram_energy),
        "row_gram_error_energy": float(gram_error),
        "row_gram_recovery": float(1.0 - gram_error / gram_energy),
        "mean_requested_update_recovery": sum(update_recoveries)
        / len(update_recoveries),
        "terminal_feedback_fro": float(feedback.float().norm()),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, dict[str, float | int]] = {}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm.name]
        chord_energy = sum(float(row["chord_energy"]) for row in selected)
        gram_energy = sum(
            float(row["row_gram_chord_energy"]) for row in selected
        )
        endpoint_error = sum(
            float(row["endpoint_error_energy"]) for row in selected
        )
        gram_error = sum(
            float(row["row_gram_error_energy"]) for row in selected
        )
        by_arm[arm.name] = {
            "cells": len(selected),
            "aggregate_endpoint_recovery": 1.0
            - endpoint_error / max(chord_energy, 1e-30),
            "aggregate_row_gram_recovery": 1.0
            - gram_error / max(gram_energy, 1e-30),
            "aggregate_endpoint_error_energy": endpoint_error,
            "aggregate_row_gram_error_energy": gram_error,
            "minimum_endpoint_recovery": min(
                float(row["endpoint_recovery"]) for row in selected
            ),
            "mean_requested_update_recovery": sum(
                float(row["mean_requested_update_recovery"])
                for row in selected
            )
            / len(selected),
            "maximum_terminal_feedback_fro": max(
                float(row["terminal_feedback_fro"]) for row in selected
            ),
            "all_finite": all(
                all(
                    math.isfinite(float(value))
                    for key, value in row.items()
                    if key not in {"arm", "layer", "phase_start", "phase_end"}
                )
                for row in selected
            ),
        }

    comparisons: dict[str, dict[str, float | int | bool]] = {}
    for candidate_name, control_name in (
        ("hidden88_output32_decay0p5", "hidden96_decay0p5"),
        ("hidden88_output64_decay0p5", "hidden104_decay0p5"),
    ):
        candidate = by_arm[candidate_name]
        control = by_arm[control_name]
        candidate_cells = {
            (int(row["layer"]), int(row["phase_start"])): row
            for row in rows
            if row["arm"] == candidate_name
        }
        control_cells = {
            (int(row["layer"]), int(row["phase_start"])): row
            for row in rows
            if row["arm"] == control_name
        }
        control_recovery = float(control["aggregate_endpoint_recovery"])
        candidate_recovery = float(candidate["aggregate_endpoint_recovery"])
        endpoint_ratio = (
            candidate_recovery / control_recovery
            if control_recovery > 0.0
            else float("-inf")
        )
        row_gram_error_reduction = 1.0 - float(
            candidate["aggregate_row_gram_error_energy"]
        ) / max(float(control["aggregate_row_gram_error_energy"]), 1e-30)
        improved_cells = sum(
            float(candidate_cells[cell]["endpoint_recovery"])
            > float(control_cells[cell]["endpoint_recovery"])
            for cell in candidate_cells
        )
        passed = bool(
            candidate["all_finite"]
            and endpoint_ratio >= 1.10
            and row_gram_error_reduction >= 0.25
            and improved_cells >= 16
        )
        comparisons[candidate_name] = {
            "control": control_name,
            "endpoint_recovery_ratio": endpoint_ratio,
            "row_gram_error_reduction": row_gram_error_reduction,
            "improved_endpoint_cells": improved_cells,
            "required_improved_cells": 16,
            "passed": passed,
        }
    if comparisons["hidden88_output32_decay0p5"]["passed"]:
        decision = "SELECT_OUTPUT32_FOR_PRODUCTION_PREFLIGHT"
    elif comparisons["hidden88_output64_decay0p5"]["passed"]:
        decision = "SELECT_OUTPUT64_FOR_PRODUCTION_PREFLIGHT"
    else:
        decision = "REJECT_ADDITIVE_SPARSE_OUTPUT_TRANSPORT"
    return {"arms": by_arm, "comparisons": comparisons, "decision": decision}


def load_snapshot(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported snapshot: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", default="0,3,6,9,11")
    parser.add_argument("--phase-boundaries", default="0,60,120,180,238")
    parser.add_argument("--neighbors", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    layers = [int(value) for value in args.layers.split(",")]
    boundaries = [int(value) for value in args.phase_boundaries.split(",")]
    if boundaries != sorted(set(boundaries)) or len(boundaries) < 2:
        raise ValueError("phase boundaries must be sorted and unique")
    paths = [args.snapshot_dir / f"step_{step:06d}.pt" for step in range(boundaries[-1] + 1)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"missing {len(missing)} snapshots; first={missing[0]}")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("schema_version") != "mai_124m_mlp_cproj_teacher_forced_bilateral_replay_plan_v1":
        raise ValueError("unexpected plan schema")

    first = load_snapshot(paths[0])
    identity = first["run_identity_sha256"]
    model_config = first["model_config"]
    config = first["run_identity"]["resolved_config"]
    expected_names = [f"transformer.h.{layer}.mlp.c_proj.weight" for layer in layers]
    if any(name not in first["parameters"] for name in expected_names):
        raise ValueError("trajectory does not contain all requested c_proj layers")
    if config.get("data_manifest_sha256") != plan["inputs"]["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest mismatch")

    rows: list[dict[str, Any]] = []
    for phase_start, phase_end in zip(boundaries[:-1], boundaries[1:], strict=True):
        start_payload = load_snapshot(paths[phase_start])
        states: dict[tuple[str, int], torch.Tensor] = {}
        feedback: dict[tuple[str, int], torch.Tensor] = {}
        update_recovery: dict[tuple[str, int], list[float]] = {}
        dense_previous: dict[int, torch.Tensor] = {}
        for layer, name in zip(layers, expected_names, strict=True):
            dense = start_payload["parameters"][name].to(args.device).float()
            dense_previous[layer] = dense
            for arm in ARMS:
                key = (arm.name, layer)
                states[key] = dense.clone()
                feedback[key] = torch.zeros_like(dense)
                update_recovery[key] = []

        for step in range(phase_start, phase_end):
            next_payload = load_snapshot(paths[step + 1])
            if next_payload["run_identity_sha256"] != identity:
                raise ValueError(f"run identity mismatch at step {step + 1}")
            lr = cosine_lr(
                step,
                learning_rate=float(config["learning_rate"]),
                min_lr=float(config["min_lr"]),
                warmup_iters=int(config["warmup_iters"]),
                decay_iters=int(config["lr_decay_iters"]),
            )
            weight_decay = float(config["weight_decay"])
            for layer, name in zip(layers, expected_names, strict=True):
                dense_before = dense_previous[layer]
                dense_after = next_payload["parameters"][name].to(args.device).float()
                dense_delta = dense_after - dense_before
                dense_nondecay = dense_delta + lr * weight_decay * dense_before
                for arm in ARMS:
                    key = (arm.name, layer)
                    candidate = states[key]
                    requested = dense_nondecay - lr * weight_decay * candidate
                    updated, new_feedback, recovery = structured_step(
                        candidate,
                        requested,
                        feedback[key],
                        arm=arm,
                        learning_rate=lr,
                        weight_decay=weight_decay,
                        neighbors=args.neighbors,
                        seed=args.seed + layer * 100000 + step * 10,
                    )
                    states[key] = updated
                    feedback[key] = new_feedback
                    update_recovery[key].append(recovery)
                dense_previous[layer] = dense_after
            if step == phase_start or (step + 1) % 10 == 0 or step + 1 == phase_end:
                print(
                    json.dumps({"phase": [phase_start, phase_end], "step": step + 1}),
                    flush=True,
                )
            del next_payload

        for layer, name in zip(layers, expected_names, strict=True):
            start = start_payload["parameters"][name].to(args.device).float()
            dense_end = dense_previous[layer]
            for arm in ARMS:
                key = (arm.name, layer)
                row = {
                    "arm": arm.name,
                    "layer": layer,
                    "phase_start": phase_start,
                    "phase_end": phase_end,
                    **cell_metrics(
                        start,
                        dense_end,
                        states[key],
                        update_recovery[key],
                        feedback[key],
                    ),
                }
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
        del start_payload, states, feedback, update_recovery, dense_previous
        torch.cuda.empty_cache()

    aggregate = aggregate_rows(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    cells_path = args.output / "cproj_teacher_forced_bilateral_cells.csv"
    aggregate_path = args.output / "cproj_teacher_forced_bilateral_result.json"
    write_csv(cells_path, rows)
    aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    script = Path(__file__).resolve()
    inventory = {
        "count": len(paths),
        "first_sha256": file_sha256(paths[0]),
        "last_sha256": file_sha256(paths[-1]),
        "total_bytes": sum(path.stat().st_size for path in paths),
    }
    metadata = {
        "schema_version": "nanogpt_cproj_teacher_forced_bilateral_replay_v1",
        "run_identity_sha256": identity,
        "model_config": model_config,
        "resolved_training_config": config,
        "layers": layers,
        "phase_boundaries": boundaries,
        "snapshot_inventory": inventory,
        "plan": {"path": str(args.plan), "sha256": file_sha256(args.plan)},
        "analysis_execution": {
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": str(script),
            "entrypoint_sha256": file_sha256(script),
            "command": sys.argv,
            "device": args.device,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
        },
        "outputs": {
            "cells_sha256": file_sha256(cells_path),
            "aggregate_sha256": file_sha256(aggregate_path),
        },
        "limitations": [
            "Directions are teacher-forced from a dense run, so this is an optimistic representation and transport oracle.",
            "Only five preregistered layers and one dense trajectory are tested.",
            "No language-model loss is evaluated and no model parameter or optimizer state is updated.",
        ],
    }
    metadata_path = args.output / "cproj_teacher_forced_bilateral_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": aggregate["decision"], "metadata": str(metadata_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
