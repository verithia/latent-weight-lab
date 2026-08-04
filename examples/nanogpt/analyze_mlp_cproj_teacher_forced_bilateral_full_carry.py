#!/usr/bin/env python3
"""Replay task-conditioned bilateral c_proj rotations with full carry."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.nanogpt.analyze_mlp_cproj_teacher_forced_bilateral_replay import (
    cell_metrics,
    cosine_lr,
    file_sha256,
    fit_right_pass,
    git_commit,
    load_snapshot,
)
from examples.nanogpt.analyze_parameter_trajectory import write_csv


@dataclass(frozen=True)
class Arm:
    name: str
    output_stages: int


ARMS = (
    Arm("hidden88_full_carry", 0),
    Arm("hidden88_output32_full_carry", 32),
    Arm("hidden88_output64_full_carry", 64),
)
CONTROL = ARMS[0].name
CANDIDATE_ORDER = tuple(arm.name for arm in ARMS[1:])


def fit_output_pass(
    weight: torch.Tensor,
    target_update: torch.Tensor,
    *,
    stages: int,
    neighbors: int,
    seed: int,
) -> torch.Tensor:
    """Fit a left/output rotation by applying the right matcher to W^T."""
    if stages == 0:
        return weight
    updated_t = fit_right_pass(
        weight.T.contiguous(),
        target_update.T.contiguous(),
        stages=stages,
        neighbors=min(neighbors, weight.shape[0] - 1),
        seed=seed,
    )
    return updated_t.T.contiguous()


def structured_step(
    weight: torch.Tensor,
    requested_update: torch.Tensor,
    feedback: torch.Tensor,
    *,
    parent_connectivity_update: torch.Tensor | None = None,
    output_stages: int,
    learning_rate: float,
    weight_decay: float,
    neighbors: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    corrected = requested_update.float() + feedback.float()
    current = weight.float()
    residual = corrected
    for pass_index, stages in enumerate((64, 24)):
        updated = fit_right_pass(
            current,
            residual,
            matching_target_update=(
                parent_connectivity_update if pass_index == 0 else None
            ),
            stages=stages,
            neighbors=neighbors,
            seed=seed + pass_index,
        )
        residual = residual - (updated - current)
        current = updated
    if output_stages:
        updated = fit_output_pass(
            current,
            residual,
            stages=output_stages,
            neighbors=neighbors,
            seed=seed + 2,
        )
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


def _scores(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    scores: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        for score_step in sorted({int(row["score_step"]) for row in rows}):
            selected = [
                row
                for row in rows
                if row["arm"] == arm.name
                and int(row["score_step"]) == score_step
            ]
            chord_energy = sum(float(row["chord_energy"]) for row in selected)
            endpoint_error = sum(
                float(row["endpoint_error_energy"]) for row in selected
            )
            gram_energy = sum(
                float(row["row_gram_chord_energy"]) for row in selected
            )
            gram_error = sum(
                float(row["row_gram_error_energy"]) for row in selected
            )
            scores[f"{arm.name}@{score_step}"] = {
                "arm": arm.name,
                "score_step": score_step,
                "layers": len(selected),
                "aggregate_endpoint_recovery": 1.0
                - endpoint_error / max(chord_energy, 1e-30),
                "aggregate_endpoint_error_energy": endpoint_error,
                "aggregate_chord_energy": chord_energy,
                "aggregate_row_gram_recovery": 1.0
                - gram_error / max(gram_energy, 1e-30),
                "aggregate_row_gram_error_energy": gram_error,
                "minimum_layer_endpoint_recovery": min(
                    float(row["endpoint_recovery"]) for row in selected
                ),
                "maximum_feedback_fro": max(
                    float(row["terminal_feedback_fro"]) for row in selected
                ),
                "mean_requested_update_recovery": sum(
                    float(row["mean_requested_update_recovery"])
                    for row in selected
                )
                / len(selected),
                "all_finite": all(
                    all(
                        math.isfinite(float(value))
                        for key, value in row.items()
                        if key not in {"arm", "layer", "score_step"}
                    )
                    for row in selected
                ),
            }
    return scores


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = _scores(rows)
    steps = sorted({int(row["score_step"]) for row in rows})
    terminal_step = steps[-1]
    control_terminal = scores[f"{CONTROL}@{terminal_step}"]
    comparisons: dict[str, dict[str, Any]] = {}
    for candidate in CANDIDATE_ORDER:
        candidate_terminal = scores[f"{candidate}@{terminal_step}"]
        recovery_ratios = {
            str(step): float(
                scores[f"{candidate}@{step}"]["aggregate_endpoint_recovery"]
            )
            / max(
                float(scores[f"{CONTROL}@{step}"]["aggregate_endpoint_recovery"]),
                1e-30,
            )
            for step in steps
        }
        candidate_cells = {
            int(row["layer"]): row
            for row in rows
            if row["arm"] == candidate
            and int(row["score_step"]) == terminal_step
        }
        control_cells = {
            int(row["layer"]): row
            for row in rows
            if row["arm"] == CONTROL
            and int(row["score_step"]) == terminal_step
        }
        comparison = {
            "endpoint_recovery_ratios_by_step": recovery_ratios,
            "all_score_steps_no_worse": all(
                value >= 1.0 for value in recovery_ratios.values()
            ),
            "terminal_endpoint_error_energy_ratio": float(
                candidate_terminal["aggregate_endpoint_error_energy"]
            )
            / max(
                float(control_terminal["aggregate_endpoint_error_energy"]),
                1e-30,
            ),
            "terminal_row_gram_error_energy_ratio": float(
                candidate_terminal["aggregate_row_gram_error_energy"]
            )
            / max(
                float(control_terminal["aggregate_row_gram_error_energy"]),
                1e-30,
            ),
            "terminal_feedback_ratio": float(
                candidate_terminal["maximum_feedback_fro"]
            )
            / max(float(control_terminal["maximum_feedback_fro"]), 1e-30),
            "terminal_layers_won": sum(
                float(candidate_cells[layer]["endpoint_recovery"])
                > float(control_cells[layer]["endpoint_recovery"])
                for layer in candidate_cells
            ),
            "required_terminal_layers_won": 4,
            "terminal_requested_update_recovery_no_worse": float(
                candidate_terminal["mean_requested_update_recovery"]
            )
            >= float(control_terminal["mean_requested_update_recovery"]),
            "all_finite": all(
                bool(scores[f"{candidate}@{step}"]["all_finite"])
                and bool(scores[f"{CONTROL}@{step}"]["all_finite"])
                for step in steps
            ),
        }
        comparison["passed"] = bool(
            comparison["all_finite"]
            and comparison["all_score_steps_no_worse"]
            and comparison["terminal_endpoint_error_energy_ratio"] <= 0.85
            and comparison["terminal_row_gram_error_energy_ratio"] <= 0.85
            and comparison["terminal_feedback_ratio"] <= 0.95
            and comparison["terminal_layers_won"] >= 4
            and comparison["terminal_requested_update_recovery_no_worse"]
        )
        comparisons[candidate] = comparison
    selected = next(
        (name for name in CANDIDATE_ORDER if comparisons[name]["passed"]), None
    )
    return {
        "scores": scores,
        "comparisons": comparisons,
        "selected_arm": selected if selected is not None else CONTROL,
        "decision": (
            f"SELECT_{selected.upper()}_FOR_PRODUCTION_PREFLIGHT"
            if selected is not None
            else "CLOSE_PRACTICAL_BILATERAL_ORTHOGONAL_BRANCH"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", default="0,3,6,9,11")
    parser.add_argument("--score-steps", default="60,120,180,238")
    parser.add_argument("--neighbors", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    layers = [int(value) for value in args.layers.split(",")]
    score_steps = [int(value) for value in args.score_steps.split(",")]
    if score_steps != sorted(set(score_steps)) or score_steps[-1] != 238:
        raise ValueError("score steps must be sorted, unique, and end at 238")
    paths = [args.snapshot_dir / f"step_{step:06d}.pt" for step in range(239)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"missing {len(missing)} snapshots; first={missing[0]}")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("schema_version") != (
        "mai_124m_mlp_cproj_teacher_forced_bilateral_full_carry_plan_v1"
    ):
        raise ValueError("unexpected plan schema")

    first = load_snapshot(paths[0])
    identity = first["run_identity_sha256"]
    if identity != plan["inputs"]["trajectory_run_identity_sha256"]:
        raise ValueError("trajectory identity mismatch")
    model_config = first["model_config"]
    config = first["run_identity"]["resolved_config"]
    if config.get("data_manifest_sha256") != plan["inputs"]["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest mismatch")
    names = [f"transformer.h.{layer}.mlp.c_proj.weight" for layer in layers]
    if any(name not in first["parameters"] for name in names):
        raise ValueError("trajectory does not contain all requested c_proj layers")

    starts = {
        layer: first["parameters"][name].to(args.device).float()
        for layer, name in zip(layers, names, strict=True)
    }
    dense_previous = {layer: value.clone() for layer, value in starts.items()}
    states = {
        (arm.name, layer): starts[layer].clone()
        for arm in ARMS
        for layer in layers
    }
    feedback = {key: torch.zeros_like(value) for key, value in states.items()}
    recoveries: dict[tuple[str, int], list[float]] = {key: [] for key in states}
    rows: list[dict[str, Any]] = []
    for step in range(238):
        payload = load_snapshot(paths[step + 1])
        if payload["run_identity_sha256"] != identity:
            raise ValueError(f"run identity mismatch at step {step + 1}")
        lr = cosine_lr(
            step,
            learning_rate=float(config["learning_rate"]),
            min_lr=float(config["min_lr"]),
            warmup_iters=int(config["warmup_iters"]),
            decay_iters=int(config["lr_decay_iters"]),
        )
        weight_decay = float(config["weight_decay"])
        for layer, name in zip(layers, names, strict=True):
            dense_before = dense_previous[layer]
            dense_after = payload["parameters"][name].to(args.device).float()
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
                    output_stages=arm.output_stages,
                    learning_rate=lr,
                    weight_decay=weight_decay,
                    neighbors=args.neighbors,
                    seed=args.seed + layer * 100000 + step * 10,
                )
                states[key] = updated
                feedback[key] = new_feedback
                recoveries[key].append(recovery)
            dense_previous[layer] = dense_after
        score_step = step + 1
        if score_step in score_steps:
            for layer in layers:
                for arm in ARMS:
                    key = (arm.name, layer)
                    row = {
                        "arm": arm.name,
                        "layer": layer,
                        "score_step": score_step,
                        **cell_metrics(
                            starts[layer],
                            dense_previous[layer],
                            states[key],
                            recoveries[key],
                            feedback[key],
                        ),
                    }
                    rows.append(row)
                    print(json.dumps(row, sort_keys=True), flush=True)
        elif score_step == 1 or score_step % 10 == 0:
            print(json.dumps({"step": score_step}), flush=True)
        del payload

    aggregate = aggregate_rows(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    cells_path = args.output / "cproj_bilateral_full_carry_cells.csv"
    aggregate_path = args.output / "cproj_bilateral_full_carry_result.json"
    write_csv(cells_path, rows)
    aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_cproj_teacher_forced_bilateral_full_carry_v1",
        "run_identity_sha256": identity,
        "model_config": model_config,
        "resolved_training_config": config,
        "layers": layers,
        "score_steps": score_steps,
        "neighbors": args.neighbors,
        "snapshot_inventory": {
            "count": len(paths),
            "first_sha256": file_sha256(paths[0]),
            "last_sha256": file_sha256(paths[-1]),
            "total_bytes": sum(path.stat().st_size for path in paths),
        },
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
            "Directions are teacher-forced from one dense run, so this is an optimistic temporal transport oracle.",
            "No language-model loss is evaluated and no parameter or optimizer state is updated.",
        ],
    }
    metadata_path = args.output / "cproj_bilateral_full_carry_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"decision": aggregate["decision"], "metadata": str(metadata_path)},
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
