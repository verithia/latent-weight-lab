#!/usr/bin/env python3
"""Test parallel transport of c_proj compression-error temporal state."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.nanogpt.analyze_mlp_cproj_activation_weighted_output_selector import (
    all_finite,
)
from examples.nanogpt.analyze_mlp_cproj_teacher_forced_bilateral_replay import (
    cell_metrics,
    cosine_lr,
    file_sha256,
    git_commit,
)
from examples.nanogpt.analyze_mlp_muon_matched_functional_metric import (
    evaluate_with_updates,
)
from examples.nanogpt.analyze_parameter_trajectory import write_csv
from examples.nanogpt.analyze_residual_compatibility import (
    fixed_validation_batches,
    load_model,
)
from examples.nanogpt.fast_task_matching import (
    fast_muon_matched_permutations,
)
from examples.nanogpt.muon_matched_givens import (
    apply_givens_flow,
    diagonal_metric_angles,
)


SCHEMA_VERSION = "mai_124m_mlp_cproj_parallel_transport_error_feedback_result_v1"
EXPECTED_PLAN_SCHEMA = "mai_124m_mlp_cproj_parallel_transport_error_feedback_plan_v2"
ARMS = ("ambient_carry", "pushforward_carry", "pullback_carry")
CANDIDATE_ORDER = ARMS[1:]
WINDOWS = ("fit", "holdout")


def validate_plan(plan: dict[str, Any]) -> None:
    analysis = plan.get("analysis", {})
    chart = analysis.get("shared_chart", {})
    expected = {
        "schema_version": EXPECTED_PLAN_SCHEMA,
        "parameter_updates": 0,
        "layers": [0, 3, 6, 9, 11],
        "score_steps": [60, 120, 180, 238],
        "finite_ce_score_steps": [238],
        "fit_seed": 20260804,
        "holdout_seed": 20260805,
        "parent_stages": 64,
        "residual_stages": 24,
        "neighbors": 64,
        "feedback_decay": 1.0,
        "smallest_pass_order": list(CANDIDATE_ORDER),
        "training_authorized": False,
    }
    observed = {
        "schema_version": plan.get("schema_version"),
        "parameter_updates": analysis.get("parameter_updates"),
        "layers": analysis.get("layers"),
        "score_steps": analysis.get("score_steps"),
        "finite_ce_score_steps": analysis.get("finite_ce_score_steps"),
        "fit_seed": analysis.get("fit_window", {}).get("seed"),
        "holdout_seed": analysis.get("holdout_window", {}).get("seed"),
        "parent_stages": chart.get("hidden_parent_stages"),
        "residual_stages": chart.get("hidden_residual_stages"),
        "neighbors": chart.get("neighbors"),
        "feedback_decay": chart.get("feedback_decay"),
        "smallest_pass_order": analysis.get("smallest_pass_order"),
        "training_authorized": plan.get("authorization", {}).get(
            "run_language_model_training"
        ),
    }
    if observed != expected:
        raise ValueError(
            "parallel-transport plan drifted: "
            f"observed={observed!r} expected={expected!r}"
        )


def parameter_name(layer: int) -> str:
    return f"transformer.h.{layer}.mlp.c_proj.weight"


def load_tracked_snapshot(path: Path, names: list[str]) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "nanogpt_parameter_trajectory_v1"
        or payload.get("all_parameters") is not False
    ):
        raise ValueError(f"not a tracked-parameter trajectory snapshot: {path}")
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict) or set(parameters) != set(names):
        raise ValueError(
            f"tracked snapshot parameter inventory mismatch: {path}"
        )
    return payload


def fit_right_pass_with_recipe(
    weight: torch.Tensor,
    target_update: torch.Tensor,
    *,
    stages: int,
    neighbors: int,
    seed: int,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    permutations, _diagnostics = fast_muon_matched_permutations(
        weight,
        target_update,
        stages=stages,
        neighbors=neighbors,
        seed=seed,
    )
    permutations = permutations.to(device=weight.device)
    inverse = torch.argsort(permutations, dim=1)
    angles = diagonal_metric_angles(weight, target_update, permutations)
    updated = apply_givens_flow(weight, angles, permutations, inverse)
    return updated, (angles, permutations, inverse)


def apply_transport_recipe(
    values: torch.Tensor,
    recipes: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    inverse: bool,
) -> torch.Tensor:
    """Apply the composed right flow or its exact inverse to ``values``."""
    output = values.float()
    ordered = reversed(recipes) if inverse else recipes
    for angles, permutations, inverse_permutations in ordered:
        if inverse:
            output = apply_givens_flow(
                output,
                -torch.flip(angles, dims=(0,)),
                torch.flip(permutations, dims=(0,)),
                torch.flip(inverse_permutations, dims=(0,)),
            )
        else:
            output = apply_givens_flow(
                output, angles, permutations, inverse_permutations
            )
    return output.contiguous()


def structured_step(
    weight: torch.Tensor,
    requested_update: torch.Tensor,
    feedback: torch.Tensor,
    *,
    arm: str,
    learning_rate: float,
    weight_decay: float,
    neighbors: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, float, dict[str, float]]:
    if arm not in ARMS:
        raise ValueError(f"unknown transport arm: {arm}")
    corrected = requested_update.float() + feedback.float()
    current = weight.float()
    residual = corrected
    recipes = []
    for pass_index, stages in enumerate((64, 24)):
        updated, recipe = fit_right_pass_with_recipe(
            current,
            residual,
            stages=stages,
            neighbors=neighbors,
            seed=seed + pass_index,
        )
        residual = residual - (updated - current)
        current = updated
        recipes.append(recipe)
    if weight_decay:
        current = current * (1.0 - learning_rate * weight_decay)
    actual = current - weight.float()
    raw_feedback = corrected - actual
    if arm == "ambient_carry":
        next_feedback = raw_feedback
    elif arm == "pushforward_carry":
        next_feedback = apply_transport_recipe(
            raw_feedback, recipes, inverse=False
        )
    else:
        next_feedback = apply_transport_recipe(
            raw_feedback, recipes, inverse=True
        )
    requested_energy = requested_update.float().square().sum().clamp_min(1e-30)
    recovery = float(
        1.0
        - (requested_update.float() - actual).square().sum()
        / requested_energy
    )
    diagnostics = {
        "raw_feedback_fro": float(raw_feedback.norm()),
        "stored_feedback_fro": float(next_feedback.norm()),
        "transport_norm_ratio": float(
            next_feedback.norm() / raw_feedback.norm().clamp_min(1e-30)
        ),
    }
    return current, next_feedback, recovery, diagnostics


def aggregate_results(
    rows: list[dict[str, Any]],
    finite_rows: list[dict[str, Any]],
    decision_rule: dict[str, Any],
) -> dict[str, Any]:
    scores: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        for score_step in sorted({int(row["score_step"]) for row in rows}):
            selected = [
                row
                for row in rows
                if row["arm"] == arm and int(row["score_step"]) == score_step
            ]
            chord_energy = sum(float(row["chord_energy"]) for row in selected)
            endpoint_error = sum(
                float(row["endpoint_error_energy"]) for row in selected
            )
            scores[f"{arm}@{score_step}"] = {
                "arm": arm,
                "score_step": score_step,
                "layers": len(selected),
                "aggregate_endpoint_recovery": 1.0
                - endpoint_error / max(chord_energy, 1e-30),
                "minimum_layer_endpoint_recovery": min(
                    float(row["endpoint_recovery"]) for row in selected
                ),
                "maximum_feedback_fro": max(
                    float(row["terminal_feedback_fro"]) for row in selected
                ),
                "maximum_transport_norm_error": max(
                    abs(float(row["transport_norm_ratio"]) - 1.0)
                    for row in selected
                ),
                "all_finite": all(all_finite(row) for row in selected),
            }

    indexed_losses = {
        (int(row["score_step"]), str(row["window"]), str(row["arm"])): float(
            row["loss"]
        )
        for row in finite_rows
    }
    requirements = decision_rule["candidate_requirements"]
    ambient_terminal = scores["ambient_carry@238"]
    finite_cells = sorted(
        {
            (int(row["score_step"]), str(row["window"]))
            for row in finite_rows
            if row["arm"] == "ambient_carry"
        }
    )
    ambient_cells = {
        int(row["layer"]): row
        for row in rows
        if row["arm"] == "ambient_carry" and int(row["score_step"]) == 238
    }
    candidates = {}
    selected_arm = None
    for arm in CANDIDATE_ORDER:
        comparisons = []
        for score_step, window in finite_cells:
            control = indexed_losses[(score_step, window, "ambient_carry")]
            candidate = indexed_losses[(score_step, window, arm)]
            comparisons.append(
                {
                    "score_step": score_step,
                    "window": window,
                    "control_loss": control,
                    "candidate_loss": candidate,
                    "gain": control - candidate,
                }
            )
        gains = [float(row["gain"]) for row in comparisons]
        holdout_gains = [
            float(row["gain"])
            for row in comparisons
            if row["window"] == "holdout"
        ]
        terminal = scores[f"{arm}@238"]
        candidate_cells = {
            int(row["layer"]): row
            for row in rows
            if row["arm"] == arm and int(row["score_step"]) == 238
        }
        endpoint_ratio = float(terminal["aggregate_endpoint_recovery"]) / max(
            float(ambient_terminal["aggregate_endpoint_recovery"]), 1e-30
        )
        winning_layers = sum(
            float(candidate_cells[layer]["endpoint_recovery"])
            > float(ambient_cells[layer]["endpoint_recovery"])
            for layer in candidate_cells
        )
        feedback_ratio = float(terminal["maximum_feedback_fro"]) / max(
            float(ambient_terminal["maximum_feedback_fro"]), 1e-30
        )
        gate = {
            "mean_finite_step_ce_gain": sum(gains) / len(gains)
            >= float(requirements["mean_finite_step_ce_gain_over_ambient_minimum"]),
            "finite_step_wins": sum(gain > 0.0 for gain in gains)
            >= int(requirements["finite_step_wins_minimum"]),
            "holdout_wins": sum(gain > 0.0 for gain in holdout_gains)
            >= int(requirements["holdout_wins_minimum"]),
            "minimum_holdout_gain": min(holdout_gains)
            >= float(requirements["minimum_holdout_finite_step_ce_gain"]),
            "terminal_endpoint_recovery": endpoint_ratio
            >= float(
                requirements[
                    "terminal_endpoint_recovery_ratio_over_ambient_minimum"
                ]
            ),
            "terminal_layers": winning_layers
            >= int(requirements["terminal_layers_beating_ambient_minimum"]),
            "maximum_feedback_fro": feedback_ratio
            <= float(requirements["maximum_feedback_fro_ratio_over_ambient"]),
            "finite": bool(
                terminal["all_finite"]
                and all_finite(comparisons)
                and all_finite(
                    {
                        "endpoint_ratio": endpoint_ratio,
                        "feedback_ratio": feedback_ratio,
                    }
                )
            ),
        }
        passed = all(gate.values())
        candidates[arm] = {
            "mean_finite_step_ce_gain": sum(gains) / len(gains),
            "finite_step_wins": sum(gain > 0.0 for gain in gains),
            "holdout_wins": sum(gain > 0.0 for gain in holdout_gains),
            "minimum_holdout_gain": min(holdout_gains),
            "terminal_endpoint_recovery_ratio_over_ambient": endpoint_ratio,
            "terminal_layers_beating_ambient": winning_layers,
            "maximum_feedback_fro_ratio_over_ambient": feedback_ratio,
            "comparisons": comparisons,
            "gate": gate,
            "passed": passed,
        }
        if selected_arm is None and passed:
            selected_arm = arm
    return {
        "scores": scores,
        "candidates": candidates,
        "selected_arm": selected_arm,
        "passed": selected_arm is not None,
        "classification": (
            "PASS_PARALLEL_TRANSPORT_ERROR_FEEDBACK"
            if selected_arm is not None
            else "REJECT_PARALLEL_TRANSPORT_ERROR_FEEDBACK"
        ),
        "authorization": {
            "production_implementation_authorized": selected_arm is not None,
            "exact_config_mfu_preflight_authorized": selected_arm is not None,
            "language_model_training_authorized": False,
        },
    }


def evaluate_model_loss(
    model: torch.nn.Module, batches: list[torch.Tensor], device: str
) -> float:
    losses = []
    with torch.no_grad():
        for tokens in batches:
            tokens = tokens.to(device)
            _logits, loss = model(
                tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
            )
            if loss is None:
                raise RuntimeError("model did not return a loss")
            losses.append(float(loss))
    return sum(losses) / len(losses)


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
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    if args.output.exists():
        raise FileExistsError(f"output directory already exists: {args.output}")
    manifest = args.data_dir / "manifest.json"
    if file_sha256(manifest) != plan["identity"]["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest SHA-256 mismatch")

    analysis = plan["analysis"]
    layers = [int(value) for value in analysis["layers"]]
    score_steps = [int(value) for value in analysis["score_steps"]]
    paths = [args.snapshot_dir / f"step_{step:06d}.pt" for step in range(239)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"missing {len(missing)} snapshots; first={missing[0]}")
    if file_sha256(paths[0]) != plan["identity"]["first_snapshot_sha256"]:
        raise ValueError("first snapshot SHA-256 mismatch")
    if file_sha256(paths[-1]) != plan["identity"]["last_snapshot_sha256"]:
        raise ValueError("last snapshot SHA-256 mismatch")
    if file_sha256(args.terminal_checkpoint) != plan["identity"][
        "terminal_checkpoint_sha256"
    ]:
        raise ValueError("terminal checkpoint SHA-256 mismatch")

    windows = {
        name: fixed_validation_batches(
            args.data_dir,
            int(analysis[f"{name}_window"]["batch_size"]),
            int(analysis[f"{name}_window"]["block_size"]) + 1,
            int(analysis[f"{name}_window"]["batches"]),
            int(analysis[f"{name}_window"]["seed"]),
        )
        for name in WINDOWS
    }
    names = [parameter_name(layer) for layer in layers]
    first = load_tracked_snapshot(paths[0], names)
    run_identity = plan["identity"]["trajectory_run_identity_sha256"]
    if first.get("run_identity_sha256") != run_identity:
        raise ValueError("trajectory identity mismatch")
    config = first["run_identity"]["resolved_config"]
    starts = {
        layer: first["parameters"][name].to(args.device).float()
        for layer, name in zip(layers, names, strict=True)
    }
    dense_previous = {layer: value.clone() for layer, value in starts.items()}
    states = {
        (arm, layer): starts[layer].clone() for arm in ARMS for layer in layers
    }
    feedback = {key: torch.zeros_like(value) for key, value in states.items()}
    recoveries = {key: [] for key in states}
    last_diagnostics = {key: {} for key in states}
    rows: list[dict[str, Any]] = []
    finite_rows: list[dict[str, Any]] = []

    for step in range(238):
        payload = load_tracked_snapshot(paths[step + 1], names)
        if payload.get("run_identity_sha256") != run_identity:
            raise ValueError(f"trajectory identity mismatch at step {step + 1}")
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
                key = (arm, layer)
                candidate = states[key]
                requested = dense_nondecay - lr * weight_decay * candidate
                updated, new_feedback, recovery, diagnostics = structured_step(
                    candidate,
                    requested,
                    feedback[key],
                    arm=arm,
                    learning_rate=lr,
                    weight_decay=weight_decay,
                    neighbors=int(analysis["shared_chart"]["neighbors"]),
                    seed=20260804 + layer * 100000 + step * 10,
                )
                states[key] = updated
                feedback[key] = new_feedback
                recoveries[key].append(recovery)
                last_diagnostics[key] = diagnostics
            dense_previous[layer] = dense_after

        score_step = step + 1
        if score_step in score_steps:
            for layer in layers:
                for arm in ARMS:
                    key = (arm, layer)
                    rows.append(
                        {
                            "arm": arm,
                            "layer": layer,
                            "score_step": score_step,
                            **cell_metrics(
                                starts[layer],
                                dense_previous[layer],
                                states[key],
                                recoveries[key],
                                feedback[key],
                            ),
                            **last_diagnostics[key],
                        }
                    )
        if score_step in analysis["finite_ce_score_steps"]:
            model = load_model(args.terminal_checkpoint, args.device)
            for layer, name in zip(layers, names, strict=True):
                torch.testing.assert_close(
                    model.transformer.h[layer].mlp.c_proj.weight.detach().cpu(),
                    payload["parameters"][name].float(),
                    rtol=0.0,
                    atol=0.0,
                )
            for window in WINDOWS:
                finite_rows.append(
                    {
                        "score_step": score_step,
                        "window": window,
                        "arm": "dense_baseline",
                        "loss": evaluate_model_loss(
                            model, windows[window], args.device
                        ),
                    }
                )
                for arm in ARMS:
                    updates = {
                        layer: (
                            states[(arm, layer)]
                            - model.transformer.h[
                                layer
                            ].mlp.c_proj.weight.detach().float()
                        ).cpu()
                        for layer in layers
                    }
                    finite_rows.append(
                        {
                            "score_step": score_step,
                            "window": window,
                            "arm": arm,
                            "loss": evaluate_with_updates(
                                model, windows[window], updates, args.device
                            ),
                        }
                    )
            del model
            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()
        elif score_step == 1 or score_step % 10 == 0:
            print(json.dumps({"step": score_step}), flush=True)
        del payload

    aggregate = aggregate_results(rows, finite_rows, plan["decision_rule"])
    args.output.mkdir(parents=True)
    cells_path = args.output / "parallel_transport_cells.csv"
    finite_path = args.output / "parallel_transport_finite_ce.csv"
    result_path = args.output / "parallel_transport_result.json"
    write_csv(cells_path, rows)
    write_csv(finite_path, finite_rows)
    result = {
        "schema_version": SCHEMA_VERSION,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "classification": aggregate["classification"],
        "execution": {
            "host": "PRO6",
            "device": args.device,
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": "examples.nanogpt.analyze_mlp_cproj_parallel_transport_error_feedback",
            "parameter_updates": 0,
            "direct_foreground_polling": True,
            "watchdog": False,
            "callback": False,
            "started_at": started_at,
            "elapsed_seconds": time.time() - started,
        },
        "identity": {
            "plan_path": str(args.plan),
            "plan_sha256": file_sha256(args.plan),
            "dataset_manifest_sha256": file_sha256(manifest),
            "run_identity_sha256": run_identity,
            "first_snapshot_sha256": file_sha256(paths[0]),
            "last_snapshot_sha256": file_sha256(paths[-1]),
            "terminal_checkpoint_sha256": file_sha256(
                args.terminal_checkpoint
            ),
        },
        "aggregate": aggregate,
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    result["artifacts"] = {
        "cells_sha256": file_sha256(cells_path),
        "finite_ce_sha256": file_sha256(finite_path),
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "classification": aggregate["classification"],
                "selected_arm": aggregate["selected_arm"],
                "output": str(result_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
