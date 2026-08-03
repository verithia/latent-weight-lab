#!/usr/bin/env python3
"""Derive and validate a true 2x2 block-output metric for MLP steps.

One production gradient supplies exact c_fc and c_proj update directions and
their task-gradient dot products.  On disjoint metric windows, their finite
block-output deltas define a normalized 2x2 Gram matrix.  Full and diagonal
damped metric solves are normalized to the production materialized-update
budget, then scored on untouched held-out windows without CE-based tuning.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_mlp_cfc_exact_current_matcher import (
    file_sha256,
    fixed_batches,
    git_commit,
    load_model_and_optimizer,
)
from examples.nanogpt.analyze_mlp_joint_prospective_step import (
    assert_joint_matches_singletons,
    extract_production_updates,
    forward_capture,
)
from examples.nanogpt.analyze_mlp_joint_step_response_surface import (
    ScaledUpdateApplier,
    evaluate_points,
    paired_comparison,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "nanogpt_mlp_joint_block_output_metric_v1"


def _accumulate_gram(
    gram: torch.Tensor,
    base: torch.Tensor,
    cfc: torch.Tensor,
    cproj: torch.Tensor,
) -> None:
    delta_fc = (cfc - base).double()
    delta_proj = (cproj - base).double()
    denominator = max(float(base.double().square().sum()), 1e-30)
    gram[0, 0] += delta_fc.square().sum() / denominator
    gram[0, 1] += (delta_fc * delta_proj).sum() / denominator
    gram[1, 0] = gram[0, 1]
    gram[1, 1] += delta_proj.square().sum() / denominator


@torch.no_grad()
def estimate_block_output_metric(
    model,
    applier: ScaledUpdateApplier,
    batches_by_window: dict[str, list[torch.Tensor]],
    probe_layers: list[int],
    *,
    device: str,
    dtype: torch.dtype,
) -> dict[str, Any]:
    per_window: dict[str, Any] = {}
    model.eval()
    model.prepare_block_fht_cache(dtype=dtype)
    try:
        for window, batches in batches_by_window.items():
            gram = torch.zeros((2, 2), dtype=torch.float64, device=device)
            contributions = 0
            for tokens in batches:
                _base_loss, base_values = forward_capture(
                    model, tokens, probe_layers, device=device, dtype=dtype
                )
                with applier.apply(1.0, 0.0):
                    _cfc_loss, cfc_values = forward_capture(
                        model, tokens, probe_layers, device=device, dtype=dtype
                    )
                with applier.apply(0.0, 1.0):
                    _cproj_loss, cproj_values = forward_capture(
                        model, tokens, probe_layers, device=device, dtype=dtype
                    )
                for layer in probe_layers:
                    _accumulate_gram(
                        gram,
                        base_values[(layer, "block")],
                        cfc_values[(layer, "block")],
                        cproj_values[(layer, "block")],
                    )
                    contributions += 1
            gram /= contributions
            correlation = float(
                gram[0, 1]
                / max(float((gram[0, 0] * gram[1, 1]).sqrt()), 1e-30)
            )
            per_window[window] = {
                "gram": [[float(value) for value in row] for row in gram],
                "correlation": correlation,
                "contributions": contributions,
            }
    finally:
        applier.restore()
        model.flush_block_fht_cache()
    matrices = [
        torch.tensor(value["gram"], dtype=torch.float64)
        for value in per_window.values()
    ]
    mean = sum(matrices, torch.zeros((2, 2), dtype=torch.float64)) / len(
        matrices
    )
    return {
        "mean_gram": [[float(value) for value in row] for row in mean],
        "mean_correlation": float(
            mean[0, 1]
            / max(float((mean[0, 0] * mean[1, 1]).sqrt()), 1e-30)
        ),
        "per_window": per_window,
    }


def damping_for_condition(gram: torch.Tensor, target: float) -> float:
    if target <= 1.0:
        raise ValueError("target condition must exceed one")
    eigenvalues = torch.linalg.eigvalsh(gram)
    minimum = float(eigenvalues[0])
    maximum = float(eigenvalues[-1])
    epsilon = max(maximum, 1.0) * 1e-12
    damping = max(0.0, (maximum - target * minimum) / (target - 1.0))
    return max(damping, epsilon - minimum)


def normalize_coefficients(
    coefficients: torch.Tensor,
    cfc_norm: float,
    cproj_norm: float,
) -> torch.Tensor:
    radius = math.sqrt(cfc_norm * cfc_norm + cproj_norm * cproj_norm)
    realized = math.sqrt(
        (float(coefficients[0]) * cfc_norm) ** 2
        + (float(coefficients[1]) * cproj_norm) ** 2
    )
    if realized <= 0.0 or not math.isfinite(realized):
        raise ValueError("metric solve produced a zero or nonfinite direction")
    return coefficients * (radius / realized)


def solve_metric_coefficients(
    gram_values: list[list[float]],
    gradient_update_dot: list[float],
    cfc_norm: float,
    cproj_norm: float,
    condition_target: float,
) -> dict[str, Any]:
    gram = torch.tensor(gram_values, dtype=torch.float64)
    gradient = torch.tensor(gradient_update_dot, dtype=torch.float64)
    damping = damping_for_condition(gram, condition_target)
    regularized = gram + damping * torch.eye(2, dtype=torch.float64)
    full_raw = torch.linalg.solve(regularized, -gradient)
    diagonal_raw = -gradient / regularized.diag()
    full = normalize_coefficients(full_raw, cfc_norm, cproj_norm)
    diagonal = normalize_coefficients(diagonal_raw, cfc_norm, cproj_norm)
    return {
        "damping": damping,
        "condition_target": condition_target,
        "regularized_condition": float(torch.linalg.cond(regularized)),
        "gradient_update_dot": [float(value) for value in gradient],
        "full_raw": [float(value) for value in full_raw],
        "diagonal_raw": [float(value) for value in diagonal_raw],
        "full_constant_budget": [float(value) for value in full],
        "diagonal_constant_budget": [float(value) for value in diagonal],
        "production_constant_budget": [1.0, 1.0],
    }


def decide_metric(
    validation_rows: list[dict[str, Any]],
    points: dict[str, dict[str, Any]],
    confidence_z: float,
) -> dict[str, Any]:
    production = str(points["production"]["point_id"])
    full = str(points["full_metric"]["point_id"])
    diagonal = str(points["diagonal_metric"]["point_id"])
    full_vs_production = paired_comparison(
        validation_rows, full, production, confidence_z
    )
    diagonal_vs_production = paired_comparison(
        validation_rows, diagonal, production, confidence_z
    )
    full_vs_diagonal = paired_comparison(
        validation_rows, full, diagonal, confidence_z
    )
    if (
        full_vs_production["candidate_reliably_better"]
        and full_vs_diagonal["candidate_reliably_better"]
    ):
        classification = "FULL_2X2_BLOCK_OUTPUT_METRIC_SUPPORTED"
        next_action = "IMPLEMENT_AND_PERFORMANCE_GATE_COUPLED_2X2_PRECONDITIONER"
    elif diagonal_vs_production["candidate_reliably_better"]:
        classification = "DIAGONAL_BLOCK_OUTPUT_PRECONDITIONING_ONLY"
        next_action = "IMPLEMENT_AND_GATE_DIAGONAL_OUTPUT_PRECONDITIONER"
    else:
        classification = "BLOCK_OUTPUT_METRIC_REJECTED"
        next_action = "RETURN_TO_REPRESENTATIONAL_CAPACITY_NOT_STEP_COORDINATION"
    return {
        "classification": classification,
        "next_action": next_action,
        "comparisons": {
            "full_vs_production": full_vs_production,
            "diagonal_vs_production": diagonal_vs_production,
            "full_vs_diagonal": full_vs_diagonal,
        },
    }


def validate_plan(
    plan_path: Path,
    checkpoint: Path,
    config_path: Path,
    data_dir: Path,
) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    actual = {
        "checkpoint_sha256": file_sha256(checkpoint),
        "config_sha256": file_sha256(config_path),
        "dataset_manifest_sha256": file_sha256(data_dir / "manifest.json"),
        "entrypoint_sha256": file_sha256(Path(__file__).resolve()),
    }
    for key, value in actual.items():
        if value != plan["identity"][key]:
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
    plan = validate_plan(args.plan, args.checkpoint, args.config, args.data_dir)
    protocol = plan["protocol"]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[str(config["dtype"])]
    train_batches = fixed_batches(
        args.data_dir,
        "train",
        batch_size=int(config["batch_size"]),
        block_size=int(config["block_size"]) + 1,
        batches=int(protocol["gradient_accumulation_steps"]),
        seed=int(protocol["train_seed"]),
    )
    updates, extracted = extract_production_updates(
        args.checkpoint, config, train_batches, device=args.device, dtype=dtype
    )
    exactness = assert_joint_matches_singletons(
        updates["cfc_only"]["c_fc"],
        updates["cproj_only"]["c_proj"],
        updates["joint"],
    )
    cfc = updates["cfc_only"]["c_fc"]
    cproj = updates["cproj_only"]["c_proj"]
    cfc_norm = float(extracted["variants"]["cfc_only"]["update_fro"]["c_fc"])
    cproj_norm = float(
        extracted["variants"]["cproj_only"]["update_fro"]["c_proj"]
    )
    gradient_dot = [
        float(
            extracted["variants"]["cfc_only"]["gradient_update_dot"]["c_fc"]
        ),
        float(
            extracted["variants"]["cproj_only"]["gradient_update_dot"][
                "c_proj"
            ]
        ),
    ]
    if not all(value < 0.0 for value in gradient_dot):
        raise ValueError("registered production directions are not both descending")
    model, _optimizer, checkpoint_payload = load_model_and_optimizer(
        args.checkpoint, config, args.device
    )
    applier = ScaledUpdateApplier(model, cfc, cproj)

    def windows(seeds: list[int], batches: int) -> dict[str, list[torch.Tensor]]:
        return {
            f"window_{index + 1}": fixed_batches(
                args.data_dir,
                "val",
                batch_size=int(protocol["evaluation_batch_size"]),
                block_size=int(protocol["evaluation_block_size"]) + 1,
                batches=batches,
                seed=seed,
            )
            for index, seed in enumerate(seeds)
        }

    metric = estimate_block_output_metric(
        model,
        applier,
        windows(
            [int(seed) for seed in protocol["metric_seeds"]],
            int(protocol["metric_batches_per_window"]),
        ),
        [int(layer) for layer in protocol["probe_layers"]],
        device=args.device,
        dtype=dtype,
    )
    solve = solve_metric_coefficients(
        metric["mean_gram"],
        gradient_dot,
        cfc_norm,
        cproj_norm,
        float(protocol["condition_target"]),
    )
    points = {
        "production": {
            "point_id": "production",
            "cfc_scale": 1.0,
            "cproj_scale": 1.0,
        },
        "full_metric": {
            "point_id": "full_metric",
            "cfc_scale": solve["full_constant_budget"][0],
            "cproj_scale": solve["full_constant_budget"][1],
        },
        "diagonal_metric": {
            "point_id": "diagonal_metric",
            "cfc_scale": solve["diagonal_constant_budget"][0],
            "cproj_scale": solve["diagonal_constant_budget"][1],
        },
    }
    validation_rows = evaluate_points(
        model,
        applier,
        windows(
            [int(seed) for seed in protocol["validation_seeds"]],
            int(protocol["validation_batches_per_window"]),
        ),
        list(points.values()),
        device=args.device,
        dtype=dtype,
    )
    decision = decide_metric(
        validation_rows,
        points,
        float(plan["decision_rule"]["confidence_z"]),
    )
    args.output.mkdir(parents=True, exist_ok=False)
    paths = {
        "metric": args.output / "block_output_metric.json",
        "solve": args.output / "metric_solve.json",
        "validation": args.output / "heldout_validation.json",
        "prospective_step_metadata": args.output / "prospective_step_metadata.json",
    }
    paths["metric"].write_text(json.dumps(metric, indent=2, sort_keys=True) + "\n")
    paths["solve"].write_text(
        json.dumps({"solve": solve, "points": points, "decision": decision}, indent=2, sort_keys=True) + "\n"
    )
    paths["validation"].write_text(
        json.dumps(validation_rows, indent=2, sort_keys=True) + "\n"
    )
    paths["prospective_step_metadata"].write_text(
        json.dumps(extracted, indent=2, sort_keys=True) + "\n"
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "metric": metric,
        "solve": solve,
        "points": points,
        "cfc_update_fro": cfc_norm,
        "cproj_update_fro": cproj_norm,
        "parameter_updates_to_checkpoint": 0,
        "disposable_optimizer_steps": 3,
        "checkpoint_next_iter": int(checkpoint_payload["next_iter"]),
        "joint_singleton_update_max_abs_error": exactness,
        "identity": {
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "config_sha256": file_sha256(args.config),
            "dataset_manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
            "plan_sha256": file_sha256(args.plan),
        },
        "outputs": {f"{name}_sha256": file_sha256(path) for name, path in paths.items()},
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
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
