#!/usr/bin/env python3
"""Compare one-, two-, and three-stage directed c_fc mixers at fixed budgets."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_mlp_cfc_directed_sparse import fit_directed_sparse_mixer
from examples.nanogpt.analyze_mlp_cfc_exact_current_matcher import (
    file_sha256,
    fixed_batches,
    git_commit,
    load_model_and_optimizer,
)
from examples.nanogpt.analyze_mlp_cfc_product_directed import (
    fit_product_directed_sparse_mixer,
)
from examples.nanogpt.analyze_mlp_dense_oracle_gap import (
    ExactVariantApplier,
    aggregate_direction_metrics,
    evaluate_candidates,
    family_fro,
    merge_updates,
    scale_family,
)
from examples.nanogpt.analyze_mlp_fixed_radius_capacity import (
    extract_reconstructed_capacity_updates,
    normalize_family_to_radius,
    quantized_update,
)
from examples.nanogpt.analyze_mlp_joint_step_response_surface import paired_comparison


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "nanogpt_mlp_cfc_multistage_directed_v1"


@torch.no_grad()
def fit_multistage_directed_sparse_mixer(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    incoming_schedule: list[int],
    ridge_ratio: float,
    chunk_size: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Greedily fit a product of sparse residual mixers to a target update."""
    if len(incoming_schedule) < 2 or any(value <= 0 for value in incoming_schedule):
        raise ValueError("incoming_schedule must contain at least two positive stages")
    transformed = source.float().clone()
    prediction = torch.zeros_like(target, dtype=torch.float32)
    stage_rows = []
    for stage_index, incoming in enumerate(incoming_schedule):
        remaining = target.float() - prediction
        stage_update, row = fit_directed_sparse_mixer(
            transformed,
            remaining,
            incoming=int(incoming),
            ridge_ratio=ridge_ratio,
            chunk_size=chunk_size,
        )
        transformed.add_(stage_update)
        prediction.add_(stage_update)
        row = dict(row)
        row["stage_index"] = int(stage_index)
        row["stage_update_fro"] = float(stage_update.norm())
        stage_rows.append(row)
    residual = target.float() - prediction
    target_energy = target.float().square().sum().clamp_min(1e-30)
    return prediction, {
        "stages": len(incoming_schedule),
        "incoming_schedule": [int(value) for value in incoming_schedule],
        "incoming_total_per_target": int(sum(incoming_schedule)),
        "coordinates": int(sum(incoming_schedule) * source.shape[1]),
        "target_recovery": float(1.0 - residual.square().sum() / target_energy),
        "target_cosine": float(
            (prediction * target.float()).sum()
            / (prediction.norm() * target.float().norm()).clamp_min(1e-30)
        ),
        "endpoint_update_fro": float(prediction.norm()),
        "stage_rows": stage_rows,
    }


def candidate_order(totals: list[int]) -> list[str]:
    names = [
        "baseline",
        "production_cfc",
        "production_cproj",
        "production_joint",
        "dense_norm_cfc",
        "hybrid_norm_cfc",
        "directed44_cfc",
        "hybrid_directed44_cfc",
        "directed88_cfc",
        "hybrid_directed88_cfc",
        "product22x2_cfc",
        "hybrid_product22x2_cfc",
        "product44x2_cfc",
        "hybrid_product44x2_cfc",
    ]
    for total in totals:
        names.extend(
            (f"product{total}totalx3_cfc", f"hybrid_product{total}totalx3_cfc")
        )
    return names


def classify(
    rows: list[dict[str, Any]],
    totals: list[int],
    *,
    confidence_z: float,
    minimum_fraction: float,
    mean_fraction: float,
) -> dict[str, Any]:
    names = candidate_order(totals)
    means = {
        point: sum(float(row["ce"]) for row in rows if row["point_id"] == point)
        / sum(1 for row in rows if row["point_id"] == point)
        for point in names
    }
    comparisons = {
        "dense_single": paired_comparison(rows, "dense_norm_cfc", "production_cfc", confidence_z),
        "dense_hybrid": paired_comparison(rows, "hybrid_norm_cfc", "production_joint", confidence_z),
    }
    oracle_valid = all(
        comparisons[name]["candidate_reliably_better"]
        for name in ("dense_single", "dense_hybrid")
    )
    for point, reference in (
        ("directed44_cfc", "production_cfc"),
        ("hybrid_directed44_cfc", "production_joint"),
        ("directed88_cfc", "production_cfc"),
        ("hybrid_directed88_cfc", "production_joint"),
        ("product22x2_cfc", "production_cfc"),
        ("hybrid_product22x2_cfc", "production_joint"),
        ("product44x2_cfc", "production_cfc"),
        ("hybrid_product44x2_cfc", "production_joint"),
    ):
        comparisons[point] = paired_comparison(rows, point, reference, confidence_z)
    results = []
    for total in totals:
        single_id = f"product{total}totalx3_cfc"
        hybrid_id = f"hybrid_product{total}totalx3_cfc"
        single_key = f"product{total}totalx3_single"
        hybrid_key = f"product{total}totalx3_hybrid"
        comparisons[single_key] = paired_comparison(
            rows, single_id, "production_cfc", confidence_z
        )
        comparisons[hybrid_key] = paired_comparison(
            rows, hybrid_id, "production_joint", confidence_z
        )
        single_gap = means["production_cfc"] - means["dense_norm_cfc"]
        hybrid_gap = means["production_joint"] - means["hybrid_norm_cfc"]
        fractions = {
            "single": (
                (means["production_cfc"] - means[single_id]) / single_gap
                if single_gap > 0.0 else math.nan
            ),
            "hybrid": (
                (means["production_joint"] - means[hybrid_id]) / hybrid_gap
                if hybrid_gap > 0.0 else math.nan
            ),
        }
        finite = all(math.isfinite(value) for value in fractions.values())
        reliable = all(
            comparisons[key]["candidate_reliably_better"]
            for key in (single_key, hybrid_key)
        )
        fraction_pass = finite and (
            min(fractions.values()) >= float(minimum_fraction)
            and sum(fractions.values()) / 2.0 >= float(mean_fraction)
        )
        results.append(
            {
                "incoming_total_per_target": int(total),
                "coordinates_per_layer": int(total * 3072),
                "reliable_singleton_and_hybrid": reliable,
                "oracle_gap_fraction_recovered": {
                    key: value if math.isfinite(value) else None
                    for key, value in fractions.items()
                },
                "fraction_pass": fraction_pass,
                "passes": oracle_valid and reliable and fraction_pass,
            }
        )
    selected = next((row for row in results if row["passes"]), None)
    if not oracle_valid:
        classification = "HELDOUT_DENSE_CFC_ORACLE_NOT_STABLE"
        next_action = "DO_NOT_TRAIN_RESELECT_DISCRIMINATING_WINDOWS"
    elif selected is not None:
        classification = "THREE_STAGE_DIRECTED_CFC_PASSES"
        next_action = "IMPLEMENT_SMALLEST_PASSING_THREE_STAGE_PRODUCT_AND_PREFLIGHT_ONLY"
    elif any(row["reliable_singleton_and_hybrid"] for row in results):
        classification = "THREE_STAGE_DIRECTED_CFC_GAIN_TOO_SMALL"
        next_action = "DO_NOT_TRAIN_REASSESS_GENERATOR_STRUCTURE"
    else:
        classification = "THREE_STAGE_DIRECTED_CFC_REJECTED"
        next_action = "DO_NOT_TRAIN_REASSESS_GENERATOR_STRUCTURE"
    return {
        "classification": classification,
        "selected_incoming_total_per_target": (
            None if selected is None else selected["incoming_total_per_target"]
        ),
        "next_action": next_action,
        "candidate_means": means,
        "comparisons": comparisons,
        "levels": results,
        "dense_cfc_oracle_valid": oracle_valid,
    }


def validate_plan(path: Path, checkpoint: Path, config: Path, data_dir: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    actual = {
        "checkpoint_sha256": file_sha256(checkpoint),
        "config_sha256": file_sha256(config),
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
    bases, dense_historical, reconstructed = extract_reconstructed_capacity_updates(
        args.checkpoint, config, train_batches,
        [int(protocol["production_residual_stages"])],
        device=args.device, dtype=dtype,
    )
    prod_cfc = reconstructed["production"]["c_fc"]
    prod_cproj = reconstructed["production"]["c_proj"]
    control_level = str(protocol["production_residual_stages"])
    control_cfc = {
        layer: quantized_update(bases["c_fc"][layer], update)
        for layer, update in reconstructed["raw"][control_level]["c_fc"].items()
    }
    reconstruction_error = max(
        float((prod_cfc[layer] - control_cfc[layer]).abs().max()) for layer in prod_cfc
    )
    if reconstruction_error > float(protocol["control_max_abs_tolerance"]):
        raise RuntimeError(f"production c_fc reconstruction failed: {reconstruction_error}")
    schedules = {
        int(total): [int(value) for value in schedule]
        for total, schedule in protocol["three_stage_schedules"].items()
    }
    totals = sorted(schedules)
    if any(len(schedule) != 3 or sum(schedule) != total for total, schedule in schedules.items()):
        raise ValueError("three-stage schedules must contain three entries summing to the key")
    raw: dict[str, dict[str, dict[int, torch.Tensor]]] = {
        "directed": {str(level): {} for level in (44, 88)},
        "two_stage": {str(total): {} for total in totals},
        "three_stage": {str(total): {} for total in totals},
    }
    fits: dict[str, dict[str, dict[int, Any]]] = {
        family: {str(total): {} for total in totals}
        for family in ("two_stage", "three_stage")
    }
    fits["directed"] = {str(level): {} for level in (44, 88)}
    for layer in sorted(bases["c_fc"]):
        source = bases["c_fc"][layer].to(args.device).float().T.contiguous()
        target = dense_historical["c_fc"][layer].to(args.device).float().T.contiguous()
        for level in (44, 88):
            predicted, row = fit_directed_sparse_mixer(
                source, target, incoming=level,
                ridge_ratio=float(protocol["ridge_ratio"]),
                chunk_size=int(protocol["solver_chunk_size"]),
            )
            raw["directed"][str(level)][layer] = predicted.T.contiguous().cpu()
            fits["directed"][str(level)][layer] = row
        for total in totals:
            half = total // 2
            predicted, row = fit_product_directed_sparse_mixer(
                source, target, incoming_per_stage=half,
                ridge_ratio=float(protocol["ridge_ratio"]),
                chunk_size=int(protocol["solver_chunk_size"]),
            )
            raw["two_stage"][str(total)][layer] = predicted.T.contiguous().cpu()
            fits["two_stage"][str(total)][layer] = row
            predicted, row = fit_multistage_directed_sparse_mixer(
                source, target, incoming_schedule=schedules[total],
                ridge_ratio=float(protocol["ridge_ratio"]),
                chunk_size=int(protocol["solver_chunk_size"]),
            )
            raw["three_stage"][str(total)][layer] = predicted.T.contiguous().cpu()
            fits["three_stage"][str(total)][layer] = row
        print(json.dumps({"multistage_directed_layer_complete": layer}), flush=True)
    cfc_radius = family_fro(prod_cfc)
    normalized: dict[str, dict[str, dict[int, torch.Tensor]]] = {
        family: {} for family in raw
    }
    normalization: dict[str, Any] = {family: {} for family in raw}
    for family, levels in raw.items():
        for level, updates in levels.items():
            candidate, row = normalize_family_to_radius(bases["c_fc"], updates, cfc_radius)
            if row["relative_radius_error"] > float(protocol["maximum_relative_radius_error"]):
                raise RuntimeError(f"{family}{level} radius normalization failed")
            normalized[family][level] = candidate
            normalization[family][level] = row
    norm_dense_cfc = scale_family(
        dense_historical["c_fc"], cfc_radius / family_fro(dense_historical["c_fc"])
    )
    candidates: dict[str, dict[str, dict[int, torch.Tensor]]] = {
        "baseline": {},
        "production_cfc": {"c_fc": prod_cfc},
        "production_cproj": {"c_proj": prod_cproj},
        "production_joint": merge_updates(prod_cfc, prod_cproj),
        "dense_norm_cfc": {"c_fc": norm_dense_cfc},
        "hybrid_norm_cfc": merge_updates(norm_dense_cfc, prod_cproj),
        "directed44_cfc": {"c_fc": normalized["directed"]["44"]},
        "hybrid_directed44_cfc": merge_updates(normalized["directed"]["44"], prod_cproj),
        "directed88_cfc": {"c_fc": normalized["directed"]["88"]},
        "hybrid_directed88_cfc": merge_updates(normalized["directed"]["88"], prod_cproj),
        "product22x2_cfc": {"c_fc": normalized["two_stage"]["44"]},
        "hybrid_product22x2_cfc": merge_updates(normalized["two_stage"]["44"], prod_cproj),
        "product44x2_cfc": {"c_fc": normalized["two_stage"]["88"]},
        "hybrid_product44x2_cfc": merge_updates(normalized["two_stage"]["88"], prod_cproj),
    }
    for total in totals:
        candidate = normalized["three_stage"][str(total)]
        candidates[f"product{total}totalx3_cfc"] = {"c_fc": candidate}
        candidates[f"hybrid_product{total}totalx3_cfc"] = merge_updates(candidate, prod_cproj)
    if list(candidates) != candidate_order(totals):
        raise RuntimeError("candidate order differs from registration")
    model, _optimizer, checkpoint_payload = load_model_and_optimizer(
        args.checkpoint, config, args.device
    )
    windows = {
        f"window_{index + 1}": fixed_batches(
            args.data_dir, "val",
            batch_size=int(protocol["evaluation_batch_size"]),
            block_size=int(protocol["evaluation_block_size"]) + 1,
            batches=int(protocol["validation_batches_per_window"]), seed=int(seed),
        )
        for index, seed in enumerate(protocol["validation_seeds"])
    }
    ce_rows = evaluate_candidates(
        model, ExactVariantApplier(model), windows, candidates,
        device=args.device, dtype=dtype,
    )
    rule = plan["decision_rule"]
    decision = classify(
        ce_rows, totals, confidence_z=float(rule["confidence_z"]),
        minimum_fraction=float(rule["minimum_oracle_gap_fraction"]),
        mean_fraction=float(rule["mean_oracle_gap_fraction"]),
    )
    direction_recovery = {
        family: {
            level: aggregate_direction_metrics(norm_dense_cfc, candidate)
            for level, candidate in levels.items()
        }
        for family, levels in normalized.items()
    }
    args.output.mkdir(parents=True, exist_ok=False)
    paths = {
        "ce": args.output / "heldout_ce.json",
        "fits": args.output / "multistage_directed_fits.json",
        "replay": args.output / "prospective_step_metadata.json",
    }
    paths["ce"].write_text(json.dumps(ce_rows, indent=2, sort_keys=True) + "\n")
    paths["fits"].write_text(json.dumps(fits, indent=2, sort_keys=True) + "\n")
    replay = {
        "checkpoint_next_iter": int(checkpoint_payload["next_iter"]),
        "production_cfc_reconstruction_max_abs_error": reconstruction_error,
        "normalization": normalization,
        "direction_recovery_against_norm_dense": direction_recovery,
        "gradient_replay": reconstructed["metadata"],
    }
    paths["replay"].write_text(json.dumps(replay, indent=2, sort_keys=True) + "\n")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "parameter_updates_to_checkpoint": 0,
        "disposable_optimizer_steps": 2,
        "identity": {
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "config_sha256": file_sha256(args.config),
            "dataset_manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
            "plan_sha256": file_sha256(args.plan),
        },
        "replay": replay,
        "outputs": {f"{name}_sha256": file_sha256(path) for name, path in paths.items()},
        "execution": {
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": str(Path(__file__).resolve()),
            "entrypoint_sha256": file_sha256(Path(__file__).resolve()),
            "command": sys.argv, "device": args.device,
            "started_at_unix": started, "finished_at_unix": time.time(),
            "direct_foreground_polling": True,
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
