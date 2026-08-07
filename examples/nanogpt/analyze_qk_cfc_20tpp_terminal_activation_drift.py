#!/usr/bin/env python3
"""Gauge-invariant terminal activation audit for QK-only versus QK+c_fc."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_mlp_cfc_exact_current_matcher import (
    file_sha256,
    git_commit,
)
from examples.nanogpt.analyze_residual_compatibility import (
    REGIMES,
    ResidualCollector,
    compatibility_metrics,
    effective_ranks,
    fixed_validation_batches,
    load_model,
    token_regimes_from_validation,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "mai_124m_qk_cfc_20tpp_terminal_activation_drift_v1"
BANDS = {"early": range(0, 4), "middle": range(4, 8), "late": range(8, 12)}
RATIO_METRICS = (
    "pregelu_hard_rank",
    "postgelu_hard_rank",
    "postgelu_to_pregelu_hard_rank",
    "update_to_residual_rms",
    "residual_update_parallel_energy",
)


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("geometric mean requires positive finite values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def compare_rows(
    rows: list[dict[str, Any]], parent: str, candidate: str
) -> list[dict[str, Any]]:
    index = {
        (str(row["run"]), int(row["layer"]), str(row["regime"])): row
        for row in rows
    }
    compared: list[dict[str, Any]] = []
    for layer in range(12):
        band = next(name for name, values in BANDS.items() if layer in values)
        for regime in REGIMES:
            left = index[(parent, layer, regime)]
            right = index[(candidate, layer, regime)]
            row: dict[str, Any] = {"layer": layer, "band": band, "regime": regime}
            for metric in RATIO_METRICS:
                denominator = float(left[metric])
                numerator = float(right[metric])
                if denominator <= 0 or numerator <= 0:
                    raise ValueError(f"nonpositive metric for ratio: {metric}")
                row[f"{metric}_candidate_to_parent"] = numerator / denominator
            row["residual_update_cos_delta"] = float(
                right["residual_update_cos_mean"]
            ) - float(left["residual_update_cos_mean"])
            row["residual_update_cka_delta"] = float(
                right["residual_update_cka"]
            ) - float(left["residual_update_cka"])
            compared.append(row)
    return compared


def aggregate_comparisons(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {"all": rows}
    groups.update(
        {name: [row for row in rows if row["band"] == name] for name in BANDS}
    )
    groups.update(
        {regime: [row for row in rows if row["regime"] == regime] for regime in REGIMES}
    )
    result: dict[str, Any] = {}
    for name, values in groups.items():
        item: dict[str, Any] = {"cells": len(values)}
        for metric in RATIO_METRICS:
            key = f"{metric}_candidate_to_parent"
            item[key] = geometric_mean([float(row[key]) for row in values])
        for metric in ("residual_update_cos_delta", "residual_update_cka_delta"):
            observed = [float(row[metric]) for row in values]
            item[f"{metric}_mean"] = sum(observed) / len(observed)
            item[f"{metric}_mean_abs"] = sum(abs(value) for value in observed) / len(observed)
        result[name] = item
    return result


def classify(
    aggregate: dict[str, Any],
    ce: dict[str, float],
    *,
    parent: str,
    candidate: str,
    minimum_material_ratio: float,
    minimum_cosine_shift: float,
) -> dict[str, Any]:
    overall = aggregate["all"]
    pre = float(overall["pregelu_hard_rank_candidate_to_parent"])
    conversion = float(
        overall["postgelu_to_pregelu_hard_rank_candidate_to_parent"]
    )
    residual_scale = float(overall["update_to_residual_rms_candidate_to_parent"])
    cosine_shift = float(overall["residual_update_cos_delta_mean_abs"])
    lower = 1.0 - minimum_material_ratio
    upper = 1.0 + minimum_material_ratio
    if conversion <= lower and pre >= lower:
        classification = "POST_GELU_CONVERSION_DEFICIT"
    elif pre <= lower:
        classification = "PRE_GELU_CAPACITY_DEFICIT"
    elif residual_scale < lower or residual_scale > upper:
        classification = "RESIDUAL_BRANCH_SCALE_DEFICIT"
    elif cosine_shift >= minimum_cosine_shift:
        classification = "RESIDUAL_DIRECTION_COMPATIBILITY_DEFICIT"
    else:
        classification = "DISTRIBUTED_OR_UNRESOLVED_TERMINAL_DRIFT"
    return {
        "classification": classification,
        "probe_ce": ce,
        "candidate_minus_parent_probe_ce": float(ce[candidate] - ce[parent]),
        "probe_reproduces_terminal_order": ce[candidate] > ce[parent],
        "minimum_material_ratio": minimum_material_ratio,
        "minimum_cosine_shift": minimum_cosine_shift,
        "training_or_structure_authorized": False,
        "interpretation_boundary": (
            "Terminal functional localization only; cross-run tensor directions and "
            "parameter transplantation are not interpreted."
        ),
    }


def collect_checkpoint(
    name: str,
    checkpoint_path: Path,
    batches: list[torch.Tensor],
    *,
    data_dir: Path,
    device: str,
    dtype: torch.dtype,
    layers: list[int],
    sample_cap: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model = load_model(checkpoint_path, device)
    if model.config.block_fht:
        model.prepare_block_fht_cache(dtype=dtype)
    regimes = token_regimes_from_validation(data_dir, model.config.vocab_size)
    collector = ResidualCollector(model, regimes, layers, sample_cap)
    losses: list[float] = []
    context = (
        torch.amp.autocast("cuda", dtype=dtype)
        if device.startswith("cuda")
        else nullcontext()
    )
    try:
        with torch.no_grad():
            for batch in batches:
                inputs = batch[:, :-1].contiguous().to(device)
                targets = batch[:, 1:].contiguous().to(device)
                collector.set_tokens(inputs)
                with context:
                    _logits, loss = model(inputs, targets)
                if loss is None or not torch.isfinite(loss):
                    raise RuntimeError("non-finite or absent probe loss")
                losses.append(float(loss))
    finally:
        collector.close()

    rows: list[dict[str, Any]] = []
    for layer in layers:
        for regime in REGIMES:
            values = {
                point: collector.values(layer, point, regime)
                for point in (
                    "residual_in",
                    "ln2",
                    "pre_gelu",
                    "post_gelu",
                    "mlp_out",
                    "residual_out",
                )
            }
            if any(value is None for value in values.values()):
                raise RuntimeError(f"incomplete activation capture: layer={layer} regime={regime}")
            residual, update, output, pre, post, ln2 = (
                values[key].to(device)  # type: ignore[union-attr]
                for key in (
                    "residual_in",
                    "mlp_out",
                    "residual_out",
                    "pre_gelu",
                    "post_gelu",
                    "ln2",
                )
            )
            metrics = compatibility_metrics(residual, update, output)
            pre_soft, pre_hard = effective_ranks(pre)
            post_soft, post_hard = effective_ranks(post)
            ln2_soft, ln2_hard = effective_ranks(ln2)
            rows.append(
                {
                    "run": name,
                    "checkpoint": str(checkpoint_path),
                    "layer": layer,
                    "regime": regime,
                    **metrics,
                    "ln2_soft_rank": ln2_soft,
                    "ln2_hard_rank": ln2_hard,
                    "pregelu_soft_rank": pre_soft,
                    "pregelu_hard_rank": pre_hard,
                    "postgelu_soft_rank": post_soft,
                    "postgelu_hard_rank": post_hard,
                    "postgelu_to_pregelu_hard_rank": post_hard / max(pre_hard, 1e-30),
                }
            )
    model.flush_block_fht_cache()
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return rows, {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "mean_probe_ce": sum(losses) / len(losses),
        "batch_probe_ce": losses,
    }


def validate_plan(args: argparse.Namespace) -> dict[str, Any]:
    plan = json.loads(args.plan.read_text())
    observed = {
        "entrypoint_sha256": file_sha256(Path(__file__)),
        "parent_checkpoint_sha256": file_sha256(args.parent_checkpoint),
        "candidate_checkpoint_sha256": file_sha256(args.candidate_checkpoint),
        "parent_config_sha256": file_sha256(args.parent_config),
        "candidate_config_sha256": file_sha256(args.candidate_config),
        "dataset_manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
    }
    if observed != plan["identity"]:
        raise ValueError(f"plan identity mismatch: observed={observed}")
    return plan


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--parent-checkpoint", required=True, type=Path)
    parser.add_argument("--candidate-checkpoint", required=True, type=Path)
    parser.add_argument("--parent-config", required=True, type=Path)
    parser.add_argument("--candidate-config", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    started = time.time()
    plan = validate_plan(args)
    protocol = plan["protocol"]
    dtype = getattr(torch, str(protocol["dtype"]))
    batches = fixed_validation_batches(
        args.data_dir,
        int(protocol["batch_size"]),
        int(protocol["block_size"]) + 1,
        int(protocol["batches"]),
        int(protocol["sample_seed"]),
    )
    rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    for name, checkpoint in (
        (str(plan["names"]["parent"]), args.parent_checkpoint),
        (str(plan["names"]["candidate"]), args.candidate_checkpoint),
    ):
        current, observed = collect_checkpoint(
            name,
            checkpoint,
            batches,
            data_dir=args.data_dir,
            device=args.device,
            dtype=dtype,
            layers=list(range(12)),
            sample_cap=int(protocol["sample_cap"]),
        )
        rows.extend(current)
        metadata[name] = observed
    comparison = compare_rows(
        rows, str(plan["names"]["parent"]), str(plan["names"]["candidate"])
    )
    aggregate = aggregate_comparisons(comparison)
    ce = {name: float(value["mean_probe_ce"]) for name, value in metadata.items()}
    decision = classify(
        aggregate,
        ce,
        parent=str(plan["names"]["parent"]),
        candidate=str(plan["names"]["candidate"]),
        minimum_material_ratio=float(plan["decision_rule"]["minimum_material_ratio"]),
        minimum_cosine_shift=float(plan["decision_rule"]["minimum_cosine_shift"]),
    )
    args.output.mkdir(parents=True)
    rows_path = args.output / "activation_rows.csv"
    comparison_path = args.output / "comparison_rows.json"
    write_csv(rows_path, rows)
    comparison_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    result = {
        "schema_version": SCHEMA_VERSION,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "parameter_updates": 0,
        "decision": decision,
        "aggregate": aggregate,
        "checkpoints": metadata,
        "identity": {
            **plan["identity"],
            "plan_sha256": file_sha256(args.plan),
            "activation_rows_sha256": file_sha256(rows_path),
            "comparison_rows_sha256": file_sha256(comparison_path),
        },
        "execution": {
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": str(Path(__file__).resolve()),
            "command": sys.argv,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "device": args.device,
            "direct_foreground_polling": True,
            "watchdog": False,
            "callback": False,
        },
    }
    result_path = args.output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
