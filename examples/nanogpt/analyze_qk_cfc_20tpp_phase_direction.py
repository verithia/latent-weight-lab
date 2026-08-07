#!/usr/bin/env python3
"""Locate QK+c_fc late drift in same-run phase and task geometry.

The acquisition snapshots are one exact replay of a rejected 20TPP run.  This
zero-update analysis uses only same-run, same-gauge comparisons.  It combines:

* fixed-token residual-write and activation geometry at every saved phase;
* descriptive raw c_fc/c_proj path geometry (never called manifold dimension);
* a 2x2 c_fc/c_proj phase recombination in each phase's otherwise-current
  model, measured directly in validation CE.

The factorial intervention is the task-metric counterpart to trajectory PCA:
it distinguishes an intrinsic c_fc path problem from harmful co-adaptation
with the dense c_proj writer.  No outcome authorizes training by itself.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_attention_functional_manifold import trajectory_metrics
from examples.nanogpt.analyze_mlp_activation_update_alignment import (
    load_snapshot,
    model_from_snapshot,
)
from examples.nanogpt.analyze_mlp_cproj_activation_weighted_output_selector import (
    file_sha256,
    git_commit,
)
from examples.nanogpt.analyze_residual_compatibility import (
    REGIMES,
    ResidualCollector,
    compatibility_metrics,
    effective_ranks,
    fixed_validation_batches,
    token_regimes_from_validation,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "mai_124m_qk_cfc_20tpp_phase_direction_v1"
BANDS = {"early": range(0, 4), "middle": range(4, 8), "late": range(8, 12)}
RATIO_METRICS = (
    "pregelu_hard_rank",
    "postgelu_hard_rank",
    "postgelu_to_pregelu_hard_rank",
    "update_to_residual_rms",
    "residual_update_parallel_energy",
)
DELTA_METRICS = (
    "residual_update_cos_mean",
    "residual_update_cka",
)
TRAJECTORY_SUMMARY_KEYS = (
    "pc1_energy",
    "pc1_pc2_energy",
    "participation_dimension",
    "path_length_over_chord",
    "median_relative_terminal_ray_residual",
    "mean_terminal_ray_recovery",
    "minimum_terminal_ray_recovery",
    "mean_consecutive_increment_cosine",
    "median_turn_degrees",
    "maximum_turn_degrees",
    "monotone_terminal_progress_fraction",
)


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("geometric mean requires positive finite values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def compare_phase_rows(
    rows: list[dict[str, Any]], reference_step: int
) -> list[dict[str, Any]]:
    index = {
        (int(row["step"]), int(row["layer"]), str(row["regime"])): row
        for row in rows
    }
    steps = sorted({key[0] for key in index})
    if reference_step not in steps:
        raise ValueError("reference step is absent")
    compared: list[dict[str, Any]] = []
    for step in steps:
        if step <= reference_step:
            continue
        for layer in range(12):
            band = next(name for name, values in BANDS.items() if layer in values)
            for regime in REGIMES:
                reference = index[(reference_step, layer, regime)]
                current = index[(step, layer, regime)]
                row: dict[str, Any] = {
                    "step": step,
                    "reference_step": reference_step,
                    "layer": layer,
                    "band": band,
                    "regime": regime,
                }
                for metric in RATIO_METRICS:
                    denominator = float(reference[metric])
                    numerator = float(current[metric])
                    if denominator <= 0 or numerator <= 0:
                        raise ValueError(f"nonpositive phase metric: {metric}")
                    row[f"{metric}_current_to_reference"] = numerator / denominator
                for metric in DELTA_METRICS:
                    row[f"{metric}_delta"] = float(current[metric]) - float(
                        reference[metric]
                    )
                compared.append(row)
    return compared


def aggregate_phase_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for step in sorted({int(row["step"]) for row in rows}):
        selected_step = [row for row in rows if int(row["step"]) == step]
        groups: dict[str, list[dict[str, Any]]] = {"all": selected_step}
        groups.update(
            {
                name: [row for row in selected_step if row["band"] == name]
                for name in BANDS
            }
        )
        groups.update(
            {
                regime: [row for row in selected_step if row["regime"] == regime]
                for regime in REGIMES
            }
        )
        phase: dict[str, Any] = {}
        for name, values in groups.items():
            item: dict[str, Any] = {"cells": len(values)}
            for metric in RATIO_METRICS:
                key = f"{metric}_current_to_reference"
                item[key] = geometric_mean([float(row[key]) for row in values])
            for metric in DELTA_METRICS:
                key = f"{metric}_delta"
                observed = [float(row[key]) for row in values]
                item[f"{key}_mean"] = sum(observed) / len(observed)
                item[f"{key}_mean_abs"] = sum(abs(value) for value in observed) / len(
                    observed
                )
            phase[name] = item
        output[str(step)] = phase
    return output


def factorial_effects(
    *,
    step: int,
    reference_pair_current_context_ce: float,
    current_cfc_reference_cproj_ce: float,
    reference_cfc_current_cproj_ce: float,
    native_current_ce: float,
) -> dict[str, float | int]:
    a = float(reference_pair_current_context_ce)
    b = float(current_cfc_reference_cproj_ce)
    c = float(reference_cfc_current_cproj_ce)
    d = float(native_current_ce)
    values = (a, b, c, d)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("factorial CE values must be finite")
    return {
        "step": step,
        "reference_pair_current_context_ce": a,
        "current_cfc_reference_cproj_ce": b,
        "reference_cfc_current_cproj_ce": c,
        "native_current_ce": d,
        "cfc_effect_with_reference_cproj_ce": b - a,
        "cfc_effect_with_current_cproj_ce": d - c,
        "cproj_effect_with_reference_cfc_ce": c - a,
        "cproj_effect_with_current_cfc_ce": d - b,
        "cfc_cproj_interaction_ce": d - b - c + a,
    }


def classify(
    aggregate: dict[str, Any],
    factorial_rows: list[dict[str, Any]],
    *,
    failure_step: int,
    minimum_material_ratio: float,
    minimum_cosine_shift: float,
    minimum_factorial_ce: float,
) -> dict[str, Any]:
    phase = aggregate[str(failure_step)]["late"]
    factorial = next(row for row in factorial_rows if int(row["step"]) == failure_step)
    pre = float(phase["pregelu_hard_rank_current_to_reference"])
    conversion = float(
        phase["postgelu_to_pregelu_hard_rank_current_to_reference"]
    )
    scale = float(phase["update_to_residual_rms_current_to_reference"])
    direction = float(phase["residual_update_cos_mean_delta_mean_abs"])
    interaction = float(factorial["cfc_cproj_interaction_ce"])
    cfc_effect = float(factorial["cfc_effect_with_reference_cproj_ce"])
    cproj_effect = float(factorial["cproj_effect_with_reference_cfc_ce"])
    lower = 1.0 - minimum_material_ratio
    upper = 1.0 + minimum_material_ratio
    shifted_steps = [
        int(step)
        for step, value in sorted(aggregate.items(), key=lambda item: int(item[0]))
        if float(value["late"]["residual_update_cos_mean_delta_mean_abs"])
        >= minimum_cosine_shift
    ]
    if conversion <= lower and pre >= lower:
        classification = "PHASE_POST_GELU_CONVERSION_DRIFT"
    elif pre <= lower:
        classification = "PHASE_PRE_GELU_CAPACITY_DRIFT"
    elif scale < lower or scale > upper:
        classification = "PHASE_RESIDUAL_BRANCH_SCALE_DRIFT"
    elif direction >= minimum_cosine_shift and interaction >= minimum_factorial_ce:
        classification = "COADAPTED_RESIDUAL_DIRECTION_DRIFT"
    elif direction >= minimum_cosine_shift and cfc_effect >= minimum_factorial_ce:
        classification = "CFC_INTRINSIC_DIRECTION_DRIFT"
    elif direction >= minimum_cosine_shift and cproj_effect >= minimum_factorial_ce:
        classification = "CPROJ_CONTEXT_DIRECTION_DRIFT"
    elif direction >= minimum_cosine_shift:
        classification = "DIRECTION_DRIFT_WITHOUT_TASK_FACTORIAL_PENALTY"
    else:
        classification = "PHASE_DRIFT_UNRESOLVED"
    return {
        "classification": classification,
        "failure_step": failure_step,
        "earliest_material_late_direction_shift_step": shifted_steps[0]
        if shifted_steps
        else None,
        "late_direction_shift_at_failure": direction,
        "factorial_interaction_ce_at_failure": interaction,
        "cfc_effect_with_reference_cproj_ce_at_failure": cfc_effect,
        "cproj_effect_with_reference_cfc_ce_at_failure": cproj_effect,
        "thresholds": {
            "minimum_material_ratio": minimum_material_ratio,
            "minimum_cosine_shift": minimum_cosine_shift,
            "minimum_factorial_ce": minimum_factorial_ce,
        },
        "parameter_updates": 0,
        "training_or_structure_authorized": False,
        "interpretation_boundary": (
            "Same-run phase localization and task-metric intervention only; raw PCA is "
            "descriptive and no cross-run tensor direction or transplant is interpreted."
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_ce(
    model: torch.nn.Module,
    batches: list[torch.Tensor],
    *,
    device: str,
    dtype: torch.dtype,
) -> tuple[float, list[float]]:
    losses: list[float] = []
    if getattr(model.config, "block_fht", False):
        model.prepare_block_fht_cache(dtype=dtype)
    try:
        with torch.no_grad():
            for batch in batches:
                inputs = batch[:, :-1].contiguous().to(device)
                targets = batch[:, 1:].contiguous().to(device)
                context = (
                    torch.amp.autocast("cuda", dtype=dtype)
                    if device.startswith("cuda")
                    else contextlib.nullcontext()
                )
                with context:
                    _logits, loss = model(inputs, targets)
                if loss is None or not torch.isfinite(loss):
                    raise RuntimeError("non-finite or absent probe loss")
                losses.append(float(loss))
    finally:
        if getattr(model.config, "block_fht", False):
            model.flush_block_fht_cache()
    return sum(losses) / len(losses), losses


def collect_native(
    payload: dict[str, Any],
    step: int,
    batches: list[torch.Tensor],
    *,
    data_dir: Path,
    device: str,
    dtype: torch.dtype,
    layers: list[int],
    sample_cap: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model = model_from_snapshot(payload, device)
    if model.config.block_fht:
        model.prepare_block_fht_cache(dtype=dtype)
    regimes = token_regimes_from_validation(data_dir, model.config.vocab_size)
    collector = ResidualCollector(model, regimes, layers, sample_cap)
    losses: list[float] = []
    try:
        with torch.no_grad():
            for batch in batches:
                inputs = batch[:, :-1].contiguous().to(device)
                targets = batch[:, 1:].contiguous().to(device)
                collector.set_tokens(inputs)
                context = (
                    torch.amp.autocast("cuda", dtype=dtype)
                    if device.startswith("cuda")
                    else contextlib.nullcontext()
                )
                with context:
                    _logits, loss = model(inputs, targets)
                if loss is None or not torch.isfinite(loss):
                    raise RuntimeError("non-finite or absent native probe loss")
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
                raise RuntimeError(
                    f"incomplete activation capture: step={step} layer={layer} regime={regime}"
                )
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
                    "step": step,
                    "layer": layer,
                    "band": next(name for name, band in BANDS.items() if layer in band),
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
    if model.config.block_fht:
        model.flush_block_fht_cache()
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return rows, {
        "step": step,
        "mean_probe_ce": sum(losses) / len(losses),
        "batch_probe_ce": losses,
    }


def copy_mlp_pair(
    model: torch.nn.Module,
    *,
    cfc_payload: dict[str, Any],
    cproj_payload: dict[str, Any],
) -> None:
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    with torch.no_grad():
        for layer in range(12):
            cfc_name = f"transformer.h.{layer}.mlp.c_fc.weight"
            cproj_name = f"transformer.h.{layer}.mlp.c_proj.weight"
            buffers[cfc_name].copy_(
                cfc_payload["buffers"][cfc_name].to(
                    device=buffers[cfc_name].device, dtype=buffers[cfc_name].dtype
                )
            )
            parameters[cproj_name].copy_(
                cproj_payload["parameters"][cproj_name].to(
                    device=parameters[cproj_name].device,
                    dtype=parameters[cproj_name].dtype,
                )
            )


def combination_ce(
    current_payload: dict[str, Any],
    cfc_payload: dict[str, Any],
    cproj_payload: dict[str, Any],
    batches: list[torch.Tensor],
    *,
    device: str,
    dtype: torch.dtype,
) -> float:
    model = model_from_snapshot(current_payload, device)
    copy_mlp_pair(model, cfc_payload=cfc_payload, cproj_payload=cproj_payload)
    mean, _losses = evaluate_ce(model, batches, device=device, dtype=dtype)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return mean


def weight_trajectory_rows(
    payloads: dict[int, dict[str, Any]], steps: list[int], layers: list[int]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer in layers:
        for target, container, name in (
            (
                "c_fc",
                "buffers",
                f"transformer.h.{layer}.mlp.c_fc.weight",
            ),
            (
                "c_proj",
                "parameters",
                f"transformer.h.{layer}.mlp.c_proj.weight",
            ),
        ):
            sequence = torch.stack(
                [payloads[step][container][name].float().flatten() for step in steps]
            )
            metrics = trajectory_metrics(sequence)
            rows.append(
                {
                    "layer": layer,
                    "band": next(band for band, values in BANDS.items() if layer in values),
                    "target": target,
                    "sampled_path_rank_upper_bound": len(steps) - 1,
                    "dimension_interpretation": "descriptive five-point path; not manifold dimension",
                    **metrics,
                }
            )
            del sequence
    return rows


def aggregate_weight_trajectories(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for target in ("c_fc", "c_proj"):
        target_rows = [row for row in rows if row["target"] == target]
        groups = {"all": target_rows}
        groups.update(
            {
                band: [row for row in target_rows if row["band"] == band]
                for band in BANDS
            }
        )
        output[target] = {}
        for name, values in groups.items():
            weights = [float(row["terminal_displacement_fro"]) ** 2 for row in values]
            denominator = sum(weights)
            if denominator <= 0:
                raise ValueError("zero trajectory displacement energy")
            output[target][name] = {
                "cells": len(values),
                **{
                    key: sum(
                        weight * float(row[key]) for weight, row in zip(weights, values)
                    )
                    / denominator
                    for key in TRAJECTORY_SUMMARY_KEYS
                },
            }
    return output


def validate_inputs(args: argparse.Namespace, plan: dict[str, Any]) -> dict[str, Any]:
    identity = plan["identity"]
    observed = {
        "entrypoint_sha256": file_sha256(Path(__file__)),
        "acquisition_plan_sha256": file_sha256(args.acquisition_plan),
        "acquisition_config_sha256": file_sha256(args.config),
        "acquisition_verifier_sha256": file_sha256(args.acquisition_verifier),
        "source_result_sha256": file_sha256(args.source_result),
        "terminal_audit_sha256": file_sha256(args.terminal_audit),
        "dataset_manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
    }
    if observed != identity:
        raise ValueError(f"phase-analysis identity mismatch: {observed}")
    verification = json.loads(args.verification.read_text())
    if (
        verification.get("passed") is not True
        or verification.get("classification")
        != "ACCEPTED_QK_CFC_20TPP_PHASE_ACQUISITION"
        or verification.get("authorization", {}).get("phase_analysis") is not True
    ):
        raise ValueError("phase acquisition was not accepted for analysis")
    if verification["identity"]["plan_sha256"] != identity["acquisition_plan_sha256"]:
        raise ValueError("verification plan identity mismatch")
    if verification["identity"]["config_sha256"] != identity["acquisition_config_sha256"]:
        raise ValueError("verification config identity mismatch")
    if verification["identity"]["dataset_manifest_sha256"] != identity["dataset_manifest_sha256"]:
        raise ValueError("verification dataset identity mismatch")
    return verification


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--verification", required=True, type=Path)
    parser.add_argument("--acquisition-plan", required=True, type=Path)
    parser.add_argument("--acquisition-verifier", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source-result", required=True, type=Path)
    parser.add_argument("--terminal-audit", required=True, type=Path)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    plan = json.loads(args.plan.read_text())
    if plan.get("schema_version") != "mai_124m_qk_cfc_20tpp_phase_direction_plan_v1":
        raise ValueError("unexpected phase-direction plan schema")
    verification = validate_inputs(args, plan)
    protocol = plan["protocol"]
    steps = [int(value) for value in protocol["steps"]]
    reference_step = int(protocol["reference_step"])
    paths = [args.snapshot_dir / f"step_{step:06d}.pt" for step in steps]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError("missing phase snapshots: " + ", ".join(missing))
    expected_hashes = verification["inventory"]["snapshot_sha256_by_step"]
    for step, path in zip(steps, paths):
        if file_sha256(path) != expected_hashes[str(step)]:
            raise ValueError(f"snapshot hash mismatch: {step}")
    payloads = {step: load_snapshot(path) for step, path in zip(steps, paths)}
    run_ids = {str(payload["run_identity_sha256"]) for payload in payloads.values()}
    if len(run_ids) != 1:
        raise ValueError("phase snapshots do not share one run identity")
    dtype = getattr(torch, str(protocol["dtype"]))
    batches = fixed_validation_batches(
        args.data_dir,
        int(protocol["batch_size"]),
        int(protocol["block_size"]) + 1,
        int(protocol["batches"]),
        int(protocol["sample_seed"]),
    )
    started = time.time()
    activation_rows: list[dict[str, Any]] = []
    native: dict[int, dict[str, Any]] = {}
    layers = [int(value) for value in protocol["layers"]]
    for step in steps:
        print(f"collecting native phase {step}", flush=True)
        rows, metadata = collect_native(
            payloads[step],
            step,
            batches,
            data_dir=args.data_dir,
            device=args.device,
            dtype=dtype,
            layers=layers,
            sample_cap=int(protocol["sample_cap"]),
        )
        activation_rows.extend(rows)
        native[step] = metadata
    comparison_rows = compare_phase_rows(activation_rows, reference_step)
    aggregate = aggregate_phase_rows(comparison_rows)
    factorial_rows: list[dict[str, Any]] = []
    reference_payload = payloads[reference_step]
    for step in steps:
        if step <= reference_step:
            continue
        print(f"evaluating c_fc/c_proj factorial phase {step}", flush=True)
        current = payloads[step]
        a = combination_ce(
            current,
            reference_payload,
            reference_payload,
            batches,
            device=args.device,
            dtype=dtype,
        )
        b = combination_ce(
            current,
            current,
            reference_payload,
            batches,
            device=args.device,
            dtype=dtype,
        )
        c = combination_ce(
            current,
            reference_payload,
            current,
            batches,
            device=args.device,
            dtype=dtype,
        )
        factorial_rows.append(
            factorial_effects(
                step=step,
                reference_pair_current_context_ce=a,
                current_cfc_reference_cproj_ce=b,
                reference_cfc_current_cproj_ce=c,
                native_current_ce=float(native[step]["mean_probe_ce"]),
            )
        )
    print("measuring descriptive weight paths", flush=True)
    trajectory_rows = weight_trajectory_rows(payloads, steps, layers)
    trajectory_aggregate = aggregate_weight_trajectories(trajectory_rows)
    decision_rule = plan["decision_rule"]
    decision = classify(
        aggregate,
        factorial_rows,
        failure_step=int(decision_rule["failure_step"]),
        minimum_material_ratio=float(decision_rule["minimum_material_ratio"]),
        minimum_cosine_shift=float(decision_rule["minimum_cosine_shift"]),
        minimum_factorial_ce=float(decision_rule["minimum_factorial_ce"]),
    )
    args.output.mkdir(parents=True)
    activation_path = args.output / "activation_rows.csv"
    comparison_path = args.output / "phase_comparison_rows.json"
    factorial_path = args.output / "factorial_rows.json"
    trajectory_path = args.output / "weight_trajectory_rows.json"
    write_csv(activation_path, activation_rows)
    comparison_path.write_text(json.dumps(comparison_rows, indent=2, sort_keys=True) + "\n")
    factorial_path.write_text(json.dumps(factorial_rows, indent=2, sort_keys=True) + "\n")
    trajectory_path.write_text(json.dumps(trajectory_rows, indent=2, sort_keys=True) + "\n")
    result = {
        "schema_version": SCHEMA_VERSION,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "parameter_updates": 0,
        "decision": decision,
        "native_phase_probe": native,
        "phase_aggregate": aggregate,
        "factorial_rows": factorial_rows,
        "weight_trajectory_aggregate": trajectory_aggregate,
        "identity": {
            **plan["identity"],
            "plan_sha256": file_sha256(args.plan),
            "verification_sha256": file_sha256(args.verification),
            "snapshot_run_identity_sha256": next(iter(run_ids)),
            "activation_rows_sha256": file_sha256(activation_path),
            "phase_comparison_rows_sha256": file_sha256(comparison_path),
            "factorial_rows_sha256": file_sha256(factorial_path),
            "weight_trajectory_rows_sha256": file_sha256(trajectory_path),
        },
        "execution": {
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": str(Path(__file__).resolve()),
            "command": sys.argv,
            "device": args.device,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "direct_foreground_polling": True,
            "watchdog": False,
            "callback": False,
        },
        "authorization": {
            "candidate_structure": False,
            "language_model_training": False,
            "larger_rung": False,
            "separate_theory_reconciliation_required": True,
        },
    }
    result_path = args.output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
