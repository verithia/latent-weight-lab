#!/usr/bin/env python3
"""Localize prospective c_fc/c_proj interaction by transformer depth.

This is a zero-persistent-update diagnostic.  It extracts one exact pair of
production custom-optimizer moves from a checkpoint and scores c_fc-only,
c_proj-only, and joint applications in early, middle, late, and all-layer
bands.  Every temporary BF16 weight application is restored by exact copy,
not add/subtract, so repeated finite evaluations cannot drift the checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

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
    family_weights,
    forward_capture,
)
from examples.nanogpt.model import GPT


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "nanogpt_mlp_joint_prospective_step_by_depth_v1"
FAMILIES = ("c_fc", "c_proj")
KINDS = ("cfc_only", "cproj_only", "joint")


def make_band_variants(
    cfc: dict[int, torch.Tensor],
    cproj: dict[int, torch.Tensor],
    bands: dict[str, list[int]],
) -> dict[str, dict[str, dict[str, dict[int, torch.Tensor]]]]:
    if set(cfc) != set(cproj):
        raise ValueError("c_fc and c_proj layer sets differ")
    available = set(cfc)
    result: dict[str, dict[str, dict[str, dict[int, torch.Tensor]]]] = {}
    for band, layers in bands.items():
        selected = set(layers)
        if not selected or not selected <= available:
            raise ValueError(f"invalid layers for band {band}: {layers}")
        band_cfc = {layer: cfc[layer] for layer in layers}
        band_cproj = {layer: cproj[layer] for layer in layers}
        result[band] = {
            "cfc_only": {"c_fc": band_cfc},
            "cproj_only": {"c_proj": band_cproj},
            "joint": {"c_fc": band_cfc, "c_proj": band_cproj},
        }
    return result


@contextmanager
def applied_updates_exact_restore(
    model: GPT,
    updates: dict[str, dict[int, torch.Tensor]],
) -> Iterator[None]:
    weights = family_weights(model)
    originals = {
        (family, layer): weights[family][layer].detach().clone()
        for family, by_layer in updates.items()
        for layer in by_layer
    }
    try:
        with torch.no_grad():
            for family, by_layer in updates.items():
                for layer, update in by_layer.items():
                    weight = weights[family][layer]
                    weight.add_(update.to(device=weight.device, dtype=weight.dtype))
        yield
    finally:
        with torch.no_grad():
            for (family, layer), original in originals.items():
                weights[family][layer].copy_(original)


def _interaction_row(
    *,
    window: str,
    batch_index: int,
    band: str,
    baseline: float,
    cfc: float,
    cproj: float,
    joint: float,
) -> dict[str, Any]:
    cfc_delta = cfc - baseline
    cproj_delta = cproj - baseline
    joint_delta = joint - baseline
    interaction = joint_delta - cfc_delta - cproj_delta
    scale = max(abs(cfc_delta) + abs(cproj_delta), 1e-12)
    return {
        "window": window,
        "batch_index": batch_index,
        "band": band,
        "baseline_ce": baseline,
        "cfc_ce": cfc,
        "cproj_ce": cproj,
        "joint_ce": joint,
        "cfc_loss_change": cfc_delta,
        "cproj_loss_change": cproj_delta,
        "joint_loss_change": joint_delta,
        "finite_additive_prediction": cfc_delta + cproj_delta,
        "finite_interaction": interaction,
        "normalized_interaction": interaction / scale,
    }


def _accumulate_output_interaction(
    sums: dict[str, float],
    base: torch.Tensor,
    cfc: torch.Tensor,
    cproj: torch.Tensor,
    joint: torch.Tensor,
) -> None:
    delta_fc = cfc - base
    delta_proj = cproj - base
    delta_joint = joint - base
    additive = delta_fc + delta_proj
    interaction = delta_joint - additive
    sums["base_energy"] += float(base.double().square().sum())
    sums["cfc_energy"] += float(delta_fc.double().square().sum())
    sums["cproj_energy"] += float(delta_proj.double().square().sum())
    sums["joint_energy"] += float(delta_joint.double().square().sum())
    sums["additive_energy"] += float(additive.double().square().sum())
    sums["interaction_energy"] += float(interaction.double().square().sum())
    sums["cfc_cproj_dot"] += float(
        (delta_fc.double() * delta_proj.double()).sum()
    )
    sums["joint_additive_dot"] += float(
        (delta_joint.double() * additive.double()).sum()
    )


def evaluate_depth_windows(
    model: GPT,
    batches_by_window: dict[str, list[torch.Tensor]],
    variants: dict[str, dict[str, dict[str, dict[int, torch.Tensor]]]],
    probe_layers_by_band: dict[str, list[int]],
    metric_batches: int,
    *,
    device: str,
    dtype: torch.dtype,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    output_sums: dict[tuple[str, str, int, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    all_probe_layers = sorted(
        {layer for layers in probe_layers_by_band.values() for layer in layers}
    )
    model.eval()
    model.prepare_block_fht_cache(dtype=dtype)
    try:
        for window, batches in batches_by_window.items():
            for batch_index, tokens in enumerate(batches):
                capture_layers = all_probe_layers if batch_index < metric_batches else []
                baseline, base_values = forward_capture(
                    model,
                    tokens,
                    capture_layers,
                    device=device,
                    dtype=dtype,
                )
                for band, band_variants in variants.items():
                    losses: dict[str, float] = {}
                    captured: dict[str, dict[tuple[int, str], torch.Tensor]] = {}
                    for kind in KINDS:
                        with applied_updates_exact_restore(
                            model, band_variants[kind]
                        ):
                            loss, values = forward_capture(
                                model,
                                tokens,
                                capture_layers,
                                device=device,
                                dtype=dtype,
                            )
                        losses[kind] = loss
                        captured[kind] = values
                    rows.append(
                        _interaction_row(
                            window=window,
                            batch_index=batch_index,
                            band=band,
                            baseline=baseline,
                            cfc=losses["cfc_only"],
                            cproj=losses["cproj_only"],
                            joint=losses["joint"],
                        )
                    )
                    if batch_index >= metric_batches:
                        continue
                    for layer in probe_layers_by_band[band]:
                        for output_kind in ("mlp", "block"):
                            key = (layer, output_kind)
                            _accumulate_output_interaction(
                                output_sums[(window, band, layer, output_kind)],
                                base_values[key],
                                captured["cfc_only"][key],
                                captured["cproj_only"][key],
                                captured["joint"][key],
                            )
    finally:
        model.flush_block_fht_cache()

    output_rows: list[dict[str, Any]] = []
    for (window, band, layer, kind), sums in sorted(output_sums.items()):
        cfc_norm = math.sqrt(sums["cfc_energy"])
        cproj_norm = math.sqrt(sums["cproj_energy"])
        joint_norm = math.sqrt(sums["joint_energy"])
        additive_norm = math.sqrt(sums["additive_energy"])
        interaction_norm = math.sqrt(sums["interaction_energy"])
        output_rows.append(
            {
                "window": window,
                "band": band,
                "layer": layer,
                "kind": kind,
                **dict(sums),
                "cfc_cproj_cosine": sums["cfc_cproj_dot"]
                / max(cfc_norm * cproj_norm, 1e-30),
                "joint_additive_cosine": sums["joint_additive_dot"]
                / max(joint_norm * additive_norm, 1e-30),
                "interaction_to_additive_norm": interaction_norm
                / max(additive_norm, 1e-30),
                "joint_to_base_norm": joint_norm
                / max(math.sqrt(sums["base_energy"]), 1e-30),
            }
        )
    return rows, output_rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _mean_sem_ci(values: list[float], z: float) -> dict[str, float]:
    mean = _mean(values)
    if len(values) < 2:
        sem = 0.0
    else:
        variance = sum((value - mean) ** 2 for value in values) / (
            len(values) - 1
        )
        sem = math.sqrt(variance / len(values))
    return {
        "mean": mean,
        "sem": sem,
        "ci_low": mean - z * sem,
        "ci_high": mean + z * sem,
    }


def classify_depth_interactions(
    rows: list[dict[str, Any]],
    *,
    bands: list[str],
    confidence_z: float,
    additive_tolerance: float,
) -> dict[str, Any]:
    by_band: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_band[str(row["band"])].append(row)
    if set(by_band) != set(bands):
        raise ValueError("finite rows do not cover every registered band")
    decisions: dict[str, Any] = {}
    for band in bands:
        band_rows = by_band[band]
        windows = sorted({str(row["window"]) for row in band_rows})
        window_metrics: dict[str, Any] = {}
        for window in windows:
            selected = [row for row in band_rows if row["window"] == window]
            metrics = {
                key: _mean([float(row[key]) for row in selected])
                for key in (
                    "cfc_loss_change",
                    "cproj_loss_change",
                    "joint_loss_change",
                    "finite_interaction",
                )
            }
            metrics["normalized_interaction"] = metrics[
                "finite_interaction"
            ] / max(
                abs(metrics["cfc_loss_change"])
                + abs(metrics["cproj_loss_change"]),
                1e-12,
            )
            window_metrics[window] = metrics
        interactions = [float(row["finite_interaction"]) for row in band_rows]
        interaction_ci = _mean_sem_ci(interactions, confidence_z)
        cfc_mean = _mean([float(row["cfc_loss_change"]) for row in band_rows])
        cproj_mean = _mean([float(row["cproj_loss_change"]) for row in band_rows])
        normalized = interaction_ci["mean"] / max(
            abs(cfc_mean) + abs(cproj_mean), 1e-12
        )
        window_interactions = [
            float(metric["finite_interaction"])
            for metric in window_metrics.values()
        ]
        if interaction_ci["ci_low"] > 0.0 and all(
            value > 0.0 for value in window_interactions
        ):
            classification = "DESTRUCTIVE_CFC_CPROJ_UPDATE_INTERACTION"
        elif interaction_ci["ci_high"] < 0.0 and all(
            value < 0.0 for value in window_interactions
        ):
            classification = "COOPERATIVE_CFC_CPROJ_UPDATE_INTERACTION"
        elif (
            interaction_ci["ci_low"] <= 0.0 <= interaction_ci["ci_high"]
            and abs(normalized) <= additive_tolerance
            and all(
                abs(float(metric["normalized_interaction"]))
                <= additive_tolerance
                for metric in window_metrics.values()
            )
        ):
            classification = "CFC_CPROJ_UPDATES_ARE_FINITE_CE_ADDITIVE"
        else:
            classification = "MIXED_CFC_CPROJ_UPDATE_INTERACTION"
        decisions[band] = {
            "classification": classification,
            "interaction": interaction_ci,
            "normalized_mean_interaction": normalized,
            "cfc_mean_loss_change": cfc_mean,
            "cproj_mean_loss_change": cproj_mean,
            "joint_mean_loss_change": _mean(
                [float(row["joint_loss_change"]) for row in band_rows]
            ),
            "joint_helpful_on_every_window": all(
                float(metric["joint_loss_change"]) < 0.0
                for metric in window_metrics.values()
            ),
            "window_metrics": window_metrics,
            "finite_batch_count": len(band_rows),
        }
    destructive = [
        band
        for band, decision in decisions.items()
        if decision["classification"]
        == "DESTRUCTIVE_CFC_CPROJ_UPDATE_INTERACTION"
    ]
    cooperative = [
        band
        for band, decision in decisions.items()
        if decision["classification"]
        == "COOPERATIVE_CFC_CPROJ_UPDATE_INTERACTION"
    ]
    mixed = [
        band
        for band, decision in decisions.items()
        if decision["classification"] == "MIXED_CFC_CPROJ_UPDATE_INTERACTION"
    ]
    if destructive:
        overall = "DEPTH_LOCALIZED_DESTRUCTIVE_INTERACTION"
        next_action = "TEST_ONLY_THE_PREREGISTERED_DESTRUCTIVE_BANDS_WITH_A_JOINT_OUTPUT_METRIC_CHART"
    elif mixed:
        overall = "NO_STABLE_DEPTH_LOCALIZED_INTERACTION"
        next_action = "DO_NOT_CHANGE_RESIDUAL_TOPOLOGY_OR_ADD_A_JOINT_CHART"
    elif cooperative:
        overall = "NONDESTRUCTIVE_COOPERATIVE_INTERACTION"
        next_action = "PRESERVE_EXISTING_COUPLING_AND_DIAGNOSE_REPRESENTATIONAL_CAPACITY"
    else:
        overall = "FINITE_CE_ADDITIVE_BY_DEPTH"
        next_action = "DIAGNOSE_INDIVIDUAL_DIRECTION_QUALITY_NOT_JOINT_INTERACTION"
    return {
        "classification": overall,
        "destructive_bands": destructive,
        "cooperative_bands": cooperative,
        "mixed_bands": mixed,
        "bands": decisions,
        "next_action": next_action,
    }


def validate_plan(
    plan_path: Path,
    checkpoint: Path,
    config_path: Path,
    data_dir: Path,
) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    identity = plan["identity"]
    actual = {
        "checkpoint_sha256": file_sha256(checkpoint),
        "config_sha256": file_sha256(config_path),
        "dataset_manifest_sha256": file_sha256(data_dir / "manifest.json"),
        "entrypoint_sha256": file_sha256(Path(__file__).resolve()),
    }
    for key, value in actual.items():
        if value != identity[key]:
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
    if int(protocol["gradient_accumulation_steps"]) != int(
        config["gradient_accumulation_steps"]
    ):
        raise ValueError("plan does not match production accumulation")
    train_batches = fixed_batches(
        args.data_dir,
        "train",
        batch_size=int(config["batch_size"]),
        block_size=int(config["block_size"]) + 1,
        batches=int(protocol["gradient_accumulation_steps"]),
        seed=int(protocol["train_seed"]),
    )
    extracted_updates, extracted = extract_production_updates(
        args.checkpoint,
        config,
        train_batches,
        device=args.device,
        dtype=dtype,
    )
    exactness = assert_joint_matches_singletons(
        extracted_updates["cfc_only"]["c_fc"],
        extracted_updates["cproj_only"]["c_proj"],
        extracted_updates["joint"],
    )
    bands = {
        str(name): [int(layer) for layer in layers]
        for name, layers in protocol["bands"].items()
    }
    variants = make_band_variants(
        extracted_updates["cfc_only"]["c_fc"],
        extracted_updates["cproj_only"]["c_proj"],
        bands,
    )
    batches_by_window = {
        f"validation_{index + 1}": fixed_batches(
            args.data_dir,
            "val",
            batch_size=int(protocol["evaluation_batch_size"]),
            block_size=int(protocol["evaluation_block_size"]) + 1,
            batches=int(protocol["evaluation_batches_per_window"]),
            seed=int(seed),
        )
        for index, seed in enumerate(protocol["validation_seeds"])
    }
    model, _optimizer, checkpoint_payload = load_model_and_optimizer(
        args.checkpoint, config, args.device
    )
    finite_rows, output_rows = evaluate_depth_windows(
        model,
        batches_by_window,
        variants,
        {
            str(name): [int(layer) for layer in layers]
            for name, layers in protocol["probe_layers_by_band"].items()
        },
        int(protocol["output_metric_batches_per_window"]),
        device=args.device,
        dtype=dtype,
    )
    decision = classify_depth_interactions(
        finite_rows,
        bands=list(bands),
        confidence_z=float(plan["decision_rule"]["confidence_z"]),
        additive_tolerance=float(plan["decision_rule"]["additive_tolerance"]),
    )
    args.output.mkdir(parents=True, exist_ok=False)
    finite_path = args.output / "finite_ce_by_batch.json"
    output_path = args.output / "output_interaction_by_depth.json"
    extraction_path = args.output / "prospective_step_metadata.json"
    decision_path = args.output / "depth_decision.json"
    finite_path.write_text(json.dumps(finite_rows, indent=2, sort_keys=True) + "\n")
    output_path.write_text(json.dumps(output_rows, indent=2, sort_keys=True) + "\n")
    extraction_path.write_text(json.dumps(extracted, indent=2, sort_keys=True) + "\n")
    decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "parameter_updates_to_checkpoint": 0,
        "disposable_optimizer_steps": 3,
        "checkpoint_next_iter": int(checkpoint_payload["next_iter"]),
        "identity": {
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "config_sha256": file_sha256(args.config),
            "dataset_manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
            "plan_sha256": file_sha256(args.plan),
        },
        "protocol": protocol,
        "joint_singleton_update_max_abs_error": exactness,
        "outputs": {
            "finite_ce_by_batch_sha256": file_sha256(finite_path),
            "output_interaction_by_depth_sha256": file_sha256(output_path),
            "prospective_step_metadata_sha256": file_sha256(extraction_path),
            "depth_decision_sha256": file_sha256(decision_path),
        },
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
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
