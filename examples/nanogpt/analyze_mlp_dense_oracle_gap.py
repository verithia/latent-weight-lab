#!/usr/bin/env python3
"""Attribute the exact-current MLP chart gap against dense Muon oracles.

One registered gradient and the persisted momentum states define both the
deployed c_fc/c_proj custom-optimizer updates and their uncompressed dense
Muon targets.  Native and per-family norm-matched dense controls are scored
on untouched windows with exact tensor restoration.  This separates chart
direction/capacity, update magnitude, family attribution, historical double
decay, and accumulated training-path effects without changing a checkpoint.
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
    _autocast,
    assert_joint_matches_singletons,
    extract_production_updates,
    family_weights,
    forward_capture,
)
from examples.nanogpt.analyze_mlp_joint_step_response_surface import (
    paired_comparison,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "nanogpt_mlp_dense_oracle_gap_v1"


def family_fro(updates: dict[int, torch.Tensor]) -> float:
    return math.sqrt(
        sum(float(update.double().square().sum()) for update in updates.values())
    )


def scale_family(
    updates: dict[int, torch.Tensor], scale: float
) -> dict[int, torch.Tensor]:
    return {layer: update * float(scale) for layer, update in updates.items()}


def merge_updates(
    cfc: dict[int, torch.Tensor], cproj: dict[int, torch.Tensor]
) -> dict[str, dict[int, torch.Tensor]]:
    return {"c_fc": cfc, "c_proj": cproj}


def aggregate_direction_metrics(
    target: dict[int, torch.Tensor], prediction: dict[int, torch.Tensor]
) -> dict[str, float]:
    if set(target) != set(prediction):
        raise ValueError("direction metric layers do not match")
    target_energy = 0.0
    prediction_energy = 0.0
    dot = 0.0
    residual_energy = 0.0
    for layer in sorted(target):
        target_value = target[layer].double()
        prediction_value = prediction[layer].double()
        target_energy += float(target_value.square().sum())
        prediction_energy += float(prediction_value.square().sum())
        dot += float((target_value * prediction_value).sum())
        residual_energy += float((target_value - prediction_value).square().sum())
    denominator = max(target_energy * prediction_energy, 1e-60)
    return {
        "target_fro": math.sqrt(target_energy),
        "prediction_fro": math.sqrt(prediction_energy),
        "cosine": dot / math.sqrt(denominator),
        "fixed_scale_recovery": 1.0
        - residual_energy / max(target_energy, 1e-30),
        "positive_line_recovery": max(dot, 0.0) ** 2 / denominator,
        "least_squares_scale": dot / max(prediction_energy, 1e-30),
    }


class ExactVariantApplier:
    def __init__(self, model) -> None:
        self.weights = family_weights(model)
        self.base = {
            family: {
                layer: weight.detach().clone()
                for layer, weight in by_layer.items()
            }
            for family, by_layer in self.weights.items()
        }

    @torch.no_grad()
    def restore(self) -> None:
        for family, by_layer in self.weights.items():
            for layer, weight in by_layer.items():
                weight.copy_(self.base[family][layer])

    @contextmanager
    def apply(
        self, updates: dict[str, dict[int, torch.Tensor]]
    ) -> Iterator[None]:
        self.restore()
        try:
            for family, by_layer in updates.items():
                for layer, update in by_layer.items():
                    weight = self.weights[family][layer]
                    value = self.base[family][layer].float() + update.to(
                        device=weight.device, dtype=torch.float32
                    )
                    weight.copy_(value.to(dtype=weight.dtype))
            yield
        finally:
            self.restore()


@torch.no_grad()
def evaluate_candidates(
    model,
    applier: ExactVariantApplier,
    batches_by_window: dict[str, list[torch.Tensor]],
    candidates: dict[str, dict[str, dict[int, torch.Tensor]]],
    *,
    device: str,
    dtype: torch.dtype,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model.eval()
    model.prepare_block_fht_cache(dtype=dtype)
    try:
        for window, batches in batches_by_window.items():
            for batch_index, tokens in enumerate(batches):
                tokens = tokens.to(device)
                for point_id, updates in candidates.items():
                    with applier.apply(updates):
                        with _autocast(device, dtype):
                            _logits, loss = model(
                                tokens[:, :-1].contiguous(),
                                tokens[:, 1:].contiguous(),
                            )
                        if loss is None:
                            raise RuntimeError("model did not return CE")
                        rows.append(
                            {
                                "window": window,
                                "batch_index": batch_index,
                                "point_id": point_id,
                                "ce": float(loss),
                            }
                        )
    finally:
        applier.restore()
        model.flush_block_fht_cache()
    return rows


def _finish_output_accumulator(values: dict[str, float]) -> dict[str, float]:
    denominator = max(
        values["target_energy"] * values["prediction_energy"], 1e-60
    )
    return {
        **values,
        "target_fro": math.sqrt(values["target_energy"]),
        "prediction_fro": math.sqrt(values["prediction_energy"]),
        "cosine": values["dot"] / math.sqrt(denominator),
        "fixed_scale_recovery": 1.0
        - values["residual_energy"] / max(values["target_energy"], 1e-30),
        "positive_line_recovery": max(values["dot"], 0.0) ** 2 / denominator,
    }


@torch.no_grad()
def output_effect_recovery(
    model,
    applier: ExactVariantApplier,
    batches_by_window: dict[str, list[torch.Tensor]],
    candidates: dict[str, dict[str, dict[int, torch.Tensor]]],
    probe_layers: list[int],
    *,
    device: str,
    dtype: torch.dtype,
) -> dict[str, Any]:
    target_id = "dense_historical_joint"
    prediction_ids = ("production_joint", "dense_norm_joint")
    accumulators: dict[tuple[str, str, int, str], dict[str, float]] = defaultdict(
        lambda: {
            "target_energy": 0.0,
            "prediction_energy": 0.0,
            "dot": 0.0,
            "residual_energy": 0.0,
        }
    )
    model.eval()
    model.prepare_block_fht_cache(dtype=dtype)
    try:
        for window, batches in batches_by_window.items():
            for tokens in batches:
                with applier.apply({}):
                    _loss, base = forward_capture(
                        model, tokens, probe_layers, device=device, dtype=dtype
                    )
                with applier.apply(candidates[target_id]):
                    _loss, target = forward_capture(
                        model, tokens, probe_layers, device=device, dtype=dtype
                    )
                predictions = {}
                for candidate in prediction_ids:
                    with applier.apply(candidates[candidate]):
                        _loss, predictions[candidate] = forward_capture(
                            model,
                            tokens,
                            probe_layers,
                            device=device,
                            dtype=dtype,
                        )
                for layer in probe_layers:
                    for kind in ("mlp", "block"):
                        target_delta = target[(layer, kind)] - base[(layer, kind)]
                        for candidate in prediction_ids:
                            prediction_delta = (
                                predictions[candidate][(layer, kind)]
                                - base[(layer, kind)]
                            )
                            key = (window, candidate, layer, kind)
                            row = accumulators[key]
                            target_d = target_delta.double()
                            prediction_d = prediction_delta.double()
                            row["target_energy"] += float(target_d.square().sum())
                            row["prediction_energy"] += float(
                                prediction_d.square().sum()
                            )
                            row["dot"] += float((target_d * prediction_d).sum())
                            row["residual_energy"] += float(
                                (target_d - prediction_d).square().sum()
                            )
    finally:
        applier.restore()
        model.flush_block_fht_cache()
    rows = []
    for (window, candidate, layer, kind), values in sorted(accumulators.items()):
        rows.append(
            {
                "window": window,
                "candidate": candidate,
                "layer": layer,
                "kind": kind,
                **_finish_output_accumulator(values),
            }
        )
    return {"rows": rows}


def classify_dense_oracle(
    rows: list[dict[str, Any]], confidence_z: float
) -> dict[str, Any]:
    pairs = {
        "dense_native_joint_vs_production": (
            "dense_historical_joint",
            "production_joint",
        ),
        "dense_norm_joint_vs_production": (
            "dense_norm_joint",
            "production_joint",
        ),
        "dense_norm_cfc_vs_production_cfc": (
            "dense_norm_cfc",
            "production_cfc",
        ),
        "dense_norm_cproj_vs_production_cproj": (
            "dense_norm_cproj",
            "production_cproj",
        ),
        "hybrid_norm_cfc_vs_production_joint": (
            "hybrid_norm_cfc",
            "production_joint",
        ),
        "hybrid_norm_cproj_vs_production_joint": (
            "hybrid_norm_cproj",
            "production_joint",
        ),
        "canonical_joint_vs_production": (
            "dense_canonical_joint",
            "production_joint",
        ),
        "canonical_joint_vs_historical_dense": (
            "dense_canonical_joint",
            "dense_historical_joint",
        ),
    }
    comparisons = {
        name: paired_comparison(rows, candidate, reference, confidence_z)
        for name, (candidate, reference) in pairs.items()
    }
    native = comparisons["dense_native_joint_vs_production"][
        "candidate_reliably_better"
    ]
    norm = comparisons["dense_norm_joint_vs_production"][
        "candidate_reliably_better"
    ]
    cfc_single = comparisons["dense_norm_cfc_vs_production_cfc"][
        "candidate_reliably_better"
    ]
    cproj_single = comparisons["dense_norm_cproj_vs_production_cproj"][
        "candidate_reliably_better"
    ]
    cfc_hybrid = comparisons["hybrid_norm_cfc_vs_production_joint"][
        "candidate_reliably_better"
    ]
    cproj_hybrid = comparisons["hybrid_norm_cproj_vs_production_joint"][
        "candidate_reliably_better"
    ]
    if not native:
        classification = "LOCAL_DENSE_ORACLE_NOT_BETTER_THAN_PRODUCTION"
        next_action = "ATTRIBUTE_ACCUMULATED_REPRESENTATION_NOT_LOCAL_STEP"
    elif not norm:
        classification = "DENSE_ORACLE_GAIN_IS_UPDATE_MAGNITUDE_ONLY"
        next_action = "TEST_CHART_RADIUS_CAPACITY_WITHOUT_NEW_TOPOLOGY"
    elif cfc_single and cfc_hybrid and cproj_single and cproj_hybrid:
        classification = "BOTH_MLP_CHARTS_LIMIT_DENSE_ORACLE"
        next_action = "BRACKET_TASK_SELECTED_CAPACITY_FOR_BOTH_MLP_FAMILIES"
    elif cfc_single and cfc_hybrid:
        classification = "CFC_CHART_LIMITS_DENSE_ORACLE"
        next_action = "BRACKET_TASK_SELECTED_CFC_CAPACITY"
    elif cproj_single and cproj_hybrid:
        classification = "CPROJ_CHART_LIMITS_DENSE_ORACLE"
        next_action = "BRACKET_TASK_SELECTED_CPROJ_CAPACITY"
    else:
        classification = "JOINT_ONLY_MLP_CHART_CAPACITY_DEFECT"
        next_action = "DESIGN_COUPLED_REPRESENTATIONAL_CHART_NOT_STEP_METRIC"
    return {
        "classification": classification,
        "next_action": next_action,
        "comparisons": comparisons,
    }


def validate_plan(
    plan_path: Path, checkpoint: Path, config_path: Path, data_dir: Path
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
    production, dense, extracted = extract_production_updates(
        args.checkpoint,
        config,
        train_batches,
        device=args.device,
        dtype=dtype,
        return_dense_oracle=True,
    )
    exactness = assert_joint_matches_singletons(
        production["cfc_only"]["c_fc"],
        production["cproj_only"]["c_proj"],
        production["joint"],
    )
    prod_cfc = production["cfc_only"]["c_fc"]
    prod_cproj = production["cproj_only"]["c_proj"]
    hist_cfc = dense["historical_double_decay"]["c_fc"]
    hist_cproj = dense["historical_double_decay"]["c_proj"]
    canonical_cfc = dense["canonical_single_decay"]["c_fc"]
    canonical_cproj = dense["canonical_single_decay"]["c_proj"]
    cfc_scale = family_fro(prod_cfc) / family_fro(hist_cfc)
    cproj_scale = family_fro(prod_cproj) / family_fro(hist_cproj)
    norm_cfc = scale_family(hist_cfc, cfc_scale)
    norm_cproj = scale_family(hist_cproj, cproj_scale)
    candidates = {
        "baseline": {},
        "production_cfc": {"c_fc": prod_cfc},
        "production_cproj": {"c_proj": prod_cproj},
        "production_joint": merge_updates(prod_cfc, prod_cproj),
        "dense_historical_cfc": {"c_fc": hist_cfc},
        "dense_historical_cproj": {"c_proj": hist_cproj},
        "dense_historical_joint": merge_updates(hist_cfc, hist_cproj),
        "dense_norm_cfc": {"c_fc": norm_cfc},
        "dense_norm_cproj": {"c_proj": norm_cproj},
        "dense_norm_joint": merge_updates(norm_cfc, norm_cproj),
        "hybrid_norm_cfc": merge_updates(norm_cfc, prod_cproj),
        "hybrid_norm_cproj": merge_updates(prod_cfc, norm_cproj),
        "dense_canonical_joint": merge_updates(canonical_cfc, canonical_cproj),
    }
    model, _optimizer, checkpoint_payload = load_model_and_optimizer(
        args.checkpoint, config, args.device
    )
    applier = ExactVariantApplier(model)

    def windows(seeds: list[int], batches: int) -> dict[str, list[torch.Tensor]]:
        return {
            f"window_{index + 1}": fixed_batches(
                args.data_dir,
                "val",
                batch_size=int(protocol["evaluation_batch_size"]),
                block_size=int(protocol["evaluation_block_size"]) + 1,
                batches=batches,
                seed=int(seed),
            )
            for index, seed in enumerate(seeds)
        }

    ce_rows = evaluate_candidates(
        model,
        applier,
        windows(
            [int(seed) for seed in protocol["validation_seeds"]],
            int(protocol["validation_batches_per_window"]),
        ),
        candidates,
        device=args.device,
        dtype=dtype,
    )
    output_recovery = output_effect_recovery(
        model,
        applier,
        windows(
            [int(seed) for seed in protocol["output_seeds"]],
            int(protocol["output_batches_per_window"]),
        ),
        candidates,
        [int(layer) for layer in protocol["probe_layers"]],
        device=args.device,
        dtype=dtype,
    )
    decision = classify_dense_oracle(
        ce_rows, float(plan["decision_rule"]["confidence_z"])
    )
    weight_recovery = {
        "c_fc": aggregate_direction_metrics(hist_cfc, prod_cfc),
        "c_proj": aggregate_direction_metrics(hist_cproj, prod_cproj),
        "c_fc_norm_matched": aggregate_direction_metrics(hist_cfc, norm_cfc),
        "c_proj_norm_matched": aggregate_direction_metrics(
            hist_cproj, norm_cproj
        ),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    paths = {
        "ce": args.output / "heldout_ce.json",
        "output": args.output / "output_effect_recovery.json",
        "prospective": args.output / "prospective_step_metadata.json",
    }
    paths["ce"].write_text(json.dumps(ce_rows, indent=2, sort_keys=True) + "\n")
    paths["output"].write_text(
        json.dumps(output_recovery, indent=2, sort_keys=True) + "\n"
    )
    paths["prospective"].write_text(
        json.dumps(extracted, indent=2, sort_keys=True) + "\n"
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "weight_recovery": weight_recovery,
        "norm_match_scales": {"c_fc": cfc_scale, "c_proj": cproj_scale},
        "output_effect_recovery": output_recovery,
        "parameter_updates_to_checkpoint": 0,
        "disposable_optimizer_steps": 3,
        "checkpoint_next_iter": int(checkpoint_payload["next_iter"]),
        "joint_singleton_update_max_abs_error": exactness,
        "identity": {
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "config_sha256": file_sha256(args.config),
            "dataset_manifest_sha256": file_sha256(
                args.data_dir / "manifest.json"
            ),
            "plan_sha256": file_sha256(args.plan),
        },
        "outputs": {
            f"{name}_sha256": file_sha256(path) for name, path in paths.items()
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
