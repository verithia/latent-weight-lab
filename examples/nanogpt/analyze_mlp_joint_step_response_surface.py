#!/usr/bin/env python3
"""Measure a held-out c_fc/c_proj prospective-step response surface.

The surface is parameterized by total update scale and by the fraction of
the squared materialized update norm allocated to c_fc.  Calibration windows
select fixed-budget, common-scale, full-surface, and axis controls; disjoint
validation windows decide whether a cheap scalar coordination fix is real.
The checkpoint is never persistently updated.
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
from examples.nanogpt.analyze_mlp_joint_prospective_step_by_depth import (
    _mean_sem_ci,
)
from examples.nanogpt.model import GPT


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "nanogpt_mlp_joint_step_response_surface_v1"


def allocation_coefficients(
    fraction_cfc: float,
    total_scale: float,
    cfc_norm: float,
    cproj_norm: float,
) -> tuple[float, float]:
    if not 0.0 <= fraction_cfc <= 1.0:
        raise ValueError("c_fc allocation fraction must be in [0, 1]")
    if total_scale < 0.0 or cfc_norm <= 0.0 or cproj_norm <= 0.0:
        raise ValueError("scale must be nonnegative and update norms positive")
    radius = math.sqrt(cfc_norm * cfc_norm + cproj_norm * cproj_norm)
    cfc_scale = total_scale * radius * math.sqrt(fraction_cfc) / cfc_norm
    cproj_scale = (
        total_scale * radius * math.sqrt(1.0 - fraction_cfc) / cproj_norm
    )
    return cfc_scale, cproj_scale


def make_surface_points(
    total_scales: list[float],
    allocation_fractions: list[float],
    production_fraction: float,
    cfc_norm: float,
    cproj_norm: float,
) -> list[dict[str, Any]]:
    fractions = sorted(set(allocation_fractions + [production_fraction]))
    points = []
    for scale in total_scales:
        for fraction in fractions:
            cfc_scale, cproj_scale = allocation_coefficients(
                fraction, scale, cfc_norm, cproj_norm
            )
            points.append(
                {
                    "point_id": f"s{scale:.6f}_f{fraction:.12f}",
                    "total_scale": scale,
                    "fraction_cfc": fraction,
                    "cfc_scale": cfc_scale,
                    "cproj_scale": cproj_scale,
                }
            )
    return points


class ScaledUpdateApplier:
    def __init__(
        self,
        model: GPT,
        cfc: dict[int, torch.Tensor],
        cproj: dict[int, torch.Tensor],
    ) -> None:
        weights = family_weights(model)
        self.weights = {"c_fc": weights["c_fc"], "c_proj": weights["c_proj"]}
        self.originals = {
            family: {
                layer: weight.detach().clone()
                for layer, weight in by_layer.items()
            }
            for family, by_layer in self.weights.items()
        }
        self.updates = {
            "c_fc": {
                layer: tensor.to(
                    device=self.weights["c_fc"][layer].device,
                    dtype=self.weights["c_fc"][layer].dtype,
                )
                for layer, tensor in cfc.items()
            },
            "c_proj": {
                layer: tensor.to(
                    device=self.weights["c_proj"][layer].device,
                    dtype=self.weights["c_proj"][layer].dtype,
                )
                for layer, tensor in cproj.items()
            },
        }

    def restore(self) -> None:
        with torch.no_grad():
            for family, by_layer in self.originals.items():
                for layer, original in by_layer.items():
                    self.weights[family][layer].copy_(original)

    @contextmanager
    def apply(self, cfc_scale: float, cproj_scale: float) -> Iterator[None]:
        self.restore()
        try:
            with torch.no_grad():
                for layer, update in self.updates["c_fc"].items():
                    self.weights["c_fc"][layer].add_(update, alpha=cfc_scale)
                for layer, update in self.updates["c_proj"].items():
                    self.weights["c_proj"][layer].add_(update, alpha=cproj_scale)
            yield
        finally:
            self.restore()


def evaluate_points(
    model: GPT,
    applier: ScaledUpdateApplier,
    batches_by_window: dict[str, list[torch.Tensor]],
    points: list[dict[str, Any]],
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
                baseline, _values = forward_capture(
                    model, tokens, [], device=device, dtype=dtype
                )
                for point in points:
                    with applier.apply(
                        float(point["cfc_scale"]),
                        float(point["cproj_scale"]),
                    ):
                        loss, _values = forward_capture(
                            model, tokens, [], device=device, dtype=dtype
                        )
                    rows.append(
                        {
                            "window": window,
                            "batch_index": batch_index,
                            **point,
                            "baseline_ce": baseline,
                            "ce": loss,
                            "loss_change": loss - baseline,
                        }
                    )
    finally:
        applier.restore()
        model.flush_block_fht_cache()
    return rows


def _means_by_point(rows: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["point_id"])].append(float(row["loss_change"]))
    return {point: sum(values) / len(values) for point, values in grouped.items()}


def _select_best(
    points: list[dict[str, Any]],
    means: dict[str, float],
) -> dict[str, Any]:
    if not points:
        raise ValueError("cannot select from an empty surface subset")
    return min(
        points,
        key=lambda point: (
            means[str(point["point_id"])],
            float(point["total_scale"]),
            float(point["fraction_cfc"]),
        ),
    )


def select_calibration_controls(
    rows: list[dict[str, Any]],
    points: list[dict[str, Any]],
    production_fraction: float,
) -> dict[str, dict[str, Any]]:
    means = _means_by_point(rows)
    fixed_budget = [p for p in points if float(p["total_scale"]) == 1.0]
    common_scale = [
        p
        for p in points
        if math.isclose(
            float(p["fraction_cfc"]), production_fraction, abs_tol=1e-12
        )
    ]
    axes = [
        p for p in points if float(p["fraction_cfc"]) in {0.0, 1.0}
    ]
    production = next(
        p
        for p in points
        if float(p["total_scale"]) == 1.0
        and math.isclose(
            float(p["fraction_cfc"]), production_fraction, abs_tol=1e-12
        )
    )
    selected = {
        "production": production,
        "fixed_budget": _select_best(fixed_budget, means),
        "common_scale": _select_best(common_scale, means),
        "surface": _select_best(points, means),
        "axis": _select_best(axes, means),
    }
    return {
        name: {**point, "calibration_mean_loss_change": means[point["point_id"]]}
        for name, point in selected.items()
    }


def paired_comparison(
    rows: list[dict[str, Any]],
    candidate_id: str,
    reference_id: str,
    confidence_z: float,
) -> dict[str, Any]:
    values: dict[str, dict[tuple[str, int], float]] = defaultdict(dict)
    for row in rows:
        point_id = str(row["point_id"])
        if point_id in {candidate_id, reference_id}:
            values[point_id][(str(row["window"]), int(row["batch_index"]))] = float(
                row["ce"]
            )
    if set(values[candidate_id]) != set(values[reference_id]):
        raise ValueError("paired validation keys do not match")
    differences = [
        values[candidate_id][key] - values[reference_id][key]
        for key in sorted(values[candidate_id])
    ]
    result: dict[str, Any] = _mean_sem_ci(differences, confidence_z)
    result["window_means"] = {
        window: sum(
            difference
            for (row_window, _index), difference in zip(
                sorted(values[candidate_id]), differences
            )
            if row_window == window
        )
        / sum(1 for row_window, _index in values[candidate_id] if row_window == window)
        for window in sorted({window for window, _index in values[candidate_id]})
    }
    result["candidate_better_on_every_window"] = all(
        value < 0.0 for value in result["window_means"].values()
    )
    result["candidate_reliably_better"] = (
        result["ci_high"] < 0.0 and result["candidate_better_on_every_window"]
    )
    return result


def fit_quadratic_surface(rows: list[dict[str, Any]]) -> dict[str, Any]:
    means = _means_by_point(rows)
    point_by_id = {str(row["point_id"]): row for row in rows}
    design = []
    targets = []
    for point_id, loss_change in sorted(means.items()):
        row = point_by_id[point_id]
        a = float(row["cfc_scale"])
        b = float(row["cproj_scale"])
        design.append([1.0, a, b, 0.5 * a * a, a * b, 0.5 * b * b])
        targets.append(loss_change)
    matrix = torch.tensor(design, dtype=torch.float64)
    target = torch.tensor(targets, dtype=torch.float64)
    coefficients = torch.linalg.lstsq(matrix, target).solution
    prediction = matrix @ coefficients
    residual_rms = float((prediction - target).square().mean().sqrt())
    hessian = torch.tensor(
        [
            [coefficients[3], coefficients[4]],
            [coefficients[4], coefficients[5]],
        ],
        dtype=torch.float64,
    )
    eigenvalues = torch.linalg.eigvalsh(hessian)
    return {
        "coefficient_names": [
            "intercept",
            "cfc_linear",
            "cproj_linear",
            "cfc_quadratic",
            "cross_quadratic",
            "cproj_quadratic",
        ],
        "coefficients": [float(value) for value in coefficients],
        "hessian_eigenvalues": [float(value) for value in eigenvalues],
        "residual_rms": residual_rms,
    }


def decide_response_surface(
    controls: dict[str, dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    *,
    confidence_z: float,
    registered_maximum_scale: float,
) -> dict[str, Any]:
    production = str(controls["production"]["point_id"])
    comparisons = {
        name: paired_comparison(
            validation_rows,
            str(point["point_id"]),
            production,
            confidence_z,
        )
        for name, point in controls.items()
        if name != "production"
    }
    fixed = controls["fixed_budget"]
    surface = controls["surface"]
    fixed_reliable = bool(comparisons["fixed_budget"]["candidate_reliably_better"])
    common_reliable = bool(comparisons["common_scale"]["candidate_reliably_better"])
    surface_reliable = bool(comparisons["surface"]["candidate_reliably_better"])
    if fixed_reliable and float(fixed["fraction_cfc"]) in {0.0, 1.0}:
        classification = "SINGLE_FAMILY_ALLOCATION_SUPPORTED"
        next_action = "FIX_OR_REPLACE_THE_SUPPRESSED_FAMILY_DIRECTION"
    elif fixed_reliable:
        classification = "RELATIVE_FAMILY_SCALING_SUPPORTED"
        next_action = "IMPLEMENT_CONSTANT_COST_CFC_CPROJ_LR_RATIO"
    elif common_reliable:
        classification = "COMMON_STEP_SCALE_SUPPORTED"
        next_action = "ADJUST_COMMON_CUSTOM_OPTIMIZER_STEP_SCALE"
    elif surface_reliable:
        classification = "COMBINED_SCALE_AND_ALLOCATION_SUPPORTED"
        next_action = "IMPLEMENT_CONSTANT_COST_COMMON_AND_RELATIVE_SCALING"
    else:
        classification = "NO_SCALAR_STEP_COORDINATION_FIX"
        next_action = "TEST_A_TRUE_2X2_BLOCK_OUTPUT_METRIC_CHART"
    return {
        "classification": classification,
        "next_action": next_action,
        "controls": controls,
        "comparisons_to_production": comparisons,
        "surface_selected_at_maximum_scale": math.isclose(
            float(surface["total_scale"]), registered_maximum_scale
        ),
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
    extracted_updates, extracted = extract_production_updates(
        args.checkpoint, config, train_batches, device=args.device, dtype=dtype
    )
    exactness = assert_joint_matches_singletons(
        extracted_updates["cfc_only"]["c_fc"],
        extracted_updates["cproj_only"]["c_proj"],
        extracted_updates["joint"],
    )
    cfc = extracted_updates["cfc_only"]["c_fc"]
    cproj = extracted_updates["cproj_only"]["c_proj"]
    cfc_norm = float(extracted["variants"]["cfc_only"]["update_fro"]["c_fc"])
    cproj_norm = float(
        extracted["variants"]["cproj_only"]["update_fro"]["c_proj"]
    )
    production_fraction = cfc_norm * cfc_norm / (
        cfc_norm * cfc_norm + cproj_norm * cproj_norm
    )
    points = make_surface_points(
        [float(value) for value in protocol["total_scales"]],
        [float(value) for value in protocol["allocation_fractions"]],
        production_fraction,
        cfc_norm,
        cproj_norm,
    )
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

    calibration_rows = evaluate_points(
        model,
        applier,
        windows(
            [int(seed) for seed in protocol["calibration_seeds"]],
            int(protocol["calibration_batches_per_window"]),
        ),
        points,
        device=args.device,
        dtype=dtype,
    )
    controls = select_calibration_controls(
        calibration_rows, points, production_fraction
    )
    unique_validation_points = {
        str(point["point_id"]): {
            key: value
            for key, value in point.items()
            if key != "calibration_mean_loss_change"
        }
        for point in controls.values()
    }
    validation_rows = evaluate_points(
        model,
        applier,
        windows(
            [int(seed) for seed in protocol["validation_seeds"]],
            int(protocol["validation_batches_per_window"]),
        ),
        list(unique_validation_points.values()),
        device=args.device,
        dtype=dtype,
    )
    decision = decide_response_surface(
        controls,
        validation_rows,
        confidence_z=float(plan["decision_rule"]["confidence_z"]),
        registered_maximum_scale=max(
            float(value) for value in protocol["total_scales"]
        ),
    )
    quadratic = fit_quadratic_surface(calibration_rows)
    args.output.mkdir(parents=True, exist_ok=False)
    paths = {
        "calibration_surface": args.output / "calibration_surface.json",
        "validation": args.output / "heldout_validation.json",
        "selection": args.output / "selection_and_decision.json",
        "prospective_step_metadata": args.output / "prospective_step_metadata.json",
    }
    paths["calibration_surface"].write_text(
        json.dumps(calibration_rows, indent=2, sort_keys=True) + "\n"
    )
    paths["validation"].write_text(
        json.dumps(validation_rows, indent=2, sort_keys=True) + "\n"
    )
    paths["selection"].write_text(
        json.dumps(
            {
                "decision": decision,
                "quadratic_fit": quadratic,
                "production_fraction_cfc": production_fraction,
                "cfc_update_fro": cfc_norm,
                "cproj_update_fro": cproj_norm,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    paths["prospective_step_metadata"].write_text(
        json.dumps(extracted, indent=2, sort_keys=True) + "\n"
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "quadratic_fit": quadratic,
        "production_fraction_cfc": production_fraction,
        "cfc_update_fro": cfc_norm,
        "cproj_update_fro": cproj_norm,
        "surface_point_count": len(points),
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
