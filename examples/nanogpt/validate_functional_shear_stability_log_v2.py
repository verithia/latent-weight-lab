#!/usr/bin/env python3
"""Fail-closed V2 functional-shear gate with correct clip-activity semantics."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from examples.nanogpt.validate_functional_shear_stability_log import (
    all_numbers_finite,
    atomic_json,
    parse_functional_rows,
    parse_losses,
)


def validate(
    text: str,
    *,
    expected_layers: int,
    expected_steps: int,
    maximum_condition_number: float,
    maximum_weight_rms_ratio: float,
    maximum_weight_abs_growth: float,
    maximum_weight_abs_floor: float,
) -> dict[str, Any]:
    rows = parse_functional_rows(text)
    losses = parse_losses(text)
    failures: list[str] = []
    expected_coordinates = {
        (step, layer)
        for step in range(expected_steps)
        for layer in range(expected_layers)
    }
    observed_coordinates = {(row.get("step"), row.get("layer")) for row in rows}
    expected_count = expected_layers * expected_steps
    if len(rows) != expected_count:
        failures.append(f"expected {expected_count} functional rows, observed {len(rows)}")
    if observed_coordinates != expected_coordinates:
        failures.append("functional rows do not form the exact registered step/layer grid")
    if len(observed_coordinates) != len(rows):
        failures.append("duplicate functional step/layer rows")
    if len(losses) < expected_steps or any(not math.isfinite(value) for value in losses):
        failures.append("training/evaluation losses are missing or nonfinite")

    log_limit = math.log(maximum_condition_number)
    finite_rows = sum(all_numbers_finite(row) for row in rows)
    internal_limiter_rows = sum(
        row.get("functional_fit_condition_projection_active") is True for row in rows
    )
    fallback_rows = sum(
        row.get("functional_fallback_to_weight_recipe") is True for row in rows
    )
    bound_violations = 0
    weight_growth_violations = 0
    for row in rows:
        if not all_numbers_finite(row):
            continue
        if row.get("functional_fit_context_finite") is not True:
            failures.append("functional fit context was nonfinite")
        for field in (
            "coordinate_finite_fraction",
            "functional_fit_coordinate_finite_fraction",
            "functional_recipe_finite_fraction",
            "update_finite_fraction",
            "weight_recipe_finite_fraction",
        ):
            if float(row.get(field, float("nan"))) != 1.0:
                failures.append(f"{field} was not exactly one")
        if row.get("functional_fallback_to_weight_recipe") is not False:
            failures.append("functional recipe fell back to the weight recipe")
        if not math.isclose(
            float(row.get("maximum_condition_number", float("nan"))),
            maximum_condition_number,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            failures.append("row maximum-condition contract drifted")
        minimum_scale = float(
            row.get("functional_fit_condition_projection_min_scale", float("nan"))
        )
        active = row.get("functional_fit_condition_projection_active") is True
        if not (0.0 <= minimum_scale <= 1.0):
            failures.append("internal condition projection scale was outside [0,1]")
        if active != (minimum_scale < 1.0):
            failures.append("internal condition projection activity/scale disagreed")
        if (
            float(row.get("functional_fit_log_condition_bound", float("inf")))
            > log_limit + 1e-12
            or float(row.get("mixed_log_condition_bound_after_projection", float("inf")))
            > log_limit + 1e-12
        ):
            bound_violations += 1
        before_rms = float(row.get("weight_rms_before", float("nan")))
        after_rms = float(row.get("weight_rms_after", float("nan")))
        before_abs = float(row.get("weight_max_abs_before", float("nan")))
        after_abs = float(row.get("weight_max_abs_after", float("nan")))
        if (
            before_rms <= 0.0
            or after_rms <= 0.0
            or after_rms / before_rms > maximum_weight_rms_ratio
            or after_rms / before_rms < 1.0 / maximum_weight_rms_ratio
            or after_abs > max(maximum_weight_abs_floor, maximum_weight_abs_growth * before_abs)
        ):
            weight_growth_violations += 1
    if finite_rows != len(rows):
        failures.append("one or more functional diagnostic rows contain nonfinite numbers")
    if internal_limiter_rows < 1:
        failures.append("the internal condition limiter was never exercised")
    if fallback_rows:
        failures.append("one or more functional rows used fail-closed fallback")
    if bound_violations:
        failures.append("one or more functional rows exceeded the registered condition bound")
    if weight_growth_violations:
        failures.append("one or more functional rows exceeded the registered weight-growth bound")

    return {
        "schema_version": "functional_shear_stability_validation_v2",
        "decision": "PASS" if not failures else "REJECT",
        "passed": not failures,
        "registered_contract": {
            "expected_layers": expected_layers,
            "expected_steps": expected_steps,
            "expected_rows": expected_count,
            "maximum_condition_number": maximum_condition_number,
            "maximum_log_condition": log_limit,
            "maximum_weight_rms_ratio": maximum_weight_rms_ratio,
            "maximum_weight_abs_growth": maximum_weight_abs_growth,
            "maximum_weight_abs_floor": maximum_weight_abs_floor,
            "require_internal_limiter_at_least_one_row": True,
            "require_zero_fallback": True,
        },
        "observed": {
            "rows": len(rows),
            "unique_step_layer_coordinates": len(observed_coordinates),
            "finite_rows": finite_rows,
            "internal_limiter_rows": internal_limiter_rows,
            "fallback_rows": fallback_rows,
            "condition_bound_violations": bound_violations,
            "weight_growth_violations": weight_growth_violations,
            "loss_values": len(losses),
            "finite_loss_values": sum(math.isfinite(value) for value in losses),
        },
        "failures": sorted(set(failures)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-layers", type=int, default=24)
    parser.add_argument("--expected-steps", type=int, default=25)
    parser.add_argument("--maximum-condition-number", type=float, default=1.01)
    parser.add_argument("--maximum-weight-rms-ratio", type=float, default=2.0)
    parser.add_argument("--maximum-weight-abs-growth", type=float, default=2.0)
    parser.add_argument("--maximum-weight-abs-floor", type=float, default=1.0)
    args = parser.parse_args()
    if args.expected_layers < 1 or args.expected_steps < 1:
        parser.error("expected layers and steps must be positive")
    if args.maximum_condition_number <= 1.0:
        parser.error("maximum condition number must exceed one")
    if args.maximum_weight_rms_ratio < 1.0 or args.maximum_weight_abs_growth < 1.0:
        parser.error("weight-growth multipliers must be at least one")
    result = validate(
        args.log.read_text(errors="replace"),
        expected_layers=args.expected_layers,
        expected_steps=args.expected_steps,
        maximum_condition_number=args.maximum_condition_number,
        maximum_weight_rms_ratio=args.maximum_weight_rms_ratio,
        maximum_weight_abs_growth=args.maximum_weight_abs_growth,
        maximum_weight_abs_floor=args.maximum_weight_abs_floor,
    )
    atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True), flush=True)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
