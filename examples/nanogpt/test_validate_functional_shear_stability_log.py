from __future__ import annotations

import json
import math

from examples.nanogpt.validate_functional_shear_stability_log import validate


def row(step: int, layer: int) -> dict:
    limit = math.log(1.01)
    return {
        "optimizer": "muon_functional_shear",
        "step": step,
        "layer": layer,
        "coordinate_finite_fraction": 1.0,
        "functional_fit_coordinate_finite_fraction": 1.0,
        "functional_recipe_finite_fraction": 1.0,
        "update_finite_fraction": 1.0,
        "weight_recipe_finite_fraction": 1.0,
        "functional_fit_context_finite": True,
        "functional_fit_condition_projection_active": True,
        "functional_fallback_to_weight_recipe": False,
        "maximum_condition_number": 1.01,
        "functional_fit_log_condition_bound": limit,
        "mixed_log_condition_bound_after_projection": limit,
        "weight_rms_before": 0.02,
        "weight_rms_after": 0.019,
        "weight_max_abs_before": 0.1,
        "weight_max_abs_after": 0.11,
    }


def text(rows: list[dict]) -> str:
    lines = [
        "muon_matched_givens_refresh " + json.dumps(value, sort_keys=True)
        for value in rows
    ]
    lines.extend(f"iter {step}: loss {8.0 - step / 10:.4f}" for step in range(2))
    return "\n".join(lines) + "\n"


def run(rows: list[dict]) -> dict:
    return validate(
        text(rows),
        expected_layers=2,
        expected_steps=2,
        maximum_condition_number=1.01,
        maximum_weight_rms_ratio=2.0,
        maximum_weight_abs_growth=2.0,
        maximum_weight_abs_floor=1.0,
    )


def test_exact_finite_grid_passes() -> None:
    result = run([row(step, layer) for step in range(2) for layer in range(2)])
    assert result["passed"] is True
    assert result["observed"]["rows"] == 4
    assert result["observed"]["internal_limiter_rows"] == 4


def test_missing_row_and_duplicate_fail_closed() -> None:
    rows = [row(0, 0), row(0, 1), row(1, 0), row(1, 0)]
    result = run(rows)
    assert result["passed"] is False
    assert "exact registered step/layer grid" in " ".join(result["failures"])
    assert "duplicate" in " ".join(result["failures"])


def test_inactive_limiter_fallback_and_growth_reject() -> None:
    rows = [row(step, layer) for step in range(2) for layer in range(2)]
    rows[0]["functional_fit_condition_projection_active"] = False
    rows[1]["functional_fallback_to_weight_recipe"] = True
    rows[2]["mixed_log_condition_bound_after_projection"] = math.log(1.02)
    rows[3]["weight_rms_after"] = 0.05
    result = run(rows)
    assert result["passed"] is False
    assert result["observed"]["internal_limiter_rows"] == 3
    assert result["observed"]["fallback_rows"] == 1
    assert result["observed"]["condition_bound_violations"] == 1
    assert result["observed"]["weight_growth_violations"] == 1


def test_nonfinite_diagnostic_and_loss_reject() -> None:
    rows = [row(step, layer) for step in range(2) for layer in range(2)]
    rows[0]["weight_rms_after"] = float("nan")
    payload = text(rows).replace("iter 1: loss 7.9000", "iter 1: loss nan")
    result = validate(
        payload,
        expected_layers=2,
        expected_steps=2,
        maximum_condition_number=1.01,
        maximum_weight_rms_ratio=2.0,
        maximum_weight_abs_growth=2.0,
        maximum_weight_abs_floor=1.0,
    )
    assert result["passed"] is False
    assert result["observed"]["finite_rows"] == 3
    assert result["observed"]["finite_loss_values"] == 1
