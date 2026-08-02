from __future__ import annotations

from examples.nanogpt.test_validate_functional_shear_stability_log import row, text
from examples.nanogpt.validate_functional_shear_stability_log_v2 import validate


def run(rows: list[dict]) -> dict:
    for value in rows:
        value.setdefault(
            "functional_fit_condition_projection_min_scale",
            0.5 if value["functional_fit_condition_projection_active"] else 1.0,
        )
    return validate(
        text(rows),
        expected_layers=2,
        expected_steps=2,
        maximum_condition_number=1.01,
        maximum_weight_rms_ratio=2.0,
        maximum_weight_abs_growth=2.0,
        maximum_weight_abs_floor=1.0,
    )


def test_safe_noop_row_passes_when_other_rows_exercise_limiter() -> None:
    rows = [row(step, layer) for step in range(2) for layer in range(2)]
    rows[0]["functional_fit_condition_projection_active"] = False
    rows[0]["functional_fit_condition_projection_min_scale"] = 1.0
    result = run(rows)
    assert result["passed"] is True
    assert result["observed"]["internal_limiter_rows"] == 3


def test_never_exercised_limiter_rejects() -> None:
    rows = [row(step, layer) for step in range(2) for layer in range(2)]
    for value in rows:
        value["functional_fit_condition_projection_active"] = False
        value["functional_fit_condition_projection_min_scale"] = 1.0
    result = run(rows)
    assert result["passed"] is False
    assert "never exercised" in " ".join(result["failures"])


def test_activity_must_agree_with_projection_scale() -> None:
    rows = [row(step, layer) for step in range(2) for layer in range(2)]
    rows[0]["functional_fit_condition_projection_min_scale"] = 1.0
    result = run(rows)
    assert result["passed"] is False
    assert "activity/scale disagreed" in " ".join(result["failures"])
