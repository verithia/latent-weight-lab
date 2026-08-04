from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest

from examples.nanogpt.fixed_model_compute_equivalence import (
    EvalPoint,
    common_loss_ratios,
    invert_monotone_curve,
    local_slope,
    parse_eval_log,
    terminal_dense_penalty,
)


def test_parse_log_uses_last_value_per_positive_step() -> None:
    with tempfile.TemporaryDirectory() as raw:
        path = Path(raw) / "train.log"
        path.write_text(
            "step 0: train loss 9.0, val loss 9.1\n"
            "step 10: train loss 4.2, val loss 4.3\n"
            "step 10: train loss 4.0, val loss 4.1\n"
            "step 20: train loss 3.7, val loss 3.8\n"
        )
        assert parse_eval_log(path) == [EvalPoint(10.0, 4.1), EvalPoint(20.0, 3.8)]


def test_power_law_recovers_total_loss_alpha() -> None:
    alpha = 0.12
    points = [EvalPoint(float(c), 5.0 * c ** (-alpha)) for c in (10, 20, 40, 80)]
    fit = local_slope(points, 4)
    # L itself is not affine in ln(C), so alpha_eff is local/window dependent.
    assert fit["loss_slope_per_log_compute"] > 0
    assert 0.09 < fit["alpha_eff_total_loss"] < 0.16


def test_log_linear_curve_interpolation_and_terminal_penalty() -> None:
    dense = [EvalPoint(10, 4.0), EvalPoint(20, 3.5), EvalPoint(40, 3.0)]
    target = 3.25
    matched, method = invert_monotone_curve(dense, target)
    assert method == "interpolation"
    assert matched == pytest.approx(math.sqrt(20 * 40))
    candidate = [EvalPoint(10, 4.2), EvalPoint(40, target)]
    penalty = terminal_dense_penalty(dense, candidate)
    assert penalty["candidate_over_dense_compute"] == pytest.approx(
        math.sqrt(2.0)
    )


def test_common_loss_ratio_recovers_horizontal_shift() -> None:
    dense = [EvalPoint(10, 4.0), EvalPoint(20, 3.5), EvalPoint(40, 3.0)]
    candidate = [EvalPoint(20, 4.0), EvalPoint(40, 3.5), EvalPoint(80, 3.0)]
    result = common_loss_ratios(dense, candidate)
    assert result["median_candidate_over_dense_compute"] == pytest.approx(2.0)
    assert result["min_candidate_over_dense_compute"] == pytest.approx(2.0)
    assert result["max_candidate_over_dense_compute"] == pytest.approx(2.0)


def test_rejects_extrapolation_beyond_one_adjacent_interval() -> None:
    dense = [EvalPoint(10, 4.0), EvalPoint(20, 3.5), EvalPoint(40, 3.0)]
    with pytest.raises(ValueError, match="one-adjacent-interval"):
        invert_monotone_curve(dense, 2.0, allow_one_interval_extrapolation=True)
