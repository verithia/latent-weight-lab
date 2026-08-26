from __future__ import annotations

from examples.nanogpt.analyze_mlp_raw_gradient_rolling_prediction import (
    phase_for_step,
    summarize_rows,
)


def test_phase_for_step_uses_inclusive_boundaries() -> None:
    assert phase_for_step(119, 119, 179) == "discovery"
    assert phase_for_step(120, 119, 179) == "validation"
    assert phase_for_step(179, 119, 179) == "validation"
    assert phase_for_step(180, 119, 179) == "test"


def test_summarize_rows_reports_energy_weighted_capture() -> None:
    base = {
        "parameter": "transformer.h.6.mlp.c_fc.weight",
        "target": "mlp.c_fc",
        "rank": 1,
        "eval_phase": "test",
        "left_capture": 0.0,
        "right_capture": 0.0,
        "bilinear_core_capture": 0.0,
        "left_current_overlap_mean_squared_cosine": 0.2,
        "right_current_overlap_mean_squared_cosine": 0.4,
    }
    rows = [
        {
            **base,
            "eval_step": 10,
            "direction_energy": 1.0,
            "rank_manifold_tangent_capture": 0.0,
        },
        {
            **base,
            "eval_step": 12,
            "direction_energy": 3.0,
            "rank_manifold_tangent_capture": 1.0,
        },
    ]
    summaries = summarize_rows(rows)
    test = next(row for row in summaries if row["eval_phase"] == "test")
    assert test["sample_count"] == 2
    assert abs(test["rank_manifold_tangent_capture_energy_weighted_mean"] - 0.75) < 1e-12
    assert abs(test["rank_manifold_tangent_capture_median"] - 0.5) < 1e-12
