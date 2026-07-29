from __future__ import annotations

from examples.nanogpt.analyze_mlp_fast_fresh_hidden_capacity_holdout import (
    CANDIDATES,
    aggregate_results,
)


def synthetic_rows(
    *,
    ratio88: float,
    ratio112: float,
    future88: float,
    future112: float,
    positive88: int = 20,
    positive112: int = 20,
) -> tuple[list[dict], list[dict]]:
    values = {
        "dense_exact": 2.0,
        "fresh_hidden64": 1.0,
        "fresh_hidden88": ratio88,
        "fresh_hidden112": ratio112,
    }
    futures = {
        "dense_exact": 1.0,
        "fresh_hidden64": 0.0,
        "fresh_hidden88": future88,
        "fresh_hidden112": future112,
    }
    positives = {
        "dense_exact": 20,
        "fresh_hidden64": 0,
        "fresh_hidden88": positive88,
        "fresh_hidden112": positive112,
    }
    rows = []
    for window in ("fit", "holdout"):
        for cell in range(20):
            for candidate in CANDIDATES:
                rows.append(
                    {
                        "candidate": candidate,
                        "window": window,
                        "coordinates_per_layer": {
                            "dense_exact": 2359296,
                            "fresh_hidden64": 98304,
                            "fresh_hidden88": 135168,
                            "fresh_hidden112": 172032,
                        }[candidate],
                        "current_weight_recovery": values[candidate],
                        "current_weight_energy": 1.0,
                        "current_residual_fixed_scale_recovery": (
                            values[candidate]
                        ),
                        "current_residual_energy": 1.0,
                        "current_output_positive_line_recovery": (
                            values[candidate]
                        ),
                        "current_output_fixed_scale_recovery": (
                            values[candidate]
                        ),
                        "current_output_energy": 1.0,
                        "future_residual_positive_line_recovery": (
                            futures[candidate]
                        ),
                        "future_residual_energy": 1.0,
                        "future_residual_cosine": (
                            1.0
                            if cell < positives[candidate]
                            else -1.0
                        ),
                        "validation_gradient_predicted_ce_decrease": (
                            values[candidate]
                        ),
                        "train_gradient_predicted_ce_decrease": (
                            values[candidate]
                        ),
                    }
                )
    finite = []
    for phase in (30, 90, 150, 210):
        for window in ("fit", "holdout"):
            for candidate in CANDIDATES:
                finite.append(
                    {
                        "base_update": phase,
                        "window": window,
                        "candidate": candidate,
                        "loss": (
                            2.0
                            if candidate == "fresh_hidden64"
                            else 1.9
                        ),
                    }
                )
    return rows, finite


def test_selects_smallest_passing_hidden88() -> None:
    rows, finite = synthetic_rows(
        ratio88=1.16,
        ratio112=1.30,
        future88=0.016,
        future112=0.030,
    )
    result = aggregate_results(rows, finite)
    assert result["passing_depths"] == [88, 112]
    assert result["selected_branch"] == "fresh_hidden88"
    assert result["decision"] == (
        "SELECT_TWO_PASS_FRESH_HIDDEN_FOR_IMPLEMENTATION_PREFLIGHT"
    )


def test_selects_hidden112_only_when_hidden88_fails() -> None:
    rows, finite = synthetic_rows(
        ratio88=1.14,
        ratio112=1.20,
        future88=0.014,
        future112=0.020,
    )
    result = aggregate_results(rows, finite)
    assert result["passing_depths"] == [112]
    assert result["selected_branch"] == "fresh_hidden112"


def test_rejects_insufficient_positive_future_cells() -> None:
    rows, finite = synthetic_rows(
        ratio88=1.20,
        ratio112=1.20,
        future88=0.020,
        future112=0.020,
        positive88=16,
        positive112=16,
    )
    result = aggregate_results(rows, finite)
    assert result["passing_depths"] == []
    assert result["decision"] == "REJECT_SPARSE_FRESH_HIDDEN_DEPTH"
