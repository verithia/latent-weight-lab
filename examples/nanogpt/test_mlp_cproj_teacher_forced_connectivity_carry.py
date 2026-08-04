from __future__ import annotations

from examples.nanogpt.analyze_mlp_cproj_teacher_forced_connectivity_carry import (
    aggregate_rows,
)


def _row(arm: str, layer: int, step: int, error: float, feedback: float):
    return {
        "arm": arm,
        "layer": layer,
        "score_step": step,
        "chord_energy": 1.0,
        "endpoint_error_energy": error,
        "endpoint_recovery": 1.0 - error,
        "endpoint_cosine": 1.0,
        "row_gram_chord_energy": 1.0,
        "row_gram_error_energy": 0.5,
        "row_gram_recovery": 0.5,
        "mean_requested_update_recovery": -2.0 if arm.endswith("64_full_carry") else -1.0,
        "terminal_feedback_fro": feedback,
    }


def _rows(candidate128_error: float, candidate128_feedback: float):
    rows = []
    for step in (60, 120, 180, 238):
        for layer in range(5):
            rows.extend(
                [
                    _row("neighbors64_full_carry", layer, step, 0.2, 10.0),
                    _row(
                        "neighbors128_full_carry",
                        layer,
                        step,
                        candidate128_error,
                        candidate128_feedback,
                    ),
                    _row("neighbors256_full_carry", layer, step, 0.1, 8.0),
                ]
            )
    return rows


def test_selects_smallest_passing_radius() -> None:
    result = aggregate_rows(_rows(0.16, 9.0))
    assert result["comparisons"]["neighbors128_full_carry"]["passed"] is True
    assert result["selected_arm"] == "neighbors128_full_carry"


def test_falls_through_to_larger_radius() -> None:
    result = aggregate_rows(_rows(0.18, 9.6))
    assert result["comparisons"]["neighbors128_full_carry"]["passed"] is False
    assert result["comparisons"]["neighbors256_full_carry"]["passed"] is True
    assert result["selected_arm"] == "neighbors256_full_carry"
