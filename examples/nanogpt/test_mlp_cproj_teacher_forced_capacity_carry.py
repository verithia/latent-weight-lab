from __future__ import annotations

from examples.nanogpt.analyze_mlp_cproj_teacher_forced_capacity_carry import (
    CANDIDATE,
    CONTROL,
    aggregate_rows,
)


def _row(
    arm: str,
    layer: int,
    step: int,
    recovery: float,
    feedback: float,
    gram_recovery: float,
    requested_recovery: float,
):
    return {
        "arm": arm,
        "layer": layer,
        "score_step": step,
        "chord_energy": 1.0,
        "endpoint_error_energy": 1.0 - recovery,
        "endpoint_recovery": recovery,
        "endpoint_cosine": 1.0,
        "row_gram_chord_energy": 1.0,
        "row_gram_error_energy": 1.0 - gram_recovery,
        "row_gram_recovery": gram_recovery,
        "mean_requested_update_recovery": requested_recovery,
        "terminal_feedback_fro": feedback,
    }


def _rows(candidate_terminal_recovery: float, candidate_feedback: float):
    rows = []
    for step in (60, 120, 180, 238):
        for layer in range(5):
            rows.append(_row(CONTROL, layer, step, 0.8, 10.0, 0.5, -2.0))
            candidate_recovery = (
                candidate_terminal_recovery if step == 238 else 0.81
            )
            rows.append(
                _row(
                    CANDIDATE,
                    layer,
                    step,
                    candidate_recovery,
                    candidate_feedback,
                    0.51,
                    -1.5,
                )
            )
    return rows


def test_hidden112_passes_only_all_registered_rules() -> None:
    result = aggregate_rows(_rows(0.85, 9.0))
    assert result["comparison"]["passed"] is True
    assert result["selected_arm"] == CANDIDATE
    assert result["decision"] == (
        "SELECT_HIDDEN112_FULL_CARRY_FOR_PRODUCTION_PREFLIGHT"
    )


def test_hidden112_rejects_insufficient_feedback_reduction() -> None:
    result = aggregate_rows(_rows(0.85, 9.6))
    assert result["comparison"]["passed"] is False
    assert result["selected_arm"] == CONTROL
