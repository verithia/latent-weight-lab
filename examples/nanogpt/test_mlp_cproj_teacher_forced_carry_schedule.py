from __future__ import annotations

import math

import torch

from examples.nanogpt.analyze_mlp_cproj_teacher_forced_carry_schedule import (
    ARMS,
    aggregate_rows,
)


def test_registered_decay_schedules() -> None:
    arms = {arm.name: arm for arm in ARMS}
    assert arms["constant_decay0p5"].decay(0, 238) == 0.5
    assert arms["constant_decay1p0"].decay(237, 238) == 1.0
    assert arms["half_switch"].decay(119, 238) == 1.0
    assert arms["half_switch"].decay(120, 238) == 0.5
    assert arms["late_linear"].decay(119, 238) == 1.0
    assert arms["late_linear"].decay(120, 238) == 1.0
    assert 0.5 < arms["late_linear"].decay(237, 238) < 0.51
    assert arms["full_cosine"].decay(0, 238) == 1.0
    assert math.isclose(arms["full_cosine"].decay(237, 238), 0.5)


def _row(arm: str, layer: int, step: int, recovery: float, feedback: float):
    return {
        "arm": arm,
        "layer": layer,
        "score_step": step,
        "chord_energy": 1.0,
        "endpoint_error_energy": 1.0 - recovery,
        "endpoint_recovery": recovery,
        "endpoint_cosine": 1.0,
        "row_gram_chord_energy": 1.0,
        "row_gram_error_energy": 0.5,
        "row_gram_recovery": 0.5,
        "mean_requested_update_recovery": 0.25,
        "terminal_feedback_fro": feedback,
    }


def test_selection_uses_registered_smallest_pass_order() -> None:
    rows = []
    for step in (120, 238):
        for layer in range(5):
            rows.extend(
                [
                    _row("constant_decay0p5", layer, step, 0.5, 1.0),
                    _row("constant_decay1p0", layer, step, 0.8, 4.0),
                    _row("half_switch", layer, step, 0.8 if step == 120 else 0.6, 1.5),
                    _row("late_linear", layer, step, 0.8 if step == 120 else 0.65, 1.0),
                    _row("full_cosine", layer, step, 0.79 if step == 120 else 0.7, 1.0),
                ]
            )
    result = aggregate_rows(rows)
    assert result["comparisons"]["half_switch"]["passed"] is True
    assert result["selected_arm"] == "half_switch"
    assert result["decision"] == "SELECT_HALF_SWITCH_FOR_PRODUCTION_PREFLIGHT"


def test_tensor_dependency_is_available() -> None:
    assert torch.tensor([1.0]).isfinite().all()
