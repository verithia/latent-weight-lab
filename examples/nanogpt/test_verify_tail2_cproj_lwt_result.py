from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from examples.nanogpt.verify_tail2_cproj_lwt_result import (
    fixed_curve_decision,
    parse_logged_losses,
)


def plan() -> dict:
    return {
        "candidate": {"fixed_evaluation_steps": [10, 20, 30, 40]},
        "decision_rule": {
            "accepted_cfc_only_validation_ce": [4.0, 3.8, 3.7, 3.6],
            "primary_terminal_validation_ce_maximum": 3.605,
            "fixed_curve_gap_to_cfc_only_maximum": 0.01,
        },
    }


def test_parse_and_pass_fixed_curve() -> None:
    text = "\n".join(
        [
            "step 0: train loss 9.0, val loss 9.1",
            "step 10: train loss 4.0, val loss 4.005",
            "step 20: train loss 3.8, val loss 3.805",
            "step 30: train loss 3.7, val loss 3.705",
            "step 40: train loss 3.6, val loss 3.604",
        ]
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "train.log"
        path.write_text(text)
        decision = fixed_curve_decision(plan(), parse_logged_losses(path))
    assert decision["terminal_passed"] is True
    assert decision["curve_passed"] is True
    assert decision["scientific_gate_passed"] is True


def test_decision_fails_terminal_and_curve_independently() -> None:
    logged = {
        10: {"train": 4.0, "validation": 4.011},
        20: {"train": 3.8, "validation": 3.8},
        30: {"train": 3.7, "validation": 3.7},
        40: {"train": 3.6, "validation": 3.606},
    }
    decision = fixed_curve_decision(plan(), logged)
    assert decision["terminal_passed"] is False
    assert decision["curve_passed"] is False
    assert decision["scientific_gate_passed"] is False


def test_missing_or_nonfinite_fixed_loss_fails_closed() -> None:
    with pytest.raises(ValueError, match="lacks fixed evaluations"):
        fixed_curve_decision(plan(), {10: {"train": 4.0, "validation": 4.0}})
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "train.log"
        path.write_text("step 10: train loss nan, val loss 4.0\n")
        with pytest.raises(ValueError, match="non-finite"):
            parse_logged_losses(path)
