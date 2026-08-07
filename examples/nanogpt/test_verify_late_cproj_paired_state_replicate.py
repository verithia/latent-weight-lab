from __future__ import annotations

from examples.nanogpt.verify_late_cproj_paired_state_replicate import (
    prediction_interval_rows,
)


def test_prediction_interval_rows_pass_and_fail_closed() -> None:
    plan = {
        "acceptance": {
            "replicate_prediction_intervals_by_step": {
                "594": {"lower": 4.21, "upper": 4.23},
                "1188": {"lower": 3.84, "upper": 3.86},
            }
        }
    }
    rows = prediction_interval_rows(
        plan,
        {594: {"val": 4.22}, 1188: {"val": 3.87}},
    )
    assert rows[0]["inside_interval"] is True
    assert rows[1]["inside_interval"] is False
