from __future__ import annotations

from examples.nanogpt.audit_late_cproj_paired_state_curve_rejection import (
    fixed_curve_rows,
)


def test_fixed_curve_rows_freeze_signed_and_absolute_deltas() -> None:
    plan = {
        "acceptance": {
            "curve_absolute_tolerance_ce": 0.005,
            "accepted_validation_ce_by_step": {
                "594": 4.2239,
                "1188": 3.8510,
            },
        }
    }
    rows = fixed_curve_rows(
        plan,
        {
            594: {"val": 4.2227},
            1188: {"val": 3.8596},
        },
    )
    assert [row["step"] for row in rows] == [594, 1188]
    assert rows[0]["within_tolerance"] is True
    assert rows[0]["delta_ce"] < 0
    assert rows[1]["within_tolerance"] is False
    assert abs(rows[1]["absolute_delta_ce"] - 0.0086) < 1e-12
