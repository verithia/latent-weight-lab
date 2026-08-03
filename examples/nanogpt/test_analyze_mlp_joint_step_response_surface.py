from __future__ import annotations

import math

from examples.nanogpt.analyze_mlp_joint_step_response_surface import (
    allocation_coefficients,
    decide_response_surface,
)


def test_production_fraction_recovers_unit_coefficients() -> None:
    cfc_norm = 3.0
    cproj_norm = 4.0
    fraction = cfc_norm**2 / (cfc_norm**2 + cproj_norm**2)
    cfc_scale, cproj_scale = allocation_coefficients(
        fraction, 1.0, cfc_norm, cproj_norm
    )
    assert math.isclose(cfc_scale, 1.0)
    assert math.isclose(cproj_scale, 1.0)


def _point(name: str, fraction: float, scale: float):
    return {
        "point_id": name,
        "fraction_cfc": fraction,
        "total_scale": scale,
        "cfc_scale": 1.0,
        "cproj_scale": 1.0,
        "calibration_mean_loss_change": -0.01,
    }


def _rows(losses: dict[str, float]):
    rows = []
    for window in ("window_1", "window_2"):
        for index in range(8):
            for point, loss in losses.items():
                rows.append(
                    {
                        "window": window,
                        "batch_index": index,
                        "point_id": point,
                        "ce": loss,
                    }
                )
    return rows


def test_heldout_decision_prefers_reliable_fixed_budget_ratio() -> None:
    controls = {
        "production": _point("production", 0.8, 1.0),
        "fixed_budget": _point("fixed", 0.7, 1.0),
        "common_scale": _point("common", 0.8, 1.25),
        "surface": _point("surface", 0.7, 1.25),
        "axis": _point("axis", 1.0, 1.0),
    }
    decision = decide_response_surface(
        controls,
        _rows(
            {
                "production": 5.0,
                "fixed": 4.99,
                "common": 5.001,
                "surface": 4.995,
                "axis": 5.01,
            }
        ),
        confidence_z=2.576,
    )
    assert decision["classification"] == "RELATIVE_FAMILY_SCALING_SUPPORTED"
    assert decision["next_action"] == "IMPLEMENT_CONSTANT_COST_CFC_CPROJ_LR_RATIO"
