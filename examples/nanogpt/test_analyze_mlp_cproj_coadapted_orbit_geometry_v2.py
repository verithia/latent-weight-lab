from __future__ import annotations

import copy

import pytest

from examples.nanogpt.analyze_mlp_cproj_coadapted_orbit_geometry_v2 import (
    classify,
    validate_plan,
)


def valid_plan() -> dict:
    return {
        "schema_version": "mai_124m_mlp_cproj_coadapted_orbit_geometry_v2_plan_v1",
        "analysis": {
            "parameter_updates": 0,
            "late_layers": [8, 9, 10, 11],
            "phase_steps": [0, 594, 1188, 1782, 2373],
            "cross_run_parameter_displacements": False,
        },
        "decision_rule": {
            "thresholds": {
                "procedural_minimum_orbit_recovery": 0.995,
                "late_minus_early_orbit_recovery": 0.05,
                "late_to_early_gram_drift_ratio": 0.8,
                "reusable_support_retention_fraction": 0.10,
                "reusable_support_enrichment": 2.0,
            }
        },
        "authorization": {
            "run_gauge_invariant_zero_update_v2": True,
            "implement_candidate_structure": False,
            "run_exact_config_mfu": False,
            "run_language_model_training": False,
            "larger_rung": False,
        },
    }


def test_plan_is_fail_closed_and_prohibits_cross_run_displacement() -> None:
    validate_plan(valid_plan())
    changed = copy.deepcopy(valid_plan())
    changed["analysis"]["cross_run_parameter_displacements"] = True
    with pytest.raises(ValueError):
        validate_plan(changed)


@pytest.mark.parametrize(
    ("gates", "expected"),
    [
        (
            {
                "accepted_path_is_scaled_right_orbit": False,
                "right_orbit_localizes_late_band": False,
                "support_is_reusable": False,
            },
            "ACCEPTED_PATH_EXCEEDS_SCALED_RIGHT_ORBIT",
        ),
        (
            {
                "accepted_path_is_scaled_right_orbit": True,
                "right_orbit_localizes_late_band": True,
                "support_is_reusable": True,
            },
            "RIGHT_ORBIT_LOCALIZES_LWT_WITH_REUSABLE_SUPPORT",
        ),
        (
            {
                "accepted_path_is_scaled_right_orbit": True,
                "right_orbit_localizes_late_band": True,
                "support_is_reusable": False,
            },
            "RIGHT_ORBIT_LOCALIZES_LWT_BUT_SUPPORT_IS_MOVING",
        ),
        (
            {
                "accepted_path_is_scaled_right_orbit": True,
                "right_orbit_localizes_late_band": False,
                "support_is_reusable": False,
            },
            "ADAPTIVE_RIGHT_ORBIT_WITH_MOVING_SUPPORT_NOT_LAYER_LOCALIZED",
        ),
    ],
)
def test_classification_is_frozen(gates: dict[str, bool], expected: str) -> None:
    assert classify(gates) == expected
