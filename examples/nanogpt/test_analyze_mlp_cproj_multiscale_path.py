from __future__ import annotations

import copy

import pytest
import torch

from examples.nanogpt.analyze_mlp_cproj_multiscale_path import (
    classify,
    polynomial_predict,
    validate_plan,
)


def valid_plan() -> dict:
    return {
        "schema_version": "mai_124m_mlp_cproj_multiscale_path_plan_v1",
        "analysis": {
            "parameter_updates": 0,
            "layers": [8, 9, 10, 11],
            "discovery_steps": [0, 99, 198, 297, 396, 495, 594, 693, 792, 891, 990, 1089, 1188, 1287, 1386, 1485, 1584, 1683, 1782],
            "holdout_steps": [1881, 1980, 2079, 2178, 2277, 2373],
            "ranks": [1, 2, 4, 8, 16],
            "primary_rank": 8,
            "rolling_horizon_indices": [1, 2, 3, 4, 5, 6],
            "polynomial_degree": 2,
            "polynomial_coordinate": "cumulative_learning_rate",
            "learned_basis_role": "diagnostic_oracle_only",
        },
        "decision_rule": {"thresholds": {
            "secant_weight_recovery": 0.80,
            "secant_functional_recovery": 0.80,
            "online_max_updates": 198,
            "phase_max_updates": 594,
            "polynomial_endpoint_weight_recovery": 0.80,
            "polynomial_endpoint_functional_recovery": 0.80,
        }},
        "authorization": {
            "run_zero_update_multiscale_analysis": True,
            "use_learned_basis_in_candidate": False,
            "use_training_time_as_candidate_latent": False,
            "implement_candidate_structure": False,
            "run_exact_config_mfu": False,
            "run_language_model_training": False,
            "larger_rung": False,
        },
    }


def test_plan_is_frozen() -> None:
    validate_plan(valid_plan())
    changed = copy.deepcopy(valid_plan())
    changed["analysis"]["polynomial_degree"] = 3
    with pytest.raises(ValueError):
        validate_plan(changed)


def test_quadratic_control_extrapolates_exact_curve() -> None:
    discovery = torch.linspace(0, 0.75, 10)
    holdout = torch.tensor([0.8, 0.9, 1.0])
    coordinates = torch.stack((2 * discovery + 3 * discovery.square(), -discovery.square()), dim=1)
    predicted = polynomial_predict(coordinates, discovery, holdout, 2)
    expected = torch.stack((2 * holdout + 3 * holdout.square(), -holdout.square()), dim=1)
    torch.testing.assert_close(predicted, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    ("onset", "poly", "expected"),
    [
        (99, True, "ONLINE_LOW_BANDWIDTH_SMOOTH_CURVE"),
        (198, False, "ONLINE_LOW_BANDWIDTH_NONSTATIONARY_CURVE"),
        (396, True, "PHASE_SCALE_SMOOTH_CURVE"),
        (594, False, "PHASE_SCALE_ENVELOPE_ONLY"),
        (None, True, "SCHEDULED_ENDPOINT_CURVE_WITHOUT_LOCAL_SECANT_TRANSPORT"),
        (None, False, "ENDPOINT_ENVELOPE_WITHOUT_CAUSAL_SECANT_TRANSPORT"),
    ],
)
def test_classification(onset: int | None, poly: bool, expected: str) -> None:
    assert classify(onset, poly) == expected
