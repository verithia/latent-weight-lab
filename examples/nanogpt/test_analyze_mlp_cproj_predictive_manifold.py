from __future__ import annotations

import copy

import pytest
import torch

from examples.nanogpt.analyze_mlp_cproj_predictive_manifold import (
    classify,
    fit_through_origin_basis,
    inverse_givens_flow,
    pooled_recovery,
    project_rows,
    validate_plan,
)
from examples.nanogpt.muon_matched_givens import apply_givens_flow


def valid_plan() -> dict:
    return {
        "schema_version": "mai_124m_mlp_cproj_predictive_manifold_v2_plan_v1",
        "analysis": {
            "parameter_updates": 0,
            "layers": [8, 9, 10, 11],
            "discovery_steps": [0, 99, 198, 297, 396, 495, 594, 693, 792, 891, 990, 1089, 1188, 1287, 1386, 1485, 1584, 1683, 1782],
            "holdout_steps": [1881, 1980, 2079, 2178, 2277, 2373],
            "ranks": [1, 2, 4, 8, 16],
            "learned_basis_role": "diagnostic_oracle_only",
        },
        "decision_rule": {"thresholds": {
            "normalization_schedule_max_relative_error": 0.03,
            "last_step_replay_max_relative_error": 3e-5,
            "rank8_holdout_endpoint_weight_recovery": 0.80,
            "rank8_holdout_endpoint_functional_recovery": 0.80,
            "rank8_holdout_tangent_functional_recovery": 0.25,
            "rank8_endpoint_functional_minus_weight_recovery": 0.10,
        }},
        "authorization": {
            "run_zero_update_predictive_analysis": True,
            "use_learned_basis_in_candidate": False,
            "implement_candidate_structure": False,
            "run_exact_config_mfu": False,
            "run_language_model_training": False,
            "larger_rung": False,
        },
    }


def test_plan_is_frozen_and_learned_basis_is_diagnostic_only() -> None:
    validate_plan(valid_plan())
    changed = copy.deepcopy(valid_plan())
    changed["analysis"]["learned_basis_role"] = "candidate"
    with pytest.raises(ValueError):
        validate_plan(changed)


def test_inverse_flow_round_trip() -> None:
    torch.manual_seed(17)
    values = torch.randn(5, 8)
    permutations = torch.stack([torch.randperm(8) for _ in range(3)])
    inverse = torch.argsort(permutations, dim=1)
    angles = torch.randn(3, 4) * 0.1
    transformed = apply_givens_flow(values, angles, permutations, inverse)
    recovered = inverse_givens_flow(transformed, angles, permutations, inverse)
    torch.testing.assert_close(recovered, values, rtol=1e-5, atol=1e-6)


def test_compact_gram_basis_recovers_known_rank_two_rows() -> None:
    torch.manual_seed(19)
    basis = torch.linalg.qr(torch.randn(30, 2), mode="reduced").Q.T
    rows = torch.randn(8, 2) @ basis
    fitted = fit_through_origin_basis(rows, 2)
    estimate = project_rows(rows, fitted)
    assert pooled_recovery(rows, estimate) > 0.99999


@pytest.mark.parametrize(
    ("gates", "expected"),
    [
        ({"normalization_schedule_valid": False, "last_step_replay_valid": True, "endpoint_weight_predictive": True, "endpoint_function_predictive": True, "tangent_function_predictive": True}, "INVALID_MECHANICAL_RECONSTRUCTION"),
        ({"normalization_schedule_valid": True, "last_step_replay_valid": True, "endpoint_weight_predictive": True, "endpoint_function_predictive": True, "tangent_function_predictive": False}, "SMOOTH_ENDPOINT_WITH_NONTRANSPORTABLE_TANGENTS"),
        ({"normalization_schedule_valid": True, "last_step_replay_valid": True, "endpoint_weight_predictive": False, "endpoint_function_predictive": True, "tangent_function_predictive": False}, "FUNCTION_SPACE_ENDPOINT_WITHOUT_WEIGHT_CHART"),
        ({"normalization_schedule_valid": True, "last_step_replay_valid": True, "endpoint_weight_predictive": False, "endpoint_function_predictive": False, "tangent_function_predictive": False}, "NO_PREDICTIVE_FIXED_AFFINE_CHART"),
    ],
)
def test_classification(gates: dict[str, bool], expected: str) -> None:
    assert classify(gates) == expected
