from __future__ import annotations

import copy

import pytest
import torch

from examples.nanogpt.analyze_mlp_cproj_global_orthogonal_residual import (
    basis_coefficients,
    classify,
    deterministic_orthogonal3,
    global_tensor_fht,
    local_block_fht256,
    support_energy,
    topk_energy,
    validate_plan,
)


def valid_plan() -> dict:
    return {
        "schema_version": "mai_124m_mlp_cproj_5tpp_global_orthogonal_residual_plan_v1",
        "analysis": {
            "parameter_updates": 0,
            "layers": list(range(8)),
            "phases": [[0, 594], [594, 1188], [1188, 1782], [1782, 2373]],
            "coordinate_budget": 147456,
            "orthogonal_seeds": [20260807, 20260808, 20260809],
            "chart": {
                "hidden_parent_stages": 64,
                "hidden_residual_stages": 24,
                "output_stages": 32,
                "neighbors": 64,
                "matching_seed": 20260806,
                "weight_decay_application": "identical production ordering",
            },
        },
        "decision_rule": {
            "thresholds": {
                "global_oracle_recovery_minimum": 0.4,
                "global_over_gaussian_enrichment_minimum": 1.25,
                "global_over_local_ratio_minimum": 1.1,
                "global_over_local_absolute_minimum": 0.02,
                "previous_support_recovery_minimum": 0.1,
                "previous_support_over_random_minimum": 1.5,
                "combined_exact_update_recovery_minimum": 0.6,
            }
        },
        "authorization": {
            "run_zero_update_global_orthogonal_analysis": True,
            "implement_candidate_structure": False,
            "run_exact_config_mfu": False,
            "run_language_model_training": False,
            "larger_rung": False,
        },
    }


def test_plan_is_fail_closed() -> None:
    validate_plan(valid_plan())
    changed = copy.deepcopy(valid_plan())
    changed["analysis"]["chart"]["neighbors"] = 32
    with pytest.raises(ValueError):
        validate_plan(changed)


def test_orthogonal_three_is_deterministic_and_orthogonal() -> None:
    first = deterministic_orthogonal3(7, torch.device("cpu")).double()
    second = deterministic_orthogonal3(7, torch.device("cpu")).double()
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        first.T @ first,
        torch.eye(3, dtype=torch.float64),
        rtol=1e-6,
        atol=1e-6,
    )


def test_local_transform_conserves_energy_and_is_deterministic() -> None:
    values = torch.arange(512, dtype=torch.float32).reshape(16, 32)
    first = local_block_fht256(values, seed=13)
    second = local_block_fht256(values, seed=13)
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert first.double().square().sum() == pytest.approx(
        values.double().square().sum(), rel=1e-6
    )


def test_global_transform_conserves_energy() -> None:
    values = torch.zeros(768, 3072, dtype=torch.float32)
    values[0, 0] = 1.0
    transformed = global_tensor_fht(values, seed=19)
    assert transformed.numel() == values.numel()
    assert transformed.double().square().sum() == pytest.approx(1.0, rel=2e-5)


def test_topk_and_previous_support_are_exact() -> None:
    coefficients = torch.tensor([4.0, 3.0, 0.0, 0.0])
    recovery, support = topk_energy(coefficients, 1)
    assert recovery == pytest.approx(16.0 / 25.0)
    assert support_energy(coefficients, support) == pytest.approx(recovery)
    permuted = torch.tensor([0.0, 4.0, 3.0, 0.0])
    assert support_energy(permuted, support) == pytest.approx(0.0)


def test_basis_dispatch_rejects_unknown_basis() -> None:
    with pytest.raises(ValueError):
        basis_coefficients(torch.ones(16, 16), "unknown", 0)


def test_classification_never_authorizes_implementation_or_training() -> None:
    thresholds = valid_plan()["decision_rule"]["thresholds"]
    metrics = {
        "global_oracle_recovery": 0.5,
        "global_over_gaussian_enrichment": 1.5,
        "global_over_local_ratio": 1.2,
        "global_over_local_absolute": 0.03,
        "global_previous_support_recovery": 0.2,
        "global_previous_support_over_random": 2.0,
        "global_combined_exact_update_recovery": 0.7,
    }
    result = classify(metrics, thresholds)
    assert result["passed"] is True
    assert result["authorization"]["global_orthogonal_structure_theory"] is True
    assert result["authorization"]["implement_candidate_structure"] is False
    assert result["authorization"]["run_language_model_training"] is False
