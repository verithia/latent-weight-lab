from __future__ import annotations

import copy

import pytest
import torch

from examples.nanogpt.analyze_mlp_cproj_residual_state_budget import (
    classify,
    intrinsic_rank_dimension,
    largest_rank_within_budget,
    rank_for_energy,
    spectrum_budget_metrics,
    validate_plan,
)


def valid_plan() -> dict:
    return {
        "schema_version": "mai_124m_mlp_cproj_residual_state_budget_plan_v1",
        "analysis": {
            "parameter_updates": 0,
            "layers": list(range(8)),
            "phases": [[0, 594], [594, 1188], [1188, 1782], [1782, 2373]],
            "matrix_shape": [768, 3072],
            "energy_thresholds": [0.5, 0.8, 0.9, 0.95, 0.99],
            "chart_coordinate_budget_per_layer": 147456,
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
                "equal_budget_best_rank_recovery_minimum": 0.8,
                "rank80_intrinsic_dof_ratio_maximum": 0.25,
            }
        },
        "authorization": {
            "run_zero_update_state_budget_analysis": True,
            "implement_candidate_structure": False,
            "run_exact_config_mfu": False,
            "run_language_model_training": False,
            "larger_rung": False,
        },
    }


def test_rank_dimension_and_budget_are_exact() -> None:
    assert intrinsic_rank_dimension(1, 3, 5) == 7
    assert intrinsic_rank_dimension(3, 3, 5) == 15
    assert largest_rank_within_budget(147456, 768, 3072) == 38


def test_rank_for_energy_and_spectrum_metrics() -> None:
    values = torch.tensor([4.0, 3.0, 2.0, 1.0])
    assert rank_for_energy(values, 0.5) == 2
    residual = torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0]))
    metrics = spectrum_budget_metrics(residual, coordinate_budget=6)
    assert metrics["equal_budget_intrinsic_rank"] == 0
    metrics = spectrum_budget_metrics(residual, coordinate_budget=7)
    assert metrics["equal_budget_intrinsic_rank"] == 1
    assert metrics["rank_80pct"] == 2


def test_plan_is_fail_closed() -> None:
    validate_plan(valid_plan())
    changed = copy.deepcopy(valid_plan())
    changed["analysis"]["energy_thresholds"][-1] = 0.98
    with pytest.raises(ValueError):
        validate_plan(changed)


def test_classification_never_authorizes_execution() -> None:
    thresholds = valid_plan()["decision_rule"]["thresholds"]
    result = classify(
        {
            "equal_budget_best_rank_recovery": 0.1,
            "rank80_intrinsic_dof_ratio": 0.8,
        },
        thresholds,
    )
    assert result["classification"] == "EXPLICIT_LOW_RANK_STATE_IS_DENSE_SCALE"
    assert result["authorization"]["task_conditioned_procedural_selector_theory"]
    assert result["authorization"]["implement_candidate_structure"] is False
    assert result["authorization"]["run_language_model_training"] is False
