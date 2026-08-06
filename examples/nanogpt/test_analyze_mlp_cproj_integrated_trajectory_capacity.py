from __future__ import annotations

import copy

import pytest
import torch

from examples.nanogpt.analyze_mlp_cproj_integrated_trajectory_capacity import (
    classify,
    geometric_metrics,
    validate_plan,
)


def valid_plan() -> dict:
    return {
        "schema_version": "mai_124m_mlp_cproj_5tpp_integrated_trajectory_capacity_plan_v1",
        "analysis": {
            "parameter_updates": 0,
            "layers": list(range(8)),
            "phases": [[0, 594], [594, 1188], [1188, 1782], [1782, 2373]],
            "straight_chord_substeps": [1, 8, 32],
            "chart": {
                "hidden_parent_stages": 64,
                "hidden_residual_stages": 24,
                "output_stages": 32,
                "neighbors": 64,
                "matching_seed": 20260807,
                "feedback": "full within each oracle path",
                "weight_decay": 0.0,
                "learning_rate": 1.0,
            },
            "fixed_validation": {
                "split": "validation",
                "eval_iters": 400,
                "eval_batch_size": 16,
                "block_size": 1024,
                "eval_seed": 20260715,
                "fixed_eval_indices_sha256": "5ca31b59768e43de808ad5e206ed152a4a0a3515ad68d29a0b2338c4db140747",
            },
        },
        "decision_rule": {
            "thresholds": {
                "phase_straight32_geometric_recovery_minimum": 0.85,
                "phase_straight32_maximum_validation_ce_gap": 0.01,
                "phase_straight32_terminal_validation_ce_gap": 0.005,
                "sequential_straight32_terminal_validation_ce_gap": 0.02,
                "straight32_terminal_ce_improvement_over_direct_minimum": 0.01,
            }
        },
        "authorization": {
            "run_zero_update_integrated_trajectory_analysis": True,
            "acquire_higher_cadence_trajectory": False,
            "implement_candidate_structure": False,
            "run_exact_config_mfu": False,
            "run_language_model_training": False,
            "larger_rung": False,
        },
    }


def test_plan_is_fail_closed() -> None:
    validate_plan(valid_plan())
    changed = copy.deepcopy(valid_plan())
    changed["analysis"]["straight_chord_substeps"] = [1, 16, 32]
    with pytest.raises(ValueError):
        validate_plan(changed)


def test_geometric_metrics_are_exact() -> None:
    start = torch.zeros(2, 2)
    dense = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    exact = geometric_metrics(start, dense, dense)
    assert exact["endpoint_recovery"] == pytest.approx(1.0)
    assert exact["endpoint_cosine"] == pytest.approx(1.0)
    half = geometric_metrics(start, dense, dense * 0.5)
    assert half["endpoint_recovery"] == pytest.approx(0.75)
    assert half["endpoint_cosine"] == pytest.approx(1.0)


def test_classification_separates_capacity_and_transport() -> None:
    thresholds = valid_plan()["decision_rule"]["thresholds"]
    metrics = {
        "phase_straight32_geometric_recovery": 0.9,
        "phase_straight32_maximum_validation_ce_gap": 0.008,
        "phase_straight32_terminal_validation_ce_gap": 0.004,
        "sequential_straight32_terminal_validation_ce_gap": 0.03,
        "straight32_terminal_ce_improvement_over_direct": 0.02,
    }
    result = classify(metrics, thresholds)
    assert result["classification"] == "MATURE_PHASE_CAPACITY_PASS_TRANSPORT_FAIL"
    assert result["authorization"]["acquire_higher_cadence_trajectory"] is True
    assert result["authorization"]["run_language_model_training"] is False
    metrics["sequential_straight32_terminal_validation_ce_gap"] = 0.01
    result = classify(metrics, thresholds)
    assert result["classification"] == "MATURE_PHASE_CAPACITY_AND_COARSE_TRANSPORT_PASS"
    assert result["authorization"]["candidate_structure_theory"] is True
    assert result["authorization"]["implement_candidate_structure"] is False
