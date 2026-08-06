from __future__ import annotations

import copy

import pytest
import torch

import examples.nanogpt.analyze_mlp_cproj_bounded_integrated_trajectory as module
from examples.nanogpt.analyze_mlp_cproj_bounded_integrated_trajectory import (
    bounded_structured_step,
    classify,
    validate_plan,
)


def valid_plan() -> dict:
    return {
        "schema_version": (
            "mai_124m_mlp_cproj_5tpp_bounded_integrated_trajectory_plan_v1"
        ),
        "analysis": {
            "parameter_updates": 0,
            "layers": list(range(8)),
            "phases": [[0, 594], [594, 1188], [1188, 1782], [1782, 2373]],
            "straight_chord_substeps": 8,
            "variants": {
                "phase_zero_feedback": {"feedback_decay": 0.0},
                "phase_decay0p5": {"feedback_decay": 0.5},
                "sequential_decay0p5": {"feedback_decay": 0.5},
            },
            "chart": {
                "hidden_parent_stages": 64,
                "hidden_residual_stages": 24,
                "output_stages": 32,
                "neighbors": 64,
                "matching_seed": 20260807,
                "weight_decay": 0.0,
                "learning_rate": 1.0,
            },
            "fixed_validation": {
                "split": "validation",
                "eval_iters": 400,
                "eval_batch_size": 16,
                "block_size": 1024,
                "eval_seed": 20260715,
                "fixed_eval_indices_sha256": (
                    "5ca31b59768e43de808ad5e206ed152a4a0a3515ad68d29a0b2338c4db140747"
                ),
            },
        },
        "decision_rule": {
            "thresholds": {
                "phase_decay0p5_geometric_recovery_minimum": 0.55,
                "phase_decay0p5_minimum_layer_phase_recovery": 0.0,
                "phase_decay0p5_maximum_validation_ce_gap": 0.01,
                "phase_decay0p5_terminal_validation_ce_gap": 0.005,
                "sequential_decay0p5_terminal_validation_ce_gap": 0.02,
                "phase_decay0p5_maximum_feedback_fro": 2.6153,
            }
        },
        "authorization": {
            "run_zero_update_bounded_feedback_analysis": True,
            "implement_candidate_structure": False,
            "run_exact_config_mfu": False,
            "run_language_model_training": False,
            "larger_rung": False,
        },
    }


def test_plan_is_fail_closed() -> None:
    validate_plan(valid_plan())
    changed = copy.deepcopy(valid_plan())
    changed["analysis"]["straight_chord_substeps"] = 16
    with pytest.raises(ValueError):
        validate_plan(changed)


def test_feedback_decay_is_applied_before_chart(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "fit_right_pass", lambda weight, *_args, **_kwargs: weight)
    monkeypatch.setattr(module, "fit_output_pass", lambda weight, *_args, **_kwargs: weight)
    weight = torch.zeros(2, 2)
    requested = torch.ones(2, 2)
    feedback = torch.full((2, 2), 2.0)
    _updated, new_feedback, _recovery = bounded_structured_step(
        weight,
        requested,
        feedback,
        feedback_decay=0.5,
        output_stages=32,
        learning_rate=1.0,
        weight_decay=0.0,
        neighbors=1,
        seed=0,
    )
    assert torch.equal(new_feedback, torch.full((2, 2), 2.0))


def test_classification_requires_phase_capacity_and_transport() -> None:
    thresholds = valid_plan()["decision_rule"]["thresholds"]
    metrics = {
        "phase_decay0p5_geometric_recovery": 0.60,
        "phase_decay0p5_minimum_layer_phase_recovery": 0.1,
        "phase_decay0p5_maximum_feedback_fro": 1.0,
        "phase_decay0p5_maximum_validation_ce_gap": 0.008,
        "phase_decay0p5_terminal_validation_ce_gap": 0.004,
        "phase_zero_feedback_terminal_validation_ce_gap": 0.006,
        "sequential_decay0p5_terminal_validation_ce_gap": 0.03,
    }
    result = classify(metrics, thresholds)
    assert result["classification"] == (
        "BOUNDED_CARRY_PHASE_CAPACITY_PASS_TRANSPORT_FAIL"
    )
    assert result["authorization"]["run_language_model_training"] is False
    metrics["sequential_decay0p5_terminal_validation_ce_gap"] = 0.01
    result = classify(metrics, thresholds)
    assert result["classification"] == (
        "BOUNDED_CARRY_MATURE_CAPACITY_AND_TRANSPORT_PASS"
    )
    assert result["authorization"]["bounded_transport_theory"] is True


def test_zero_feedback_control_is_reported() -> None:
    thresholds = valid_plan()["decision_rule"]["thresholds"]
    metrics = {
        "phase_decay0p5_geometric_recovery": 0.60,
        "phase_decay0p5_minimum_layer_phase_recovery": 0.1,
        "phase_decay0p5_maximum_feedback_fro": 1.0,
        "phase_decay0p5_maximum_validation_ce_gap": 0.008,
        "phase_decay0p5_terminal_validation_ce_gap": 0.004,
        "phase_zero_feedback_terminal_validation_ce_gap": 0.003,
        "sequential_decay0p5_terminal_validation_ce_gap": 0.01,
    }
    assert classify(metrics, thresholds)["zero_feedback_terminal_closes"] is True
