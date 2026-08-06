from __future__ import annotations

import copy

import pytest
import torch

from examples.nanogpt.analyze_mlp_cproj_temporal_residual import (
    classify,
    matrix_structure,
    span_recovery,
    temporal_structure,
    validate_plan,
    vector_metrics,
)


def valid_plan() -> dict:
    return {
        "schema_version": "mai_124m_mlp_cproj_5tpp_temporal_residual_plan_v1",
        "analysis": {
            "parameter_updates": 0,
            "layers": list(range(8)),
            "phases": [[0, 594], [594, 1188], [1188, 1782], [1782, 2373]],
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
                "temporal_pc2_energy_fraction_minimum": 0.8,
                "causal_previous_line_recovery_minimum": 0.1,
                "causal_prior_span_recovery_minimum": 0.25,
                "future_chord_residual_line_recovery_minimum": 0.05,
                "singular_top64_energy_fraction_minimum": 0.5,
                "channel_top_quarter_energy_fraction_minimum": 0.5,
            }
        },
        "authorization": {
            "run_zero_update_temporal_decomposition": True,
            "implement_candidate_structure": False,
            "run_exact_config_mfu": False,
            "run_language_model_training": False,
            "larger_rung": False,
        },
    }


def test_plan_is_fail_closed() -> None:
    validate_plan(valid_plan())
    changed = copy.deepcopy(valid_plan())
    changed["decision_rule"]["thresholds"][
        "causal_prior_span_recovery_minimum"
    ] = 0.1
    with pytest.raises(ValueError):
        validate_plan(changed)


def test_vector_and_span_metrics_are_exact() -> None:
    target = torch.tensor([1.0, 2.0, 0.0])
    same = vector_metrics(target, target)
    assert same["cosine"] == pytest.approx(1.0)
    assert same["positive_line_recovery"] == pytest.approx(1.0)
    assert same["fixed_scale_recovery"] == pytest.approx(1.0)
    assert span_recovery(target, [torch.tensor([1.0, 0.0, 0.0])]) == pytest.approx(
        0.2
    )
    assert span_recovery(target, [target]) == pytest.approx(1.0)


def test_matrix_structure_detects_rank_one_and_channel_concentration() -> None:
    residual = torch.zeros(8, 16)
    residual[0] = 1.0
    metrics = matrix_structure(residual)
    assert metrics["singular_top1_energy_fraction"] == pytest.approx(1.0)
    assert metrics["stable_rank"] == pytest.approx(1.0)
    assert metrics["output_top_quarter_energy_fraction"] == pytest.approx(1.0)


def test_temporal_structure_distinguishes_static_and_transport() -> None:
    basis_a = torch.tensor([1.0, 0.0, 0.0, 0.0])
    basis_b = torch.tensor([0.0, 1.0, 0.0, 0.0])
    static = temporal_structure([basis_a, basis_a, basis_a, basis_a])
    assert static["pc1_energy_fraction"] == pytest.approx(1.0)
    assert static["mean_previous_residual_positive_line_recovery"] == pytest.approx(
        1.0
    )
    transported = temporal_structure([basis_a, basis_b, basis_a, basis_b])
    assert transported["pc2_energy_fraction"] == pytest.approx(1.0)
    assert transported["mean_previous_residual_positive_line_recovery"] == pytest.approx(
        0.0
    )
    assert transported["mean_prior_residual_span_recovery"] > 0.6


def test_classification_never_authorizes_implementation_or_training() -> None:
    thresholds = valid_plan()["decision_rule"]["thresholds"]
    aggregate = {
        "temporal_pc2_energy_fraction": 0.9,
        "causal_previous_line_recovery": 0.05,
        "causal_prior_span_recovery": 0.1,
        "future_chord_residual_line_recovery": 0.01,
        "singular_top64_energy_fraction": 0.2,
        "output_top_quarter_energy_fraction": 0.3,
        "hidden_top_quarter_energy_fraction": 0.3,
    }
    result = classify(aggregate, thresholds)
    assert result["classification"] == "LOW_DIMENSIONAL_BUT_PHASE_TRANSPORTED"
    assert result["authorization"]["phase_local_transport_design"] is True
    assert result["authorization"]["implement_candidate_structure"] is False
    assert result["authorization"]["run_language_model_training"] is False
