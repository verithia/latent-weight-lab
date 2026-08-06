from __future__ import annotations

import copy

import pytest

from examples.nanogpt.analyze_mlp_cproj_same_gauge_lwt_allocation import (
    choose_prefix,
    classify,
    validate_plan,
)


def valid_plan() -> dict:
    return {
        "schema_version": "mai_124m_mlp_cproj_same_gauge_lwt_allocation_plan_v1",
        "analysis": {
            "parameter_updates": 0,
            "difficult_layers": list(range(8)),
            "always_procedural_layers": list(range(8, 12)),
            "phases": [[0, 594], [594, 1188], [1188, 1782], [1782, 2373]],
            "straight_chord_substeps": 8,
            "feedback_decay": 0.5,
            "chart": {
                "hidden_parent_stages": 64,
                "hidden_residual_stages": 24,
                "output_stages": 32,
                "neighbors": 64,
                "matching_seed": 20260807,
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
                "discovery_slice": [0, 64],
                "holdout_slice": [64, 192],
                "confirmation_slice": [0, 400],
            },
            "selection": {
                "ranking": "terminal discovery single-dense repair descending",
                "prefix_sizes": [1, 2, 3, 4],
                "selected_prefix": "smallest discovery prefix within 0.005 CE of dense",
                "failure_fallback": "top-four prefix for diagnostic confirmation only",
            },
        },
        "decision_rule": {
            "thresholds": {
                "maximum_dense_exceptions": 4,
                "discovery_terminal_validation_ce_gap": 0.005,
                "holdout_terminal_validation_ce_gap": 0.005,
                "confirmation_terminal_validation_ce_gap": 0.005,
                "confirmation_maximum_phase_validation_ce_gap": 0.01,
                "minimum_terminal_repair_over_all_approx": 0.002,
                "predecessor_must_fail_terminal_gap": 0.005,
            }
        },
        "authorization": {
            "run_zero_update_same_gauge_lwt_attribution": True,
            "implement_candidate_mask": False,
            "run_exact_config_mfu": False,
            "run_language_model_training": False,
            "larger_rung": False,
        },
    }


def test_plan_is_fail_closed() -> None:
    validate_plan(valid_plan())
    changed = copy.deepcopy(valid_plan())
    changed["analysis"]["selection"]["prefix_sizes"] = [1, 2, 3, 4, 5]
    with pytest.raises(ValueError):
        validate_plan(changed)


def test_choose_prefix_selects_smallest_passing_mask() -> None:
    ranking = [3, 1, 0, 2]
    prefix_ce = {1: 3.008, 2: 3.004, 3: 3.002, 4: 3.0}
    layers, k = choose_prefix(
        ranking,
        prefix_ce,
        3.0,
        maximum_k=4,
        maximum_gap=0.005,
    )
    assert layers == [3, 1]
    assert k == 2


def test_choose_prefix_returns_diagnostic_fallback_on_failure() -> None:
    ranking = [3, 1, 0, 2]
    layers, k = choose_prefix(
        ranking,
        {1: 3.02, 2: 3.015, 3: 3.01, 4: 3.006},
        3.0,
        maximum_k=4,
        maximum_gap=0.005,
    )
    assert layers == ranking
    assert k is None


def test_classification_requires_minimal_replicated_mask() -> None:
    thresholds = valid_plan()["decision_rule"]["thresholds"]
    metrics = {
        "selected_k": 2,
        "discovery_selected_terminal_gap": 0.004,
        "holdout_selected_terminal_gap": 0.004,
        "confirmation_selected_terminal_gap": 0.004,
        "confirmation_maximum_phase_gap": 0.008,
        "confirmation_terminal_repair_over_all_approx": 0.003,
        "confirmation_predecessor_terminal_gap": 0.006,
    }
    result = classify(metrics, thresholds)
    assert result["classification"] == "SAME_GAUGE_LWT_MASK_CAPACITY_PASS"
    assert result["authorization"]["run_language_model_training"] is False
    metrics["confirmation_predecessor_terminal_gap"] = 0.004
    result = classify(metrics, thresholds)
    assert result["classification"] == (
        "MASK_OVERSELECTED_REQUIRES_FRESH_PREREGISTRATION"
    )
