from __future__ import annotations

import copy

import pytest

from examples.nanogpt.analyze_mlp_cproj_late_band_phase_validity import (
    band_passes,
    classify,
    validate_plan,
)


def valid_plan() -> dict:
    return {
        "schema_version": "mai_124m_mlp_cproj_late_band_phase_validity_plan_v1",
        "analysis": {
            "parameter_updates": 0,
            "phases": [[0, 594], [594, 1188], [1188, 1782], [1782, 2373]],
            "early_dense_layers": [0, 1, 2, 3],
            "middle_test_layers": [4, 5, 6, 7],
            "late_test_layers": [8, 9, 10, 11],
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
            },
        },
        "decision_rule": {
            "thresholds": {
                "maximum_phase_validation_ce_gap": 0.01,
                "terminal_validation_ce_gap": 0.005,
            }
        },
        "authorization": {
            "run_zero_update_band_factorization": True,
            "run_coadapted_trajectory_acquisition": False,
            "implement_candidate_mask": False,
            "run_exact_config_mfu": False,
            "run_language_model_training": False,
            "larger_rung": False,
        },
    }


def test_plan_is_fail_closed() -> None:
    validate_plan(valid_plan())
    changed = copy.deepcopy(valid_plan())
    changed["analysis"]["middle_test_layers"] = [3, 4, 5, 6]
    with pytest.raises(ValueError):
        validate_plan(changed)


def test_band_passes_both_phase_and_terminal_gates() -> None:
    assert band_passes(
        {594: 0.009, 1188: 0.008, 1782: 0.006, 2373: 0.004},
        maximum_phase_gap=0.01,
        terminal_gap=0.005,
    )
    assert not band_passes(
        {594: 0.011, 1188: 0.008, 1782: 0.006, 2373: 0.004},
        maximum_phase_gap=0.01,
        terminal_gap=0.005,
    )


def test_classification_requires_coadapted_path_when_late_proxy_fails() -> None:
    metrics = {
        "late_only_gap_by_end_step": {
            594: 0.02,
            1188: 0.01,
            1782: 0.006,
            2373: 0.004,
        },
        "middle_only_gap_by_end_step": {
            594: 0.02,
            1188: 0.01,
            1782: 0.006,
            2373: 0.004,
        },
        "combined_gap_by_end_step": {
            594: 0.03,
            1188: 0.02,
            1782: 0.01,
            2373: 0.004,
        },
    }
    decision = classify(
        metrics,
        valid_plan()["decision_rule"]["thresholds"],
    )
    assert decision["classification"] == "REQUIRE_COADAPTED_LATE_BAND_TRAJECTORY"
    assert decision["authorization"]["run_coadapted_trajectory_acquisition"]


def test_classification_localizes_middle_band() -> None:
    metrics = {
        "late_only_gap_by_end_step": {
            594: 0.009,
            1188: 0.008,
            1782: 0.006,
            2373: 0.004,
        },
        "middle_only_gap_by_end_step": {
            594: 0.02,
            1188: 0.01,
            1782: 0.006,
            2373: 0.004,
        },
        "combined_gap_by_end_step": {
            594: 0.03,
            1188: 0.02,
            1782: 0.01,
            2373: 0.004,
        },
    }
    decision = classify(
        metrics,
        valid_plan()["decision_rule"]["thresholds"],
    )
    assert decision["classification"] == "LOCALIZE_DENSE_PATH_FAILURE_TO_MIDDLE_4_7"
    assert not decision["authorization"]["run_language_model_training"]
