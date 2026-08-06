from __future__ import annotations

import copy

import pytest
import torch

from examples.nanogpt.analyze_mlp_cproj_polynomial_oracle_ce import (
    classify,
    restore_radius,
    validate_plan,
)


def valid_plan() -> dict:
    return {
        "schema_version": "mai_124m_mlp_cproj_polynomial_oracle_ce_plan_v1",
        "analysis": {
            "parameter_updates": 0,
            "layers": [8, 9, 10, 11],
            "discovery_steps": [0, 99, 198, 297, 396, 495, 594, 693, 792, 891, 990, 1089, 1188, 1287, 1386, 1485, 1584, 1683, 1782],
            "terminal_step": 2373,
            "ranks": [1, 2, 4, 8, 16],
            "polynomial_degree": 2,
            "polynomial_coordinate": "cumulative_learning_rate",
            "restore_terminal_radius": True,
            "validation_batches": 400,
            "learned_basis_role": "diagnostic_oracle_only",
        },
        "decision_rule": {"thresholds": {
            "exact_replay_absolute_tolerance_ce": 0.005,
            "candidate_maximum_validation_ce_gap": 0.005,
        }},
        "authorization": {
            "run_zero_update_terminal_oracle_ce": True,
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
    changed["analysis"]["validation_batches"] = 100
    with pytest.raises(ValueError):
        validate_plan(changed)


def test_radius_restoration() -> None:
    values = torch.tensor([[3.0, 4.0]])
    restored = restore_radius(values, torch.tensor(2.0))
    torch.testing.assert_close(restored.norm(), torch.tensor(2.0))


def test_classification_is_fail_closed_and_selects_minimum_rank() -> None:
    rows = [
        {"rank": 1, "passes_ce_gate": False},
        {"rank": 2, "passes_ce_gate": True},
        {"rank": 4, "passes_ce_gate": True},
    ]
    assert classify(rows, False) == ("INVALID_EXACT_REPLAY", None)
    assert classify(rows, True) == (
        "SMOOTH_SCHEDULED_ENVELOPE_FUNCTIONALLY_SUFFICIENT", 2
    )
