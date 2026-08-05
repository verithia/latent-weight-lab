from __future__ import annotations

import copy

import torch

from examples.nanogpt.analyze_attention_vo_functional_direction import (
    factor_induced_direction,
    summarize,
)


def test_factor_induced_direction_matches_finite_difference() -> None:
    generator = torch.Generator().manual_seed(7)
    value = torch.randn(3, 5, generator=generator, dtype=torch.float64)
    output = torch.randn(5, 3, generator=generator, dtype=torch.float64)
    d_value = torch.randn(3, 5, generator=generator, dtype=torch.float64)
    d_output = torch.randn(5, 3, generator=generator, dtype=torch.float64)
    epsilon = 1e-6
    measured = (
        (output + epsilon * d_output) @ (value + epsilon * d_value)
        - output @ value
    ) / epsilon
    expected = factor_induced_direction(value, output, d_value, d_output)
    assert torch.allclose(measured, expected, atol=1e-5, rtol=1e-5)


def _plan() -> dict:
    return {
        "protocol": {"guards": {"minimum_valid_cells": 2}},
        "preregistered_gate": {
            "minimum_aggregate_direct_heldout_task_cosine": 0.20,
            "minimum_aggregate_direct_minus_factor_heldout_task_cosine": 0.05,
            "minimum_direct_over_positive_factor_heldout_alignment_multiplier": 1.25,
            "minimum_aggregate_direct_fit_holdout_direction_cosine": 0.15,
            "maximum_aggregate_factor_line_recovery_of_direct": 0.81,
            "minimum_layers_with_positive_direct_advantage": 1,
            "minimum_cells_with_positive_direct_heldout_alignment": 2,
            "pass_classification": "PASS",
            "fail_classification": "FAIL",
            "pass_action": "design",
            "fail_action": "stop",
        },
    }


def test_summary_passes_clear_direct_advantage() -> None:
    keys = [(0, 0), (0, 1)]
    direct = {
        keys[0]: torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        keys[1]: torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
    }
    factor = {
        keys[0]: torch.tensor([[0.0, 1.0], [-1.0, 0.0]]),
        keys[1]: torch.tensor([[1.0, 0.0], [0.0, -1.0]]),
    }
    fit = {
        "direct_gradients": copy.deepcopy(direct),
        "direct_directions": copy.deepcopy(direct),
        "factor_directions": factor,
    }
    holdout = {
        "direct_gradients": copy.deepcopy(direct),
        "direct_directions": copy.deepcopy(direct),
        "factor_directions": copy.deepcopy(factor),
    }
    rows, summary = summarize(fit, holdout, [0], _plan())
    assert len(rows) == 2
    assert summary["passed"] is True
    assert summary["decision"] == "PASS"
    assert summary["language_model_training_authorized"] is False


def test_summary_fails_when_factor_matches_direct() -> None:
    direct = {
        (0, 0): torch.eye(2),
        (0, 1): torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
    }
    window = {
        "direct_gradients": copy.deepcopy(direct),
        "direct_directions": copy.deepcopy(direct),
        "factor_directions": copy.deepcopy(direct),
    }
    _rows, summary = summarize(window, window, [0], _plan())
    assert summary["passed"] is False
    assert summary["decision"] == "FAIL"
    assert not summary["gate"][
        "aggregate_direct_minus_factor_heldout_task_cosine"
    ]
    assert not summary["gate"]["aggregate_factor_line_recovery_of_direct"]
