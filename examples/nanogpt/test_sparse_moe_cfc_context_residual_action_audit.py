from __future__ import annotations

import torch

from examples.nanogpt.analyze_sparse_moe_cfc_context_residual_action_audit import (
    residual_direction_metrics,
    result_authorization,
)


def test_exact_residual_direction_has_unit_recovery() -> None:
    static = torch.tensor([1.0, 2.0])
    target = torch.tensor([2.0, 4.0])
    gated = target.clone()
    metrics = residual_direction_metrics(static, gated, target)
    assert abs(metrics["optimal_positive_alpha"] - 1.0) < 1e-12
    assert abs(metrics["optimal_residual_recovery"] - 1.0) < 1e-12
    assert abs(metrics["optimal_total_recovery"] - 1.0) < 1e-12


def test_negative_direction_is_clipped_to_zero_alpha() -> None:
    static = torch.zeros(3)
    target = torch.ones(3)
    gated = -torch.ones(3)
    metrics = residual_direction_metrics(static, gated, target)
    assert metrics["signed_cosine"] < -0.999999
    assert metrics["optimal_positive_alpha"] == 0.0
    assert metrics["optimal_residual_recovery"] == 0.0


def test_transferred_alpha_is_scored_unchanged() -> None:
    static = torch.zeros(2)
    target = torch.tensor([2.0, 0.0])
    gated = torch.tensor([1.0, 0.0])
    metrics = residual_direction_metrics(
        static, gated, target, transferred_alpha=0.5
    )
    assert metrics["optimal_positive_alpha"] == 2.0
    assert metrics["transferred_alpha"] == 0.5
    assert metrics["transferred_alpha_residual_recovery"] < 1.0


def test_pass_does_not_authorize_fit_or_training() -> None:
    authorization = result_authorization(True)
    assert authorization["direct_sum_preregistration"]
    assert not authorization["new_coordinate_fit"]
    assert not authorization["production_implementation"]
    assert not authorization["language_model_training"]
