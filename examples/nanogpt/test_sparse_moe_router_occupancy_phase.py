from __future__ import annotations

import torch

from examples.nanogpt.analyze_sparse_moe_router_occupancy_phase import (
    coefficient_from_ratio,
    occupancy_statistics,
    persistent_collapse_onset,
)


def test_soft_uniform_objective_can_hide_hard_top2_collapse() -> None:
    router = torch.zeros(8, 4)
    activations = torch.randn(256, 4)
    result = occupancy_statistics(router, activations, 2, 0.01, 0.001)
    assert result["expert_counts"][:2] == [256, 256]
    assert result["expert_counts"][2:] == [0] * 6
    assert result["minimum_fraction"] == 0.0
    assert result["experts_below_one_percent"] == 6
    assert abs(result["load_balance_loss"] - 1.0) < 1e-7
    assert abs(result["soft_effective_expert_count"] - 8.0) < 1e-5


def test_persistent_collapse_onset_is_not_interpolated() -> None:
    rows = [
        {"step": 0, "minimum_fraction": 0.03},
        {"step": 10, "minimum_fraction": 0.008},
        {"step": 20, "minimum_fraction": 0.02},
        {"step": 30, "minimum_fraction": 0.006},
        {"step": 40, "minimum_fraction": 0.004},
    ]
    assert persistent_collapse_onset(rows) == 30


def test_coefficient_rule_rounds_up_to_frozen_grid() -> None:
    assert coefficient_from_ratio(0.20) == 0.04
    assert coefficient_from_ratio(0.10) == 0.08
    assert coefficient_from_ratio(0.01) == 0.16
    assert coefficient_from_ratio(1.00) == 0.02
