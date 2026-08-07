import torch

from examples.nanogpt.analyze_qk_cfc_optimizer_state_budget import (
    classify,
    intrinsic_rank_dimension,
    largest_rank_within_budget,
    participation_rank,
    rank_for_energy,
    state_budget_metrics,
)


def test_rank_budget_math() -> None:
    assert intrinsic_rank_dimension(2, 4, 3) == 10
    assert largest_rank_within_budget(10, 4, 3) == 2
    assert rank_for_energy(torch.tensor([4.0, 3.0, 2.0, 1.0]), 0.5) == 2


def test_participation_rank_bounds() -> None:
    assert participation_rank(torch.ones(4)) == 4.0
    assert participation_rank(torch.tensor([1.0, 0.0, 0.0])) == 1.0


def test_state_budget_metrics_detect_low_rank_and_sparse() -> None:
    state = torch.zeros(3072, 768)
    state[0, 0] = 3.0
    state[1, 0] = 4.0
    metrics = state_budget_metrics(state, coordinate_budget=10)
    assert metrics["equal_budget_intrinsic_rank"] == 0
    assert metrics["equal_budget_low_rank_recovery"] == 0.0
    assert metrics["equal_budget_sparse_recovery_ignoring_indices"] == 1.0
    assert metrics["rank_50pct"] == 1


def test_classification_requires_both_state_families() -> None:
    compact = {
        "all": {
            "energy_weighted_low_rank_recovery": 0.9,
            "energy_weighted_sparse_recovery_ignoring_indices": 0.2,
        },
        "late": {
            "minimum_layer_low_rank_recovery": 0.8,
            "minimum_layer_sparse_recovery_ignoring_indices": 0.1,
        },
    }
    dense = {
        "all": {
            "energy_weighted_low_rank_recovery": 0.5,
            "energy_weighted_sparse_recovery_ignoring_indices": 0.5,
        },
        "late": {
            "minimum_layer_low_rank_recovery": 0.4,
            "minimum_layer_sparse_recovery_ignoring_indices": 0.4,
        },
    }
    rule = {
        "minimum_aggregate_recovery": 0.8,
        "minimum_late_layer_recovery": 0.7,
        "threshold_changes_after_measurement": False,
    }
    rejected = classify(
        {"momentum_buffer": compact, "compression_residual": dense}, rule
    )
    assert rejected["classification"] == "CFC_TEMPORAL_STATE_IS_DENSE_SCALE"
    accepted = classify(
        {"momentum_buffer": compact, "compression_residual": compact}, rule
    )
    assert accepted["all_state_families_compact"] is True
