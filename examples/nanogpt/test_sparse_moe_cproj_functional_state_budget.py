from __future__ import annotations

import torch

from examples.nanogpt.analyze_sparse_moe_cproj_functional_state_budget import (
    action_spectrum,
    intrinsic_rank_dimension,
    largest_rank_within_budget,
    rank_for_energy,
    recovery_at_rank,
    spectrum_metrics,
    subspace_overlap,
)


def test_registered_sparse_cproj_budgets() -> None:
    dense = 768 * 1536
    assert largest_rank_within_budget(dense // 200, 768, 1536) == 2
    assert largest_rank_within_budget(4194, 768, 1536) == 1
    assert largest_rank_within_budget(dense // 500, 768, 1536) == 1
    assert intrinsic_rank_dimension(2, 768, 1536) == 4604


def test_spectrum_and_energy_rank() -> None:
    action = torch.diag(torch.tensor([4.0, 2.0, 1.0]))
    values, vectors = action_spectrum(action)
    assert torch.allclose(values, torch.tensor([16.0, 4.0, 1.0]))
    assert vectors.shape == (3, 3)
    assert rank_for_energy(values, 0.80) == 2
    assert abs(recovery_at_rank(values, 1) - 16.0 / 21.0) < 1e-7


def test_subspace_overlap_extremes() -> None:
    identity = torch.eye(4)
    swapped = identity[:, [2, 3, 0, 1]]
    assert abs(subspace_overlap(identity, identity, 2) - 1.0) < 1e-7
    assert abs(subspace_overlap(identity, swapped, 2)) < 1e-7


def test_joint_budget_uses_sum_of_expert_ranks() -> None:
    values = torch.ones(32)
    metrics = spectrum_metrics(
        values,
        energy_thresholds=[0.5, 0.8, 0.9, 0.95, 0.99],
        compression_targets=[200.0, 281.27038626609444, 500.0],
        experts=8,
        output_width=768,
        input_width=1536,
        joint=True,
    )
    assert metrics["compression_200p0x_ordinary_rank_per_expert"] == 2
    assert metrics["compression_200p0x_optimistic_action_rank"] == 16
    assert abs(float(metrics["compression_200p0x_best_recovery"]) - 0.5) < 1e-7
    assert metrics["compression_281p27038626609444x_optimistic_action_rank"] == 8
