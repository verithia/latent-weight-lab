from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_trajectory import sample_indices, trajectory_summary


def test_sample_indices_are_fixed_per_layer_and_unique():
    first = sample_indices(100, 30, 7, 3)
    second = sample_indices(100, 30, 7, 3)
    assert torch.equal(first, second)
    assert first.unique().numel() == 30


def test_subspace_is_deferred_until_three_snapshots():
    deferred = trajectory_summary({3: [(0, torch.zeros(4)), (10, torch.ones(4))]})[0]
    assert deferred["trajectory_subspace_status"].startswith("deferred")
    measured = trajectory_summary(
        {3: [(0, torch.zeros(4)), (10, torch.tensor([1.0, 0.0, 0.0, 0.0])), (20, torch.tensor([0.0, 1.0, 0.0, 0.0]))]}
    )[0]
    assert measured["trajectory_subspace_status"] == "measured_sampled_entry_displacements"
    assert measured["trajectory_linear_rank"] == 2
