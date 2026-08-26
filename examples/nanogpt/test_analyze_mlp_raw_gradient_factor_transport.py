from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_raw_gradient_factor_transport import (
    aggregate_capture,
    canonical_overlap,
    fit_shared_factors,
    tangent_capture,
)


def test_tangent_capture_exact_components() -> None:
    left = torch.eye(4)[:, :1]
    right = torch.eye(5)[:, :1]
    direction = left @ torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    direction = direction + torch.tensor([[0.0], [1.0], [2.0], [3.0]]) @ right.T
    result = tangent_capture(direction, left, right)
    assert abs(result["rank_manifold_tangent_capture"] - 1.0) < 1e-6


def test_fit_shared_factors_recovers_stable_rank_one() -> None:
    left_true = torch.tensor([1.0, 2.0, -1.0])
    left_true = left_true / left_true.norm()
    right_true = torch.tensor([2.0, 1.0, -1.0, 0.5])
    right_true = right_true / right_true.norm()
    left = [left_true[:, None] for _ in range(3)]
    values = [torch.tensor([scale]) for scale in (1.0, 2.0, 3.0)]
    right = [right_true[:, None] for _ in range(3)]
    fitted_left, fitted_right = fit_shared_factors(
        left, values, right, [0, 1, 2], rank=1, component_rank=1
    )
    left_overlap = float((fitted_left.T @ left_true).square())
    right_overlap = float((fitted_right.T @ right_true).square())
    assert abs(left_overlap - 1.0) < 1e-6
    assert abs(right_overlap - 1.0) < 1e-6


def test_aggregate_capture_is_energy_weighted() -> None:
    directions = torch.stack((torch.eye(2), 2 * torch.eye(2)))
    left = torch.eye(2)[:, :1]
    right = torch.eye(2)[:, :1]
    one = aggregate_capture(directions, [0], left, right)
    both = aggregate_capture(directions, [0, 1], left, right)
    assert abs(one["rank_manifold_tangent_capture"] - 0.5) < 1e-6
    assert abs(both["rank_manifold_tangent_capture"] - 0.5) < 1e-6


def test_canonical_overlap_identity() -> None:
    basis = torch.eye(5)[:, :3]
    mean, minimum, maximum = canonical_overlap(basis, basis)
    assert abs(mean - 1.0) < 1e-7
    assert abs(minimum - 1.0) < 1e-7
    assert abs(maximum - 1.0) < 1e-7
