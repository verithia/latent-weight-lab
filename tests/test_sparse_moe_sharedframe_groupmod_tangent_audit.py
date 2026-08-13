import torch

from examples.nanogpt.analyze_sparse_moe_sharedframe_groupmod_tangent_audit import (
    conflict_metrics,
    corresponding_cosines,
    group_tangent_projection,
    side_passes,
)


def test_group_tangent_projection_recovers_left_group_direction() -> None:
    torch.manual_seed(7)
    frame = torch.randn(8, 8)
    gradient = torch.zeros_like(frame)
    gradient[:2] = frame[:2] * 3.0
    coefficients, explained = group_tangent_projection(
        gradient, frame, groups=4, side="left"
    )
    assert coefficients.shape == (4,)
    assert abs(explained - 1.0) < 1e-6


def test_group_tangent_projection_recovers_right_group_direction() -> None:
    torch.manual_seed(8)
    frame = torch.randn(8, 8)
    gradient = torch.zeros_like(frame)
    gradient[:, 2:4] = frame[:, 2:4] * -2.0
    _, explained = group_tangent_projection(
        gradient, frame, groups=4, side="right"
    )
    assert abs(explained - 1.0) < 1e-6


def test_conflict_metrics_detect_exact_cancellation() -> None:
    gradients = torch.tensor([[[1.0, 0.0]], [[-1.0, 0.0]]])
    row = conflict_metrics(gradients)
    assert row["pairwise_cosine_mean"] == -1.0
    assert row["cancellation_ratio"] == 0.0
    assert row["finite_nonzero_gradient_count"] == 2


def test_corresponding_cosines_and_side_gate() -> None:
    values = torch.eye(3).repeat(2, 1, 1).unsqueeze(-1)
    row = corresponding_cosines(values, values)
    assert row["mean"] == 1.0
    gates = {
        "group_tangent_explained_energy_mean_min_each_diagonal_cell": 0.1,
        "group_tangent_explained_energy_minimum_layer_mean_min_each_diagonal_cell": 0.05,
        "fixed_endpoint_cross_bank_group_coefficient_cosine_mean_min": 0.5,
        "fixed_endpoint_cross_bank_group_coefficient_cosine_minimum_layer_mean_min": 0.35,
    }
    assert side_passes(
        [{"mean": 0.2, "minimum_layer_mean": 0.1}] * 2,
        [{"mean": 0.8, "minimum_layer_mean": 0.7}] * 2,
        gates,
    )
