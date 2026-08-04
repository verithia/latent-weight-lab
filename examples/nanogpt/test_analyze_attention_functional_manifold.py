from __future__ import annotations

import torch

from examples.nanogpt.analyze_attention_functional_manifold import (
    product_delta_singular_values,
    product_kernel,
    trajectory_metrics,
)


def test_product_kernel_is_invariant_to_orthogonal_factor_gauge() -> None:
    torch.manual_seed(20260804)
    first = torch.randn(3, 7)
    second = torch.randn(3, 7)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    torch.testing.assert_close(
        product_kernel(rotation @ first, rotation @ second),
        product_kernel(first, second),
        rtol=1e-5,
        atol=1e-5,
    )


def test_lowrank_product_delta_spectrum_matches_dense_delta() -> None:
    torch.manual_seed(20260805)
    first0 = torch.randn(3, 7)
    second0 = torch.randn(3, 7)
    first1 = torch.randn(3, 7)
    second1 = torch.randn(3, 7)
    expected = torch.linalg.svdvals(
        product_kernel(first1, second1) - product_kernel(first0, second0)
    )
    observed = product_delta_singular_values(
        first0, second0, first1, second1
    )
    torch.testing.assert_close(
        observed, expected[: observed.numel()], rtol=2e-5, atol=2e-5
    )
    assert float(expected[observed.numel() :].max()) < 2e-5


def test_trajectory_metrics_recognizes_a_straight_ray() -> None:
    direction = torch.tensor([1.0, -2.0, 3.0])
    rows = torch.stack([scale * direction for scale in (0.0, 1.0, 2.0, 3.0)])
    metrics = trajectory_metrics(rows)
    assert metrics["pc1_energy"] > 0.999999
    assert abs(float(metrics["path_length_over_chord"]) - 1.0) < 1e-7
    assert metrics["mean_terminal_ray_recovery"] > 0.999999
