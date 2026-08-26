from __future__ import annotations

import torch

from examples.nanogpt.analyze_gradient_seeded_lowrank_terminal import (
    terminal_gradient_metrics,
)


def test_terminal_gradient_metrics_distinguishes_tangent_and_pullback() -> None:
    gradient = torch.zeros(4, 4)
    gradient[0, 0] = 2.0
    gradient[1, 2] = 1.0
    left = torch.tensor([[1.0], [0.0], [0.0], [0.0]])
    right = torch.tensor([[1.0], [0.0], [0.0], [0.0]])
    # These are the exact factor derivatives for scale=1 at this A/B pair.
    left_gradient = gradient @ right
    right_gradient = gradient.T @ left

    metrics, tensors = terminal_gradient_metrics(
        gradient,
        left,
        right,
        left_gradient,
        right_gradient,
        scale=1.0,
    )

    assert abs(metrics["tangent_capture"] - 0.8) < 1e-6
    assert abs(metrics["residual_gradient_fraction"] - 0.2) < 1e-6
    assert metrics["best_rank_capture"] == 0.8
    assert metrics["factor_pullback_gradient_cosine"] > 0.89
    assert metrics["factor_pullback_tangent_cosine"] == 1.0
    torch.testing.assert_close(
        tensors["tangent_projection"],
        torch.diag(torch.tensor([2.0, 0.0, 0.0, 0.0])),
    )
