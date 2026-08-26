from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_residual_block_atlas_basis import (
    evaluate,
    fit_tangent,
    tangent_projection,
)


def test_tangent_projection_recovers_left_and_right_directions() -> None:
    torch.manual_seed(3)
    left, _ = torch.linalg.qr(torch.randn(8, 2))
    right, _ = torch.linalg.qr(torch.randn(6, 2))
    left_direction = left @ torch.randn(2, 6)
    right_direction = torch.randn(8, 2) @ right.T
    targets = torch.stack((left_direction, right_direction))
    projected = tangent_projection(targets, left, right)
    assert torch.allclose(projected, targets, atol=1e-5, rtol=1e-5)


def test_alternating_fit_improves_structured_basis_capture() -> None:
    torch.manual_seed(7)
    left, _ = torch.linalg.qr(torch.randn(9, 2))
    right, _ = torch.linalg.qr(torch.randn(7, 2))
    matrices = torch.stack(
        (
            left @ torch.randn(2, 7),
            torch.randn(9, 2) @ right.T,
            left @ torch.randn(2, 7) + torch.randn(9, 2) @ right.T,
        )
    )
    probabilities = torch.tensor([0.5, 0.3, 0.2])
    fitted_left, fitted_right, history = fit_tangent(
        matrices, probabilities, rank=2, iterations=3
    )
    weighted, minimum, _maximum, _captures = evaluate(
        matrices, probabilities, fitted_left, fitted_right
    )
    assert history[-1]["weighted_tangent_capture"] > 0.999
    assert weighted > 0.999
    assert minimum > 0.999
