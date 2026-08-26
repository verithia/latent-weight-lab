from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_low_displacement_rank import (
    LowDisplacementRankChart,
    cyclic_displacement,
    gradient_seeded_displacement_factors,
    inverse_cyclic_displacement,
)
from examples.nanogpt.analyze_mlp_nonlinear_bilateral_kernel import project_target


def test_cyclic_displacement_inverse() -> None:
    torch.manual_seed(8)
    matrix = torch.randn(9, 7)
    displaced = cyclic_displacement(matrix, rho=0.9)
    reconstructed = inverse_cyclic_displacement(displaced, rho=0.9)
    torch.testing.assert_close(reconstructed, matrix, atol=2e-5, rtol=2e-5)


def test_ldr_chart_is_zero_at_seed_and_full_rank_after_motion() -> None:
    torch.manual_seed(9)
    gradient = torch.randn(10, 8)
    left, right = gradient_seeded_displacement_factors(
        gradient, rank=3, rho=0.9, decoded_rms=0.5
    )
    module = LowDisplacementRankChart(
        left, right, rho=0.9, output_scale=0.02
    )
    torch.testing.assert_close(module(), torch.zeros_like(gradient))
    with torch.no_grad():
        module.left.add_(torch.randn_like(module.left), alpha=0.1)
    assert torch.linalg.matrix_rank(module()).item() == 8


def test_ldr_projector_recovers_own_tangent() -> None:
    torch.manual_seed(10)
    gradient = torch.randn(8, 6)
    left, right = gradient_seeded_displacement_factors(
        gradient, rank=2, rho=0.9, decoded_rms=0.5
    )
    module = LowDisplacementRankChart(
        left, right, rho=0.9, output_scale=0.2
    )
    direction = (torch.randn_like(module.left), torch.randn_like(module.right))

    def materialize(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return module.output_scale * (
            module.decode(a, b)
            - module.decode(module.initial_left, module.initial_right)
        )

    _, target = torch.func.jvp(
        materialize, (module.left.detach(), module.right.detach()), direction
    )
    _, action, metrics = project_target(
        module, target, cg_steps=50, damping_ratio=1e-8
    )
    assert metrics["action_capture"] > 0.999
    assert torch.nn.functional.cosine_similarity(
        action.flatten(), target.flatten(), dim=0
    ) > 0.999
