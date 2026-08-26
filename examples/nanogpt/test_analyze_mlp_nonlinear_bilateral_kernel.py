from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_nonlinear_bilateral_kernel import (
    MultiatomNonlinearBilateralKernel,
    NonlinearBilateralKernel,
    apply_normalized_step,
    gradient_seeded_factors,
    gradient_seeded_multiatom_factors,
    project_target,
)


def test_kernel_is_zero_at_seed_and_generically_full_rank_after_motion() -> None:
    torch.manual_seed(3)
    gradient = torch.randn(9, 7)
    left, right = gradient_seeded_factors(gradient, rank=3, product_rms=0.5)
    module = NonlinearBilateralKernel(left, right, output_scale=0.02)
    torch.testing.assert_close(module(), torch.zeros_like(gradient))
    with torch.no_grad():
        module.left.add_(torch.randn_like(module.left), alpha=0.1)
    assert torch.linalg.matrix_rank(module()).item() == 7


def test_projector_recovers_an_exact_tangent_direction() -> None:
    torch.manual_seed(4)
    gradient = torch.randn(8, 6)
    left, right = gradient_seeded_factors(gradient, rank=2, product_rms=0.5)
    module = NonlinearBilateralKernel(left, right, output_scale=0.2)
    direction = (torch.randn_like(module.left), torch.randn_like(module.right))
    primals = (module.left.detach(), module.right.detach())

    def materialize(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        current = a @ b.T
        initial = module.initial_left @ module.initial_right.T
        return module.output_scale * (current.sin() - initial.sin())

    _, target = torch.func.jvp(materialize, primals, direction)
    _, action, metrics = project_target(
        module, target, cg_steps=40, damping_ratio=1e-8
    )
    assert metrics["action_capture"] > 0.999
    assert torch.nn.functional.cosine_similarity(
        action.flatten(), target.flatten(), dim=0
    ) > 0.999


def test_normalized_step_respects_coordinate_cap() -> None:
    torch.manual_seed(5)
    gradient = torch.randn(8, 6)
    left, right = gradient_seeded_factors(gradient, rank=2, product_rms=0.5)
    module = NonlinearBilateralKernel(left, right, output_scale=0.2)
    coordinates, action, _ = project_target(
        module, gradient, cg_steps=20, damping_ratio=1e-6
    )
    diagnostics = apply_normalized_step(
        module,
        coordinates,
        action,
        norm_reference=gradient * 1000,
        learning_rate=1.0,
        coordinate_cap=0.02,
    )
    assert diagnostics["cap_scale"] < 1.0
    assert diagnostics["applied_maximum_coordinate_update"] <= 0.02000001


def test_multiatom_chart_preserves_budget_and_gains_rank() -> None:
    torch.manual_seed(6)
    gradient = torch.randn(12, 9)
    left, right = gradient_seeded_multiatom_factors(
        gradient, atoms=3, rank_per_atom=1, product_rms=0.5
    )
    module = MultiatomNonlinearBilateralKernel(
        left, right, output_scale=0.02 / (3**0.5)
    )
    assert module.coordinate_count == 3 * (12 + 9)
    torch.testing.assert_close(module(), torch.zeros_like(gradient))
    with torch.no_grad():
        module.left.add_(torch.randn_like(module.left), alpha=0.2)
        module.right.add_(torch.randn_like(module.right), alpha=0.2)
    assert torch.linalg.matrix_rank(module()).item() == 9
