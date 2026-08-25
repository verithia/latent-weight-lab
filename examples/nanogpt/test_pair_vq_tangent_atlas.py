from __future__ import annotations

import torch

from examples.nanogpt.pair_vq_tangent_atlas import (
    AtlasProtocol,
    FullRankTangentPanel,
    conjugate_gradient,
    fixed_matchings,
    givens_transform,
    join_panels,
    split_panels,
)


def protocol(*, atoms: int = 1, fit_steps: int = 4) -> AtlasProtocol:
    return AtlasProtocol(
        width=8,
        stages=3,
        atoms=atoms,
        fit_steps=fit_steps,
        fit_learning_rate=0.02,
        fit_weight_decay=0.0,
        fit_gradient_clip=10.0,
        cg_iterations=20,
        cg_tolerance=1e-8,
        cg_ridge=1e-8,
        seed=17,
    )


def test_arbitrary_even_width_givens_preserves_norm_and_gradient() -> None:
    permutations = fixed_matchings(width=6, stages=4, seed=3)
    angles = torch.randn(4, 3, requires_grad=True)
    values = torch.randn(5, 6)
    output = givens_transform(values, angles, permutations)
    torch.testing.assert_close(output.norm(dim=-1), values.norm(dim=-1))
    output.square().sum().backward()
    assert angles.grad is not None
    assert torch.isfinite(angles.grad).all()


def test_coordinate_count_matches_full_rank_atlas_contract() -> None:
    candidate = AtlasProtocol(
        width=768,
        stages=10,
        atoms=1,
        fit_steps=1,
        fit_learning_rate=0.01,
        fit_weight_decay=0.0,
        fit_gradient_clip=1.0,
        cg_iterations=1,
        cg_tolerance=1e-4,
        cg_ridge=1e-6,
        seed=1,
    )
    assert candidate.coordinates_per_panel == 8448
    assert (768 * 768) / candidate.coordinates_per_panel == 69.81818181818181


def test_panel_is_full_rank_for_nonzero_diagonal() -> None:
    chart = FullRankTangentPanel(protocol(), identity="rank", device="cpu")
    state = chart.initial_state(torch.eye(8)).detached()
    state.diagonal.fill_(1.0)
    matrix = chart.materialize(state)
    assert torch.linalg.matrix_rank(matrix) == 8


def test_cg_solves_positive_diagonal_system() -> None:
    rhs = (torch.tensor([2.0, 6.0]),)

    def operator(values: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        return (values[0] * torch.tensor([2.0, 3.0]),)

    solution, diagnostics = conjugate_gradient(
        operator, rhs, maximum_iterations=4, tolerance=1e-10
    )
    torch.testing.assert_close(solution[0], torch.tensor([1.0, 2.0]))
    assert diagnostics["cg_final_residual_norm"] < 1e-6


def test_panel_split_and_join_are_exact() -> None:
    c_fc = torch.arange(32 * 8).reshape(32, 8)
    c_proj = torch.arange(8 * 32).reshape(8, 32)
    assert torch.equal(join_panels(split_panels(c_fc, 8), c_fc.shape), c_fc)
    assert torch.equal(join_panels(split_panels(c_proj, 8), c_proj.shape), c_proj)


def test_tangent_projection_recovers_diagonal_direction() -> None:
    chart = FullRankTangentPanel(protocol(fit_steps=2), identity="diag", device="cpu")
    state = chart.initial_state(torch.eye(8)).detached()
    state.left_angles.zero_()
    state.right_angles.zero_()
    state.diagonal.fill_(1.0)
    requested = torch.diag(torch.linspace(0.1, 0.8, 8))
    projected, diagnostics = chart.project_tangent(state, requested)
    assert diagnostics["cosine"] > 0.999999
    torch.testing.assert_close(projected, requested, rtol=1e-4, atol=1e-5)
