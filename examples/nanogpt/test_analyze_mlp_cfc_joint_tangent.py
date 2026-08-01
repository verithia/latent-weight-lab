import torch

from examples.nanogpt.analyze_mlp_cfc_joint_tangent import (
    joint_jvp,
    joint_vjp,
    solve_joint_tangent,
)
from examples.nanogpt.muon_matched_givens import apply_givens_flow


def test_joint_jvp_vjp_are_adjoint() -> None:
    torch.manual_seed(7)
    weight = torch.randn(6, 4)
    output_pairs = torch.tensor([[0, 1], [2, 5], [3, 4]])
    input_pairs = torch.tensor([[0, 3], [1, 2]])
    coordinates = torch.randn(5)
    cotangent = torch.randn_like(weight)
    lhs = (joint_jvp(weight, output_pairs, input_pairs, coordinates) * cotangent).sum()
    rhs = (coordinates * joint_vjp(weight, output_pairs, input_pairs, cotangent)).sum()
    torch.testing.assert_close(lhs, rhs, rtol=1e-5, atol=1e-5)


def test_joint_solver_recovers_a_representable_tangent() -> None:
    torch.manual_seed(11)
    weight = torch.randn(8, 6)
    output_pairs = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])
    input_pairs = torch.tensor([[0, 1], [2, 3], [4, 5]])
    coordinates = torch.randn(7) * 0.01
    target = joint_jvp(weight, output_pairs, input_pairs, coordinates)
    solved, diagnostics = solve_joint_tangent(
        weight,
        target,
        output_pairs,
        input_pairs,
        iterations=24,
        damping=1e-8,
    )
    relative_error = (solved - target).norm() / target.norm()
    assert float(relative_error) < 1e-4
    assert diagnostics["coordinates"] == 7


def test_joint_tangent_matches_finite_givens_signs() -> None:
    torch.manual_seed(13)
    weight = torch.randn(6, 4)
    epsilon = 1e-3
    output_permutation = torch.arange(6).reshape(1, 6)
    output_pairs = output_permutation.reshape(-1, 2)
    output_angles = torch.tensor([[epsilon, 0.0, 0.0]])
    finite_output = (
        apply_givens_flow(
            weight.T,
            output_angles,
            output_permutation,
        ).T
        - weight
    )
    tangent_output = joint_jvp(
        weight,
        output_pairs,
        torch.empty(0, 2, dtype=torch.long),
        output_angles.reshape(-1),
    )
    torch.testing.assert_close(
        finite_output / epsilon,
        tangent_output / epsilon,
        rtol=1e-3,
        atol=1e-3,
    )

    input_permutation = torch.arange(4).reshape(1, 4)
    input_pairs = input_permutation.reshape(-1, 2)
    input_angles = torch.tensor([[epsilon, 0.0]])
    finite_input = (
        apply_givens_flow(weight, input_angles, input_permutation) - weight
    )
    tangent_input = joint_jvp(
        weight,
        torch.empty(0, 2, dtype=torch.long),
        input_pairs,
        input_angles.reshape(-1),
    )
    torch.testing.assert_close(
        finite_input / epsilon,
        tangent_input / epsilon,
        rtol=1e-3,
        atol=1e-3,
    )


def test_bilateral_allocations_are_equal_coordinate() -> None:
    assert 88 * (3072 // 2) == 135168
    assert 80 * (3072 // 2) + 32 * (768 // 2) == 135168
    assert 72 * (3072 // 2) + 64 * (768 // 2) == 135168
