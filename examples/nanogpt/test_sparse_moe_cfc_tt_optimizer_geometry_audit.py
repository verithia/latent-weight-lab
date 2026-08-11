from __future__ import annotations

import torch

from examples.nanogpt.analyze_sparse_moe_cfc_tt_optimizer_geometry_audit import (
    canonicalize_raw_cores,
    classify_gates,
    pcg,
    score_image,
    tuple_dot,
)


def test_canonicalization_makes_nonterminal_cores_orthonormal() -> None:
    generator = torch.Generator().manual_seed(7)
    cores = (
        torch.randn(1, 4, 3, generator=generator),
        torch.randn(3, 3, 2, generator=generator),
        torch.randn(2, 2, 1, generator=generator),
    )
    canonical = canonicalize_raw_cores(cores)
    for core in canonical[:-1]:
        matrix = core.reshape(-1, core.shape[-1])
        identity = torch.eye(matrix.shape[-1])
        assert torch.allclose(matrix.T @ matrix, identity, atol=1e-5)
    assert torch.equal(canonical[-1], cores[-1])


def test_pcg_solves_diagonal_tuple_system() -> None:
    diagonal = (torch.tensor([2.0, 5.0]), torch.tensor([7.0]))
    rhs = (torch.tensor([4.0, 10.0]), torch.tensor([21.0]))

    def operator(values: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        return tuple(
            value * scale
            for value, scale in zip(values, diagonal, strict=True)
        )

    inverse = tuple(value.reciprocal() for value in diagonal)
    solution, diagnostics = pcg(operator, rhs, inverse, iterations=4)
    assert torch.allclose(solution[0], torch.tensor([2.0, 2.0]))
    assert torch.allclose(solution[1], torch.tensor([3.0]))
    assert diagnostics["relative_normal_residual"] < 1e-7
    assert float(tuple_dot(solution, solution)) > 0.0


def test_score_image_separates_direction_and_transferred_scale() -> None:
    residual = torch.tensor([1.0, 2.0, -1.0])
    image = residual * 2.0
    score = score_image(image, residual, transferred_alpha=0.25)
    assert abs(score["directional_cosine"] - 1.0) < 1e-6
    assert abs(score["optimal_positive_alpha"] - 0.5) < 1e-6
    assert abs(score["optimal_positive_scalar_recovery"] - 1.0) < 1e-6
    assert abs(score["transferred_alpha_recovery"] - 0.75) < 1e-6


def test_classification_distinguishes_numeric_local_and_metric_results() -> None:
    base = {
        "all_values_and_gradients_finite": True,
        "pcg_converged": True,
        "gauss_newton_source_recovery_pass": True,
        "gauss_newton_heldout_recovery_pass": True,
        "gauss_newton_gain_pass": True,
        "gauss_newton_transferred_alpha_pass": True,
        "same_endpoint_stability_pass": True,
        "cross_endpoint_stability_pass": True,
        "diagonal_simple_repair_pass": True,
    }
    assert classify_gates(base) == "DIAGONAL_METRIC_REPAIR_AUTHORIZED"
    coupled = {**base, "diagonal_simple_repair_pass": False}
    assert classify_gates(coupled) == "COUPLED_METRIC_REPAIR_AUTHORIZED"
    local = {**base, "gauss_newton_source_recovery_pass": False}
    assert classify_gates(local) == "LOCAL_TT_TANGENT_INSUFFICIENT"
    numeric = {**base, "pcg_converged": False}
    assert classify_gates(numeric) == "OPTIMIZER_GEOMETRY_NUMERICALLY_INCONCLUSIVE"
