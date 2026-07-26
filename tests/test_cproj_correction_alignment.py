from __future__ import annotations

import math

import torch

from examples.nanogpt.analyze_cproj_correction_alignment import (
    correction_alignment_metrics,
    paired_basis_projection,
)


def test_paired_basis_projection_recovers_exact_diagonal_span() -> None:
    in_basis = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
    )
    out_basis = torch.tensor(
        [[1.0, 0.0], [1.0, 1.0]],
    )
    coefficients = torch.tensor([2.0, -0.5])
    target = (out_basis * coefficients.unsqueeze(0)) @ in_basis.T

    recovered, projection, gram = paired_basis_projection(
        target, in_basis, out_basis
    )

    torch.testing.assert_close(recovered, coefficients)
    torch.testing.assert_close(projection, target)
    assert torch.linalg.matrix_rank(gram) == 2


def test_alignment_metrics_measure_unrepresentable_target_and_learned_fit() -> None:
    in_basis = torch.tensor([[1.0], [0.0]])
    out_basis = torch.tensor([[1.0], [0.0]])
    target = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    learned = torch.tensor([[1.0, 0.0], [0.0, 0.0]])

    metrics = correction_alignment_metrics(target, in_basis, out_basis, learned)

    assert math.isclose(metrics["explainable_energy_fraction"], 0.5)
    assert math.isclose(metrics["unexplained_residual_fraction"], 0.5)
    assert math.isclose(
        metrics["learned_target_cosine"], 2.0 ** -0.5, rel_tol=1e-6
    )
    assert math.isclose(metrics["learned_optimal_cosine"], 1.0)
    assert math.isclose(metrics["learned_to_optimal_norm"], 0.5)
    assert math.isclose(metrics["learned_error_reduction_fraction"], 0.375)
