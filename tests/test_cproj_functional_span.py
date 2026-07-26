from __future__ import annotations

import math

import torch

from examples.nanogpt.analyze_cproj_functional_span import (
    functional_span_metrics,
    solve_full_core,
    solve_paired_diagonal,
)


def test_full_core_represents_cross_basis_map_that_diagonal_cannot() -> None:
    hidden = torch.eye(2)
    in_basis = torch.eye(2)
    out_basis = torch.eye(2)
    target = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    _, diagonal_prediction = solve_paired_diagonal(
        hidden, target, in_basis, out_basis
    )
    core, core_prediction = solve_full_core(hidden, target, in_basis, out_basis)

    torch.testing.assert_close(diagonal_prediction, torch.zeros_like(target))
    torch.testing.assert_close(core, target)
    torch.testing.assert_close(core_prediction, target)


def test_functional_metrics_recover_exact_diagonal_and_learned_solution() -> None:
    hidden = torch.eye(2)
    in_basis = torch.eye(2)
    out_basis = torch.eye(2)
    diagonal = torch.tensor([2.0, -0.5])
    target = torch.diag(diagonal)

    metrics = functional_span_metrics(
        hidden,
        target,
        in_basis,
        out_basis,
        diagonal,
        torch.tensor(1.0),
    )

    assert math.isclose(metrics["paired_optimal_explained_energy"], 1.0)
    assert math.isclose(metrics["full_core_optimal_explained_energy"], 1.0)
    assert math.isclose(metrics["learned_explained_energy"], 1.0)
    assert math.isclose(metrics["learned_target_cosine"], 1.0)
    assert math.isclose(metrics["learned_paired_optimal_cosine"], 1.0)
