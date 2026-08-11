from __future__ import annotations

import torch

from examples.nanogpt.analyze_sparse_moe_stepzero_kfac_factor_oracle import (
    build_kfac_basis,
    gelu_derivative,
    weighted_top_eigenbasis,
)
from examples.nanogpt.analyze_sparse_moe_rolling_tangent_oracle import LayerState


def test_weighted_top_eigenbasis_recovers_diagonal_order() -> None:
    rows = torch.eye(5)
    weights = torch.tensor([1.0, 5.0, 2.0, 4.0, 3.0])
    basis, stats = weighted_top_eigenbasis(rows, weights, rank=2, ridge_ratio=1e-8)
    expected = torch.zeros(5, 2)
    expected[1, 0] = 1.0
    expected[3, 1] = 1.0
    torch.testing.assert_close(basis.abs(), expected)
    assert 0.0 < stats["top_rank_energy_fraction"] < 1.0


def test_gelu_derivative_matches_autograd() -> None:
    values = torch.linspace(-3.0, 3.0, 17, requires_grad=True)
    observed = torch.autograd.grad(torch.nn.functional.gelu(values).sum(), values)[0]
    torch.testing.assert_close(gelu_derivative(values.detach()), observed, atol=1e-6, rtol=1e-6)


def test_kfac_basis_has_expected_paired_shapes() -> None:
    torch.manual_seed(61)
    state = LayerState(
        torch.randn(2, 4),
        torch.randn(2, 5, 4) * 0.1,
        torch.randn(2, 4, 5) * 0.1,
    )
    inputs = torch.randn(64, 4)
    errors = torch.randn(64, 4)
    bank, rows = build_kfac_basis(
        state, inputs, errors, rank=3, ridge_ratio=1e-6,
        minimum_assignments=8, device="cpu",
    )
    assert len(bank) == 3
    assert len(rows) == 2
    assert all(direction.router.shape == state.router.shape for direction in bank)
    assert all(direction.c_fc.shape == state.c_fc.shape for direction in bank)
    assert all(direction.c_proj.shape == state.c_proj.shape for direction in bank)


def test_kfac_basis_records_but_does_not_abort_below_occupancy_gate() -> None:
    torch.manual_seed(62)
    state = LayerState(
        torch.zeros(2, 4),
        torch.randn(2, 5, 4) * 0.1,
        torch.randn(2, 4, 5) * 0.1,
    )
    inputs = torch.randn(8, 4)
    errors = torch.randn(8, 4)
    _bank, rows = build_kfac_basis(
        state, inputs, errors, rank=2, ridge_ratio=1e-6,
        minimum_assignments=128, device="cpu",
    )
    assert min(int(row["assignments"]) for row in rows) == 8
