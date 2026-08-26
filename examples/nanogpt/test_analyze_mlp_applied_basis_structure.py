from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_applied_basis_structure import (
    bilateral_diagonal_projection,
    blockwise_fht_2d,
    low_rank_for_budget,
    selected_support_capture,
)


def test_blockwise_fht_is_energy_preserving() -> None:
    values = torch.randn((3, 6, 12), generator=torch.Generator().manual_seed(4))
    transformed = blockwise_fht_2d(
        values, row_block=2, column_block=4
    )
    torch.testing.assert_close(
        transformed.square().sum(), values.square().sum(), rtol=2e-6, atol=2e-6
    )


def test_selected_support_uses_fit_only() -> None:
    coefficients = torch.tensor(
        [[10.0, 0.0, 0.0], [9.0, 0.0, 0.0], [0.0, 3.0, 4.0]]
    )
    capture = selected_support_capture(
        coefficients,
        fit_indices=[0, 1],
        eval_indices=[2],
        coordinates=1,
    )
    assert capture == 0.0


def test_bilateral_projection_recovers_exact_tangent() -> None:
    weight = torch.randn((5, 7), generator=torch.Generator().manual_seed(5))
    a = torch.linspace(-0.2, 0.3, 5)
    b = torch.linspace(0.4, -0.1, 7)
    direction = weight * (a.unsqueeze(1) + b.unsqueeze(0))
    capture = bilateral_diagonal_projection(weight, direction, iterations=30)
    assert capture > 0.99999


def test_low_rank_budget_accounting() -> None:
    assert low_rank_for_budget(768, 3072, 0.001) == 0
    assert low_rank_for_budget(768, 3072, 0.01) == 6
