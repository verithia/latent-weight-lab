from __future__ import annotations

import math

import torch

from examples.nanogpt.analyze_mlp_residual_multiresolution_basis import (
    dct_ii,
    evaluate_coefficients,
    haar_1d,
)


def test_dct_matches_explicit_orthonormal_matrix() -> None:
    torch.manual_seed(3)
    values = torch.randn(4, 8)
    sample = torch.arange(8, dtype=values.dtype)
    frequency = torch.arange(8, dtype=values.dtype).view(-1, 1)
    matrix = torch.cos(math.pi / 8 * (sample + 0.5) * frequency)
    matrix[0] *= math.sqrt(1.0 / 8)
    matrix[1:] *= math.sqrt(2.0 / 8)
    expected = values @ matrix.T
    torch.testing.assert_close(dct_ii(values), expected, rtol=2e-5, atol=2e-5)


def test_haar_is_orthonormal() -> None:
    torch.manual_seed(5)
    values = torch.randn(3, 16)
    transformed = haar_1d(values)
    torch.testing.assert_close(
        transformed.square().sum(dim=-1),
        values.square().sum(dim=-1),
        rtol=2e-6,
        atol=2e-6,
    )


def test_exact_support_capture() -> None:
    coefficients = torch.zeros(2, 4, 4)
    coefficients[0, 0, 0] = 1.0
    coefficients[1, 3, 3] = 1.0
    probabilities = torch.tensor([0.75, 0.25])
    order = torch.arange(16)
    rows = evaluate_coefficients(
        coefficients,
        probabilities,
        order,
        family="test",
        budgets=[1 / 16],
        retained_fraction=1.0,
        expansion_ops=32,
    )
    assert rows[0]["stored_scalars"] == 1
    assert rows[0]["weighted_pc_capture"] == 0.75
    assert rows[0]["minimum_pc_capture"] == 0.0
