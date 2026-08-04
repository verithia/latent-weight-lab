from __future__ import annotations

import torch

from examples.nanogpt.analyze_attention_product_fht_residual_tangent import (
    ProductFHTResidualTangent,
    coordinate_dot,
)


def test_exact_batched_adjoint_identity() -> None:
    torch.manual_seed(7)
    chart = ProductFHTResidualTangent(
        in_features=3,
        out_features=4,
        factors=2,
        seed=19,
        device="cpu",
    )
    coordinates = (
        torch.randn(3, 2, 4),
        torch.randn(3, 4),
    )
    target = torch.randn(3, 4, 3)
    left = (chart.jvp(coordinates) * target).flatten(1).sum(dim=1)
    right = coordinate_dot(coordinates, chart.adjoint(target))
    torch.testing.assert_close(left, right, atol=2e-5, rtol=2e-5)


def test_cg_projection_matches_explicit_small_tangent() -> None:
    torch.manual_seed(11)
    chart = ProductFHTResidualTangent(
        in_features=3,
        out_features=4,
        factors=2,
        seed=23,
        device="cpu",
    )
    columns = []
    for index in range(chart.coordinate_count):
        coordinates = chart.zeros(1)
        if index < chart.factors * chart.padded_features:
            coordinates[0].view(-1)[index] = 1.0
        else:
            coordinates[1].view(-1)[
                index - chart.factors * chart.padded_features
            ] = 1.0
        columns.append(chart.jvp(coordinates).reshape(-1))
    tangent = torch.stack(columns, dim=1)
    target = torch.randn(2, 4, 3)
    expected = []
    for value in target:
        solution = torch.linalg.lstsq(tangent, value.reshape(-1)).solution
        expected.append((tangent @ solution).reshape_as(value))
    expected_projection = torch.stack(expected)
    observed_projection, diagnostics = chart.project(
        target,
        maximum_iterations=200,
        tolerance=1e-7,
        ridge=1e-10,
    )
    torch.testing.assert_close(
        observed_projection,
        expected_projection,
        atol=2e-4,
        rtol=2e-4,
    )
    assert float(diagnostics["projection_orthogonality_error"].max()) < 2e-4
