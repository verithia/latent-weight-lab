from __future__ import annotations

import torch

from examples.nanogpt.analyze_attention_fht_block_skew_tangent import (
    FixedFHTBlockSkewSide,
    TargetedBilateralTangent,
    coordinate_dot,
    project,
)


def test_side_jvp_matches_finite_difference() -> None:
    torch.manual_seed(7)
    weight = torch.randn(8, 8, dtype=torch.float64)
    for side in ("input", "output"):
        chart = FixedFHTBlockSkewSide(
            weight=weight,
            side=side,
            stages=2,
            rotation_block_size=4,
            basis_block_size=8,
            seed=11,
        )
        coordinates = torch.randn_like(chart.zeros()) * 0.1
        epsilon = 1e-6
        with torch.no_grad():
            chart.mixer.coordinates.copy_((epsilon * coordinates).reshape(-1))
            if side == "input":
                moved = chart.mixer.inverse(weight)
            else:
                moved = chart.mixer(weight.T).T
            finite = (moved - weight) / epsilon
        assert torch.allclose(
            chart.jvp(coordinates), finite, atol=2e-6, rtol=2e-6
        )


def test_adjoint_identity_and_projection() -> None:
    torch.manual_seed(13)
    weight = torch.randn(8, 8)
    chart = TargetedBilateralTangent(
        weight=weight,
        sides=("input", "output"),
        stages=2,
        rotation_block_size=4,
        basis_block_size=8,
        seed=17,
    )
    coordinates = tuple(torch.randn_like(value) for value in chart.zeros())
    direction = torch.randn_like(weight)
    left = (chart.jvp(coordinates) * direction).sum()
    right = coordinate_dot(coordinates, chart.adjoint(direction))
    assert torch.allclose(left, right, atol=2e-5, rtol=2e-5)
    projected, diagnostics = project(
        chart,
        direction,
        maximum_iterations=200,
        tolerance=1e-6,
        ridge=0.0,
    )
    assert projected.shape == weight.shape
    assert diagnostics["projection_orthogonality_error"] < 2e-5
    assert diagnostics["relative_normal_residual"] < 2e-5
