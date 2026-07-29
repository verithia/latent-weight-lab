from __future__ import annotations

import torch
from torch.func import functional_call, jvp

from examples.nanogpt.analyze_mlp_bilateral_phase_capture import (
    project_target,
)
from examples.nanogpt.analyze_mlp_natural_axis_phase_capture import (
    GroupedCayleyRotation,
    NaturalAxisBilateralWeightChart,
)


def tiny_chart() -> NaturalAxisBilateralWeightChart:
    return NaturalAxisBilateralWeightChart(
        hidden_features=8,
        output_features=4,
        hidden_groups=2,
        output_groups=2,
        coordinate_scale=2.0,
        gain_scale=2.0,
    ).to(dtype=torch.float64)


def test_grouped_cayley_rotation_is_orthogonal() -> None:
    rotation = GroupedCayleyRotation(
        features=8,
        groups=2,
        coordinate_scale=2.0,
    ).to(dtype=torch.float64)
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        rotation.local_coordinates.copy_(
            torch.randn(
                rotation.local_coordinates.shape,
                generator=generator,
                dtype=torch.float64,
            )
            * 0.05
        )
        rotation.group_coordinates.copy_(
            torch.randn(
                rotation.group_coordinates.shape,
                generator=generator,
                dtype=torch.float64,
            )
            * 0.05
        )
    matrix = rotation.matrix(torch.empty((), dtype=torch.float64))
    torch.testing.assert_close(
        matrix.transpose(0, 1) @ matrix,
        torch.eye(8, dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )


def test_natural_axis_chart_is_exact_identity() -> None:
    generator = torch.Generator().manual_seed(11)
    base = torch.randn(4, 8, generator=generator, dtype=torch.float64)
    torch.testing.assert_close(
        tiny_chart()(base),
        base,
        rtol=0.0,
        atol=0.0,
    )


def test_natural_axis_projection_recovers_chart_tangent() -> None:
    generator = torch.Generator().manual_seed(13)
    base = torch.randn(4, 8, generator=generator, dtype=torch.float64)
    chart = tiny_chart()
    named = dict(chart.named_parameters())
    names = sorted(named)
    primals = tuple(named[name].detach() for name in names)
    direction = tuple(
        torch.randn(
            value.shape,
            generator=generator,
            dtype=value.dtype,
        )
        for value in primals
    )

    def materialize(*coordinates: torch.Tensor) -> torch.Tensor:
        replacements = {
            name: value
            for name, value in zip(names, coordinates, strict=True)
        }
        return functional_call(
            chart,
            replacements,
            (base,),
            strict=False,
        )

    _, target = jvp(materialize, primals, direction)
    metrics = project_target(
        chart,
        base,
        target,
        damping_ratio=1e-10,
        cg_steps=64,
        trace_seed=17,
    )
    assert metrics["recovered_energy_fraction"] > 0.999999
    assert metrics["target_projected_cosine"] > 0.999999


def test_registered_chart_coordinate_count() -> None:
    chart = NaturalAxisBilateralWeightChart(
        hidden_features=3072,
        output_features=768,
        hidden_groups=48,
        output_groups=12,
        coordinate_scale=4.0,
        gain_scale=4.0,
    )
    assert sum(parameter.numel() for parameter in chart.parameters()) == 125994
