from __future__ import annotations

import torch
from torch.func import functional_call, jvp

from examples.nanogpt.analyze_mlp_bilateral_phase_capture import (
    BilateralWeightChart,
    project_target,
    singular_frame_components,
)
from examples.nanogpt.model import GPTConfig, MLP


def tiny_chart(dtype: torch.dtype = torch.float64) -> BilateralWeightChart:
    return BilateralWeightChart(
        hidden_features=8,
        output_features=4,
        hidden_stages=1,
        output_stages=1,
        rotation_block_size=4,
        basis_block_size=4,
        hidden_seed=11,
        output_seed=19,
        coordinate_scale=2.0,
        gain_scale=2.0,
    ).to(dtype=dtype)


def test_bilateral_weight_chart_is_exact_identity() -> None:
    generator = torch.Generator().manual_seed(29)
    base = torch.randn(4, 8, generator=generator, dtype=torch.float64)
    chart = tiny_chart()
    actual = chart(base)
    torch.testing.assert_close(actual, base, rtol=0.0, atol=0.0)


def test_bilateral_weight_chart_matches_production_materializer() -> None:
    config = GPTConfig(
        n_embd=4,
        n_head=1,
        block_fht=False,
        block_fht_mlp_hidden_block_rotation_stages=1,
        block_fht_mlp_hidden_block_rotation_size=4,
        block_fht_mlp_hidden_block_rotation_basis_size=4,
        block_fht_mlp_hidden_block_rotation_coordinate_scale=2.0,
        block_fht_mlp_hidden_gain=True,
        block_fht_mlp_hidden_gain_scale=2.0,
        block_fht_mlp_output_block_rotation_stages=1,
        block_fht_mlp_output_block_rotation_size=4,
        block_fht_mlp_output_block_rotation_basis_size=4,
        block_fht_mlp_output_block_rotation_coordinate_scale=2.0,
        block_fht_mlp_residual_output_gain=True,
        block_fht_mlp_residual_output_gain_scale=2.0,
    )
    layer = 2
    production = MLP(config, layer).to(dtype=torch.float64)
    chart = BilateralWeightChart(
        hidden_features=16,
        output_features=4,
        hidden_stages=1,
        output_stages=1,
        rotation_block_size=4,
        basis_block_size=4,
        hidden_seed=config.block_fht_mlp_hidden_block_rotation_seed + layer * 64,
        output_seed=config.block_fht_mlp_output_rotation_seed + layer * 64,
        coordinate_scale=2.0,
        gain_scale=2.0,
    ).to(dtype=torch.float64)
    generator = torch.Generator().manual_seed(23)
    with torch.no_grad():
        production.hidden_block_rotation.coordinates.copy_(
            torch.randn(
                production.hidden_block_rotation.coordinates.shape,
                generator=generator,
                dtype=torch.float64,
            )
            * 0.05
        )
        production.output_block_rotation.coordinates.copy_(
            torch.randn(
                production.output_block_rotation.coordinates.shape,
                generator=generator,
                dtype=torch.float64,
            )
            * 0.05
        )
        production.hidden_log_gain.copy_(
            torch.randn(16, generator=generator, dtype=torch.float64) * 0.05
        )
        production.residual_output_log_gain.copy_(
            torch.randn(4, generator=generator, dtype=torch.float64) * 0.05
        )
        chart.hidden_rotation.coordinates.copy_(
            production.hidden_block_rotation.coordinates
        )
        chart.output_rotation.coordinates.copy_(
            production.output_block_rotation.coordinates
        )
        chart.hidden_log_gain.copy_(production.hidden_log_gain)
        chart.output_log_gain.copy_(production.residual_output_log_gain)
    base = torch.randn(4, 16, generator=generator, dtype=torch.float64)
    expected = production._materialize_charted_cproj_weight(base)
    actual = chart(base)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_projection_recovers_a_chart_tangent() -> None:
    generator = torch.Generator().manual_seed(31)
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
            device=value.device,
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
        trace_seed=37,
    )
    assert metrics["recovered_energy_fraction"] > 0.999999
    assert metrics["target_projected_cosine"] > 0.999999
    assert metrics["cg_relative_normal_residual"] < 1e-6


def test_singular_components_close_exactly() -> None:
    generator = torch.Generator().manual_seed(41)
    base = torch.randn(4, 8, generator=generator, dtype=torch.float64)
    delta = torch.randn(4, 8, generator=generator, dtype=torch.float64)
    components = singular_frame_components(base, delta)
    reconstructed = (
        components["singular_value"]
        + components["in_frame_mixing"]
        + components["subspace_rotation"]
    )
    torch.testing.assert_close(reconstructed, delta, rtol=1e-12, atol=1e-12)
