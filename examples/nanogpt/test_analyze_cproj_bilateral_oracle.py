from __future__ import annotations

import torch

from examples.nanogpt.analyze_cproj_bilateral_oracle import (
    BilateralCProjChart,
    ChartSpec,
    fit_spec,
    parse_spec,
)


def test_parse_spec() -> None:
    assert parse_spec("in2_out4_act") == ChartSpec(2, 4, True)
    assert parse_spec("in0_out0_noact") == ChartSpec(0, 0, False)


def test_bilateral_chart_is_exact_identity_at_zero() -> None:
    generator = torch.Generator().manual_seed(11)
    pre_gelu = torch.randn(7, 32, generator=generator)
    weight = torch.randn(16, 32, generator=generator)
    expected = torch.nn.functional.linear(
        torch.nn.functional.gelu(pre_gelu), weight
    )
    for spec in (
        ChartSpec(1, 0, False),
        ChartSpec(0, 1, False),
        ChartSpec(1, 1, False),
        ChartSpec(1, 1, True),
    ):
        chart = BilateralCProjChart(
            hidden_features=32,
            output_features=16,
            spec=spec,
            rotation_block_size=8,
            basis_block_size=16,
            seed=17,
        )
        actual = chart(pre_gelu, weight, None)
        assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_identity_fit_has_zero_recovery() -> None:
    generator = torch.Generator().manual_seed(23)
    train_pre = torch.randn(32, 16, generator=generator)
    holdout_pre = torch.randn(16, 16, generator=generator)
    weight = torch.randn(8, 16, generator=generator)
    train_identity = torch.nn.functional.linear(
        torch.nn.functional.gelu(train_pre), weight
    )
    holdout_identity = torch.nn.functional.linear(
        torch.nn.functional.gelu(holdout_pre), weight
    )
    train_target = train_identity + 0.1
    holdout_target = holdout_identity + 0.1
    row = fit_spec(
        ChartSpec(0, 0, False),
        train_pre,
        train_target,
        holdout_pre,
        holdout_target,
        weight,
        None,
        rotation_block_size=8,
        basis_block_size=16,
        seed=29,
        steps=2,
        batch_size=8,
        learning_rate=0.01,
    )
    assert row["holdout_identity_error_recovery"] == 0.0
    assert row["parameter_count"] == 0.0
