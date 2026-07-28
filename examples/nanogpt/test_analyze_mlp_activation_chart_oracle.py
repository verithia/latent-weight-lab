from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_activation_chart_oracle import (
    ActivationChart,
    prediction_metrics,
)


def test_common_gauge_centered_log_scales() -> None:
    chart = ActivationChart(4, "common_gauge_centered")
    with torch.no_grad():
        chart.common.fill_(0.25)
        chart.gauge.fill_(0.10)
        chart.centered.copy_(torch.tensor([-3.0, -1.0, 1.0, 3.0]))
    pre, post = chart.log_scales()
    assert torch.allclose(pre.mean(), torch.tensor(0.15))
    assert torch.allclose(post.mean(), torch.tensor(0.35))
    assert torch.allclose(post - pre, torch.full((4,), 0.20))
    assert torch.allclose(
        pre - pre.mean(), chart.centered - chart.centered.mean()
    )


def test_zero_chart_is_identity() -> None:
    generator = torch.Generator().manual_seed(7)
    pre_gelu = torch.randn(5, 8, generator=generator)
    weight = torch.randn(3, 8, generator=generator)
    expected = torch.nn.functional.linear(
        torch.nn.functional.gelu(pre_gelu), weight
    )
    for family in (
        "global_common",
        "global_common_gauge",
        "centered_common",
        "common_gauge_centered",
        "independent_channels",
    ):
        chart = ActivationChart(8, family)
        actual = chart(pre_gelu, weight, None)
        assert torch.equal(actual, expected)


def test_identity_error_recovery() -> None:
    target = torch.tensor([[1.0, 2.0]])
    identity = torch.tensor([[0.0, 0.0]])
    halfway = torch.tensor([[0.5, 1.0]])
    metrics = prediction_metrics(target, halfway, identity)
    assert abs(metrics["identity_error_recovery"] - 0.75) < 1e-7
