import torch
import torch.nn.functional as F

from examples.nanogpt.analyze_mlp_paired_gauge_oracle import (
    ChartSpec,
    PairedGaugeChart,
    parse_spec,
    source_prediction,
)


def test_paired_gauge_spec_parser() -> None:
    assert parse_spec("identity") == ChartSpec("identity", 0, 0)
    assert parse_spec("post2_out4") == ChartSpec("post", 2, 4)
    assert parse_spec("paired1_out0") == ChartSpec("paired", 1, 0)


def test_paired_hidden_rotation_cancels_in_linear_activation_limit() -> None:
    torch.manual_seed(41)
    chart = PairedGaugeChart(
        hidden_features=16,
        output_features=8,
        spec=ChartSpec("paired", 2, 0),
        rotation_block_size=4,
        basis_block_size=8,
        seed=17,
    )
    with torch.no_grad():
        chart.hidden_mixer.coordinates.normal_(std=0.1)
    values = torch.randn(3, 5, 8)
    c_fc_weight = torch.randn(16, 8)
    c_proj_weight = torch.randn(8, 16)
    pre = F.linear(values, c_fc_weight)
    paired = chart.hidden_mixer(chart.hidden_mixer.inverse(pre))
    observed = F.linear(paired, c_proj_weight)
    expected = F.linear(pre, c_proj_weight)
    torch.testing.assert_close(observed, expected, atol=3e-5, rtol=3e-5)


def test_paired_chart_is_exact_source_function_at_initialization() -> None:
    torch.manual_seed(43)
    chart = PairedGaugeChart(
        hidden_features=16,
        output_features=8,
        spec=ChartSpec("paired", 2, 2),
        rotation_block_size=4,
        basis_block_size=8,
        seed=19,
    )
    values = torch.randn(3, 5, 8)
    c_fc_weight = torch.randn(16, 8)
    c_proj_weight = torch.randn(8, 16)
    observed = chart(
        values,
        c_fc_weight,
        None,
        c_proj_weight,
        None,
    )
    expected = source_prediction(
        values,
        c_fc_weight,
        None,
        c_proj_weight,
        None,
    )
    torch.testing.assert_close(observed, expected)
