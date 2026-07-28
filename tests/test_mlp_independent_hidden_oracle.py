import torch

from examples.nanogpt.analyze_mlp_independent_hidden_oracle import (
    ChartSpec,
    IndependentHiddenChart,
    parse_spec,
)
from examples.nanogpt.analyze_mlp_paired_gauge_oracle import (
    source_prediction,
)


def test_independent_hidden_spec_parser() -> None:
    assert parse_spec("identity") == ChartSpec(0, 0, 0)
    assert parse_spec("pre2_post1_out4") == ChartSpec(2, 1, 4)


def test_independent_hidden_chart_is_identity_at_initialization() -> None:
    torch.manual_seed(47)
    chart = IndependentHiddenChart(
        hidden_features=16,
        output_features=8,
        spec=ChartSpec(2, 1, 2),
        rotation_block_size=4,
        basis_block_size=8,
        seed=23,
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
