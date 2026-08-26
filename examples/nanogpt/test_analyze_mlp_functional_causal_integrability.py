from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_functional_causal_integrability import (
    direction_metrics,
    gate_outcome,
    gelu_derivative,
    mlp_input_jvp,
    mlp_output,
    right_project,
    summarize,
    truncated_svd_factors,
    truncated_svd_reconstruct,
)


def test_direction_metrics_distinguishes_scale_and_line() -> None:
    target = torch.tensor([1.0, 2.0])
    prediction = 2.0 * target
    result = direction_metrics(target, prediction)
    assert result["cosine"] > 0.999999
    assert result["positive_line_recovery"] > 0.999999
    assert result["fixed_scale_recovery"] < 0.0


def test_right_projection_is_exact() -> None:
    matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    basis = torch.tensor([[1.0], [0.0]])
    torch.testing.assert_close(
        right_project(matrix, basis),
        torch.tensor([[1.0, 0.0], [3.0, 0.0]]),
    )


def test_truncated_svd_reconstruction_has_requested_rank() -> None:
    matrix = torch.diag(torch.tensor([4.0, 3.0, 2.0]))
    factors = truncated_svd_factors(matrix, 2)
    rank_one = truncated_svd_reconstruct(*factors, 1)
    rank_two = truncated_svd_reconstruct(*factors, 2)
    assert torch.linalg.matrix_rank(rank_one) == 1
    assert torch.linalg.matrix_rank(rank_two) == 2
    torch.testing.assert_close(rank_two, torch.diag(torch.tensor([4.0, 3.0, 0.0])))


def test_mlp_jvp_matches_finite_difference() -> None:
    torch.manual_seed(7)
    inputs = torch.randn(4, 3)
    directions = torch.randn_like(inputs)
    c_fc = torch.randn(5, 3)
    c_proj = torch.randn(3, 5)
    c_fc_bias = torch.randn(5)
    c_proj_bias = torch.randn(3)
    analytic = mlp_input_jvp(inputs, directions, c_fc, c_proj, c_fc_bias)
    epsilon = 1e-3
    finite = (
        mlp_output(inputs + epsilon * directions, c_fc, c_proj, c_fc_bias, c_proj_bias)
        - mlp_output(inputs - epsilon * directions, c_fc, c_proj, c_fc_bias, c_proj_bias)
    ) / (2.0 * epsilon)
    torch.testing.assert_close(analytic, finite, rtol=2e-3, atol=2e-3)


def test_summary_and_gate_require_every_bank() -> None:
    rows = []
    for bank, recovery in ((0, 0.9), (1, 0.5)):
        for _index in range(2):
            row = {
                "chart_kind": "causal",
                "split": "test",
                "bank": bank,
                "union_rank": 6,
            }
            for kind in ("output", "jvp"):
                row[f"{kind}_cosine"] = 0.9
                row[f"{kind}_positive_line_recovery"] = 0.8
                row[f"{kind}_fixed_scale_recovery"] = recovery
            rows.append(row)
    table = summarize(rows)
    gate = gate_outcome(
        table,
        rank=6,
        output_mean_gate=0.8,
        output_minimum_gate=0.7,
        jvp_mean_gate=0.6,
    )
    assert gate["passed"] is False
    assert gate["banks"][0]["passed"] is True
    assert gate["banks"][1]["passed"] is False
