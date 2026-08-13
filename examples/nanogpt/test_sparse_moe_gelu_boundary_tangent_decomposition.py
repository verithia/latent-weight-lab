from __future__ import annotations

import json
from pathlib import Path

import torch

from examples.nanogpt.analyze_sparse_moe_gelu_boundary_tangent_decomposition import (
    gelu_second_derivative,
    tangent_counterfactuals,
)
from examples.nanogpt.analyze_sparse_moe_input_atlas_ceiling import gelu_derivative


def test_gelu_second_derivative_matches_finite_difference() -> None:
    values = torch.linspace(-3.0, 3.0, 101)
    epsilon = 1e-3
    finite = (gelu_derivative(values + epsilon) - gelu_derivative(values - epsilon)) / (2 * epsilon)
    torch.testing.assert_close(gelu_second_derivative(values), finite, rtol=3e-3, atol=3e-4)


def test_counterfactuals_collapse_at_exact_teacher() -> None:
    generator = torch.Generator().manual_seed(29)
    width, hidden, samples = 8, 12, 7
    basis = torch.eye(width)
    c_fc = torch.randn(hidden, width, generator=generator) * 0.2
    c_proj = torch.randn(width, hidden, generator=generator) * 0.2
    coefficients = c_fc.T
    x = torch.randn(samples, width, generator=generator)
    direction = torch.randn(samples, width, generator=generator)
    rows = tangent_counterfactuals(x, direction, c_fc, c_proj, basis, coefficients)
    for key in (
        "teacher_gate_projected_normal",
        "projected_gate_dense_normal",
        "self_consistent",
        "first_order_normal_plus_boundary",
    ):
        torch.testing.assert_close(rows[key], rows["dense_target"])
    assert torch.count_nonzero(rows["normal_term"]) == 0
    assert torch.count_nonzero(rows["boundary_term"]) == 0


def test_first_order_terms_predict_small_normal_perturbation() -> None:
    generator = torch.Generator().manual_seed(31)
    width, hidden, samples = 7, 11, 9
    basis = torch.eye(width)
    c_fc = torch.randn(hidden, width, generator=generator) * 0.2
    c_proj = torch.randn(width, hidden, generator=generator) * 0.2
    delta = torch.randn(c_fc.shape, generator=generator) * 1e-4
    coefficients = (c_fc + delta).T
    x = torch.randn(samples, width, generator=generator)
    direction = torch.randn(samples, width, generator=generator)
    rows = tangent_counterfactuals(x, direction, c_fc, c_proj, basis, coefficients)
    torch.testing.assert_close(
        rows["first_order_normal_plus_boundary"],
        rows["self_consistent"],
        rtol=5e-3,
        atol=2e-6,
    )


def test_plan_is_diagnostic_only() -> None:
    path = Path(__file__).parent / "configs" / "selection_artifacts" / "124m_sparse_moe_gelu_boundary_tangent_decomposition_plan.json"
    plan = json.loads(path.read_text(encoding="utf-8"))
    assert plan["identity"]["theory_preregistration_git_commit"] == (
        "5c9e85f5ccb8ccbf81fc5509c53d7195fc0c4e63"
    )
    assert plan["authorization"]["new_parameter_candidate"] is False
    assert plan["authorization"]["language_model_training"] is False
