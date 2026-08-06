from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_cproj_modulation_residual_oracle import (
    cg_solve,
    fit_hidden_additive,
    fit_hidden_gain,
    fit_output_additive,
    fit_output_gain,
    fit_paper_literal_dc,
    fit_two_way_additive,
    fit_two_way_gain,
)


COMMON = {"ridge_ratio": 1e-10, "cg_iterations": 96, "alternating_iterations": 8}


def _relative_output_error(hidden: torch.Tensor, target: torch.Tensor, delta: torch.Tensor) -> float:
    predicted = hidden @ delta.T
    return float((target - predicted).norm() / target.norm().clamp_min(1e-30))


def test_cg_solves_spd_system() -> None:
    matrix = torch.tensor([[4.0, 1.0], [1.0, 3.0]])
    rhs = torch.tensor([1.0, 2.0])
    actual = cg_solve(lambda value: matrix @ value, rhs, ridge=0.0, iterations=8)
    expected = torch.linalg.solve(matrix, rhs)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_literal_and_output_families_recover_exact_outputs() -> None:
    torch.manual_seed(1)
    hidden = torch.randn(48, 7)
    weight = torch.randn(5, 7)
    literal = torch.full_like(weight, 0.125)
    literal_target = hidden @ literal.T
    fitted_literal = fit_paper_literal_dc(hidden, literal_target, weight, **COMMON)
    assert _relative_output_error(hidden, literal_target, fitted_literal) < 1e-5

    additive = torch.randn(5, 1).expand_as(weight).clone()
    additive_target = hidden @ additive.T
    fitted_additive = fit_output_additive(hidden, additive_target, weight, **COMMON)
    assert _relative_output_error(hidden, additive_target, fitted_additive) < 1e-5

    gain = torch.randn(5, 1) * weight
    gain_target = hidden @ gain.T
    fitted_gain = fit_output_gain(hidden, gain_target, weight, **COMMON)
    assert _relative_output_error(hidden, gain_target, fitted_gain) < 1e-5


def test_hidden_and_two_way_families_recover_exact_outputs() -> None:
    torch.manual_seed(2)
    hidden = torch.randn(64, 6)
    weight = torch.randn(4, 6)

    hidden_additive = torch.randn(1, 6).expand_as(weight).clone()
    target = hidden @ hidden_additive.T
    fitted = fit_hidden_additive(hidden, target, weight, **COMMON)
    assert _relative_output_error(hidden, target, fitted) < 2e-4

    hidden_gain = weight * torch.randn(1, 6)
    target = hidden @ hidden_gain.T
    fitted = fit_hidden_gain(hidden, target, weight, **COMMON)
    assert _relative_output_error(hidden, target, fitted) < 2e-4

    two_way_additive = torch.randn(4, 1) + torch.randn(1, 6)
    target = hidden @ two_way_additive.T
    fitted = fit_two_way_additive(hidden, target, weight, **COMMON)
    assert _relative_output_error(hidden, target, fitted) < 2e-4

    two_way_gain = weight * (torch.randn(4, 1) + torch.randn(1, 6))
    target = hidden @ two_way_gain.T
    fitted = fit_two_way_gain(hidden, target, weight, **COMMON)
    assert _relative_output_error(hidden, target, fitted) < 2e-4
