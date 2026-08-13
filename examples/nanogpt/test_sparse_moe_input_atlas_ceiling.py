from __future__ import annotations

import json
from pathlib import Path

import torch

from examples.nanogpt.analyze_sparse_moe_input_atlas_ceiling import (
    equal_node_input_atlas,
    gelu_derivative,
    metric_recovery,
    oracle_function_and_jvp,
    ridge_coefficients,
)


def test_gelu_derivative_matches_finite_difference() -> None:
    values = torch.linspace(-3.0, 3.0, 101)
    epsilon = 1e-3
    finite = (torch.nn.functional.gelu(values + epsilon) - torch.nn.functional.gelu(values - epsilon)) / (2 * epsilon)
    torch.testing.assert_close(gelu_derivative(values), finite, rtol=2e-3, atol=2e-4)


def test_equal_node_atlas_normalizes_node_trace() -> None:
    first = torch.diag(torch.tensor([100.0, 0.0, 0.0, 0.0]))
    second = torch.diag(torch.tensor([0.0, 1.0, 0.0, 0.0]))
    basis, _ = equal_node_input_atlas({0: torch.stack((first, second))}, 2, "cpu")
    recovery = metric_recovery({0: torch.stack((first, second))}, basis)
    assert recovery["jacobian_energy_recovery"] > 0.999999


def test_dense_coefficients_reconstruct_weights_inside_atlas() -> None:
    generator = torch.Generator().manual_seed(17)
    width, rank, hidden, experts, samples = 12, 7, 15, 2, 64
    basis, _ = torch.linalg.qr(torch.randn(width, rank, generator=generator))
    basis = basis.T.contiguous()
    truth = torch.randn(experts, rank, hidden, generator=generator)
    c_fc = torch.einsum("erh,rd->ehd", truth, basis)
    x = torch.randn(experts, samples, width, generator=generator)
    fitted, diagnostics = ridge_coefficients(
        x, c_fc, basis, ridge_scale=1e-10, device="cpu"
    )
    assert diagnostics["minimum_preactivation_recovery"] > 0.999999
    torch.testing.assert_close(fitted, truth, rtol=3e-4, atol=3e-4)


def test_oracle_analytic_jvp_matches_finite_difference() -> None:
    generator = torch.Generator().manual_seed(19)
    width, rank, hidden, samples = 10, 6, 14, 8
    basis, _ = torch.linalg.qr(torch.randn(width, rank, generator=generator))
    basis = basis.T.contiguous()
    coefficients = torch.randn(rank, hidden, generator=generator) * 0.2
    c_proj = torch.randn(width, hidden, generator=generator) * 0.2
    x = torch.randn(samples, width, generator=generator)
    direction = torch.randn(samples, width, generator=generator)
    _output, jvp = oracle_function_and_jvp(
        x, direction, basis, coefficients, c_proj
    )
    epsilon = 1e-3
    plus, _ = oracle_function_and_jvp(
        x + epsilon * direction, torch.zeros_like(x), basis, coefficients, c_proj
    )
    minus, _ = oracle_function_and_jvp(
        x - epsilon * direction, torch.zeros_like(x), basis, coefficients, c_proj
    )
    finite = (plus - minus) / (2 * epsilon)
    torch.testing.assert_close(jvp, finite, rtol=6e-3, atol=8e-5)


def test_preregistered_plan_has_no_candidate_authorization() -> None:
    path = Path(__file__).parent / "configs" / "selection_artifacts" / "124m_sparse_moe_input_atlas_ceiling_plan.json"
    plan = json.loads(path.read_text(encoding="utf-8"))
    assert plan["identity"]["theory_preregistration_git_commit"] == (
        "72bd4b0dd17c43df67b1c7d66707ebbafe9f0ecc"
    )
    assert plan["authorization"]["compact_candidate"] is False
    assert plan["authorization"]["language_model_training"] is False
