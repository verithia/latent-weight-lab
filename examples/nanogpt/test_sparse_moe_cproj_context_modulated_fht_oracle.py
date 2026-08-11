from __future__ import annotations

import torch

from examples.nanogpt.analyze_sparse_moe_cproj_context_modulated_fht_oracle import (
    ContextModulatedCProjOperator,
    ExpertFrame,
    action_cosine,
    cgls_fit,
)


def _operator(beta: float, factors: int = 2) -> ContextModulatedCProjOperator:
    torch.manual_seed(71)
    frame = ExpertFrame(
        tokens=torch.arange(24),
        probabilities=torch.ones(24),
        hidden=torch.randn(24, 8),
    )
    return ContextModulatedCProjOperator(
        [frame], token_count=24, output_width=4, factors=factors,
        beta=beta, seed=101, layer=0, device="cpu",
    )


def test_context_operator_adjoint_identity() -> None:
    operator = _operator(beta=1.0)
    coordinates = torch.randn(operator.coordinate_shape)
    cotangent = torch.randn(operator.token_count, operator.output_width)
    left = (operator.apply(coordinates) * cotangent).sum()
    right = (coordinates * operator.adjoint(cotangent)).sum()
    torch.testing.assert_close(left, right, atol=2e-5, rtol=2e-5)


def test_beta_zero_is_identical_coordinate_static_ablation() -> None:
    operator = _operator(beta=0.0)
    assert all(torch.equal(gate, torch.ones_like(gate)) for gate in operator.gates)
    coordinates = torch.randn(operator.coordinate_shape)
    assert torch.isfinite(operator.apply(coordinates)).all()


def test_cgls_recovers_synthetic_context_action() -> None:
    operator = _operator(beta=1.0, factors=1)
    truth = torch.randn(operator.coordinate_shape)
    target = operator.apply(truth)
    fitted, diagnostics = cgls_fit(
        operator,
        target,
        maximum_iterations=80,
        tolerance=1e-7,
        ridge_ratio=1e-10,
        probe_seed=103,
    )
    predicted = operator.apply(fitted)
    assert action_cosine(predicted, target) > 0.999
    assert diagnostics["relative_normal_residual"] < 1e-4
