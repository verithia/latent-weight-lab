from __future__ import annotations

import torch

from examples.nanogpt.analyze_sparse_moe_cproj_context_modulated_fht_oracle import (
    ContextModulatedCProjOperator,
    ExpertFrame,
    action_cosine,
    cgls_fit,
)
from examples.nanogpt.analyze_sparse_moe_cproj_context_residual_decomposition import (
    DirectSumResidualOperator,
)


def _direct_sum() -> DirectSumResidualOperator:
    torch.manual_seed(81)
    frame = ExpertFrame(
        tokens=torch.arange(32),
        probabilities=torch.ones(32),
        hidden=torch.randn(32, 8),
    )
    arguments = dict(
        frames=[frame], token_count=32, output_width=4, factors=2,
        seed=211, layer=0, device="cpu",
    )
    static = ContextModulatedCProjOperator(beta=0.0, **arguments)
    coupled = ContextModulatedCProjOperator(beta=1.0, **arguments)
    return DirectSumResidualOperator(static, coupled, static_factors=1)


def test_direct_sum_adjoint_identity() -> None:
    operator = _direct_sum()
    coordinates = torch.randn(operator.coordinate_shape)
    cotangent = torch.randn(operator.token_count, operator.output_width)
    left = (operator.apply(coordinates) * cotangent).sum()
    right = (coordinates * operator.adjoint(cotangent)).sum()
    torch.testing.assert_close(left, right, atol=3e-5, rtol=3e-5)


def test_direct_sum_cgls_recovers_synthetic_action() -> None:
    operator = _direct_sum()
    truth = torch.randn(operator.coordinate_shape)
    target = operator.apply(truth)
    fitted, diagnostics = cgls_fit(
        operator, target, maximum_iterations=100, tolerance=1e-7,
        ridge_ratio=1e-10, probe_seed=223,
    )
    assert action_cosine(operator.apply(fitted), target) > 0.999
    assert diagnostics["relative_normal_residual"] < 1e-4
