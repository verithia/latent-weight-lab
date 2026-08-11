from __future__ import annotations

import torch

from examples.nanogpt.analyze_sparse_moe_cfc_butterfly_expert_conflict import (
    UnsharedButterflyState,
    batched_butterfly_transform,
    classify,
    fit_unshared_state,
    gradient_conflict,
    unshared_preactivation,
)
from examples.nanogpt.analyze_sparse_moe_cfc_learned_butterfly_frame_oracle import (
    LearnedButterflyCFC,
)


def _operator() -> LearnedButterflyCFC:
    return LearnedButterflyCFC(
        experts=2,
        input_width=4,
        hidden_width=6,
        input_padded_width=4,
        hidden_padded_width=8,
        seed=7,
        layer=0,
        device="cpu",
    )


def test_batched_butterfly_preserves_each_expert_norm() -> None:
    generator = torch.Generator().manual_seed(3)
    values = torch.randn(3, 5, 16, generator=generator)
    angles = torch.randn(3, 4, 8, generator=generator, requires_grad=True)
    output = batched_butterfly_transform(values, angles)
    torch.testing.assert_close(
        output.square().sum(dim=-1),
        values.square().sum(dim=-1),
        atol=2e-5,
        rtol=2e-5,
    )
    output[..., 0].sum().backward()
    assert angles.grad is not None and torch.isfinite(angles.grad).all()


def test_unshared_clone_matches_shared_initial_function() -> None:
    operator = _operator()
    shared = operator.initial_state(requires_grad=False)
    unshared = UnsharedButterflyState.from_shared(
        shared, 2, "cpu", requires_grad=False
    )
    inputs = torch.randn(2, 7, 4)
    expected = operator.preactivation(inputs, shared)
    actual = unshared_preactivation(operator, inputs, unshared)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_unshared_fit_reduces_synthetic_objective() -> None:
    torch.manual_seed(11)
    operator = _operator()
    shared = operator.initial_state(requires_grad=False).cpu()
    inputs = torch.randn(2, 24, 4)
    c_fc = torch.randn(2, 6, 4) * 0.1
    c_proj = torch.randn(2, 4, 6) * 0.1
    _state, diagnostics = fit_unshared_state(
        operator,
        shared,
        inputs,
        c_fc,
        c_proj,
        steps=20,
        learning_rate=0.03,
        weight_decay=0.0,
        gradient_clip=10.0,
    )
    assert diagnostics["final_loss"] < diagnostics["initial_loss"]


def test_gradient_conflict_returns_finite_nonzero_expert_gradients() -> None:
    torch.manual_seed(17)
    operator = _operator()
    shared = operator.initial_state(requires_grad=False).cpu()
    inputs = torch.randn(2, 12, 4)
    c_fc = torch.randn(2, 6, 4) * 0.1
    c_proj = torch.randn(2, 4, 6) * 0.1
    diagnostics = gradient_conflict(operator, shared, inputs, c_fc, c_proj)
    assert diagnostics["finite_nonzero_gradient_count"] == 2
    assert -1.0 <= diagnostics["pairwise_cosine_mean"] <= 1.0
    assert 0.0 <= diagnostics["cancellation_ratio"] <= 1.0 + 1e-6


def test_classification_never_implies_implementation() -> None:
    assert classify(True, True) == "EXPERT_SHARING_CONFLICT_CONFIRMED"
    assert classify(False, False) == "ONE_SWEEP_BUTTERFLY_TOPOLOGY_INSUFFICIENT"
    assert classify(False, True) == "SHARING_DIAGNOSIS_AMBIGUOUS"
