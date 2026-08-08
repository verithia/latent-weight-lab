from __future__ import annotations

import torch

from examples.nanogpt.analyze_attention_residual_cubic_oracle import (
    classify_target,
    inverse_residual_cubic,
    residual_cubic,
)


def test_residual_cubic_inverse_round_trip() -> None:
    target = torch.linspace(-4.0, 4.0, 101, dtype=torch.float64)
    inverse = inverse_residual_cubic(target, 0.7)
    mapped, derivative = residual_cubic(inverse, 0.7)
    assert torch.allclose(mapped, target, atol=1e-11, rtol=1e-11)
    assert bool((derivative >= 1.0).all())


def test_residual_cubic_derivative_matches_autograd() -> None:
    value = torch.randn(29, dtype=torch.float64, requires_grad=True)
    mapped, derivative = residual_cubic(value, 1.3)
    gradient = torch.autograd.grad(mapped.sum(), value)[0]
    assert torch.allclose(gradient, derivative, atol=1e-12, rtol=1e-12)


def test_classification_requires_disjoint_gain_and_condition() -> None:
    thresholds = {
        "maximum_jacobian_diagonal": 10.0,
        "eval_functional_image_recovery_minimum": 0.8,
        "eval_cubic_tangent_recovery_minimum": 0.8,
        "eval_cubic_gain_over_identity_minimum": 0.05,
    }
    classification, checks = classify_target(
        {
            "maximum_jacobian_diagonal": 10.01,
            "eval_functional_image_recovery": 0.9,
            "eval_cubic_tangent_recovery": 0.9,
            "eval_cubic_gain_over_identity": 0.049,
        },
        thresholds,
    )
    assert classification == "ATTENTION_RESIDUAL_CUBIC_ORACLE_REJECT"
    assert checks == {
        "condition": False,
        "image": True,
        "tangent": True,
        "cubic_gain": False,
    }
