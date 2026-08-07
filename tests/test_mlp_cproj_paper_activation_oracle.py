from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_cproj_paper_activation_oracle import (
    activated_weight_and_derivative,
    activation_bias,
    activation_scale,
    cgls,
    classify,
)


def test_signed_tanh_bias_reproduces_step_zero_exactly() -> None:
    initial = torch.tensor([[-0.25, 0.0, 0.5], [0.1, -0.4, 0.2]])
    scale = activation_scale(initial)
    bias = activation_bias(initial, scale)
    reconstructed, derivative = activated_weight_and_derivative(bias, scale)
    torch.testing.assert_close(reconstructed, initial, atol=1e-7, rtol=1e-6)
    assert bool((derivative > 0).all())
    assert bool((derivative <= 1).all())


def test_signed_tanh_derivative_matches_finite_difference() -> None:
    preactivation = torch.tensor([-0.4, 0.2, 0.7], dtype=torch.float64)
    scale = torch.tensor(1.5, dtype=torch.float64)
    direction = torch.tensor([0.3, -0.2, 0.1], dtype=torch.float64)
    _, derivative = activated_weight_and_derivative(preactivation, scale)
    epsilon = 1e-6
    plus, _ = activated_weight_and_derivative(
        preactivation + epsilon * direction, scale
    )
    minus, _ = activated_weight_and_derivative(
        preactivation - epsilon * direction, scale
    )
    finite = (plus - minus) / (2 * epsilon)
    torch.testing.assert_close(finite, derivative * direction, atol=1e-9, rtol=1e-8)


def test_tight_condition_scale_has_derivative_floor_one_tenth() -> None:
    initial = torch.tensor([-2.0, 0.0, 1.0])
    scale = activation_scale(initial, multiplier=(10.0 / 9.0) ** 0.5)
    bias = activation_bias(initial, scale)
    _, derivative = activated_weight_and_derivative(bias, scale)
    torch.testing.assert_close(derivative.amin(), torch.tensor(0.1), atol=1e-6, rtol=1e-6)


def test_cgls_recovers_explicit_linear_system() -> None:
    matrix = torch.tensor(
        [[2.0, -1.0], [0.5, 1.0], [1.0, 1.5]], dtype=torch.float32
    )
    truth = torch.tensor([0.75, -0.5])
    target = matrix @ truth
    coordinate, prediction, _ = cgls(
        lambda value: matrix @ value,
        lambda value: matrix.T @ value,
        target,
        torch.zeros_like(truth),
        iterations=8,
    )
    torch.testing.assert_close(coordinate, truth, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(prediction, target, atol=1e-5, rtol=1e-5)


def test_classification_is_fail_closed() -> None:
    assert classify(False, 1.0, 1.0, 0.8) == "PAPER_ACTIVATION_RANGE_INVALID"
    assert classify(True, 0.79, 1.0, 0.8) == "PAPER_ACTIVATION_IMAGE_INSUFFICIENT"
    assert classify(True, 0.9, 0.79, 0.8) == "PAPER_ACTIVATION_TANGENT_INSUFFICIENT"
    assert classify(True, 0.9, 0.9, 0.8) == "PAPER_ACTIVATION_ORACLE_PASS"
