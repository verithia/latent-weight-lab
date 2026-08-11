from __future__ import annotations

import math

import torch

from examples.nanogpt.analyze_sparse_moe_cfc_learned_butterfly_frame_oracle import (
    LearnedButterflyCFC,
    butterfly_angle_count,
    butterfly_transform,
    candidate_coordinate_count,
    fit_butterfly_state,
    result_authorization,
)


def test_registered_coordinate_budget_is_exactly_256x() -> None:
    coordinates = candidate_coordinate_count(
        layers=12,
        experts=8,
        input_padded_width=1024,
        hidden_padded_width=2048,
        hidden_width=1536,
    )
    assert butterfly_angle_count(1024) == 5120
    assert butterfly_angle_count(2048) == 11264
    assert coordinates == 442368
    assert (12 * 8 * 1536 * 768) / coordinates == 256.0


def test_butterfly_preserves_norm_and_gradients() -> None:
    generator = torch.Generator().manual_seed(3)
    values = torch.randn(5, 16, generator=generator)
    angles = torch.randn(4, 8, generator=generator, requires_grad=True)
    output = butterfly_transform(values, angles)
    torch.testing.assert_close(
        output.square().sum(dim=-1),
        values.square().sum(dim=-1),
        atol=2e-5,
        rtol=2e-5,
    )
    output.square().mean().backward()
    assert angles.grad is not None and torch.isfinite(angles.grad).all()


def _small_operator() -> LearnedButterflyCFC:
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


def test_initial_state_is_finite_full_rank_map() -> None:
    operator = _small_operator()
    state = operator.initial_state(requires_grad=False)
    inputs = torch.eye(4).repeat(2, 1, 1)
    preactivation = operator.preactivation(inputs, state)
    assert preactivation.shape == (2, 4, 6)
    assert torch.isfinite(preactivation).all()
    assert torch.linalg.matrix_rank(preactivation[0]).item() == 4
    torch.testing.assert_close(
        state.input_angles,
        torch.full_like(state.input_angles, math.pi / 4),
    )


def test_fit_reduces_synthetic_objective() -> None:
    torch.manual_seed(11)
    operator = _small_operator()
    inputs = torch.randn(2, 24, 4)
    c_fc = torch.randn(2, 6, 4) * 0.1
    c_proj = torch.randn(2, 4, 6) * 0.1
    _state, diagnostics = fit_butterfly_state(
        operator,
        inputs,
        c_fc,
        c_proj,
        steps=20,
        learning_rate=0.03,
        weight_decay=0.0,
        gradient_clip=10.0,
    )
    assert diagnostics["final_loss"] < diagnostics["initial_loss"]
    assert math.isfinite(diagnostics["maximum_preclip_gradient_norm"])


def test_authorization_requires_separate_training_gate() -> None:
    passed = result_authorization(True)
    rejected = result_authorization(False)
    assert passed["implementation"]
    assert passed["initialization_and_mapping_loss_shadow"]
    assert not passed["mfu_preflight"]
    assert not passed["language_model_training"]
    assert not rejected["implementation"]
