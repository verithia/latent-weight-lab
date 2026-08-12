from __future__ import annotations

import math

import torch

from examples.nanogpt.analyze_sparse_moe_coupled_biplane_transport_oracle import (
    CoupledBiplaneTransportAtom,
    _rotate_with_jvp,
    coordinate_count,
)


def test_coordinate_accounting_stays_above_200x() -> None:
    candidate = coordinate_count(
        experts=8, planes=2, input_width=768, hidden_width=1536,
        conditional=True,
    )
    control = coordinate_count(
        experts=8, planes=2, input_width=768, hidden_width=1536,
        conditional=False,
    )
    assert candidate == 79_936
    assert control == 30_720
    assert 18_874_368 / candidate > 200.0


def test_exact_plane_rotation_preserves_norm() -> None:
    generator = torch.Generator().manual_seed(17)
    value = torch.randn(2, 5, 8, generator=generator)
    first = torch.randn(2, 5, 8, generator=generator)
    v = torch.nn.functional.normalize(first, dim=-1)
    second = torch.randn(2, 5, 8, generator=generator)
    second = second - v * (v * second).sum(dim=-1, keepdim=True)
    w = torch.nn.functional.normalize(second, dim=-1)
    theta = torch.tensor([0.3, -0.7])
    rotated, _ = _rotate_with_jvp(
        value, torch.zeros_like(value), v, torch.zeros_like(v),
        w, torch.zeros_like(w), theta,
    )
    torch.testing.assert_close(
        rotated.square().sum(dim=-1), value.square().sum(dim=-1),
        atol=2e-5, rtol=2e-5,
    )


def test_analytic_input_jvp_matches_finite_difference() -> None:
    torch.manual_seed(23)
    module = CoupledBiplaneTransportAtom(
        experts=2, planes=2, input_width=8, hidden_width=8,
        padded_width=8, tensor_layers=2, seed=29, layer=0,
        device="cpu", conditional=True,
    )
    inputs = torch.randn(2, 4, 8)
    direction = torch.randn_like(inputs)
    _, analytic = module.function_and_jvp(
        inputs, direction, conditional=True
    )
    epsilon = 2e-3
    plus, _ = module.function_and_jvp(
        inputs + epsilon * direction, torch.zeros_like(direction),
        conditional=True,
    )
    minus, _ = module.function_and_jvp(
        inputs - epsilon * direction, torch.zeros_like(direction),
        conditional=True,
    )
    finite = (plus - minus) / (2.0 * epsilon)
    torch.testing.assert_close(analytic, finite, atol=3e-3, rtol=3e-2)


def test_candidate_and_control_have_finite_live_gradients() -> None:
    torch.manual_seed(31)
    for conditional in (False, True):
        module = CoupledBiplaneTransportAtom(
            experts=2, planes=2, input_width=8, hidden_width=8,
            padded_width=8, tensor_layers=2, seed=37, layer=1,
            device="cpu", conditional=conditional,
        )
        inputs = torch.randn(2, 3, 8)
        direction = torch.randn_like(inputs)
        output, jvp = module.function_and_jvp(
            inputs, direction, conditional=conditional
        )
        loss = output.square().mean() + 0.1 * jvp.square().mean()
        loss.backward()
        parameters = module.trainable_parameters(conditional=conditional)
        assert all(parameter.grad is not None for parameter in parameters)
        assert all(torch.isfinite(parameter.grad).all() for parameter in parameters)
        assert math.isfinite(float(loss))
