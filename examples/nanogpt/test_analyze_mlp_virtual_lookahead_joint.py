from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_virtual_lookahead_joint import (
    calibrate_lookahead_scales,
    lookahead_accounting,
    make_generic_lookahead_program,
)


def toy_problem():
    torch.manual_seed(29)
    weights = (torch.randn(7, 5), torch.randn(4, 7))
    prompt = torch.randn(1, 6, 5)
    target = torch.randn(1, 6, 4)

    def loss_fn(active: tuple[torch.Tensor, ...], values: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.gelu(values @ active[0].T)
        prediction = hidden @ active[1].T
        return 0.5 * (prediction - target).square().mean()

    return weights, prompt, loss_fn


def test_lookahead_accounting_is_below_one_percent() -> None:
    accounting = lookahead_accounting(737, 768)
    assert accounting["prompt_scalars"] == 566_016
    assert accounting["lookahead_scale_scalars"] == 24
    assert accounting["output_coefficient_scalars"] == 48
    assert accounting["total_scalars"] == 566_088
    assert accounting["deployable_scalar_fraction"] < 0.01


def test_lookahead_calibration_matches_first_step_norms() -> None:
    weights, prompt, loss_fn = toy_problem()
    target_norms = torch.tensor([0.08, 0.06])
    scales, manifest = calibrate_lookahead_scales(
        weights,
        loss_fn,
        prompt,
        target_norms,
        ns_steps=5,
        momentum=0.95,
    )
    assert tuple(scales.shape) == (2,)
    assert torch.isfinite(scales).all() and (scales > 0).all()
    assert manifest["maximum_norm_match_relative_error"] < 1e-6
    assert manifest["lookahead_is_not_output_scale"] is True


def test_lookahead_scale_and_output_mix_are_decoupled() -> None:
    weights, prompt, loss_fn = toy_problem()
    function = make_generic_lookahead_program(
        weights, loss_fn, ns_steps=5, momentum=0.95
    )
    coefficients = torch.ones(2, 2)
    near = function(prompt, torch.tensor([0.001, 0.001]), coefficients)
    far = function(prompt, torch.tensor([0.1, 0.1]), coefficients)
    assert not torch.allclose(near, far)
    first_only = function(
        prompt,
        torch.tensor([0.001, 0.001]),
        torch.tensor([[1.0, 1.0], [0.0, 0.0]]),
    )
    assert not torch.allclose(near, first_only)


def test_lookahead_program_jvp_is_finite() -> None:
    weights, prompt, loss_fn = toy_problem()
    function = make_generic_lookahead_program(
        weights, loss_fn, ns_steps=5, momentum=0.95
    )
    scales = torch.tensor([0.001, 0.001])
    coefficients = torch.ones(2, 2)
    value, tangent = torch.func.jvp(
        function,
        (prompt, scales, coefficients),
        (torch.randn_like(prompt), torch.randn_like(scales), torch.randn_like(coefficients)),
    )
    assert value.shape == tangent.shape == (63,)
    assert torch.isfinite(value).all()
    assert torch.isfinite(tangent).all()
