from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_synthetic_muon_momentum_twostep_joint import (
    calibrate_momentum_amplitudes,
    make_generic_momentum_program,
)
from examples.nanogpt.analyze_mlp_synthetic_muon_program_twostep_joint import (
    make_generic_two_step_program,
)


def toy_problem() -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor, object]:
    torch.manual_seed(17)
    weights = (torch.randn(7, 5), torch.randn(4, 7))
    prompt = torch.randn(1, 6, 5)
    target = torch.randn(1, 6, 4)

    def loss_fn(active: tuple[torch.Tensor, ...], values: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.gelu(values @ active[0].T)
        prediction = hidden @ active[1].T
        return 0.5 * (prediction - target).square().mean()

    return weights, prompt, target, loss_fn


def test_zero_momentum_matches_memoryless_program() -> None:
    weights, prompt, _, loss_fn = toy_problem()
    amplitudes = torch.tensor([[0.03, 0.02], [0.04, 0.01]])
    memoryless = make_generic_two_step_program(weights, loss_fn, ns_steps=5)
    zero_momentum = make_generic_momentum_program(
        weights, loss_fn, ns_steps=5, momentum=0.0
    )
    torch.testing.assert_close(memoryless(prompt, amplitudes), zero_momentum(prompt, amplitudes))


def test_parent_momentum_changes_second_step_program() -> None:
    weights, prompt, _, loss_fn = toy_problem()
    amplitudes = torch.tensor([[0.03, 0.02], [0.04, 0.01]])
    memoryless = make_generic_two_step_program(weights, loss_fn, ns_steps=5)
    momentum = make_generic_momentum_program(weights, loss_fn, ns_steps=5, momentum=0.95)
    assert not torch.allclose(memoryless(prompt, amplitudes), momentum(prompt, amplitudes))


def test_momentum_calibration_matches_step_norms() -> None:
    weights, prompt, _, loss_fn = toy_problem()
    target_norms = torch.tensor([[0.08, 0.06], [0.05, 0.04]])
    amplitudes, manifest = calibrate_momentum_amplitudes(
        weights,
        loss_fn,
        prompt,
        target_norms,
        ns_steps=5,
        momentum=0.95,
    )
    assert tuple(amplitudes.shape) == (2, 2)
    assert torch.isfinite(amplitudes).all()
    assert (amplitudes > 0).all()
    assert manifest["maximum_norm_match_relative_error"] < 1e-6
    assert manifest["stores_dense_direction_or_buffer"] is False


def test_momentum_program_jvp_is_finite() -> None:
    weights, prompt, _, loss_fn = toy_problem()
    amplitudes = torch.tensor([[0.03, 0.02], [0.04, 0.01]])
    function = make_generic_momentum_program(
        weights, loss_fn, ns_steps=5, momentum=0.95
    )
    value, tangent = torch.func.jvp(
        function,
        (prompt, amplitudes),
        (torch.randn_like(prompt), torch.randn_like(amplitudes)),
    )
    assert value.shape == tangent.shape == (63,)
    assert torch.isfinite(value).all()
    assert torch.isfinite(tangent).all()
