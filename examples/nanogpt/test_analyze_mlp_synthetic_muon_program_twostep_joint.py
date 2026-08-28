from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_synthetic_muon_program_twostep_joint import (
    calibrate_two_step_amplitudes,
    make_generic_two_step_program,
    two_step_latent_accounting,
)


def test_two_step_accounting_is_below_one_percent() -> None:
    accounting = two_step_latent_accounting(737, 768)
    assert accounting["prompt_scalars"] == 566_016
    assert accounting["amplitude_scalars"] == 48
    assert accounting["total_scalars"] == 566_064
    assert accounting["deployable_scalar_fraction"] < 0.01


def test_calibration_matches_both_step_norms() -> None:
    torch.manual_seed(9)
    weights = (torch.randn(7, 5), torch.randn(4, 7))
    prompt = torch.randn(1, 6, 5)
    target = torch.randn(1, 6, 4)

    def loss_fn(active: tuple[torch.Tensor, ...], values: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.gelu(values @ active[0].T)
        prediction = hidden @ active[1].T
        return 0.5 * (prediction - target).square().mean()

    target_norms = torch.tensor([[0.08, 0.06], [0.05, 0.04]])
    amplitudes, manifest = calibrate_two_step_amplitudes(
        weights, loss_fn, prompt, target_norms, ns_steps=5
    )
    assert tuple(amplitudes.shape) == (2, 2)
    assert torch.isfinite(amplitudes).all()
    assert (amplitudes > 0).all()
    assert manifest["maximum_norm_match_relative_error"] < 1e-6
    assert manifest["stores_dense_direction"] is False


def test_second_step_changes_with_first_step_state() -> None:
    torch.manual_seed(11)
    weights = (torch.randn(7, 5), torch.randn(4, 7))
    prompt = torch.randn(1, 6, 5)
    target = torch.randn(1, 6, 4)

    def loss_fn(active: tuple[torch.Tensor, ...], values: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.gelu(values @ active[0].T)
        prediction = hidden @ active[1].T
        return 0.5 * (prediction - target).square().mean()

    function = make_generic_two_step_program(weights, loss_fn, ns_steps=5)
    amplitudes = torch.tensor([[0.03, 0.02], [0.04, 0.01]])
    with_first = function(prompt, amplitudes)
    without_first = function(prompt, amplitudes * torch.tensor([[0.0], [1.0]]))
    assert not torch.allclose(with_first, without_first)


def test_two_step_program_jvp_is_finite() -> None:
    torch.manual_seed(13)
    weights = (torch.randn(7, 5), torch.randn(4, 7))
    prompt = torch.randn(1, 6, 5)
    target = torch.randn(1, 6, 4)

    def loss_fn(active: tuple[torch.Tensor, ...], values: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.gelu(values @ active[0].T)
        prediction = hidden @ active[1].T
        return 0.5 * (prediction - target).square().mean()

    function = make_generic_two_step_program(weights, loss_fn, ns_steps=5)
    amplitudes = torch.tensor([[0.03, 0.02], [0.04, 0.01]])
    value, tangent = torch.func.jvp(
        function,
        (prompt, amplitudes),
        (torch.randn_like(prompt), torch.randn_like(amplitudes)),
    )
    assert value.shape == tangent.shape == (63,)
    assert torch.isfinite(value).all()
    assert torch.isfinite(tangent).all()
