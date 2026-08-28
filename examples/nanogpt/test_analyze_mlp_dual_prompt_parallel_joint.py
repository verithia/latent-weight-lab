from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_dual_prompt_parallel_joint import (
    dual_prompt_accounting,
    make_generic_parallel_program,
    split_registered_prompt,
)


def test_dual_prompt_accounting_is_below_one_percent() -> None:
    accounting = dual_prompt_accounting(768)
    assert accounting["prompt_scalars"] == 565_248
    assert accounting["amplitude_scalars"] == 48
    assert accounting["total_scalars"] == 565_296
    assert accounting["deployable_scalar_fraction"] < 0.01


def test_registered_prompt_split_is_disjoint() -> None:
    prompt = torch.arange(737 * 3, dtype=torch.float32).reshape(1, 737, 3)
    targets = torch.arange(737).reshape(1, 737)
    prompts, target_parts, manifest = split_registered_prompt(prompt, targets)
    assert prompts[0].shape == prompts[1].shape == (1, 368, 3)
    assert target_parts[0][0, -1].item() == 367
    assert target_parts[1][0, 0].item() == 369
    assert manifest["omitted_source_positions"] == [368]


def test_parallel_program_adds_both_branches() -> None:
    torch.manual_seed(19)
    weights = (torch.randn(7, 5), torch.randn(4, 7))
    prompt1 = torch.randn(1, 5, 5)
    prompt2 = torch.randn(1, 6, 5)
    target1 = torch.randn(1, 5, 4)
    target2 = torch.randn(1, 6, 4)

    def make_loss(target: torch.Tensor):
        def loss_fn(active: tuple[torch.Tensor, ...], values: torch.Tensor) -> torch.Tensor:
            hidden = torch.nn.functional.gelu(values @ active[0].T)
            prediction = hidden @ active[1].T
            return 0.5 * (prediction - target).square().mean()
        return loss_fn

    function = make_generic_parallel_program(
        weights, (make_loss(target1), make_loss(target2)), ns_steps=5
    )
    both = function(prompt1, prompt2, torch.ones(2, 2))
    first = function(prompt1, prompt2, torch.tensor([[1.0, 1.0], [0.0, 0.0]]))
    second = function(prompt1, prompt2, torch.tensor([[0.0, 0.0], [1.0, 1.0]]))
    torch.testing.assert_close(both, first + second)


def test_parallel_program_jvp_is_finite() -> None:
    torch.manual_seed(23)
    weights = (torch.randn(7, 5), torch.randn(4, 7))
    prompt1 = torch.randn(1, 5, 5)
    prompt2 = torch.randn(1, 6, 5)
    targets = (torch.randn(1, 5, 4), torch.randn(1, 6, 4))

    def make_loss(target: torch.Tensor):
        def loss_fn(active: tuple[torch.Tensor, ...], values: torch.Tensor) -> torch.Tensor:
            hidden = torch.nn.functional.gelu(values @ active[0].T)
            prediction = hidden @ active[1].T
            return 0.5 * (prediction - target).square().mean()
        return loss_fn

    function = make_generic_parallel_program(
        weights, (make_loss(targets[0]), make_loss(targets[1])), ns_steps=5
    )
    amplitudes = torch.ones(2, 2)
    value, tangent = torch.func.jvp(
        function,
        (prompt1, prompt2, amplitudes),
        (torch.randn_like(prompt1), torch.randn_like(prompt2), torch.randn_like(amplitudes)),
    )
    assert value.shape == tangent.shape == (63,)
    assert torch.isfinite(value).all()
    assert torch.isfinite(tangent).all()
