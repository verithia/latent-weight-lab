from __future__ import annotations

import torch
import torch.nn.functional as F

from examples.nanogpt.model import GPT, GPTConfig, freeze_non_block_fht
from examples.nanogpt.muon import Muon


def tiny_config(**updates) -> GPTConfig:
    values = dict(
        block_size=8,
        vocab_size=64,
        n_layer=2,
        n_head=2,
        n_embd=16,
        bias=False,
        compact_native_mlp="shared_recurrent_core_residual720x2",
        compact_native_mlp_core_width=12,
        compact_native_mlp_seed=29,
    )
    values.update(updates)
    return GPTConfig(**values)


def production_config(**updates) -> GPTConfig:
    values = dict(
        block_size=8,
        vocab_size=64,
        n_layer=12,
        n_head=12,
        n_embd=768,
        bias=False,
        compact_native_mlp="shared_recurrent_core_residual720x2",
        compact_native_mlp_core_width=720,
        compact_native_mlp_seed=20260829,
    )
    values.update(updates)
    return GPTConfig(**values)


def test_frozen_production_state_and_single_core_registration() -> None:
    model = GPT(production_config())
    core = model.shared_compact_mlp_recurrent_core
    assert core is not None
    assert all(block.mlp.shared_core is core for block in model.transformer.h)
    local = sum(
        parameter.numel()
        for block in model.transformer.h
        for parameter in block.mlp.parameters()
    )
    shared = sum(parameter.numel() for parameter in core.parameters())
    dense = 12 * 8 * 768 * 768
    assert local == 34_560
    assert shared == 518_400
    assert local + shared == 552_960
    assert abs((local + shared) / dense - 0.009765625) < 1e-15
    keys = set(model.state_dict())
    assert "shared_compact_mlp_recurrent_core.weight" in keys
    assert not any("mlp._shared_core" in key for key in keys)
    assert not any(
        "permutation" in key
        or key.endswith("mlp.c_fc.weight")
        or key.endswith("mlp.c_proj.weight")
        for key in keys
    )


def test_folded_recurrence_matches_activation_formula_and_gradients() -> None:
    torch.manual_seed(17)
    model = GPT(tiny_config(n_layer=1)).eval()
    mlp = model.transformer.h[0].mlp
    x = torch.randn(2, 3, 16)

    actual = mlp(x)
    reference_state = x
    for step in range(mlp.depth):
        reference_state = reference_state + mlp.activation_space_step(
            reference_state,
            step,
        )
    reference = reference_state - x
    torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-6)

    parameters = (
        mlp.shared_core.weight,
        *mlp.input_gain,
        *mlp.output_gain,
    )
    actual_grads = torch.autograd.grad(actual.square().sum(), parameters)
    reference_grads = torch.autograd.grad(reference.square().sum(), parameters)
    for actual_grad, reference_grad in zip(actual_grads, reference_grads):
        torch.testing.assert_close(
            actual_grad,
            reference_grad,
            rtol=4e-5,
            atol=4e-6,
        )


def test_outer_residual_is_not_double_counted() -> None:
    model = GPT(tiny_config(n_layer=1)).eval()
    mlp = model.transformer.h[0].mlp
    x = torch.randn(2, 3, 16)
    state = x
    for step in range(mlp.depth):
        state = state + mlp.activation_space_step(state, step)
    torch.testing.assert_close(x + mlp(x), state, rtol=2e-5, atol=2e-6)


def test_forward_backward_freeze_and_private_gradients_are_noncollapsed() -> None:
    config = production_config(
        block_fht=True,
        block_fht_targets=("attn.c_attn.q", "attn.c_attn.k"),
        block_fht_match_gpt_init=True,
    )
    model = GPT(config)
    first = model.transformer.h[0].mlp
    second = model.transformer.h[1].mlp
    assert not torch.equal(first.permutations[0], first.permutations[1])
    assert not torch.equal(first.permutations[0], second.permutations[0])
    freeze_non_block_fht(model)
    tokens = torch.randint(0, config.vocab_size, (2, config.block_size))
    _, loss = model(tokens, tokens)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    compact = [
        *model.shared_compact_mlp_recurrent_core.parameters(),
        *(
            parameter
            for block in model.transformer.h
            for parameter in block.mlp.parameters()
        ),
    ]
    assert all(parameter.requires_grad for parameter in compact)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in compact
    )
    first_gradient = torch.cat(
        [parameter.grad.flatten() for parameter in first.input_gain]
    )
    second_gradient = torch.cat(
        [parameter.grad.flatten() for parameter in second.input_gain]
    )
    assert torch.linalg.vector_norm(first_gradient) > 0
    assert torch.linalg.vector_norm(second_gradient) > 0
    assert abs(F.cosine_similarity(first_gradient, second_gradient, dim=0)) < 0.999
    assert all(
        torch.linalg.vector_norm(first.input_gain[step].grad) > 0
        for step in range(first.depth)
    )


def test_checkpoint_round_trip_is_exact() -> None:
    config = tiny_config()
    torch.manual_seed(5)
    first = GPT(config).eval()
    torch.manual_seed(11)
    second = GPT(config).eval()
    second.load_state_dict(first.state_dict())
    tokens = torch.randint(0, config.vocab_size, (2, config.block_size))
    with torch.no_grad():
        first_logits, _ = first(tokens)
        second_logits, _ = second(tokens)
    torch.testing.assert_close(first_logits, second_logits, rtol=0, atol=0)


def test_core_uses_muon_and_gains_use_adamw() -> None:
    model = GPT(production_config())
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=0.0024,
        betas=(0.9, 0.95),
        device_type="cpu",
        optimizer="muon",
    )
    optimizers = getattr(optimizer, "optimizers", [optimizer])
    muon = [item for item in optimizers if isinstance(item, Muon)]
    assert len(muon) == 1
    matrix_ids = {
        id(parameter)
        for group in muon[0].param_groups
        for parameter in group["params"]
    }
    assert id(model.shared_compact_mlp_recurrent_core.weight) in matrix_ids
    assert all(
        id(parameter) not in matrix_ids
        for block in model.transformer.h
        for parameter in (*block.mlp.input_gain, *block.mlp.output_gain)
    )
