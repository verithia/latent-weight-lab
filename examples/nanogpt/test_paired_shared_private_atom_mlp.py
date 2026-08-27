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
        compact_native_mlp="paired_shared_private_atoms236x10",
        compact_native_mlp_shared_width=8,
        compact_native_mlp_private_width=2,
        compact_native_mlp_seed=31,
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
        compact_native_mlp="paired_shared_private_atoms236x10",
        compact_native_mlp_shared_width=236,
        compact_native_mlp_private_width=10,
        compact_native_mlp_seed=20260830,
    )
    values.update(updates)
    return GPTConfig(**values)


def test_frozen_production_state_and_single_bank_registration() -> None:
    model = GPT(production_config())
    bank = model.shared_compact_mlp_paired_atom_bank
    assert bank is not None
    assert all(block.mlp.shared_bank is bank for block in model.transformer.h)
    local = sum(
        parameter.numel()
        for block in model.transformer.h
        for parameter in block.mlp.parameters()
    )
    shared = sum(parameter.numel() for parameter in bank.parameters())
    dense = 12 * 8 * 768 * 768
    assert local == 196_368
    assert shared == 362_496
    assert local + shared == 558_864
    assert abs((local + shared) / dense - 0.009869893391927084) < 1e-15
    keys = set(model.state_dict())
    assert "shared_compact_mlp_paired_atom_bank.input_weight" in keys
    assert "shared_compact_mlp_paired_atom_bank.output_weight" in keys
    assert not any("mlp._shared_bank" in key for key in keys)
    assert not any(
        "permutation" in key
        or key.endswith("mlp.c_fc.weight")
        or key.endswith("mlp.c_proj.weight")
        for key in keys
    )


def test_folded_shared_path_matches_activation_formula_and_gradients() -> None:
    torch.manual_seed(17)
    model = GPT(tiny_config(n_layer=1)).eval()
    mlp = model.transformer.h[0].mlp
    x = torch.randn(2, 3, 16)

    shared_hidden = F.gelu(
        F.linear(x, mlp._fold_input_weight(mlp.shared_bank.input_weight))
    )
    shared_hidden = shared_hidden * mlp.shared_hidden_gain
    actual = F.linear(
        shared_hidden,
        mlp._fold_output_weight(mlp.shared_bank.output_weight),
    )
    reference = mlp.activation_space_shared_write(x)
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)

    parameters = (
        mlp.shared_bank.input_weight,
        mlp.shared_bank.output_weight,
        mlp.shared_hidden_gain,
    )
    actual_grads = torch.autograd.grad(actual.square().sum(), parameters)
    reference_grads = torch.autograd.grad(reference.square().sum(), parameters)
    for actual_grad, reference_grad in zip(actual_grads, reference_grads):
        torch.testing.assert_close(
            actual_grad,
            reference_grad,
            rtol=2e-5,
            atol=2e-6,
        )


def test_full_formula_includes_private_input_and_output_atom_pair() -> None:
    torch.manual_seed(19)
    model = GPT(tiny_config(n_layer=1)).eval()
    mlp = model.transformer.h[0].mlp
    x = torch.randn(2, 3, 16)
    private = F.linear(
        F.gelu(F.linear(x, mlp.private_input_weight)),
        mlp.private_output_weight,
    )
    expected = (
        mlp.activation_space_shared_write(x) + private
    ) * mlp.output_gain
    torch.testing.assert_close(mlp(x), expected, rtol=1e-5, atol=1e-6)


def test_forward_backward_freeze_and_private_pair_gradients_noncollapsed() -> None:
    config = production_config(
        block_fht=True,
        block_fht_targets=("attn.c_attn.q", "attn.c_attn.k"),
        block_fht_match_gpt_init=True,
    )
    model = GPT(config)
    first = model.transformer.h[0].mlp
    second = model.transformer.h[1].mlp
    assert not torch.equal(first.permutation, second.permutation)
    freeze_non_block_fht(model)
    tokens = torch.randint(0, config.vocab_size, (2, config.block_size))
    _, loss = model(tokens, tokens)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    compact = [
        *model.shared_compact_mlp_paired_atom_bank.parameters(),
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
    for name in ("private_input_weight", "private_output_weight"):
        first_gradient = getattr(first, name).grad.flatten()
        second_gradient = getattr(second, name).grad.flatten()
        assert torch.linalg.vector_norm(first_gradient) > 0
        assert torch.linalg.vector_norm(second_gradient) > 0
        assert abs(F.cosine_similarity(first_gradient, second_gradient, dim=0)) < 0.999


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


def test_all_atom_matrices_use_muon_and_gains_use_adamw() -> None:
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
    bank = model.shared_compact_mlp_paired_atom_bank
    assert id(bank.input_weight) in matrix_ids
    assert id(bank.output_weight) in matrix_ids
    for block in model.transformer.h:
        assert id(block.mlp.private_input_weight) in matrix_ids
        assert id(block.mlp.private_output_weight) in matrix_ids
        assert id(block.mlp.shared_hidden_gain) not in matrix_ids
        assert id(block.mlp.output_gain) not in matrix_ids
