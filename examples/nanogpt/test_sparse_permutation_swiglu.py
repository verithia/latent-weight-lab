from __future__ import annotations

import torch

from examples.nanogpt.model import (
    GPT,
    GPTConfig,
    SparsePermutationLinear,
    SparsePermutationSwiGLUMLP,
    freeze_non_block_fht,
)


def tiny_config(**updates) -> GPTConfig:
    values = dict(
        block_size=8,
        vocab_size=64,
        n_layer=2,
        n_head=2,
        n_embd=16,
        bias=False,
        compact_native_mlp="sparse_permutation_swiglu4",
        compact_native_mlp_branches=4,
        compact_native_mlp_paths=3,
        compact_native_mlp_seed=1234,
    )
    values.update(updates)
    return GPTConfig(**values)


def test_sparse_map_has_procedural_connectivity_and_exact_state() -> None:
    module = SparsePermutationLinear(width=16, paths=3, seed=7, init_std=0.2)
    assert sum(parameter.numel() for parameter in module.parameters()) == 3 * 16
    assert set(module.state_dict()) == {"gains.0", "gains.1", "gains.2"}
    assert torch.equal(module.permutation_0, torch.arange(16))


def test_sparse_mlp_exact_budget_and_no_permutations_in_checkpoint() -> None:
    config = tiny_config()
    module = SparsePermutationSwiGLUMLP(config, layer_id=0)
    assert sum(parameter.numel() for parameter in module.parameters()) == 36 * 16
    assert not any("permutation" in name for name in module.state_dict())
    dense_mlp_parameters = 8 * config.n_embd * config.n_embd
    assert sum(parameter.numel() for parameter in module.parameters()) / dense_mlp_parameters == 36 / (8 * 16)


def test_sparse_mlp_is_deterministic_from_seed_and_state() -> None:
    config = tiny_config()
    torch.manual_seed(5)
    first = SparsePermutationSwiGLUMLP(config, layer_id=1)
    torch.manual_seed(99)
    second = SparsePermutationSwiGLUMLP(config, layer_id=1)
    second.load_state_dict(first.state_dict())
    x = torch.randn(2, 3, 16)
    assert torch.equal(first(x), second(x))


def test_sparse_mlp_forward_and_backward_are_finite() -> None:
    module = SparsePermutationSwiGLUMLP(tiny_config(), layer_id=0)
    x = torch.randn(2, 3, 16, requires_grad=True)
    loss = module(x).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
    )


def test_qk_blockfht_freeze_keeps_sparse_mlp_trainable() -> None:
    config = tiny_config(
        block_fht=True,
        block_fht_targets=("attn.c_attn.q", "attn.c_attn.k"),
        block_fht_match_gpt_init=True,
    )
    model = GPT(config)
    freeze_non_block_fht(model)
    sparse_parameters = [
        parameter
        for block in model.transformer.h
        for parameter in block.mlp.parameters()
    ]
    assert sparse_parameters
    assert all(parameter.requires_grad for parameter in sparse_parameters)
    tokens = torch.randint(0, config.vocab_size, (2, config.block_size))
    _, loss = model(tokens, tokens)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()


def test_frozen_production_shape_uses_point_586_percent_state() -> None:
    width = 768
    module = SparsePermutationSwiGLUMLP(
        tiny_config(n_embd=width, n_head=12, n_layer=12), layer_id=0
    )
    compact = sum(parameter.numel() for parameter in module.parameters())
    dense = 8 * width * width
    assert compact == 27_648
    assert abs(compact / dense - 0.005859375) < 1e-12
