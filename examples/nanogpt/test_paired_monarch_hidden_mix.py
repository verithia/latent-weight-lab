from __future__ import annotations

import torch

from examples.nanogpt.model import (
    GPT,
    GPTConfig,
    LearnedMonarchHiddenMix,
)


def test_monarch_identity_and_transpose_are_exact_at_initialization() -> None:
    module = LearnedMonarchHiddenMix(
        features=32,
        block_width=8,
        seed=17,
        coordinate_scale=2.0,
    )
    values = torch.randn(5, 32)
    assert torch.equal(module(values), values)
    assert torch.equal(module.apply_transpose(values), values)

    with torch.no_grad():
        module.coordinates.normal_(mean=0.0, std=0.02)
    left = torch.randn(7, 32)
    right = torch.randn(7, 32)
    lhs = (module(left) * right).sum()
    rhs = (left * module.apply_transpose(right)).sum()
    torch.testing.assert_close(lhs, rhs, rtol=1e-5, atol=1e-5)


def _tiny_config(monarch_width: int) -> GPTConfig:
    return GPTConfig(
        block_size=8,
        vocab_size=64,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=False,
        block_fht=True,
        block_fht_targets=("mlp.c_fc", "mlp.c_proj"),
        block_fht_latent_ratio=0.25,
        block_fht_match_gpt_init=True,
        block_fht_mlp_paired_monarch_block_width=monarch_width,
        block_fht_mlp_paired_monarch_coordinate_scale=2.0,
    )


def test_paired_monarch_preserves_step_zero_function_and_gets_both_vjps() -> None:
    torch.manual_seed(123)
    control = GPT(_tiny_config(0))
    torch.manual_seed(123)
    candidate = GPT(_tiny_config(8))

    tokens = torch.randint(0, 64, (2, 8))
    control.prepare_block_fht_cache(dtype=torch.float32)
    candidate.prepare_block_fht_cache(dtype=torch.float32)
    try:
        control_logits, _ = control(tokens, tokens)
        candidate_logits, loss = candidate(tokens, tokens)
        torch.testing.assert_close(
            candidate_logits, control_logits, rtol=0.0, atol=0.0
        )
        assert loss is not None
        loss.backward()
    finally:
        control.flush_block_fht_cache()
        candidate.flush_block_fht_cache()

    for block in candidate.transformer.h:
        coordinates = block.mlp.paired_monarch.coordinates
        assert coordinates.grad is not None
        assert torch.isfinite(coordinates.grad).all()
        assert float(coordinates.grad.abs().sum()) > 0.0


def test_paired_monarch_requires_both_generated_mlp_matrices() -> None:
    config = _tiny_config(8)
    config.block_fht_targets = ("mlp.c_proj",)
    try:
        GPT(config)
    except ValueError as error:
        assert "paired Monarch requires generated plain" in str(error)
    else:
        raise AssertionError("expected paired Monarch scope validation")
