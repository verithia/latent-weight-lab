from __future__ import annotations

from dataclasses import replace

import torch

from examples.nanogpt.model import (
    GPT,
    GPTConfig,
    LearnedLowRankCayleyMix,
)


def test_low_rank_cayley_is_identity_initialized_with_live_gradient() -> None:
    mix = LearnedLowRankCayleyMix(
        features=8,
        rank=1,
        seed=17,
    ).double()
    values = torch.randn(3, 8, dtype=torch.float64)
    target = torch.randn_like(values)
    output = mix(values)
    torch.testing.assert_close(output, values, rtol=0.0, atol=0.0)

    (output * target).sum().backward()
    assert mix.left.grad is not None
    assert float(mix.left.grad.norm()) > 0.0


def test_low_rank_cayley_remains_orthogonal_after_motion() -> None:
    mix = LearnedLowRankCayleyMix(
        features=8,
        rank=2,
        seed=29,
        coordinate_scale=1.5,
    ).double()
    with torch.no_grad():
        mix.left.normal_(mean=0.0, std=0.2)
        mix.right.add_(0.1 * torch.randn_like(mix.right))
    operator = mix(torch.eye(8, dtype=torch.float64))
    torch.testing.assert_close(
        operator.transpose(0, 1) @ operator,
        torch.eye(8, dtype=torch.float64),
        rtol=1e-9,
        atol=1e-9,
    )


def test_attention_cayley_preserves_initial_gpt_function() -> None:
    base = GPTConfig(
        block_size=8,
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_embd=8,
        block_fht=True,
        block_fht_targets=(
            "attn.c_attn.qk_headwise",
            "attn.c_attn.v",
            "attn.c_proj",
        ),
        block_fht_latent_ratio=0.25,
        block_fht_layers=2,
        block_fht_match_gpt_init=True,
    )
    cayley = replace(
        base,
        block_fht_attn_cayley_targets=base.block_fht_targets,
        block_fht_attn_cayley_rank=1,
    )
    torch.manual_seed(20260730)
    control = GPT(base).eval()
    torch.manual_seed(20260730)
    candidate = GPT(cayley).eval()
    tokens = torch.randint(0, base.vocab_size, (2, base.block_size))
    with torch.no_grad():
        control_logits, _ = control(tokens)
        candidate_logits, _ = candidate(tokens)
    torch.testing.assert_close(candidate_logits, control_logits)

    cayley_parameters = [
        name
        for name, _parameter in candidate.named_parameters()
        if "input_cayley" in name
    ]
    assert len(cayley_parameters) == 6
