import math

import torch

from examples.nanogpt.analyze_attention_endpoint_attribution import (
    evaluate_ce,
    project_attention,
    qkv,
    select_structural_gate,
    shapley_improvements,
)
from examples.nanogpt.model import CausalSelfAttention, GPT, GPTConfig


def test_dense_decomposition_matches_native_attention() -> None:
    torch.manual_seed(7)
    config = GPTConfig(
        block_size=8,
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_embd=8,
        dropout=0.0,
        bias=False,
    )
    attention = CausalSelfAttention(config, layer_id=0).eval()
    values = torch.randn(2, 6, 8)
    q, k, value = qkv(attention, values)
    scores = q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1])
    mask = torch.ones((6, 6), dtype=torch.bool).tril()
    probabilities = torch.softmax(scores.masked_fill(~mask, -torch.inf), dim=-1)
    decomposed = project_attention(attention, probabilities @ value)
    native = attention(values)
    torch.testing.assert_close(decomposed, native, rtol=1e-5, atol=1e-6)


def test_structured_decomposition_matches_native_attention() -> None:
    torch.manual_seed(11)
    targets = (
        "attn.c_attn.qk_headwise",
        "attn.c_attn.v",
        "attn.c_proj",
    )
    config = GPTConfig(
        block_size=8,
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_embd=8,
        dropout=0.0,
        bias=False,
        block_fht=True,
        block_fht_targets=targets,
        block_fht_latent_ratio=0.25,
        block_fht_layers=2,
        block_fht_match_gpt_init=True,
        block_fht_output_gain_targets=targets,
        block_fht_attn_cayley_targets=targets,
        block_fht_attn_cayley_output_targets=(
            "attn.c_attn.qk_headwise",
            "attn.c_proj",
        ),
        block_fht_attn_cayley_bilateral_targets=(
            "attn.c_attn.qk_headwise",
            "attn.c_attn.v",
        ),
        block_fht_attn_cayley_rank=1,
    )
    attention = CausalSelfAttention(config, layer_id=0).eval()
    values = torch.randn(2, 6, 8)
    q, k, value = qkv(attention, values)
    scores = q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1])
    mask = torch.ones((6, 6), dtype=torch.bool).tril()
    probabilities = torch.softmax(scores.masked_fill(~mask, -torch.inf), dim=-1)
    decomposed = project_attention(attention, probabilities @ value)
    native = attention(values)
    torch.testing.assert_close(decomposed, native, rtol=1e-5, atol=1e-6)


def test_shapley_telescopes_and_recovers_additive_contributions() -> None:
    effects = (0.2, -0.1, 0.05)
    losses = {}
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                mask = (a, b, c)
                losses[mask] = 4.0 - sum(
                    effect * active for effect, active in zip(effects, mask)
                )
    attribution = shapley_improvements(losses)
    assert math.isclose(attribution["score"], effects[0])
    assert math.isclose(attribution["value"], effects[1])
    assert math.isclose(attribution["projection"], effects[2])
    assert math.isclose(
        sum(attribution.values()), losses[(0, 0, 0)] - losses[(1, 1, 1)]
    )


def test_selection_requires_replication_or_stable_joint_interaction() -> None:
    protocol = {
        "minimum_stable_component_ce": 0.005,
        "minimum_value_projection_interaction_ce": 0.003,
    }
    result = {
        "primary": {
            "shapley_ce_improvement": {
                "score": 0.001,
                "value": 0.010,
                "projection": 0.002,
            },
            "hybrid_ce": {
                "000": 4.0,
                "010": 3.995,
                "001": 3.998,
                "011": 3.985,
            },
        },
        "confirmation": {
            "shapley_ce_improvement": {
                "score": 0.001,
                "value": 0.009,
                "projection": 0.002,
            },
            "hybrid_ce": {
                "000": 4.0,
                "010": 3.995,
                "001": 3.998,
                "011": 3.986,
            },
        },
    }
    decision = select_structural_gate(result, protocol)
    assert decision["classification"] == "STABLE_SINGLE_COMPONENT"
    assert decision["selected_component"] == "value"


def test_evaluate_ce_accepts_sliced_next_token_targets() -> None:
    config = GPTConfig(
        block_size=8,
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_embd=8,
        dropout=0.0,
        bias=False,
    )
    model = GPT(config).eval()
    tokens = torch.randint(0, config.vocab_size, (2, config.block_size + 1))
    value = evaluate_ce(model, [tokens], "cpu")
    assert math.isfinite(value)


def test_harmful_pair_is_not_selected_from_positive_interaction_only() -> None:
    protocol = {
        "minimum_stable_component_ce": 0.005,
        "minimum_value_projection_interaction_ce": 0.003,
    }
    window = {
        "shapley_ce_improvement": {
            "score": -1.0,
            "value": -0.6,
            "projection": -1.8,
        },
        "hybrid_ce": {
            "000": 3.6,
            "010": 6.8,
            "001": 8.0,
            "011": 7.1,
        },
    }
    decision = select_structural_gate(
        {"primary": window, "confirmation": window}, protocol
    )
    assert decision["classification"] == "NO_STABLE_ENDPOINT_COMPONENT"
    assert decision["selected_component"] is None
