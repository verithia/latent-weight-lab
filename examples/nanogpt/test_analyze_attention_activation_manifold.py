from __future__ import annotations

import torch

from examples.nanogpt.analyze_attention_activation_manifold import (
    causal_head_function,
)


def test_causal_head_function_shapes_and_probability_normalization() -> None:
    torch.manual_seed(7)
    inputs = torch.randn(2, 5, 8)
    q = torch.randn(4, 8)
    k = torch.randn(4, 8)
    v = torch.randn(4, 8)
    output = torch.randn(8, 4)
    logits, probabilities, contribution = causal_head_function(
        inputs, q, k, v, output
    )
    assert logits.shape == (2, 15)
    assert probabilities.shape == (2, 15)
    assert contribution.shape == (2, 5, 8)
    cursor = 0
    for length in range(1, 6):
        selected = probabilities[:, cursor : cursor + length]
        torch.testing.assert_close(selected.sum(dim=-1), torch.ones(2))
        centered = logits[:, cursor : cursor + length]
        torch.testing.assert_close(centered.mean(dim=-1), torch.zeros(2), atol=1e-6, rtol=1e-6)
        cursor += length


def test_querywise_logit_offset_does_not_change_probability() -> None:
    scores = torch.tensor([[[1.0, 9.0], [2.0, 3.0]]])
    mask = torch.ones((2, 2), dtype=torch.bool).tril()
    base = torch.softmax(scores.masked_fill(~mask, -torch.inf), dim=-1)
    offsets = torch.tensor([[[8.0], [-5.0]]])
    shifted = torch.softmax((scores + offsets).masked_fill(~mask, -torch.inf), dim=-1)
    torch.testing.assert_close(base, shifted)
