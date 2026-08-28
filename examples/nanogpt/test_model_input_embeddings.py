from __future__ import annotations

import pytest
import torch

from examples.nanogpt.model import GPT, GPTConfig


def tiny_model() -> GPT:
    torch.manual_seed(17)
    return GPT(
        GPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=2,
            n_head=2,
            n_embd=8,
            dropout=0.0,
            bias=False,
            block_fht=False,
            tie_word_embeddings=True,
        )
    ).eval()


def test_input_embeddings_reproduce_token_lookup() -> None:
    model = tiny_model()
    idx = torch.randint(0, model.config.vocab_size, (2, 7))
    targets = torch.randint(0, model.config.vocab_size, (2, 7))
    embeddings = model.transformer.wte(idx)
    token_logits, token_loss = model(idx, targets)
    embed_logits, embed_loss = model(None, targets, input_embeddings=embeddings)
    torch.testing.assert_close(embed_logits, token_logits, rtol=0, atol=0)
    torch.testing.assert_close(embed_loss, token_loss, rtol=0, atol=0)


def test_input_embeddings_validate_exclusive_source_and_width() -> None:
    model = tiny_model()
    idx = torch.zeros(1, 4, dtype=torch.long)
    embeddings = torch.zeros(1, 4, model.config.n_embd)
    with pytest.raises(ValueError, match="exactly one"):
        model(idx, input_embeddings=embeddings)
    with pytest.raises(ValueError, match="exactly one"):
        model(None)
    with pytest.raises(ValueError, match="width mismatch"):
        model(None, input_embeddings=torch.zeros(1, 4, model.config.n_embd + 1))
