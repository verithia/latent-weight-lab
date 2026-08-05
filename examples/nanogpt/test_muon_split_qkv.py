import pytest
import torch

from examples.nanogpt.model import GPT, GPTConfig
from examples.nanogpt.muon import Muon, muon_update


def test_blockwise_update_matches_independent_qk_and_value_updates() -> None:
    torch.manual_seed(7)
    update = torch.randn(12, 4)
    actual = muon_update(update, steps=5, row_splits=(8, 4))
    expected = torch.cat(
        (
            muon_update(update[:8], steps=5),
            muon_update(update[8:], steps=5),
        )
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("splits", [(), (0, 12), (7, 4)])
def test_invalid_row_splits_are_rejected(splits: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="row splits"):
        muon_update(torch.randn(12, 4), steps=5, row_splits=splits)


def test_muon_step_uses_parameter_row_split_metadata() -> None:
    torch.manual_seed(11)
    parameter = torch.nn.Parameter(torch.randn(12, 4))
    parameter._muon_row_splits = (8, 4)
    gradient = torch.randn_like(parameter)
    parameter.grad = gradient.clone()
    before = parameter.detach().clone()
    optimizer = Muon(
        [parameter], lr=0.2, momentum=0.0, weight_decay=0.0, ns_steps=5
    )
    optimizer.step()
    expected = before - 0.2 * muon_update(
        gradient, steps=5, row_splits=(8, 4)
    )
    torch.testing.assert_close(parameter, expected)


def test_gpt_optimizer_marks_only_dense_packed_attention_weights() -> None:
    config = GPTConfig(
        block_size=8,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_embd=8,
        block_fht=False,
    )
    model = GPT(config)
    model.configure_optimizers(
        0.1,
        1e-3,
        (0.9, 0.95),
        "cpu",
        optimizer="muon",
        muon_split_attention_qkv_rows=True,
    )
    marked = [
        (name, getattr(parameter, "_muon_row_splits", None))
        for name, parameter in model.named_parameters()
        if getattr(parameter, "_muon_row_splits", None) is not None
    ]
    assert marked == [
        ("transformer.h.0.attn.c_attn.weight", (16, 8)),
        ("transformer.h.1.attn.c_attn.weight", (16, 8)),
    ]


def test_split_qkv_update_rejects_blockfht_attention() -> None:
    model = GPT(
        GPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=1,
            n_head=2,
            n_embd=8,
            block_fht=True,
            block_fht_targets=("attn.c_attn.qk_headwise",),
            block_fht_match_gpt_init=True,
        )
    )
    with pytest.raises(ValueError, match="dense packed c_attn"):
        model.configure_optimizers(
            0.1,
            1e-3,
            (0.9, 0.95),
            "cpu",
            optimizer="muon",
            muon_split_attention_qkv_rows=True,
        )
