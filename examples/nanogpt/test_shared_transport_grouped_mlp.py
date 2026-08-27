from __future__ import annotations

import torch

from examples.nanogpt.model import GPT, GPTConfig, freeze_non_block_fht
from examples.nanogpt.muon import (
    GroupedMuon,
    zeropower_batched_via_newtonschulz5,
    zeropower_via_newtonschulz5,
)


def tiny_config(**updates) -> GPTConfig:
    values = dict(
        block_size=8,
        vocab_size=64,
        n_layer=2,
        n_head=2,
        n_embd=16,
        bias=False,
        compact_native_mlp="shared_transport_grouped_gelu4",
        compact_native_mlp_group_size=4,
    )
    values.update(updates)
    return GPTConfig(**values)


def test_shared_transport_is_registered_once_and_reused() -> None:
    model = GPT(tiny_config())
    transport = model.shared_compact_mlp_transport
    assert transport is not None
    assert all(block.mlp.shared_transport is transport for block in model.transformer.h)
    keys = set(model.state_dict())
    assert "shared_compact_mlp_transport.pre_delta" in keys
    assert "shared_compact_mlp_transport.post_delta" in keys
    assert not any("mlp._shared_transport" in key for key in keys)


def test_exact_production_state_budget() -> None:
    config = tiny_config(n_embd=768, n_head=12, n_layer=12)
    model = GPT(config)
    local = sum(
        parameter.numel()
        for block in model.transformer.h
        for parameter in block.mlp.parameters()
    )
    shared = sum(
        parameter.numel()
        for parameter in model.shared_compact_mlp_transport.parameters()
    )
    dense = config.n_layer * 8 * config.n_embd * config.n_embd
    assert local == 294_912
    assert shared == 73_728
    assert local + shared == 368_640
    assert abs((local + shared) / dense - 0.006510416666666667) < 1e-15
    assert not any(
        key.endswith("mlp.c_fc.weight") or key.endswith("mlp.c_proj.weight")
        for key in model.state_dict()
    )


def test_forward_backward_and_freeze_are_finite() -> None:
    config = tiny_config(
        block_fht=True,
        block_fht_targets=("attn.c_attn.q", "attn.c_attn.k"),
        block_fht_match_gpt_init=True,
    )
    model = GPT(config)
    freeze_non_block_fht(model)
    tokens = torch.randint(0, config.vocab_size, (2, config.block_size))
    _, loss = model(tokens, tokens)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    compact = [
        *model.shared_compact_mlp_transport.parameters(),
        *(
            parameter
            for block in model.transformer.h
            for parameter in block.mlp.parameters()
        ),
    ]
    assert compact
    assert all(parameter.requires_grad for parameter in compact)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in compact
    )


def test_checkpoint_round_trip_is_exact() -> None:
    config = tiny_config()
    torch.manual_seed(7)
    first = GPT(config).eval()
    torch.manual_seed(19)
    second = GPT(config).eval()
    second.load_state_dict(first.state_dict())
    tokens = torch.randint(0, config.vocab_size, (2, config.block_size))
    with torch.no_grad():
        first_logits, _ = first(tokens)
        second_logits, _ = second(tokens)
    torch.testing.assert_close(first_logits, second_logits, rtol=0, atol=0)


def test_grouped_muon_polar_matches_independent_matrices() -> None:
    torch.manual_seed(23)
    gradient = torch.randn(7, 16, 4, dtype=torch.float64)
    actual = zeropower_batched_via_newtonschulz5(gradient, steps=5)
    expected = torch.stack(
        [zeropower_via_newtonschulz5(item, steps=5) for item in gradient]
    )
    # Batched and independent CUDA/BLAS reductions may accumulate FP32 in a
    # different order; the algorithm and blockwise normalization are the same.
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)


def test_configured_optimizer_uses_grouped_muon_for_local_blocks() -> None:
    model = GPT(tiny_config())
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=0.0024,
        betas=(0.9, 0.95),
        device_type="cpu",
        optimizer="muon",
    )
    optimizers = getattr(optimizer, "optimizers", [optimizer])
    grouped = [item for item in optimizers if isinstance(item, GroupedMuon)]
    assert len(grouped) == 1
    grouped_ids = {
        id(parameter)
        for group in grouped[0].param_groups
        for parameter in group["params"]
    }
    expected_ids = {
        id(parameter)
        for block in model.transformer.h
        for name, parameter in block.mlp.named_parameters()
        if name in {"grouped_c_fc_weight", "grouped_c_proj_weight"}
    }
    assert grouped_ids == expected_ids
