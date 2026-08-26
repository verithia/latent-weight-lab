from __future__ import annotations

import torch

from examples.nanogpt.model import (
    GPT,
    GPTConfig,
    GradientSeededLowRankLinear,
    MultiOptimizer,
)


def test_gradient_bootstrap_preserves_function_and_seeds_right_space() -> None:
    torch.manual_seed(7)
    layer = GradientSeededLowRankLinear(
        5,
        7,
        bias=False,
        rank=2,
        scale=1.0,
        target_name="mlp.c_fc",
    )
    torch.nn.init.normal_(layer.weight, std=0.02)
    x = torch.randn(11, 5)
    before = layer(x).detach()
    layer.enable_gradient_bootstrap()
    loss = layer(x).square().mean()
    loss.backward()
    gradient = layer.weight.grad.detach().clone()
    singular_values = torch.linalg.svdvals(gradient)
    expected_capture = float(
        singular_values[:2].double().square().sum()
        / singular_values.double().square().sum()
    )
    record = layer.finish_gradient_bootstrap()
    after = layer(x).detach()

    torch.testing.assert_close(after, before)
    torch.testing.assert_close(
        layer.gradient_seeded_right.T @ layer.gradient_seeded_right,
        torch.eye(2),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.count_nonzero(layer.gradient_seeded_left) == 0
    assert abs(float(record["gradient_energy_capture"]) - expected_capture) < 1e-7
    assert "weight" not in layer.state_dict()
    assert set(layer.state_dict()) == {
        "gradient_seeded_left",
        "gradient_seeded_right",
    }


def test_gpt_uses_compact_factors_and_routes_them_to_adamw() -> None:
    config = GPTConfig(
        block_size=8,
        vocab_size=32,
        n_layer=2,
        n_head=4,
        n_embd=16,
        compact_mlp_gradient_seeded_rank=2,
    )
    model = GPT(config)
    assert model.enable_compact_mlp_gradient_bootstrap() == 4
    tokens = torch.randint(0, config.vocab_size, (2, config.block_size))
    _, loss = model(tokens, tokens)
    assert loss is not None
    loss.backward()
    records = model.finish_compact_mlp_gradient_bootstrap()
    model.zero_grad(set_to_none=True)
    assert len(records) == 4
    stats = model.compact_mlp_gradient_seeded_stats()
    assert stats["modules"] == 4
    assert stats["trainable_factor_scalars"] == 4 * 2 * (16 + 64)
    state_keys = set(model.state_dict())
    assert "transformer.h.0.mlp.c_fc.weight" not in state_keys
    assert "transformer.h.0.mlp.c_proj.weight" not in state_keys

    optimizer = model.configure_optimizers(
        0.1,
        1e-3,
        (0.9, 0.95),
        "cpu",
        optimizer="muon",
    )
    assert isinstance(optimizer, MultiOptimizer)
    factor_ids = {
        id(parameter)
        for name, parameter in model.named_parameters()
        if "gradient_seeded_" in name
    }
    adamw_ids = {
        id(parameter)
        for child in optimizer.optimizers
        if isinstance(child, torch.optim.AdamW)
        for group in child.param_groups
        for parameter in group["params"]
    }
    assert factor_ids
    assert factor_ids.issubset(adamw_ids)


def test_gradient_seeded_checkpoint_reconstructs_procedural_seed_exactly() -> None:
    config = GPTConfig(
        block_size=8,
        vocab_size=32,
        n_layer=2,
        n_head=4,
        n_embd=16,
        compact_mlp_gradient_seeded_rank=2,
    )
    torch.manual_seed(20260826)
    source = GPT(config)
    tokens = torch.arange(16).reshape(2, 8) % config.vocab_size
    assert source.enable_compact_mlp_gradient_bootstrap() == 4
    _, loss = source(tokens, tokens)
    assert loss is not None
    loss.backward()
    source.finish_compact_mlp_gradient_bootstrap()
    source.zero_grad(set_to_none=True)
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            if "gradient_seeded_left" in name:
                parameter.normal_(mean=0.0, std=0.01)
    source.eval()
    expected, _ = source(tokens)
    checkpoint = {
        key: value.detach().clone() for key, value in source.state_dict().items()
    }

    torch.manual_seed(20260826)
    restored = GPT(config)
    restored.load_state_dict(checkpoint)
    restored.eval()
    actual, _ = restored(tokens)

    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    for name, module in restored.named_modules():
        if isinstance(module, GradientSeededLowRankLinear):
            assert f"{name}.weight" not in checkpoint
