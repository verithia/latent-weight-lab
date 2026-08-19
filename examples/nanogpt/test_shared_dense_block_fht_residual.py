from __future__ import annotations

import math

import torch

from examples.nanogpt.model import BlockFHTLinear, GPT, GPTConfig


def _tiny_config(**overrides: object) -> GPTConfig:
    values: dict[str, object] = {
        "block_size": 8,
        "vocab_size": 64,
        "n_layer": 3,
        "n_head": 2,
        "n_embd": 16,
        "dropout": 0.0,
        "bias": False,
        "block_fht": True,
        "block_fht_targets": (
            "attn.c_proj",
            "mlp.c_fc",
            "mlp.c_proj",
        ),
        "block_fht_latent_ratio": 0.25,
        "block_fht_match_gpt_init": True,
        "block_fht_ffn_pregelu_gain": True,
        "block_fht_mlp_residual_output_gain": True,
        "mlp_shared_dense_block_fht_residual": True,
        "mlp_shared_dense_block_fht_residual_scale": math.sqrt(0.5),
    }
    values.update(overrides)
    return GPTConfig(**values)


def test_hybrid_ties_dense_bases_but_keeps_private_latents() -> None:
    model = GPT(_tiny_config())
    mlps = [block.mlp for block in model.transformer.h]

    assert all(isinstance(mlp.c_fc, BlockFHTLinear) for mlp in mlps)
    assert all(isinstance(mlp.c_proj, BlockFHTLinear) for mlp in mlps)
    assert len({id(mlp.c_fc.residual_base_weight) for mlp in mlps}) == 1
    assert len({id(mlp.c_proj.residual_base_weight) for mlp in mlps}) == 1
    assert len({id(mlp.c_fc.generator.latent) for mlp in mlps}) == 3
    assert len({id(mlp.c_proj.generator.latent) for mlp in mlps}) == 3


def test_cached_backward_splits_shared_and_private_gradients() -> None:
    torch.manual_seed(29)
    model = GPT(_tiny_config())
    tokens = torch.randint(0, 64, (2, 8))
    model.prepare_block_fht_cache(dtype=torch.float32)
    try:
        logits, loss = model(tokens, tokens)
        assert torch.isfinite(logits).all()
        assert loss is not None and torch.isfinite(loss)
        loss.backward()
    finally:
        model.flush_block_fht_cache()

    root = model.transformer.h[0].mlp
    for base in (
        root.c_fc.residual_base_weight,
        root.c_proj.residual_base_weight,
    ):
        assert base is not None and base.grad is not None
        assert torch.isfinite(base.grad).all()
        assert float(base.grad.abs().sum()) > 0.0
    for block in model.transformer.h:
        for latent in (
            block.mlp.c_fc.generator.latent,
            block.mlp.c_proj.generator.latent,
        ):
            assert latent.grad is not None
            assert torch.isfinite(latent.grad).all()
            assert float(latent.grad.abs().sum()) > 0.0


def test_equal_mix_preserves_target_initialization_variance() -> None:
    model = GPT(_tiny_config(n_layer=12, n_head=4, n_embd=64))
    c_fc = model.transformer.h[0].mlp.c_fc
    c_proj = model.transformer.h[0].mlp.c_proj
    assert isinstance(c_fc, BlockFHTLinear)
    assert isinstance(c_proj, BlockFHTLinear)
    assert math.isclose(c_fc.residual_base_scale, math.sqrt(0.5))
    assert math.isclose(c_proj.residual_base_scale, math.sqrt(0.5))
    assert math.isclose(float(c_fc.weight.float().std()), 0.02, rel_tol=0.12)
    target_cproj = 0.02 / math.sqrt(24)
    assert math.isclose(
        float(c_proj.weight.float().std()), target_cproj, rel_tol=0.12
    )


def test_hybrid_scope_rejections_are_explicit() -> None:
    invalid = (
        (
            {"block_fht_targets": ("attn.c_proj", "mlp.c_fc")},
            "requires mlp.c_fc and mlp.c_proj targets",
        ),
        (
            {"block_fht_match_gpt_init": False},
            "requires GPT init matching",
        ),
        (
            {"mlp_shared_dense_block_fht_residual_scale": 1.0},
            "must be finite and in (0, 1)",
        ),
        (
            {"mlp_shared_dense_trunk": True},
            "mutually exclusive",
        ),
    )
    for overrides, expected in invalid:
        try:
            GPT(_tiny_config(**overrides))
        except ValueError as error:
            assert expected in str(error)
        else:
            raise AssertionError(
                f"expected scope rejection containing {expected!r}"
            )
