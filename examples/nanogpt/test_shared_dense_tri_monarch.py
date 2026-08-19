from __future__ import annotations

import io

import torch

from examples.nanogpt.model import GPT, GPTConfig


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
        "block_fht_targets": ("attn.c_proj",),
        "block_fht_latent_ratio": 0.25,
        "block_fht_match_gpt_init": True,
        "block_fht_ffn_pregelu_gain": True,
        "block_fht_mlp_residual_output_gain": True,
        "mlp_shared_dense_trunk": True,
        "mlp_shared_dense_tri_monarch_block_width": 4,
    }
    values.update(overrides)
    return GPTConfig(**values)


def _transport_parameters(model: GPT) -> list[torch.nn.Parameter]:
    return [
        parameter
        for block in model.transformer.h
        for module in (
            block.mlp.paired_monarch,
            block.mlp.input_monarch,
            block.mlp.output_monarch,
        )
        for parameter in module.parameters()
    ]


def test_tri_monarch_shares_core_but_keeps_all_transports_private() -> None:
    model = GPT(_tiny_config())
    mlps = [block.mlp for block in model.transformer.h]

    assert len({id(mlp.c_fc.weight) for mlp in mlps}) == 1
    assert len({id(mlp.c_proj.weight) for mlp in mlps}) == 1
    assert len({id(mlp.paired_monarch.coordinates) for mlp in mlps}) == 3
    assert len({id(mlp.input_monarch.coordinates) for mlp in mlps}) == 3
    assert len({id(mlp.output_monarch.coordinates) for mlp in mlps}) == 3
    assert sum(parameter.numel() for parameter in _transport_parameters(model)) == 2304


def test_tri_monarch_is_exact_identity_at_initialization() -> None:
    model = GPT(_tiny_config())
    for block in model.transformer.h:
        mlp = block.mlp
        assert torch.equal(
            mlp._materialize_charted_cfc_weight(mlp.c_fc.weight),
            mlp.c_fc.weight,
        )
        assert torch.equal(
            mlp._materialize_charted_cproj_weight(mlp.c_proj.weight),
            mlp.c_proj.weight,
        )


def test_tri_monarch_backward_reaches_shared_core_and_every_transport() -> None:
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
    for parameter in (root.c_fc.weight, root.c_proj.weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert float(parameter.grad.abs().sum()) > 0.0
    for parameter in _transport_parameters(model):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert float(parameter.grad.abs().sum()) > 0.0


def test_tri_monarch_optimizer_has_unique_chart_ownership() -> None:
    model = GPT(_tiny_config())
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=0.0024,
        betas=(0.9, 0.95),
        device_type="cpu",
        optimizer="muon",
        muon_adamw_lr_scale=0.3,
        block_fht_mlp_chart_lr_scale=4.0,
    )
    transports = {id(parameter) for parameter in _transport_parameters(model)}
    routed: dict[int, float] = {}
    all_ids: list[int] = []
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            parameter_id = id(parameter)
            all_ids.append(parameter_id)
            if parameter_id in transports:
                routed[parameter_id] = float(group["lr_scale"])

    assert len(all_ids) == len(set(all_ids))
    assert routed.keys() == transports
    assert set(routed.values()) == {1.2}


def test_tri_monarch_checkpoint_roundtrip_and_scope_rejection() -> None:
    model = GPT(_tiny_config())
    with torch.no_grad():
        model.transformer.h[2].mlp.output_monarch.coordinates[3] = 0.125
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = GPT(_tiny_config())
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    assert torch.equal(
        restored.transformer.h[2].mlp.output_monarch.coordinates,
        model.transformer.h[2].mlp.output_monarch.coordinates,
    )

    for overrides, expected in (
        ({"mlp_shared_dense_trunk": False}, "requires the shared dense MLP trunk"),
        ({"block_fht_targets": ("mlp.c_fc",)}, "requires plain dense c_fc"),
        ({"block_fht_mlp_paired_monarch_block_width": 4}, "mutually exclusive"),
    ):
        try:
            GPT(_tiny_config(**overrides))
        except ValueError as error:
            assert expected in str(error)
        else:
            raise AssertionError(f"expected scope rejection containing {expected!r}")
