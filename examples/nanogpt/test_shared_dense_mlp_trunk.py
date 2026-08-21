from __future__ import annotations

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
    }
    values.update(overrides)
    return GPTConfig(**values)


def test_shared_dense_trunk_ties_only_full_rank_mlp_matrices() -> None:
    model = GPT(_tiny_config())
    mlps = [block.mlp for block in model.transformer.h]

    assert len({id(mlp.c_fc.weight) for mlp in mlps}) == 1
    assert len({id(mlp.c_proj.weight) for mlp in mlps}) == 1
    assert len({id(mlp.pregelu_gain) for mlp in mlps}) == 3
    assert len({id(mlp.residual_output_log_gain) for mlp in mlps}) == 3

    shared_names = [
        name
        for name, _ in model.named_parameters()
        if name.endswith("mlp.c_fc.weight") or name.endswith("mlp.c_proj.weight")
    ]
    assert shared_names == [
        "transformer.h.0.mlp.c_fc.weight",
        "transformer.h.0.mlp.c_proj.weight",
    ]


def test_shared_dense_trunk_supports_contiguous_depth_groups() -> None:
    model = GPT(_tiny_config(n_layer=6, mlp_shared_dense_trunk_groups=3))
    mlps = [block.mlp for block in model.transformer.h]

    assert len({id(mlp.c_fc.weight) for mlp in mlps}) == 3
    assert len({id(mlp.c_proj.weight) for mlp in mlps}) == 3
    assert id(mlps[0].c_fc.weight) == id(mlps[1].c_fc.weight)
    assert id(mlps[2].c_fc.weight) == id(mlps[3].c_fc.weight)
    assert id(mlps[4].c_fc.weight) == id(mlps[5].c_fc.weight)
    assert len({id(mlp.pregelu_gain) for mlp in mlps}) == 6
    assert len({id(mlp.residual_output_log_gain) for mlp in mlps}) == 6


def test_shared_dense_trunk_backward_aggregates_shared_and_private_gradients() -> None:
    torch.manual_seed(17)
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
    for block in model.transformer.h:
        for parameter in (
            block.mlp.pregelu_gain,
            block.mlp.residual_output_log_gain,
        ):
            assert parameter is not None and parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert float(parameter.grad.abs().sum()) > 0.0


def test_shared_dense_trunk_optimizer_routes_private_gains_as_charts() -> None:
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
    gains = {
        id(parameter)
        for block in model.transformer.h
        for parameter in (
            block.mlp.pregelu_gain,
            block.mlp.residual_output_log_gain,
        )
        if parameter is not None
    }
    routed: dict[int, float] = {}
    all_parameter_ids: list[int] = []
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            parameter_id = id(parameter)
            all_parameter_ids.append(parameter_id)
            if parameter_id in gains:
                routed[parameter_id] = float(group["lr_scale"])

    assert len(all_parameter_ids) == len(set(all_parameter_ids))
    assert routed.keys() == gains
    assert set(routed.values()) == {1.2}


def test_shared_dense_trunk_rejects_generated_mlp_and_missing_private_charts() -> None:
    invalid = (
        (
            {"block_fht_targets": ("mlp.c_fc",)},
            "requires dense MLP matrices",
        ),
        (
            {"block_fht_ffn_pregelu_gain": False},
            "requires layer-private pre-GELU gain",
        ),
        (
            {"block_fht_mlp_residual_output_gain": False},
            "requires layer-private residual-output gain",
        ),
        (
            {"moe_num_experts": 2},
            "incompatible with MoE",
        ),
        (
            {"n_layer": 3, "mlp_shared_dense_trunk_groups": 2},
            "evenly divide n_layer",
        ),
    )
    for overrides, expected in invalid:
        try:
            GPT(_tiny_config(**overrides))
        except ValueError as error:
            assert expected in str(error)
        else:
            raise AssertionError(f"expected scope rejection containing {expected!r}")
