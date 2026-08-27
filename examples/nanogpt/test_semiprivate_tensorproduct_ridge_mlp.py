from __future__ import annotations

import torch

from examples.nanogpt.model import GPT, GPTConfig, freeze_non_block_fht
from examples.nanogpt.muon import Muon


def tiny_config(**updates) -> GPTConfig:
    values = dict(
        block_size=8,
        vocab_size=64,
        n_layer=2,
        n_head=2,
        n_embd=16,
        bias=False,
        compact_native_mlp="semiprivate_tensorproduct_r20",
        compact_native_mlp_factor_rank=20,
        compact_native_mlp_seed=20260828,
    )
    values.update(updates)
    return GPTConfig(**values)


def production_config(**updates) -> GPTConfig:
    values = dict(
        block_size=8,
        vocab_size=64,
        n_layer=12,
        n_head=12,
        n_embd=768,
        bias=False,
        compact_native_mlp="semiprivate_tensorproduct_r20",
        compact_native_mlp_factor_rank=20,
        compact_native_mlp_seed=20260828,
    )
    values.update(updates)
    return GPTConfig(**values)


def test_frozen_production_state_and_single_shared_registration() -> None:
    model = GPT(production_config())
    field = model.shared_compact_mlp_tensor_product
    assert field is not None
    assert all(block.mlp.shared_field is field for block in model.transformer.h)
    local = sum(
        parameter.numel()
        for block in model.transformer.h
        for parameter in block.mlp.parameters()
    )
    shared = sum(parameter.numel() for parameter in field.parameters())
    dense = 12 * 8 * 768 * 768
    assert local == 198_816
    assert shared == 353_280
    assert local + shared == 552_096
    assert abs((local + shared) / dense - 0.0097503662109375) < 1e-15
    keys = set(model.state_dict())
    assert "shared_compact_mlp_tensor_product.shared_input_weight" in keys
    assert "shared_compact_mlp_tensor_product.write_weight" in keys
    assert not any("mlp._shared_field" in key for key in keys)
    assert not any(
        "permutation" in key
        or key.endswith("mlp.c_fc.weight")
        or key.endswith("mlp.c_proj.weight")
        for key in keys
    )


def test_tensor_product_feature_order_is_exact() -> None:
    torch.manual_seed(3)
    model = GPT(tiny_config(n_layer=1)).eval()
    mlp = model.transformer.h[0].mlp
    x = torch.randn(2, 3, 16)
    private = torch.nn.functional.gelu(
        torch.nn.functional.linear(x, mlp.private_input_weight)
    )
    shared = mlp.shared_field.shared_features(x)
    expected = torch.cat(
        (
            private,
            shared,
            (private.unsqueeze(-1) * shared.unsqueeze(-2)).flatten(-2),
        ),
        dim=-1,
    )
    torch.testing.assert_close(mlp.tensor_product_features(x), expected)
    assert expected.shape[-1] == 440


def test_weight_folded_forward_and_gradients_equal_activation_conjugation() -> None:
    torch.manual_seed(17)
    model = GPT(tiny_config(n_layer=1)).eval()
    mlp = model.transformer.h[0].mlp
    x = torch.randn(2, 3, 16)

    actual = mlp(x)
    sign = mlp.sign.to(dtype=x.dtype)
    signed = x.index_select(-1, mlp.permutation) * sign
    features = mlp.tensor_product_features(signed)
    features = features * mlp.feature_gain.to(dtype=x.dtype)
    conjugated = mlp.shared_field.write(features)
    conjugated = conjugated * mlp.output_gain.to(dtype=x.dtype)
    reference = (conjugated * sign).index_select(
        -1,
        mlp.inverse_permutation,
    )
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)

    parameters = (
        mlp.private_input_weight,
        mlp.feature_gain,
        mlp.output_gain,
        mlp.shared_field.shared_input_weight,
        mlp.shared_field.write_weight,
    )
    actual_grads = torch.autograd.grad(actual.square().sum(), parameters)
    reference_grads = torch.autograd.grad(reference.square().sum(), parameters)
    for actual_grad, reference_grad in zip(actual_grads, reference_grads):
        torch.testing.assert_close(
            actual_grad,
            reference_grad,
            rtol=2e-5,
            atol=2e-6,
        )


def test_forward_backward_freeze_and_private_layer_gradients() -> None:
    config = production_config(
        block_fht=True,
        block_fht_targets=("attn.c_attn.q", "attn.c_attn.k"),
        block_fht_match_gpt_init=True,
    )
    model = GPT(config)
    assert not torch.equal(
        model.transformer.h[0].mlp.permutation,
        model.transformer.h[1].mlp.permutation,
    )
    freeze_non_block_fht(model)
    tokens = torch.randint(0, config.vocab_size, (2, config.block_size))
    _, loss = model(tokens, tokens)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    compact = [
        *model.shared_compact_mlp_tensor_product.parameters(),
        *(
            parameter
            for block in model.transformer.h
            for parameter in block.mlp.parameters()
        ),
    ]
    assert all(parameter.requires_grad for parameter in compact)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in compact
    )
    first = model.transformer.h[0].mlp.private_input_weight.grad.flatten()
    second = model.transformer.h[1].mlp.private_input_weight.grad.flatten()
    assert torch.linalg.vector_norm(first) > 0
    assert torch.linalg.vector_norm(second) > 0
    assert abs(torch.nn.functional.cosine_similarity(first, second, dim=0)) < 0.999


def test_checkpoint_round_trip_is_exact() -> None:
    config = tiny_config()
    torch.manual_seed(5)
    first = GPT(config).eval()
    torch.manual_seed(11)
    second = GPT(config).eval()
    second.load_state_dict(first.state_dict())
    tokens = torch.randint(0, config.vocab_size, (2, config.block_size))
    with torch.no_grad():
        first_logits, _ = first(tokens)
        second_logits, _ = second(tokens)
    torch.testing.assert_close(first_logits, second_logits, rtol=0, atol=0)


def test_matrices_use_muon_and_gains_use_adamw() -> None:
    model = GPT(production_config())
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=0.0024,
        betas=(0.9, 0.95),
        device_type="cpu",
        optimizer="muon",
    )
    optimizers = getattr(optimizer, "optimizers", [optimizer])
    muon = [item for item in optimizers if isinstance(item, Muon)]
    assert len(muon) == 1
    matrix_ids = {
        id(parameter)
        for group in muon[0].param_groups
        for parameter in group["params"]
    }
    field = model.shared_compact_mlp_tensor_product
    assert id(field.shared_input_weight) in matrix_ids
    assert id(field.write_weight) in matrix_ids
    for block in model.transformer.h:
        assert id(block.mlp.private_input_weight) in matrix_ids
        assert id(block.mlp.feature_gain) not in matrix_ids
        assert id(block.mlp.output_gain) not in matrix_ids
