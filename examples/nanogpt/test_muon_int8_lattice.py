from __future__ import annotations

import copy

import torch

from examples.nanogpt.model import GPT, GPTConfig, MultiOptimizer
from examples.nanogpt.muon_int8_lattice import (
    MuonInt8Lattice,
    MuonInt8LatticeLinear,
)


def make_module(
    seed: int = 17, *, error_feedback: bool = False
) -> MuonInt8LatticeLinear:
    return MuonInt8LatticeLinear(
        5,
        4,
        bias=False,
        block_size=8,
        base_seed=seed,
        weight_std=0.02,
        layer_id=2,
        error_feedback=error_feedback,
    )


def make_optimizer(module: MuonInt8LatticeLinear) -> MuonInt8Lattice:
    return MuonInt8Lattice(
        [module],
        lr=0.01,
        momentum=0.5,
        weight_decay=0.1,
        ns_steps=1,
    )


def test_codec_state_is_compact_and_dense_materialization_is_transient() -> None:
    module = make_module()
    state = module.state_dict()
    assert set(state) == {
        "base_seed",
        "base_weight_std",
        "codes",
        "scales",
        "optimizer_step",
    }
    assert module.persistent_codec_bytes == 20 + 3 * 2
    assert module.fp32_weight_bytes == 20 * 4
    assert module.codec_storage_ratio == (20 + 3 * 2) / (20 * 4)
    assert "weight" not in state
    assert "base_weight" not in state


def test_device_style_buffer_migration_preserves_optimizer_leaf() -> None:
    module = make_module()
    module._apply(lambda tensor: tensor.clone())
    assert module.weight.is_leaf
    assert module.weight.requires_grad
    make_optimizer(module)


def test_projection_has_monotone_fp16_scales_and_deterministic_decode() -> None:
    torch.manual_seed(11)
    module = make_module()
    requested = module.base_weight + 0.1 * torch.randn_like(module.weight)
    module.project_weight_(requested)
    first_scales = module.scales.clone()
    first_weight = module.weight.clone()
    module.rematerialize_weight_()
    torch.testing.assert_close(module.weight, first_weight, rtol=0.0, atol=0.0)
    assert int(module.codes.abs().max()) <= 127

    smaller = module.base_weight + 0.001 * torch.randn_like(module.weight)
    module.project_weight_(smaller)
    assert torch.all(module.scales >= first_scales)
    assert int(module.optimizer_step) == 2


def test_model_and_optimizer_resume_are_bit_exact_for_the_next_step() -> None:
    torch.manual_seed(23)
    module = make_module(seed=29)
    optimizer = make_optimizer(module)
    first_gradient = torch.randn_like(module.weight)
    module.weight.grad = first_gradient.clone()
    optimizer.step()

    model_state = copy.deepcopy(module.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    expected_weight = module.weight.clone()

    restored = make_module(seed=999)
    restored.load_state_dict(model_state, strict=True)
    restored_optimizer = make_optimizer(restored)
    restored_optimizer.load_state_dict(optimizer_state)
    torch.testing.assert_close(restored.weight, expected_weight, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        restored.base_weight, module.base_weight, rtol=0.0, atol=0.0
    )

    next_gradient = torch.randn_like(module.weight)
    module.weight.grad = next_gradient.clone()
    restored.weight.grad = next_gradient.clone()
    optimizer.step()
    restored_optimizer.step()
    assert torch.equal(restored.codes, module.codes)
    assert torch.equal(restored.scales, module.scales)
    torch.testing.assert_close(restored.weight, module.weight, rtol=0.0, atol=0.0)
    assert int(restored.optimizer_step) == int(module.optimizer_step) == 2


def test_error_feedback_accumulates_subquantum_muon_updates() -> None:
    torch.manual_seed(31)
    direct = make_module(seed=37, error_feedback=False)
    feedback = make_module(seed=37, error_feedback=True)
    coarse = direct.base_weight + 0.1 * torch.randn_like(direct.weight)
    direct.project_weight_(coarse)
    feedback.load_state_dict(copy.deepcopy(direct.state_dict()), strict=True)
    initial_codes = direct.codes.clone()

    direct_optimizer = MuonInt8Lattice(
        [direct], lr=1e-6, momentum=0.0, weight_decay=0.0, ns_steps=1
    )
    feedback_optimizer = MuonInt8Lattice(
        [feedback], lr=1e-6, momentum=0.0, weight_decay=0.0, ns_steps=1
    )
    gradient = torch.randn_like(direct.weight)
    for _ in range(2048):
        direct.weight.grad = gradient.clone()
        feedback.weight.grad = gradient.clone()
        direct_optimizer.step()
        feedback_optimizer.step()

    assert torch.equal(direct.codes, initial_codes)
    assert not torch.equal(feedback.codes, initial_codes)
    feedback_state = feedback_optimizer.state[feedback.weight]
    assert feedback_state["compression_residual"].dtype == torch.float16
    assert torch.isfinite(feedback_state["compression_residual"]).all()


def test_error_feedback_optimizer_resume_is_bit_exact() -> None:
    torch.manual_seed(41)
    module = make_module(seed=43, error_feedback=True)
    optimizer = make_optimizer(module)
    module.weight.grad = torch.randn_like(module.weight)
    optimizer.step()

    model_state = copy.deepcopy(module.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    restored = make_module(seed=47, error_feedback=True)
    restored.load_state_dict(model_state, strict=True)
    restored_optimizer = make_optimizer(restored)
    restored_optimizer.load_state_dict(optimizer_state)

    gradient = torch.randn_like(module.weight)
    module.weight.grad = gradient.clone()
    restored.weight.grad = gradient.clone()
    optimizer.step()
    restored_optimizer.step()
    assert torch.equal(restored.codes, module.codes)
    assert torch.equal(restored.scales, module.scales)
    torch.testing.assert_close(restored.weight, module.weight, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        restored_optimizer.state[restored.weight]["compression_residual"],
        optimizer.state[module.weight]["compression_residual"],
        rtol=0.0,
        atol=0.0,
    )


def test_gpt_routes_attention_cproj_to_its_own_muon_optimizer() -> None:
    model = GPT(
        GPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=1,
            n_head=2,
            n_embd=8,
            bias=False,
            block_fht=True,
            block_fht_targets=(),
            block_fht_attn_cproj_int8_lattice=True,
            block_fht_attn_cproj_int8_lattice_block_size=16,
        )
    )
    module = model.transformer.h[0].attn.c_proj
    assert isinstance(module, MuonInt8LatticeLinear)
    assert model.attention_int8_lattice_stats() == {
        "modules": 1,
        "elements": 64,
        "codec_bytes": 72,
        "fp32_weight_bytes": 256,
        "storage_ratio": 72 / 256,
    }
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=0.001,
        betas=(0.9, 0.95),
        device_type="cpu",
        optimizer="muon",
        muon_momentum=0.95,
        muon_ns_steps=1,
    )
    assert isinstance(optimizer, MultiOptimizer)
    lattice_optimizers = [
        item for item in optimizer.optimizers if isinstance(item, MuonInt8Lattice)
    ]
    assert len(lattice_optimizers) == 1
    tokens = torch.randint(0, 32, (2, 8))
    _logits, loss = model(tokens, tokens)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert module.weight.grad is not None
    optimizer.step()
    assert int(module.optimizer_step) == 1


def test_conflicting_cproj_representations_are_rejected() -> None:
    try:
        GPT(
            GPTConfig(
                block_size=8,
                vocab_size=32,
                n_layer=1,
                n_head=2,
                n_embd=8,
                bias=False,
                block_fht=True,
                block_fht_targets=("attn.c_proj",),
                block_fht_attn_cproj_int8_lattice=True,
            )
        )
    except ValueError as error:
        assert "remove attn.c_proj" in str(error)
    else:
        raise AssertionError("conflicting c_proj representations must fail")


def test_gpt_routes_qk_headwise_plus_v_lattice_with_error_feedback() -> None:
    model = GPT(
        GPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=1,
            n_head=2,
            n_embd=8,
            bias=False,
            block_fht=True,
            block_fht_targets=("attn.c_attn.qk_headwise",),
            block_fht_attn_v_int8_lattice=True,
            block_fht_attn_v_int8_lattice_block_size=16,
            block_fht_attn_v_int8_lattice_error_feedback=True,
        )
    )
    module = model.transformer.h[0].attn.c_attn_v
    assert isinstance(module, MuonInt8LatticeLinear)
    assert module.error_feedback is True
    assert model.attention_int8_lattice_stats() == {
        "modules": 1,
        "elements": 64,
        "codec_bytes": 72,
        "fp32_weight_bytes": 256,
        "storage_ratio": 72 / 256,
    }
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=0.001,
        betas=(0.9, 0.95),
        device_type="cpu",
        optimizer="muon",
        muon_momentum=0.95,
        muon_ns_steps=1,
    )
    lattice_optimizers = [
        item for item in optimizer.optimizers if isinstance(item, MuonInt8Lattice)
    ]
    assert len(lattice_optimizers) == 1
    tokens = torch.randint(0, 32, (2, 8))
    _logits, loss = model(tokens, tokens)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert module.weight.grad is not None
    optimizer.step()
    state = lattice_optimizers[0].state[module.weight]
    assert state["compression_residual"].dtype == torch.float16
    assert int(module.optimizer_step) == 1


def test_gpt_routes_attention_cproj_lattice_with_error_feedback() -> None:
    model = GPT(
        GPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=1,
            n_head=2,
            n_embd=8,
            bias=False,
            block_fht=True,
            block_fht_targets=(),
            block_fht_attn_cproj_int8_lattice=True,
            block_fht_attn_cproj_int8_lattice_block_size=16,
            block_fht_attn_cproj_int8_lattice_error_feedback=True,
        )
    )
    module = model.transformer.h[0].attn.c_proj
    assert isinstance(module, MuonInt8LatticeLinear)
    assert module.error_feedback is True
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=0.001,
        betas=(0.9, 0.95),
        device_type="cpu",
        optimizer="muon",
        muon_momentum=0.95,
        muon_ns_steps=1,
    )
    lattice_optimizers = [
        item for item in optimizer.optimizers if isinstance(item, MuonInt8Lattice)
    ]
    assert len(lattice_optimizers) == 1
    tokens = torch.randint(0, 32, (2, 8))
    _logits, loss = model(tokens, tokens)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    state = lattice_optimizers[0].state[module.weight]
    assert state["compression_residual"].dtype == torch.float16
    assert int(module.optimizer_step) == 1


def test_cproj_error_feedback_without_lattice_is_rejected() -> None:
    try:
        GPT(
            GPTConfig(
                block_size=8,
                vocab_size=32,
                n_layer=1,
                n_head=2,
                n_embd=8,
                bias=False,
                block_fht=True,
                block_fht_targets=(),
                block_fht_attn_cproj_int8_lattice_error_feedback=True,
            )
        )
    except ValueError as error:
        assert "requires the c_proj lattice" in str(error)
    else:
        raise AssertionError("c_proj feedback without a lattice must fail")


def test_conflicting_v_lattice_representations_are_rejected() -> None:
    try:
        GPT(
            GPTConfig(
                block_size=8,
                vocab_size=32,
                n_layer=1,
                n_head=2,
                n_embd=8,
                bias=False,
                block_fht=True,
                block_fht_targets=("attn.c_attn.qk", "attn.c_attn.v"),
                block_fht_attn_v_int8_lattice=True,
            )
        )
    except ValueError as error:
        assert "remove attn.c_attn.v" in str(error)
    else:
        raise AssertionError("conflicting attention V representations must fail")


def test_v_lattice_without_split_attention_is_rejected() -> None:
    try:
        GPT(
            GPTConfig(
                block_size=8,
                vocab_size=32,
                n_layer=1,
                n_head=2,
                n_embd=8,
                bias=False,
                block_fht=True,
                block_fht_targets=(),
                block_fht_attn_v_int8_lattice=True,
            )
        )
    except ValueError as error:
        assert "split or structured QKV" in str(error)
    else:
        raise AssertionError("V lattice without split attention must fail")


def test_gpt_routes_both_mlp_matrices_to_the_lattice_optimizer() -> None:
    model = GPT(
        GPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=1,
            n_head=2,
            n_embd=8,
            bias=False,
            block_fht=True,
            block_fht_targets=(),
            block_fht_mlp_int8_lattice_targets=(
                "mlp.c_fc",
                "mlp.c_proj",
            ),
            block_fht_mlp_int8_lattice_block_size=16,
            block_fht_mlp_int8_lattice_error_feedback=True,
        )
    )
    mlp = model.transformer.h[0].mlp
    assert isinstance(mlp.c_fc, MuonInt8LatticeLinear)
    assert isinstance(mlp.c_proj, MuonInt8LatticeLinear)
    assert mlp.c_fc.error_feedback is True
    assert mlp.c_proj.error_feedback is True
    assert model.mlp_int8_lattice_stats() == {
        "modules": 2,
        "elements": 512,
        "codec_bytes": 576,
        "fp32_weight_bytes": 2048,
        "storage_ratio": 576 / 2048,
    }
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=0.001,
        betas=(0.9, 0.95),
        device_type="cpu",
        optimizer="muon",
        muon_momentum=0.95,
        muon_ns_steps=1,
    )
    lattice_optimizers = [
        item for item in optimizer.optimizers if isinstance(item, MuonInt8Lattice)
    ]
    assert len(lattice_optimizers) == 1
    tokens = torch.randint(0, 32, (2, 8))
    _logits, loss = model(tokens, tokens)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    assert int(mlp.c_fc.optimizer_step) == 1
    assert int(mlp.c_proj.optimizer_step) == 1


def test_mlp_lattice_rejects_overlapping_block_fht_targets() -> None:
    try:
        GPT(
            GPTConfig(
                block_size=8,
                vocab_size=32,
                n_layer=1,
                n_head=2,
                n_embd=8,
                bias=False,
                block_fht=True,
                block_fht_targets=("mlp.c_fc",),
                block_fht_mlp_int8_lattice_targets=("mlp.c_fc",),
            )
        )
    except ValueError as error:
        assert "cannot be combined" in str(error)
    else:
        raise AssertionError("overlapping MLP representations must fail")
