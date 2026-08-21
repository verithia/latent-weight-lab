from __future__ import annotations

import copy

import torch

from examples.nanogpt.model import GPT, GPTConfig
from examples.nanogpt.muon_pair_vq import (
    MuonPairVQ,
    MuonPairVQLinear,
    _nearest_cartesian_codes,
    _nearest_codes_exact,
    _normal_cartesian_codebook,
)


def make_module(*, stages: int = 2, seed: int = 101) -> MuonPairVQLinear:
    return MuonPairVQLinear(
        8,
        6,
        bias=False,
        stages=stages,
        base_seed=seed,
        weight_std=0.02,
        layer_id=3,
        neighbor_candidates=16,
        code_refresh_interval=8,
    )


def make_optimizer(module: MuonPairVQLinear) -> MuonPairVQ:
    return MuonPairVQ(
        [module], lr=0.01, momentum=0.5, weight_decay=0.1, ns_steps=1
    )


def test_codec_state_excludes_transient_dense_weight() -> None:
    module = make_module()
    state = module.state_dict()
    assert set(state) == {"codebooks", "codes", "optimizer_step"}
    assert state["codebooks"].dtype == torch.float32
    assert state["codes"].dtype == torch.uint8
    assert "weight" not in state
    assert module.persistent_codec_bytes == 2 * 256 * 2 * 4 + 2 * 24 + 8


def test_projection_moves_toward_request_and_refreshes_codes() -> None:
    module = make_module(stages=1)
    old_codes = module.codes.clone()
    requested = module.weight + 0.05 * torch.randn_like(module.weight)
    diagnostics = module.project_requested_weight_(requested, refresh_codes=True)
    assert diagnostics["requested_step_energy_recovery"] > 0.0
    assert diagnostics["requested_update_cosine"] > 0.0
    assert diagnostics["code_changes"] > 0
    assert not torch.equal(module.codes, old_codes)


def test_optimizer_state_is_only_compact_code_momentum() -> None:
    module = make_module()
    optimizer = make_optimizer(module)
    module.weight.grad = torch.randn_like(module.weight)
    optimizer.step()
    state = optimizer.state[module.weight]
    assert set(state) == {"compact_momentum"}
    assert state["compact_momentum"].shape == module.codebooks.shape
    assert state["compact_momentum"].numel() == 2 * 256 * 2


def test_model_and_optimizer_resume_are_bit_exact_for_next_step() -> None:
    torch.manual_seed(103)
    module = make_module(seed=107)
    optimizer = make_optimizer(module)
    module.weight.grad = torch.randn_like(module.weight)
    optimizer.step()
    model_state = copy.deepcopy(module.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())

    restored = make_module(seed=109)
    restored.load_state_dict(model_state, strict=True)
    restored_optimizer = make_optimizer(restored)
    restored_optimizer.load_state_dict(optimizer_state)
    torch.testing.assert_close(restored.weight, module.weight, rtol=0.0, atol=0.0)

    gradient = torch.randn_like(module.weight)
    module.weight.grad = gradient.clone()
    restored.weight.grad = gradient.clone()
    optimizer.step()
    restored_optimizer.step()
    assert torch.equal(restored.codes, module.codes)
    torch.testing.assert_close(
        restored.codebooks, module.codebooks, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(restored.weight, module.weight, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        restored_optimizer.state[restored.weight]["compact_momentum"],
        optimizer.state[module.weight]["compact_momentum"],
        rtol=0.0,
        atol=0.0,
    )


def test_device_style_migration_preserves_weight_leaf() -> None:
    module = make_module()
    module._apply(lambda tensor: tensor.clone())
    assert module.weight.is_leaf and module.weight.requires_grad
    make_optimizer(module)


def test_cartesian_initialization_is_exact_nearest_neighbor() -> None:
    torch.manual_seed(113)
    vectors = torch.randn(4096, 2) * 0.02
    codebook = _normal_cartesian_codebook(0.02, device=torch.device("cpu"))
    assert torch.equal(
        _nearest_cartesian_codes(vectors, codebook),
        _nearest_codes_exact(vectors, codebook),
    )


def test_gpt_routes_complete_mlp_and_optimizer_through_pair_vq() -> None:
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
            block_fht_mlp_pair_vq=True,
            block_fht_mlp_pair_vq_neighbor_candidates=16,
            block_fht_mlp_pair_vq_code_refresh_interval=8,
        )
    )
    mlp = model.transformer.h[0].mlp
    assert isinstance(mlp.c_fc, MuonPairVQLinear)
    assert isinstance(mlp.c_proj, MuonPairVQLinear)
    assert mlp.c_fc.stages == 2
    assert mlp.c_proj.stages == 1
    stats = model.mlp_pair_vq_stats()
    assert stats["modules"] == 2
    assert stats["dense_master_weight"] == "disabled"
    assert stats["dense_optimizer_momentum"] == "disabled"
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=0.001,
        betas=(0.9, 0.95),
        device_type="cpu",
        optimizer="muon",
        muon_momentum=0.95,
        muon_ns_steps=1,
    )
    pair_optimizers = [
        item for item in optimizer.optimizers if isinstance(item, MuonPairVQ)
    ]
    assert len(pair_optimizers) == 1
    tokens = torch.randint(0, 32, (2, 8))
    _logits, loss = model(tokens, tokens)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    assert int(mlp.c_fc.optimizer_step) == 1
    assert int(mlp.c_proj.optimizer_step) == 1
