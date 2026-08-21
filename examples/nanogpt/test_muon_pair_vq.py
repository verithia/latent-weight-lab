from __future__ import annotations

import copy

import torch

from examples.nanogpt.muon_pair_vq import MuonPairVQ, MuonPairVQLinear


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
