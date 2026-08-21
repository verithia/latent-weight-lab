from __future__ import annotations

import copy
from argparse import Namespace

import torch

from examples.nanogpt.model import GPT, GPTConfig
from examples.nanogpt.muon_pair_vq import (
    MuonPairVQ,
    MuonPairVQLinear,
    _nearest_cartesian_codes,
    _nearest_codes_exact,
    _normal_cartesian_codebook,
)
from examples.nanogpt.train import pair_vq_model_kwargs


def make_module(
    *,
    stages: int = 2,
    seed: int = 101,
    fast_residual: bool = False,
    error_feedback: bool = False,
    feedback_codec: str = "cartesian4x4",
    feedback_output_group_size: int = 0,
) -> MuonPairVQLinear:
    return MuonPairVQLinear(
        8,
        6,
        bias=False,
        stages=stages,
        base_seed=seed,
        weight_std=0.02,
        layer_id=3,
        fast_residual=fast_residual,
        error_feedback=error_feedback,
        feedback_codec=feedback_codec,
        feedback_output_group_size=feedback_output_group_size,
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
    assert set(state) == {
        "codebooks",
        "codes",
        "fast_levels",
        "fast_codes",
        "optimizer_step",
    }
    assert state["codebooks"].dtype == torch.float32
    assert state["codes"].dtype == torch.uint8
    assert "weight" not in state
    assert module.persistent_codec_bytes == 2 * 256 * 2 * 4 + 2 * 24 + 8


def test_fast_residual_repairs_small_step_tangent() -> None:
    torch.manual_seed(1019)
    module = make_module(stages=2, fast_residual=True)
    requested = module.weight + 1e-4 * torch.randn_like(module.weight)
    diagnostics = module.project_requested_weight_(requested, refresh_codes=True)
    assert diagnostics["requested_step_energy_recovery"] > 0.8
    assert diagnostics["requested_update_cosine"] > 0.9
    assert diagnostics["fast_code_changes"] > 0
    assert module.fast_levels.shape == (2, 16)
    assert module.fast_codes.numel() == module.element_count // 2


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


def test_pair_coded_feedback_conserves_discarded_motion_compactly() -> None:
    torch.manual_seed(127)
    module = make_module(stages=1, error_feedback=True)
    optimizer = make_optimizer(module)
    diagnostics = None
    for _step in range(12):
        module.weight.grad = 1e-3 * torch.randn_like(module.weight)
        optimizer.step()
        diagnostics = optimizer.consume_diagnostics()[0]
    assert diagnostics is not None
    assert diagnostics["error_feedback"] == 1
    assert diagnostics["feedback_codec_energy_recovery"] > 0.95
    assert diagnostics["conserved_requested_step_energy_recovery"] > 0.90
    assert diagnostics["feedback_code_changes"] > 0
    state = optimizer.state[module.weight]
    assert set(state) == {
        "compact_momentum",
        "feedback_levels",
        "feedback_codes",
    }
    assert state["feedback_levels"].shape == (2, 16)
    assert state["feedback_levels"].dtype == torch.float32
    assert state["feedback_codes"].shape == (module.element_count // 2,)
    assert state["feedback_codes"].dtype == torch.uint8
    assert module.compact_feedback_bytes == module.element_count // 2 + 128
    assert all(value.numel() != module.element_count for value in state.values())


def test_pair_coded_feedback_resume_is_bit_exact_for_next_step() -> None:
    torch.manual_seed(131)
    module = make_module(
        stages=1,
        seed=137,
        error_feedback=True,
        feedback_output_group_size=2,
    )
    optimizer = make_optimizer(module)
    for _step in range(3):
        module.weight.grad = torch.randn_like(module.weight)
        optimizer.step()
    model_state = copy.deepcopy(module.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())

    restored = make_module(
        stages=1,
        seed=139,
        error_feedback=True,
        feedback_output_group_size=2,
    )
    restored.load_state_dict(model_state, strict=True)
    restored_optimizer = make_optimizer(restored)
    restored_optimizer.load_state_dict(optimizer_state)
    restored_state = restored_optimizer.state[restored.weight]
    assert restored_state["feedback_codes"].dtype == torch.uint8
    assert restored_state["feedback_levels"].dtype == torch.float32

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
    original_state = optimizer.state[module.weight]
    restored_state = restored_optimizer.state[restored.weight]
    assert torch.equal(
        restored_state["feedback_codes"], original_state["feedback_codes"]
    )
    torch.testing.assert_close(
        restored_state["feedback_levels"],
        original_state["feedback_levels"],
        rtol=0.0,
        atol=0.0,
    )


def test_output_grouped_feedback_is_compact_and_conserves_motion() -> None:
    torch.manual_seed(139)
    module = make_module(
        stages=1,
        error_feedback=True,
        feedback_output_group_size=2,
    )
    optimizer = make_optimizer(module)
    diagnostics = None
    for _step in range(12):
        row_scale = torch.linspace(0.2, 2.0, module.out_features)[:, None]
        module.weight.grad = row_scale * torch.randn_like(module.weight)
        optimizer.step()
        diagnostics = optimizer.consume_diagnostics()[0]
    assert diagnostics is not None
    assert diagnostics["feedback_codec_energy_recovery"] > 0.95
    assert diagnostics["conserved_requested_step_energy_recovery"] > 0.90
    state = optimizer.state[module.weight]
    assert state["feedback_levels"].shape == (3, 2, 16)
    assert state["feedback_codes"].dtype == torch.uint8
    expected_bytes = module.element_count // 2 + 3 * 2 * 16 * 4
    assert module.compact_feedback_bytes == expected_bytes
    assert all(value.numel() != module.element_count for value in state.values())


def test_polar_feedback_preserves_pair_direction_at_same_code_rate() -> None:
    torch.manual_seed(149)
    module = make_module(
        stages=1,
        error_feedback=True,
        feedback_codec="polar32x8",
    )
    optimizer = make_optimizer(module)
    diagnostics = None
    for _step in range(12):
        module.weight.grad = 1e-3 * torch.randn_like(module.weight)
        optimizer.step()
        diagnostics = optimizer.consume_diagnostics()[0]
    assert diagnostics is not None
    assert diagnostics["feedback_codec_energy_recovery"] > 0.98
    state = optimizer.state[module.weight]
    assert state["feedback_levels"].shape == (8,)
    assert state["feedback_center"].shape == (2,)
    assert state["feedback_codes"].dtype == torch.uint8
    assert module.compact_feedback_bytes == module.element_count // 2 + 40
    assert all(value.numel() != module.element_count for value in state.values())


def test_polar_feedback_resume_is_bit_exact_for_next_step() -> None:
    torch.manual_seed(151)
    module = make_module(
        stages=1,
        seed=157,
        error_feedback=True,
        feedback_codec="polar32x8",
    )
    optimizer = make_optimizer(module)
    for _step in range(3):
        module.weight.grad = torch.randn_like(module.weight)
        optimizer.step()
    model_state = copy.deepcopy(module.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())

    restored = make_module(
        stages=1,
        seed=163,
        error_feedback=True,
        feedback_codec="polar32x8",
    )
    restored.load_state_dict(model_state, strict=True)
    restored_optimizer = make_optimizer(restored)
    restored_optimizer.load_state_dict(optimizer_state)
    gradient = torch.randn_like(module.weight)
    module.weight.grad = gradient.clone()
    restored.weight.grad = gradient.clone()
    optimizer.step()
    restored_optimizer.step()
    torch.testing.assert_close(restored.weight, module.weight, rtol=0.0, atol=0.0)
    original_state = optimizer.state[module.weight]
    restored_state = restored_optimizer.state[restored.weight]
    assert torch.equal(restored_state["feedback_codes"], original_state["feedback_codes"])
    torch.testing.assert_close(
        restored_state["feedback_levels"],
        original_state["feedback_levels"],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        restored_state["feedback_center"],
        original_state["feedback_center"],
        rtol=0.0,
        atol=0.0,
    )


def test_rvq_feedback_learns_joint_pair_atoms_at_same_code_rate() -> None:
    torch.manual_seed(167)
    module = make_module(
        stages=1,
        error_feedback=True,
        feedback_codec="rvq4x4",
    )
    optimizer = make_optimizer(module)
    diagnostics = None
    for _step in range(12):
        base = torch.randn(module.out_features, module.in_features // 2, 1)
        paired = torch.cat((base, 0.7 * base + 0.2 * torch.randn_like(base)), dim=2)
        module.weight.grad = 1e-3 * paired.reshape_as(module.weight)
        optimizer.step()
        diagnostics = optimizer.consume_diagnostics()[0]
    assert diagnostics is not None
    assert diagnostics["feedback_codec_energy_recovery"] > 0.98
    state = optimizer.state[module.weight]
    assert state["feedback_levels"].shape == (2, 16, 2)
    assert state["feedback_center"].shape == (2,)
    assert state["feedback_codes"].dtype == torch.uint8
    assert module.compact_feedback_bytes == module.element_count // 2 + 264
    assert all(value.numel() != module.element_count for value in state.values())


def test_rvq_feedback_resume_is_bit_exact_for_next_step() -> None:
    torch.manual_seed(173)
    module = make_module(
        stages=1,
        seed=179,
        error_feedback=True,
        feedback_codec="rvq4x4",
    )
    optimizer = make_optimizer(module)
    for _step in range(3):
        module.weight.grad = torch.randn_like(module.weight)
        optimizer.step()
    model_state = copy.deepcopy(module.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())

    restored = make_module(
        stages=1,
        seed=181,
        error_feedback=True,
        feedback_codec="rvq4x4",
    )
    restored.load_state_dict(model_state, strict=True)
    restored_optimizer = make_optimizer(restored)
    restored_optimizer.load_state_dict(optimizer_state)
    gradient = torch.randn_like(module.weight)
    module.weight.grad = gradient.clone()
    restored.weight.grad = gradient.clone()
    optimizer.step()
    restored_optimizer.step()
    torch.testing.assert_close(restored.weight, module.weight, rtol=0.0, atol=0.0)
    original_state = optimizer.state[module.weight]
    restored_state = restored_optimizer.state[restored.weight]
    assert torch.equal(restored_state["feedback_codes"], original_state["feedback_codes"])
    torch.testing.assert_close(
        restored_state["feedback_levels"],
        original_state["feedback_levels"],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        restored_state["feedback_center"],
        original_state["feedback_center"],
        rtol=0.0,
        atol=0.0,
    )


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
            block_fht_mlp_pair_vq_error_feedback=True,
        )
    )
    mlp = model.transformer.h[0].mlp
    assert isinstance(mlp.c_fc, MuonPairVQLinear)
    assert isinstance(mlp.c_proj, MuonPairVQLinear)
    assert mlp.c_fc.stages == 2
    assert mlp.c_proj.stages == 1
    assert mlp.c_fc.fast_residual is True
    assert mlp.c_proj.fast_residual is False
    assert mlp.c_fc.error_feedback is True
    assert mlp.c_proj.error_feedback is True
    stats = model.mlp_pair_vq_stats()
    assert stats["modules"] == 2
    assert stats["dense_master_weight"] == "disabled"
    assert stats["dense_optimizer_momentum"] == "disabled"
    assert stats["dense_ambient_error_buffer"] == "disabled"
    assert stats["compact_feedback_bytes"] > 0
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


def test_gpt_routes_optional_fast_residual_through_cproj() -> None:
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
            block_fht_mlp_pair_vq_error_feedback=True,
            block_fht_mlp_pair_vq_cproj_fast_residual=True,
            block_fht_mlp_pair_vq_feedback_codec="cartesian4x4",
            block_fht_mlp_pair_vq_feedback_output_group_size=2,
        )
    )
    mlp = model.transformer.h[0].mlp
    assert isinstance(mlp.c_proj, MuonPairVQLinear)
    assert mlp.c_proj.stages == 1
    assert mlp.c_proj.fast_residual is True
    assert mlp.c_proj.error_feedback is True
    assert mlp.c_proj.feedback_output_group_size == 2
    assert mlp.c_fc.feedback_output_group_size == 2
    pair_count = mlp.c_proj.element_count // mlp.c_proj.vector_length
    assert mlp.c_proj.fast_codes.numel() == pair_count
    assert mlp.c_proj.fast_levels.shape == (2, 16)
    assert all(
        value.numel() != mlp.c_proj.element_count
        for value in mlp.c_proj.state_dict().values()
    )


def test_pair_vq_training_boundary_forwards_cproj_fast_residual() -> None:
    namespace = Namespace(
        block_fht_mlp_pair_vq=True,
        block_fht_mlp_pair_vq_seed=20261020,
        block_fht_mlp_pair_vq_neighbor_candidates=16,
        block_fht_mlp_pair_vq_code_refresh_interval=8,
        block_fht_mlp_pair_vq_error_feedback=True,
        block_fht_mlp_pair_vq_cproj_fast_residual=True,
        block_fht_mlp_pair_vq_feedback_codec="polar32x8",
        block_fht_mlp_pair_vq_feedback_output_group_size=0,
    )
    kwargs = pair_vq_model_kwargs(namespace)
    assert kwargs == {
        "block_fht_mlp_pair_vq": True,
        "block_fht_mlp_pair_vq_seed": 20261020,
        "block_fht_mlp_pair_vq_neighbor_candidates": 16,
        "block_fht_mlp_pair_vq_code_refresh_interval": 8,
        "block_fht_mlp_pair_vq_error_feedback": True,
        "block_fht_mlp_pair_vq_cproj_fast_residual": True,
        "block_fht_mlp_pair_vq_feedback_codec": "polar32x8",
        "block_fht_mlp_pair_vq_feedback_output_group_size": 0,
    }
    config = GPTConfig(
        block_size=8,
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_embd=8,
        bias=False,
        block_fht=True,
        block_fht_targets=(),
        **kwargs,
    )
    model = GPT(config)
    assert model.transformer.h[0].mlp.c_proj.fast_residual is True
    assert model.transformer.h[0].mlp.c_proj.feedback_codec == "polar32x8"
