from __future__ import annotations

import copy

import torch

import pytest

from examples.nanogpt.analyze_mlp_cfc_multistage_directed import (
    fit_multistage_directed_sparse_mixer,
)
from examples.nanogpt.model import GPT, GPTConfig
from examples.nanogpt.train import require_block_fht_native_extension
from examples.nanogpt.muon_matched_givens import (
    MuonDirectedProduct,
    MuonDirectedProductLinear,
    MuonMatchedGivens,
    batched_multistage_directed_sparse_update,
)


def make_module(*, layer_id: int = 0) -> MuonDirectedProductLinear:
    torch.manual_seed(301 + layer_id)
    return MuonDirectedProductLinear(
        4,
        8,
        bias=False,
        incoming_schedule=(1, 1, 1),
        ridge_ratio=1e-6,
        chunk_size=3,
        family_radius_ratio=0.65,
        weight_std=0.02,
        layer_id=layer_id,
    )


def make_optimizer(
    modules: list[MuonDirectedProductLinear],
) -> MuonDirectedProduct:
    return MuonDirectedProduct(
        modules,
        lr=1e-3,
        momentum=0.95,
        weight_decay=0.1,
        ns_steps=5,
    )


def make_gpt_config() -> GPTConfig:
    return GPTConfig(
        block_size=8,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_embd=8,
        block_fht=True,
        block_fht_targets=("mlp.c_proj",),
        block_fht_mlp_cproj_muon_matched_givens=True,
        block_fht_mlp_cproj_muon_matched_givens_stages=1,
        block_fht_mlp_cproj_muon_matched_givens_residual_stages=0,
        block_fht_mlp_cproj_muon_matched_givens_neighbors=2,
        block_fht_mlp_cproj_muon_matched_givens_refresh_interval=3,
        block_fht_mlp_cfc_directed_product=True,
        block_fht_mlp_cfc_directed_product_schedule=(1, 1, 1),
        block_fht_mlp_cfc_directed_product_ridge_ratio=1e-6,
        block_fht_mlp_cfc_directed_product_chunk_size=5,
        block_fht_mlp_cfc_directed_product_family_radius_ratio=0.65,
    )


def test_batched_solver_matches_registered_scalar_solver() -> None:
    torch.manual_seed(307)
    source = torch.randn(2, 6, 8)
    target = torch.randn_like(source) * 0.1
    schedule = (2, 1, 1)
    actual, rows = batched_multistage_directed_sparse_update(
        source,
        target,
        incoming_schedule=schedule,
        ridge_ratio=1e-6,
        chunk_size=3,
    )
    expected = []
    for member in range(source.shape[0]):
        prediction, _row = fit_multistage_directed_sparse_mixer(
            source[member],
            target[member],
            incoming_schedule=list(schedule),
            ridge_ratio=1e-6,
            chunk_size=3,
        )
        expected.append(prediction)
    assert torch.allclose(actual, torch.stack(expected), atol=2e-5, rtol=2e-5)
    assert [row["incoming_per_target"] for row in rows] == list(schedule)


def test_module_exposes_only_sparse_coordinate_budget() -> None:
    module = make_module()
    assert module.coordinate_count == 24
    assert not isinstance(module.weight, torch.nn.Parameter)
    assert set(module.state_dict()) == {"weight", "optimizer_step"}
    assert all(
        tensor.shape != (module.out_features, module.out_features)
        for tensor in module.state_dict().values()
    )


def test_forward_gradient_and_family_radius_are_finite() -> None:
    modules = [make_module(layer_id=index) for index in range(2)]
    optimizer = make_optimizer(modules)
    before = [module.weight.clone() for module in modules]
    loss = sum(module(torch.randn(3, 4)).square().mean() for module in modules)
    loss.backward()
    assert all(torch.isfinite(module.weight.grad).all() for module in modules)
    optimizer.step()
    diagnostics = optimizer.consume_diagnostics()
    actual_radius = torch.stack(
        [
            (module.weight - old).float().square().sum()
            for module, old in zip(modules, before, strict=True)
        ]
    ).sum().sqrt()
    assert torch.isfinite(actual_radius)
    assert torch.allclose(
        actual_radius,
        torch.tensor(diagnostics[0]["target_family_fro"]),
        atol=1e-7,
        rtol=2e-5,
    )
    assert all(row["coordinates"] == 24 for row in diagnostics)
    assert all(int(module.optimizer_step) == 1 for module in modules)


def test_optimizer_resume_is_exact() -> None:
    modules = [make_module(layer_id=index) for index in range(2)]
    optimizer = make_optimizer(modules)
    generator = torch.Generator().manual_seed(313)
    for module in modules:
        module.weight.grad = torch.randn(
            module.weight.shape, generator=generator
        )
    optimizer.step()
    module_states = [copy.deepcopy(module.state_dict()) for module in modules]
    optimizer_state = copy.deepcopy(optimizer.state_dict())

    resumed = [make_module(layer_id=index) for index in range(2)]
    resumed_optimizer = make_optimizer(resumed)
    for module, state in zip(resumed, module_states, strict=True):
        module.load_state_dict(state)
    resumed_optimizer.load_state_dict(optimizer_state)

    next_gradients = [
        torch.randn(module.weight.shape, generator=generator)
        for module in modules
    ]
    for module, gradient in zip(modules, next_gradients, strict=True):
        module.weight.grad = gradient.clone()
    for module, gradient in zip(resumed, next_gradients, strict=True):
        module.weight.grad = gradient.clone()
    optimizer.step()
    resumed_optimizer.step()
    for original, restored in zip(modules, resumed, strict=True):
        assert torch.equal(original.weight, restored.weight)
        assert torch.equal(original.optimizer_step, restored.optimizer_step)


def test_gpt_wiring_optimizer_assignment_and_stats() -> None:
    model = GPT(make_gpt_config())
    modules = [block.mlp.c_fc for block in model.transformer.h]
    assert all(isinstance(module, MuonDirectedProductLinear) for module in modules)
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=1e-3,
        betas=(0.9, 0.95),
        device_type="cpu",
        optimizer="muon",
    )
    directed = next(
        item
        for item in optimizer.optimizers
        if isinstance(item, MuonDirectedProduct)
    )
    cproj = next(
        item for item in optimizer.optimizers if isinstance(item, MuonMatchedGivens)
    )
    assert optimizer.optimizers.index(directed) < optimizer.optimizers.index(cproj)
    tokens = torch.randint(0, 32, (2, 8))
    _logits, loss = model(tokens, tokens)
    assert loss is not None
    loss.backward()
    optimizer.step()
    assert all(int(module.optimizer_step) == 1 for module in modules)
    # Per layer: 3*32 directed c_fc coordinates and 1*16 c_proj angles.
    assert model.block_fht_stats()["latent"] == 2 * (96 + 16)


def test_directed_cfc_preserves_dense_paired_seed_initialization() -> None:
    dense_config = make_gpt_config()
    dense_config.block_fht_mlp_cfc_directed_product = False
    directed_config = copy.deepcopy(dense_config)
    directed_config.block_fht_mlp_cfc_directed_product = True
    torch.manual_seed(317)
    dense = GPT(dense_config)
    torch.manual_seed(317)
    directed = GPT(directed_config)
    for dense_block, directed_block in zip(
        dense.transformer.h, directed.transformer.h, strict=True
    ):
        assert torch.equal(
            dense_block.mlp.c_fc.weight,
            directed_block.mlp.c_fc.weight,
        )


def test_native_extension_guard_fails_closed(monkeypatch) -> None:
    from latent_weight_lab import block_fht as block_fht_module

    assert require_block_fht_native_extension(False) is False
    monkeypatch.setattr(block_fht_module, "_load_block_fht_ext", lambda: object())
    assert require_block_fht_native_extension(True) is True
    monkeypatch.setattr(block_fht_module, "_load_block_fht_ext", lambda: None)
    monkeypatch.setattr(
        block_fht_module,
        "_BLOCK_FHT_EXT_ERROR",
        RuntimeError("missing native test backend"),
    )
    with pytest.raises(RuntimeError, match="missing native test backend"):
        require_block_fht_native_extension(True)
