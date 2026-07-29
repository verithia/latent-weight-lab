from __future__ import annotations

import copy

import torch

from examples.nanogpt.model import GPT, GPTConfig
from examples.nanogpt.muon_matched_givens import (
    MuonMatchedGivens,
    MuonMatchedGivensLinear,
    apply_givens_flow,
    diagonal_metric_angles,
    random_unique_matchings,
)


def make_module(*, layer_id: int = 0) -> MuonMatchedGivensLinear:
    torch.manual_seed(17)
    return MuonMatchedGivensLinear(
        8,
        4,
        bias=False,
        stages=1,
        neighbors=2,
        refresh_interval=3,
        matching_seed=23,
        weight_std=0.02,
        layer_id=layer_id,
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
        block_fht_mlp_cproj_muon_matched_givens_neighbors=2,
        block_fht_mlp_cproj_muon_matched_givens_refresh_interval=3,
    )


def test_diagonal_metric_function_recovers_small_in_chart_update() -> None:
    torch.manual_seed(13)
    source = torch.randn(5, 8)
    permutations = torch.arange(8).view(1, 8)
    requested = torch.zeros_like(source)
    angle = 0.001
    requested[:, 0] = -angle * source[:, 1]
    requested[:, 1] = angle * source[:, 0]
    angles = diagonal_metric_angles(
        source, requested, permutations
    )
    predicted = (
        apply_givens_flow(source, angles, permutations) - source
    )
    cosine = torch.nn.functional.cosine_similarity(
        requested.flatten(), predicted.flatten(), dim=0
    )
    recovery = 1.0 - (
        requested - predicted
    ).square().sum() / requested.square().sum()
    assert float(cosine) > 0.999999
    assert float(recovery) > 0.99999


def test_random_unique_matchings_are_deterministic_and_edge_disjoint() -> None:
    matchings = random_unique_matchings(width=8, stages=7, seed=29)
    assert torch.equal(
        matchings,
        random_unique_matchings(width=8, stages=7, seed=29),
    )
    edges = {
        tuple(sorted(pair.tolist()))
        for matching in matchings
        for pair in matching.view(-1, 2)
    }
    assert len(edges) == 8 * 7 // 2
    assert all(
        torch.equal(torch.sort(row).values, torch.arange(8))
        for row in matchings
    )


def test_folded_weight_is_a_buffer_with_compact_coordinate_count() -> None:
    module = make_module()
    assert dict(module.named_parameters()) == {}
    assert "weight" in dict(module.named_buffers())
    assert module.coordinate_count == 4


def test_optimizer_refreshes_folds_and_round_trips_exact_state() -> None:
    module = make_module(layer_id=3)
    optimizer = MuonMatchedGivens(
        [module],
        lr=0.001,
        momentum=0.95,
        weight_decay=0.1,
        ns_steps=2,
    )
    original = module.weight.detach().clone()
    module.weight.grad = torch.randn_like(module.weight)
    optimizer.step()
    diagnostics = optimizer.consume_diagnostics()
    assert len(diagnostics) == 1
    assert diagnostics[0]["layer"] == 3
    assert diagnostics[0]["refresh"] is True
    assert int(module.optimizer_step) == 1
    assert int(module.last_refresh_step) == 0
    assert int(module.refresh_count) == 1
    assert bool(module.matching_valid)
    assert not torch.equal(module.weight, original)
    assert torch.isfinite(module.last_angles).all()

    module_state = copy.deepcopy(module.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    restored = make_module(layer_id=3)
    restored.load_state_dict(module_state)
    restored_optimizer = MuonMatchedGivens(
        [restored],
        lr=0.001,
        momentum=0.95,
        weight_decay=0.1,
        ns_steps=2,
    )
    restored_optimizer.load_state_dict(optimizer_state)
    for key, value in module.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
    original_momentum = optimizer.state[module.weight][
        "momentum_buffer"
    ]
    restored_momentum = restored_optimizer.state[restored.weight][
        "momentum_buffer"
    ]
    assert torch.equal(original_momentum, restored_momentum)


def test_stage64_optimizer_state_round_trip() -> None:
    torch.manual_seed(31)
    module = MuonMatchedGivensLinear(
        128,
        16,
        bias=False,
        stages=64,
        neighbors=64,
        refresh_interval=60,
        matching_seed=161803,
        weight_std=0.02,
        layer_id=7,
    )
    optimizer = MuonMatchedGivens(
        [module],
        lr=0.0024,
        momentum=0.95,
        weight_decay=0.1,
        ns_steps=2,
    )
    module.weight.grad = torch.randn_like(module.weight)
    optimizer.step()
    assert module.selected_permutations.shape == (64, 128)
    assert module.selected_inverse_permutations.shape == (64, 128)
    assert module.last_angles.shape == (64, 64)
    assert module.coordinate_count == 4096
    assert int(module.optimizer_step) == 1
    assert int(module.refresh_count) == 1
    assert bool(module.matching_valid)

    module_state = copy.deepcopy(module.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    restored = MuonMatchedGivensLinear(
        128,
        16,
        bias=False,
        stages=64,
        neighbors=64,
        refresh_interval=60,
        matching_seed=161803,
        weight_std=0.02,
        layer_id=7,
    )
    restored.load_state_dict(module_state)
    restored_optimizer = MuonMatchedGivens(
        [restored],
        lr=0.0024,
        momentum=0.95,
        weight_decay=0.1,
        ns_steps=2,
    )
    restored_optimizer.load_state_dict(optimizer_state)
    for key, value in module_state.items():
        assert torch.equal(value, restored.state_dict()[key])
    assert torch.equal(
        optimizer.state[module.weight]["momentum_buffer"],
        restored_optimizer.state[restored.weight]["momentum_buffer"],
    )


def test_gpt_wires_custom_cproj_into_muon_optimizer_and_stats() -> None:
    model = GPT(make_gpt_config())
    modules = [
        module
        for module in model.modules()
        if isinstance(module, MuonMatchedGivensLinear)
    ]
    assert len(modules) == 2
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=0.001,
        betas=(0.9, 0.95),
        device_type="cpu",
        optimizer="muon",
    )
    custom = [
        candidate
        for candidate in optimizer.optimizers
        if isinstance(candidate, MuonMatchedGivens)
    ]
    assert len(custom) == 1
    stats = model.block_fht_stats()
    assert stats["modules"] == 2
    assert stats["generated"] == 2 * 8 * 32
    assert stats["latent"] == 2 * 16
    clip_ids = {
        id(tensor) for tensor in model.product_fht_clip_parameters()
    }
    assert all(id(module.weight) in clip_ids for module in modules)


def test_gpt_backward_step_and_full_checkpoint_round_trip() -> None:
    torch.manual_seed(41)
    config = make_gpt_config()
    model = GPT(config)
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=0.001,
        betas=(0.9, 0.95),
        device_type="cpu",
        optimizer="muon",
    )
    tokens = torch.randint(0, config.vocab_size, (2, config.block_size))
    _logits, loss = model(tokens, tokens)
    assert loss is not None
    loss.backward()
    custom_modules = [
        module
        for module in model.modules()
        if isinstance(module, MuonMatchedGivensLinear)
    ]
    assert all(module.weight.grad is not None for module in custom_modules)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert all(int(module.optimizer_step) == 1 for module in custom_modules)
    assert all(module.weight.grad is None for module in custom_modules)

    model_state = copy.deepcopy(model.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    restored = GPT(config)
    restored.load_state_dict(model_state)
    restored_optimizer = restored.configure_optimizers(
        weight_decay=0.1,
        learning_rate=0.001,
        betas=(0.9, 0.95),
        device_type="cpu",
        optimizer="muon",
    )
    restored_optimizer.load_state_dict(optimizer_state)
    for key, value in model_state.items():
        assert torch.equal(value, restored.state_dict()[key])
    assert optimizer_state.keys() == restored_optimizer.state_dict().keys()
    original_custom = next(
        candidate
        for candidate in optimizer.optimizers
        if isinstance(candidate, MuonMatchedGivens)
    )
    restored_custom = next(
        candidate
        for candidate in restored_optimizer.optimizers
        if isinstance(candidate, MuonMatchedGivens)
    )
    for original_module, restored_module in zip(
        custom_modules,
        (
            module
            for module in restored.modules()
            if isinstance(module, MuonMatchedGivensLinear)
        ),
        strict=True,
    ):
        assert torch.equal(
            original_custom.state[original_module.weight][
                "momentum_buffer"
            ],
            restored_custom.state[restored_module.weight][
                "momentum_buffer"
            ],
        )
