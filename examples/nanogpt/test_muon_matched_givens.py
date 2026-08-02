from __future__ import annotations

import copy
import math

import torch

from examples.nanogpt.model import GPT, GPTConfig
from examples.nanogpt.muon_matched_givens import (
    MuonFunctionalShear,
    MuonFunctionalShearLinear,
    MuonMatchedGivens,
    MuonMatchedGivensLinear,
    _apply_symmetric_shear_stage,
    _fit_functional_shear_recipe,
    _fit_weight_shear_recipe,
    apply_givens_flow,
    diagonal_metric_angles,
    mix_shear_recipes,
    random_unique_matchings,
)
from examples.nanogpt.analyze_mlp_cfc_functional_shear_fit import (
    fit_functional_shear_recipe,
)
from examples.nanogpt.analyze_mlp_cfc_task_shear_fit import (
    apply_pair_stage,
    fit_pair_recipe,
)


def make_module(*, layer_id: int = 0) -> MuonMatchedGivensLinear:
    torch.manual_seed(17)
    return MuonMatchedGivensLinear(
        8,
        4,
        bias=False,
        stages=1,
        residual_stages=0,
        neighbors=2,
        refresh_interval=3,
        fast_fresh_matching=False,
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
        block_fht_mlp_cproj_muon_matched_givens_residual_stages=0,
        block_fht_mlp_cproj_muon_matched_givens_neighbors=2,
        block_fht_mlp_cproj_muon_matched_givens_refresh_interval=3,
        block_fht_mlp_cproj_muon_matched_givens_fast_fresh=False,
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
    assert not any(
        key.startswith("residual_") for key in module.state_dict()
    )


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


def test_fast_fresh_optimizer_reselects_every_step(
    monkeypatch,
) -> None:
    calls: list[int] = []

    def fake_fast_matching(
        weight: torch.Tensor,
        direction: torch.Tensor,
        *,
        stages: int,
        neighbors: int,
        seed: int,
    ):
        del direction, neighbors
        calls.append(seed)
        return (
            torch.arange(weight.shape[1]).repeat(stages, 1),
            {
                "candidate_edge_fraction": 1.0,
                "minimum_stage_candidate_edge_fraction": 1.0,
                "prepared_seconds": 0.001,
                "native_seconds": 0.002,
                "total_seconds": 0.003,
                "native_output_validated": True,
                "native_library_sha256": "library",
                "source_sha256": "source",
            },
        )

    monkeypatch.setattr(
        "examples.nanogpt.muon_matched_givens."
        "fast_muon_matched_permutations",
        fake_fast_matching,
    )
    module = MuonMatchedGivensLinear(
        8,
        4,
        bias=False,
        stages=1,
        residual_stages=0,
        neighbors=2,
        refresh_interval=1,
        fast_fresh_matching=True,
        matching_seed=23,
        weight_std=0.02,
        layer_id=3,
    )
    optimizer = MuonMatchedGivens(
        [module],
        lr=0.001,
        momentum=0.95,
        weight_decay=0.1,
        ns_steps=2,
    )
    report_flags: list[bool] = []
    for _step in range(2):
        module.weight.grad = torch.randn_like(module.weight)
        optimizer.step()
        diagnostics = optimizer.consume_diagnostics()
        assert len(diagnostics) == 1
        assert diagnostics[0]["fast_fresh_matching"] is True
        assert diagnostics[0]["refresh"] is True
        assert diagnostics[0]["matching"]["selector"] == (
            "fast_fresh_single_pass"
        )
        report_flags.append(bool(diagnostics[0]["report_refresh"]))
    assert calls == [23, 24]
    assert report_flags == [True, False]
    assert int(module.last_refresh_step) == 1
    assert int(module.refresh_count) == 2


def test_fast_fresh_residual_pass_fits_after_parent_residual(
    monkeypatch,
) -> None:
    calls: list[tuple[torch.Tensor, torch.Tensor, int, int]] = []

    def fake_fast_matching(
        weight: torch.Tensor,
        direction: torch.Tensor,
        *,
        stages: int,
        neighbors: int,
        seed: int,
    ):
        del neighbors
        calls.append(
            (
                weight.detach().clone(),
                direction.detach().clone(),
                stages,
                seed,
            )
        )
        return (
            torch.arange(weight.shape[1]).repeat(stages, 1),
            {
                "candidate_edge_fraction": 1.0,
                "minimum_stage_candidate_edge_fraction": 1.0,
                "prepared_seconds": 0.001,
                "native_seconds": 0.002,
                "total_seconds": 0.003,
                "native_output_validated": True,
                "native_library_sha256": "library",
                "source_sha256": "source",
            },
        )

    monkeypatch.setattr(
        "examples.nanogpt.muon_matched_givens."
        "fast_muon_matched_permutations",
        fake_fast_matching,
    )
    module = MuonMatchedGivensLinear(
        8,
        4,
        bias=False,
        stages=1,
        residual_stages=1,
        neighbors=2,
        refresh_interval=1,
        fast_fresh_matching=True,
        matching_seed=23,
        weight_std=0.02,
        layer_id=3,
    )
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

    assert len(calls) == 2
    parent_source, parent_direction, parent_stages, parent_seed = calls[0]
    residual_source, residual_direction, residual_stages, residual_seed = (
        calls[1]
    )
    requested = 0.001 * (
        parent_direction - 0.1 * parent_source.float()
    )
    parent_angles = diagonal_metric_angles(
        parent_source,
        requested,
        torch.arange(8).view(1, 8),
    )
    expected_after_parent = apply_givens_flow(
        parent_source,
        parent_angles,
        torch.arange(8).view(1, 8),
    )
    expected_residual = (
        requested - (expected_after_parent - parent_source)
    )
    assert parent_stages == 1
    assert parent_seed == 23
    assert residual_stages == 1
    assert residual_seed == 24
    assert torch.allclose(residual_source, expected_after_parent)
    assert torch.allclose(residual_direction, expected_residual)
    assert module.coordinate_count == 8
    assert module.residual_selected_permutations.shape == (1, 8)
    assert module.residual_selected_inverse_permutations.shape == (1, 8)
    assert module.residual_last_angles.shape == (1, 4)
    assert torch.isfinite(module.residual_last_angles).all()
    assert diagnostics[0]["parent_stages"] == 1
    assert diagnostics[0]["residual_stages"] == 1
    assert diagnostics[0]["residual_matching"]["selector"] == (
        "fast_fresh_residual_pass"
    )
    assert not torch.equal(module.weight, original)

    module_state = copy.deepcopy(module.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    restored = MuonMatchedGivensLinear(
        8,
        4,
        bias=False,
        stages=1,
        residual_stages=1,
        neighbors=2,
        refresh_interval=1,
        fast_fresh_matching=True,
        matching_seed=23,
        weight_std=0.02,
        layer_id=3,
    )
    restored.load_state_dict(module_state)
    restored_optimizer = MuonMatchedGivens(
        [restored],
        lr=0.001,
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


def test_stage64_optimizer_state_round_trip() -> None:
    torch.manual_seed(31)
    module = MuonMatchedGivensLinear(
        128,
        16,
        bias=False,
        stages=64,
        residual_stages=0,
        neighbors=64,
        refresh_interval=60,
        fast_fresh_matching=False,
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
        residual_stages=0,
        neighbors=64,
        refresh_interval=60,
        fast_fresh_matching=False,
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


def test_gpt_wires_residual_stages_into_coordinate_stats() -> None:
    config = make_gpt_config()
    config.block_fht_mlp_cproj_muon_matched_givens_refresh_interval = 1
    config.block_fht_mlp_cproj_muon_matched_givens_fast_fresh = True
    config.block_fht_mlp_cproj_muon_matched_givens_residual_stages = 1
    model = GPT(config)
    modules = [
        module
        for module in model.modules()
        if isinstance(module, MuonMatchedGivensLinear)
    ]
    assert len(modules) == 2
    assert all(module.residual_stages == 1 for module in modules)
    assert all(module.coordinate_count == 32 for module in modules)
    stats = model.block_fht_stats()
    assert stats["latent"] == 64


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


def test_production_shear_recipes_match_registered_diagnostic() -> None:
    torch.manual_seed(101)
    source = torch.randn(5, 8) * 0.02
    requested = torch.randn_like(source) * 0.001
    inputs = torch.randn(11, 5)
    pre_gelu = inputs @ source
    cproj = torch.randn(5, 8) * 0.02
    permutations = random_unique_matchings(
        width=8, stages=2, seed=103
    )
    production_weight = _fit_weight_shear_recipe(
        source, requested, permutations
    )
    _update, _diagnostics, registered_weight = fit_pair_recipe(
        source,
        requested,
        permutations,
        stages=2,
        family="shear",
    )
    production_function = _fit_functional_shear_recipe(
        source,
        requested,
        inputs,
        pre_gelu,
        cproj,
        permutations,
    )
    _update, _diagnostics, registered_function = (
        fit_functional_shear_recipe(
            source,
            requested,
            inputs,
            pre_gelu,
            cproj,
            permutations,
            stages=2,
        )
    )
    for production, registered in (
        (production_weight, registered_weight),
        (production_function, registered_function),
    ):
        current_production = source
        current_registered = source
        for (pairs_a, coordinate_a), (pairs_b, coordinate_b) in zip(
            production, registered, strict=True
        ):
            assert torch.equal(pairs_a, pairs_b)
            assert torch.allclose(
                coordinate_a,
                coordinate_b[:, 0],
                rtol=2e-6,
                atol=2e-9,
            )
            current_production = _apply_symmetric_shear_stage(
                current_production, pairs_a, coordinate_a
            )
            current_registered, _finite = apply_pair_stage(
                current_registered, pairs_b, coordinate_b
            )
        assert torch.allclose(
            current_production,
            current_registered,
            rtol=2e-6,
            atol=2e-7,
        )


def test_functional_coordinate_diagnostics_bound_pair_condition() -> None:
    torch.manual_seed(107)
    source = torch.randn(8, 5) * 0.02
    requested = torch.randn_like(source) * 0.0001
    inputs = torch.randn(11, 5)
    pre_gelu = inputs @ source.T
    cproj = torch.randn(5, 8) * 0.02
    from examples.nanogpt.muon_matched_givens import (
        functional_coordinate_mix_update,
    )

    update, diagnostics = functional_coordinate_mix_update(
        source,
        requested,
        requested,
        inputs,
        pre_gelu,
        cproj,
        parent_stages=2,
        shear_stages=1,
        neighbors=2,
        seed=109,
        beta=0.5,
        project_to_weight_norm=False,
        learning_rate=0.001,
        weight_decay=0.1,
    )
    assert torch.isfinite(update).all()
    assert diagnostics["coordinate_finite_fraction"] == 1.0
    assert diagnostics["update_finite_fraction"] == 1.0
    assert diagnostics["shear_max_abs"] >= diagnostics["shear_rms"]
    assert diagnostics["shear_log_condition_bound"] >= (
        2.0 * diagnostics["shear_max_abs"]
    )
    assert math.isfinite(diagnostics["weight_rms_ratio"])
    assert diagnostics["weight_rms_ratio"] > 0.0


def test_mixed_functional_direction_projects_to_weight_recipe_norm() -> None:
    pairs = torch.tensor([[0, 1], [2, 3]])
    weight_recipe = [(pairs, torch.tensor([0.2, -0.1]))]
    functional_recipe = [(pairs, torch.tensor([1.0, 0.5]))]
    mixed, diagnostics = mix_shear_recipes(
        weight_recipe,
        functional_recipe,
        beta=0.5,
        project_to_weight_norm=True,
    )
    raw = 0.5 * weight_recipe[0][1] + 0.5 * functional_recipe[0][1]
    projected = mixed[0][1]
    assert torch.allclose(projected / projected.norm(), raw / raw.norm())
    assert torch.allclose(projected.norm(), weight_recipe[0][1].norm())
    assert diagnostics["coordinate_norm_projection_active"] is True
    assert diagnostics["coordinate_norm_projection_scale"] < 1.0


def test_gpt_wires_functional_cfc_and_consumes_bounded_context(
    monkeypatch,
) -> None:
    config = make_gpt_config()
    config.block_fht_mlp_cfc_functional_shear = True
    config.block_fht_mlp_cfc_functional_shear_parent_stages = 2
    config.block_fht_mlp_cfc_functional_shear_stages = 1
    config.block_fht_mlp_cfc_functional_shear_neighbors = 2
    config.block_fht_mlp_cfc_functional_shear_sample_cap = 5
    model = GPT(config)
    modules = [
        block.mlp.c_fc for block in model.transformer.h
    ]
    assert all(
        isinstance(module, MuonFunctionalShearLinear)
        for module in modules
    )
    assert all(module.coordinate_count == 48 for module in modules)

    observed_samples: list[int] = []

    def fake_update(
        weight,
        requested_update,
        selection_direction,
        inputs,
        pre_gelu,
        cproj_weight,
        **kwargs,
    ):
        del selection_direction, pre_gelu, cproj_weight, kwargs
        observed_samples.append(int(inputs.shape[0]))
        return requested_update.to(weight), {
            "coordinates": 48,
            "functional_samples": int(inputs.shape[0]),
        }

    monkeypatch.setattr(
        "examples.nanogpt.muon_matched_givens."
        "functional_coordinate_mix_update",
        fake_update,
    )
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=0.001,
        betas=(0.9, 0.95),
        device_type="cpu",
        optimizer="muon",
    )
    functional_optimizer = next(
        candidate
        for candidate in optimizer.optimizers
        if isinstance(candidate, MuonFunctionalShear)
    )
    cproj_optimizer = next(
        candidate
        for candidate in optimizer.optimizers
        if isinstance(candidate, MuonMatchedGivens)
    )
    assert optimizer.optimizers.index(functional_optimizer) < (
        optimizer.optimizers.index(cproj_optimizer)
    )
    tokens = torch.randint(0, config.vocab_size, (2, config.block_size))
    _logits, loss = model(tokens, tokens)
    assert loss is not None
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert observed_samples == [5, 5]
    assert all(int(module.optimizer_step) == 1 for module in modules)
    assert all(module._functional_inputs is None for module in modules)
    assert model.block_fht_stats()["latent"] == 2 * (48 + 16)


def test_functional_cfc_preserves_dense_paired_seed_initialization() -> None:
    dense_config = make_gpt_config()
    functional_config = copy.deepcopy(dense_config)
    functional_config.block_fht_mlp_cfc_functional_shear = True
    functional_config.block_fht_mlp_cfc_functional_shear_parent_stages = 2
    functional_config.block_fht_mlp_cfc_functional_shear_stages = 1
    functional_config.block_fht_mlp_cfc_functional_shear_neighbors = 2
    functional_config.block_fht_mlp_cfc_functional_shear_sample_cap = 5
    torch.manual_seed(211)
    dense = GPT(dense_config)
    torch.manual_seed(211)
    functional = GPT(functional_config)
    for dense_block, functional_block in zip(
        dense.transformer.h, functional.transformer.h, strict=True
    ):
        assert torch.equal(
            dense_block.mlp.c_fc.weight,
            functional_block.mlp.c_fc.weight,
        )
