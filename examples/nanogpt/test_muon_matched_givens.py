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
    diagonal_input_metric_angles,
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


def make_activation_metric_module(
    *, layer_id: int = 0
) -> MuonMatchedGivensLinear:
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
        activation_energy_metric=True,
        activation_energy_metric_decay=0.95,
        activation_energy_metric_minimum=0.25,
        activation_energy_metric_maximum=4.0,
        activation_energy_metric_epsilon=1e-6,
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


def test_diagonal_input_metric_recovers_small_in_chart_update() -> None:
    torch.manual_seed(131)
    source = torch.randn(5, 8)
    permutations = torch.arange(8).view(1, 8)
    requested = torch.zeros_like(source)
    angle = 0.001
    requested[:, 0] = -angle * source[:, 1]
    requested[:, 1] = angle * source[:, 0]
    metric = torch.tensor([0.25, 4.0, 0.5, 2.0, 1.0, 3.0, 0.75, 1.5])
    angles = diagonal_input_metric_angles(
        source, requested, permutations, metric
    )
    assert torch.allclose(angles[0, 0], torch.tensor(angle), atol=1e-7)
    predicted = apply_givens_flow(source, angles, permutations) - source
    weighted_target = requested.square() * metric
    weighted_error = (requested - predicted).square() * metric
    recovery = 1.0 - weighted_error.sum() / weighted_target.sum()
    assert float(recovery) > 0.99999


def test_activation_energy_metric_aggregates_training_microbatches() -> None:
    module = make_activation_metric_module()
    first = torch.tensor(
        [[[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]]]
    )
    second = torch.tensor(
        [[[3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0]]]
    )
    module.record_activation_energy_context(first)
    module.record_activation_energy_context(second)
    metric, count = module.consume_activation_energy_metric()
    assert count == 2
    assert metric is not None
    observed = (first.square().sum((0, 1)) + second.square().sum((0, 1))) / 2
    expected = (observed / (observed.mean() + 1e-6)).clamp(0.25, 4.0)
    assert torch.allclose(module.activation_energy_ema, observed)
    assert torch.allclose(metric, expected)
    assert int(module.activation_energy_updates) == 1
    assert module._activation_energy_sum is None
    assert module._activation_energy_count == 0

    module.eval()
    module.record_activation_energy_context(torch.full((2, 8), 100.0))
    assert module._activation_energy_sum is None
    assert module._activation_energy_count == 0


def test_activation_metric_state_and_optimizer_round_trip_exactly() -> None:
    module = make_activation_metric_module(layer_id=3)
    optimizer = MuonMatchedGivens(
        [module],
        lr=0.001,
        momentum=0.95,
        weight_decay=0.1,
        ns_steps=2,
    )
    module.record_activation_energy_context(torch.randn(2, 3, 8))
    module.weight.grad = torch.randn_like(module.weight)
    optimizer.step()
    diagnostics = optimizer.consume_diagnostics()
    assert diagnostics[0]["activation_energy_metric"] is True
    assert diagnostics[0]["activation_energy_metric_samples"] == 6
    assert 0.25 <= diagnostics[0]["activation_energy_metric_min"]
    assert diagnostics[0]["activation_energy_metric_max"] <= 4.0
    assert diagnostics[0]["activation_energy_metric_condition"] <= 16.0
    assert math.isfinite(
        diagnostics[0]["activation_weighted_corrected_target_recovery"]
    )

    module_state = copy.deepcopy(module.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    restored = make_activation_metric_module(layer_id=3)
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
    assert torch.equal(
        optimizer.state[module.weight]["momentum_buffer"],
        restored_optimizer.state[restored.weight]["momentum_buffer"],
    )


def test_activation_metric_default_off_preserves_state_and_update() -> None:
    control = make_module()
    torch.manual_seed(17)
    explicit = MuonMatchedGivensLinear(
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
        activation_energy_metric=False,
    )
    assert set(control.state_dict()) == set(explicit.state_dict())
    gradient = torch.randn_like(control.weight)
    optimizers = [
        MuonMatchedGivens(
            [module],
            lr=0.001,
            momentum=0.95,
            weight_decay=0.1,
            ns_steps=2,
        )
        for module in (control, explicit)
    ]
    control.weight.grad = gradient.clone()
    explicit.weight.grad = gradient.clone()
    optimizers[0].step()
    optimizers[1].step()
    assert torch.equal(control.weight, explicit.weight)


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


def test_cproj_error_feedback_preserves_first_step_and_resumes_exactly(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "examples.nanogpt.muon_matched_givens.zeropower_via_newtonschulz5",
        lambda matrix, steps: matrix,
    )
    control = make_module(layer_id=3)
    candidate = make_module(layer_id=3)
    candidate.load_state_dict(copy.deepcopy(control.state_dict()))
    control_optimizer = MuonMatchedGivens(
        [control],
        lr=0.001,
        momentum=0.95,
        weight_decay=0.1,
        ns_steps=2,
    )
    candidate_optimizer = MuonMatchedGivens(
        [candidate],
        lr=0.001,
        momentum=0.95,
        weight_decay=0.1,
        ns_steps=2,
        error_feedback=True,
        error_feedback_decay=1.0,
    )
    original = control.weight.detach().clone()
    torch.manual_seed(41)
    gradient = torch.randn_like(control.weight)
    control.weight.grad = gradient.clone()
    candidate.weight.grad = gradient.clone()
    control_optimizer.step()
    candidate_optimizer.step()
    assert torch.equal(control.weight, candidate.weight)

    combined = gradient + 0.95 * gradient
    requested = 0.001 * (-combined - 0.1 * original.float())
    applied = candidate.weight.float() - original.float()
    expected_residual = requested - applied
    residual = candidate_optimizer.state[candidate.weight][
        "compression_residual"
    ]
    torch.testing.assert_close(residual, expected_residual)
    diagnostics = candidate_optimizer.consume_diagnostics()
    assert diagnostics[0]["error_feedback"] is True
    assert diagnostics[0]["feedback_input_fro"] == 0.0
    assert math.isclose(
        diagnostics[0]["feedback_output_fro"],
        float(expected_residual.detach().norm()),
        rel_tol=1e-6,
    )

    module_state = copy.deepcopy(candidate.state_dict())
    optimizer_state = copy.deepcopy(candidate_optimizer.state_dict())
    restored = make_module(layer_id=3)
    restored.load_state_dict(module_state)
    restored_optimizer = MuonMatchedGivens(
        [restored],
        lr=0.001,
        momentum=0.95,
        weight_decay=0.1,
        ns_steps=2,
        error_feedback=True,
        error_feedback_decay=1.0,
    )
    restored_optimizer.load_state_dict(optimizer_state)
    torch.manual_seed(43)
    next_gradient = torch.randn_like(candidate.weight)
    candidate.weight.grad = next_gradient.clone()
    restored.weight.grad = next_gradient.clone()
    candidate_optimizer.step()
    restored_optimizer.step()
    assert torch.equal(candidate.weight, restored.weight)
    assert torch.equal(
        candidate_optimizer.state[candidate.weight]["momentum_buffer"],
        restored_optimizer.state[restored.weight]["momentum_buffer"],
    )
    assert torch.equal(
        candidate_optimizer.state[candidate.weight]["compression_residual"],
        restored_optimizer.state[restored.weight]["compression_residual"],
    )


def test_cproj_compact_optimizer_state_persists_and_resumes_exactly(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "examples.nanogpt.muon_matched_givens.zeropower_via_newtonschulz5",
        lambda matrix, steps: matrix,
    )
    module = make_module(layer_id=5)
    kwargs = {
        "lr": 0.001,
        "momentum": 0.95,
        "weight_decay": 0.1,
        "ns_steps": 2,
        "error_feedback": True,
        "error_feedback_decay": 1.0,
        "momentum_state_dtype": "float16",
        "feedback_state_codec": "int8_blockwise",
        "feedback_state_block_size": 8,
    }
    optimizer = MuonMatchedGivens([module], **kwargs)
    generator = torch.Generator().manual_seed(1311)
    module.weight.grad = torch.randn(module.weight.shape, generator=generator)
    optimizer.step()
    state = optimizer.state[module.weight]
    assert state["momentum_buffer"].dtype == torch.float16
    assert state["compression_residual"].dtype == torch.int8
    assert state["compression_residual_block_scale"].dtype == torch.float16

    module_state = copy.deepcopy(module.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    restored = make_module(layer_id=5)
    restored.load_state_dict(module_state)
    restored_optimizer = MuonMatchedGivens([restored], **kwargs)
    restored_optimizer.load_state_dict(optimizer_state)
    restored_state = restored_optimizer.state[restored.weight]
    assert restored_state["momentum_buffer"].dtype == torch.float16
    assert restored_state["compression_residual"].dtype == torch.int8
    assert restored_state["compression_residual_block_scale"].dtype == torch.float16

    gradient = torch.randn(module.weight.shape, generator=generator)
    module.weight.grad = gradient.clone()
    restored.weight.grad = gradient.clone()
    optimizer.step()
    restored_optimizer.step()
    assert torch.equal(module.weight, restored.weight)
    for key in (
        "momentum_buffer",
        "compression_residual",
        "compression_residual_block_scale",
    ):
        assert torch.equal(
            optimizer.state[module.weight][key],
            restored_optimizer.state[restored.weight][key],
        )


def test_cproj_feedback_nominal_step_cap_preserves_direction_and_resumes(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "examples.nanogpt.muon_matched_givens.zeropower_via_newtonschulz5",
        lambda matrix, steps: matrix,
    )
    uncapped = make_module(layer_id=3)
    capped = make_module(layer_id=3)
    capped.load_state_dict(copy.deepcopy(uncapped.state_dict()))
    kwargs = {
        "lr": 0.001,
        "momentum": 0.95,
        "weight_decay": 0.1,
        "ns_steps": 2,
        "error_feedback": True,
        "error_feedback_decay": 1.0,
    }
    uncapped_optimizer = MuonMatchedGivens([uncapped], **kwargs)
    capped_optimizer = MuonMatchedGivens(
        [capped],
        **kwargs,
        error_feedback_max_nominal_steps=0.25,
    )
    torch.manual_seed(47)
    gradient = torch.randn_like(uncapped.weight)
    uncapped.weight.grad = gradient.clone()
    capped.weight.grad = gradient.clone()
    uncapped_optimizer.step()
    capped_optimizer.step()

    # The cap changes only next-step state, not the update that produced it.
    assert torch.equal(uncapped.weight, capped.weight)
    raw = uncapped_optimizer.state[uncapped.weight]["compression_residual"]
    bounded = capped_optimizer.state[capped.weight]["compression_residual"]
    maximum = 0.25 * 0.001 * math.sqrt(capped.weight.shape[0])
    assert math.isclose(float(bounded.norm()), maximum, rel_tol=1e-6)
    cosine = torch.nn.functional.cosine_similarity(
        raw.flatten(), bounded.flatten(), dim=0
    )
    assert float(cosine) > 0.999999
    diagnostics = capped_optimizer.consume_diagnostics()[0]
    assert diagnostics["feedback_cap_active"] is True
    assert diagnostics["feedback_output_nominal_steps_post_cap"] <= (
        0.25 + 1e-6
    )
    assert diagnostics["feedback_cap_scale"] < 1.0
    assert diagnostics["feedback_output_fro_pre_cap"] > maximum

    module_state = copy.deepcopy(capped.state_dict())
    optimizer_state = copy.deepcopy(capped_optimizer.state_dict())
    restored = make_module(layer_id=3)
    restored.load_state_dict(module_state)
    restored_optimizer = MuonMatchedGivens(
        [restored],
        **kwargs,
        error_feedback_max_nominal_steps=0.25,
    )
    restored_optimizer.load_state_dict(optimizer_state)
    torch.manual_seed(53)
    next_gradient = torch.randn_like(capped.weight)
    capped.weight.grad = next_gradient.clone()
    restored.weight.grad = next_gradient.clone()
    capped_optimizer.step()
    restored_optimizer.step()
    assert torch.equal(capped.weight, restored.weight)
    assert torch.equal(
        capped_optimizer.state[capped.weight]["compression_residual"],
        restored_optimizer.state[restored.weight]["compression_residual"],
    )


def test_inactive_cproj_feedback_cap_is_bitwise_identical(monkeypatch) -> None:
    monkeypatch.setattr(
        "examples.nanogpt.muon_matched_givens.zeropower_via_newtonschulz5",
        lambda matrix, steps: matrix,
    )
    control = make_module(layer_id=3)
    candidate = make_module(layer_id=3)
    candidate.load_state_dict(copy.deepcopy(control.state_dict()))
    kwargs = {
        "lr": 0.001,
        "momentum": 0.95,
        "weight_decay": 0.1,
        "ns_steps": 2,
        "error_feedback": True,
        "error_feedback_decay": 1.0,
    }
    control_optimizer = MuonMatchedGivens([control], **kwargs)
    candidate_optimizer = MuonMatchedGivens(
        [candidate],
        **kwargs,
        error_feedback_max_nominal_steps=1e9,
    )
    for seed in (59, 61):
        torch.manual_seed(seed)
        gradient = torch.randn_like(control.weight)
        control.weight.grad = gradient.clone()
        candidate.weight.grad = gradient.clone()
        control_optimizer.step()
        candidate_optimizer.step()
        assert torch.equal(control.weight, candidate.weight)
        assert torch.equal(
            control_optimizer.state[control.weight]["compression_residual"],
            candidate_optimizer.state[candidate.weight]["compression_residual"],
        )
        diagnostics = candidate_optimizer.consume_diagnostics()[0]
        assert diagnostics["feedback_cap_active"] is False
        assert diagnostics["feedback_cap_scale"] == 1.0


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


def test_fast_fresh_output_pass_fits_transposed_post_hidden_residual(
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
        output_stages=1,
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
    module.weight.grad = torch.randn_like(module.weight)
    optimizer.step()
    diagnostics = optimizer.consume_diagnostics()

    assert len(calls) == 3
    parent_source, parent_direction, _, _ = calls[0]
    residual_source, residual_direction, _, _ = calls[1]
    output_source, output_direction, output_stages, output_seed = calls[2]
    requested = 0.001 * (
        parent_direction - 0.1 * parent_source.float()
    )
    parent_angles = diagonal_metric_angles(
        parent_source,
        requested,
        torch.arange(8).view(1, 8),
    )
    after_parent = apply_givens_flow(
        parent_source,
        parent_angles,
        torch.arange(8).view(1, 8),
    )
    expected_residual = requested - (after_parent - parent_source)
    residual_angles = diagonal_metric_angles(
        residual_source,
        residual_direction,
        torch.arange(8).view(1, 8),
    )
    after_residual = apply_givens_flow(
        residual_source,
        residual_angles,
        torch.arange(8).view(1, 8),
    )
    expected_output_residual = (
        requested - (after_residual - parent_source)
    )
    torch.testing.assert_close(output_source, after_residual.T.contiguous())
    torch.testing.assert_close(
        output_direction,
        expected_output_residual.T.contiguous(),
    )
    assert output_stages == 1
    assert output_seed == 25
    assert module.coordinate_count == 10
    assert module.output_selected_permutations.shape == (1, 4)
    assert module.output_selected_inverse_permutations.shape == (1, 4)
    assert module.output_last_angles.shape == (1, 2)
    assert torch.isfinite(module.output_last_angles).all()
    assert diagnostics[0]["output_stages"] == 1
    assert diagnostics[0]["output_matching"]["selector"] == (
        "fast_fresh_output_pass"
    )

    module_state = copy.deepcopy(module.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    restored = MuonMatchedGivensLinear(
        8,
        4,
        bias=False,
        stages=1,
        residual_stages=1,
        output_stages=1,
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
    assert all(
        group.get("cproj_error_feedback_decay_schedule") is True
        for group in custom[0].param_groups
    )
    assert all(
        not group.get("cproj_error_feedback_decay_schedule", False)
        for candidate in optimizer.optimizers
        if candidate is not custom[0]
        for group in candidate.param_groups
    )
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


def test_gpt_wires_activation_energy_metric_and_records_postgelu() -> None:
    torch.manual_seed(409)
    config = make_gpt_config()
    config.block_fht_mlp_cproj_activation_energy_metric = True
    model = GPT(config)
    modules = [
        module
        for module in model.modules()
        if isinstance(module, MuonMatchedGivensLinear)
    ]
    assert len(modules) == 2
    assert all(module.activation_energy_metric for module in modules)
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
    assert all(module._activation_energy_count > 0 for module in modules)
    optimizer.step()
    assert all(int(module.activation_energy_updates) == 1 for module in modules)
    assert all(module._activation_energy_sum is None for module in modules)
    assert all(
        torch.isfinite(module.activation_energy_ema).all()
        for module in modules
    )


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
        max_condition_number=None,
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
        max_condition_number=None,
    )
    raw = 0.5 * weight_recipe[0][1] + 0.5 * functional_recipe[0][1]
    projected = mixed[0][1]
    assert torch.allclose(projected / projected.norm(), raw / raw.norm())
    assert torch.allclose(projected.norm(), weight_recipe[0][1].norm())
    assert diagnostics["coordinate_norm_projection_active"] is True
    assert diagnostics["coordinate_norm_projection_scale"] < 1.0


def test_mixed_direction_respects_composed_condition_bound() -> None:
    pairs = torch.tensor([[0, 1], [2, 3]])
    weight_recipe = [
        (pairs, torch.tensor([0.002, -0.001])),
        (pairs.flip(0), torch.tensor([0.001, 0.002])),
    ]
    functional_recipe = [
        (pairs, torch.tensor([1.0, 0.5])),
        (pairs.flip(0), torch.tensor([0.75, -1.0])),
    ]
    mixed, diagnostics = mix_shear_recipes(
        weight_recipe,
        functional_recipe,
        beta=0.5,
        project_to_weight_norm=False,
        max_condition_number=1.01,
    )
    raw = torch.cat(
        [
            0.5 * weight_coordinates + 0.5 * functional_coordinates
            for (_pairs, weight_coordinates), (
                _functional_pairs,
                functional_coordinates,
            ) in zip(weight_recipe, functional_recipe, strict=True)
        ]
    )
    projected = torch.cat([coordinates for _pairs, coordinates in mixed])
    assert torch.allclose(projected / projected.norm(), raw / raw.norm())
    assert diagnostics["condition_projection_active"] is True
    assert diagnostics["mixed_log_condition_bound_after_projection"] <= (
        math.log(1.01) + 1e-12
    )


def test_functional_fit_is_bounded_inside_recursive_chart() -> None:
    torch.manual_seed(113)
    source = torch.randn(5, 8) * 0.02
    requested = torch.randn_like(source) * 2.0
    inputs = torch.randn(17, 5)
    pre_gelu = inputs @ source
    cproj = torch.randn(5, 8) * 0.02
    permutations = random_unique_matchings(width=8, stages=4, seed=127)
    diagnostics: dict[str, float | bool] = {}
    recipe = _fit_functional_shear_recipe(
        source,
        requested,
        inputs,
        pre_gelu,
        cproj,
        permutations,
        max_condition_number=1.01,
        fit_diagnostics=diagnostics,
    )
    assert all(
        torch.isfinite(coordinates).all()
        for _pairs, coordinates in recipe
    )
    assert diagnostics["functional_fit_condition_projection_active"] is True
    assert diagnostics["functional_fit_log_condition_bound"] <= (
        math.log(1.01) + 1e-12
    )


def test_nonfinite_functional_recipe_falls_back_to_weight_recipe() -> None:
    pairs = torch.tensor([[0, 1], [2, 3]])
    weight_recipe = [(pairs, torch.tensor([0.002, -0.001]))]
    functional_recipe = [(pairs, torch.tensor([float("nan"), 1.0]))]
    mixed, diagnostics = mix_shear_recipes(
        weight_recipe,
        functional_recipe,
        beta=0.5,
        project_to_weight_norm=False,
        max_condition_number=1.01,
    )
    assert torch.equal(mixed[0][1], weight_recipe[0][1])
    assert diagnostics["functional_fallback_to_weight_recipe"] is True
    assert diagnostics["weight_recipe_finite_fraction"] == 1.0
    assert diagnostics["functional_recipe_finite_fraction"] == 0.5


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
