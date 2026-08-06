from __future__ import annotations

import copy

import pytest
import torch

from examples.nanogpt import fast_task_matching
from examples.nanogpt.muon_matched_givens import (
    MuonMatchedGivens,
    MuonMatchedGivensLinear,
    _apply_symmetric_shear_stage,
    _fit_weight_shear_flow,
)


def test_selector_uses_symmetric_tangent_and_returns_perfect_matchings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[torch.Tensor] = []

    def fake_color(
        edges: torch.Tensor,
        *,
        width: int,
        stages: int,
        seed: int,
        cache_dir=None,
    ):
        del seed, cache_dir
        captured.append(edges.clone())
        assert width == 4
        assert stages == 1
        return torch.tensor([[0, 1, 2, 3]]), {
            "candidate_edge_fraction": 1.0,
            "minimum_stage_candidate_edge_fraction": 1.0,
            "native_seconds": 0.0,
            "native_output_validated": True,
            "native_library_sha256": "test",
            "source_sha256": "test",
        }

    monkeypatch.setattr(fast_task_matching, "color_sorted_edges", fake_color)
    source = torch.eye(4)
    symmetric = torch.tensor(
        [[0.0, 3.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 2.0], [0.0, 0.0, 2.0, 0.0]]
    )
    first, diagnostics = fast_task_matching.fast_symmetric_shear_permutations(
        source, symmetric, stages=1, neighbors=2, seed=7
    )
    second, _ = fast_task_matching.fast_symmetric_shear_permutations(
        source, symmetric, stages=1, neighbors=2, seed=7
    )
    assert torch.equal(first, second)
    assert sorted(first[0].tolist()) == [0, 1, 2, 3]
    assert len(set(first[0].tolist())) == 4
    assert captured[0][0].tolist() == [0, 1]
    assert diagnostics["score_family"] == (
        "symmetric_shear_normalized_squared_tangent"
    )


def test_symmetric_flow_recovers_symmetric_but_rejects_skew_direction() -> None:
    source = torch.eye(4)
    permutations = torch.tensor([[0, 1, 2, 3]])
    pairs = permutations.reshape(-1, 2)
    target = _apply_symmetric_shear_stage(
        source, pairs, torch.tensor([0.02, -0.015])
    )
    updated, recipe, diagnostics = _fit_weight_shear_flow(
        source,
        target - source,
        permutations,
        max_condition_number=1.1,
    )
    assert len(recipe) == 1
    assert diagnostics["requested_update_recovery"] > 0.999
    assert diagnostics["condition_number_upper_bound"] <= 1.1 + 1e-7
    assert diagnostics["all_coordinates_finite"] is True
    assert torch.isfinite(updated).all()

    skew = torch.tensor(
        [[0.0, 0.02, 0.0, 0.0], [-0.02, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, -0.01], [0.0, 0.0, 0.01, 0.0]]
    )
    _updated, skew_recipe, _ = _fit_weight_shear_flow(
        source, skew, permutations, max_condition_number=1.1
    )
    torch.testing.assert_close(
        skew_recipe[0][1], torch.zeros(2, dtype=torch.float64),
        rtol=0.0, atol=0.0,
    )


def make_module(seed: int = 13) -> MuonMatchedGivensLinear:
    torch.manual_seed(seed)
    module = MuonMatchedGivensLinear(
        8,
        4,
        bias=False,
        stages=1,
        residual_stages=0,
        output_stages=0,
        neighbors=2,
        refresh_interval=100,
        fast_fresh_matching=True,
        matching_seed=19,
        weight_std=0.02,
        layer_id=0,
        output_symmetric_shear_stages=2,
        output_symmetric_shear_neighbors=2,
        output_symmetric_shear_max_condition_number=1.1,
    )
    module.matching_valid.fill_(True)
    return module


def make_optimizer(module: MuonMatchedGivensLinear) -> MuonMatchedGivens:
    return MuonMatchedGivens(
        [module],
        lr=1e-3,
        momentum=0.9,
        weight_decay=0.1,
        ns_steps=2,
        error_feedback=True,
        error_feedback_decay=0.5,
    )


def one_step(
    module: MuonMatchedGivensLinear,
    optimizer: MuonMatchedGivens,
    values: torch.Tensor,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    module(values).float().square().mean().backward()
    optimizer.step()


def test_module_accounting_state_and_exact_resume() -> None:
    first_values = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 17.0
    second_values = torch.arange(16, 32, dtype=torch.float32).reshape(2, 8) / 19.0
    uninterrupted = make_module()
    optimizer = make_optimizer(uninterrupted)
    assert uninterrupted.coordinate_count == 8
    one_step(uninterrupted, optimizer, first_values)
    assert torch.isfinite(optimizer.state[uninterrupted.weight]["momentum_buffer"]).all()
    assert torch.isfinite(
        optimizer.state[uninterrupted.weight]["compression_residual"]
    ).all()
    assert torch.isfinite(uninterrupted.output_shear_last_coordinates).all()
    saved_model = copy.deepcopy(uninterrupted.state_dict())
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    one_step(uninterrupted, optimizer, second_values)

    resumed = make_module(seed=999)
    resumed.load_state_dict(saved_model)
    resumed_optimizer = make_optimizer(resumed)
    resumed_optimizer.load_state_dict(saved_optimizer)
    one_step(resumed, resumed_optimizer, second_values)
    for key, value in uninterrupted.state_dict().items():
        torch.testing.assert_close(
            value, resumed.state_dict()[key], rtol=0.0, atol=0.0
        )
    original_state = optimizer.state[uninterrupted.weight]
    resumed_state = resumed_optimizer.state[resumed.weight]
    assert set(original_state) == set(resumed_state)
    for key, value in original_state.items():
        torch.testing.assert_close(
            value, resumed_state[key], rtol=0.0, atol=0.0
        )


def test_invalid_symmetric_output_combinations_fail_closed() -> None:
    with pytest.raises(ValueError, match="invalid output symmetric-shear"):
        MuonMatchedGivensLinear(
            8, 4, bias=False, stages=1, residual_stages=0,
            output_stages=1, neighbors=2, refresh_interval=1,
            fast_fresh_matching=True, matching_seed=1, weight_std=0.02,
            output_symmetric_shear_stages=1,
            output_symmetric_shear_neighbors=2,
        )
    with pytest.raises(ValueError, match="invalid output symmetric-shear"):
        MuonMatchedGivensLinear(
            8, 4, bias=False, stages=1, residual_stages=0,
            output_stages=0, neighbors=2, refresh_interval=1,
            fast_fresh_matching=True, matching_seed=1, weight_std=0.02,
            activation_energy_metric=True,
            output_symmetric_shear_stages=1,
            output_symmetric_shear_neighbors=2,
        )
