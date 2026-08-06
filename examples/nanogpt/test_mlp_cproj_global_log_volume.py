from __future__ import annotations

import copy
import math

import pytest
import torch

from examples.nanogpt.muon_matched_givens import (
    MuonMatchedGivens,
    MuonMatchedGivensLinear,
    _fit_global_log_volume,
)


def test_global_log_volume_exact_fit_and_geometry() -> None:
    source = torch.diag(torch.tensor([3.0, 2.0, 1.0])).repeat(1, 2)
    scale = 1.005
    remaining = source * (scale - 1.0)
    updated, coordinate, diagnostics = _fit_global_log_volume(
        source,
        remaining,
        max_abs_log_coordinate=math.log(1.01),
    )
    torch.testing.assert_close(
        updated, source * scale, rtol=2e-6, atol=2e-6
    )
    assert float(coordinate) == pytest.approx(math.log(scale), rel=2e-6)
    singular_before = torch.linalg.svdvals(source)
    singular_after = torch.linalg.svdvals(updated)
    torch.testing.assert_close(
        (singular_after.log() - singular_before.log()).mean(),
        coordinate,
        rtol=2e-6,
        atol=2e-6,
    )
    assert float(torch.linalg.cond(updated)) == pytest.approx(
        float(torch.linalg.cond(source)), rel=2e-6
    )
    assert diagnostics["remaining_update_recovery"] > 0.999999
    assert diagnostics["condition_number_ratio"] == 1.0
    assert diagnostics["all_finite"] is True


def test_global_log_volume_signed_clamp_is_deterministic() -> None:
    source = torch.arange(1, 17, dtype=torch.float32).reshape(4, 4)
    bound = math.log(1.01)
    first, coordinate, diagnostics = _fit_global_log_volume(
        source,
        source,
        max_abs_log_coordinate=bound,
    )
    second, second_coordinate, _ = _fit_global_log_volume(
        source,
        source,
        max_abs_log_coordinate=bound,
    )
    assert float(coordinate) == pytest.approx(bound, rel=1e-7)
    assert diagnostics["clamp_active"] is True
    torch.testing.assert_close(first, source * 1.01, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    torch.testing.assert_close(coordinate, second_coordinate, rtol=0.0, atol=0.0)

    shrunk, negative_coordinate, _ = _fit_global_log_volume(
        source,
        source * -0.005,
        max_abs_log_coordinate=bound,
    )
    assert float(negative_coordinate) < 0.0
    assert torch.linalg.norm(shrunk) < torch.linalg.norm(source)


def make_module(seed: int = 13, *, enabled: bool = True) -> MuonMatchedGivensLinear:
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
        global_log_volume=enabled,
        global_log_volume_max_abs=math.log(1.01),
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


def test_module_accounting_state_residual_and_exact_resume() -> None:
    values1 = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 17.0
    values2 = torch.arange(16, 32, dtype=torch.float32).reshape(2, 8) / 19.0
    uninterrupted = make_module()
    optimizer = make_optimizer(uninterrupted)
    assert uninterrupted.coordinate_count == 5
    one_step(uninterrupted, optimizer, values1)
    state = optimizer.state[uninterrupted.weight]
    assert torch.isfinite(state["momentum_buffer"]).all()
    assert torch.isfinite(state["compression_residual"]).all()
    assert torch.isfinite(uninterrupted.last_global_log_volume_coordinate)
    assert optimizer.last_step_diagnostics[0]["global_log_volume"] is True
    assert optimizer.last_step_diagnostics[0]["global_log_volume_fit"][
        "all_finite"
    ] is True
    saved_model = copy.deepcopy(uninterrupted.state_dict())
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    one_step(uninterrupted, optimizer, values2)

    resumed = make_module(seed=999)
    resumed.load_state_dict(saved_model)
    resumed_optimizer = make_optimizer(resumed)
    resumed_optimizer.load_state_dict(saved_optimizer)
    one_step(resumed, resumed_optimizer, values2)
    for key, value in uninterrupted.state_dict().items():
        torch.testing.assert_close(
            value, resumed.state_dict()[key], rtol=0.0, atol=0.0
        )
    for key, value in optimizer.state[uninterrupted.weight].items():
        torch.testing.assert_close(
            value,
            resumed_optimizer.state[resumed.weight][key],
            rtol=0.0,
            atol=0.0,
        )


def test_default_off_and_invalid_combinations_fail_closed() -> None:
    disabled = make_module(enabled=False)
    assert disabled.coordinate_count == 4
    assert not hasattr(disabled, "last_global_log_volume_coordinate")
    with pytest.raises(ValueError, match="global log-volume"):
        MuonMatchedGivensLinear(
            8,
            4,
            bias=False,
            stages=1,
            residual_stages=0,
            output_stages=0,
            neighbors=2,
            refresh_interval=1,
            fast_fresh_matching=True,
            matching_seed=1,
            weight_std=0.02,
            output_symmetric_shear_stages=1,
            output_symmetric_shear_neighbors=2,
            global_log_volume=True,
        )
    with pytest.raises(ValueError, match="global log-volume"):
        MuonMatchedGivensLinear(
            8,
            4,
            bias=False,
            stages=1,
            residual_stages=0,
            output_stages=0,
            neighbors=2,
            refresh_interval=1,
            fast_fresh_matching=True,
            matching_seed=1,
            weight_std=0.02,
            global_log_volume=True,
            global_log_volume_max_abs=0.0,
        )
