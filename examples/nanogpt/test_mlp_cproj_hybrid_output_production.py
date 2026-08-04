from __future__ import annotations

import copy

import pytest
import torch

from examples.nanogpt.analyze_mlp_cproj_global_directed_affine_output import (
    fit_global_directed_map,
)
from examples.nanogpt.analyze_mlp_cproj_global_directed_minimax_output import (
    minimax_support_score,
)
from examples.nanogpt.analyze_mlp_cproj_task_gradient_output_selector import (
    fit_task_gradient_hybrid_pass,
)
from examples.nanogpt.muon_matched_givens import (
    MuonMatchedGivens,
    MuonMatchedGivensLinear,
    hybrid_task_directed_output_update,
    minimax_directed_output_pass,
    task_gradient_output_pass,
)
from examples.nanogpt.model import GPTConfig, MLP


def tensors(seed: int = 17):
    generator = torch.Generator().manual_seed(seed)
    source = torch.eye(4) + 0.02 * torch.randn(4, 4, generator=generator)
    residual = 0.01 * torch.randn(4, 4, generator=generator)
    activation = torch.randn(6, 4, generator=generator)
    current = torch.randn(4, 4, generator=generator)
    momentum = torch.randn(4, 4, generator=generator)
    return source, residual, activation, current, momentum


def test_task_output_pass_matches_registered_analysis() -> None:
    source, residual, _activation, current, _momentum = tensors()
    expected, diagnostics = fit_task_gradient_hybrid_pass(
        source, residual, current.T, stages=1, neighbors=2, seed=23
    )
    actual, permutations, angles, production = task_gradient_output_pass(
        source, residual, current.T, stages=1, neighbors=2, seed=23
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(permutations.cpu(), diagnostics["permutations"])
    torch.testing.assert_close(angles.cpu(), diagnostics["angles"])
    assert production["coordinates"] == diagnostics["coordinates"]


def test_directed_output_pass_matches_registered_analysis() -> None:
    source, residual, activation, current, momentum = tensors()
    score, _ = minimax_support_score(
        source, residual, activation, current, momentum
    )
    expected, _supports, diagnostics = fit_global_directed_map(
        source, residual, activation, score,
        incoming=1, trust_output_energy=0.05,
    )
    actual, production = minimax_directed_output_pass(
        source, residual, activation, current, momentum,
        incoming=1, ridge_ratio=1e-6, trust_output_energy=0.05,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
    assert production["coordinates"] == diagnostics["coordinates"]
    assert production["trust_energy_obeyed"] is True


def test_hybrid_update_is_finite_and_obeys_control_energy() -> None:
    source, residual, activation, current, momentum = tensors()
    updated, permutations, angles, diagnostics = (
        hybrid_task_directed_output_update(
            source, residual, activation, current, momentum,
            task_stages=1, directed_incoming=1, control_stages=2,
            neighbors=2, ridge_ratio=1e-6, seed=29,
        )
    )
    assert torch.isfinite(updated).all()
    assert permutations.shape == (1, 4)
    assert angles.shape == (1, 2)
    assert diagnostics["coordinates"] == 6
    assert diagnostics["combined_trust_energy_obeyed"] is True


def make_module(seed: int = 31) -> MuonMatchedGivensLinear:
    torch.manual_seed(seed)
    return MuonMatchedGivensLinear(
        8, 4, bias=False, stages=1, residual_stages=1,
        output_stages=1, neighbors=2, refresh_interval=1,
        fast_fresh_matching=True, matching_seed=37, weight_std=0.02,
        layer_id=0, hybrid_output=True, hybrid_directed_incoming=1,
        hybrid_control_output_stages=2, hybrid_ridge_ratio=1e-6,
        hybrid_functional_sample_cap=3,
    )


def one_step(
    module: MuonMatchedGivensLinear,
    optimizer: MuonMatchedGivens,
    values: torch.Tensor,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    module.record_hybrid_output_context(values)
    loss = module(values).float().square().mean()
    loss.backward()
    optimizer.step()


def make_optimizer(module: MuonMatchedGivensLinear) -> MuonMatchedGivens:
    return MuonMatchedGivens(
        [module], lr=1e-3, momentum=0.9, weight_decay=0.1,
        ns_steps=2, error_feedback=True, error_feedback_decay=1.0,
    )


def test_context_is_bounded_transient_and_consumed() -> None:
    module = make_module()
    values = torch.randn(5, 8)
    module.train()
    module.record_hybrid_output_context(values)
    assert module._hybrid_output_inputs is not None
    assert module._hybrid_output_inputs.shape == (3, 8)
    assert all("hybrid_output_inputs" not in key for key in module.state_dict())
    optimizer = make_optimizer(module)
    module(values).float().square().mean().backward()
    optimizer.step()
    assert module._hybrid_output_inputs is None
    state = optimizer.state[module.weight]
    assert torch.isfinite(state["momentum_buffer"]).all()
    assert torch.isfinite(state["compression_residual"]).all()


def test_exact_resume_matches_uninterrupted_two_steps() -> None:
    first_values = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 17.0
    second_values = torch.arange(16, 32, dtype=torch.float32).reshape(2, 8) / 19.0
    uninterrupted = make_module()
    uninterrupted_optimizer = make_optimizer(uninterrupted)
    one_step(uninterrupted, uninterrupted_optimizer, first_values)
    saved_model = copy.deepcopy(uninterrupted.state_dict())
    saved_optimizer = copy.deepcopy(uninterrupted_optimizer.state_dict())
    one_step(uninterrupted, uninterrupted_optimizer, second_values)

    resumed = make_module(seed=999)
    resumed.load_state_dict(saved_model)
    resumed_optimizer = make_optimizer(resumed)
    resumed_optimizer.load_state_dict(saved_optimizer)
    one_step(resumed, resumed_optimizer, second_values)

    for key, value in uninterrupted.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0.0, atol=0.0)
    uninterrupted_state = uninterrupted_optimizer.state[uninterrupted.weight]
    resumed_state = resumed_optimizer.state[resumed.weight]
    assert set(uninterrupted_state) == set(resumed_state)
    for key, value in uninterrupted_state.items():
        torch.testing.assert_close(value, resumed_state[key], rtol=0.0, atol=0.0)


def test_hybrid_constructor_fails_closed() -> None:
    with pytest.raises(ValueError):
        MuonMatchedGivensLinear(
            8, 4, bias=False, stages=1, residual_stages=1,
            output_stages=1, neighbors=2, refresh_interval=1,
            fast_fresh_matching=True, matching_seed=1, weight_std=0.02,
            hybrid_output=True, hybrid_directed_incoming=1,
            hybrid_control_output_stages=3,
        )


def test_mlp_hybrid_requires_muon_matched_cproj() -> None:
    config = GPTConfig(block_fht_mlp_cproj_hybrid_output=True)
    with pytest.raises(
        ValueError, match="requires Muon-matched Givens"
    ):
        MLP(config, layer_id=0)
