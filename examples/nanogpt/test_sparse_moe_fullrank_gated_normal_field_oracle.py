from __future__ import annotations

import json
from pathlib import Path

import torch

from examples.nanogpt.analyze_sparse_moe_fullrank_gated_normal_field_oracle import (
    FullRankGatedNormalField,
    coordinate_count,
    fit_field,
    validate_plan,
)


def _module(seed: int = 7) -> FullRankGatedNormalField:
    return FullRankGatedNormalField(
        dense_cproj=torch.randn(2, 8, 12) * 0.1,
        input_width=8,
        hidden_width=12,
        padded_width=16,
        tensor_layers=2,
        experts=2,
        seed=seed,
        layer=0,
        device="cpu",
    )


def test_registered_coordinate_count_is_exact() -> None:
    assert coordinate_count(tensor_layers=12, experts=8, hidden_width=1536) == 442464
    assert 113246208 / 442464 > 255.94


def test_live_count_excludes_procedural_maps_and_dense_cproj() -> None:
    module = _module()
    assert module.counted_coordinates() == 2 * (3 * 12 + 1)
    assert "signs" not in module.state_dict()
    assert "dense_cproj" not in module.state_dict()


def test_same_seed_and_cproj_are_exact() -> None:
    torch.manual_seed(11)
    cproj = torch.randn(2, 8, 12)
    kwargs = dict(
        dense_cproj=cproj, input_width=8, hidden_width=12, padded_width=16,
        tensor_layers=2, experts=2, seed=13, layer=1, device="cpu",
    )
    left, right = FullRankGatedNormalField(**kwargs), FullRankGatedNormalField(**kwargs)
    torch.testing.assert_close(left.signs, right.signs)
    for key, value in left.state_dict().items():
        torch.testing.assert_close(value, right.state_dict()[key])


def test_candidate_and_control_shapes_and_gradients() -> None:
    module = _module()
    inputs = torch.randn(2, 5, 8)
    directions = torch.randn_like(inputs)
    for multiplicative in (False, True):
        output, jvp = module.function_and_jvp(
            inputs, directions, multiplicative=multiplicative
        )
        assert output.shape == jvp.shape == inputs.shape
        loss = output.square().mean() + jvp.square().mean()
        module.zero_grad(set_to_none=True)
        loss.backward()
        for parameter in module.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()


def test_analytic_jvp_matches_finite_difference() -> None:
    torch.manual_seed(17)
    module = _module()
    with torch.no_grad():
        module.gate_gain_delta.normal_(std=0.05)
        module.value_gain_delta.normal_(std=0.05)
        module.gate_bias.normal_(std=0.05)
        module.log_scale.normal_(std=0.02)
    inputs = torch.randn(2, 4, 8)
    directions = torch.randn_like(inputs)
    for multiplicative in (False, True):
        _, jvp = module.function_and_jvp(
            inputs, directions, multiplicative=multiplicative
        )
        epsilon = 1e-3
        plus, _ = module.function_and_jvp(
            inputs + epsilon * directions, torch.zeros_like(inputs),
            multiplicative=multiplicative,
        )
        minus, _ = module.function_and_jvp(
            inputs - epsilon * directions, torch.zeros_like(inputs),
            multiplicative=multiplicative,
        )
        finite = (plus - minus) / (2.0 * epsilon)
        torch.testing.assert_close(jvp, finite, rtol=8e-3, atol=2e-4)


def test_synthetic_fit_reduces_registered_objective() -> None:
    torch.manual_seed(23)
    module = _module()
    inputs = torch.randn(2, 16, 8)
    c_fc = torch.randn(2, 12, 8) * 0.1
    c_proj = module.dense_cproj.detach().clone()
    diagnostics = fit_field(
        module, inputs, c_fc, c_proj, multiplicative=True,
        steps=20, learning_rate=0.02, weight_decay=0.0,
        gradient_clip=10.0, jvp_weight=0.1, probe_seed=29,
    )
    assert diagnostics["final_loss"] < diagnostics["initial_loss"]


def test_preregistered_plan_validates() -> None:
    path = (
        Path(__file__).parent / "configs" / "selection_artifacts"
        / "124m_sparse_moe_fullrank_gated_normal_field_oracle_plan.json"
    )
    plan = json.loads(path.read_text(encoding="utf-8"))
    validate_plan(plan, path)
    assert plan["identity"]["preregistration_parent_commit"] == (
        "83ecfe8c69d79c9fd515a4a464829744732b2015"
    )
