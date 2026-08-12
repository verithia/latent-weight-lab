from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from examples.nanogpt.analyze_sparse_moe_signed_complete_atom_oracle import (
    SignedCompleteAtoms,
    fit_atoms,
    result_authorization,
    validate_plan,
)


def _small(seed: int = 7) -> SignedCompleteAtoms:
    return SignedCompleteAtoms(
        experts=2,
        atoms=2,
        input_width=8,
        hidden_width=12,
        padded_width=16,
        router_feature_width=3,
        tensor_layers=2,
        seed=seed,
        layer=0,
        device="cpu",
    )


def test_registered_accounting_exceeds_200x() -> None:
    module = SignedCompleteAtoms(
        experts=8,
        atoms=4,
        input_width=768,
        hidden_width=1536,
        padded_width=2048,
        router_feature_width=8,
        tensor_layers=12,
        seed=11,
        layer=0,
        device="cpu",
    )
    assert module.compact_parameter_count(conditional=True) == 86304
    assert module.compact_parameter_count(conditional=False) == 86048
    assert (8 * 2 * 1536 * 768) / 86304 > 218.69


def test_signed_output_and_router_gradient_are_finite() -> None:
    module = _small()
    inputs = torch.randn(2, 5, 8)
    directions = torch.randn_like(inputs)
    output, jvp = module.function_and_jvp(
        inputs, directions, conditional=True
    )
    assert output.shape == inputs.shape
    assert jvp.shape == inputs.shape
    (output.square().mean() + jvp.square().mean()).backward()
    assert torch.isfinite(module.router_weight.grad).all()
    assert float(module.router_weight.grad.abs().sum()) > 0.0
    assert torch.isfinite(module.hidden_gain_delta.grad).all()
    assert torch.isfinite(module.output_gain_delta.grad).all()


def test_signed_analytic_jvp_matches_finite_difference() -> None:
    torch.manual_seed(17)
    module = _small()
    with torch.no_grad():
        module.router_weight.normal_(std=0.2)
        module.router_bias.normal_(std=0.1)
        module.hidden_gain_delta.normal_(std=0.05)
        module.output_gain_delta.normal_(std=0.05)
    inputs = torch.randn(2, 4, 8)
    directions = torch.randn_like(inputs)
    output, jvp = module.function_and_jvp(
        inputs, directions, conditional=True
    )
    epsilon = 1e-3
    plus, _ = module.function_and_jvp(
        inputs + epsilon * directions,
        torch.zeros_like(inputs),
        conditional=True,
    )
    minus, _ = module.function_and_jvp(
        inputs - epsilon * directions,
        torch.zeros_like(inputs),
        conditional=True,
    )
    finite = (plus - minus) / (2.0 * epsilon)
    torch.testing.assert_close(jvp, finite, rtol=3e-3, atol=3e-4)
    assert torch.isfinite(output).all()


def test_static_control_ignores_signed_router_weight() -> None:
    module = _small()
    inputs = torch.randn(2, 4, 8)
    directions = torch.randn_like(inputs)
    left = module.function_and_jvp(inputs, directions, conditional=False)
    with torch.no_grad():
        module.router_weight.normal_(std=10.0)
    right = module.function_and_jvp(inputs, directions, conditional=False)
    torch.testing.assert_close(left[0], right[0])
    torch.testing.assert_close(left[1], right[1])


def test_zero_signed_router_matches_static_control() -> None:
    module = _small()
    inputs = torch.randn(2, 4, 8)
    directions = torch.randn_like(inputs)
    conditional = module.function_and_jvp(
        inputs, directions, conditional=True
    )
    static = module.function_and_jvp(inputs, directions, conditional=False)
    torch.testing.assert_close(conditional[0], static[0])
    torch.testing.assert_close(conditional[1], static[1])


def test_same_seed_is_exact() -> None:
    left = _small(seed=29)
    right = _small(seed=29)
    for key, value in left.state_dict().items():
        torch.testing.assert_close(value, right.state_dict()[key])


def test_synthetic_fit_reduces_registered_objective() -> None:
    torch.manual_seed(31)
    module = _small()
    inputs = torch.randn(2, 16, 8)
    c_fc = torch.randn(2, 12, 8) * 0.1
    c_proj = torch.randn(2, 8, 12) * 0.1
    diagnostics = fit_atoms(
        module, inputs, c_fc, c_proj,
        conditional=True, steps=20, learning_rate=0.02,
        weight_decay=0.0, gradient_clip=10.0, jvp_weight=0.1,
        probe_seed=37,
    )
    assert diagnostics["final_loss"] < diagnostics["initial_loss"]
    assert math.isfinite(diagnostics["maximum_preclip_gradient_norm"])
    assert diagnostics["coefficient_abs_max"] > 0.0


def test_authorization_stops_before_training() -> None:
    passed = result_authorization(True)
    failed = result_authorization(False)
    assert passed["implementation"]
    assert passed["initialization_and_mapping_loss_shadow"]
    assert not passed["language_model_training"]
    assert not passed["mfu_preflight"]
    assert not failed["implementation"]


def test_preregistered_plan_is_hash_sealed() -> None:
    plan_path = (
        Path(__file__).parent / "configs" / "selection_artifacts"
        / "124m_sparse_moe_signed_complete_atom_oracle_plan.json"
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_plan(plan, plan_path)
    assert plan["identity"]["preregistration_git_commit"] == (
        "de750b008e0a872c435b69ea97273a8f54cb2dbf"
    )
    assert plan["candidate"]["paired_parameter_compression_ratio"] > 200.0
