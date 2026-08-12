from __future__ import annotations

import torch

from examples.nanogpt.analyze_sparse_moe_residual_coordinate_modulation_oracle import (
    ResidualCoordinateModulatedAtoms,
    coordinate_count,
    fit_atoms,
)


def _small() -> ResidualCoordinateModulatedAtoms:
    return ResidualCoordinateModulatedAtoms(
        experts=2, atoms=3, input_width=8, hidden_width=16,
        padded_width=16, tensor_layers=2, seed=17, layer=0, device="cpu",
    )


def test_coordinate_accounting_matches_registered_budget() -> None:
    candidate = coordinate_count(
        experts=8, atoms=3, input_width=768, hidden_width=1536, conditional=True
    )
    control = coordinate_count(
        experts=8, atoms=3, input_width=768, hidden_width=1536, conditional=False
    )
    assert candidate == 86016
    assert control == 67584
    assert (8 * 2 * 1536 * 768) / candidate > 219.42


def test_conditional_zero_modulation_equals_static_control() -> None:
    module = _small()
    x = torch.randn(2, 5, 8)
    v = torch.randn_like(x)
    conditional = module.function_and_jvp(x, v, conditional=True)
    control = module.function_and_jvp(x, v, conditional=False)
    torch.testing.assert_close(conditional[0], control[0])
    torch.testing.assert_close(conditional[1], control[1])


def test_output_and_all_candidate_gradients_are_finite() -> None:
    module = _small()
    x = torch.randn(2, 5, 8)
    v = torch.randn_like(x)
    output, jvp = module.function_and_jvp(x, v, conditional=True)
    (output.square().mean() + jvp.square().mean()).backward()
    assert torch.isfinite(output).all()
    assert torch.isfinite(jvp).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.trainable_parameters(conditional=True))


def test_analytic_jvp_matches_centered_finite_difference() -> None:
    module = _small()
    with torch.no_grad():
        module.residual_modulation.normal_(0.0, 0.2)
        module.output_gain_delta.normal_(0.0, 0.1)
    x = torch.randn(2, 4, 8)
    v = torch.randn_like(x)
    output, jvp = module.function_and_jvp(x, v, conditional=True)
    eps = 1e-3
    plus = module.function_and_jvp(x + eps * v, v, conditional=True)[0]
    minus = module.function_and_jvp(x - eps * v, v, conditional=True)[0]
    finite = (plus - minus) / (2.0 * eps)
    assert output.shape == x.shape
    torch.testing.assert_close(jvp, finite, rtol=2e-2, atol=2e-3)


def test_fit_descends_and_updates_conditional_modulation() -> None:
    torch.manual_seed(3)
    module = _small()
    inputs = torch.randn(2, 8, 8)
    c_fc = torch.randn(2, 16, 8) * 0.02
    c_proj = torch.randn(2, 8, 16) * 0.01
    diagnostics = fit_atoms(
        module, inputs, c_fc, c_proj, conditional=True, steps=5,
        learning_rate=0.03, weight_decay=1e-4, gradient_clip=10.0,
        jvp_weight=0.1, probe_seed=19,
    )
    assert diagnostics["final_loss"] < diagnostics["initial_loss"]
    assert diagnostics["residual_modulation_rms"] > 0.0
