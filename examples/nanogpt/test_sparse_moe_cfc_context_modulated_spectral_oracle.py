from __future__ import annotations

import torch

from examples.nanogpt.analyze_sparse_moe_cfc_context_modulated_spectral_oracle import (
    fit_is_healthy,
    result_authorization,
)
from examples.nanogpt.analyze_sparse_moe_cfc_spectral_feature_oracle import (
    CompactCFCState,
    SpectralCFC,
)


def _operator(beta: float) -> SpectralCFC:
    return SpectralCFC(
        experts=2,
        input_width=4,
        hidden_width=8,
        padded_width=8,
        seed=101,
        layer=0,
        device="cpu",
        context_beta=beta,
        context_seed_offset=37,
    )


def test_beta_zero_preserves_exact_static_parent() -> None:
    torch.manual_seed(91)
    inputs = torch.randn(2, 11, 4)
    state = CompactCFCState(
        0.1 * torch.randn(2, 8),
        0.1 * torch.randn(2, 8),
        0.01 * torch.randn(2, 8),
    )
    legacy = SpectralCFC(
        experts=2,
        input_width=4,
        hidden_width=8,
        padded_width=8,
        seed=101,
        layer=0,
        device="cpu",
    )
    control = _operator(0.0)
    torch.testing.assert_close(
        legacy.preactivation(inputs, state, spectral=True),
        control.preactivation(inputs, state, spectral=True),
        atol=0,
        rtol=0,
    )


def test_context_gate_changes_image_without_adding_coordinates() -> None:
    torch.manual_seed(93)
    inputs = torch.randn(2, 13, 4)
    state = CompactCFCState(
        0.2 * torch.randn(2, 8),
        0.2 * torch.randn(2, 8),
        torch.zeros(2, 8),
    )
    dynamic = _operator(1.0)
    static = _operator(0.0)
    assert dynamic.coordinates_per_expert == static.coordinates_per_expert == 24
    dynamic_values = dynamic.preactivation(inputs, state, spectral=True)
    static_values = static.preactivation(inputs, state, spectral=True)
    assert torch.isfinite(dynamic_values).all()
    assert not torch.equal(dynamic_values, static_values)


def test_context_gate_has_unit_expected_rms_scale() -> None:
    torch.manual_seed(95)
    inputs = torch.randn(2, 256, 4)
    gate = _operator(1.0).context_gate(inputs)
    assert torch.isfinite(gate).all()
    assert abs(float(gate.square().mean()) - 1.0) < 0.1


def test_expert_view_preserves_context_state() -> None:
    operator = _operator(1.0)
    expert = operator.for_expert(1)
    assert expert.context_beta == 1.0
    torch.testing.assert_close(expert.signs, operator.signs[1:2])
    torch.testing.assert_close(expert.context_signs, operator.context_signs[1:2])


def test_fit_health_requires_finite_decrease() -> None:
    assert fit_is_healthy({"initial_loss": 1.0, "final_loss": 0.8, "minimum_loss": 0.7})
    assert not fit_is_healthy({"initial_loss": 1.0, "final_loss": 1.0, "minimum_loss": 0.7})


def test_passing_oracle_does_not_authorize_training() -> None:
    passed = result_authorization(True)
    assert passed["production_implementation"]
    assert passed["initialization_fit_shadow"]
    assert not passed["mfu_preflight"]
    assert not passed["language_model_training"]
