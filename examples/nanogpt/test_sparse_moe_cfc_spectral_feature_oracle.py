from __future__ import annotations

import torch

from examples.nanogpt.analyze_sparse_moe_cfc_spectral_feature_oracle import (
    CompactCFCState,
    SpectralCFC,
    action_cosine,
    fit_compact_state,
    normalized_fit_loss,
)


def _operator() -> SpectralCFC:
    return SpectralCFC(
        experts=2,
        input_width=4,
        hidden_width=8,
        padded_width=8,
        seed=101,
        layer=0,
        device="cpu",
    )


def test_coordinate_count_matches_two_spectra_plus_bias() -> None:
    operator = _operator()
    assert operator.coordinates_per_expert == 24


def test_fixed_operator_is_deterministic_and_finite() -> None:
    torch.manual_seed(7)
    inputs = torch.randn(2, 12, 4)
    left = _operator().fixed_features(inputs)
    right = _operator().fixed_features(inputs)
    torch.testing.assert_close(left, right, atol=0, rtol=0)
    assert torch.isfinite(left).all()


def test_zero_spectra_match_fixed_path() -> None:
    torch.manual_seed(9)
    operator = _operator()
    inputs = torch.randn(2, 10, 4)
    zeros = torch.zeros(2, 8)
    state = CompactCFCState(zeros, zeros, torch.zeros(2, 8))
    spectral = operator.preactivation(inputs, state, spectral=True)
    fixed = operator.preactivation(inputs, state, spectral=False)
    torch.testing.assert_close(spectral, fixed, atol=0, rtol=0)


def test_fit_reduces_a_representable_synthetic_objective() -> None:
    torch.manual_seed(13)
    operator = _operator()
    inputs = torch.randn(2, 48, 4)
    truth = CompactCFCState(
        0.15 * torch.randn(2, 8),
        0.15 * torch.randn(2, 8),
        0.02 * torch.randn(2, 8),
    )
    c_proj = torch.randn(2, 4, 8) / 4
    target_pre, target_output = operator.expert_output(
        inputs, c_proj, truth, spectral=True
    )
    zeros = torch.zeros(2, 8)
    initial = CompactCFCState(zeros, zeros, torch.zeros(2, 8))
    initial_pre, initial_output = operator.expert_output(
        inputs, c_proj, initial, spectral=True
    )
    initial_loss = normalized_fit_loss(
        initial_pre, initial_output, target_pre, target_output
    )
    # Use a dense c_fc target synthesized from the exact target preactivation
    # least-squares fit; the test checks optimization behavior, not an exact
    # architectural identity.
    c_fc = torch.linalg.lstsq(
        inputs.reshape(-1, 4), target_pre.reshape(-1, 8)
    ).solution.T.reshape(1, 8, 4).repeat(2, 1, 1)
    fitted, diagnostics = fit_compact_state(
        operator,
        inputs,
        c_fc,
        c_proj,
        spectral=True,
        steps=30,
        learning_rate=0.03,
        weight_decay=0.0,
    )
    assert diagnostics["final_loss"] < diagnostics["initial_loss"]
    predicted_pre, predicted_output = operator.expert_output(
        inputs, c_proj, fitted, spectral=True
    )
    dense_pre = torch.bmm(inputs, c_fc.transpose(1, 2))
    dense_output = torch.bmm(torch.nn.functional.gelu(dense_pre), c_proj.transpose(1, 2))
    assert normalized_fit_loss(
        predicted_pre, predicted_output, dense_pre, dense_output
    ) < initial_loss + 1.0


def test_action_cosine_identity() -> None:
    values = torch.randn(4, 7)
    assert action_cosine(values, values) > 0.999999
