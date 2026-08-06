from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_cproj_optimizer_state_transport import (
    classify,
    functional_metric,
    metric,
    output_additive_projection,
    reconstruct_components,
)


def test_metric_reports_exact_and_orthogonal_recovery() -> None:
    target = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    exact = metric(target, target)
    assert exact["positive_line_recovery"] == 1.0
    assert exact["fixed_scale_recovery"] == 1.0
    orthogonal = metric(target, torch.tensor([[0.0, 1.0], [0.0, 0.0]]))
    assert orthogonal["positive_line_recovery"] == 0.0


def test_functional_metric_ignores_input_nullspace() -> None:
    hidden = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    target = torch.tensor([[1.0, 3.0]])
    candidate = torch.tensor([[1.0, -7.0]])
    result = functional_metric(target, candidate, hidden)
    assert result["fixed_scale_recovery"] == 1.0


def test_output_additive_projection_recovers_output_additive_component() -> None:
    torch.manual_seed(2)
    hidden = torch.randn(64, 5)
    component = torch.randn(3, 1).expand(3, 5).clone()
    projected = output_additive_projection(component, hidden)
    result = functional_metric(component, projected, hidden)
    assert result["fixed_scale_recovery"] > 0.99999


def test_reconstruct_components_matches_uncapped_feedback_identity() -> None:
    torch.manual_seed(3)
    weight = torch.randn(4, 3)
    combined = torch.randn(4, 3)
    before_feedback = torch.randn(4, 3) * 0.01
    hyper = {
        "ns_steps": 3,
        "polar_scale": 1.2,
        "lr": 0.02,
        "weight_decay": 0.1,
        "error_feedback_decay": 0.5,
    }
    polar = __import__(
        "examples.nanogpt.muon", fromlist=["zeropower_via_newtonschulz5"]
    ).zeropower_via_newtonschulz5(combined, steps=3)
    requested = 0.02 * (-1.2 * polar - 0.1 * weight)
    corrected = requested + 0.5 * before_feedback
    realized = corrected * 0.75
    after = weight + realized
    residual = corrected - realized
    state = {
        "weight_before_step": weight,
        "combined_momentum_update": combined,
        "compression_residual_before_step": before_feedback,
        "weight_after_step": after,
        "compression_residual_after_step": residual,
    }
    components, error = reconstruct_components(state, hyper)
    torch.testing.assert_close(components["requested"], requested)
    torch.testing.assert_close(components["corrected"], corrected)
    torch.testing.assert_close(components["realized"], realized)
    assert error < 1e-6


def test_classification_requires_heldout_functional_threshold() -> None:
    aggregate = {
        name: {"heldout_terminal_functional_positive_line_recovery": value}
        for name, value in {
            "requested": 0.2,
            "feedback": 0.81,
            "corrected": 0.6,
            "realized": 0.7,
            "unrepresented": 0.4,
        }.items()
    }
    assert classify(aggregate, True, 0.8) == (
        "CAUSAL_OPTIMIZER_STATE_TRANSPORT_SUFFICIENT", "feedback"
    )
    assert classify(aggregate, False, 0.8)[0] == (
        "INVALID_OPTIMIZER_STATE_RECONSTRUCTION"
    )
