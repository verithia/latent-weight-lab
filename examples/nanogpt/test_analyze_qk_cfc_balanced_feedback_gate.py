import torch

from examples.nanogpt.analyze_qk_cfc_balanced_feedback_gate import (
    balanced_target,
    classify,
)


def test_layer_balancing_equalizes_component_norms() -> None:
    requested = {0: torch.tensor([3.0, 4.0]), 1: torch.tensor([0.0, 2.0])}
    feedback = {0: torch.tensor([12.0, 5.0]), 1: torch.tensor([6.0, 8.0])}
    target, diagnostics = balanced_target(
        requested, feedback, gamma=1.0, layerwise=True
    )
    assert diagnostics["layerwise"] is True
    for layer in requested:
        added = target[layer] - requested[layer]
        assert torch.allclose(added.norm(), requested[layer].norm())


def test_global_balancing_uses_one_scale() -> None:
    requested = {0: torch.tensor([1.0]), 1: torch.tensor([2.0])}
    feedback = {0: torch.tensor([10.0]), 1: torch.tensor([20.0])}
    _target, diagnostics = balanced_target(
        requested, feedback, gamma=0.5, layerwise=False
    )
    assert len(set(diagnostics["feedback_scales"].values())) == 1


def test_classification_requires_all_frozen_gates() -> None:
    def row(recovery: float, residual: float):
        return {
            "late_action_vs_requested": {"positive_line_recovery": recovery},
            "outgoing_to_incoming_feedback_fro_ratio": residual,
        }

    candidates = (
        "layer_equal_balance",
        "layer_half_balance",
        "global_equal_balance",
    )
    rows = {
        window: {
            "current_corrected": row(0.01, 1.0),
            **{name: row(0.20, 0.99) for name in candidates},
        }
        for window in ("fit", "holdout")
    }
    rows["cross_window"] = {
        name: {"late_action_cosine": 0.30} for name in candidates
    }
    comparison = {
        "candidate_minus_current_mean_ce": -0.001,
        "upper_confidence_bound": -0.0001,
    }
    functional = {
        window: {name: {"vs_current": comparison} for name in candidates}
        for window in ("fit", "holdout")
    }
    rule = {
        "minimum_late_requested_recovery_improvement": 0.10,
        "minimum_late_action_cross_window_cosine": 0.10,
        "maximum_outgoing_to_incoming_feedback_ratio": 1.0,
        "maximum_functional_upper_bound_regression_ce": 0.0,
        "minimum_one_window_mean_ce_improvement": 0.0001,
        "candidate_priority": list(candidates),
    }
    assert classify(rows, functional, rule)["selected_candidate"] == candidates[0]
