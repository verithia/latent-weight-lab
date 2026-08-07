import torch

from examples.nanogpt.analyze_qk_cfc_selector_fitter_factorial import (
    classify,
    fit_with_separate_selection_target,
    support_overlap,
)
from examples.nanogpt.muon_matched_givens import MuonDirectedProductLinear


def _module() -> MuonDirectedProductLinear:
    return MuonDirectedProductLinear(
        4,
        4,
        bias=False,
        incoming_schedule=(1, 1),
        ridge_ratio=1e-4,
        chunk_size=4,
        family_radius_ratio=1.0,
        error_feedback=True,
        error_feedback_decay=1.0,
        weight_std=0.02,
        layer_id=0,
    )


def test_same_selection_and_fit_matches_production_solver() -> None:
    torch.manual_seed(7)
    module = _module()
    target = {0: torch.randn_like(module.weight)}
    observed, supports = fit_with_separate_selection_target(
        [module], target, target, schedule=[1, 1], ridge_ratio=1e-4, chunk_size=4
    )
    from examples.nanogpt.analyze_mlp_cfc_directed_product_terminal_capacity import (
        fit_schedule,
    )

    expected, _ = fit_schedule(
        [module], target, schedule=[1, 1], ridge_ratio=1e-4, chunk_size=4
    )
    assert torch.equal(observed[0], expected[0])
    assert len(supports) == 2


def test_support_overlap_is_set_based() -> None:
    first = [torch.tensor([[[0, 1], [2, 3]]], dtype=torch.int32)]
    second = [torch.tensor([[[2, 3], [0, 1]]], dtype=torch.int32)]
    assert support_overlap(first, second, [0])["aggregate"] == 1.0


def test_classification_promotes_only_passing_factorial_cell() -> None:
    def row(recovery: float, residual: float):
        return {
            "late_action_vs_requested": {"positive_line_recovery": recovery},
            "outgoing_to_incoming_feedback_fro_ratio": residual,
        }

    rows = {
        "fit": {
            "current_corrected": row(0.10, 0.80),
            "task_select_corrected_fit_corrected_radius": row(0.20, 0.82),
            "task_select_corrected_fit_requested_radius": row(0.18, 1.20),
        },
        "holdout": {
            "current_corrected": row(0.10, 0.80),
            "task_select_corrected_fit_corrected_radius": row(0.20, 0.82),
            "task_select_corrected_fit_requested_radius": row(0.18, 1.20),
        },
        "cross_window": {
            "task_select_corrected_fit_corrected_radius": {
                "late_action_cosine": 0.30
            },
            "task_select_corrected_fit_requested_radius": {
                "late_action_cosine": 0.30
            },
        },
    }
    comparison = {
        "candidate_minus_current_mean_ce": -0.001,
        "upper_confidence_bound": -0.0001,
    }
    functional = {
        window: {
            "task_only": {"vs_current": comparison},
            "task_select_corrected_fit_corrected_radius": {
                "vs_current": comparison
            },
            "task_select_corrected_fit_requested_radius": {
                "vs_current": comparison
            },
        }
        for window in ("fit", "holdout")
    }
    rule = {
        "minimum_late_requested_recovery_improvement": 0.05,
        "minimum_late_action_cross_window_cosine": 0.1,
        "maximum_outgoing_to_incoming_feedback_ratio": 1.0,
        "maximum_residual_ratio_increase_over_current": 0.05,
        "maximum_functional_upper_bound_regression_ce": 0.0,
        "minimum_one_window_mean_ce_improvement": 0.0001,
        "candidate_priority": list(
            (
                "task_select_corrected_fit_corrected_radius",
                "task_select_corrected_fit_requested_radius",
            )
        ),
    }
    decision = classify(rows, functional, rule)
    assert decision["selected_candidate"] == (
        "task_select_corrected_fit_corrected_radius"
    )
