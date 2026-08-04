from __future__ import annotations

import copy

import pytest
import torch

from examples.nanogpt.muon_matched_givens import diagonal_metric_angles
from examples.nanogpt.analyze_mlp_cproj_task_gradient_output_selector import (
    aggregate_results,
    fit_task_gradient_hybrid_pass,
    task_gradient_pair_scores,
    validate_plan,
)


def valid_plan() -> dict:
    plan = {
        "schema_version": "mai_124m_mlp_cproj_task_gradient_output_selector_plan_v1",
        "authorization": {"implement_and_run_zero_update_analysis": True},
        "analysis": {
            "layers": [0, 3, 6, 9, 11],
            "phases": [[0, 60], [60, 120], [120, 180], [180, 238]],
            "fit_window": {
                "split": "validation",
                "seed": 20260804,
                "batch_size": 2,
                "block_size": 256,
                "batches": 4,
                "rows_per_layer": 2048,
            },
            "holdout_window": {
                "split": "validation",
                "seed": 20260805,
                "batch_size": 2,
                "block_size": 256,
                "batches": 4,
                "rows_per_layer": 2048,
            },
            "shared_chart": {
                "hidden_parent_stages": 64,
                "hidden_residual_stages": 24,
                "output_stages": 32,
                "neighbors": 64,
                "matching_seed": 20260804,
                "coordinate_count_per_layer": 147456,
                "feedback": "zero for this one-step prospective diagnostic",
                "weight_decay_application": "identical production ordering in both arms",
            },
            "candidate": {
                "name": "task_gradient_hybrid_output32",
                "source": "S = W_after_hidden^T",
                "residual": "R = remaining requested update^T after shared hidden64+24",
                "fit_task_gradient": "G = current fit-window validation gradient^T; holdout gradient is scoring-only",
                "per_edge_residual_inner": "r_ij = dot(S_i,R_j)-dot(S_j,R_i)",
                "per_edge_coordinate_norm": "q_ij = ||S_i||^2+||S_j||^2",
                "per_edge_angle": "a_ij = r_ij/max(q_ij,1e-30)",
                "per_edge_residual_score": "u_ij = r_ij^2/max(q_ij,1e-30)",
                "per_edge_task_inner": "g_ij = dot(S_i,G_j)-dot(S_j,G_i)",
                "per_edge_task_score": "v_ij = -a_ij*g_ij",
                "normalization": "divide u and v independently by their RMS over finite strict-upper-triangle edges, each clamped at 1e-30",
                "combined_score": "score_ij = normalized(u_ij)+normalized(v_ij)",
                "candidate_graph": "top 64 combined-score neighbors per output vertex, then the existing deterministic compiled edge coloring",
                "angle_fit": "After selecting 32 matchings, recompute ordinary Frobenius diagonal_metric_angles(S,R,pairs); do not fit angles in activation or task metric.",
                "application": "Apply selected pairs and Frobenius angles to unprojected S, transpose, then use the same production weight-decay ordering as control.",
            },
            "parameter_updates": 0,
        },
    }
    return plan


def test_plan_validation_fails_closed() -> None:
    validate_plan(valid_plan())
    changed = copy.deepcopy(valid_plan())
    changed["analysis"]["candidate"]["combined_score"] = "post-hoc mixture"
    with pytest.raises(ValueError):
        validate_plan(changed)


def test_task_score_sign_prefers_predicted_descent() -> None:
    source = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    residual = torch.tensor([[0.0, 0.1, 0.0, 0.0]])
    gradient = torch.tensor([[0.0, -1.0, 0.0, 0.0]])
    scores, diagnostics = task_gradient_pair_scores(source, residual, gradient)
    assert scores[0, 1] > scores[0, 2]
    assert diagnostics["task_score_rms"] > 0.0
    assert diagnostics["positive_task_edge_fraction"] > 0.0


def test_hybrid_pass_keeps_frobenius_angles_and_budget() -> None:
    generator = torch.Generator().manual_seed(19)
    source = torch.randn(6, 16, generator=generator)
    residual = 0.01 * torch.randn(6, 16, generator=generator)
    gradient = torch.randn(6, 16, generator=generator)
    updated, diagnostics = fit_task_gradient_hybrid_pass(
        source,
        residual,
        gradient,
        stages=4,
        neighbors=6,
        seed=23,
    )
    assert updated.shape == source.shape
    assert torch.isfinite(updated).all()
    assert diagnostics["coordinates"] == 32
    assert diagnostics["maximum_abs_angle"] < 1.0
    expected = diagonal_metric_angles(
        source, residual, diagnostics["permutations"]
    )
    torch.testing.assert_close(diagnostics["angles"], expected)


def synthetic_rows(task_multiplier: float = 1.5) -> tuple[list[dict], list[dict]]:
    rows = []
    finite = []
    for phase in (0, 60, 120, 180):
        for layer in (0, 3, 6, 9, 11):
            for window in ("fit", "holdout"):
                for candidate, task, residual, update in (
                    ("frobenius_output32", 1.0, 1.0, 1.0),
                    (
                        "task_gradient_hybrid_output32",
                        task_multiplier,
                        1.1,
                        1.1,
                    ),
                ):
                    rows.append(
                        {
                            "phase_start": phase,
                            "layer": layer,
                            "window": window,
                            "candidate": candidate,
                            "validation_gradient_predicted_ce_decrease": task,
                            "activation_output_residual_energy": residual,
                            "update_energy": update,
                            "weight_error_energy": residual,
                            "coordinates_per_layer": 147456,
                        }
                    )
        for window in ("fit", "holdout"):
            finite.extend(
                [
                    {
                        "phase_start": phase,
                        "window": window,
                        "candidate": "frobenius_output32",
                        "loss": 2.0,
                    },
                    {
                        "phase_start": phase,
                        "window": window,
                        "candidate": "task_gradient_hybrid_output32",
                        "loss": 1.99,
                    },
                ]
            )
    return rows, finite


def test_aggregate_requires_every_preregistered_gate() -> None:
    rows, finite = synthetic_rows()
    result = aggregate_results(rows, finite)
    assert result["passed"] is True
    assert result["finite_step"]["candidate_wins"] == 8
    assert result["authorization"]["language_model_training_authorized"] is False

    failed_rows, failed_finite = synthetic_rows(task_multiplier=1.1)
    failed = aggregate_results(failed_rows, failed_finite)
    assert failed["passed"] is False
    assert failed["decision"] == "REJECT_TASK_GRADIENT_OUTPUT_SELECTOR"
