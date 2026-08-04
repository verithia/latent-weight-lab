from __future__ import annotations

import copy

import pytest
import torch

from examples.nanogpt.analyze_mlp_cproj_cross_batch_consensus_selector import (
    aggregate_results,
    normalized_consensus_gradient,
    validate_plan,
)


def valid_plan() -> dict:
    return {
        "schema_version": "mai_124m_mlp_cproj_cross_batch_consensus_selector_plan_v1",
        "authorization": {"implement_and_run_zero_update_analysis": True},
        "analysis": {
            "layers": [0, 3, 6, 9, 11],
            "phases": [[0, 60], [60, 120], [120, 180], [180, 238]],
            "fit_window": {"split": "validation", "seed": 20260804, "batch_size": 2, "block_size": 256, "batches": 4, "rows_per_layer": 2048},
            "holdout_window": {"split": "validation", "seed": 20260805, "batch_size": 2, "block_size": 256, "batches": 4, "rows_per_layer": 2048},
            "shared_chart": {
                "hidden_parent_stages": 64,
                "hidden_residual_stages": 24,
                "output_stages": 32,
                "neighbors": 64,
                "matching_seed": 20260804,
                "coordinate_count_per_layer": 147456,
                "feedback": "zero for this one-step prospective diagnostic",
                "weight_decay_application": "identical production ordering in all arms",
            },
            "consensus": {
                "formula": "G_consensus = 0.5 * (G_recorded_train/max(||G_recorded_train||_F,1e-30) + G_fit_validation/max(||G_fit_validation||_F,1e-30))",
                "weights": [0.5, 0.5],
                "tunable": False,
                "holdout_used": False,
            },
            "parameter_updates": 0,
        },
    }


def test_plan_validation_fails_closed() -> None:
    validate_plan(valid_plan())
    changed = copy.deepcopy(valid_plan())
    changed["analysis"]["consensus"]["weights"] = [0.25, 0.75]
    with pytest.raises(ValueError):
        validate_plan(changed)


def test_consensus_normalizes_sources_before_equal_weighting() -> None:
    recorded = torch.tensor([[3.0, 0.0]])
    fit = torch.tensor([[0.0, 4.0]])
    consensus, diagnostics = normalized_consensus_gradient(recorded, fit)
    torch.testing.assert_close(consensus, torch.tensor([[0.5, 0.5]]))
    assert diagnostics["recorded_fit_gradient_cosine"] == pytest.approx(0.0)
    assert diagnostics["consensus_fro"] == pytest.approx(2.0**-0.5)


def synthetic_rows() -> tuple[list[dict], list[dict], list[dict]]:
    rows = []
    finite = []
    consensus = []
    values = {
        "frobenius_output32": (1.0, 1.0, 1.0),
        "fit_task_output32": (1.3, 1.05, 1.05),
        "consensus_task_output32": (1.5, 1.05, 1.05),
    }
    for phase in (0, 60, 120, 180):
        for layer in (0, 3, 6, 9, 11):
            for window in ("fit", "holdout"):
                for arm, (task, residual, update) in values.items():
                    rows.append(
                        {
                            "phase_start": phase,
                            "layer": layer,
                            "window": window,
                            "arm": arm,
                            "coordinates_per_layer": 147456,
                            "validation_gradient_predicted_ce_decrease": task,
                            "activation_output_residual_energy": residual,
                            "update_energy": update,
                            "weight_error_energy": residual,
                        }
                    )
            consensus.append(
                {
                    "phase_start": phase,
                    "layer": layer,
                    "recorded_fit_gradient_cosine": 0.2,
                }
            )
        for window in ("fit", "holdout"):
            for arm, loss in (
                ("frobenius_output32", 2.0),
                ("fit_task_output32", 1.9995),
                ("consensus_task_output32", 1.999),
            ):
                finite.append(
                    {
                        "phase_start": phase,
                        "window": window,
                        "arm": arm,
                        "loss": loss,
                    }
                )
    return rows, finite, consensus


def test_aggregate_requires_all_consensus_gates() -> None:
    rows, finite, consensus = synthetic_rows()
    result = aggregate_results(rows, finite, consensus)
    assert result["passed"] is True
    assert result["finite_step"]["holdout_wins_vs_fit_task"] == 4
    assert result["authorization"]["language_model_training_authorized"] is False

    finite[0]["loss"] = 1.0
    failed = aggregate_results(rows, finite, consensus)
    assert failed["passed"] is False
