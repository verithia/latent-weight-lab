from __future__ import annotations

import copy

import pytest
import torch

from examples.nanogpt.analyze_mlp_cproj_task_gradient_sequential_trust import (
    aggregate_results,
    sequential_refit_pass,
    validate_plan,
)


def valid_plan() -> dict:
    return {
        "schema_version": "mai_124m_mlp_cproj_task_gradient_sequential_trust_plan_v1",
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
            "trust_radius": {
                "definition": "For each layer-phase cell, max_abs(diagonal_metric_angles(S,R,frobenius_control_pairs)) from the existing simultaneous Frobenius output32 control.",
                "application": "Clamp every sequentially refitted scalar angle in both sequential arms to [-radius,+radius].",
                "minimum": 0.0,
                "tunable": False,
                "shared_between_sequential_arms": True,
            },
            "sequential_refit": {
                "connectivity_fixed_before_refit": True,
                "stages": 32,
                "stage_angle": "diagonal_metric_angles(current_source,current_remaining_residual,current_single_matching)",
                "after_stage": "remaining_residual -= next_source-current_source; current_source = next_source",
                "task_gradient_used_for_angles": False,
                "holdout_used_for_selection_or_angles": False,
            },
            "parameter_updates": 0,
        },
    }


def test_plan_validation_fails_closed() -> None:
    validate_plan(valid_plan())
    changed = copy.deepcopy(valid_plan())
    changed["analysis"]["trust_radius"]["tunable"] = True
    with pytest.raises(ValueError):
        validate_plan(changed)


def test_sequential_refit_obeys_trust_and_reduces_residual() -> None:
    source = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    residual = torch.tensor([[0.0, 0.2, 0.0, 0.0]])
    permutations = torch.tensor([[0, 1, 2, 3]])
    updated, diagnostics = sequential_refit_pass(
        source, residual, permutations, trust_radius=0.05
    )
    realized = updated - source
    assert diagnostics["trust_obeyed"] is True
    assert diagnostics["clipped_coordinates"] == 1
    assert diagnostics["maximum_abs_angle"] == pytest.approx(0.05)
    assert (residual - realized).square().sum() < residual.square().sum()


def synthetic_rows() -> tuple[list[dict], list[dict], list[dict]]:
    rows = []
    finite = []
    trust = []
    values = {
        "frobenius_simultaneous": (1.0, 1.0, 1.0),
        "frobenius_sequential_trust": (1.1, 0.9, 0.95),
        "task_gradient_sequential_trust": (1.5, 0.95, 1.0),
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
            for arm in values:
                trust.append(
                    {
                        "phase_start": phase,
                        "layer": layer,
                        "arm": arm,
                        "trust_obeyed": True,
                        "trust_radius": 0.1,
                        "maximum_abs_angle": 0.1,
                    }
                )
        for window in ("fit", "holdout"):
            for arm, loss in (
                ("frobenius_simultaneous", 2.001),
                ("frobenius_sequential_trust", 2.0),
                ("task_gradient_sequential_trust", 1.999),
            ):
                finite.append(
                    {
                        "phase_start": phase,
                        "window": window,
                        "arm": arm,
                        "loss": loss,
                    }
                )
    return rows, finite, trust


def test_aggregate_requires_every_sequential_gate() -> None:
    rows, finite, trust = synthetic_rows()
    result = aggregate_results(rows, finite, trust)
    assert result["passed"] is True
    assert result["finite_step"]["candidate_wins"] == 8
    assert result["finite_step"]["holdout_wins"] == 4
    assert result["authorization"]["language_model_training_authorized"] is False

    finite[0]["loss"] = 1.0
    failed = aggregate_results(rows, finite, trust)
    assert failed["passed"] is False
