from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from examples.nanogpt.analyze_mlp_cproj_global_directed_minimax_output import (
    aggregate_results,
    minimax_support_score,
    validate_plan,
)


PLAN = Path(__file__).parent / "configs/selection_artifacts/124m_mlp_cproj_global_directed_minimax_output_plan.json"


def valid_plan() -> dict:
    return json.loads(PLAN.read_text())


def test_plan_validation_fails_closed() -> None:
    validate_plan(valid_plan())
    changed = copy.deepcopy(valid_plan())
    changed["analysis"]["directed_map"]["incoming_coordinates_per_target"] = 8
    with pytest.raises(ValueError):
        validate_plan(changed)


def test_minimax_score_penalizes_task_disagreement() -> None:
    source = torch.eye(2)
    residual = 0.1 * torch.eye(2)
    activation = torch.eye(2)
    train_gradient = -torch.eye(2)
    fit_gradient = torch.eye(2)
    score, diagnostics = minimax_support_score(
        source, residual, activation, train_gradient, fit_gradient,
        relative_ridge=0.0,
    )
    assert diagnostics["positive_task_agreement_fraction"] == 0.0
    assert score[0, 0] < 1.0
    assert score[1, 1] < 1.0


def synthetic_rows() -> tuple[list[dict], list[dict], list[dict]]:
    rows: list[dict] = []
    finite: list[dict] = []
    chart: list[dict] = []
    for phase in (0, 60, 120, 180):
        for layer in (0, 3, 6, 9, 11):
            for arm in ("global_directed16_average_consensus", "global_directed16_minimax_consensus"):
                chart.append({
                    "phase_start": phase, "layer": layer, "arm": arm,
                    "trust_energy_obeyed": True, "trust_scale": 0.8,
                    "minimum_singular_value_i_plus_b": 0.99,
                    "train_fit_gradient_cosine": 0.3,
                    "support_overlap_with_other_arm": 0.5,
                    "positive_task_agreement_fraction": 0.25,
                })
            for window in ("fit", "holdout"):
                for arm, residual, task in (
                    ("frobenius_output32", 1.0, 0.0035),
                    ("global_directed16_average_consensus", 0.90, 0.0045),
                    ("global_directed16_minimax_consensus", 0.88, 0.0047),
                ):
                    rows.append({"phase_start": phase, "layer": layer, "window": window, "arm": arm, "coordinates_per_layer": 147456, "activation_output_residual_energy": residual, "validation_gradient_predicted_ce_decrease": task, "update_energy": 1.0})
        for window in ("fit", "holdout"):
            finite.extend([
                {"phase_start": phase, "window": window, "arm": "frobenius_output32", "loss": 7.1815},
                {"phase_start": phase, "window": window, "arm": "global_directed16_average_consensus", "loss": 7.1807},
                {"phase_start": phase, "window": window, "arm": "global_directed16_minimax_consensus", "loss": 7.1800},
            ])
    return rows, finite, chart


def test_aggregate_requires_all_frozen_gates() -> None:
    rows, finite, chart = synthetic_rows()
    result = aggregate_results(rows, finite, chart)
    assert result["passed"] is True
    assert result["authorization"]["language_model_training_authorized"] is False
    for row in chart:
        if row["arm"] == "global_directed16_minimax_consensus":
            row["minimum_singular_value_i_plus_b"] = 0.9
    assert aggregate_results(rows, finite, chart)["passed"] is False
