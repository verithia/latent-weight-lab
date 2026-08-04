from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from examples.nanogpt.analyze_mlp_cproj_global_directed_affine_output import (
    aggregate_results,
    directed_support_scores,
    fit_global_directed_map,
    support_overlap,
    validate_plan,
)


PLAN = Path(__file__).parent / "configs/selection_artifacts/124m_mlp_cproj_global_directed_affine_output_plan.json"


def valid_plan() -> dict:
    return json.loads(PLAN.read_text())


def test_plan_validation_fails_closed() -> None:
    validate_plan(valid_plan())
    changed = copy.deepcopy(valid_plan())
    changed["analysis"]["directed_map"]["incoming_coordinates_per_target"] = 32
    with pytest.raises(ValueError):
        validate_plan(changed)


def test_global_support_finds_exact_directed_sources_and_caps_energy() -> None:
    source = torch.eye(4)
    residual = torch.zeros(4, 4)
    residual[:, 0] = 0.2 * source[:, 3]
    residual[:, 1] = -0.1 * source[:, 2]
    activation = torch.eye(4)
    gradient = torch.ones(4, 4)
    activation_score, _combined, diagnostics = directed_support_scores(source, residual, activation, gradient)
    updated, supports, fit = fit_global_directed_map(
        source, residual, activation, activation_score,
        incoming=1, trust_output_energy=1.0, relative_ridge=0.0,
    )
    assert supports[0, 0].item() == 3
    assert supports[0, 1].item() == 2
    torch.testing.assert_close(updated - source, residual, rtol=0.0, atol=1e-6)
    assert diagnostics["residual_score_rms"] > 0.0
    assert fit["trust_energy_obeyed"] is True


def test_support_overlap() -> None:
    first = torch.tensor([[0, 1], [2, 3]])
    second = torch.tensor([[0, 3], [1, 2]])
    assert support_overlap(first, second) == pytest.approx(0.5)


def synthetic_rows() -> tuple[list[dict], list[dict], list[dict]]:
    rows: list[dict] = []
    finite: list[dict] = []
    chart: list[dict] = []
    for phase in (0, 60, 120, 180):
        for layer in (0, 3, 6, 9, 11):
            for candidate in ("global_directed16_activation", "global_directed16_activation_task_consensus"):
                chart.append({
                    "phase_start": phase, "layer": layer, "arm": candidate,
                    "trust_energy_obeyed": True, "trust_scale": 0.8,
                    "minimum_singular_value_i_plus_b": 0.99,
                    "train_fit_gradient_cosine": 0.3,
                    "support_overlap_with_other_arm": 0.5,
                    "coordinate_energy": 1.0, "skew_coordinate_energy": 0.3,
                    "symmetric_offdiag_coordinate_energy": 0.5,
                    "diagonal_coordinate_energy": 0.2,
                })
            for window in ("fit", "holdout"):
                for arm, residual, task in (
                    ("frobenius_output32", 1.00, 0.0035),
                    ("global_directed16_activation", 0.90, 0.0045),
                    ("global_directed16_activation_task_consensus", 0.88, 0.0047),
                ):
                    rows.append({
                        "phase_start": phase, "layer": layer, "window": window,
                        "arm": arm, "coordinates_per_layer": 147456,
                        "activation_output_residual_energy": residual,
                        "validation_gradient_predicted_ce_decrease": task,
                        "update_energy": 1.0,
                    })
        for window in ("fit", "holdout"):
            finite.extend([
                {"phase_start": phase, "window": window, "arm": "frobenius_output32", "loss": 7.1815},
                {"phase_start": phase, "window": window, "arm": "global_directed16_activation", "loss": 7.1805},
                {"phase_start": phase, "window": window, "arm": "global_directed16_activation_task_consensus", "loss": 7.1800},
            ])
    return rows, finite, chart


def test_aggregate_uses_smallest_pass_order() -> None:
    rows, finite, chart = synthetic_rows()
    result = aggregate_results(rows, finite, chart)
    assert result["passed"] is True
    assert result["selected"] == "global_directed16_activation"
    for row in chart:
        if row["arm"] == "global_directed16_activation":
            row["minimum_singular_value_i_plus_b"] = 0.9
    result = aggregate_results(rows, finite, chart)
    assert result["selected"] == "global_directed16_activation_task_consensus"
    assert result["authorization"]["language_model_training_authorized"] is False
