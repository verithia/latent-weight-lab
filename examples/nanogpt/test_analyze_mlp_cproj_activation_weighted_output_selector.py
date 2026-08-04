from __future__ import annotations

import copy

import pytest
import torch

from examples.nanogpt.analyze_mlp_cproj_activation_weighted_output_selector import (
    aggregate_results,
    fit_activation_weighted_pass,
    fit_frobenius_pass,
    output_residual_energy,
    validate_plan,
)


def valid_plan() -> dict:
    return {
        "schema_version": (
            "mai_124m_mlp_cproj_activation_weighted_output_selector_plan_v1"
        ),
        "authorization": {"zero_update_selector_analysis": True},
        "selector_analysis": {
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
                "weight_decay_application": (
                    "identical production ordering in both arms"
                ),
            },
            "parameter_updates": 0,
            "prohibited": [
                "learned basis",
                "inverse JtJ or conjugate-gradient pullback",
                "dense residual",
                "extra chart coordinates",
                "selection on holdout activations",
            ],
        },
    }


def test_plan_validation_is_fail_closed() -> None:
    validate_plan(valid_plan())
    changed = copy.deepcopy(valid_plan())
    changed["selector_analysis"]["shared_chart"]["output_stages"] = 64
    with pytest.raises(ValueError):
        validate_plan(changed)


def test_activation_selector_prefers_the_active_conflicting_edge(tmp_path) -> None:
    source = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [100.0, 0.0, 0.0, 0.0]]
    )
    residual = torch.tensor(
        [[0.0, 0.05, 0.0, 0.0], [0.0, 0.0, 5.0, 0.0]]
    )
    hidden = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    control, control_diagnostics = fit_frobenius_pass(
        source, residual, stages=1, neighbors=2, seed=17
    )
    candidate, candidate_diagnostics = fit_activation_weighted_pass(
        source, residual, hidden, stages=1, neighbors=2, seed=17
    )
    control_error = output_residual_energy(
        hidden, residual.T, (control - source).T
    )
    candidate_error = output_residual_energy(
        hidden, residual.T, (candidate - source).T
    )
    assert candidate_error < 0.05 * control_error
    assert control_diagnostics["coordinates"] == 2
    assert candidate_diagnostics["coordinates"] == 2
    assert candidate_diagnostics["fit_rows"] == 2


def synthetic_rows(candidate_ratio: float = 0.90) -> tuple[list[dict], list[dict]]:
    rows = []
    finite = []
    for phase in (0, 60, 120, 180):
        for layer in (0, 3, 6, 9, 11):
            for window in ("fit", "holdout"):
                for candidate, ratio, descent in (
                    ("frobenius_output32", 1.0, 1.0),
                    ("activation_output32", candidate_ratio, 1.1),
                ):
                    rows.append(
                        {
                            "phase_start": phase,
                            "layer": layer,
                            "window": window,
                            "candidate": candidate,
                            "activation_output_residual_energy": ratio,
                            "validation_gradient_predicted_ce_decrease": descent,
                            "recorded_train_gradient_predicted_ce_decrease": descent,
                            "target_output_energy": 2.0,
                            "weight_error_energy": 1.0,
                            "target_weight_energy": 2.0,
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
                        "candidate": "activation_output32",
                        "loss": 1.99,
                    },
                ]
            )
    return rows, finite


def test_aggregate_passes_only_when_every_registered_rule_passes() -> None:
    rows, finite = synthetic_rows()
    result = aggregate_results(rows, finite)
    assert result["passed"] is True
    assert result["finite_step"]["activation_wins"] == 8
    assert result["authorization"]["language_model_training_authorized"] is False

    failed_rows, failed_finite = synthetic_rows(candidate_ratio=0.97)
    failed = aggregate_results(failed_rows, failed_finite)
    assert failed["passed"] is False
    assert failed["decision"] == "REJECT_ACTIVATION_WEIGHTED_OUTPUT_SELECTOR"
