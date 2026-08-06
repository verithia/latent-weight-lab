from __future__ import annotations

import copy

import pytest
import torch

from examples.nanogpt.analyze_mlp_cproj_diagonal_kfac_selector import (
    aggregate_results,
    apply_plan_authorization,
    fit_metric_selected_raw_angle_pass,
    normalized_output_error_scale,
    require_full_state_snapshot,
    validate_plan,
)
from examples.nanogpt.muon_matched_givens import diagonal_metric_angles
from examples.nanogpt.parameter_trajectory import FULL_STATE_SCHEMA_VERSION


def valid_plan() -> dict:
    return {
        "schema_version": "mai_124m_mlp_cproj_diagonal_kfac_selector_plan_v1",
        "authorization": {
            "implement_and_run_zero_update_analysis": True,
            "run_language_model_training": False,
        },
        "analysis": {
            "parameter_updates": 0,
            "layers": [0, 3, 6, 9, 11],
            "phases": [[0, 60], [60, 120], [120, 180], [180, 238]],
            "fit_window": {
                "split": "validation",
                "seed": 20260804,
                "batch_size": 2,
                "block_size": 256,
                "batches": 4,
                "rows_per_layer": 2048,
                "participates_in_selection": True,
            },
            "holdout_window": {
                "split": "validation",
                "seed": 20260805,
                "batch_size": 2,
                "block_size": 256,
                "batches": 4,
                "rows_per_layer": 2048,
                "participates_in_selection": False,
            },
            "shared_chart": {
                "hidden_parent_stages": 64,
                "hidden_residual_stages": 24,
                "output_stages": 32,
                "neighbors": 64,
                "matching_seed": 20260804,
                "coordinate_count_per_layer": 147456,
                "feedback": "zero for this one-step prospective diagnostic",
                "weight_decay_application": "identical production ordering in every arm",
            },
            "output_error_scale": {"bounds": [0.25, 4.0]},
            "smallest_pass_order": [
                "activation_selector_output32",
                "error_selector_output32",
                "diagonal_kfac_selector_output32",
            ],
        },
    }


def test_plan_validation_fails_closed() -> None:
    validate_plan(valid_plan())
    changed = copy.deepcopy(valid_plan())
    changed["analysis"]["output_error_scale"]["bounds"] = [0.1, 10.0]
    with pytest.raises(ValueError):
        validate_plan(changed)


def test_functional_calibration_requires_full_state_snapshot() -> None:
    with pytest.raises(ValueError, match="full-state v2"):
        require_full_state_snapshot(
            {
                "schema_version": "nanogpt_parameter_trajectory_v1",
                "all_parameters": True,
            }
        )
    require_full_state_snapshot(
        {
            "schema_version": FULL_STATE_SCHEMA_VERSION,
            "all_parameters": True,
            "all_buffers": True,
            "buffers": {"transformer.h.0.mlp.c_fc.weight": torch.ones(1)},
        }
    )


def test_5tpp_calibration_plan_is_fail_closed_and_never_authorizes_training() -> None:
    plan = valid_plan()
    plan["schema_version"] = (
        "mai_124m_mlp_cproj_5tpp_functional_metric_calibration_plan_v1"
    )
    plan["authorization"] = {
        "run_zero_update_metric_calibration": True,
        "implement_candidate_structure": False,
        "run_language_model_training": False,
    }
    plan["analysis"]["layers"] = list(range(8))
    plan["analysis"]["phases"] = [
        [0, 594],
        [594, 1188],
        [1188, 1782],
        [1782, 2373],
    ]
    plan["analysis"]["fit_window"]["seed"] = 20260806
    plan["analysis"]["holdout_window"]["seed"] = 20260807
    plan["analysis"]["shared_chart"]["matching_seed"] = 20260806
    validate_plan(plan)
    aggregate = {"selected_variant": "activation_selector_output32"}
    authorized = apply_plan_authorization(aggregate, plan["schema_version"])
    assert authorized["authorization"]["short_shadow_rollout_authorized"] is True
    assert authorized["authorization"]["production_implementation_authorized"] is False
    assert authorized["authorization"]["language_model_training_authorized"] is False
    changed = copy.deepcopy(plan)
    changed["authorization"]["implement_candidate_structure"] = True
    with pytest.raises(ValueError):
        validate_plan(changed)


def test_output_error_scale_is_unit_rms_bounded_and_finite() -> None:
    errors = torch.tensor([[1.0, 2.0, 100.0], [1.0, 2.0, 100.0]])
    scale, diagnostics = normalized_output_error_scale(
        errors, minimum=0.25, maximum=4.0
    )
    assert torch.isfinite(scale).all()
    assert float(scale.min()) >= 0.25
    assert float(scale.max()) <= 4.0
    assert diagnostics["clamp_fraction"] > 0.0


def test_metric_selection_refits_raw_frobenius_angles() -> None:
    generator = torch.Generator().manual_seed(19)
    source = torch.randn(8, 16, generator=generator)
    residual = 0.01 * torch.randn(8, 16, generator=generator)
    hidden = torch.randn(12, 8, generator=generator)
    scale = torch.linspace(0.5, 1.5, 16)
    updated, diagnostics = fit_metric_selected_raw_angle_pass(
        source,
        residual,
        hidden=hidden,
        output_scale=scale,
        stages=4,
        neighbors=6,
        seed=23,
    )
    expected = diagonal_metric_angles(
        source, residual, diagnostics["permutations"]
    )
    torch.testing.assert_close(diagnostics["angles"], expected)
    assert updated.shape == source.shape
    assert diagnostics["coordinates"] == 32
    assert diagnostics["output_error_scaled"] is True


def synthetic_inputs(gain: float = 0.001) -> tuple[list[dict], list[dict], list[dict]]:
    rows = []
    finite = []
    variants = (
        "frobenius_output32",
        "activation_selector_output32",
        "error_selector_output32",
        "diagonal_kfac_selector_output32",
    )
    for phase in (0, 60, 120, 180):
        for layer in (0, 3, 6, 9, 11):
            for window in ("fit", "holdout"):
                for variant in variants:
                    candidate = variant != "frobenius_output32"
                    rows.append(
                        {
                            "phase_start": phase,
                            "layer": layer,
                            "window": window,
                            "variant": variant,
                            "task_predicted_ce_decrease": 1.1 if candidate else 1.0,
                            "activation_residual_energy": 1.0,
                            "update_energy": 1.0,
                            "weight_error_energy": 1.0,
                        }
                    )
        for window in ("fit", "holdout"):
            finite.append(
                {
                    "phase_start": phase,
                    "window": window,
                    "variant": "baseline",
                    "loss": 2.1,
                }
            )
            for variant in variants:
                finite.append(
                    {
                        "phase_start": phase,
                        "window": window,
                        "variant": variant,
                        "loss": 2.0 if variant == "frobenius_output32" else 2.0 - gain,
                    }
                )
    scales = [
        {"phase_start": phase, "layer": layer, "clamp_fraction": 0.0}
        for phase in (0, 60, 120, 180)
        for layer in (0, 3, 6, 9, 11)
    ]
    return rows, finite, scales


def decision_rule() -> dict:
    return {
        "candidate_requirements": {
            "mean_finite_step_ce_gain_over_frobenius_minimum": 0.0005,
            "finite_step_wins_minimum": 6,
            "holdout_wins_minimum": 3,
            "minimum_holdout_finite_step_ce_gain": 0.0,
            "holdout_activation_residual_energy_ratio_maximum": 1.25,
            "candidate_update_energy_ratio_maximum": 1.25,
            "holdout_to_fit_task_advantage_retention_minimum": 0.25,
            "output_error_scale_clamp_fraction_maximum": 0.25,
        }
    }


def test_aggregate_selects_smallest_passing_arm() -> None:
    rows, finite, scales = synthetic_inputs()
    result = aggregate_results(rows, finite, scales, decision_rule())
    assert result["passed"] is True
    assert result["selected_variant"] == "activation_selector_output32"
    assert result["authorization"]["language_model_training_authorized"] is False


def test_aggregate_rejects_subthreshold_gain() -> None:
    rows, finite, scales = synthetic_inputs(gain=0.0001)
    result = aggregate_results(rows, finite, scales, decision_rule())
    assert result["passed"] is False
    assert result["classification"] == "REJECT_DIAGONAL_KFAC_SELECTOR"


def test_nonpositive_fit_advantage_uses_json_null_retention() -> None:
    rows, finite, scales = synthetic_inputs()
    for row in rows:
        if row["variant"] == "error_selector_output32":
            row["task_predicted_ce_decrease"] = 0.9
    result = aggregate_results(rows, finite, scales, decision_rule())
    candidate = result["candidates"]["error_selector_output32"]
    assert candidate["holdout_to_fit_task_advantage_retention"] is None
    assert candidate["gate"]["task_advantage_retention"] is False
    assert candidate["passed"] is False
