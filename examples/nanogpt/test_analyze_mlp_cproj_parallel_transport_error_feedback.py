from __future__ import annotations

import copy

import pytest
import torch

from examples.nanogpt.analyze_mlp_cproj_parallel_transport_error_feedback import (
    aggregate_results,
    apply_transport_recipe,
    structured_step,
    validate_plan,
)
from examples.nanogpt.muon_matched_givens import apply_givens_flow


def valid_plan() -> dict:
    return {
        "schema_version": "mai_124m_mlp_cproj_parallel_transport_error_feedback_plan_v2",
        "analysis": {
            "parameter_updates": 0,
            "layers": [0, 3, 6, 9, 11],
            "score_steps": [60, 120, 180, 238],
            "finite_ce_score_steps": [238],
            "fit_window": {"seed": 20260804},
            "holdout_window": {"seed": 20260805},
            "shared_chart": {
                "hidden_parent_stages": 64,
                "hidden_residual_stages": 24,
                "neighbors": 64,
                "feedback_decay": 1.0,
            },
            "smallest_pass_order": ["pushforward_carry", "pullback_carry"],
        },
        "authorization": {"run_language_model_training": False},
    }


def test_plan_validation_fails_closed() -> None:
    validate_plan(valid_plan())
    changed = copy.deepcopy(valid_plan())
    changed["analysis"]["shared_chart"]["feedback_decay"] = 0.5
    with pytest.raises(ValueError):
        validate_plan(changed)


def test_transport_recipe_is_norm_preserving_and_exactly_invertible() -> None:
    generator = torch.Generator().manual_seed(17)
    values = torch.randn(6, 8, generator=generator)
    permutations = torch.stack((torch.randperm(8, generator=generator),) * 2)
    inverse = torch.argsort(permutations, dim=1)
    angles = 0.1 * torch.randn(2, 4, generator=generator)
    recipe = [(angles, permutations, inverse)]
    forward = apply_transport_recipe(values, recipe, inverse=False)
    expected = apply_givens_flow(values, angles, permutations, inverse)
    torch.testing.assert_close(forward, expected)
    restored = apply_transport_recipe(forward, recipe, inverse=True)
    torch.testing.assert_close(restored, values, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(forward.norm(), values.norm())


def test_zero_feedback_first_weight_step_is_identical_across_arms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = torch.Generator().manual_seed(23)
    weight = torch.randn(4, 8, generator=generator)
    update = 0.01 * torch.randn(4, 8, generator=generator)

    def fake_match(weight, target, *, stages, neighbors, seed):
        del target, neighbors, seed
        base = torch.arange(weight.shape[1])
        return base.repeat(stages, 1), {}

    monkeypatch.setattr(
        "examples.nanogpt.analyze_mlp_cproj_parallel_transport_error_feedback.fast_muon_matched_permutations",
        fake_match,
    )
    outputs = []
    for arm in ("ambient_carry", "pushforward_carry", "pullback_carry"):
        outputs.append(
            structured_step(
                weight,
                update,
                torch.zeros_like(weight),
                arm=arm,
                learning_rate=0.01,
                weight_decay=0.1,
                neighbors=4,
                seed=7,
            )
        )
    torch.testing.assert_close(outputs[0][0], outputs[1][0])
    torch.testing.assert_close(outputs[0][0], outputs[2][0])
    torch.testing.assert_close(outputs[0][1].norm(), outputs[1][1].norm())
    torch.testing.assert_close(outputs[0][1].norm(), outputs[2][1].norm())


def synthetic_rows(gain: float = 0.001) -> tuple[list[dict], list[dict]]:
    rows = []
    finite = []
    for step in (60, 120, 180, 238):
        for layer in (0, 3, 6, 9, 11):
            for arm in ("ambient_carry", "pushforward_carry", "pullback_carry"):
                candidate = arm != "ambient_carry"
                rows.append(
                    {
                        "arm": arm,
                        "layer": layer,
                        "score_step": step,
                        "chord_energy": 1.0,
                        "endpoint_error_energy": 0.4 if candidate else 0.5,
                        "endpoint_recovery": 0.6 if candidate else 0.5,
                        "terminal_feedback_fro": 1.0,
                        "transport_norm_ratio": 1.0,
                    }
                )
        if step == 238:
            for window in ("fit", "holdout"):
                for arm in (
                    "ambient_carry",
                    "pushforward_carry",
                    "pullback_carry",
                ):
                    finite.append(
                        {
                            "score_step": step,
                            "window": window,
                            "arm": arm,
                            "loss": (
                                2.0
                                if arm == "ambient_carry"
                                else 2.0 - gain
                            ),
                        }
                    )
    return rows, finite


def rule() -> dict:
    return {
        "candidate_requirements": {
            "mean_finite_step_ce_gain_over_ambient_minimum": 0.0005,
            "finite_step_wins_minimum": 2,
            "holdout_wins_minimum": 1,
            "minimum_holdout_finite_step_ce_gain": 0.0,
            "terminal_endpoint_recovery_ratio_over_ambient_minimum": 1.02,
            "terminal_layers_beating_ambient_minimum": 4,
            "maximum_feedback_fro_ratio_over_ambient": 1.01,
        }
    }


def test_aggregate_selects_smallest_passing_transport() -> None:
    rows, finite = synthetic_rows()
    result = aggregate_results(rows, finite, rule())
    assert result["passed"] is True
    assert result["selected_arm"] == "pushforward_carry"
    assert result["authorization"]["language_model_training_authorized"] is False


def test_aggregate_rejects_subthreshold_task_gain() -> None:
    rows, finite = synthetic_rows(gain=0.0001)
    result = aggregate_results(rows, finite, rule())
    assert result["passed"] is False
    assert result["classification"] == "REJECT_PARALLEL_TRANSPORT_ERROR_FEEDBACK"
