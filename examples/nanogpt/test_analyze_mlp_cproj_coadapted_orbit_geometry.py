from __future__ import annotations

import copy

import pytest
import torch

from examples.nanogpt.analyze_mlp_cproj_coadapted_orbit_geometry import (
    classify,
    scaled_right_orbit_metrics,
    support_metrics,
    validate_plan,
)


def valid_plan() -> dict:
    return {
        "schema_version": "mai_124m_mlp_cproj_coadapted_orbit_geometry_plan_v1",
        "analysis": {
            "parameter_updates": 0,
            "late_layers": [8, 9, 10, 11],
            "phase_steps": [0, 594, 1188, 1782, 2373],
        },
        "decision_rule": {
            "thresholds": {
                "same_initialization_max_absolute_error": 0.0,
                "procedural_minimum_orbit_recovery": 0.995,
                "late_minus_early_orbit_recovery": 0.05,
                "late_to_early_gram_drift_ratio": 0.8,
                "reusable_support_retention_fraction": 0.10,
                "reusable_support_enrichment": 2.0,
                "path_alignment_cosine": 0.5,
            }
        },
        "authorization": {
            "run_zero_update_orbit_geometry": True,
            "implement_candidate_structure": False,
            "run_exact_config_mfu": False,
            "run_language_model_training": False,
            "larger_rung": False,
        },
    }


def test_plan_is_fail_closed() -> None:
    validate_plan(valid_plan())
    changed = copy.deepcopy(valid_plan())
    changed["decision_rule"]["thresholds"][
        "procedural_minimum_orbit_recovery"
    ] = 0.9
    with pytest.raises(ValueError):
        validate_plan(changed)


def test_scaled_right_orbit_recovers_exact_rotation_and_scale() -> None:
    generator = torch.Generator().manual_seed(17)
    source = torch.randn(4, 8, generator=generator, dtype=torch.float64)
    orthogonal, _ = torch.linalg.qr(
        torch.randn(8, 8, generator=generator, dtype=torch.float64)
    )
    target = 0.91 * source @ orthogonal
    metrics = scaled_right_orbit_metrics(source, target)
    assert metrics["optimal_scale"] == pytest.approx(0.91, abs=1e-10)
    assert metrics["orbit_recovery"] >= 1.0 - 1e-10
    assert metrics["normalized_output_gram_drift"] <= 1e-10


def test_output_axis_deformation_leaves_the_right_orbit() -> None:
    generator = torch.Generator().manual_seed(23)
    source = torch.randn(4, 8, generator=generator, dtype=torch.float64)
    target = torch.diag(torch.tensor([0.7, 0.9, 1.1, 1.3])).double() @ source
    metrics = scaled_right_orbit_metrics(source, target)
    assert metrics["orbit_recovery"] < 0.99
    assert metrics["normalized_output_gram_drift"] > 0.01


def test_support_retention_is_measured_against_random_edge_overlap() -> None:
    left = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 2, 1, 4, 3, 5]])
    same = support_metrics(left, left.clone())
    assert same["retention_fraction"] == 1.0
    assert same["retention_enrichment"] > 1.0

    right = torch.tensor([[0, 3, 1, 5, 2, 4], [0, 4, 1, 3, 2, 5]])
    moved = support_metrics(left, right)
    assert moved["retention_fraction"] < 1.0


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"same_initialization": False}, "INVALID_CROSS_RUN_GAUGE"),
        (
            {"accepted_path_is_scaled_right_orbit": False},
            "ACCEPTED_PATH_EXCEEDS_SCALED_RIGHT_ORBIT",
        ),
        (
            {
                "right_orbit_localizes_late_band": True,
                "support_is_reusable": True,
            },
            "RIGHT_ORBIT_LOCALIZES_LWT_WITH_REUSABLE_SUPPORT",
        ),
        (
            {
                "right_orbit_localizes_late_band": True,
                "support_is_reusable": False,
            },
            "RIGHT_ORBIT_LOCALIZES_LWT_BUT_SUPPORT_IS_MOVING",
        ),
    ],
)
def test_classification_is_frozen(
    overrides: dict[str, bool], expected: str
) -> None:
    gates = {
        "same_initialization": True,
        "accepted_path_is_scaled_right_orbit": True,
        "right_orbit_localizes_late_band": False,
        "support_is_reusable": True,
        "coadapted_path_tracks_dense_parent": False,
    }
    gates.update(overrides)
    assert classify(gates) == expected
