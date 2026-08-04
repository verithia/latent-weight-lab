from __future__ import annotations

import torch

from examples.nanogpt.analyze_attention_gradient_selected_givens import (
    gate_passes,
    replace_selection_direction,
)


def test_replace_selection_direction_is_nonmutating_and_uses_descent() -> None:
    original_applied = torch.tensor([1.0, 2.0])
    gradient = torch.tensor([3.0, 4.0])
    probe = {
        "parameters": {
            "x": {
                "applied_direction_per_lr": original_applied,
                "gradient_after_clip": gradient,
            }
        }
    }
    selected = replace_selection_direction(probe)
    torch.testing.assert_close(
        selected["parameters"]["x"]["applied_direction_per_lr"],
        -gradient,
    )
    assert (
        probe["parameters"]["x"]["applied_direction_per_lr"]
        is original_applied
    )


def test_gate_passes_only_when_every_registered_check_passes() -> None:
    summary = {
        "energy_recovery": 0.25,
        "normalized_enrichment": 3.5,
        "maximum_orthogonality_error": 1e-8,
        "maximum_relative_normal_residual": 1e-6,
    }
    thresholds = {
        "task_gradient_recovery_minimum": 0.2,
        "task_gradient_enrichment_minimum": 3.0,
        "dense_muon_recovery_minimum": 0.15,
        "dense_muon_over_random_minimum": 2.0,
        "future_chord_recovery_minimum": 0.1,
        "future_chord_over_random_minimum": 1.75,
        "per_target_chord_recovery_minimum": 0.05,
        "maximum_projection_error": 1e-4,
        "maximum_normal_residual": 1e-4,
    }
    passed, failures = gate_passes(
        task_gradient=summary,
        dense_muon=summary,
        dense_muon_over_random=2.1,
        future_chord=summary,
        future_chord_over_random=1.8,
        future_by_target={"qk": summary, "v": summary, "cproj": summary},
        thresholds=thresholds,
    )
    assert passed
    assert failures == []

    failed = {**summary, "energy_recovery": 0.04}
    passed, failures = gate_passes(
        task_gradient=summary,
        dense_muon=summary,
        dense_muon_over_random=2.1,
        future_chord=summary,
        future_chord_over_random=1.8,
        future_by_target={"qk": summary, "v": summary, "cproj": failed},
        thresholds=thresholds,
    )
    assert not passed
    assert failures == ["per_target_future_chord"]


def test_python_result_booleans_are_json_serializable() -> None:
    import json

    payload = {"uses_dense_muon_target": False}
    assert json.loads(json.dumps(payload)) == {
        "uses_dense_muon_target": False
    }
