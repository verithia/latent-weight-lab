from __future__ import annotations

import json
from pathlib import Path

import torch

from examples.nanogpt.analyze_mlp_cproj_teacher_forced_directed_product_carry import (
    CANDIDATES,
    OUTPUT_COORDINATES,
    aggregate_rows,
    fit_directed_product,
    validate_plan,
)


PLAN = Path(__file__).parent / "configs" / "selection_artifacts" / "124m_mlp_cproj_teacher_forced_directed_product_carry_plan.json"


def test_registered_plan_is_valid() -> None:
    validate_plan(json.loads(PLAN.read_text()))


def test_directed_product_obeys_budget_energy_and_invertibility() -> None:
    torch.manual_seed(7)
    weight = torch.randn(24, 64)
    residual = 0.01 * torch.randn_like(weight)
    updated, diagnostics = fit_directed_product(
        weight,
        residual,
        incoming_by_stage=(8, 8),
        trust_output_energy=0.02,
    )
    assert updated.shape == weight.shape
    assert int(diagnostics["coordinate_count"]) == 24 * 16
    assert diagnostics["bounded_output_delta_energy"] <= 0.0200002
    assert diagnostics["minimum_singular_value_lower_bound"] >= 0.95
    assert torch.isfinite(updated).all()


def _synthetic_rows(candidate_recovery: float) -> tuple[list[dict], list[dict]]:
    arms = (
        "hidden88_full_carry",
        "hidden88_output32_full_carry",
        *CANDIDATES,
    )
    rows = []
    charts = []
    for step in (60, 120, 180, 238):
        for layer in range(5):
            for arm in arms:
                recovery = 0.93 if arm == "hidden88_output32_full_carry" else candidate_recovery
                if arm == "hidden88_full_carry":
                    recovery = 0.84
                row = {
                    "arm": arm,
                    "layer": layer,
                    "score_step": step,
                    "chord_energy": 1.0,
                    "endpoint_error_energy": 1.0 - recovery,
                    "endpoint_recovery": recovery,
                    "endpoint_cosine": 0.9,
                    "row_gram_chord_energy": 1.0,
                    "row_gram_error_energy": 0.4 if arm in CANDIDATES else 0.49,
                    "row_gram_recovery": 0.6 if arm in CANDIDATES else 0.51,
                    "mean_requested_update_recovery": -1.0 if arm in CANDIDATES else -2.0,
                    "terminal_feedback_fro": 0.7 if arm in CANDIDATES else 0.85,
                }
                rows.append(row)
            for arm in CANDIDATES:
                charts.append({
                    "arm": arm,
                    "layer": layer,
                    "step": step,
                    "coordinate_count": OUTPUT_COORDINATES,
                    "bounded_output_delta_energy": 0.9,
                    "trust_output_energy": 1.0,
                    "minimum_singular_value_lower_bound": 0.96,
                    "trust_scale": 0.8,
                })
    return rows, charts


def test_smallest_passing_directed_arm_is_selected() -> None:
    plan = json.loads(PLAN.read_text())
    rows, charts = _synthetic_rows(0.95)
    result = aggregate_rows(rows, charts, plan)
    assert result["selected_arm"] == CANDIDATES[0]


def test_subthreshold_directed_arms_are_rejected() -> None:
    plan = json.loads(PLAN.read_text())
    rows, charts = _synthetic_rows(0.935)
    result = aggregate_rows(rows, charts, plan)
    assert result["selected_arm"] is None
