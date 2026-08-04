from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from examples.nanogpt.analyze_mlp_cproj_hybrid_output import (
    ARMS,
    CANDIDATES,
    aggregate_results,
    bound_composed_transform,
    fit_hybrid,
    task_givens_component,
    validate_plan,
)


PLAN = Path(__file__).parent / "configs/selection_artifacts/124m_mlp_cproj_hybrid_output_plan.json"


def valid_plan() -> dict:
    return json.loads(PLAN.read_text())


def test_plan_validation_fails_closed() -> None:
    validate_plan(valid_plan())
    changed = copy.deepcopy(valid_plan())
    changed["analysis"]["output_budget"]["hybrid_minimax_incoming_per_target"] = 9
    with pytest.raises(ValueError):
        validate_plan(changed)


def test_task_component_exposes_exact_transform() -> None:
    generator = torch.Generator().manual_seed(7)
    source = torch.randn(8, 4, generator=generator)
    residual = 0.01 * torch.randn(8, 4, generator=generator)
    gradient = torch.randn(8, 4, generator=generator)
    updated, transform, diagnostics = task_givens_component(
        source, residual, gradient, stages=1, neighbors=2, seed=11
    )
    torch.testing.assert_close(updated, source @ transform, rtol=2e-5, atol=2e-6)
    assert diagnostics["coordinates"] == 2


def test_combined_bound_caps_energy_and_preserves_identity_ray() -> None:
    source = torch.eye(4)
    transform = torch.eye(4)
    transform[0, 1] = 0.4
    updated, bounded, diagnostics = bound_composed_transform(
        source, transform, trust_output_energy=0.04
    )
    assert diagnostics["trust_scale"] == pytest.approx(0.5)
    assert diagnostics["trust_energy_obeyed"] is True
    torch.testing.assert_close(updated, source @ bounded)
    assert diagnostics["minimum_singular_value"] > 0.0


@pytest.mark.parametrize("order", CANDIDATES)
def test_hybrid_orders_are_finite_and_equal_budget(order: str) -> None:
    generator = torch.Generator().manual_seed(19)
    source = torch.eye(4) + 0.01 * torch.randn(4, 4, generator=generator)
    residual = 0.02 * torch.randn(4, 4, generator=generator)
    activation = torch.eye(4)
    train_gradient = torch.randn(4, 4, generator=generator)
    fit_gradient = torch.randn(4, 4, generator=generator)
    updated, diagnostics = fit_hybrid(
        source, residual, activation, train_gradient, fit_gradient,
        order=order, task_stages=1, incoming=1, neighbors=2, seed=23,
        trust_output_energy=0.1,
    )
    assert torch.isfinite(updated).all()
    assert diagnostics["coordinates"] == 6
    assert diagnostics["trust_energy_obeyed"] is True


def synthetic_rows() -> tuple[list[dict], list[dict], list[dict]]:
    rows: list[dict] = []
    finite: list[dict] = []
    chart: list[dict] = []
    loss_by_arm = {
        ARMS[0]: 7.1815,
        ARMS[1]: 7.1807,
        ARMS[2]: 7.1808,
        CANDIDATES[0]: 7.1800,
        CANDIDATES[1]: 7.1799,
    }
    for phase in (0, 60, 120, 180):
        for layer in (0, 3, 6, 9, 11):
            for candidate in CANDIDATES:
                chart.append({
                    "phase_start": phase, "layer": layer, "arm": candidate,
                    "trust_energy_obeyed": True, "trust_scale": 0.8,
                    "minimum_singular_value": 0.99,
                })
            for window in ("fit", "holdout"):
                for arm in ARMS:
                    is_candidate = arm in CANDIDATES
                    rows.append({
                        "phase_start": phase, "layer": layer, "window": window,
                        "arm": arm, "coordinates_per_layer": 147456,
                        "activation_output_residual_energy": 0.90 if is_candidate else 1.0,
                        "validation_gradient_predicted_ce_decrease": 0.0047 if is_candidate else 0.0035,
                    })
        for window in ("fit", "holdout"):
            for arm in ARMS:
                finite.append({
                    "phase_start": phase, "window": window,
                    "arm": arm, "loss": loss_by_arm[arm],
                })
    return rows, finite, chart


def test_aggregate_uses_frozen_order_and_fails_over() -> None:
    rows, finite, chart = synthetic_rows()
    result = aggregate_results(rows, finite, chart)
    assert result["selected"] == CANDIDATES[0]
    assert result["authorization"]["language_model_training_authorized"] is False
    for row in chart:
        if row["arm"] == CANDIDATES[0]:
            row["minimum_singular_value"] = 0.9
    result = aggregate_results(rows, finite, chart)
    assert result["selected"] == CANDIDATES[1]
