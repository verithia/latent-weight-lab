from __future__ import annotations

import math

import torch

from examples.nanogpt.analyze_mlp_joint_block_output_metric import (
    _accumulate_gram,
    decide_metric,
    normalize_coefficients,
    solve_metric_coefficients,
)


def test_gram_accumulator_accepts_outputs_on_accumulator_device() -> None:
    gram = torch.zeros((2, 2), dtype=torch.float64)
    base = torch.ones((2, 3), dtype=torch.float32)
    _accumulate_gram(gram, base, base + 1.0, base + 2.0)
    assert torch.allclose(
        gram,
        torch.tensor([[1.0, 2.0], [2.0, 4.0]], dtype=torch.float64),
    )


def test_metric_coefficients_preserve_materialized_update_budget() -> None:
    result = solve_metric_coefficients(
        [[2.0, 0.8], [0.8, 1.0]],
        [-0.4, -0.3],
        3.0,
        4.0,
        100.0,
    )
    for name in ("full_constant_budget", "diagonal_constant_budget"):
        cfc, cproj = result[name]
        assert math.isclose(
            math.sqrt((3.0 * cfc) ** 2 + (4.0 * cproj) ** 2),
            5.0,
            rel_tol=1e-12,
        )


def _point(name: str):
    return {"point_id": name, "cfc_scale": 1.0, "cproj_scale": 1.0}


def _rows(losses: dict[str, float]):
    return [
        {
            "window": window,
            "batch_index": index,
            "point_id": point,
            "ce": loss,
        }
        for window in ("window_1", "window_2")
        for index in range(8)
        for point, loss in losses.items()
    ]


def test_metric_decision_requires_full_to_beat_production_and_diagonal() -> None:
    points = {
        "production": _point("production"),
        "full_metric": _point("full"),
        "diagonal_metric": _point("diagonal"),
    }
    decision = decide_metric(
        _rows({"production": 5.0, "full": 4.98, "diagonal": 4.99}),
        points,
        2.576,
    )
    assert decision["classification"] == "FULL_2X2_BLOCK_OUTPUT_METRIC_SUPPORTED"
    assert decision["next_action"] == (
        "IMPLEMENT_AND_PERFORMANCE_GATE_COUPLED_2X2_PRECONDITIONER"
    )
