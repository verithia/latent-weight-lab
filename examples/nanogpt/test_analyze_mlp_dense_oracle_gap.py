from __future__ import annotations

import math

import torch

from examples.nanogpt.analyze_mlp_dense_oracle_gap import (
    aggregate_direction_metrics,
    classify_dense_oracle,
)


def test_aggregate_direction_metrics_reports_exact_projection_geometry() -> None:
    target = {0: torch.tensor([3.0, 4.0])}
    prediction = {0: torch.tensor([3.0, 0.0])}
    result = aggregate_direction_metrics(target, prediction)
    assert math.isclose(result["target_fro"], 5.0)
    assert math.isclose(result["prediction_fro"], 3.0)
    assert math.isclose(result["cosine"], 0.6)
    assert math.isclose(result["fixed_scale_recovery"], 0.36)
    assert math.isclose(result["positive_line_recovery"], 0.36)


def _rows(losses: dict[str, float]) -> list[dict[str, object]]:
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


def _base_losses() -> dict[str, float]:
    return {
        "production_joint": 5.0,
        "dense_historical_joint": 5.0,
        "dense_norm_joint": 5.0,
        "production_cfc": 5.0,
        "production_cproj": 5.0,
        "dense_norm_cfc": 5.0,
        "dense_norm_cproj": 5.0,
        "hybrid_norm_cfc": 5.0,
        "hybrid_norm_cproj": 5.0,
        "dense_canonical_joint": 5.0,
    }


def test_dense_oracle_decision_attributes_both_families() -> None:
    losses = _base_losses()
    for name in (
        "dense_historical_joint",
        "dense_norm_joint",
        "dense_norm_cfc",
        "dense_norm_cproj",
        "hybrid_norm_cfc",
        "hybrid_norm_cproj",
        "dense_canonical_joint",
    ):
        losses[name] = 4.9
    decision = classify_dense_oracle(_rows(losses), 2.576)
    assert decision["classification"] == "BOTH_MLP_CHARTS_LIMIT_DENSE_ORACLE"


def test_dense_oracle_decision_separates_magnitude_only() -> None:
    losses = _base_losses()
    losses["dense_historical_joint"] = 4.9
    decision = classify_dense_oracle(_rows(losses), 2.576)
    assert decision["classification"] == "DENSE_ORACLE_GAIN_IS_UPDATE_MAGNITUDE_ONLY"
