from __future__ import annotations

import torch

from examples.nanogpt.analyze_sparse_moe_cfc_same_gauge_tangent_decomposition import (
    classify,
    comparison_metrics,
)


THRESHOLDS = {
    "global_rank7_cross_projection_mean_min_both_directions": 0.75,
    "global_rank7_cross_projection_minimum_layer_min_both_directions": 0.6,
    "global_rank7_cross_projection_minimum_row_min_both_directions": 0.4,
    "direct_corresponding_gradient_cosine_mean_min": 0.5,
    "direct_corresponding_gradient_cosine_minimum_layer_min": 0.35,
}


def test_identical_rank_seven_cells_pass() -> None:
    torch.manual_seed(9)
    basis, _ = torch.linalg.qr(torch.randn(32, 7))
    rows = torch.randn(3, 8, 7) @ basis.T
    rows = rows / rows.norm(dim=-1, keepdim=True)
    metrics = comparison_metrics(rows, rows, rank=7, thresholds=THRESHOLDS)
    assert metrics["gates"]["all_pass"]


def test_orthogonal_cells_fail() -> None:
    left = torch.zeros(3, 8, 16)
    right = torch.zeros(3, 8, 16)
    for layer in range(3):
        left[layer, :, :8] = torch.eye(8)
        right[layer, :, 8:] = torch.eye(8)
    metrics = comparison_metrics(left, right, rank=7, thresholds=THRESHOLDS)
    assert not metrics["gates"]["all_pass"]


def test_classification_distinguishes_data_from_endpoint_effect() -> None:
    assert classify(False, False) == "ACTIVATION_CONDITIONED_TANGENT_INSTABILITY"
    assert classify(True, False) == "NONLINEAR_ENDPOINT_CHART_DRIFT"
    assert classify(True, True) == "SAME_GAUGE_TANGENT_STABLE"
