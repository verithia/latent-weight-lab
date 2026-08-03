from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_cfc_directed_sparse import (
    fit_directed_sparse_mixer,
)
from examples.nanogpt.analyze_mlp_cfc_product_directed import (
    candidate_order,
    classify,
    fit_product_directed_sparse_mixer,
)


def test_product_directed_recovers_two_sparse_residuals() -> None:
    source = torch.eye(12)
    target = torch.stack(
        [
            0.7 * source[:, (index + 1) % 12]
            - 0.2 * source[:, (index + 3) % 12]
            for index in range(12)
        ],
        dim=1,
    )
    one_stage, _ = fit_directed_sparse_mixer(
        source,
        target,
        incoming=1,
        ridge_ratio=1e-8,
        chunk_size=4,
    )
    prediction, row = fit_product_directed_sparse_mixer(
        source,
        target,
        incoming_per_stage=1,
        ridge_ratio=1e-8,
        chunk_size=4,
    )
    assert float((prediction - target).norm()) < float((one_stage - target).norm())
    assert row["coordinates"] == 24
    assert row["target_recovery"] > 0.95


def _rows(product22: float, product22_hybrid: float) -> list[dict[str, object]]:
    offsets = {
        "baseline": 0.0,
        "production_cfc": -0.001,
        "production_cproj": -0.002,
        "production_joint": -0.003,
        "dense_norm_cfc": -0.002,
        "hybrid_norm_cfc": -0.004,
        "directed44_cfc": -0.0012,
        "hybrid_directed44_cfc": -0.0032,
        "directed88_cfc": -0.0013,
        "hybrid_directed88_cfc": -0.0033,
        "product22x2_cfc": product22,
        "hybrid_product22x2_cfc": product22_hybrid,
        "product44x2_cfc": -0.0017,
        "hybrid_product44x2_cfc": -0.0037,
    }
    return [
        {
            "window": f"window_{window}",
            "batch_index": batch,
            "point_id": point,
            "ce": 6.0 + offset,
        }
        for window in (1, 2)
        for batch in range(32)
        for point, offset in offsets.items()
    ]


def test_candidate_order_keeps_linear_controls_before_products() -> None:
    assert candidate_order([22, 44])[-4:] == [
        "product22x2_cfc",
        "hybrid_product22x2_cfc",
        "product44x2_cfc",
        "hybrid_product44x2_cfc",
    ]


def test_classify_selects_same_budget_product_when_it_passes() -> None:
    result = classify(
        _rows(-0.0016, -0.0036),
        [22, 44],
        confidence_z=2.576,
        minimum_fraction=0.25,
        mean_fraction=0.4,
    )
    assert result["classification"] == "PRODUCT_DIRECTED_CFC_PASSES"
    assert result["selected_incoming_per_stage_and_target"] == 22


def test_classify_falls_back_to_double_budget_product() -> None:
    result = classify(
        _rows(-0.0011, -0.0031),
        [22, 44],
        confidence_z=2.576,
        minimum_fraction=0.25,
        mean_fraction=0.4,
    )
    assert result["selected_incoming_per_stage_and_target"] == 44
