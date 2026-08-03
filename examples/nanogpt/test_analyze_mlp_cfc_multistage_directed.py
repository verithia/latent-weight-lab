from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_cfc_multistage_directed import (
    candidate_order,
    classify,
    fit_multistage_directed_sparse_mixer,
)
from examples.nanogpt.analyze_mlp_cfc_product_directed import (
    fit_product_directed_sparse_mixer,
)


def test_three_stage_improves_two_stage_sparse_residual_fit() -> None:
    source = torch.eye(12)
    target = torch.stack(
        [
            0.7 * source[:, (index + 1) % 12]
            - 0.2 * source[:, (index + 3) % 12]
            + 0.1 * source[:, (index + 5) % 12]
            for index in range(12)
        ], dim=1,
    )
    two_stage, _ = fit_product_directed_sparse_mixer(
        source, target, incoming_per_stage=1, ridge_ratio=1e-8, chunk_size=4
    )
    three_stage, row = fit_multistage_directed_sparse_mixer(
        source, target, incoming_schedule=[1, 1, 1],
        ridge_ratio=1e-8, chunk_size=4,
    )
    assert float((three_stage - target).norm()) < float((two_stage - target).norm())
    assert row["coordinates"] == 36
    assert row["incoming_total_per_target"] == 3


def _rows(three44: float, three44_hybrid: float) -> list[dict[str, object]]:
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
        "product22x2_cfc": -0.0013,
        "hybrid_product22x2_cfc": -0.0033,
        "product44x2_cfc": -0.0014,
        "hybrid_product44x2_cfc": -0.0034,
        "product44totalx3_cfc": three44,
        "hybrid_product44totalx3_cfc": three44_hybrid,
        "product88totalx3_cfc": -0.0017,
        "hybrid_product88totalx3_cfc": -0.0037,
    }
    return [
        {"window": f"window_{window}", "batch_index": batch,
         "point_id": point, "ce": 6.0 + offset}
        for window in (1, 2) for batch in range(32)
        for point, offset in offsets.items()
    ]


def test_candidate_order_keeps_three_stage_candidates_last() -> None:
    assert candidate_order([44, 88])[-4:] == [
        "product44totalx3_cfc", "hybrid_product44totalx3_cfc",
        "product88totalx3_cfc", "hybrid_product88totalx3_cfc",
    ]


def test_classify_selects_same_budget_three_stage_when_it_passes() -> None:
    result = classify(
        _rows(-0.0016, -0.0036), [44, 88], confidence_z=2.576,
        minimum_fraction=0.25, mean_fraction=0.4,
    )
    assert result["classification"] == "THREE_STAGE_DIRECTED_CFC_PASSES"
    assert result["selected_incoming_total_per_target"] == 44


def test_classify_falls_back_to_double_budget_three_stage() -> None:
    result = classify(
        _rows(-0.0011, -0.0031), [44, 88], confidence_z=2.576,
        minimum_fraction=0.25, mean_fraction=0.4,
    )
    assert result["selected_incoming_total_per_target"] == 88
