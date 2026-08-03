from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_cfc_directed_sparse import (
    candidate_order,
    classify,
    fit_directed_sparse_mixer,
)


def test_directed_sparse_fit_recovers_sparse_target() -> None:
    torch.manual_seed(23)
    source = torch.eye(16)
    target = torch.stack(
        [
            0.7 * source[:, (index + 1) % 16]
            - 0.2 * source[:, (index + 3) % 16]
            for index in range(16)
        ],
        dim=1,
    )
    prediction, row = fit_directed_sparse_mixer(
        source, target, incoming=2, ridge_ratio=1e-8, chunk_size=4
    )
    assert float((prediction - target).norm() / target.norm()) < 1e-4
    assert row["coordinates"] == 32


def _rows(single44: float, hybrid44: float) -> list[dict[str, object]]:
    offsets = {
        "baseline": 0.0,
        "production_cfc": -0.001,
        "production_cproj": -0.002,
        "production_joint": -0.003,
        "dense_norm_cfc": -0.002,
        "hybrid_norm_cfc": -0.004,
        "directed44_cfc": single44,
        "hybrid_directed44_cfc": hybrid44,
        "directed88_cfc": -0.0018,
        "hybrid_directed88_cfc": -0.0038,
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


def test_candidate_order_is_capacity_nested() -> None:
    assert candidate_order([44, 88])[-4:] == [
        "directed44_cfc",
        "hybrid_directed44_cfc",
        "directed88_cfc",
        "hybrid_directed88_cfc",
    ]


def test_classify_selects_smallest_passing_directed_chart() -> None:
    result = classify(
        _rows(-0.0016, -0.0036),
        [44, 88],
        confidence_z=2.576,
        minimum_fraction=0.25,
        mean_fraction=0.4,
    )
    assert result["classification"] == "DIRECTED_SPARSE_CFC_PASSES"
    assert result["selected_incoming_per_target"] == 44


def test_classify_rejects_missing_hybrid_gain() -> None:
    result = classify(
        _rows(-0.0016, -0.003),
        [44, 88],
        confidence_z=2.576,
        minimum_fraction=0.25,
        mean_fraction=0.4,
    )
    assert result["selected_incoming_per_target"] == 88
