from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_fixed_radius_capacity import (
    candidate_names,
    classify_capacity,
    normalize_family_to_radius,
    quantized_update,
)


def test_quantized_update_matches_materialized_endpoint() -> None:
    base = torch.tensor([[1.0, -0.5]], dtype=torch.bfloat16)
    update = torch.tensor([[0.007, -0.013]], dtype=torch.float32)
    delta = quantized_update(base, update)
    expected = (base.float() + update).to(torch.bfloat16).float() - base.float()
    torch.testing.assert_close(delta, expected, rtol=0.0, atol=0.0)


def test_family_normalization_matches_requested_radius() -> None:
    bases = {
        0: torch.randn(8, 8).to(torch.bfloat16),
        1: torch.randn(8, 8).to(torch.bfloat16),
    }
    raw = {0: torch.randn(8, 8) * 0.01, 1: torch.randn(8, 8) * 0.01}
    normalized, row = normalize_family_to_radius(bases, raw, 0.05)
    assert set(normalized) == {0, 1}
    assert row["relative_radius_error"] < 0.02


def _rows() -> list[dict[str, object]]:
    levels = [40, 64]
    names = candidate_names(levels)
    offsets = {name: 0.0 for name in names}
    offsets.update(
        {
            "production_cfc": -0.001,
            "production_cproj": -0.002,
            "production_joint": -0.003,
            "dense_norm_cfc": -0.002,
            "dense_norm_cproj": -0.003,
            "dense_norm_joint": -0.004,
            "hybrid_norm_cfc": -0.004,
            "hybrid_norm_cproj": -0.004,
            "cfc40_only": -0.0016,
            "cproj40_only": -0.0026,
            "hybrid_cfc40": -0.0036,
            "hybrid_cproj40": -0.0036,
            "joint40": -0.0036,
            "cfc64_only": -0.0018,
            "cproj64_only": -0.0028,
            "hybrid_cfc64": -0.0038,
            "hybrid_cproj64": -0.0038,
            "joint64": -0.0038,
        }
    )
    return [
        {
            "window": f"window_{window}",
            "batch_index": batch,
            "point_id": name,
            "ce": 6.0 + offsets[name],
        }
        for window in (1, 2)
        for batch in range(8)
        for name in names
    ]


def test_classification_selects_smallest_passing_capacity() -> None:
    result = classify_capacity(
        _rows(),
        [40, 64],
        confidence_z=2.576,
        minimum_fraction=0.25,
        mean_fraction=0.4,
    )
    assert result["classification"] == "SAME_TOPOLOGY_CAPACITY_PASSES_FIXED_RADIUS"
    assert result["selected_residual_stages"] == 40
