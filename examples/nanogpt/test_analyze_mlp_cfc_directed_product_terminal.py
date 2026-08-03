import torch

from examples.nanogpt.analyze_mlp_cfc_directed_product_terminal import (
    build_candidates,
    classify,
    interpolate,
    scaled_to_dense_ratio,
)


def test_scaled_product_and_residual_candidates() -> None:
    dense = {0: torch.tensor([[3.0, 4.0]])}
    raw = {0: torch.tensor([[1.0, 0.0]])}
    scaled = scaled_to_dense_ratio(raw, dense, 0.5)
    assert torch.allclose(scaled[0], torch.tensor([[2.5, 0.0]]))
    midpoint = interpolate(scaled, dense, 0.5)
    assert torch.allclose(midpoint[0], torch.tensor([[2.75, 2.0]]))
    candidates, geometry = build_candidates(
        dense,
        raw,
        radius_ratios=[0.5, 1.0],
        residual_fractions=[0.5],
        registered_ratio=0.5,
    )
    assert list(candidates) == [
        "baseline",
        "dense_same_radius",
        "dense_full_radius",
        "product_registered",
        "product_radius_0.500000",
        "product_radius_1.000000",
        "product_plus_residual_0.500000",
    ]
    assert geometry["registered_product_metrics"]["prediction_fro"] == 2.5


def _rows(ce_by_point: dict[str, list[float]]) -> list[dict[str, object]]:
    return [
        {
            "point_id": point,
            "window": "heldout",
            "batch_index": batch_index,
            "ce": ce,
        }
        for point, values in ce_by_point.items()
        for batch_index, ce in enumerate(values)
    ]


def test_classify_direction_limited_with_nondefault_registered_ratio() -> None:
    rows = _rows(
        {
            "product_registered": [5.0, 5.0, 5.0, 5.0],
            "product_radius_0.500000": [5.0, 5.0, 5.0, 5.0],
            "product_radius_0.750000": [5.0, 5.0, 5.0, 5.0],
            "dense_same_radius": [4.9, 4.9, 4.9, 4.9],
        }
    )
    decision = classify(
        rows,
        radius_ratios=[0.5, 0.75],
        registered_ratio=0.5,
        confidence_z=2.576,
    )
    assert decision["classification"] == "RESIDUAL_DIRECTION_LIMITED"
    assert list(decision["radius_vs_registered"]) == ["0.750000"]
