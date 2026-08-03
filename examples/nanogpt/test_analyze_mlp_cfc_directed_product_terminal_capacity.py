from examples.nanogpt.analyze_mlp_cfc_directed_product_terminal_capacity import (
    classify_capacity,
    schedule_coordinates,
)


def _row(
    seed: int,
    name: str,
    recovery: float,
    minimum_layer: float,
    coordinates: int,
) -> dict[str, object]:
    return {
        "gradient_seed": seed,
        "candidate": name,
        "positive_line_recovery": recovery,
        "minimum_layer_positive_line_recovery": minimum_layer,
        "coordinates_per_layer": coordinates,
        "radius_ratio_absolute_error": 1e-9,
    }


def test_schedule_coordinates() -> None:
    assert schedule_coordinates([30, 29, 29], 3072) == 270336


def test_classify_selects_smallest_candidate_passing_every_seed() -> None:
    rows = []
    for seed in [1, 2, 3]:
        rows.extend(
            [
                _row(seed, "current", 0.50, 0.49, 270336),
                _row(seed, "small", 0.66, 0.64, 405504),
                _row(seed, "large", 0.70, 0.68, 540672),
            ]
        )
    decision = classify_capacity(
        rows,
        current_name="current",
        maximum_coordinates_per_layer=540672,
        minimum_positive_line_recovery=0.65,
        minimum_layer_positive_line_recovery=0.62,
        minimum_improvement_over_current=0.12,
        maximum_radius_error=1e-7,
    )
    assert decision["classification"] == "TERMINAL_COMPOSITIONAL_CAPACITY_PASSES"
    assert decision["selected_candidate"] == "small"


def test_classify_rejects_candidate_with_one_weak_gradient() -> None:
    rows = [
        _row(1, "current", 0.50, 0.49, 270336),
        _row(2, "current", 0.51, 0.50, 270336),
        _row(1, "candidate", 0.67, 0.65, 405504),
        _row(2, "candidate", 0.61, 0.60, 405504),
    ]
    decision = classify_capacity(
        rows,
        current_name="current",
        maximum_coordinates_per_layer=540672,
        minimum_positive_line_recovery=0.65,
        minimum_layer_positive_line_recovery=0.62,
        minimum_improvement_over_current=0.12,
        maximum_radius_error=1e-7,
    )
    assert decision["classification"] == "TERMINAL_COMPOSITIONAL_CAPACITY_REJECTED"
    assert decision["selected_candidate"] is None
