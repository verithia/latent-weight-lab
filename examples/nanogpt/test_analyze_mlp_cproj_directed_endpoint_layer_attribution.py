from __future__ import annotations

from examples.nanogpt.analyze_mlp_cproj_directed_endpoint_layer_attribution import (
    classify_attribution,
)


LAYERS = [0, 3, 6, 9, 11]
SEEDS = [101, 102]


def _rows(
    *,
    all_gains: tuple[float, float],
    layer_gains: dict[int, tuple[float, float]],
) -> list[dict[str, object]]:
    control = {"variant": "output32_all", "val_ce_seed_101": 5.0, "val_ce_seed_102": 5.1}
    candidate = {
        "variant": "directed16_all",
        "val_ce_seed_101": 5.0 - all_gains[0],
        "val_ce_seed_102": 5.1 - all_gains[1],
    }
    rows: list[dict[str, object]] = [control, candidate]
    for layer in LAYERS:
        gains = layer_gains[layer]
        rows.append(
            {
                "variant": f"directed16_layer_{layer}",
                "val_ce_seed_101": 5.0 - gains[0],
                "val_ce_seed_102": 5.1 - gains[1],
            }
        )
    return rows


def _classify(rows: list[dict[str, object]]) -> dict[str, object]:
    return classify_attribution(
        rows,
        layers=LAYERS,
        seeds=SEEDS,
        minimum_mean_gain=0.001,
    )


def test_distributed_signal_requires_four_consistent_layers() -> None:
    result = _classify(
        _rows(
            all_gains=(0.002, 0.0021),
            layer_gains={
                0: (0.0005, 0.0005),
                3: (0.0004, 0.0004),
                6: (0.0003, 0.0003),
                9: (0.0002, 0.0002),
                11: (-0.0001, -0.0001),
            },
        )
    )
    assert result["classification"] == "DISTRIBUTED_DIRECTED_FUNCTIONAL_SIGNAL"
    assert result["consistent_improving_layers"] == [0, 3, 6, 9]


def test_top_two_concentration_classifies_layer_local_signal() -> None:
    result = _classify(
        _rows(
            all_gains=(0.002, 0.002),
            layer_gains={
                0: (0.0010, 0.0010),
                3: (0.0008, 0.0008),
                6: (0.0001, 0.0001),
                9: (0.0001, 0.0001),
                11: (0.0001, 0.0001),
            },
        )
    )
    assert result["classification"] == "LAYER_LOCAL_DIRECTED_FUNCTIONAL_SIGNAL"
    assert result["top_two_positive_gain_fraction"] > 0.75


def test_nonreplicated_gain_is_functionally_null() -> None:
    result = _classify(
        _rows(
            all_gains=(0.0012, 0.0009),
            layer_gains={layer: (0.0002, 0.0002) for layer in LAYERS},
        )
    )
    assert result["classification"] == "DIRECTED_ENDPOINT_FUNCTIONALLY_NULL"
    assert result["authorization"]["language_model_training"] is False


def test_intermediate_consistency_classifies_mixed_signal() -> None:
    result = _classify(
        _rows(
            all_gains=(0.002, 0.002),
            layer_gains={
                0: (0.0004, 0.0004),
                3: (0.0004, 0.0004),
                6: (0.0004, 0.0004),
                9: (0.0002, -0.0001),
                11: (0.0001, -0.0001),
            },
        )
    )
    assert result["classification"] == "MIXED_DIRECTED_FUNCTIONAL_SIGNAL"


def test_large_nonadditivity_is_reported_without_changing_primary_class() -> None:
    result = _classify(
        _rows(
            all_gains=(0.003, 0.003),
            layer_gains={layer: (0.0001, 0.0001) for layer in LAYERS},
        )
    )
    assert result["nonadditive_context"] is True
