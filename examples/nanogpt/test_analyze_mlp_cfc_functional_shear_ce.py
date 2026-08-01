from examples.nanogpt.analyze_mlp_cfc_functional_shear_ce import aggregate


SELECTED = "functional_mix_0p125000"


def _rows(losses: dict[str, tuple[float, float, float, float]]) -> list[dict]:
    rows = []
    for window, (fresh, weight, selected, dense) in losses.items():
        for candidate, loss in (
            ("baseline", fresh + 0.1),
            ("fresh88", fresh),
            ("fresh64_weight_shear24", weight),
            (SELECTED, selected),
            ("dense_exact", dense),
        ):
            rows.extend(
                {
                    "window": window,
                    "candidate": candidate,
                    "repeat": repeat,
                    "loss": loss,
                }
                for repeat in range(3)
            )
    return rows


def test_aggregate_promotes_consistent_additional_recovery() -> None:
    result = aggregate(
        _rows(
            {
                "validation_1": (5.2, 5.1, 5.08, 5.0),
                "validation_2": (5.3, 5.2, 5.17, 5.0),
            }
        ),
        windows=["validation_1", "validation_2"],
        selected=SELECTED,
        maximum_replicate_range=1e-7,
        minimum_recovery=0.05,
        median_recovery=0.1,
    )
    assert result["decision"] == "PROMOTE_FUNCTIONAL_COORDINATE_MIX_TO_PRODUCTION_PREFLIGHT"


def test_aggregate_rejects_one_weight_shear_regression() -> None:
    result = aggregate(
        _rows(
            {
                "validation_1": (5.2, 5.1, 5.08, 5.0),
                "validation_2": (5.3, 5.2, 5.21, 5.0),
            }
        ),
        windows=["validation_1", "validation_2"],
        selected=SELECTED,
        maximum_replicate_range=1e-7,
        minimum_recovery=0.0,
        median_recovery=0.0,
    )
    assert result["decision"] == "REJECT_FUNCTIONAL_COORDINATE_MIX_HELDOUT_CE"
    assert not result["selected_recovery_over_weight_shear"]["beats_weight_shear_every_window"]
