from examples.nanogpt.analyze_mlp_cfc_task_shear_ce import aggregate


def _rows(losses: dict[str, tuple[float, float, float]]) -> list[dict]:
    rows = []
    for window, (fresh, selected, dense) in losses.items():
        for candidate, loss in (
            ("baseline", fresh + 0.1),
            ("fresh88", fresh),
            ("fresh64_shear24", selected),
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


def test_aggregate_promotes_consistent_recovery() -> None:
    result = aggregate(
        _rows(
            {
                "validation_1": (5.1, 5.08, 5.0),
                "validation_2": (5.2, 5.18, 5.0),
            }
        ),
        windows=["validation_1", "validation_2"],
        maximum_replicate_range=1e-7,
        minimum_recovery=0.05,
        median_recovery=0.1,
    )
    assert result["decision"] == "PROMOTE_TASK_MATCHED_SHEAR_TO_PRODUCTION_PREFLIGHT"
    assert result["selected_recovery"]["minimum"] >= 0.1


def test_aggregate_rejects_one_negative_window() -> None:
    result = aggregate(
        _rows(
            {
                "validation_1": (5.1, 5.08, 5.0),
                "validation_2": (5.2, 5.21, 5.0),
            }
        ),
        windows=["validation_1", "validation_2"],
        maximum_replicate_range=1e-7,
        minimum_recovery=0.0,
        median_recovery=0.0,
    )
    assert result["decision"] == "REJECT_TASK_MATCHED_SHEAR_HELDOUT_CE"
    assert not result["selected_recovery"]["beats_fresh_every_window"]


def test_aggregate_rejects_invalid_dense_reference() -> None:
    result = aggregate(
        _rows({"validation_1": (5.0, 4.99, 5.01)}),
        windows=["validation_1"],
        maximum_replicate_range=1e-7,
        minimum_recovery=0.0,
        median_recovery=0.0,
    )
    assert result["decision"] == "TASK_SHEAR_CE_DENSE_REFERENCE_INVALID"
