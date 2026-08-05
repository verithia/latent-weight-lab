from __future__ import annotations

import math

import pytest

from examples.nanogpt.analyze_mlp_cproj_directed_product_endpoint_fixed_eval import (
    CANDIDATE,
    CONTROL,
    select_directed_endpoint,
)


def _rows(
    *,
    dense_train: float = 3.10,
    dense_val: float = 3.11,
    control_train: float = 3.108,
    control_val: float = 3.119,
    candidate_train: float = 3.104,
    candidate_val: float = 3.114,
) -> list[dict[str, object]]:
    return [
        {"variant": "dense_endpoint", "train_ce": dense_train, "val_ce": dense_val},
        {
            "variant": "hidden88_full_carry",
            "train_ce": 3.13,
            "val_ce": 3.14,
        },
        {"variant": CONTROL, "train_ce": control_train, "val_ce": control_val},
        {
            "variant": CANDIDATE,
            "train_ce": candidate_train,
            "val_ce": candidate_val,
        },
    ]


def _select(rows: list[dict[str, object]]) -> dict[str, object]:
    return select_directed_endpoint(
        rows,
        minimum_val_improvement=0.002,
        maximum_val_gap_to_dense=0.0046,
    )


def test_directed_endpoint_pass_authorizes_only_implementation_and_mfu() -> None:
    result = _select(_rows())
    assert result["passed"] is True
    assert result["decision"] == "DIRECTED16_ENDPOINT_TASK_LOSS_PASS"
    assert result["authorization"] == {
        "production_implementation": True,
        "exact_config_mfu_preflight": True,
        "language_model_training": False,
    }


@pytest.mark.parametrize(
    ("kwargs", "failed_gate"),
    [
        (
            {"candidate_val": 3.118},
            "validation_ce_improvement_over_output32",
        ),
        (
            {"candidate_train": 3.109},
            "train_ce_no_worse_than_output32",
        ),
        (
            {"candidate_val": 3.115},
            "validation_ce_gap_to_dense",
        ),
        (
            {"candidate_val": math.nan},
            "all_finite",
        ),
    ],
)
def test_directed_endpoint_rejects_each_failed_gate(
    kwargs: dict[str, float], failed_gate: str
) -> None:
    result = _select(_rows(**kwargs))
    assert result["passed"] is False
    assert result["gates"][failed_gate] is False
    assert result["authorization"]["production_implementation"] is False
    assert result["authorization"]["language_model_training"] is False


def test_variant_inventory_is_frozen() -> None:
    rows = _rows()
    rows.pop()
    with pytest.raises(ValueError, match="frozen variant set"):
        _select(rows)
