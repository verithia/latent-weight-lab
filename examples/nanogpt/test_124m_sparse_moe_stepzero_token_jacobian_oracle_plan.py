from __future__ import annotations

import json
from pathlib import Path


PLAN = Path(__file__).parent / "configs" / "selection_artifacts" / (
    "124m_sparse_moe_stepzero_token_jacobian_oracle_plan.json"
)


def _plan() -> dict:
    return json.loads(PLAN.read_text(encoding="utf-8"))


def test_jacobian_plan_is_token_compute_and_coordinate_matched() -> None:
    plan = _plan()
    for bank in plan["gradient_banks"]:
        assert bank["coordinates"] == 4
        assert bank["batches"] * bank["batch_size"] * bank["block_size"] == 2048
    assert [row["compression_ratio"] for row in plan["coordinate_families"]] == [
        383.8750813272609,
        256.0,
    ]


def test_jacobian_plan_separates_discovery_and_heldout() -> None:
    plan = _plan()
    seeds = [bank["seed"] for bank in plan["gradient_banks"]]
    assert seeds == [20260911, 20260912]
    assert plan["evaluation"]["heldout_seed"] not in seeds
    assert "endpoint tensor" in plan["jacobian_basis"]["anti_leakage"]


def test_jacobian_plan_does_not_authorize_training() -> None:
    authorization = _plan()["interpretation"]["authorization"].lower()
    assert "only this zero-update offline oracle" in authorization
    assert "generated-expert training" in authorization
    assert "remain prohibited" in authorization
