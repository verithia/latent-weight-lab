from __future__ import annotations

import json
from pathlib import Path


PLAN = Path(__file__).parent / "configs" / "selection_artifacts" / (
    "124m_sparse_moe_stepzero_task_gradient_oracle_plan.json"
)


def _plan() -> dict:
    return json.loads(PLAN.read_text(encoding="utf-8"))


def test_task_gradient_plan_preserves_coordinate_budget() -> None:
    plan = _plan()
    families = {row["name"]: row for row in plan["coordinate_families"]}
    assert families["coupled_four"]["coordinates_per_layer"] == 49184
    assert 200 <= families["coupled_four"]["compression_ratio"] <= 500
    assert families["separate_three_plus_three"]["coordinates_per_layer"] == 73752
    assert 200 <= families["separate_three_plus_three"]["compression_ratio"] <= 500


def test_task_gradient_plan_has_disjoint_banks_and_heldout_frame() -> None:
    plan = _plan()
    seeds = [bank["seed"] for bank in plan["gradient_banks"]]
    assert len(seeds) == len(set(seeds)) == 2
    assert plan["evaluation"]["heldout_seed"] not in seeds
    assert all(bank["independent_microbatches"] == 4 for bank in plan["gradient_banks"])


def test_task_gradient_plan_does_not_authorize_training() -> None:
    plan = _plan()
    authorization = plan["interpretation"]["authorization"].lower()
    assert "only the zero-update offline oracle" in authorization
    assert "does not authorize generated-expert training" in authorization
