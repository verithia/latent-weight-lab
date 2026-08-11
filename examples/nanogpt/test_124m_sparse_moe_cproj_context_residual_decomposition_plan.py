from __future__ import annotations

import json
from pathlib import Path


PLAN = Path(__file__).parent / "configs" / "selection_artifacts" / (
    "124m_sparse_moe_cproj_context_residual_decomposition_plan.json"
)


def _plan() -> dict:
    return json.loads(PLAN.read_text(encoding="utf-8"))


def test_decomposition_is_exact_and_not_a_beta_sweep() -> None:
    decomposition = _plan()["decomposition"]
    assert "sqrt(2)*A1 - A0" in decomposition["pure_modulation"]
    assert "without a new axis, seed, or beta" in decomposition["pure_modulation"]


def test_decomposition_keeps_total_coordinate_budget() -> None:
    plan = _plan()
    assert plan["controls"]["same_total_coordinates"] is True
    assert "factor 1 through Au" in plan["decomposition"]["two_factor_384x"]
    assert "factor 2 through Au" in plan["decomposition"]["three_factor_256x"]


def test_decomposition_does_not_authorize_training() -> None:
    rule = _plan()["decision_rule"]["pass_all"].lower()
    assert "training remains unauthorized" in rule
