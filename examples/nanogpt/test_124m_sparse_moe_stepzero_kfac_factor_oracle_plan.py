from __future__ import annotations

import json
from pathlib import Path


PLAN = Path(__file__).parent / "configs" / "selection_artifacts" / (
    "124m_sparse_moe_stepzero_kfac_factor_oracle_plan.json"
)


def _plan() -> dict:
    return json.loads(PLAN.read_text(encoding="utf-8"))


def test_kfac_plan_uses_discovery_only_activation_error_factors() -> None:
    basis = _plan()["kfac_basis"]
    assert "mean_j(dh_j^2)" in basis["incoming_factor"]
    assert "mean_j(h_j^2)" in basis["outgoing_factor"]
    assert "terminal parameter chord" in basis["anti_leakage"]


def test_kfac_plan_preserves_matched_budget_and_splits() -> None:
    plan = _plan()
    assert [bank["tokens"] for bank in plan["geometry_banks"]] == [2048, 2048]
    assert [bank["seed"] for bank in plan["geometry_banks"]] == [20260911, 20260912]
    assert plan["evaluation"]["heldout_seed"] == 20260913
    assert [row["compression_ratio"] for row in plan["coordinate_families"]] == [
        383.8750813272609,
        256.0,
    ]


def test_kfac_plan_does_not_authorize_training() -> None:
    authorization = _plan()["interpretation"]["authorization"].lower()
    assert "only this zero-update offline oracle" in authorization
    assert "generated-expert training" in authorization
    assert "remain prohibited" in authorization
