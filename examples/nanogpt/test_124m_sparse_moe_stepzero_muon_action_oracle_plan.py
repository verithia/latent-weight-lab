from __future__ import annotations

import json
from pathlib import Path


PLAN = Path(__file__).parent / "configs" / "selection_artifacts" / (
    "124m_sparse_moe_stepzero_muon_action_oracle_plan.json"
)


def _plan() -> dict:
    return json.loads(PLAN.read_text(encoding="utf-8"))


def test_muon_action_plan_is_matched_to_raw_gradient_control() -> None:
    plan = _plan()
    assert [bank["seed"] for bank in plan["gradient_banks"]] == [20260911, 20260912]
    assert plan["evaluation"]["heldout_seed"] == 20260913
    assert "single_change" in plan["controls"]
    assert plan["controls"]["raw_gradient_seal_sha256"] == (
        "577cdd7a526c981808afeab11466f61ae626fd0d2690bf87e234813c15f89380"
    )


def test_muon_action_plan_uses_production_optimizer_geometry() -> None:
    plan = _plan()
    action = plan["optimizer_action"]
    assert "muon_update_batched" in action["expert_matrices"]
    assert "five Newton-Schulz steps" in action["expert_matrices"]
    assert "-g/(abs(g)+1e-8)" in action["router"]


def test_muon_action_plan_does_not_authorize_training() -> None:
    authorization = _plan()["interpretation"]["authorization"].lower()
    assert "only this zero-update offline oracle" in authorization
    assert "generated-expert training" in authorization
    assert "remain prohibited" in authorization
