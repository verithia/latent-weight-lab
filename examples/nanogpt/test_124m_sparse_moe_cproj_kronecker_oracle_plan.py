from __future__ import annotations

import json
from pathlib import Path


PLAN = Path(__file__).parent / "configs" / "selection_artifacts" / (
    "124m_sparse_moe_cproj_kronecker_oracle_plan.json"
)


def test_plan_freezes_the_non_fixed_image_mechanism_and_budget() -> None:
    plan = json.loads(PLAN.read_text())
    assert plan["schema_version"] == "nanogpt_sparse_moe_cproj_kronecker_oracle_plan_v1"
    mechanism = plan["mechanism"]
    assert mechanism["kronecker_rank"] == 2
    assert mechanism["coordinates_per_expert"] == 4352
    assert 200 < mechanism["coordinate_compression_ratio"] < 500
    assert "no learned projection" in mechanism["latent_state"]
    assert "ordinary rank 768" in mechanism["ordinary_rank"]


def test_plan_freezes_two_fits_heldout_scoring_and_no_training() -> None:
    plan = json.loads(PLAN.read_text())
    assert len(plan["functional_protocol"]["discovery_banks"]) == 2
    assert plan["functional_protocol"]["heldout"]["tokens"] == 4096
    assert plan["fit"]["no_sweep"] is True
    assert plan["decision_rule"]["pass"].startswith("Passing both independent fits")
    assert "does not authorize model implementation" in plan["decision_rule"]["pass"]


def test_plan_requires_strict_functional_recovery() -> None:
    gates = json.loads(PLAN.read_text())["frozen_gates"]
    assert gates["heldout_recovery_mean_min"] == 0.9
    assert gates["heldout_recovery_every_layer_min"] == 0.8
    assert gates["improvement_over_parent_static_min"] == 0.1
    assert gates["heldout_bank_action_cosine_mean_min"] == 0.8
