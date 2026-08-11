from __future__ import annotations

import json
from pathlib import Path


PLAN = Path(__file__).parent / "configs" / "selection_artifacts" / (
    "124m_sparse_moe_cproj_context_modulated_fht_oracle_plan.json"
)


def _plan() -> dict:
    return json.loads(PLAN.read_text(encoding="utf-8"))


def test_context_modulated_plan_rotates_image_without_error_at_inference() -> None:
    mechanism = _plan()["mechanism"]
    assert "G_beta(h)" in mechanism["gate"]
    assert "linear in trainable z" in mechanism["properties"]
    assert "no labels or errors at inference" in mechanism["properties"]
    assert mechanism["fixed_operator_seed"] == 20260925


def test_context_modulated_plan_preserves_200_to_500x_budget() -> None:
    families = _plan()["coordinate_families"]
    assert [row["factors"] for row in families] == [2, 3]
    assert [row["c_proj_compression_ratio"] for row in families] == [384.0, 256.0]


def test_context_modulated_plan_has_matched_static_ablation() -> None:
    plan = _plan()
    assert "identical-coordinate" in plan["ablations_and_controls"]["beta_zero"]
    assert plan["frozen_gates"]["dynamic_minus_beta_zero_mean_min_each_bank"] == 0.1
    assert "two-probe Rademacher" in plan["evaluation"]["ridge_trace_estimator"]


def test_context_modulated_plan_does_not_authorize_training() -> None:
    rule = _plan()["decision_rule"]["pass"].lower()
    assert "does not authorize training" in rule
    assert "multi-phase shadow rollout" in rule
