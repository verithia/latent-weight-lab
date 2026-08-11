from __future__ import annotations

import json
from pathlib import Path

from examples.nanogpt.analyze_sparse_moe_cfc_context_residual_action_audit import (
    validate_plan,
)
from examples.nanogpt.analyze_sparse_moe_paired_alignment import file_sha256


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "examples/nanogpt/configs/selection_artifacts/124m_sparse_moe_cfc_context_residual_action_audit_plan.json"


def _plan() -> dict:
    return json.loads(PLAN.read_text(encoding="utf-8"))


def test_plan_identity_and_helper_hashes_are_frozen() -> None:
    plan = _plan()
    validate_plan(plan, PLAN)
    identity = plan["identity"]
    assert identity["preregistration_git_commit"].startswith("e10eff6")
    assert identity["entrypoint_sha256"] == file_sha256(
        ROOT / identity["entrypoint"]
    )


def test_audit_reuses_frozen_parent_without_fitting() -> None:
    plan = _plan()
    source = plan["source"]
    operator = plan["frozen_operator"]
    assert source["checkpoint_updates"] == 0
    assert operator["static_beta"] == 0.0
    assert operator["context_beta"] == 1.0
    assert "unchanged" in operator["coordinates"]
    assert "optimizer" in operator["coordinates"]
    assert plan["causal_reconciliation"]["no_new_fit"].startswith(
        "This audit loads the already fitted static coordinates"
    )


def test_pass_only_authorizes_direct_sum_preregistration() -> None:
    authorization = _plan()["authorization"]
    assert authorization["zero_update_residual_action_audit"]
    assert not authorization["direct_sum_preregistration"]
    assert not authorization["new_coordinate_fit"]
    assert not authorization["production_implementation"]
    assert not authorization["mfu_preflight"]
    assert not authorization["language_model_training"]
    assert not authorization["larger_rung"]
    assert not authorization["automatic_retry_or_sweep"]
