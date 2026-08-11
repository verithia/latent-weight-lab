from __future__ import annotations

import json
from pathlib import Path

from examples.nanogpt.analyze_sparse_moe_paired_alignment import file_sha256
from examples.nanogpt.analyze_sparse_moe_stepzero_kfac_maxbudget_oracle import (
    validate_plan,
)


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "examples/nanogpt/configs/selection_artifacts/124m_sparse_moe_stepzero_kfac_maxbudget_oracle_plan.json"


def _plan() -> dict:
    return json.loads(PLAN.read_text(encoding="utf-8"))


def test_plan_identity_and_helpers_are_frozen() -> None:
    plan = _plan()
    validate_plan(plan, PLAN)
    identity = plan["identity"]
    assert identity["preregistration_git_commit"].startswith("d1e1f10")
    assert identity["entrypoint_sha256"] == file_sha256(
        ROOT / identity["entrypoint"]
    )


def test_only_the_output_rank_changes() -> None:
    families = _plan()["families"]
    control = families["control_separate_3_plus_3"]
    candidate = families["candidate_separate_3_plus_4"]
    assert candidate["incoming_rank"] == control["incoming_rank"] == 3
    assert candidate["router_rank"] == control["router_rank"] == 3
    assert control["outgoing_rank"] == 3
    assert candidate["outgoing_rank"] == 4
    assert candidate["paired_coordinate_compression_ratio"] >= 200.0


def test_oracle_cannot_authorize_dense_basis_or_training() -> None:
    authorization = _plan()["authorization"]
    assert authorization["zero_update_maxbudget_oracle"]
    assert not authorization["structured_basis_approximation_preregistration"]
    assert not authorization["dense_or_lora_basis"]
    assert not authorization["production_implementation"]
    assert not authorization["mfu_preflight"]
    assert not authorization["language_model_training"]
    assert not authorization["automatic_retry_or_sweep"]
