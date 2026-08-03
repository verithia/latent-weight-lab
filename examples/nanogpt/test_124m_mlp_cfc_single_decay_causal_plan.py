from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PLAN = (
    REPO
    / "examples/nanogpt/configs/selection_artifacts/124m_mlp_cfc_single_decay_causal_plan.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_single_decay_plan_binds_all_repository_inputs() -> None:
    plan = json.loads(PLAN.read_text())
    assert plan["schema_version"] == "124m_mlp_cfc_single_decay_causal_plan_v1"
    frozen = plan["frozen_inputs"]
    for path_key, sha_key in (
        ("old_config", "old_config_sha256"),
        ("old_training_result", "old_training_result_sha256"),
        ("parent_replay_result", "parent_replay_result_sha256"),
    ):
        assert sha256(REPO / frozen[path_key]) == frozen[sha_key]
    implementation = plan["implementation"]
    assert sha256(REPO / implementation["config"]) == implementation["config_sha256"]
    assert sha256(REPO / "examples/nanogpt/muon_matched_givens.py") == (
        implementation["muon_matched_givens_sha256"]
    )
    assert sha256(REPO / implementation["stability_validator"]) == (
        implementation["stability_validator_sha256"]
    )


def test_single_decay_config_changes_only_registered_causal_metadata() -> None:
    plan = json.loads(PLAN.read_text())
    old = json.loads((REPO / plan["frozen_inputs"]["old_config"]).read_text())
    new = json.loads((REPO / plan["implementation"]["config"]).read_text())
    allowed = {
        "candidate_scope",
        "hpo_stage",
        "implementation_commit",
        "implementation_source_hashes",
        "implementation_test_evidence",
        "instability_policy",
        "ladder_role",
        "mfu_preflight_certificate",
        "monitoring_policy",
        "out_dir",
        "prelaunch_provenance_requirements",
        "preregistered_decision_rule",
        "selection_endpoint",
    }
    assert {key for key in old if old[key] != new[key]} == allowed
    old_hashes = old["implementation_source_hashes"]
    new_hashes = new["implementation_source_hashes"]
    changed_sources = {
        key for key in old_hashes if old_hashes[key] != new_hashes[key]
    }
    assert changed_sources == {"examples/nanogpt/muon_matched_givens.py"}
    gate = plan["performance_and_stability_gate"]["requirements"]
    assert gate["decoupled_weight_decay_applications_every_row"] == 1
    assert gate["expected_diagnostic_rows"] == 12 * 25
    assert gate["minimum_mfu_fraction"] == 0.2
    assert plan["order"] == ["performance_and_stability_gate", "scientific_run"]


def test_single_decay_promotion_is_stricter_than_original_success_gate() -> None:
    plan = json.loads(PLAN.read_text())
    decision = plan["decision_rule"]
    assert math.isclose(
        decision["promote_to_350m_ceiling"],
        decision["original_double_counted_candidate_validation_ce"] - 0.005,
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    assert decision["promote_to_350m_ceiling"] < (
        decision["matched_hidden88_dense_cfc_parent_validation_ce"] + 0.1
    )
    assert "no watchdog" in plan["execution"]["monitoring"]
