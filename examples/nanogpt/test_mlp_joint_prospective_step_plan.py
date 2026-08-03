from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PLAN = (
    REPO
    / "examples/nanogpt/configs/selection_artifacts/124m_mlp_joint_prospective_step_plan_v2.json"
)
PLAN_V1 = REPO / "examples/nanogpt/configs/selection_artifacts/124m_mlp_joint_prospective_step_plan.json"
FAILURE = REPO / "examples/nanogpt/configs/selection_artifacts/124m_mlp_joint_prospective_step_attempt1_failure.json"
RESULT = REPO / "examples/nanogpt/configs/selection_artifacts/124m_mlp_joint_prospective_step_result.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_joint_prospective_plan_binds_repository_inputs() -> None:
    plan = json.loads(PLAN.read_text())
    assert plan["schema_version"] == "mlp_joint_prospective_step_plan_v2"
    for path, digest in plan["source_hashes"].items():
        assert sha256(REPO / path) == digest
    identity = plan["identity"]
    assert sha256(REPO / identity["config"]) == identity["config_sha256"]
    assert sha256(REPO / identity["parent_result"]) == (
        identity["parent_result_sha256"]
    )
    assert sha256(REPO / identity["entrypoint"]) == (
        identity["entrypoint_sha256"]
    )
    assert sha256(PLAN_V1) == plan["predecessor"]["plan_sha256"]
    assert sha256(FAILURE) == plan["predecessor"]["failure_sha256"]


def test_attempt1_source_identity_is_historical_not_live() -> None:
    plan = json.loads(PLAN_V1.read_text())
    commit = plan["implementation_commit"]
    for path, digest in plan["source_hashes"].items():
        content = subprocess.check_output(
            ["git", "-C", str(REPO), "show", f"{commit}:{path}"]
        )
        assert hashlib.sha256(content).hexdigest() == digest


def test_joint_prospective_plan_uses_fresh_fixed_windows_and_no_watcher() -> None:
    plan = json.loads(PLAN.read_text())
    protocol = plan["protocol"]
    assert protocol["train_seed"] == 20260840
    assert protocol["validation_seeds"] == [20260841, 20260842]
    assert protocol["gradient_accumulation_steps"] == 8
    assert protocol["evaluation_batches_per_window"] == 32
    assert plan["execution"]["checkpoint_parameter_updates"] == 0
    assert "no watchdog" in plan["execution"]["monitoring"]
    assert "does not authorize" in plan["authorization"]


def test_joint_prospective_result_is_exact_mixed_and_nonarchitectural() -> None:
    result = json.loads(RESULT.read_text())
    assert result["schema_version"] == "mlp_joint_prospective_step_result_v1"
    assert result["decision"] == "MIXED_CFC_CPROJ_UPDATE_INTERACTION"
    assert result["joint_helpful_on_all_windows"] is True
    assert result["inputs"]["plan_sha256"] == sha256(PLAN)
    assert result["prospective_step"]["joint_singleton_update_max_abs_error"] == {
        "c_fc": 0.0,
        "c_proj": 0.0,
    }
    signs = [
        row["finite_interaction"]
        for row in result["finite_ce_by_window"].values()
    ]
    assert min(signs) < 0.0 < max(signs)
    assert "does not justify HyperConnection" in (
        result["interpretation"]["not_established"]
    )
