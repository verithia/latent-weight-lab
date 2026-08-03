from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PLAN = (
    REPO
    / "examples/nanogpt/configs/selection_artifacts/124m_mlp_joint_prospective_step_plan.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_joint_prospective_plan_binds_repository_inputs() -> None:
    plan = json.loads(PLAN.read_text())
    assert plan["schema_version"] == "mlp_joint_prospective_step_plan_v1"
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
