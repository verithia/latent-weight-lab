from __future__ import annotations

import hashlib
import json
from pathlib import Path

from examples.nanogpt.analyze_mlp_cfc_checkpoint_splice import variant_specs


REPO = Path(__file__).resolve().parents[2]
PLAN = (
    REPO
    / "examples/nanogpt/configs/selection_artifacts/350m_mlp_cfc_checkpoint_splice_plan.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_splice_plan_binds_repository_inputs_and_exact_variants() -> None:
    plan = json.loads(PLAN.read_text())
    assert plan["schema_version"] == "350m_mlp_cfc_checkpoint_splice_plan_v1"
    assert sha256(REPO / plan["source"]) == plan["source_sha256"]
    inputs = plan["inputs"]
    assert sha256(REPO / inputs["parent_config"]) == inputs["parent_config_sha256"]
    assert sha256(REPO / inputs["candidate_config"]) == inputs["candidate_config_sha256"]
    assert sha256(REPO / inputs["parent_terminal_result"]) == (
        inputs["parent_terminal_result_sha256"]
    )
    assert sha256(REPO / inputs["candidate_terminal_result"]) == (
        inputs["candidate_terminal_result_sha256"]
    )
    assert plan["variants"] == variant_specs(24)
    assert plan["execution"]["parameter_updates"] == 0
    assert plan["execution"]["monitoring"].startswith("foreground")
    assert plan["protocol"]["validation_seeds"] == [20260831, 20260832]
