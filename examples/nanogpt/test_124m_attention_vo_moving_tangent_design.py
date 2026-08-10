from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DESIGN = ROOT / "examples/nanogpt/configs/selection_artifacts/124m_attention_vo_moving_tangent_design.json"
PLAN = ROOT / "examples/nanogpt/configs/selection_artifacts/124m_attention_vo_moving_tangent_plan.json"
RESULT = ROOT / "examples/nanogpt/configs/selection_artifacts/124m_attention_vo_moving_tangent_result.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_moving_tangent_oracle_has_causal_selection_test_split() -> None:
    design = json.loads(DESIGN.read_text())
    split = design["selection_and_test"]
    assert split["window_candidates"] == [2, 4, 8, 16]
    assert "1200 through 1740" in split["window_selection_phase"]
    assert "beginning at 1800" in split["frozen_test_phase"]
    assert split["global_window"] is True
    assert "strictly earlier" in split["causality"]


def test_moving_tangent_oracle_is_unlaunching_teacher_upper_bound() -> None:
    design = json.loads(DESIGN.read_text())
    teacher = design["teacher_upper_bound"]
    assert teacher["dense_past_information_used"] is True
    assert teacher["future_information_used"] is False
    assert teacher["deployable_decoder"] is False
    assert all(value is False for value in design["authorization"].values())
    assert design["execution_policy"]["parameter_updates"] == 0


def test_moving_tangent_plan_pins_code_and_design() -> None:
    plan = json.loads(PLAN.read_text())
    identity = plan["identity"]
    entrypoint = ROOT / "examples/nanogpt/analyze_attention_vo_moving_tangent.py"
    assert identity["entrypoint_sha256"] == sha256(entrypoint)
    assert identity["design_sha256"] == sha256(DESIGN)
    assert all(value is False for value in plan["authorization"].values())


def test_moving_tangent_result_rejects_past_memory() -> None:
    result = json.loads(RESULT.read_text())
    assert result["classification"] == "ATTENTION_VO_MOVING_TANGENT_REJECT"
    assert result["identity"]["plan_sha256"] == sha256(PLAN)
    assert result["selection"]["selected_window"] == 8
    assert result["heldout_test"]["chord"]["aggregate_eval_recovery"] < 0.07
    assert result["heldout_test"]["muon_direction"]["aggregate_eval_recovery"] < 0.04
    assert all(value is False for value in result["checks"].values())
    assert result["decision"]["language_model_training_authorized"] is False
