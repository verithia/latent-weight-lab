from __future__ import annotations

import json
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DESIGN = (
    ROOT
    / "examples/nanogpt/configs/selection_artifacts/"
    "124m_attention_terminal_functional_atlas_design.json"
)
PLAN = (
    ROOT
    / "examples/nanogpt/configs/selection_artifacts/"
    "124m_attention_terminal_functional_atlas_plan.json"
)
RESULT = (
    ROOT
    / "examples/nanogpt/configs/selection_artifacts/"
    "124m_attention_terminal_functional_atlas_result.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_terminal_atlas_is_teacher_upper_bound_not_training() -> None:
    design = json.loads(DESIGN.read_text())
    teacher = design["teacher_information"]
    assert teacher["terminal_dense_checkpoint_used_for_basis"] is True
    assert teacher["deployable_causal_decoder"] is False
    assert teacher["parameter_updates"] == 0
    assert design["authorization"] == {
        "current_successor": "none",
        "language_model_training": False,
        "mfu_preflight": False,
        "larger_rung": False,
    }


def test_terminal_atlas_changes_only_basis_time_and_keeps_strict_gates() -> None:
    design = json.loads(DESIGN.read_text())
    protocol = design["frozen_protocol"]
    assert protocol["coordinate_fraction"] == 0.01
    assert protocol["changed_from_parent"] == [
        "basis model state",
        "calibration seeds",
    ]
    assert "thresholds" in protocol["unchanged_from_parent"]
    assert design["decision_rule"]["thresholds"] == {
        "aggregate_recovery_minimum": 0.8,
        "minimum_every_layer_recovery": 0.6,
        "minimum_late_layer_8_to_11_recovery": 0.6,
        "minimum_absolute_gain_over_blockfht": 0.1,
        "minimum_calibration_split_subspace_overlap": 0.75,
    }


def test_terminal_atlas_fail_rule_rejects_kfac_training_sweep() -> None:
    design = json.loads(DESIGN.read_text())
    assert "do not run a KFAC training sweep" in design["decision_rule"]["fail"]
    assert design["execution_policy"]["monitoring"].startswith(
        "foreground polling"
    )


def test_executable_plan_pins_code_design_and_teacher_identity() -> None:
    plan = json.loads(PLAN.read_text())
    identity = plan["identity"]
    entrypoint = ROOT / "examples/nanogpt/analyze_attention_terminal_functional_atlas.py"
    assert identity["entrypoint_sha256"] == sha256(entrypoint)
    assert identity["design_sha256"] == sha256(DESIGN)
    assert plan["protocol"]["basis_model_source"] == "terminal_dense_checkpoint"
    assert plan["protocol"]["parameter_updates"] == 0
    assert all(value is False for value in plan["authorization"].values())
    assert "foreground polling" in plan["execution"]["monitoring"]


def test_terminal_result_rejects_separable_kfac_without_training_authority() -> None:
    result = json.loads(RESULT.read_text())
    assert result["classification"] == "ATTENTION_TERMINAL_FUNCTIONAL_ATLAS_REJECT"
    assert result["execution"]["parameter_updates"] == 0
    assert result["execution"]["basis_uses_terminal_teacher_state"] is True
    assert result["identity"]["plan_sha256"] == sha256(PLAN)
    assert result["decision"]["passed_targets"] == []
    assert result["decision"]["language_model_training_authorized"] is False
    assert result["decision"]["online_adaptive_atlas_implementation_gate_authorized"] is False
    assert result["summaries"]["v"]["terminal_kfac"]["exact_muon"]["absolute_gain_over_blockfht"] < 0
    assert result["summaries"]["v"]["calibration_overlap"]["minimum"] < 0.1
