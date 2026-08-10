from __future__ import annotations

import json
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DESIGN = (
    ROOT
    / "examples/nanogpt/configs/selection_artifacts/"
    "124m_attention_stepzero_functional_atlas_design.json"
)
PLAN = (
    ROOT
    / "examples/nanogpt/configs/selection_artifacts/"
    "124m_attention_stepzero_functional_atlas_plan.json"
)
RESULT = (
    ROOT
    / "examples/nanogpt/configs/selection_artifacts/"
    "124m_attention_stepzero_functional_atlas_result.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_functional_atlas_is_full_attention_smallest_rung_only() -> None:
    design = json.loads(DESIGN.read_text())
    scope = design["objective_scope"]
    assert scope["project_target"].startswith("full generated attention")
    assert set(scope["targets_under_test"]) == {
        "attn.c_attn.v",
        "attn.c_proj",
    }
    assert scope["larger_rung_authorized"] is False
    assert scope["language_model_training_authorized"] is False
    assert scope["mfu_preflight_authorized"] is False


def test_functional_atlas_has_no_teacher_or_trainable_basis() -> None:
    design = json.loads(DESIGN.read_text())
    candidate = design["candidate_upper_bound"]
    assert candidate["teacher_information_for_basis"] is False
    assert candidate["later_checkpoint_information_for_basis"] is False
    assert candidate["trainable_basis"] is False
    assert candidate["learned_dense_adapter"] is False
    assert design["frozen_protocol"]["parameter_updates"] == 0


def test_functional_atlas_freezes_strict_heldout_gates() -> None:
    design = json.loads(DESIGN.read_text())
    rule = design["decision_rule"]
    threshold = rule["thresholds"]
    assert rule["no_posthoc_threshold_changes"] is True
    assert rule["all_targets_must_pass"] is True
    assert threshold == {
        "aggregate_recovery_minimum": 0.8,
        "minimum_every_layer_recovery": 0.6,
        "minimum_late_layer_8_to_11_recovery": 0.6,
        "minimum_absolute_gain_over_blockfht": 0.1,
        "minimum_calibration_split_subspace_overlap": 0.75,
    }
    assert design["execution_policy"]["launch_before_code_and_identity_seal"] is False
    assert design["execution_policy"]["run_while_training_gpu_is_occupied"] is False


def test_executable_plan_pins_the_reviewed_design_and_entrypoint() -> None:
    plan = json.loads(PLAN.read_text())
    identity = plan["identity"]
    assert plan["schema_version"] == (
        "mai_124m_attention_stepzero_functional_atlas_plan_v1"
    )
    assert identity["design_sha256"] == sha256(DESIGN)
    entrypoint = ROOT / "examples/nanogpt/analyze_attention_stepzero_functional_atlas.py"
    assert identity["entrypoint_sha256"] == sha256(entrypoint)
    assert plan["protocol"]["parameter_updates"] == 0
    assert len(plan["protocol"]["trajectory_steps"]) == 41
    assert set(plan["protocol"]["targets"]) == {"v", "cproj"}
    assert all(value is False for value in plan["authorization"].values())


def test_terminal_result_rejects_fixed_stepzero_atlas_without_authorization() -> None:
    result = json.loads(RESULT.read_text())
    assert result["classification"] == "ATTENTION_STEPZERO_FUNCTIONAL_ATLAS_REJECT"
    assert result["execution"]["git_commit"] == (
        "20e673ed0557dbf58b973ecff3d9c2ecb2064c7a"
    )
    assert result["execution"]["parameter_updates"] == 0
    assert result["execution"]["stepzero_reconstruction_target_tensors_exact"]
    assert result["identity"]["plan_sha256"] == sha256(PLAN)
    assert result["decision"]["passed_targets"] == []
    assert all(
        value is False
        for key, value in result["decision"].items()
        if key != "passed_targets"
    )
    for target in ("v", "cproj"):
        summary = result["summaries"][target]
        assert summary["passed"] is False
        assert summary["calibration_overlap"]["mean"] < 0.75
        for metric in ("state", "local_chord", "discovery_span", "exact_muon"):
            assert (
                summary["stepzero_kfac"][metric][
                    "absolute_gain_over_blockfht"
                ]
                < 0.0
            )
