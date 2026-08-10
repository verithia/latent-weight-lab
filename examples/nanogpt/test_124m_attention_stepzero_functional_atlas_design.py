from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DESIGN = (
    ROOT
    / "examples/nanogpt/configs/selection_artifacts/"
    "124m_attention_stepzero_functional_atlas_design.json"
)


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
