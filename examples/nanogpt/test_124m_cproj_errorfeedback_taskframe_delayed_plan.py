from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PLAN = (
    REPO
    / "examples/nanogpt/configs/selection_artifacts/"
    "124m_mlp_cproj_errorfeedback_task_frame_delayed_start120_plan.json"
)
RESOLUTION = (
    REPO
    / "examples/nanogpt/configs/selection_artifacts/"
    "124m_mlp_cproj_errorfeedback_task_frame_delayed_start120_resolution.json"
)
CONFIG = (
    REPO
    / "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_fullattn_plus_mlp_cproj_errorfeedback_"
    "taskframe_start120_0p5tpp.json"
)
FAILED_RESULT = (
    REPO
    / "examples/nanogpt/configs/selection_artifacts/"
    "124m_mlp_cproj_errorfeedback_task_frame_0p5tpp_result.json"
)


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    assert isinstance(value, dict)
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_blob_sha256(commit: str, path: str) -> str:
    payload = subprocess.check_output(
        ["git", "show", f"{commit}:{path}"],
        cwd=REPO,
    )
    return hashlib.sha256(payload).hexdigest()


def test_resolution_binds_plan_config_and_implementation() -> None:
    plan = load(PLAN)
    resolution = load(RESOLUTION)
    config = load(CONFIG)
    assert sha256(PLAN) == resolution["plan"]["sha256"]
    assert sha256(CONFIG) == resolution["config"]["sha256"]
    assert sha256(FAILED_RESULT) == plan["evidence"]["failed_causal_result_sha256"]
    assert config["implementation_commit"] == resolution["implementation"]["commit"]
    for relative, expected in resolution["implementation"]["source_hashes"].items():
        assert git_blob_sha256(config["implementation_commit"], relative) == expected


def test_only_registered_direction_acquisition_change_is_present() -> None:
    plan = load(PLAN)
    resolution = load(RESOLUTION)
    config = load(CONFIG)
    assert config["block_fht_mlp_task_frame_start_iter"] == 120
    assert config["max_iters"] == config["lr_decay_iters"] == 238
    assert config["scheduled_tokens"] == 62_390_272
    assert config["block_fht_mlp_chart_lr_scale"] == 0.1
    assert config["block_fht_mlp_pregelu_chart_lr_scale"] == 0.1
    assert config["block_fht_mlp_hidden_block_rotation_stages"] == 2
    assert config["block_fht_mlp_output_block_rotation_stages"] == 4
    assert config["block_fht_mlp_pregelu_block_rotation_stages"] == 2
    assert config["block_fht_mlp_hidden_gain"] is False
    assert config["block_fht_mlp_residual_output_gain"] is False
    assert plan["candidate"]["freeze_cfc_base_at_start"] is False
    assert plan["candidate"]["freeze_cproj_base_at_start"] is False
    assert resolution["scientific_invariants"]["teacher_or_endpoint_state_used"] is False
    assert resolution["scientific_invariants"]["decision_threshold_changed"] is False


def test_gates_time_active_path_and_forbid_automatic_promotion() -> None:
    plan = load(PLAN)
    resolution = load(RESOLUTION)
    config = load(CONFIG)
    transform = resolution["implementation"]["preflight_transform"]
    assert transform == {
        "scientific_start": 120,
        "scratch_start": 1,
        "warmup_updates": 1,
        "timed_updates": 8,
        "timed_active_path": True,
    }
    command = resolution["execution"]["mfu_command"]
    assert command[command.index("--min-fraction") + 1] == "0.2"
    assert command[command.index("--warmup-updates") + 1] == "1"
    assert command[command.index("--timed-updates") + 1] == "8"
    assert config["preregistered_decision_rule"]["pass_validation_ce_maximum"] == 5.522365207672119
    assert plan["decision_rule"]["no_post_hoc_start_time_or_lr_sweep"] is True
    assert resolution["authorization"]["automatic_rerun_authorized"] is False
    assert resolution["authorization"]["larger_model_or_token_rung_authorized"] is False
