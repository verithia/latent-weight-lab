from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PLAN_PATH = (
    REPO
    / "examples/nanogpt/configs/selection_artifacts/350m_full_mlp_error_feedback_0p5tpp_plan.json"
)
V2_PLAN_PATH = (
    REPO
    / "examples/nanogpt/configs/selection_artifacts/350m_full_mlp_error_feedback_0p5tpp_v2_plan.json"
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


def test_plan_binds_config_controls_dataset_and_implementation() -> None:
    plan = load(PLAN_PATH)
    assert plan["schema_version"] == "mai_350m_full_mlp_error_feedback_0p5tpp_plan_v1"
    identity = plan["identity"]
    config_path = REPO / identity["config"]
    assert sha256(config_path) == identity["config_sha256"]
    config = load(config_path)
    assert config["data_manifest_sha256"] == identity["dataset_manifest_sha256"]

    commit = identity["implementation_commit"]
    for relative, expected in identity["implementation_source_hashes"].items():
        assert git_blob_sha256(commit, relative) == expected

    controls = plan["controls"]
    for path_field, hash_field in (
        ("attention_only_config", "attention_only_config_sha256"),
        ("attention_only_ranking", "attention_only_ranking_sha256"),
        ("memoryless_cproj_result", "memoryless_cproj_result_sha256"),
        ("conditioned_full_mlp_result", "conditioned_full_mlp_result_sha256"),
        ("smallest_rung_full_mlp_result", "smallest_rung_full_mlp_result_sha256"),
    ):
        assert sha256(REPO / controls[path_field]) == controls[hash_field]


def test_width_scaling_preserves_registered_chart_capacity() -> None:
    plan = load(PLAN_PATH)
    config = load(REPO / plan["identity"]["config"])
    cfc = plan["candidate"]["cfc"]
    assert cfc["schedule"] == [30, 30, 29, 29, 29, 29]
    assert sum(cfc["schedule"]) == 176
    assert sum(cfc["schedule"]) / 4096 == cfc["incoming_coordinate_fraction"]
    assert cfc["coordinates_per_layer"] == 4096 * sum(cfc["schedule"])
    assert config["block_fht_mlp_cfc_directed_product_schedule"] == cfc["schedule"]
    assert config["block_fht_mlp_cfc_directed_product_family_radius_ratio"] == 1.0
    assert config["block_fht_mlp_cfc_directed_product_error_feedback"] is True

    cproj = plan["candidate"]["cproj"]
    assert (cproj["parent_stages"], cproj["residual_stages"], cproj["matching_neighbors"]) == (
        85,
        32,
        85,
    )
    assert cproj["coordinates_per_layer"] == 2 * 1024 * (85 + 32)
    assert config["block_fht_mlp_cproj_muon_matched_givens_stages"] == 85
    assert config["block_fht_mlp_cproj_muon_matched_givens_residual_stages"] == 32
    assert config["block_fht_mlp_cproj_muon_matched_givens_neighbors"] == 85
    assert config["block_fht_mlp_cproj_muon_matched_givens_error_feedback"] is True

    per_family_bytes = 24 * 4096 * 1024 * 4
    assert per_family_bytes == 402653184
    assert plan["candidate"]["additional_dense_optimizer_state_bytes"] == 2 * per_family_bytes
    assert config["total_additional_dense_optimizer_state_bytes"] == 2 * per_family_bytes
    assert config["block_fht_native_extension_required"] is True


def test_scientific_admission_is_exact_mfu_gated_and_callbacks_are_long_run_only() -> None:
    plan = load(PLAN_PATH)
    execution = plan["execution"]
    authorization = plan["authorization"]
    command = execution["mfu_command"]
    assert command[command.index("--warmup-updates") + 1] == "1"
    assert command[command.index("--timed-updates") + 1] == "8"
    assert command[command.index("--min-fraction") + 1] == "0.2"
    assert execution["mfu_polling"].startswith("foreground")
    assert authorization["exact_config_mfu_run_authorized"] is True
    assert authorization["full_677_update_scientific_run_authorized_after_exact_mfu_pass"] is True
    assert authorization["automatic_rerun_authorized"] is False
    assert authorization["larger_model_or_token_rung_authorized"] is False
    assert execution["callback_endpoint"].endswith("/send-opencode-test")
    assert execution["callback_mention"] == "@Codex"
    assert execution["callback_milestones"] == [20, 50, 100]
    assert execution["heartbeat_minutes"] == 90
    assert "aggregate" in execution["watchdog"]


def test_loss_and_token_endpoints_are_frozen() -> None:
    plan = load(PLAN_PATH)
    config = load(REPO / plan["identity"]["config"])
    assert plan["controls"]["attention_only_terminal_validation_ce"] == 4.3629
    assert plan["decision_rule"]["pass_validation_ce_maximum"] == 4.4629
    assert config["preregistered_decision_rule"]["pass_validation_ce_maximum"] == 4.4629
    assert config["max_iters"] == config["lr_decay_iters"] == 677
    assert config["tokens_per_iter"] == 262144
    assert config["scheduled_tokens"] == config["max_iters"] * config["tokens_per_iter"]
    assert config["planned_tokens"] == 177299968
    assert config["learning_rate"] == 0.0024
    assert config["n_layer"] == 24
    assert config["n_embd"] == 1024
    assert config["n_head"] == 16
    assert config["optimizer"] == "muon"


def test_v1_invalid_stage_allocation_is_preserved_as_a_rejection() -> None:
    result_path = (
        REPO
        / "examples/nanogpt/configs/selection_artifacts/350m_full_mlp_error_feedback_0p5tpp_v1_preflight_reject_result.json"
    )
    result = load(result_path)
    assert result["decision"] == "REJECT_INVALID_NATIVE_STAGE_ALLOCATION"
    assert sha256(REPO / result["config"]["path"]) == result["config"]["sha256"]
    assert sha256(REPO / result["plan"]["path"]) == result["plan"]["sha256"]
    assert result["execution"]["train_exit_code"] == 1
    assert result["execution"]["warmup_updates_completed"] == 0
    assert result["execution"]["timed_updates_completed"] == 0
    assert result["failure"]["scientific_training_started"] is False
    assert result["repair_constraint"]["required_split"] == [64, 53]
    assert result["repair_constraint"]["capacity_change"] is False
    assert result["repair_constraint"]["loss_threshold_change"] is False


def test_v2_redistributes_identical_capacity_into_native_legal_passes() -> None:
    v1_plan = load(PLAN_PATH)
    v2_plan = load(V2_PLAN_PATH)
    assert v2_plan["schema_version"] == "mai_350m_full_mlp_error_feedback_0p5tpp_plan_v2"
    integrity = v2_plan["registration_integrity"]
    assert sha256(REPO / integrity["v1_plan"]) == integrity["v1_plan_sha256"]
    assert sha256(REPO / integrity["v1_preflight_rejection"]) == (
        integrity["v1_preflight_rejection_sha256"]
    )
    assert integrity["capacity_changed_from_v1"] is False
    assert integrity["cfc_changed_from_v1"] is False
    assert integrity["forward_or_optimizer_family_changed_from_v1"] is False
    assert integrity["loss_thresholds_changed_from_v1"] is False

    v1_config = load(REPO / v1_plan["identity"]["config"])
    v2_config = load(REPO / v2_plan["identity"]["config"])
    assert sha256(REPO / v2_plan["identity"]["config"]) == v2_plan["identity"]["config_sha256"]
    assert v2_config["block_fht_mlp_cproj_muon_matched_givens_stages"] == 64
    assert v2_config["block_fht_mlp_cproj_muon_matched_givens_residual_stages"] == 53
    assert v2_config["block_fht_mlp_cproj_muon_matched_givens_neighbors"] == 64
    assert 64 + 53 == 85 + 32 == v2_plan["candidate"]["cproj_total_stages"]
    assert v2_config["block_fht_mlp_cfc_directed_product_schedule"] == (
        v1_config["block_fht_mlp_cfc_directed_product_schedule"]
    )
    assert v2_config["preregistered_decision_rule"] == v1_config["preregistered_decision_rule"]
    assert v2_plan["decision_rule"]["pass_validation_ce_maximum"] == 4.4629


def test_v2_exact_gate_and_long_run_monitoring_remain_separate() -> None:
    plan = load(V2_PLAN_PATH)
    command = plan["execution"]["mfu_command"]
    assert command[command.index("--warmup-updates") + 1] == "1"
    assert command[command.index("--timed-updates") + 1] == "8"
    assert command[command.index("--min-fraction") + 1] == "0.2"
    assert plan["execution"]["mfu_polling"].startswith("foreground")
    assert plan["execution"]["callback_milestones"] == [20, 50, 100]
    assert plan["execution"]["callback_mention"] == "@Codex"
    assert plan["execution"]["heartbeat_minutes"] == 90
    assert plan["authorization"]["automatic_rerun_authorized"] is False
