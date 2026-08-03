from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PLAN_PATH = (
    REPO
    / "examples/nanogpt/configs/selection_artifacts/350m_full_mlp_cfc_on_bounded_cproj_0p5tpp_plan.json"
)
MFU_RESULT_PATH = (
    REPO
    / "examples/nanogpt/configs/selection_artifacts/350m_full_mlp_cfc_on_bounded_cproj_0p5tpp_mfu_result.json"
)
RESULT_PATH = (
    REPO
    / "examples/nanogpt/configs/selection_artifacts/350m_full_mlp_cfc_on_bounded_cproj_0p5tpp_result.json"
)


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    assert isinstance(value, dict)
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_blob_sha256(commit: str, path: str) -> str:
    payload = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=REPO)
    return hashlib.sha256(payload).hexdigest()


def test_plan_binds_config_parents_and_implementation() -> None:
    plan = load(PLAN_PATH)
    identity = plan["identity"]
    for path_field, hash_field in (
        ("config", "config_sha256"),
        ("bounded_cproj_parent_config", "bounded_cproj_parent_config_sha256"),
        ("bounded_cproj_parent_result", "bounded_cproj_parent_result_sha256"),
        ("failed_full_mlp_result", "failed_full_mlp_result_sha256"),
        ("smallest_full_mlp_result", "smallest_full_mlp_result_sha256"),
    ):
        assert sha256(REPO / identity[path_field]) == identity[hash_field]
    for relative, expected in identity["implementation_source_hashes"].items():
        assert git_blob_sha256(identity["implementation_commit"], relative) == expected


def test_candidate_keeps_selected_cproj_and_adds_only_cfc_science() -> None:
    plan = load(PLAN_PATH)
    identity = plan["identity"]
    candidate = load(REPO / identity["config"])
    parent = load(REPO / identity["bounded_cproj_parent_config"])
    frozen_fields = (
        "batch_size",
        "block_fht_targets",
        "block_fht_mlp_cproj_muon_matched_givens",
        "block_fht_mlp_cproj_muon_matched_givens_stages",
        "block_fht_mlp_cproj_muon_matched_givens_residual_stages",
        "block_fht_mlp_cproj_muon_matched_givens_neighbors",
        "block_fht_mlp_cproj_muon_matched_givens_seed",
        "block_fht_mlp_cproj_muon_matched_givens_refresh_interval",
        "block_fht_mlp_cproj_muon_matched_givens_fast_fresh",
        "block_fht_mlp_cproj_muon_matched_givens_error_feedback",
        "block_fht_mlp_cproj_muon_matched_givens_error_feedback_decay",
        "gradient_accumulation_steps",
        "learning_rate",
        "max_iters",
        "min_lr",
        "model_seed",
        "muon_momentum",
        "muon_ns_steps",
        "n_embd",
        "n_head",
        "n_layer",
        "optimizer",
        "tokens_per_iter",
        "train_data_seed",
        "weight_decay",
    )
    for field in frozen_fields:
        assert candidate[field] == parent[field], field
    assert candidate["block_fht_mlp_cfc_directed_product"] is True
    assert candidate["block_fht_mlp_cfc_directed_product_schedule"] == [30, 30, 29, 29, 29, 29]
    assert candidate["block_fht_mlp_cfc_directed_product_error_feedback"] is True
    assert candidate["block_fht_mlp_cfc_directed_product_error_feedback_decay"] == 1.0
    assert candidate["block_fht_mlp_cfc_directed_product_family_radius_ratio"] == 1.0


def test_endpoint_performance_gate_and_monitoring_are_frozen() -> None:
    plan = load(PLAN_PATH)
    config = load(REPO / plan["identity"]["config"])
    assert plan["decision_rule"]["pass_validation_ce_maximum"] == 4.4629
    assert config["preregistered_decision_rule"]["pass_validation_ce_maximum"] == 4.4629
    assert config["max_iters"] == config["lr_decay_iters"] == 677
    command = plan["execution"]["mfu_command"]
    assert command[command.index("--min-fraction") + 1] == "0.2"
    assert command[command.index("--warmup-updates") + 1] == "1"
    assert command[command.index("--timed-updates") + 1] == "8"
    assert plan["execution"]["mfu_polling"].startswith("foreground")
    assert plan["execution"]["callback_milestones"] == [20, 50, 100]
    assert plan["execution"]["callback_mention"] == "@Codex"
    assert plan["execution"]["heartbeat_minutes"] == 90
    assert plan["authorization"]["automatic_rerun_authorized"] is False
    assert plan["authorization"]["larger_model_or_token_rung_authorized"] is False


def test_representation_cost_is_explicit() -> None:
    plan = load(PLAN_PATH)
    candidate = plan["candidate"]
    assert candidate["total_additional_trainable_parameters"] == 0
    assert candidate["total_additional_dense_optimizer_state_bytes"] == 2 * 24 * 4096 * 1024 * 4
    assert candidate["inference_parameter_or_flop_change_vs_materialized_parent"] == 0


def test_exact_mfu_result_authorizes_one_scientific_run() -> None:
    result = load(MFU_RESULT_PATH)
    assert result["passed"] is True
    assert result["classification"] == (
        "FULL_MLP_CFC_ON_BOUNDED_CPROJ_350M_0P5TPP_EXACT_CONFIG_MFU_PASSED"
    )
    assert sha256(REPO / result["config"]["path"]) == result["config"]["sha256"]
    assert sha256(REPO / result["plan"]["path"]) == result["plan"]["sha256"]
    assert result["measurement"]["mfu_fraction"] >= 0.2
    assert result["measurement"]["peak_mib"] < 97887
    assert result["stability"]["all_logged_losses_finite"] is True
    assert result["stability"]["native_matching_validated"] is True
    assert result["execution"]["direct_foreground_polling"] is True
    assert result["execution"]["watchdog"] is False
    assert result["decision"]["one_full_677_update_run_authorized"] is True
    assert result["decision"]["automatic_rerun_authorized"] is False
    assert result["decision"]["larger_model_or_token_rung_authorized"] is False


def test_terminal_result_selects_full_mlp_recipe_for_next_rung() -> None:
    result = load(RESULT_PATH)
    assert result["classification"] == (
        "PASS_FULL_MLP_CFC_ON_BOUNDED_CPROJ_350M_CLOSES_ATTENTION_GAP"
    )
    assert sha256(REPO / result["config"]["path"]) == result["config"]["sha256"]
    assert sha256(REPO / result["plan"]["path"]) == result["plan"]["sha256"]
    assert sha256(REPO / result["provenance"]["mfu_result_path"]) == (
        result["provenance"]["mfu_result_sha256"]
    )
    assert result["run"]["status"] == "clean"
    assert result["run"]["train_exit_code"] == 0
    assert result["checkpoint"]["exact_resume_schema"] == "nanogpt_exact_resume_v2"
    assert result["checkpoint"]["next_iter"] == 677
    assert result["measurement"]["terminal_validation_ce_exact"] <= 4.4629
    assert abs(result["measurement"]["candidate_minus_bounded_cproj_parent_ce_exact"]) <= 0.01
    assert result["residual_diagnosis"]["cfc_residual_remains_small"] is True
    assert result["residual_diagnosis"]["cproj_residual_remains_bounded"] is True
    assert result["residual_diagnosis"]["direction_problem_resolved_at_350m"] is True
    assert result["decision"]["selected_for_full_mlp_scaling"] is True
    assert result["decision"]["next_model_rung_registration_authorized"] is True
    assert result["decision"]["larger_model_or_token_rung_run_authorized"] is False
    assert result["decision"]["automatic_rerun_authorized"] is False
    assert result["watchdog"]["callback_milestones_sent"] == [20, 50, 100]
    assert result["watchdog"]["duplicate_terminal_callbacks"] == 0
