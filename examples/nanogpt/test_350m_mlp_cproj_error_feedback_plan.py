from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PLAN_PATH = (
    REPO
    / "examples/nanogpt/configs/selection_artifacts/350m_mlp_cproj_error_feedback_0p5tpp_plan.json"
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


def test_plan_binds_config_controls_results_and_implementation() -> None:
    plan = load(PLAN_PATH)
    identity = plan["identity"]
    config_path = REPO / identity["config"]
    assert sha256(config_path) == identity["config_sha256"]
    config = load(config_path)
    assert config["data_manifest_sha256"] == identity["dataset_manifest_sha256"]
    for path_field, hash_field in (
        ("matched_control_config", "matched_control_config_sha256"),
        ("matched_control_result", "matched_control_result_sha256"),
        ("negative_transfer_result", "negative_transfer_result_sha256"),
        ("smallest_rung_result", "smallest_rung_result_sha256"),
    ):
        assert sha256(REPO / identity[path_field]) == identity[hash_field]
    for relative, expected in identity["implementation_source_hashes"].items():
        assert git_blob_sha256(identity["implementation_commit"], relative) == expected


def test_candidate_changes_only_cproj_temporal_carry_in_scientific_path() -> None:
    plan = load(PLAN_PATH)
    candidate = load(REPO / plan["identity"]["config"])
    control = load(REPO / plan["identity"]["matched_control_config"])
    frozen_fields = (
        "batch_size",
        "block_fht_targets",
        "block_fht_mlp_cproj_muon_matched_givens_stages",
        "block_fht_mlp_cproj_muon_matched_givens_residual_stages",
        "block_fht_mlp_cproj_muon_matched_givens_neighbors",
        "block_fht_mlp_cproj_muon_matched_givens_seed",
        "block_fht_mlp_cproj_muon_matched_givens_refresh_interval",
        "block_fht_mlp_cproj_muon_matched_givens_fast_fresh",
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
        assert candidate[field] == control[field], field
    assert candidate["block_fht_mlp_cproj_muon_matched_givens_error_feedback"] is True
    assert candidate["block_fht_mlp_cproj_muon_matched_givens_error_feedback_decay"] == 1.0
    assert "mlp.c_fc" not in candidate["block_fht_targets"]


def test_capacity_and_decision_endpoint_are_frozen() -> None:
    plan = load(PLAN_PATH)
    config = load(REPO / plan["identity"]["config"])
    assert (plan["candidate"]["parent_stages"], plan["candidate"]["residual_stages"]) == (64, 24)
    assert plan["candidate"]["additional_dense_optimizer_state_bytes"] == 24 * 4096 * 1024 * 4
    assert config["muon_matched_givens_representation"]["total_matching_stages"] == 88
    assert plan["controls"]["attention_only_terminal_validation_ce"] == 4.3629
    assert plan["decision_rule"]["pass_validation_ce_maximum"] == 4.4629
    assert config["preregistered_decision_rule"]["pass_validation_ce_maximum"] == 4.4629
    assert config["max_iters"] == config["lr_decay_iters"] == 677
    assert config["block_fht_native_extension_required"] is True


def test_mfu_is_directly_polled_before_long_run_watchdog() -> None:
    plan = load(PLAN_PATH)
    command = plan["execution"]["mfu_command"]
    assert command[command.index("--warmup-updates") + 1] == "1"
    assert command[command.index("--timed-updates") + 1] == "8"
    assert command[command.index("--min-fraction") + 1] == "0.2"
    assert plan["execution"]["mfu_polling"].startswith("foreground")
    assert plan["execution"]["callback_milestones"] == [20, 50, 100]
    assert plan["execution"]["callback_mention"] == "@Codex"
    assert plan["execution"]["heartbeat_minutes"] == 90
    assert plan["authorization"]["automatic_rerun_authorized"] is False
    assert plan["authorization"]["larger_model_or_token_rung_authorized"] is False
