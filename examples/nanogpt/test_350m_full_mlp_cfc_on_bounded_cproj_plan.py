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
