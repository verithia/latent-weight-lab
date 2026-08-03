from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from examples.nanogpt.train import load_config


REPO = Path(__file__).resolve().parents[2]
PLAN_PATH = (
    REPO
    / "examples/nanogpt/configs/selection_artifacts/690m_full_mlp_scaled_transfer_0p5tpp_plan.json"
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


def test_plan_binds_config_control_parent_and_implementation() -> None:
    plan = load(PLAN_PATH)
    identity = plan["identity"]
    for path_field, hash_field in (
        ("config", "config_sha256"),
        ("attention_control_config", "attention_control_config_sha256"),
        ("attention_control_ranking", "attention_control_ranking_sha256"),
        ("selected_350m_result", "selected_350m_result_sha256"),
    ):
        assert sha256(REPO / identity[path_field]) == identity[hash_field]
    for relative, expected in identity["implementation_source_hashes"].items():
        assert git_blob_sha256(identity["implementation_commit"], relative) == expected


def test_width_scaling_preserves_selected_coordinate_fraction() -> None:
    plan = load(PLAN_PATH)
    candidate = plan["candidate"]
    config = load(REPO / plan["identity"]["config"])
    assert candidate["width_ratio_690m_over_350m"] == 1280 / 1024
    assert sum(candidate["cfc_schedule_350m"]) == 176
    assert sum(candidate["cfc_schedule_690m"]) == 220
    assert 176 * 1.25 == 220
    assert candidate["cproj_parent_stages_350m"] * 1.25 == 80
    assert candidate["cproj_residual_stages_350m"] * 1.25 == 30
    assert config["block_fht_mlp_cfc_directed_product_schedule"] == [37, 37, 37, 37, 36, 36]
    assert config["block_fht_mlp_cproj_muon_matched_givens_stages"] == 80
    assert config["block_fht_mlp_cproj_muon_matched_givens_residual_stages"] == 30
    assert config["directed_product_representation"]["coordinate_fraction_per_cfc"] == 11 / 256
    assert config["muon_matched_givens_representation"]["coordinate_fraction_per_cproj"] == 11 / 256


def test_candidate_inherits_690m_control_runtime_and_selected_decays() -> None:
    plan = load(PLAN_PATH)
    candidate = load(REPO / plan["identity"]["config"])
    control = load(REPO / plan["identity"]["attention_control_config"])
    for field in (
        "batch_size",
        "block_size",
        "eval_batch_size",
        "eval_iters",
        "gradient_accumulation_steps",
        "learning_rate",
        "max_iters",
        "min_lr",
        "model_seed",
        "muon_adamw_lr_scale",
        "muon_momentum",
        "muon_ns_steps",
        "n_embd",
        "n_head",
        "n_layer",
        "optimizer",
        "tokens_per_iter",
        "train_data_seed",
        "weight_decay",
    ):
        assert candidate[field] == control[field], field
    assert candidate["block_fht_mlp_cfc_directed_product_error_feedback_decay"] == 1.0
    assert candidate["block_fht_mlp_cproj_muon_matched_givens_error_feedback_decay"] == 0.5
    assert plan["candidate"]["total_additional_trainable_parameters"] == 0
    assert plan["candidate"]["total_additional_dense_optimizer_state_bytes"] == 1677721600


def test_config_is_accepted_by_production_loader() -> None:
    plan = load(PLAN_PATH)
    config = load_config(str(REPO / plan["identity"]["config"]))
    assert config["n_layer"] == 32
    assert config["n_embd"] == 1280
    assert config["max_iters"] == 1326
    assert config["terminal_eval_required"] is True
    assert config["block_fht_native_extension_required"] is True


def test_endpoint_performance_gate_monitoring_and_authorization_are_frozen() -> None:
    plan = load(PLAN_PATH)
    config = load(REPO / plan["identity"]["config"])
    assert plan["decision_rule"]["pass_validation_ce_maximum"] == 3.8003
    assert config["preregistered_decision_rule"]["pass_validation_ce_maximum"] == 3.8003
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
