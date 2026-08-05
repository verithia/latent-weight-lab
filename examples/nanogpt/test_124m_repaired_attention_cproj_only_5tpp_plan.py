from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "examples/nanogpt/configs/pro6_mai_v3_124m_repairedfullattn_plus_cprojdecay0p5_5tpp_lr24e4.json"
PLAN = REPO / "examples/nanogpt/configs/selection_artifacts/124m_repaired_attention_cproj_only_5tpp_plan.json"
JOINT_CONFIG = REPO / "examples/nanogpt/configs/pro6_mai_v3_124m_repairedfullattn_plus_fullmlp_cfcdecay1_cprojdecay0p5_5tpp_lr24e4_v2.json"
JOINT_RESULT = REPO / "examples/nanogpt/configs/selection_artifacts/124m_repaired_attention_full_mlp_5tpp_result.json"


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    assert isinstance(value, dict)
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_identity_and_terminal_evidence_are_hash_pinned() -> None:
    plan = load(PLAN)
    identity = plan["identity"]
    for path_key, hash_key in (
        ("config", "config_sha256"),
        ("attention_parent_config", "attention_parent_config_sha256"),
        ("attention_parent_result", "attention_parent_result_sha256"),
        ("joint_full_mlp_config", "joint_full_mlp_config_sha256"),
        ("joint_full_mlp_result", "joint_full_mlp_result_sha256"),
    ):
        assert sha256(REPO / identity[path_key]) == identity[hash_key]
    result = load(JOINT_RESULT)
    assert result["classification"] == "REJECT_FULL_MLP_124M_5TPP_INCREMENTAL_GATE"
    assert result["decision_rule"]["threshold_changed_after_measurement"] is False


def test_cproj_only_is_the_registered_single_factor_ablation() -> None:
    candidate = load(CONFIG)
    joint = load(JOINT_CONFIG)
    removed = set(load(PLAN)["single_factor_change"]["removed_keys"])
    for key in removed:
        assert key not in candidate
        assert key in joint
    frozen = (
        "n_layer", "n_head", "n_embd", "block_size", "batch_size",
        "gradient_accumulation_steps", "max_iters", "warmup_iters",
        "eval_interval", "eval_iters", "learning_rate", "min_lr",
        "optimizer", "weight_decay", "model_seed", "train_data_seed",
        "data_manifest_sha256", "eval_seed", "fixed_eval_index_spec_sha256",
        "block_fht_attn_cayley_targets", "block_fht_attn_cayley_ranks",
        "block_fht_attn_cayley_bilateral_targets",
        "block_fht_attn_cayley_output_targets", "block_fht_attn_cayley_scale",
        "block_fht_attn_cayley_lr_scale", "block_fht_targets",
        "block_fht_mlp_cproj_muon_matched_givens_stages",
        "block_fht_mlp_cproj_muon_matched_givens_residual_stages",
        "block_fht_mlp_cproj_muon_matched_givens_neighbors",
        "block_fht_mlp_cproj_muon_matched_givens_refresh_interval",
        "block_fht_mlp_cproj_muon_matched_givens_fast_fresh",
        "block_fht_mlp_cproj_muon_matched_givens_error_feedback",
        "block_fht_mlp_cproj_muon_matched_givens_error_feedback_decay",
        "checkpoint_wall_clock_seconds",
    )
    for key in frozen:
        assert candidate[key] == joint[key], key
    assert "mlp.c_fc" not in candidate["block_fht_targets"]
    assert candidate["additional_dense_optimizer_state_bytes"] == 12 * 768 * 3072 * 4


def test_component_attribution_rule_and_performance_gate_are_frozen() -> None:
    plan = load(PLAN)
    config = load(CONFIG)
    rule = plan["decision_rule"]
    assert rule["cproj_only_pass_validation_ce_maximum"] == 3.6478
    assert rule["joint_near_match_absolute_ce_maximum"] == 0.01
    assert rule["threshold_changed_after_measurement"] is False
    assert config["max_iters"] == config["lr_decay_iters"] == 2373
    assert config["mfu_preflight_required"] is True
    assert config["mfu_min_fraction"] >= 0.20
    command = plan["execution"]["exact_mfu_command"]
    assert command[command.index("--warmup-updates") + 1] == "1"
    assert command[command.index("--timed-updates") + 1] == "8"
    assert plan["execution"]["mfu_polling"].startswith("foreground")
    monitoring = plan["execution"]["scientific_monitoring"]
    assert monitoring["milestones"] == [20, 50, 100]
    assert monitoring["heartbeat_minutes"] == 90
    assert monitoring["progress_resets_heartbeat"] is True
    assert monitoring["callback_endpoint"].endswith("/send-opencode-test")
    assert monitoring["callback_mention"] == "@Codex"
    assert monitoring["terminal_delivery_once"] is True


def test_no_rerun_or_scale_is_pre_authorized() -> None:
    authorization = load(PLAN)["authorization"]
    assert authorization["one_scientific_run"] is True
    assert authorization["automatic_rerun"] is False
    assert authorization["larger_rung"] is False
    assert authorization["parallel_gpu_experiment"] is False
