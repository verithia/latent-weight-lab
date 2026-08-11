from __future__ import annotations

import hashlib
import json
from pathlib import Path

from examples.nanogpt.mfu_preflight import estimate_active_params


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_qk_mapped_dense_moe8_top2_0p5tpp_lr24e4.json"
)
PLAN = (
    ROOT
    / "examples/nanogpt/configs/selection_artifacts/"
    "124m_qk_mapped_dense_moe_composition_plan.json"
)
CONFIG_5TPP = (
    ROOT
    / "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_qk_mapped_dense_moe8_top2_5tpp_lr24e4.json"
)
PLAN_5TPP = (
    ROOT
    / "examples/nanogpt/configs/selection_artifacts/"
    "124m_qk_mapped_dense_moe_5tpp_confirmation_plan.json"
)
RESULT_0P5TPP = (
    ROOT
    / "examples/nanogpt/configs/selection_artifacts/"
    "124m_qk_mapped_dense_moe_composition_0p5tpp_result.json"
)


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_candidate_changes_only_attention_qk_from_dense_moe_scope() -> None:
    config = load(CONFIG)
    assert config["method"] == "block_fht"
    assert config["block_fht_targets"] == ["attn.c_attn.qk_headwise"]
    assert config["moe_num_experts"] == 8
    assert config["moe_top_k"] == 2
    assert config["moe_expert_hidden_multiplier"] == 2
    assert config["optimizer"] == "muon"
    assert config["estimated_materialized_active_params"] == 124447488
    assert estimate_active_params(config) == 124447488
    assert config["estimated_registered_trainable_params"] == 283859952
    assert config["estimated_registered_active_trainable_params"] == 113990640
    assert config["estimated_qk_mapping_trainable_state"] == 3698928
    assert config["estimated_qk_materialized_parameters"] == 14155776
    assert config["block_fht_attn_pack_cached_qkv"] is True
    assert config["moe_unpadded_expert_loop"] is True
    assert config["launch_ready"] is True


def test_candidate_horizon_identity_and_performance_gate_are_frozen() -> None:
    config = load(CONFIG)
    plan = load(PLAN)
    assert config["tokens_per_iter"] == 262144
    assert config["batch_size"] == 32
    assert config["gradient_accumulation_steps"] == 8
    assert "cuda_allocator_conf" not in config
    assert config["scheduled_tokens"] == config["max_iters"] * 262144
    assert config["max_iters"] == 238
    assert config["mfu_preflight_required"] is True
    assert config["mfu_min_fraction"] == 0.2
    assert plan["decision_rule"]["terminal_validation_ce_maximum"] == 5.4739
    assert plan["decision_rule"]["maximum_fixed_checkpoint_penalty_ce"] == 0.03
    assert plan["authorization"]["generated_expert"] is False
    assert plan["authorization"]["larger_rung"] is False


def test_preregistered_parent_hashes_still_match() -> None:
    plan = load(PLAN)
    for record in plan["immutable_evidence"].values():
        assert sha256(ROOT / record["path"]) == record["sha256"]


def test_composition_result_passes_every_frozen_short_horizon_gate() -> None:
    result = load(RESULT_0P5TPP)
    assert result["gates"]["all_pass"] is True
    assert result["gates"]["maximum_fixed_checkpoint_penalty_ce"] <= 0.03
    assert result["gates"]["terminal_penalty_to_dense_moe_ce"] <= 0.01
    assert result["decision"]["one_124m_5tpp_confirmation_authorized"] is True
    assert result["decision"]["generated_expert_authorized"] is False


def test_5tpp_confirmation_changes_only_horizon_and_recovery_metadata() -> None:
    short = load(CONFIG)
    confirmation = load(CONFIG_5TPP)
    invariant_keys = [
        "method",
        "block_fht_targets",
        "block_fht_attn_cayley_ranks",
        "block_fht_attn_cayley_bilateral_targets",
        "block_fht_output_gain_targets",
        "block_fht_attn_pack_cached_qkv",
        "moe_num_experts",
        "moe_top_k",
        "moe_expert_hidden_multiplier",
        "moe_unpadded_expert_loop",
        "optimizer",
        "learning_rate",
        "min_lr",
        "batch_size",
        "gradient_accumulation_steps",
        "model_seed",
        "train_data_seed",
        "eval_seed",
        "eval_iters",
        "fixed_eval_indices",
        "estimated_materialized_active_params",
        "estimated_registered_active_trainable_params",
    ]
    for key in invariant_keys:
        assert confirmation[key] == short[key]
    assert confirmation["max_iters"] == 2374
    assert confirmation["eval_interval"] == 594
    assert confirmation["warmup_iters"] == 23
    assert confirmation["lr_decay_iters"] == 2374
    assert confirmation["checkpoint_wall_clock_seconds"] == 7200
    assert confirmation["scheduled_tokens"] == 2374 * 262144
    assert estimate_active_params(confirmation) == 124447488


def test_5tpp_plan_hashes_and_authorization_are_frozen() -> None:
    plan = load(PLAN_5TPP)
    for record in plan["immutable_evidence"].values():
        assert sha256(ROOT / record["path"]) == record["sha256"]
    assert plan["decision_rule"]["terminal_validation_ce_maximum"] == 3.4548
    assert plan["decision_rule"]["maximum_fixed_checkpoint_penalty_ce"] == 0.03
    assert plan["authorization"]["one_scientific_5tpp_run_after_mfu_pass"] is True
    assert plan["authorization"]["generated_expert"] is False
    assert plan["authorization"]["larger_rung"] is False
