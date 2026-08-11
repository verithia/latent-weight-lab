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
    assert config["launch_ready"] is True


def test_candidate_horizon_identity_and_performance_gate_are_frozen() -> None:
    config = load(CONFIG)
    plan = load(PLAN)
    assert config["tokens_per_iter"] == 262144
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
