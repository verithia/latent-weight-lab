from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

from examples.nanogpt.train import parse_args


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / (
    "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_qk_only_qk64_outputgain_20tpp_lr24e4.json"
)
PARENT = ROOT / (
    "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_qk_only_qk64_outputgain_5tpp_lr24e4.json"
)
PLAN = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_attention_qk_only_lwt_20tpp_plan.json"
)


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    assert isinstance(value, dict)
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_plan_binds_all_immutable_evidence() -> None:
    plan = load(PLAN)
    assert plan["schema_version"] == "mai_124m_attention_qk_only_lwt_20tpp_plan_v1"
    assert sha256(CONFIG) == plan["candidate"]["config_sha256"]
    assert sha256(PARENT) == plan["candidate"]["parent_config_sha256"]
    for record in plan["immutable_evidence"].values():
        assert sha256(ROOT / record["path"]) == record["sha256"]
    assert plan["identity"]["dataset_manifest_sha256"] == (
        "1e1de075c504906a93637bd79450d30da2243797d2e1d3e33f2392d9492ddf8b"
    )


def test_candidate_changes_only_horizon_and_registered_metadata() -> None:
    parent = load(PARENT)
    candidate = load(CONFIG)
    allowed = {
        "max_iters",
        "lr_decay_iters",
        "warmup_iters",
        "eval_interval",
        "planned_tokens",
        "scheduled_tokens",
        "planned_tpp",
        "scheduled_tpp",
        "out_dir",
        "hpo_stage",
        "ladder_role",
        "ladder_slot",
        "confirmation_slot",
        "confirmation_source",
        "candidate_scope",
        "practical_equivalence_policy",
        "recipe_resolution_stage",
        "operator_override",
        "dense_fixed_validation_curve",
        "parent_fixed_validation_curve",
        "practical_equivalence_nll",
        "selected_lwt_allocation",
        "qk_only_5tpp_result",
        "qkv_only_20tpp_result",
        "attention_activation_manifold_result",
        "residual_write_joint_5tpp_result",
    }
    changed = {
        key
        for key in set(parent) | set(candidate)
        if parent.get(key) != candidate.get(key)
    }
    assert changed == allowed
    assert candidate["max_iters"] == 9489
    assert candidate["eval_interval"] == 2373
    assert candidate["warmup_iters"] == 94
    assert candidate["planned_tpp"] == 20.0


def test_functional_lwt_scope_and_training_invariants() -> None:
    parent = load(PARENT)
    candidate = load(CONFIG)
    assert candidate["block_fht_targets"] == ["attn.c_attn.qk_headwise"]
    assert candidate["selected_lwt_allocation"]["generated"] == [
        "attn.c_attn.qk_headwise"
    ]
    assert candidate["selected_lwt_allocation"]["dense_muon"] == [
        "attn.c_attn.v",
        "attn.c_proj",
    ]
    assert candidate.get("block_fht_mlp_cfc_directed_product", False) is False
    assert candidate.get("block_fht_mlp_cproj_muon_matched_givens", False) is False
    for key in (
        "learning_rate",
        "min_lr",
        "batch_size",
        "gradient_accumulation_steps",
        "block_size",
        "optimizer",
        "muon_momentum",
        "muon_ns_steps",
        "muon_adamw_lr_scale",
        "weight_decay",
        "model_seed",
        "train_data_seed",
        "data_manifest_sha256",
        "block_fht_attn_cayley_ranks",
        "block_fht_attn_cayley_lr_scale",
        "block_fht_output_gain_targets",
    ):
        assert candidate[key] == parent[key]
    with patch.object(sys, "argv", ["train.py", "--config", str(CONFIG)]):
        args = parse_args()
    assert args.max_iters == 9489
    assert args.block_fht_targets == ["attn.c_attn.qk_headwise"]


def test_frozen_gate_and_monitoring_policy() -> None:
    plan = load(PLAN)
    rule = plan["decision_rule"]
    assert rule["terminal_validation_ce_maximum"] == 3.1747
    assert rule["minimum_terminal_improvement_over_qkv"] == 0.015
    assert rule["maximum_fixed_curve_gap_to_qkv"] == 0.0
    assert rule["threshold_changed_after_measurement"] is False
    gate = plan["performance_gate"]
    assert gate["minimum_mfu_fraction"] == 0.2
    assert gate["direct_foreground_polling"] is True
    assert gate["watchdog"] is False
    monitoring = plan["monitoring"]
    assert monitoring["callbacks"][:3] == [20, 50, 100]
    assert monitoring["heartbeat_minutes"] == 90
    assert monitoring["heartbeat_resets_after_progress_callback"] is True
    assert "send-opencode-test" in monitoring["callback_endpoint"]
