from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from examples.nanogpt.train import parse_args


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "examples/nanogpt/configs/pro6_mai_v3_124m_qk_only_plus_cfc_directed_20tpp_lr24e4.json"
PARENT = ROOT / "examples/nanogpt/configs/pro6_mai_v3_124m_qk_only_qk64_outputgain_20tpp_lr24e4.json"
PLAN = ROOT / "examples/nanogpt/configs/selection_artifacts/124m_qk_only_plus_cfc_directed_20tpp_plan.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_plan_binds_config_parent_and_evidence() -> None:
    plan = load(PLAN)
    assert plan["status"] == "registered_before_performance_preflight_and_training"
    assert plan["candidate"]["config_sha256"] == sha256(CONFIG)
    assert plan["candidate"]["parent_config_sha256"] == sha256(PARENT)
    for artifact in plan["immutable_evidence"].values():
        assert sha256(ROOT / artifact["path"]) == artifact["sha256"]


def test_candidate_is_exact_accepted_composition() -> None:
    config = load(CONFIG)
    parent = load(PARENT)
    assert config["block_fht_targets"] == ["attn.c_attn.qk_headwise"]
    assert config["block_fht_attn_cayley_ranks"] == {"attn.c_attn.qk_headwise": 64}
    assert config["block_fht_mlp_cfc_directed_product"] is True
    assert config["block_fht_mlp_cfc_directed_product_schedule"] == [22] * 6
    assert config["block_fht_mlp_cfc_directed_product_error_feedback"] is True
    assert config["block_fht_mlp_cfc_directed_product_error_feedback_decay"] == 1
    assert config.get("block_fht_mlp_cproj_muon_matched_givens", False) is False
    assert config["selected_lwt_allocation"]["dense_muon"] == [
        "attn.c_attn.v", "attn.c_proj", "mlp.c_proj"
    ]
    for key in (
        "learning_rate", "min_lr", "batch_size", "gradient_accumulation_steps",
        "block_size", "optimizer", "muon_momentum", "muon_ns_steps",
        "muon_adamw_lr_scale", "weight_decay", "model_seed", "train_data_seed",
        "data_manifest_sha256", "max_iters", "eval_interval", "eval_iters",
    ):
        assert config[key] == parent[key]


def test_frozen_gate_accounting_and_monitoring() -> None:
    config = load(CONFIG)
    plan = load(PLAN)
    rule = plan["decision_rule"]
    assert rule["terminal_validation_ce_maximum"] == 3.1538
    assert rule["maximum_fixed_curve_gap_to_qk_only_parent"] == 0.005
    assert rule["threshold_changed_after_measurement"] is False
    assert config["expected_registered_trainable_parameters"] == 113916912
    assert config["cfc_component_parameter_reduction"] == 0
    assert config["inference_parameter_reduction"] == 0
    assert config["inference_flop_reduction"] == 0
    assert plan["monitoring"]["callbacks"][:3] == [20, 50, 100]
    assert plan["monitoring"]["heartbeat_minutes"] == 90
    assert plan["monitoring"]["heartbeat_resets_after_progress_callback"] is True
    assert plan["authorization"]["automatic_rerun"] is False


def test_exact_config_passes_train_argument_validation(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(CONFIG)])
    parsed = parse_args()
    assert parsed.max_iters == 9489
    assert parsed.block_fht_targets == ["attn.c_attn.qk_headwise"]
    assert parsed.block_fht_mlp_cfc_directed_product is True
    assert parsed.block_fht_mlp_cproj_muon_matched_givens is False
