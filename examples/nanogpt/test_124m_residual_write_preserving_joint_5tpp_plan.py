from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from examples.nanogpt.train import parse_args


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / (
    "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_residual_write_preserving_joint_5tpp_lr24e4.json"
)
PLAN = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_residual_write_preserving_joint_5tpp_plan.json"
)
ACCOUNTING_CORRECTION = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_residual_write_preserving_joint_5tpp_accounting_correction.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config() -> dict[str, object]:
    return json.loads(CONFIG.read_text())


def test_plan_identity_and_frozen_thresholds() -> None:
    config = load_config()
    plan = json.loads(PLAN.read_text())
    assert config["resolved_from_plan_sha256"] == sha256(PLAN)
    assert config["accounting_correction_sha256"] == sha256(
        ACCOUNTING_CORRECTION
    )
    assert plan["decision_rule"]["terminal_validation_ce_maximum"] == 3.5248
    assert plan["decision_rule"]["terminal_gap_to_qkv_parent_maximum"] == 0.01
    assert plan["decision_rule"]["maximum_fixed_curve_gap_to_qkv_parent"] == 0.015
    assert plan["decision_rule"]["threshold_changed_after_measurement"] is False
    assert plan["authorization"]["automatic_rerun"] is False
    assert plan["authorization"]["larger_rung"] is False
    assert plan["monitoring"]["milestone_callbacks"] is False
    assert plan["monitoring"]["heartbeat_callbacks"] is False


def test_only_feature_forming_maps_are_procedural() -> None:
    config = load_config()
    expected_attention = ["attn.c_attn.qk_headwise", "attn.c_attn.v"]
    assert config["block_fht_targets"] == expected_attention
    assert config["block_fht_attn_cayley_targets"] == expected_attention
    assert config["block_fht_output_gain_targets"] == expected_attention
    assert "attn.c_proj" not in config["block_fht_targets"]
    assert "mlp.c_proj" not in config["block_fht_targets"]
    assert config.get("block_fht_mlp_cproj_muon_matched_givens", False) is False
    assert config["block_fht_mlp_cfc_directed_product"] is True
    assert config["block_fht_mlp_cfc_directed_product_schedule"] == [22] * 6
    assert config["block_fht_mlp_cfc_directed_product_error_feedback"] is True
    assert config["block_fht_mlp_cfc_directed_product_error_feedback_decay"] == 1


def test_scientific_recipe_and_runtime_policy_are_frozen() -> None:
    config = load_config()
    assert config["learning_rate"] == 0.0024
    assert config["muon_adamw_lr_scale"] == 0.3
    assert config["muon_momentum"] == 0.95
    assert config["max_iters"] == 2373
    assert config["eval_interval"] == 594
    assert config["eval_iters"] == 400
    assert config["mfu_min_fraction"] >= 0.20
    assert config["block_fht_native_extension_required"] is True
    assert config["registered_resume_determinism_required"] is True
    assert config["checkpoint_wall_clock_seconds"] == 7200
    assert config["expected_registered_trainable_parameters"] == 79197288
    assert config["cfc_component_parameter_reduction"] == 0
    assert config["inference_parameter_reduction"] == 0
    assert config["inference_flop_reduction"] == 0


def test_exact_scientific_json_passes_argument_validation(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(CONFIG)])
    parsed = parse_args()
    assert parsed.block_fht_targets == [
        "attn.c_attn.qk_headwise",
        "attn.c_attn.v",
    ]
    assert parsed.block_fht_mlp_cfc_directed_product is True
    assert parsed.block_fht_mlp_cproj_muon_matched_givens is False
