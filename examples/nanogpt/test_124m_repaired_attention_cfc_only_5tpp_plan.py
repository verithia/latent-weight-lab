from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from examples.nanogpt.train import parse_args


REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / (
    "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_repairedfullattn_plus_cfconly_decay1_5tpp_lr24e4.json"
)
JOINT = REPO / (
    "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_repairedfullattn_plus_fullmlp_"
    "cfcdecay1_cprojdecay0p5_5tpp_lr24e4_v2.json"
)
PLAN = REPO / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_repaired_attention_cfc_only_5tpp_plan.json"
)
MFU_RESULT = REPO / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_repaired_attention_cfc_only_5tpp_mfu_result.json"
)


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    assert isinstance(value, dict)
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_blob_sha256(commit: str, path: str) -> str:
    payload = subprocess.check_output(
        ["git", "-C", str(REPO), "show", f"{commit}:{path}"]
    )
    return hashlib.sha256(payload).hexdigest()


def test_causal_evidence_and_implementation_are_hash_pinned() -> None:
    plan = load(PLAN)
    for evidence in plan["causal_evidence"].values():
        for path_key, hash_key in (
            ("path", "sha256"),
            ("config", "config_sha256"),
            ("result", "result_sha256"),
        ):
            if path_key in evidence:
                assert sha256(REPO / evidence[path_key]) == evidence[hash_key]
    config = load(CONFIG)
    assert config["registered_plan_sha256"] == sha256(PLAN)
    commit = config["implementation_commit"]
    for path, digest in config["implementation_source_hashes"].items():
        assert git_blob_sha256(commit, path) == digest


def test_candidate_is_exact_cfc_only_factorial_arm() -> None:
    candidate = load(CONFIG)
    joint = load(JOINT)
    removed_cproj_keys = {
        key for key in joint if key.startswith("block_fht_mlp_cproj_")
    }
    assert removed_cproj_keys
    assert all(key not in candidate for key in removed_cproj_keys)
    assert candidate["block_fht_targets"] == [
        "attn.c_attn.qk_headwise",
        "attn.c_attn.v",
        "attn.c_proj",
    ]
    assert candidate["block_fht_mlp_cfc_directed_product"] is True
    assert candidate["block_fht_mlp_cfc_directed_product_schedule"] == [22] * 6
    assert candidate["block_fht_mlp_cfc_directed_product_family_radius_ratio"] == 1
    assert candidate["block_fht_mlp_cfc_directed_product_ridge_ratio"] == 1e-6
    assert candidate["block_fht_mlp_cfc_directed_product_error_feedback"] is True
    assert candidate["block_fht_mlp_cfc_directed_product_error_feedback_decay"] == 1
    frozen = (
        "n_layer", "n_head", "n_embd", "block_size", "batch_size",
        "gradient_accumulation_steps", "max_iters", "warmup_iters",
        "eval_interval", "eval_iters", "learning_rate", "min_lr",
        "optimizer", "weight_decay", "model_seed", "train_data_seed",
        "data_manifest_sha256", "eval_seed", "fixed_eval_index_spec_sha256",
        "block_fht_attn_cayley_targets", "block_fht_attn_cayley_ranks",
        "block_fht_attn_cayley_bilateral_targets",
        "block_fht_attn_cayley_output_targets", "block_fht_attn_cayley_scale",
        "block_fht_attn_cayley_lr_scale", "checkpoint_wall_clock_seconds",
    )
    for key in frozen:
        assert candidate[key] == joint[key], key
    assert candidate["additional_dense_optimizer_state_bytes"] == 12 * 768 * 3072 * 4
    assert candidate["additional_inference_parameters_vs_materialized_attention_parent"] == 0
    assert candidate["additional_inference_flops_vs_materialized_attention_parent"] == 0


def test_production_parser_accepts_dense_control_but_not_generated_unqualified_cproj(
    tmp_path: Path,
) -> None:
    with patch.object(sys, "argv", ["train.py", "--config", str(CONFIG)]):
        parsed = parse_args()
    assert parsed.block_fht_mlp_cfc_directed_product is True
    assert parsed.block_fht_mlp_cproj_muon_matched_givens is False
    assert "mlp.c_proj" not in parsed.block_fht_targets

    invalid = load(CONFIG)
    invalid["block_fht_targets"] = [*invalid["block_fht_targets"], "mlp.c_proj"]
    invalid_path = tmp_path / "unqualified_generated_cproj.json"
    invalid_path.write_text(json.dumps(invalid))
    with patch.object(
        sys, "argv", ["train.py", "--config", str(invalid_path)]
    ):
        with pytest.raises(ValueError, match="either dense c_proj"):
            parse_args()


def test_gate_monitoring_and_authorization_are_frozen() -> None:
    plan = load(PLAN)
    config = load(CONFIG)
    rule = plan["decision_rule"]
    assert rule["primary_terminal_validation_ce_maximum"] == 3.6478
    assert rule["nonbinding_near_parent_validation_ce_maximum"] == 3.6378
    assert rule["threshold_changed_after_measurement"] is False
    assert config["max_iters"] == config["lr_decay_iters"] == 2373
    assert config["mfu_preflight_required"] is True
    assert config["mfu_min_fraction"] >= 0.20
    command = plan["execution"]["exact_mfu_command"]
    assert command[command.index("--warmup-updates") + 1] == "1"
    assert command[command.index("--timed-updates") + 1] == "8"
    monitoring = plan["scientific_run"]["monitoring"]
    assert monitoring["milestones"] == [20, 50, 100]
    assert monitoring["heartbeat_minutes"] == 90
    assert monitoring["progress_resets_heartbeat"] is True
    assert monitoring["callback_endpoint"].endswith("/send-opencode-test")
    assert monitoring["callback_mention"] == "@Codex"
    assert monitoring["terminal_delivery_once"] is True
    authorization = plan["authorization"]
    assert authorization["automatic_rerun"] is False
    assert authorization["parallel_arm"] is False
    assert authorization["larger_rung"] is False


def test_exact_config_mfu_gate_authorizes_one_scientific_run() -> None:
    result = load(MFU_RESULT)
    assert result["classification"] == "CFC_ONLY_124M_5TPP_EXACT_CONFIG_MFU_PASSED"
    assert result["passed"] is True
    assert sha256(REPO / result["identity"]["config"]) == result["identity"]["config_sha256"]
    assert sha256(REPO / result["identity"]["plan"]) == result["identity"]["plan_sha256"]
    assert result["measurement"]["mfu_fraction"] >= 0.20
    assert result["measurement"]["native_block_fht_extension"]["loaded"] is True
    assert result["structural_verification"]["cfc_directed_product_active"] is True
    assert result["structural_verification"]["cproj_dense_muon_active"] is True
    assert result["structural_verification"]["cproj_generated_target_absent"] is True
    assert result["stability"]["all_logged_losses_finite"] is True
    assert result["execution"]["direct_foreground_polling"] is True
    assert result["execution"]["watchdog_used"] is False
    assert result["decision"]["one_scientific_2373_update_run_authorized"] is True
    assert result["decision"]["automatic_rerun_authorized"] is False
