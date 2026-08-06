from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from examples.nanogpt.train import parse_args


REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / (
    "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_repairedfullattn_plus_fullmlp_"
    "volumecproj_5tpp_lr24e4.json"
)
PARENT = REPO / (
    "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_repairedfullattn_plus_fullmlp_"
    "cfcdecay1_cprojdecay0p5_5tpp_lr24e4_v2.json"
)
PLAN = REPO / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_repaired_attention_full_mlp_volume_cproj_5tpp_plan.json"
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


def test_evidence_plan_and_implementation_are_hash_pinned() -> None:
    plan = load(PLAN)
    config = load(CONFIG)
    assert config["registered_plan_sha256"] == sha256(PLAN)
    for evidence in plan["causal_evidence"].values():
        path = evidence.get("path")
        if path is not None:
            assert sha256(REPO / path) == evidence["sha256"]
    commit = config["implementation_commit"]
    for relative, digest in config["implementation_source_hashes"].items():
        assert git_blob_sha256(commit, relative) == digest


def test_candidate_preserves_parent_and_adds_only_volume_coordinate() -> None:
    candidate = load(CONFIG)
    parent = load(PARENT)
    for key in (
        "n_layer",
        "n_head",
        "n_embd",
        "block_size",
        "batch_size",
        "gradient_accumulation_steps",
        "max_iters",
        "warmup_iters",
        "eval_interval",
        "eval_iters",
        "learning_rate",
        "min_lr",
        "optimizer",
        "weight_decay",
        "model_seed",
        "train_data_seed",
        "data_manifest_sha256",
        "eval_seed",
        "fixed_eval_index_spec_sha256",
        "block_fht_targets",
        "block_fht_attn_cayley_targets",
        "block_fht_attn_cayley_ranks",
        "block_fht_attn_cayley_bilateral_targets",
        "block_fht_attn_cayley_output_targets",
        "block_fht_attn_cayley_scale",
        "block_fht_attn_cayley_lr_scale",
        "block_fht_mlp_cfc_directed_product_schedule",
        "block_fht_mlp_cfc_directed_product_error_feedback_decay",
        "block_fht_mlp_cproj_muon_matched_givens_stages",
        "block_fht_mlp_cproj_muon_matched_givens_residual_stages",
        "block_fht_mlp_cproj_muon_matched_givens_error_feedback_decay",
        "checkpoint_wall_clock_seconds",
    ):
        assert candidate[key] == parent[key], key
    assert candidate["block_fht_mlp_cproj_output_symmetric_shear_stages"] == 0
    assert candidate["block_fht_mlp_cproj_global_log_volume"] is True
    assert candidate[
        "block_fht_mlp_cproj_global_log_volume_max_abs"
    ] == 0.009950330853168092
    representation = candidate["muon_matched_givens_representation"]
    assert representation["coordinates_per_layer"] == 135169
    assert representation["coordinates_all_12_layers"] == 1622028
    assert representation["additional_persistent_trainable_coordinates"] == 0
    assert candidate["additional_trainable_parameters_vs_attention_parent"] == 0
    assert candidate[
        "additional_inference_flops_vs_materialized_attention_parent"
    ] == 0


def test_exact_config_is_accepted_by_production_parser() -> None:
    with patch.object(sys, "argv", ["train.py", "--config", str(CONFIG)]):
        parsed = parse_args()
    assert parsed.block_fht_mlp_cproj_muon_matched_givens is True
    assert parsed.block_fht_mlp_cproj_muon_matched_givens_stages == 64
    assert parsed.block_fht_mlp_cproj_muon_matched_givens_residual_stages == 24
    assert parsed.block_fht_mlp_cproj_output_symmetric_shear_stages == 0
    assert parsed.block_fht_mlp_cproj_global_log_volume is True
    assert parsed.block_fht_mlp_cproj_global_log_volume_max_abs == (
        0.009950330853168092
    )
    assert parsed.block_fht_mlp_cproj_muon_matched_givens_fast_fresh is True
    assert parsed.block_fht_native_extension_required is True


def test_frozen_gate_and_terminal_only_monitoring_policy() -> None:
    plan = load(PLAN)
    candidate = load(CONFIG)
    assert candidate["mfu_preflight_required"] is True
    assert candidate["mfu_min_fraction"] >= 0.20
    scientific = plan["scientific_run"]
    assert scientific["primary_terminal_validation_ce_maximum"] == (
        3.635838041305542
    )
    assert scientific["secondary_terminal_validation_ce_maximum"] == 3.6478
    assert scientific["threshold_changed_after_measurement"] is False
    monitoring = scientific["monitoring"]
    assert monitoring["healthy_progress_callbacks"] is False
    assert monitoring["healthy_heartbeat_callbacks"] is False
    assert monitoring["callback_on_clean_completion"] is True
    assert monitoring["callback_on_actionable_error"] is True
    assert monitoring["callback_endpoint"].endswith("/send-opencode-test")
    assert monitoring["callback_mention"] == "@Codex"
    assert monitoring["terminal_delivery_once"] is True
    authorization = plan["authorization"]
    assert authorization["automatic_rerun"] is False
    assert authorization["parallel_arm"] is False
    assert authorization["larger_rung"] is False
