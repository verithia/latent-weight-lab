from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PLAN = REPO / "examples/nanogpt/configs/selection_artifacts/124m_repaired_attention_cproj_activation_energy_metric_5tpp_plan.json"
CONFIG = REPO / "examples/nanogpt/configs/pro6_mai_v3_124m_repairedfullattn_plus_cprojdecay0p5_activationenergymetric_5tpp_lr24e4.json"
CONTROL = REPO / "examples/nanogpt/configs/pro6_mai_v3_124m_repairedfullattn_plus_cprojdecay0p5_5tpp_lr24e4.json"


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


def test_causal_inputs_are_hash_pinned() -> None:
    plan = load(PLAN)
    for item in plan["causal_evidence"].values():
        assert sha256(REPO / item["path"]) == item["sha256"]


def test_candidate_is_one_bounded_hidden_metric_change() -> None:
    plan = load(PLAN)
    change = plan["single_factor_change"]
    metric = plan["metric_definition"]
    safety = plan["safety_against_prior_failure"]
    assert change["enabled_flag"] == "block_fht_mlp_cproj_activation_energy_metric"
    assert metric["ema_decay"] == 0.95
    assert metric["minimum_weight"] == 0.25
    assert metric["maximum_weight"] == 4.0
    assert metric["epsilon"] == 1e-6
    assert safety["not_output_side"] is True
    assert safety["not_full_activation_covariance"] is True
    assert safety["bounded_metric_condition_number"] == 16.0
    assert "zero additional trainable parameters and zero inference-time transforms" in change["unchanged"]


def test_gate_and_monitoring_are_frozen() -> None:
    plan = load(PLAN)
    performance = plan["performance_gate"]
    rule = plan["decision_rule"]
    monitoring = plan["scientific_run"]["monitoring"]
    assert performance["minimum_mfu_fraction"] >= 0.20
    assert performance["protocol"].startswith("Foreground")
    assert monitoring["milestones"] == [20, 50, 100]
    assert monitoring["heartbeat_minutes"] == 90
    assert monitoring["progress_resets_heartbeat"] is True
    assert monitoring["callback_endpoint"].endswith("/send-opencode-test")
    assert monitoring["callback_mention"] == "@Codex"
    assert monitoring["terminal_delivery_once"] is True
    assert rule["promote_terminal_validation_ce_maximum"] == 3.6478
    assert rule["threshold_changed_after_measurement"] is False


def test_only_one_run_is_authorized() -> None:
    authorization = load(PLAN)["authorization"]
    assert authorization["implementation_and_tests"] is True
    assert authorization["one_exact_config_mfu_gate"] is True
    assert authorization["one_scientific_5tpp_run_after_gate"] is True
    assert authorization["automatic_rerun"] is False
    assert authorization["parallel_arm"] is False
    assert authorization["larger_rung"] is False
    assert authorization["post_hoc_metric_sweep"] is False


def test_production_config_is_the_registered_single_factor_candidate() -> None:
    plan = load(PLAN)
    candidate = load(CONFIG)
    control = load(CONTROL)
    assert candidate["registered_plan"] == str(PLAN.relative_to(REPO))
    assert candidate["registered_plan_sha256"] == sha256(PLAN)
    assert candidate["cproj_control_result_sha256"] == (
        plan["causal_evidence"]["cproj_only_terminal_result"]["sha256"]
    )
    metric = plan["metric_definition"]
    assert candidate["block_fht_mlp_cproj_activation_energy_metric"] is True
    assert candidate["block_fht_mlp_cproj_activation_energy_metric_decay"] == metric["ema_decay"]
    assert candidate["block_fht_mlp_cproj_activation_energy_metric_minimum"] == metric["minimum_weight"]
    assert candidate["block_fht_mlp_cproj_activation_energy_metric_maximum"] == metric["maximum_weight"]
    assert candidate["block_fht_mlp_cproj_activation_energy_metric_epsilon"] == metric["epsilon"]
    frozen = (
        "n_layer", "n_head", "n_embd", "block_size", "batch_size",
        "gradient_accumulation_steps", "max_iters", "warmup_iters",
        "eval_interval", "eval_iters", "learning_rate", "min_lr",
        "optimizer", "weight_decay", "model_seed", "train_data_seed",
        "data_manifest_sha256", "eval_seed", "fixed_eval_index_spec_sha256",
        "block_fht_targets", "block_fht_attn_cayley_targets",
        "block_fht_attn_cayley_ranks", "block_fht_attn_cayley_scale",
        "block_fht_attn_cayley_lr_scale",
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
        assert candidate[key] == control[key], key
    assert candidate["implementation_commit"] == "edc11023f7428ee3d3214cb3afb6a4f656e62475"
    launch_commit = candidate["implementation_commit"]
    for path, digest in candidate["implementation_source_hashes"].items():
        assert git_blob_sha256(launch_commit, path) == digest
