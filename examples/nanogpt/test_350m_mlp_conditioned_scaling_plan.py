from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PLAN_PATH = (
    REPO
    / "examples/nanogpt/configs/selection_artifacts/350m_mlp_conditioned_scaling_plan.json"
)
VALIDATOR_CONTRACT_PATH = (
    REPO
    / "examples/nanogpt/configs/selection_artifacts/350m_mlp_functional_shear_stability_validator_contract.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    assert isinstance(value, dict)
    return value


def test_350m_conditioned_scaling_plan_binds_every_input() -> None:
    plan = load(PLAN_PATH)
    assert plan["schema_version"] == "350m_conditioned_full_mlp_scaling_plan_v1"
    assert plan["order"] == [
        "cproj_hidden88_parent",
        "conditioned_full_mlp_candidate",
    ]
    assert plan["execution"]["host"] == "PRO6"
    assert plan["execution"]["performance_gate_owner"].startswith("foreground")

    stages = plan["stages"]
    configs = {}
    for name in plan["order"]:
        stage = stages[name]
        path = REPO / stage["config"]
        assert path.is_file()
        assert sha256(path) == stage["config_sha256"]
        configs[name] = load(path)

    parent = configs["cproj_hidden88_parent"]
    candidate = configs["conditioned_full_mlp_candidate"]
    shared = (
        "batch_size",
        "beta1",
        "beta2",
        "bias",
        "block_fht_latent_init_std",
        "block_fht_latent_ratio",
        "block_fht_layers",
        "block_fht_match_gpt_init",
        "block_fht_mlp_cproj_muon_matched_givens",
        "block_fht_mlp_cproj_muon_matched_givens_fast_fresh",
        "block_fht_mlp_cproj_muon_matched_givens_neighbors",
        "block_fht_mlp_cproj_muon_matched_givens_refresh_interval",
        "block_fht_mlp_cproj_muon_matched_givens_residual_stages",
        "block_fht_mlp_cproj_muon_matched_givens_seed",
        "block_fht_mlp_cproj_muon_matched_givens_stages",
        "block_fht_targets",
        "block_size",
        "compile",
        "data_dir",
        "data_manifest_sha256",
        "dropout",
        "dtype",
        "eval_batch_size",
        "eval_interval",
        "eval_iters",
        "eval_protocol_id",
        "eval_seed",
        "fixed_eval_index_spec_sha256",
        "fixed_eval_indices",
        "gradient_accumulation_steps",
        "learning_rate",
        "lr_decay_iters",
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
        "planned_tokens",
        "planned_tpp",
        "scheduled_tokens",
        "tokens_per_iter",
        "train_data_seed",
        "vocab_size",
        "warmup_iters",
        "weight_decay",
    )
    assert {field: parent[field] for field in shared} == {
        field: candidate[field] for field in shared
    }
    assert parent["max_iters"] == 677
    assert parent["tokens_per_iter"] == 262144
    assert parent["scheduled_tokens"] == parent["max_iters"] * parent["tokens_per_iter"]

    assert not parent.get("block_fht_mlp_cfc_functional_shear", False)
    assert candidate["block_fht_mlp_cfc_functional_shear"] is True
    assert candidate["block_fht_mlp_cfc_functional_shear_beta"] == 0.5
    assert candidate["block_fht_mlp_cfc_functional_shear_max_condition_number"] == 1.01
    assert candidate["block_fht_mlp_cfc_functional_shear_weight_norm_projection"] is False

    control = plan["historical_attention_control"]
    control_config = REPO / control["config"]
    ranking = REPO / control["ranking_artifact"]
    assert sha256(control_config) == control["config_sha256"]
    assert sha256(ranking) == control["ranking_artifact_sha256"]
    assert control["terminal_validation_ce"] == 4.3629
    assert parent["preregistered_decision_rule"]["success"].find("4.4629") >= 0
    assert candidate["preregistered_decision_rule"]["attention_only_absolute_ce_ceiling"] == 4.5629

    for relative, expected in plan["implementation"]["source_hashes"].items():
        assert sha256(REPO / relative) == expected
    assert (
        parent["data_manifest_sha256"]
        == candidate["data_manifest_sha256"]
        == plan["dataset"]["manifest_sha256"]
    )


def test_candidate_admission_is_sequential_and_performance_gated() -> None:
    plan = load(PLAN_PATH)
    parent = plan["stages"]["cproj_hidden88_parent"]
    candidate = plan["stages"]["conditioned_full_mlp_candidate"]
    assert parent["admission_dependency"] == "none"
    assert "must finish cleanly" in candidate["admission_dependency"]
    assert "<= 4.4629" in candidate["admission_dependency"]
    assert "--warmup-updates 1 --timed-updates 8" in parent["performance_command"]
    assert "MUON_FUNCTIONAL_SHEAR_DIAGNOSTIC_STEPS=25" in candidate["performance_command"]
    assert "--warmup-updates 1 --timed-updates 24" in candidate["performance_command"]
    assert "exactly 600 c_fc diagnostic rows" in candidate["performance_rule"]


def test_parent_mfu_result_is_bound_to_plan_and_config() -> None:
    result_path = (
        REPO
        / "examples/nanogpt/configs/selection_artifacts/350m_mlp_cproj_hidden88_mfu_result.json"
    )
    result = load(result_path)
    assert result["schema_version"] == "350m_mlp_performance_result_v1"
    assert result["decision"] == "PASS_MFU_GATE"
    assert result["measurement"]["mfu_fraction"] >= result["threshold"]["minimum_mfu_fraction"]
    assert result["stability"]["all_logged_losses_finite"] is True
    assert result["stability"]["train_exit_code"] == 0
    assert sha256(REPO / result["config"]["path"]) == result["config"]["sha256"]
    assert sha256(REPO / result["plan"]["path"]) == result["plan"]["sha256"]


def test_candidate_stability_validator_contract_is_frozen() -> None:
    contract = load(VALIDATOR_CONTRACT_PATH)
    assert (
        contract["schema_version"]
        == "350m_conditioned_full_mlp_stability_validator_contract_v1"
    )
    assert sha256(REPO / contract["paired_plan"]["path"]) == contract["paired_plan"]["sha256"]
    assert sha256(REPO / contract["candidate"]["config"]) == contract["candidate"]["config_sha256"]
    assert sha256(REPO / contract["validator"]["path"]) == contract["validator"]["sha256"]
    assert contract["paired_plan"]["sha256"] == (
        "406cae6fc38d2ef9b64ed56f612e4d00614a8934cf555ba2342926527cba0615"
    )
    rules = contract["registered_rules"]
    assert rules["expected_layers"] == 24
    assert rules["expected_steps"] == 25
    assert rules["expected_rows"] == 600
    assert rules["maximum_condition_number"] == 1.01
    assert rules["maximum_weight_rms_ratio"] == 2.0
    assert rules["maximum_weight_abs_growth"] == 2.0
    assert rules["maximum_weight_abs_floor"] == 1.0
    assert rules["require_internal_limiter_every_row"] is True
    assert rules["require_zero_fallback"] is True
    command = contract["execution"]["command"]
    assert "--expected-layers 24 --expected-steps 25" in command
    assert "--maximum-condition-number 1.01" in command
    assert "--maximum-weight-rms-ratio 2" in command
    assert "--maximum-weight-abs-growth 2" in command
    assert "--maximum-weight-abs-floor 1" in command


def test_parent_terminal_result_passes_frozen_gate() -> None:
    result_path = (
        REPO
        / "examples/nanogpt/configs/selection_artifacts/350m_mlp_cproj_hidden88_terminal_result.json"
    )
    result = load(result_path)
    assert result["schema_version"] == "350m_mlp_cproj_hidden88_terminal_result_v1"
    assert result["decision"] == "PASS_PARENT_GATE"
    assert sha256(REPO / result["config"]["path"]) == result["config"]["sha256"]
    assert sha256(REPO / result["plan"]["path"]) == result["plan"]["sha256"]
    measurement = result["measurement"]
    assert measurement["terminal_step"] == 677
    assert measurement["terminal_validation_ce"] <= measurement["parent_success_ceiling"]
    assert math.isclose(
        measurement["attention_gap_ce"],
        measurement["terminal_validation_ce"] - measurement["attention_control_validation_ce"],
        abs_tol=1e-12,
    )
    assert math.isclose(
        measurement["success_margin_ce"],
        measurement["parent_success_ceiling"] - measurement["terminal_validation_ce"],
        abs_tol=1e-12,
    )
    assert result["checkpoint"]["next_iter"] == 677
    assert result["checkpoint"]["exact_resume_schema"] == "nanogpt_exact_resume_v2"
    assert result["callback"]["sent_milestones"] == [20, 50]
    assert result["callback"]["terminal_signature"][0:2] == ["finished", 0]
