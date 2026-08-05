import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "examples/nanogpt/configs/pro6_mai_v3_124m_repairedfullattn_plus_fullmlp_cfcdecay1_cprojdecay0p5_5tpp_lr24e4.json"
PARENT = REPO / "examples/nanogpt/configs/pro6_mai_v3_124m_fullattn_cayley_horizon_capacity_qk32_v16_cproj8_targeted_bilateral_fullcayleylr_5tpp_lr24e4.json"
PLAN = REPO / "examples/nanogpt/configs/selection_artifacts/124m_repaired_attention_full_mlp_5tpp_plan.json"
MFU_RESULT = REPO / "examples/nanogpt/configs/selection_artifacts/124m_repaired_attention_full_mlp_5tpp_mfu_result.json"

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def test_identity_and_evidence_are_hash_pinned() -> None:
    plan = json.loads(PLAN.read_text())
    assert sha256(CONFIG) == plan["identity"]["config_sha256"]
    for path_key, hash_key in (
        ("attention_parent_config", "attention_parent_config_sha256"),
        ("attention_parent_result", "attention_parent_result_sha256"),
        ("smallest_full_mlp_result", "smallest_full_mlp_result_sha256"),
        ("full_mlp_350m_result", "full_mlp_350m_result_sha256"),
        ("full_mlp_690m_result", "full_mlp_690m_result_sha256"),
    ):
        assert sha256(REPO / plan["identity"][path_key]) == plan["identity"][hash_key]

def test_attention_parent_is_preserved() -> None:
    config = json.loads(CONFIG.read_text())
    parent = json.loads(PARENT.read_text())
    for key in (
        "n_layer", "n_head", "n_embd", "block_size", "batch_size",
        "gradient_accumulation_steps", "max_iters", "warmup_iters",
        "eval_interval", "eval_iters", "learning_rate", "min_lr",
        "optimizer", "weight_decay", "model_seed", "train_data_seed",
        "data_manifest_sha256", "eval_seed", "fixed_eval_index_spec_sha256",
        "block_fht_attn_cayley_targets", "block_fht_attn_cayley_ranks",
        "block_fht_attn_cayley_bilateral_targets",
        "block_fht_attn_cayley_output_targets", "block_fht_attn_cayley_scale",
        "block_fht_attn_cayley_lr_scale",
    ):
        assert config[key] == parent[key]
    assert config["scheduled_tokens"] == 622067712
    assert config["planned_tpp"] == 5.0

def test_full_mlp_and_frozen_gates_are_exact() -> None:
    config = json.loads(CONFIG.read_text())
    plan = json.loads(PLAN.read_text())
    assert config["block_fht_targets"] == ["attn.c_attn.qk_headwise", "attn.c_attn.v", "attn.c_proj", "mlp.c_proj"]
    assert config["block_fht_mlp_cfc_directed_product_schedule"] == [22] * 6
    assert config["block_fht_mlp_cfc_directed_product_error_feedback_decay"] == 1.0
    assert config["block_fht_mlp_cproj_muon_matched_givens_stages"] == 64
    assert config["block_fht_mlp_cproj_muon_matched_givens_residual_stages"] == 24
    assert config["block_fht_mlp_cproj_muon_matched_givens_error_feedback_decay"] == 0.5
    assert config["mfu_preflight_required"] is True
    assert config["block_fht_native_extension_required"] is True
    assert config["mfu_min_fraction"] >= 0.20
    assert plan["decision_rule"]["pass_validation_ce_maximum"] == 3.6478
    assert plan["decision_rule"]["threshold_changes_after_measurement"] is False
    assert plan["authorization"]["automatic_rerun"] is False
    assert config["additional_trainable_parameters_vs_attention_parent"] == 0
    assert config["additional_inference_flops_vs_materialized_attention_parent"] == 0

def test_long_run_monitoring_is_registered() -> None:
    plan = json.loads(PLAN.read_text())
    monitoring = plan["execution"]["monitoring"]
    assert monitoring["milestones"] == [20, 50, 100]
    assert monitoring["heartbeat_minutes"] == 90
    assert monitoring["progress_resets_heartbeat"] is True
    assert monitoring["terminal_callback_once"] is True
    assert monitoring["callback_endpoint"].endswith("/send-opencode-test")
    assert monitoring["callback_mention"] == "@Codex"

def test_exact_config_mfu_pass_is_bound_to_candidate() -> None:
    result = json.loads(MFU_RESULT.read_text())
    assert result["identity"]["config_sha256"] == sha256(CONFIG)
    assert result["decision"]["passed"] is True
    assert result["measurement"]["mfu_fraction"] >= 0.20
    assert result["measurement"]["native_block_fht_extension"]["loaded"] is True
    assert result["execution"]["watchdog_used"] is False
