import hashlib
import json

from examples.nanogpt.make_pro6_attention_v_int8_lattice_error_feedback_124m_20tpp import (
    BASE,
    CAYLEY_20TPP,
    OUTPUT,
    PLAN,
    QK_20TPP,
    RESULT_5TPP,
    build_config,
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_horizon_transfer_preserves_v_mechanism_and_optimizer_recipe() -> None:
    parent = json.loads(BASE.read_text())
    candidate = build_config()
    for key in (
        "block_fht_attn_v_int8_lattice",
        "block_fht_attn_v_int8_lattice_block_size",
        "block_fht_attn_v_int8_lattice_seed",
        "block_fht_attn_v_int8_lattice_error_feedback",
        "block_fht_attn_cayley_ranks",
        "block_fht_attn_cayley_lr_scale",
        "block_fht_attn_cayley_scale",
        "block_fht_targets",
        "block_fht_output_gain_targets",
        "learning_rate",
        "min_lr",
        "optimizer",
        "muon_momentum",
        "muon_ns_steps",
        "weight_decay",
        "batch_size",
        "gradient_accumulation_steps",
        "model_seed",
        "train_data_seed",
        "data_manifest_sha256",
        "fixed_eval_index_spec_sha256",
    ):
        assert candidate[key] == parent[key]
    assert candidate["max_iters"] == 9489
    assert candidate["lr_decay_iters"] == 9489
    assert candidate["warmup_iters"] == 94
    assert candidate["eval_interval"] == 2373
    assert candidate["planned_tpp"] == 20.0
    assert candidate["mfu_preflight_required"] is True
    assert candidate["mfu_min_fraction"] >= 0.20


def test_plan_pins_evidence_and_freezes_qk_relative_curve_gate() -> None:
    plan = json.loads(PLAN.read_text())
    assert sha256(OUTPUT) == plan["candidate"]["config_sha256"]
    assert sha256(RESULT_5TPP) == plan["evidence"]["v_repair_5tpp"]["sha256"]
    assert sha256(QK_20TPP) == plan["evidence"]["qk_only_20tpp"]["sha256"]
    assert sha256(CAYLEY_20TPP) == plan["evidence"]["cayley_qkv_20tpp"]["sha256"]
    gate = plan["terminal_gate"]
    assert gate["maximum_validation_ce"] == 3.1688
    assert gate["maximum_delta_to_qk_only_ce"] == 0.02
    assert gate["minimum_improvement_over_cayley_qkv_ce"] == 0.0165
    assert gate["maximum_delta_to_qk_only_at_every_fixed_evaluation_ce"] == 0.02
    mfu_result = OUTPUT.parent / (
        "selection_artifacts/"
        "124m_attention_v_int8_lattice_errorfeedback_20tpp_mfu_result.json"
    )
    assert sha256(mfu_result) == plan["mfu_result"]["sha256"]
    assert plan["mfu_result"]["mfu_fraction"] >= 0.20
    assert plan["authorization"]["combined_full_replacement"] is False
    assert plan["authorization"]["larger_model"] is False
