from __future__ import annotations

import hashlib
import json

from examples.nanogpt.make_pro6_350m_full_mlp_hybrid_state_0p5tpp import (
    BASE,
    DENSE_FP32_STATE_BYTES,
    PERSISTENT_STATE_BYTES,
    json_bytes,
    make_config,
    make_plan,
    validate_inputs,
)


def test_350m_hybrid_state_preserves_accepted_scientific_recipe() -> None:
    validate_inputs()
    base = json.loads(BASE.read_text())
    config = make_config()
    for key in (
        "n_layer",
        "n_head",
        "n_embd",
        "learning_rate",
        "max_iters",
        "tokens_per_iter",
        "block_fht_targets",
        "block_fht_mlp_cfc_directed_product_schedule",
        "block_fht_mlp_cfc_directed_product_error_feedback_decay",
        "block_fht_mlp_cproj_muon_matched_givens_stages",
        "block_fht_mlp_cproj_muon_matched_givens_residual_stages",
        "block_fht_mlp_cproj_muon_matched_givens_error_feedback_decay",
    ):
        assert config[key] == base[key]
    assert config["block_fht_mlp_muon_momentum_state_dtype"] == "float16"
    assert config["block_fht_mlp_error_feedback_state_codec"] == "int8_blockwise"
    assert config["block_fht_mlp_error_feedback_state_block_size"] == 4096


def test_350m_hybrid_state_accounting_and_gate_are_frozen() -> None:
    config = make_config()
    representation = config["temporal_state_representation"]
    assert representation["persistent_storage_bytes"] == PERSISTENT_STATE_BYTES
    assert representation["dense_fp32_storage_bytes"] == DENSE_FP32_STATE_BYTES
    assert representation["storage_ratio"] == 0.37506103515625
    plan = make_plan(hashlib.sha256(json_bytes(config)).hexdigest())
    assert plan["candidate"]["config_sha256"] == hashlib.sha256(
        json_bytes(config)
    ).hexdigest()
    assert plan["authorization"]["training_before_preflight_pass"] is False
    assert plan["authorization"]["one_350m_training_after_preflight_pass"] is True
    assert plan["authorization"]["larger_rung"] is False
    assert plan["decision_rule"]["scientific_success"].endswith("<= 4.4629")
    assert "terminal/error-only" in plan["protocol"]["training_monitor"]
