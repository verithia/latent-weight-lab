from __future__ import annotations

import hashlib

from examples.nanogpt.make_pro6_full_mlp_hybrid_state_0p5tpp import (
    DENSE_FP32_STATE_BYTES,
    PERSISTENT_STATE_BYTES,
    json_bytes,
    make_config,
    make_plan,
    validate_inputs,
)


def test_hybrid_state_config_changes_only_persistent_optimizer_storage() -> None:
    validate_inputs()
    config = make_config()
    assert config["block_fht_mlp_muon_momentum_state_dtype"] == "float16"
    assert config["block_fht_mlp_error_feedback_state_codec"] == "int8_blockwise"
    assert config["block_fht_mlp_error_feedback_state_block_size"] == 4096
    assert config["max_iters"] == 238
    assert config["mfu_preflight_required"] is True
    representation = config["temporal_state_representation"]
    assert representation["persistent_storage_bytes"] == PERSISTENT_STATE_BYTES
    assert representation["dense_fp32_storage_bytes"] == DENSE_FP32_STATE_BYTES
    assert representation["storage_ratio"] < 0.38


def test_hybrid_state_plan_requires_mfu_memory_gate_and_terminal_watchdog() -> None:
    config = make_config()
    plan = make_plan(hashlib.sha256(json_bytes(config)).hexdigest())
    assert plan["candidate"]["config_sha256"] == hashlib.sha256(
        json_bytes(config)
    ).hexdigest()
    assert plan["authorization"]["training_before_preflight_pass"] is False
    assert plan["authorization"]["one_124m_training_after_preflight_pass"] is True
    assert plan["authorization"]["larger_rung"] is False
    assert plan["protocol"]["preflight_monitor"] == "foreground polling"
    assert "terminal/error-only" in plan["protocol"]["training_monitor"]
