from __future__ import annotations

import hashlib

from examples.nanogpt.make_pro6_mlp_cfc_directed_product_error_feedback_0p5tpp import (
    MAX_ITERS,
    SCREEN_RESULT_SHA256,
    json_bytes,
    make_config,
    make_plan,
    validate_inputs,
)


def test_full_error_feedback_config_changes_only_registered_rung_contract() -> None:
    validate_inputs()
    config = make_config()
    assert config["block_fht_mlp_cfc_directed_product_schedule"] == [22] * 6
    assert config["block_fht_mlp_cfc_directed_product_family_radius_ratio"] == 1.0
    assert config["block_fht_mlp_cfc_directed_product_error_feedback"] is True
    assert config["block_fht_mlp_cfc_directed_product_error_feedback_decay"] == 1.0
    assert config["max_iters"] == MAX_ITERS
    assert config["lr_decay_iters"] == MAX_ITERS
    assert config["parent_selection_result_sha256"] == SCREEN_RESULT_SHA256
    assert config["mfu_preflight_required"] is True
    assert config["preregistered_decision_rule"]["success_ce_maximum"] == 5.5918
    assert config["directed_product_representation"][
        "additional_dense_optimizer_state_bytes"
    ] == 113_246_208


def test_full_error_feedback_plan_requires_exact_mfu_and_one_direct_run() -> None:
    config = make_config()
    digest = hashlib.sha256(json_bytes(config)).hexdigest()
    plan = make_plan(digest)
    assert plan["candidate"]["config_sha256"] == digest
    assert plan["candidate"]["error_feedback"] is True
    assert plan["protocol"]["watchdog"] is False
    assert plan["protocol"]["callback"] is False
    assert plan["authorization"]["training_requires_exact_config_mfu_pass"] is True
    assert plan["authorization"]["training_authorized_before_exact_mfu"] is False
    assert plan["authorization"]["automatic_rerun_authorized"] is False
    assert plan["authorization"]["larger_rung_authorized"] is False
    assert plan["decision_rule"]["success"].endswith("<= 5.5918")
