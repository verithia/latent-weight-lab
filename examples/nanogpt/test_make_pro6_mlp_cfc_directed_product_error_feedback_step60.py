from __future__ import annotations

import hashlib

from examples.nanogpt.make_pro6_mlp_cfc_directed_product_error_feedback_step60 import (
    ADDITIONAL_OPTIMIZER_STATE_BYTES,
    PASS_VALIDATION_CE,
    SCHEDULE,
    json_bytes,
    make_config,
    make_plan,
    validate_inputs,
)


def test_error_feedback_screen_changes_only_temporal_compression_contract() -> None:
    validate_inputs()
    config = make_config()
    assert config["block_fht_mlp_cfc_directed_product_schedule"] == SCHEDULE
    assert config["block_fht_mlp_cfc_directed_product_family_radius_ratio"] == 1.0
    assert config["block_fht_mlp_cfc_directed_product_error_feedback"] is True
    assert config["block_fht_mlp_cfc_directed_product_error_feedback_decay"] == 1.0
    assert config["max_iters"] == 60
    assert config["lr_decay_iters"] == 238
    assert config["model_seed"] == 1337
    assert config["preregistered_decision_rule"]["pass_validation_ce_maximum"] == (
        PASS_VALIDATION_CE
    )
    assert config["directed_product_representation"][
        "additional_dense_optimizer_state_bytes"
    ] == ADDITIONAL_OPTIMIZER_STATE_BYTES
    assert config["directed_product_representation"][
        "additional_trainable_parameters"
    ] == 0


def test_error_feedback_plan_is_frozen_and_does_not_authorize_full_run() -> None:
    config_hash = hashlib.sha256(json_bytes(make_config())).hexdigest()
    plan = make_plan(config_hash)
    assert plan["identity"]["config_sha256"] == config_hash
    assert plan["candidate"]["error_feedback"] is True
    assert plan["candidate"]["forward_structure_changed"] is False
    assert plan["decision_rule"]["pass_validation_ce_maximum"] == (
        PASS_VALIDATION_CE
    )
    assert plan["execution"]["exact_config_mfu_minimum"] == 0.2
    assert plan["execution"]["watchdog"] is False
    assert plan["execution"]["callback"] is False
    assert plan["authorization"]["full_238_update_run_authorized"] is False
