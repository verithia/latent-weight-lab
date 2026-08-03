from __future__ import annotations

import hashlib

from examples.nanogpt.make_pro6_mlp_cfc_directed_product_2x_error_feedback_step60 import (
    COORDINATES_PER_LAYER,
    PASS_VALIDATION_CE,
    SCHEDULE,
    json_bytes,
    make_config,
    make_plan,
    validate_inputs,
)


def test_2x_error_feedback_screen_changes_only_coordinate_reach() -> None:
    validate_inputs()
    config = make_config()
    assert config["block_fht_mlp_cfc_directed_product_schedule"] == SCHEDULE
    assert config["block_fht_mlp_cfc_directed_product_family_radius_ratio"] == 1.0
    assert config["block_fht_mlp_cfc_directed_product_error_feedback"] is True
    assert config["block_fht_mlp_cfc_directed_product_error_feedback_decay"] == 1.0
    assert config["max_iters"] == 60
    assert config["lr_decay_iters"] == 238
    assert config["directed_product_representation"][
        "coordinates_per_layer"
    ] == COORDINATES_PER_LAYER
    assert config["preregistered_decision_rule"][
        "pass_validation_ce_maximum"
    ] == PASS_VALIDATION_CE


def test_2x_error_feedback_plan_freezes_half_gap_rule_and_no_full_run() -> None:
    config = make_config()
    digest = hashlib.sha256(json_bytes(config)).hexdigest()
    plan = make_plan(digest)
    assert plan["candidate"]["config_sha256"] == digest
    assert plan["candidate"]["incoming_schedule"] == SCHEDULE
    assert plan["candidate"]["error_feedback"] is True
    assert plan["decision_rule"]["pass_validation_ce_maximum"] == PASS_VALIDATION_CE
    assert plan["execution"]["watchdog"] is False
    assert plan["execution"]["callback"] is False
    assert plan["authorization"]["full_238_update_run_authorized"] is False
    assert plan["authorization"]["automatic_rerun_authorized"] is False
