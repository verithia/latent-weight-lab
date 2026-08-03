from __future__ import annotations

import hashlib

from examples.nanogpt.make_pro6_mlp_cfc_directed_product_2x_radius1_step60 import (
    COORDINATES_PER_LAYER,
    LR_DECAY_ITERS,
    MAX_ITERS,
    PASS_VALIDATION_CE,
    SCHEDULE,
    json_bytes,
    make_config,
    make_plan,
    validate_inputs,
)


def test_2x_screen_changes_only_registered_coordinate_contract() -> None:
    validate_inputs()
    config = make_config()
    assert config["block_fht_mlp_cfc_directed_product_schedule"] == SCHEDULE
    assert config["block_fht_mlp_cfc_directed_product_family_radius_ratio"] == 1.0
    assert config["directed_product_representation"]["coordinates_per_layer"] == (
        COORDINATES_PER_LAYER
    )
    assert config["max_iters"] == MAX_ITERS
    assert config["lr_decay_iters"] == LR_DECAY_ITERS
    assert config["model_seed"] == 1337
    assert config["preregistered_decision_rule"]["pass_validation_ce_maximum"] == (
        PASS_VALIDATION_CE
    )
    assert config["monitoring_policy"].startswith("short 60-update")


def test_2x_plan_is_frozen_and_does_not_authorize_full_run() -> None:
    config_hash = hashlib.sha256(json_bytes(make_config())).hexdigest()
    plan = make_plan(config_hash)
    assert plan["identity"]["config_sha256"] == config_hash
    assert plan["candidate"]["schedule"] == SCHEDULE
    assert plan["candidate"]["coordinate_multiplier_vs_original"] == 2.0
    assert plan["decision_rule"]["pass_validation_ce_maximum"] == (
        PASS_VALIDATION_CE
    )
    assert plan["execution"]["exact_config_mfu_minimum"] == 0.2
    assert plan["execution"]["watchdog"] is False
    assert plan["execution"]["callback"] is False
    assert plan["authorization"]["full_238_update_run_authorized"] is False
