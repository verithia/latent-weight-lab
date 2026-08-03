from __future__ import annotations

import hashlib

from examples.nanogpt.make_pro6_mlp_cfc_directed_product_radius_step60 import (
    ARMS,
    CONTROL_RADIUS,
    LR_DECAY_ITERS,
    MAX_ITERS,
    PASS_VALIDATION_CE,
    json_bytes,
    make_config,
    make_plan,
    validate_inputs,
)


def test_radius_arms_change_only_registered_screen_contract() -> None:
    validate_inputs()
    expected = {"radius0p82": 0.82, "radius1p00": 1.0}
    for arm, radius in expected.items():
        config = make_config(arm)
        assert config["block_fht_mlp_cfc_directed_product_schedule"] == [
            22,
            22,
            22,
            22,
            22,
            22,
        ]
        assert config[
            "block_fht_mlp_cfc_directed_product_family_radius_ratio"
        ] == radius
        assert config["max_iters"] == MAX_ITERS
        assert config["lr_decay_iters"] == LR_DECAY_ITERS
        assert config["model_seed"] == 1337
        assert config["preregistered_decision_rule"][
            "pass_validation_ce_maximum"
        ] == PASS_VALIDATION_CE
        assert config["monitoring_policy"].startswith("short 60-update")
    assert CONTROL_RADIUS not in expected.values()


def test_plan_requires_mfu_and_direct_polling_without_full_run() -> None:
    hashes = {
        arm: hashlib.sha256(json_bytes(make_config(arm))).hexdigest()
        for arm in ARMS
    }
    plan = make_plan(hashes)
    assert plan["decision_rule"]["pass_validation_ce_maximum"] == 6.3485
    assert plan["protocol"]["exact_config_mfu_minimum"] == 0.2
    assert plan["protocol"]["watchdog"] is False
    assert plan["protocol"]["callback"] is False
    assert plan["authorization"]["full_238_update_run_authorized"] is False
    for arm in ARMS:
        assert plan["arms"][arm]["config_sha256"] == hashes[arm]
        assert plan["arms"][arm]["max_iters"] == 60
