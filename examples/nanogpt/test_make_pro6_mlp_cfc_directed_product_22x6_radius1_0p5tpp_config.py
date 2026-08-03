from __future__ import annotations

import hashlib

from examples.nanogpt.make_pro6_mlp_cfc_directed_product_22x6_radius1_0p5tpp_config import (
    MAX_ITERS,
    RADIUS,
    SCREEN_RESULT_SHA256,
    json_bytes,
    make_config,
    make_plan,
    validate_inputs,
)


def test_full_radius_config_preserves_selected_structure_and_schedule() -> None:
    validate_inputs()
    config = make_config()
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
    ] == RADIUS
    assert config["max_iters"] == MAX_ITERS
    assert config["lr_decay_iters"] == MAX_ITERS
    assert config["parent_selection_result_sha256"] == SCREEN_RESULT_SHA256
    assert config["mfu_preflight_required"] is True
    assert config["preregistered_decision_rule"]["success_ce_maximum"] == 5.5918


def test_plan_requires_exact_mfu_and_one_foreground_run() -> None:
    config = make_config()
    digest = hashlib.sha256(json_bytes(config)).hexdigest()
    plan = make_plan(digest)
    assert plan["candidate"]["config_sha256"] == digest
    assert plan["candidate"]["radius"] == 1.0
    assert plan["protocol"]["watchdog"] is False
    assert plan["protocol"]["callback"] is False
    assert plan["authorization"]["training_requires_exact_config_mfu_pass"] is True
    assert plan["authorization"]["automatic_rerun_authorized"] is False
    assert plan["decision_rule"]["success"].endswith("<= 5.5918")
