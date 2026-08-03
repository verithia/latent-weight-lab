from __future__ import annotations

import hashlib

from examples.nanogpt.make_pro6_mlp_cfc_directed_product_mfu_config import (
    DATASET_MANIFEST_SHA256,
    IMPLEMENTATION_COMMIT,
    SELECTION_RESULT_SHA256,
    json_bytes,
    make_config,
    make_plan,
    validate_inputs,
)


def test_inputs_and_selected_production_contract_are_pinned() -> None:
    validate_inputs()
    config = make_config()
    assert config["block_fht_mlp_cfc_functional_shear"] is False
    assert config["block_fht_mlp_cfc_directed_product"] is True
    assert config["block_fht_mlp_cfc_directed_product_schedule"] == [30, 29, 29]
    assert config["block_fht_mlp_cfc_directed_product_family_radius_ratio"] == (
        0.6589686140591383
    )
    assert config["implementation_commit"] == IMPLEMENTATION_COMMIT
    assert config["parent_selection_result_sha256"] == SELECTION_RESULT_SHA256
    assert config["mfu_preflight_required"] is True
    assert config["mfu_min_fraction"] == 0.2


def test_plan_is_directly_polled_and_does_not_authorize_training() -> None:
    config = make_config()
    config_sha256 = hashlib.sha256(json_bytes(config)).hexdigest()
    plan = make_plan(config_sha256)
    assert plan["candidate"]["config_sha256"] == config_sha256
    assert plan["identity"]["dataset_manifest_sha256"] == (
        DATASET_MANIFEST_SHA256
    )
    assert plan["protocol"]["host"] == "PRO6"
    assert plan["protocol"]["timed_updates"] == 8
    assert plan["protocol"]["minimum_mfu_fraction"] == 0.2
    assert plan["protocol"]["watchdog"] is False
    assert plan["protocol"]["callback"] is False
    assert plan["authorization"]["scientific_training_authorized"] is False
