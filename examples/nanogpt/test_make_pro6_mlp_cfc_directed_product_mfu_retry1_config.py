from __future__ import annotations

import hashlib

from examples.nanogpt.make_pro6_mlp_cfc_directed_product_mfu_retry1_config import (
    FAILED_RESULT_SHA256,
    IMPLEMENTATION_COMMIT,
    PYTHON,
    json_bytes,
    make_config,
    make_plan,
    validate_inputs,
)


def test_retry_keeps_structure_and_requires_native_backend() -> None:
    validate_inputs()
    config = make_config()
    assert config["block_fht_mlp_cfc_directed_product_schedule"] == [30, 29, 29]
    assert config["block_fht_mlp_cfc_directed_product_family_radius_ratio"] == (
        0.6589686140591383
    )
    assert config["block_fht_native_extension_required"] is True
    assert config["implementation_commit"] == IMPLEMENTATION_COMMIT
    assert config["failed_mfu_preflight"]["result_sha256"] == (
        FAILED_RESULT_SHA256
    )
    assert config["runtime_environment"]["python"] == PYTHON


def test_retry_plan_is_foreground_only_and_training_unauthorized() -> None:
    config = make_config()
    config_sha256 = hashlib.sha256(json_bytes(config)).hexdigest()
    plan = make_plan(config_sha256)
    assert plan["candidate"]["config_sha256"] == config_sha256
    assert plan["candidate"]["scientific_structure_changed_from_failed_attempt"] is False
    assert plan["runtime_preconditions"]["block_fht_native_extension_required"] is True
    assert plan["protocol"]["watchdog"] is False
    assert plan["protocol"]["callback"] is False
    assert plan["authorization"]["scientific_training_authorized"] is False
