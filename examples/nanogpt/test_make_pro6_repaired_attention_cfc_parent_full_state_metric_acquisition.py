from __future__ import annotations

import json

from examples.nanogpt.make_pro6_repaired_attention_cfc_parent_full_state_metric_acquisition import (
    BASE_CONFIG_SHA256,
    INVALID_CALIBRATION_AUDIT,
    INVALID_CALIBRATION_AUDIT_SHA256,
    OUTPUT_CONFIG,
    OUTPUT_PLAN,
    PHASE_BOUNDARIES,
    SOURCE_CONFIG,
    make_config,
    make_plan,
    json_bytes,
    sha256_bytes,
    sha256_file,
)


def test_full_state_config_changes_only_observational_fields() -> None:
    source = json.loads(SOURCE_CONFIG.read_text())
    candidate = make_config(source)
    assert candidate["max_iters"] == source["max_iters"] == 2373
    assert candidate["eval_interval"] == source["eval_interval"] == 594
    assert candidate["trajectory_snapshot_all_parameters"] is True
    assert candidate["trajectory_snapshot_all_buffers"] is True
    assert candidate["functional_metric_acquisition_provenance"][
        "scientific_settings_changed"
    ] is False
    assert candidate["functional_metric_acquisition_provenance"][
        "accepted_v2_config_sha256"
    ] == BASE_CONFIG_SHA256
    assert sha256_file(INVALID_CALIBRATION_AUDIT) == (
        INVALID_CALIBRATION_AUDIT_SHA256
    )


def test_full_state_plan_freezes_functional_replay_gate() -> None:
    source = json.loads(SOURCE_CONFIG.read_text())
    config = make_config(source)
    digest = sha256_bytes(json_bytes(config))
    plan = make_plan(digest)
    assert plan["identity"]["candidate_config_sha256"] == digest
    assert plan["full_state_contract"]["phase_boundaries"] == PHASE_BOUNDARIES
    assert plan["full_state_contract"]["persistent_buffer_count_expected"] == 24
    assert plan["acquisition_acceptance"][
        "functional_replay_absolute_tolerance_ce"
    ] == 0.005
    assert plan["acquisition_acceptance"][
        "threshold_changes_after_measurement"
    ] is False
    assert not plan["authorization"]["candidate_structure_implementation"]
    assert not plan["authorization"]["candidate_language_model_training"]
    assert OUTPUT_CONFIG.name.endswith("fullstate_metric_probe_5tpp.json")
    assert OUTPUT_PLAN.name.endswith("full_state_metric_acquisition_v3_plan.json")
