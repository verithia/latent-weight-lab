from __future__ import annotations

import json

from examples.nanogpt.make_pro6_repaired_attention_cfc_parent_nonintervening_metric_acquisition import (
    OUTPUT_CONFIG,
    OUTPUT_PLAN,
    PHASE_BOUNDARIES,
    PROBE_LAYERS,
    PROBE_STEPS,
    REJECTED_ACQUISITION_RESULT_SHA256,
    SOURCE_CONFIG,
    SOURCE_CONFIG_SHA256,
    make_config,
    make_plan,
    sha256_bytes,
    sha256_file,
    json_bytes,
)


def test_config_changes_only_observational_fields() -> None:
    source = json.loads(SOURCE_CONFIG.read_text())
    assert sha256_file(SOURCE_CONFIG) == SOURCE_CONFIG_SHA256
    candidate = make_config(source)
    assert candidate["max_iters"] == source["max_iters"] == 2373
    assert candidate["eval_interval"] == source["eval_interval"] == 594
    assert candidate["optimizer_probe_steps"] == PROBE_STEPS
    assert candidate["optimizer_probe_layers"] == PROBE_LAYERS
    assert candidate["trajectory_snapshot_all_parameters"] is True
    assert candidate["functional_metric_acquisition_provenance"][
        "scientific_settings_changed"
    ] is False
    assert "No alternate pre-step GPU matrix kernel" in candidate[
        "diagnostic_protocol"
    ]


def test_plan_freezes_equivalence_and_authorization() -> None:
    source = json.loads(SOURCE_CONFIG.read_text())
    config = make_config(source)
    config_sha256 = sha256_bytes(json_bytes(config))
    plan = make_plan(config_sha256)
    assert plan["identity"]["candidate_config_sha256"] == config_sha256
    assert plan["identity"]["rejected_acquisition_result_sha256"] == (
        REJECTED_ACQUISITION_RESULT_SHA256
    )
    assert plan["acquisition"]["phase_boundaries"] == PHASE_BOUNDARIES
    assert plan["nonintervention_gate"]["schema_version"] == (
        "nanogpt_optimizer_probe_v2"
    )
    assert plan["acquisition_acceptance"][
        "threshold_changes_after_measurement"
    ] is False
    assert plan["authorization"]["one_corrected_acquisition_after_all_gates_pass"]
    assert not plan["authorization"]["candidate_structure_implementation"]
    assert not plan["authorization"]["candidate_language_model_training"]
    assert OUTPUT_CONFIG.name.endswith("nonintervening_metric_probe_5tpp.json")
    assert OUTPUT_PLAN.name.endswith("functional_metric_acquisition_v2_plan.json")
