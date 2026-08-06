from __future__ import annotations

import json

from examples.nanogpt.make_pro6_repaired_attention_cfc_late_cproj_paired_state_trajectory import (
    INVALID_CROSS_RUN_RESULT,
    INVALID_CROSS_RUN_RESULT_SHA256,
    OUTPUT_CONFIG,
    OUTPUT_PLAN,
    POST_STEP_SNAPSHOTS,
    PROBE_LAYERS,
    PROBE_STEPS,
    SNAPSHOT_STEPS,
    SOURCE_CONFIG,
    SOURCE_CONFIG_SHA256,
    json_bytes,
    make_config,
    make_plan,
    sha256_bytes,
    sha256_file,
)


def test_config_changes_only_same_run_diagnostics() -> None:
    assert sha256_file(SOURCE_CONFIG) == SOURCE_CONFIG_SHA256
    assert sha256_file(INVALID_CROSS_RUN_RESULT) == INVALID_CROSS_RUN_RESULT_SHA256
    source = json.loads(SOURCE_CONFIG.read_text())
    candidate = make_config(source)
    for key in (
        "block_fht_targets",
        "block_fht_mlp_cproj_muon_matched_givens_layers",
        "learning_rate",
        "muon_adamw_lr_scale",
        "max_iters",
        "eval_interval",
        "model_seed",
        "train_data_seed",
        "batch_size",
        "gradient_accumulation_steps",
    ):
        assert candidate.get(key) == source.get(key)
    assert candidate["trajectory_snapshot_interval"] == 99
    assert candidate["trajectory_snapshot_targets"] == ["mlp.c_proj"]
    assert candidate["trajectory_snapshot_layers"] == PROBE_LAYERS
    assert candidate["optimizer_probe_steps"] == PROBE_STEPS
    assert candidate["trajectory_acquisition_provenance"][
        "scientific_settings_changed"
    ] is False


def test_plan_freezes_same_run_pairing_and_terminal_only_monitor() -> None:
    candidate = make_config(json.loads(SOURCE_CONFIG.read_text()))
    plan = make_plan(sha256_bytes(json_bytes(candidate)))
    assert plan["targeted_snapshot_contract"]["steps"] == SNAPSHOT_STEPS
    assert plan["optimizer_state_contract"]["post_step_snapshot_steps"] == POST_STEP_SNAPSHOTS
    assert plan["nonintervention_gate"]["same_run_identity_required"]
    assert plan["performance_gate"]["required_snapshot_step_in_preflight"] == 0
    assert plan["performance_gate"]["required_probe_step_in_preflight"] == 0
    assert plan["monitoring"]["milestones"] is False
    assert plan["monitoring"]["heartbeats"] is False
    assert not plan["authorization"]["candidate_language_model_training"]
    assert OUTPUT_CONFIG.name.endswith("pairedstate_trajectory_5tpp.json")
    assert OUTPUT_PLAN.name.endswith("paired_state_trajectory_acquisition_plan.json")
