from __future__ import annotations

import json

import pytest
import torch

from examples.nanogpt.make_pro6_repaired_attention_cfc_late_cproj_full_state_trajectory import (
    CAPTURE_INTERVAL,
    EXPECTED_SNAPSHOT_STEPS,
    OUTPUT_CONFIG,
    OUTPUT_PLAN,
    PHASE_BOUNDARIES,
    SOURCE_CONFIG,
    SOURCE_CONFIG_SHA256,
    SOURCE_RESULT_SHA256,
    json_bytes,
    make_config,
    make_plan,
    sha256_bytes,
    sha256_file,
)
from examples.nanogpt.verify_late_cproj_full_state_trajectory import (
    compare_terminal_checkpoint,
    expected_snapshot_steps,
    validate_snapshot,
)
from examples.nanogpt.parameter_trajectory import FULL_STATE_SCHEMA_VERSION


def test_config_changes_only_observational_state_capture() -> None:
    source = json.loads(SOURCE_CONFIG.read_text())
    assert sha256_file(SOURCE_CONFIG) == SOURCE_CONFIG_SHA256
    candidate = make_config(source)
    scientific_keys = (
        "block_fht_targets",
        "block_fht_mlp_cproj_muon_matched_givens_layers",
        "block_fht_mlp_cproj_muon_matched_givens_stages",
        "block_fht_mlp_cproj_muon_matched_givens_residual_stages",
        "block_fht_mlp_cproj_muon_matched_givens_error_feedback_decay",
        "learning_rate",
        "muon_adamw_lr_scale",
        "max_iters",
        "eval_interval",
        "eval_iters",
        "eval_seed",
        "model_seed",
        "train_data_seed",
        "micro_batch_size",
        "gradient_accumulation_steps",
    )
    for key in scientific_keys:
        assert candidate.get(key) == source.get(key)
    assert candidate["trajectory_snapshot_interval"] == CAPTURE_INTERVAL
    assert candidate["trajectory_snapshot_all_parameters"] is True
    assert candidate["trajectory_snapshot_all_buffers"] is True
    assert candidate.get("optimizer_probe_steps") is None
    assert candidate["trajectory_acquisition_provenance"][
        "scientific_settings_changed"
    ] is False
    changed = {
        key
        for key in set(source) | set(candidate)
        if source.get(key) != candidate.get(key)
        or (key in source) != (key in candidate)
    }
    assert changed == {
        "candidate_scope",
        "diagnostic_acquisition_plan",
        "diagnostic_caveat",
        "diagnostic_protocol",
        "estimated_trajectory_payload_bytes",
        "hpo_stage",
        "implementation_source_hashes",
        "ladder_role",
        "ladder_slot",
        "mfu_measurement_protocol",
        "mfu_preflight_certificate",
        "monitoring_policy",
        "operator_override",
        "out_dir",
        "registered_plan",
        "registered_plan_sha256",
        "selection_endpoint",
        "state_capture_registration_parent_commit",
        "trajectory_acquisition_provenance",
        "trajectory_snapshot_all_buffers",
        "trajectory_snapshot_all_parameters",
        "trajectory_snapshot_dtype",
        "trajectory_snapshot_interval",
        "trajectory_snapshot_layers",
        "trajectory_snapshot_targets",
    }


def test_plan_freezes_curve_replay_and_terminal_only_monitoring() -> None:
    candidate = make_config(json.loads(SOURCE_CONFIG.read_text()))
    config_sha = sha256_bytes(json_bytes(candidate))
    plan = make_plan(config_sha)
    assert plan["identity"]["candidate_config_sha256"] == config_sha
    assert plan["identity"]["source_result_sha256"] == SOURCE_RESULT_SHA256
    assert plan["full_state_contract"]["expected_snapshot_steps"] == (
        EXPECTED_SNAPSHOT_STEPS
    )
    assert plan["full_state_contract"]["required_functional_replay_steps"] == (
        PHASE_BOUNDARIES
    )
    assert plan["acceptance"]["curve_absolute_tolerance_ce"] == 0.005
    assert plan["acceptance"][
        "functional_replay_absolute_tolerance_ce"
    ] == 0.005
    assert plan["monitoring"]["milestones"] is False
    assert plan["monitoring"]["heartbeats"] is False
    assert plan["monitoring"]["callback_endpoint"].endswith(
        "/send-opencode-test"
    )
    assert not plan["authorization"]["candidate_language_model_training"]
    assert OUTPUT_CONFIG.name.endswith("fullstate_trajectory_5tpp.json")
    assert OUTPUT_PLAN.name.endswith("full_state_trajectory_acquisition_plan.json")


def test_registered_snapshot_cadence_is_exact() -> None:
    assert expected_snapshot_steps(2373, CAPTURE_INTERVAL) == (
        EXPECTED_SNAPSHOT_STEPS
    )
    assert len(EXPECTED_SNAPSHOT_STEPS) == 25
    for step in PHASE_BOUNDARIES:
        assert step in EXPECTED_SNAPSHOT_STEPS


def test_snapshot_validation_is_strict_about_inventory_and_finiteness() -> None:
    payload = {
        "schema_version": FULL_STATE_SCHEMA_VERSION,
        "all_parameters": True,
        "all_buffers": True,
        "step": 99,
        "parameters": {"weight": torch.tensor([1.0, 2.0])},
        "buffers": {"mask": torch.tensor([1], dtype=torch.int64)},
    }
    parameter_names, buffer_names = validate_snapshot(
        payload,
        expected_step=99,
        expected_parameter_names=None,
        expected_buffer_names=None,
    )
    assert parameter_names == {"weight"}
    assert buffer_names == {"mask"}

    payload["parameters"]["weight"] = torch.tensor([float("nan")])
    with pytest.raises(ValueError, match="invalid parameter tensor"):
        validate_snapshot(
            payload,
            expected_step=99,
            expected_parameter_names={"weight"},
            expected_buffer_names={"mask"},
        )


def test_terminal_snapshot_must_equal_checkpoint(tmp_path) -> None:
    snapshot = {
        "parameters": {"weight": torch.tensor([1.0, 2.0])},
        "buffers": {"mask": torch.tensor([1], dtype=torch.int64)},
    }
    checkpoint = tmp_path / "ckpt.pt"
    torch.save(
        {"model": {"weight": snapshot["parameters"]["weight"], "mask": snapshot["buffers"]["mask"]}},
        checkpoint,
    )
    assert compare_terminal_checkpoint(snapshot, checkpoint) == {
        "compared_parameters": 1,
        "compared_buffers": 1,
    }

    torch.save(
        {"model": {"weight": torch.tensor([1.0, 3.0]), "mask": snapshot["buffers"]["mask"]}},
        checkpoint,
    )
    with pytest.raises(ValueError, match="terminal checkpoint mismatch"):
        compare_terminal_checkpoint(snapshot, checkpoint)
