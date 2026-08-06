from __future__ import annotations

import json

import torch

from examples.nanogpt.make_pro6_repaired_attention_cfc_late_cproj_optimizer_state_trajectory import (
    OUTPUT_CONFIG,
    OUTPUT_PLAN,
    PROBE_LAYERS,
    PROBE_STEPS,
    REFERENCE_POST_STEPS,
    SOURCE_CONFIG,
    SOURCE_CONFIG_SHA256,
    json_bytes,
    make_config,
    make_plan,
    sha256_bytes,
    sha256_file,
)
from examples.nanogpt.parameter_trajectory import OPTIMIZER_PROBE_SCHEMA_VERSION
from examples.nanogpt.verify_late_cproj_optimizer_state_trajectory import (
    compare_post_step_to_reference,
    validate_probe,
)


def test_config_changes_only_observational_optimizer_capture() -> None:
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
        "batch_size",
        "gradient_accumulation_steps",
    )
    for key in scientific_keys:
        assert candidate.get(key) == source.get(key)
    assert candidate["optimizer_probe_steps"] == PROBE_STEPS
    assert candidate["optimizer_probe_layers"] == PROBE_LAYERS
    assert candidate["trajectory_snapshot_interval"] == 0
    assert candidate["trajectory_snapshot_all_parameters"] is False
    assert candidate["trajectory_snapshot_all_buffers"] is False
    assert candidate["trajectory_acquisition_provenance"][
        "scientific_settings_changed"
    ] is False
    allowed = {
        "candidate_scope", "diagnostic_acquisition_plan", "diagnostic_caveat",
        "diagnostic_protocol", "estimated_trajectory_payload_bytes", "hpo_stage",
        "implementation_source_hashes", "ladder_role", "ladder_slot",
        "mfu_measurement_protocol",
        "mfu_preflight_certificate", "monitoring_policy", "operator_override",
        "optimizer_probe_dtype", "optimizer_probe_layers", "optimizer_probe_steps",
        "optimizer_probe_targets", "optimizer_state_registration_parent_commit",
        "out_dir", "registered_plan", "selection_endpoint",
        "trajectory_acquisition_provenance",
        "trajectory_snapshot_all_buffers", "trajectory_snapshot_all_parameters",
        "trajectory_snapshot_interval",
    }
    changed = {
        key for key in set(source) | set(candidate)
        if source.get(key) != candidate.get(key)
        or (key in source) != (key in candidate)
    }
    assert changed == allowed


def test_plan_freezes_probe_phases_theory_and_terminal_monitoring() -> None:
    candidate = make_config(json.loads(SOURCE_CONFIG.read_text()))
    config_sha = sha256_bytes(json_bytes(candidate))
    plan = make_plan(config_sha)
    contract = plan["optimizer_state_contract"]
    assert contract["pre_step_probe_steps"] == PROBE_STEPS
    assert contract["post_step_reference_steps"] == REFERENCE_POST_STEPS
    assert plan["performance_gate"]["required_probe_step_in_preflight"] == 0
    assert plan["preregistered_zero_update_analysis"]["decision_rule"].find("0.80") >= 0
    assert plan["monitoring"]["milestones"] is False
    assert plan["monitoring"]["heartbeats"] is False
    assert not plan["authorization"]["candidate_language_model_training"]
    assert OUTPUT_CONFIG.name.endswith("optimizerstate_trajectory_5tpp.json")
    assert OUTPUT_PLAN.name.endswith("optimizer_state_trajectory_acquisition_plan.json")


def test_probe_validation_and_reference_equality() -> None:
    contract = make_plan("a" * 64)["optimizer_state_contract"]
    name = "transformer.h.8.mlp.c_proj.weight"
    state = {
        "weight_before_step": torch.ones(2, 2),
        "gradient_after_clip": torch.full((2, 2), 0.2),
        "momentum_buffer_before_step": torch.full((2, 2), 0.1),
        "compression_residual_before_step": torch.zeros(2, 2),
        "weight_after_step": torch.full((2, 2), 0.99),
        "combined_momentum_update": torch.full((2, 2), 0.48025),
        "applied_direction_per_lr": torch.full((2, 2), -0.5),
        "momentum_buffer_after_step": torch.full((2, 2), 0.295),
        "compression_residual_after_step": torch.zeros(2, 2),
    }
    parameters = {name: state}
    for layer in [9, 10, 11]:
        parameters[f"transformer.h.{layer}.mlp.c_proj.weight"] = {
            key: value.clone() for key, value in state.items()
        }
    payload = {
        "schema_version": OPTIMIZER_PROBE_SCHEMA_VERSION,
        "step": 98,
        "targets": ["mlp.c_proj"],
        "layers": PROBE_LAYERS,
        "storage_dtype": "float32",
        "run_identity_sha256": "b" * 64,
        "parameters": parameters,
        "hyperparameters": {
            key: {
                "optimizer_kind": "MuonMatchedGivens", "error_feedback": True,
                "error_feedback_decay": 0.5, "momentum": 0.95, "lr": 0.02,
            } for key in parameters
        },
    }
    identity, checked = validate_probe(payload, expected_step=98, contract=contract)
    assert identity == "b" * 64
    compare_post_step_to_reference(
        checked,
        {"buffers": {name: state["weight_after_step"] for name, state in checked.items()}},
    )
