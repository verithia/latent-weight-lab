import hashlib
import json

from examples.nanogpt.make_pro6_qk_cfc_20tpp_phase_acquisition import (
    PHASES,
    OUTPUT_CONFIG,
    OUTPUT_PLAN,
    SOURCE_CONFIG,
    VERIFIER,
    make_config,
    make_plan,
)


def test_acquisition_changes_only_observational_fields() -> None:
    source = json.loads(SOURCE_CONFIG.read_text())
    candidate = make_config(source)
    allowed = {
        "out_dir", "mfu_preflight_certificate", "hpo_stage", "ladder_role",
        "candidate_scope", "registered_plan", "implementation_source_hashes",
        "trajectory_snapshot_interval", "trajectory_snapshot_targets",
        "trajectory_snapshot_dtype", "trajectory_snapshot_layers",
        "trajectory_snapshot_all_parameters", "trajectory_snapshot_all_buffers",
        "estimated_trajectory_payload_bytes", "trajectory_acquisition_provenance",
        "diagnostic_protocol", "mfu_measurement_protocol", "monitoring_policy",
        "selection_endpoint", "operator_override", "launch_ready",
        "launch_block_reason", "screen_only", "terminal_eval_required",
    }
    changed = {key for key in set(source) | set(candidate) if source.get(key) != candidate.get(key)}
    assert changed <= allowed
    assert candidate["trajectory_snapshot_interval"] == 2373
    assert candidate["trajectory_snapshot_all_parameters"] is True
    assert candidate["trajectory_snapshot_all_buffers"] is True


def test_plan_is_fail_closed_and_has_exact_phases() -> None:
    plan = make_plan("a" * 64)
    assert plan["full_state_contract"]["expected_snapshot_steps"] == PHASES
    assert plan["authorization"]["one_exact_replay_after_mfu_pass"] is True
    assert plan["authorization"]["candidate_structure_implementation"] is False
    assert plan["acceptance"]["threshold_changes_after_measurement"] is False


def test_generated_identity_is_self_consistent() -> None:
    plan = json.loads(OUTPUT_PLAN.read_text())
    assert hashlib.sha256(OUTPUT_CONFIG.read_bytes()).hexdigest() == plan["identity"][
        "candidate_config_sha256"
    ]
    assert hashlib.sha256(VERIFIER.read_bytes()).hexdigest() == plan["identity"][
        "verifier_sha256"
    ]
    assert plan["identity"]["source_config_sha256"] == (
        "26613d1136d68be8412e3172ca90894a41b9ecc60ca8ac69842e303fb2a504b2"
    )
