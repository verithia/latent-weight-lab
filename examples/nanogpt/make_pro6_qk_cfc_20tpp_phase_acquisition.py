#!/usr/bin/env python3
"""Register a nonintervening phase acquisition of the rejected QK+c_fc run."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "examples/nanogpt/configs/selection_artifacts"
SOURCE_CONFIG = ROOT / "examples/nanogpt/configs/pro6_mai_v3_124m_qk_only_plus_cfc_directed_20tpp_lr24e4.json"
SOURCE_CONFIG_SHA256 = "26613d1136d68be8412e3172ca90894a41b9ecc60ca8ac69842e303fb2a504b2"
SOURCE_RESULT = ARTIFACTS / "124m_qk_only_plus_cfc_directed_20tpp_result.json"
SOURCE_RESULT_SHA256 = "2b7bba0e5928d38b073f7b0734a6516eb12f0f3a01be3a168cef9094f4f6582e"
TERMINAL_AUDIT = ARTIFACTS / "124m_qk_cfc_20tpp_terminal_activation_drift_result.json"
TERMINAL_AUDIT_SHA256 = "50aad2a01ade96a84d9265318c46b00cee1fd41fa7dd879a2d04a1f716ac4876"
OUTPUT_CONFIG = ROOT / "examples/nanogpt/configs/pro6_mai_v3_124m_qk_cfc_phase_acquisition_20tpp.json"
OUTPUT_PLAN = ARTIFACTS / "124m_qk_cfc_20tpp_phase_acquisition_plan.json"
VERIFIER = ROOT / "examples/nanogpt/verify_qk_cfc_20tpp_phase_acquisition.py"
WORKSPACE = Path("/mnt/ssd-data/orj/MappingNetworks")
REMOTE_REPO = WORKSPACE / "latent-weight-lab-cproj-activation-metric"
RUN_NAME = "pro6_mai_v3_124m_qk_cfc_phase_acquisition_20tpp"
OUTPUT_ROOT = WORKSPACE / "outputs/pro6_mai_v3_124m_qk_cfc_phase_acquisition_20tpp"
SCIENTIFIC_OUT = OUTPUT_ROOT / "scientific"
CERTIFICATE = OUTPUT_ROOT / "performance_preflight.json"
CAPTURE_INTERVAL = 2373
PHASES = [0, 2373, 4746, 7119, 9489]
SOURCE_PATHS = (
    "examples/nanogpt/model.py",
    "examples/nanogpt/train.py",
    "examples/nanogpt/muon.py",
    "examples/nanogpt/muon_matched_givens.py",
    "examples/nanogpt/parameter_trajectory.py",
    "examples/nanogpt/mfu_preflight.py",
    "latent_weight_lab/block_fht.py",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def make_config(source: dict[str, Any]) -> dict[str, Any]:
    for path, expected in (
        (SOURCE_CONFIG, SOURCE_CONFIG_SHA256),
        (SOURCE_RESULT, SOURCE_RESULT_SHA256),
        (TERMINAL_AUDIT, TERMINAL_AUDIT_SHA256),
    ):
        if sha256_file(path) != expected:
            raise RuntimeError(f"immutable source drifted: {path}")
    config = copy.deepcopy(source)
    config.pop("registered_plan_sha256", None)
    config.update(
        {
            "out_dir": str(SCIENTIFIC_OUT),
            "mfu_preflight_certificate": str(CERTIFICATE),
            "hpo_stage": "qk_cfc_20tpp_phase_acquisition",
            "ladder_role": "observational_replay_of_rejected_qk_cfc_curve",
            "candidate_scope": (
                "Exact rejected QK+c_fc configuration replayed only to capture "
                "full model state at 0/5/10/15/20TPP; no scientific setting changes."
            ),
            "registered_plan": str(OUTPUT_PLAN.relative_to(ROOT)),
            "implementation_source_hashes": {
                path: sha256_file(ROOT / path) for path in SOURCE_PATHS
            },
            "trajectory_snapshot_interval": CAPTURE_INTERVAL,
            "trajectory_snapshot_targets": [],
            "trajectory_snapshot_dtype": "float32",
            "trajectory_snapshot_layers": None,
            "trajectory_snapshot_all_parameters": True,
            "trajectory_snapshot_all_buffers": True,
            "estimated_trajectory_payload_bytes": 3051008160,
            "trajectory_acquisition_provenance": {
                "classification": "qk_cfc_20tpp_phase_localization_replay",
                "source_config": str(SOURCE_CONFIG.relative_to(ROOT)),
                "source_config_sha256": SOURCE_CONFIG_SHA256,
                "source_result": str(SOURCE_RESULT.relative_to(ROOT)),
                "source_result_sha256": SOURCE_RESULT_SHA256,
                "terminal_audit": str(TERMINAL_AUDIT.relative_to(ROOT)),
                "terminal_audit_sha256": TERMINAL_AUDIT_SHA256,
                "scientific_settings_changed": False,
                "state_capture_change": "five all-parameter/all-persistent-buffer snapshots",
            },
            "diagnostic_protocol": (
                "Snapshot exact phases 0,2373,4746,7119,9489. Require loss-curve "
                "reproduction, exact snapshot inventories, fixed-token functional replay, "
                "and terminal snapshot/checkpoint equality before phase analysis."
            ),
            "mfu_measurement_protocol": (
                "exact config, one warmup plus eight timed real updates, diagnostic I/O "
                "charged, native BlockFHT required, MFU >=0.20"
            ),
            "monitoring_policy": (
                "single persistent aggregate watchdog, 20/50/100 milestones plus one "
                "90-minute heartbeat reset by delivered progress, immediate error/stall"
            ),
            "selection_endpoint": "observational acquisition only; no structure acceptance",
            "operator_override": (
                "2026-08-08: terminal audit localizes directional residual mismatch but "
                "cannot identify its phase; acquire the exact rejected trajectory once."
            ),
            "launch_ready": True,
            "launch_block_reason": None,
            "screen_only": False,
            "terminal_eval_required": True,
        }
    )
    return config


def make_plan(config_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": "mai_124m_qk_cfc_20tpp_phase_acquisition_plan_v1",
        "status": "registered_before_preflight_or_execution",
        "recorded_at": "2026-08-08",
        "scientific_question": (
            "At which fixed phase does the QK+c_fc residual-direction geometry drift "
            "from its own early trajectory as the CE gap emerges after 10TPP?"
        ),
        "identity": {
            "source_config": str(SOURCE_CONFIG.relative_to(ROOT)),
            "source_config_sha256": SOURCE_CONFIG_SHA256,
            "source_result": str(SOURCE_RESULT.relative_to(ROOT)),
            "source_result_sha256": SOURCE_RESULT_SHA256,
            "terminal_audit": str(TERMINAL_AUDIT.relative_to(ROOT)),
            "terminal_audit_sha256": TERMINAL_AUDIT_SHA256,
            "candidate_config": str(OUTPUT_CONFIG.relative_to(ROOT)),
            "candidate_config_sha256": config_sha256,
            "verifier": str(VERIFIER.relative_to(ROOT)),
            "verifier_sha256": sha256_file(VERIFIER),
            "dataset_manifest_sha256": "1e1de075c504906a93637bd79450d30da2243797d2e1d3e33f2392d9492ddf8b",
            "fixed_eval_indices_sha256": "5ca31b59768e43de808ad5e206ed152a4a0a3515ad68d29a0b2338c4db140747",
        },
        "full_state_contract": {
            "trajectory_schema_version": "nanogpt_parameter_trajectory_v2",
            "capture_interval": CAPTURE_INTERVAL,
            "expected_snapshot_steps": PHASES,
            "required_functional_replay_steps": PHASES,
            "all_named_parameters": True,
            "all_persistent_named_buffers": True,
            "storage_dtype": "float32",
            "optimizer_probes": False,
        },
        "nonintervention_gate": {
            "scientific_config_changes": False,
            "only_change": "all-parameter/all-buffer phase snapshots and acquisition metadata",
            "curve_must_reproduce_before_interpretation": True,
        },
        "performance_gate": {
            "minimum_mfu_fraction": 0.2,
            "native_extension_required": True,
            "all_losses_finite": True,
            "include_diagnostic_io": True,
            "mode": "direct foreground polling",
            "watchdog": False,
            "callback": False,
        },
        "acceptance": {
            "accepted_validation_ce_by_step": {
                "2373": 3.5036,
                "4746": 3.3187,
                "7119": 3.2145,
                "9489": 3.1569,
            },
            "curve_absolute_tolerance_ce": 0.005,
            "functional_replay_absolute_tolerance_ce": 0.005,
            "terminal_checkpoint_next_iter": 9489,
            "threshold_changes_after_measurement": False,
        },
        "storage_gate": {
            "workspace_cap_bytes": 274877906944,
            "estimated_snapshot_payload_bytes": 3051008160,
            "atomic_checkpoint_bytes": 1197677360,
            "require_backing_filesystem_free_space": True,
        },
        "authorization": {
            "one_exact_replay_after_mfu_pass": True,
            "phase_analysis_after_acceptance": True,
            "candidate_structure_implementation": False,
            "candidate_language_model_training": False,
            "larger_rung": False,
        },
        "monitoring": {
            "callbacks": [20, 50, 100, "error or actionable stall"],
            "heartbeat_minutes": 90,
            "heartbeat_resets_after_progress": True,
            "callback_endpoint": "http://127.0.0.1:8766/send-opencode-test",
            "callback_prompt": (
                "@Codex verify identity, curve, snapshots, GPU/storage health, update the "
                "active note, and continue the preregistered phase-localization plan."
            ),
        },
    }


def main() -> None:
    source = json.loads(SOURCE_CONFIG.read_text())
    config = make_config(source)
    config_raw = json_bytes(config)
    plan = make_plan(hashlib.sha256(config_raw).hexdigest())
    OUTPUT_CONFIG.write_bytes(config_raw)
    OUTPUT_PLAN.write_bytes(json_bytes(plan))
    print(json.dumps({
        "config": str(OUTPUT_CONFIG),
        "config_sha256": sha256_file(OUTPUT_CONFIG),
        "plan": str(OUTPUT_PLAN),
        "plan_sha256": sha256_file(OUTPUT_PLAN),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
