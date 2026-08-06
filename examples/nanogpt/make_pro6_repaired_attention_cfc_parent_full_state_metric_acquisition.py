#!/usr/bin/env python3
"""Register the full-state repair of the 124M/5TPP metric acquisition."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from examples.nanogpt.make_pro6_repaired_attention_cfc_parent_nonintervening_metric_acquisition import (
    DATASET_MANIFEST_SHA256,
    FIXED_EVAL_INDICES_SHA256,
    PHASE_BOUNDARIES,
    PROBE_LAYERS,
    PROBE_STEPS,
    REMOTE_REPO,
    ROOT,
    SOURCE_CONFIG,
    SOURCE_CONFIG_SHA256,
    SOURCE_RESULT_SHA256,
    WORKSPACE,
    json_bytes,
    make_config as make_v2_config,
    make_plan as make_v2_plan,
    sha256_bytes,
    sha256_file,
)


BASE_CONFIG_SHA256 = (
    "d9395addcac32933c3c62a7d10aad7e26caa356b4957901482373be1150498fe"
)
BASE_PLAN_SHA256 = (
    "e01cd9bbc56787846c95c9a6948eeafc278365d9398fb47d2f3817896946d4e1"
)
INVALID_CALIBRATION_AUDIT = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_mlp_cproj_5tpp_functional_metric_calibration_invalid_result.json"
)
INVALID_CALIBRATION_AUDIT_SHA256 = (
    "b1c3b584233671e8f6e6b799fd2de4cb3a704159edbd66bf3d5750957d00c613"
)
OUTPUT_CONFIG = ROOT / (
    "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_repairedattn_cfconly_"
    "fullstate_metric_probe_5tpp.json"
)
OUTPUT_PLAN = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_repaired_attention_cfc_parent_"
    "full_state_metric_acquisition_v3_plan.json"
)
RUN_NAME = (
    "pro6_mai_v3_124m_repairedattn_cfconly_"
    "fullstate_metric_probe_5tpp"
)
OUTPUT_ROOT = WORKSPACE / "outputs/pro6_mai_v3_mlp_manifold" / RUN_NAME
SCIENTIFIC_OUT = OUTPUT_ROOT / "scientific"
CERTIFICATE = OUTPUT_ROOT / "performance_preflight.json"


def make_config(source: dict[str, Any]) -> dict[str, Any]:
    base = make_v2_config(source)
    if sha256_bytes(json_bytes(base)) != BASE_CONFIG_SHA256:
        raise RuntimeError("immutable accepted v2 acquisition config drifted")
    if sha256_bytes(json_bytes(make_v2_plan(BASE_CONFIG_SHA256))) != BASE_PLAN_SHA256:
        raise RuntimeError("immutable accepted v2 acquisition plan drifted")
    if sha256_file(INVALID_CALIBRATION_AUDIT) != INVALID_CALIBRATION_AUDIT_SHA256:
        raise RuntimeError("invalid functional-replay audit drifted")

    config = copy.deepcopy(base)
    config.update(
        {
            "out_dir": str(SCIENTIFIC_OUT),
            "mfu_preflight_certificate": str(CERTIFICATE),
            "trajectory_snapshot_all_buffers": True,
            "diagnostic_acquisition_plan": str(OUTPUT_PLAN.relative_to(ROOT)),
            "diagnostic_protocol": (
                "Replay the accepted repaired-attention plus procedural-c_fc "
                "parent unchanged. At every phase boundary, snapshot every "
                "unique named parameter and every persistent unique named "
                "buffer. Retain the nonintervening v2 c_proj optimizer probe: "
                "CPU copies only before the real Muon step and CPU-derived "
                "realized direction after it."
            ),
            "diagnostic_caveat": (
                "The predecessor's training trajectory and c_proj probes were "
                "valid, but its parameter-only snapshots could not reconstruct "
                "procedural c_fc forward state. This distinct full-state repair "
                "authorizes no metric conclusion until reconstructed fixed CE "
                "matches every same-step evaluation."
            ),
            "estimated_trajectory_payload_bytes": 2100000000,
            "functional_metric_acquisition_provenance": {
                "classification": "full_state_functional_replay_repair",
                "source_config": str(SOURCE_CONFIG.relative_to(ROOT)),
                "source_config_sha256": SOURCE_CONFIG_SHA256,
                "source_result_sha256": SOURCE_RESULT_SHA256,
                "accepted_v2_config_sha256": BASE_CONFIG_SHA256,
                "accepted_v2_plan_sha256": BASE_PLAN_SHA256,
                "invalid_calibration_audit": str(
                    INVALID_CALIBRATION_AUDIT.relative_to(ROOT)
                ),
                "invalid_calibration_audit_sha256": (
                    INVALID_CALIBRATION_AUDIT_SHA256
                ),
                "scientific_settings_changed": False,
                "state_capture_change": (
                    "add all persistent unique named model buffers to each "
                    "all-parameter trajectory snapshot"
                ),
            },
        }
    )
    changed = {
        key
        for key in set(base) | set(config)
        if base.get(key) != config.get(key)
    }
    expected = {
        "out_dir",
        "mfu_preflight_certificate",
        "trajectory_snapshot_all_buffers",
        "diagnostic_acquisition_plan",
        "diagnostic_protocol",
        "diagnostic_caveat",
        "estimated_trajectory_payload_bytes",
        "functional_metric_acquisition_provenance",
    }
    if changed != expected:
        raise RuntimeError(
            f"full-state config changed unexpected fields: {changed ^ expected}"
        )
    return config


def make_plan(config_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": (
            "mai_124m_repaired_attention_cfc_parent_"
            "full_state_metric_acquisition_v3_plan_v1"
        ),
        "status": "registered_before_preflight_or_execution",
        "recorded_at": "2026-08-06",
        "scientific_question": (
            "Can a nonintervening full-state replay preserve the accepted "
            "5TPP parent trajectory and exactly reconstruct its procedural "
            "c_fc forward function at every phase?"
        ),
        "causal_basis": [
            "The accepted v2 run is parent-equivalent and its c_proj optimizer probes are internally exact.",
            "Its v1 trajectory snapshots omitted 24 persistent c_fc buffers, so reconstructed mature-phase CE was approximately 10-11 instead of 3.69-4.21.",
            "The invalid calibration is preserved only as an audit and provides no metric evidence.",
            "The minimal repair is observational: capture persistent named buffers in addition to the unchanged all-parameter and optimizer-probe payloads.",
        ],
        "identity": {
            "source_config": str(SOURCE_CONFIG.relative_to(ROOT)),
            "source_config_sha256": SOURCE_CONFIG_SHA256,
            "source_parent_result_sha256": SOURCE_RESULT_SHA256,
            "accepted_v2_config_sha256": BASE_CONFIG_SHA256,
            "accepted_v2_plan_sha256": BASE_PLAN_SHA256,
            "invalid_calibration_audit": str(
                INVALID_CALIBRATION_AUDIT.relative_to(ROOT)
            ),
            "invalid_calibration_audit_sha256": (
                INVALID_CALIBRATION_AUDIT_SHA256
            ),
            "candidate_config": str(OUTPUT_CONFIG.relative_to(ROOT)),
            "candidate_config_sha256": config_sha256,
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "fixed_eval_indices_sha256": FIXED_EVAL_INDICES_SHA256,
            "execution_commit_rule": (
                "use one clean pushed commit containing the implementation, "
                "tests, invalid audit, immutable plan, and config"
            ),
        },
        "full_state_contract": {
            "trajectory_schema_version": "nanogpt_parameter_trajectory_v2",
            "phase_boundaries": PHASE_BOUNDARIES,
            "named_parameter_count_expected": 327,
            "persistent_buffer_count_expected": 24,
            "required_buffer_patterns": [
                "transformer.h.{0..11}.mlp.c_fc.weight",
                "transformer.h.{0..11}.mlp.c_fc.optimizer_step",
            ],
            "floating_buffer_storage_dtype": "float32",
            "nonfloating_buffer_dtype_policy": "preserve exact source dtype",
            "nonpersistent_caches": "excluded",
            "tied_parameter_aliases": "not duplicated",
        },
        "nonintervention_gate": {
            "required": [
                "full-state snapshotting leaves all CPU model parameters and buffers bitwise unchanged",
                "full-state snapshotting leaves all CUDA model parameters and buffers bitwise unchanged",
                "v2 optimizer-probe CPU and CUDA trajectory-identity tests remain bitwise exact",
                "all focused tests pass on PRO6 before preflight",
            ]
        },
        "acquisition": {
            "host": "PRO6",
            "gpu": 0,
            "phase_boundaries": PHASE_BOUNDARIES,
            "optimizer_probe_steps": PROBE_STEPS,
            "optimizer_probe_layers": PROBE_LAYERS,
            "out_dir": str(SCIENTIFIC_OUT),
            "expected_duration_minutes": [80, 90],
            "command": [
                str(REMOTE_REPO / "examples/nanogpt/launch_y400_ladder_detached.sh"),
                "--foreground",
                str(REMOTE_REPO / OUTPUT_CONFIG.relative_to(ROOT)),
                "0",
                RUN_NAME,
                str(WORKSPACE),
            ],
        },
        "performance_gate": {
            "minimum_mfu_fraction": 0.2,
            "native_extension_required": True,
            "all_losses_finite": True,
            "include_diagnostic_io": True,
            "direct_foreground_polling": True,
            "watchdog": False,
            "command": [
                "env",
                "CUDA_VISIBLE_DEVICES=0",
                str(WORKSPACE / ".venv/bin/python"),
                "-u",
                "-m",
                "examples.nanogpt.mfu_preflight",
                "--config",
                str(REMOTE_REPO / OUTPUT_CONFIG.relative_to(ROOT)),
                "--output",
                str(CERTIFICATE),
                "--log-output",
                str(OUTPUT_ROOT / "performance_preflight.log"),
                "--min-fraction",
                "0.2",
                "--warmup-updates",
                "1",
                "--timed-updates",
                "8",
                "--include-diagnostic-io",
            ],
        },
        "acquisition_acceptance": {
            "required": [
                "exact-config MFU >=0.20 with diagnostic I/O charged and native BlockFHT loaded",
                "clean terminal 2373/2373 and complete exact-resume checkpoint",
                "all five full-state snapshots and all four v2 optimizer probes exist, are finite, and share one run identity",
                "exactly 327 named parameters and 24 required persistent buffers occur at every phase",
                "each reconstructed snapshot fixed validation CE differs from the run's same-step fixed validation CE by <=0.005",
                "absolute validation CE difference from the accepted parent is <=0.005 at every nonzero fixed checkpoint",
                "terminal full-state snapshot equals the checkpoint model state modulo the tied lm_head alias",
            ],
            "accepted_parent_validation_ce": [
                4.2028,
                3.8395,
                3.6888,
                3.625838041305542,
            ],
            "functional_replay_absolute_tolerance_ce": 0.005,
            "parent_equivalence_absolute_tolerance_ce": 0.005,
            "threshold_changes_after_measurement": False,
        },
        "authorization": {
            "blind_rerun": False,
            "one_full_state_acquisition_after_all_gates_pass": True,
            "zero_update_metric_calibration_after_both_acceptance_gates_pass": True,
            "candidate_structure_implementation": False,
            "candidate_language_model_training": False,
            "larger_rung": False,
        },
        "monitoring": {
            "callbacks": ["100% clean completion", "error or actionable stall"],
            "milestones": False,
            "heartbeats": False,
            "scientific_run_policy": "one idempotent terminal-only watchdog",
            "callback_action": (
                "verify terminal artifacts, exact buffer inventory, parent "
                "equivalence, and same-step functional replay before any next step"
            ),
        },
    }


def main() -> None:
    source = json.loads(SOURCE_CONFIG.read_text())
    config = make_config(source)
    config_bytes = json_bytes(config)
    plan = make_plan(sha256_bytes(config_bytes))
    OUTPUT_CONFIG.write_bytes(config_bytes)
    OUTPUT_PLAN.write_bytes(json_bytes(plan))
    print(
        json.dumps(
            {
                "config": str(OUTPUT_CONFIG),
                "config_sha256": sha256_file(OUTPUT_CONFIG),
                "plan": str(OUTPUT_PLAN),
                "plan_sha256": sha256_file(OUTPUT_PLAN),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
