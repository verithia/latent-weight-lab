#!/usr/bin/env python3
"""Register one confirmatory paired-state replay with a replicate-calibrated gate."""

from __future__ import annotations

import copy
import datetime as dt
import json
import math
import statistics
import subprocess
from pathlib import Path
from typing import Any

from examples.nanogpt.make_pro6_repaired_attention_cfc_late_cproj_paired_state_trajectory import (
    POST_STEP_SNAPSHOTS,
    PROBE_LAYERS,
    PROBE_STEPS,
    REMOTE_REPO,
    SNAPSHOT_STEPS,
    WORKSPACE,
    json_bytes,
    sha256_bytes,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONFIG = ROOT / (
    "examples/nanogpt/configs/pro6_mai_v3_124m_repairedattn_cfc_"
    "latecproj_pairedstate_trajectory_5tpp.json"
)
SOURCE_CONFIG_SHA256 = "8a96082e45ec6990d2aa579f53636542be761147cf74b74d95ca70f52aed86e8"
SOURCE_RESULT = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/124m_repaired_attention_"
    "cfc_late_cproj_lwt_5tpp_result.json"
)
SOURCE_RESULT_SHA256 = "f3a436af76d7bc85bee8301fae8277f878f4b5fad5c7bf24aacefb1c2b288e4f"
FULL_STATE_RESULT = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/124m_repaired_attention_"
    "cfc_late_cproj_full_state_trajectory_verification_result.json"
)
FULL_STATE_RESULT_SHA256 = "d34a06734733587db069c2de0424e93e16ce952bd71f5c854de91d803ca98489"
OPTIMIZER_STATE_RESULT = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/124m_repaired_attention_"
    "cfc_late_cproj_optimizer_state_cross_run_invalid_result.json"
)
OPTIMIZER_STATE_RESULT_SHA256 = (
    "379c4a114a7246261932fcc5182f150880776bf947d45a87a42a0142c897a868"
)
PAIRED_REJECTION_RESULT = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/124m_repaired_attention_"
    "cfc_late_cproj_paired_state_curve_rejection_result.json"
)
PAIRED_REJECTION_RESULT_SHA256 = (
    "76aea2883429f4678be4933cdc3affc10f198655d7b9226c981daa05ed910b72"
)
OUTPUT_CONFIG = ROOT / (
    "examples/nanogpt/configs/pro6_mai_v3_124m_repairedattn_cfc_"
    "latecproj_pairedstate_replicate2_5tpp.json"
)
OUTPUT_PLAN = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/124m_repaired_attention_"
    "cfc_late_cproj_paired_state_replicate2_acquisition_plan.json"
)
VERIFIER = ROOT / "examples/nanogpt/verify_late_cproj_paired_state_replicate.py"
RUN_NAME = (
    "pro6_mai_v3_124m_repairedattn_cfc_latecproj_"
    "pairedstate_replicate2_5tpp"
)
OUTPUT_ROOT = WORKSPACE / "outputs/pro6_mai_v3_mlp_manifold" / RUN_NAME
SCIENTIFIC_OUT = OUTPUT_ROOT / "scientific"
CERTIFICATE = OUTPUT_ROOT / "performance_preflight.json"
T_CRITICAL_DF3_975 = 3.182446305
HISTORICAL_CURVES = {
    594: [4.2239, 4.2256, 4.2225, 4.2227],
    1188: [3.8510, 3.8554, 3.8530, 3.8596],
    1782: [3.6998, 3.7046, 3.7001, 3.7081],
    2373: [3.6369, 3.6412, 3.6371, 3.6449],
}
NONSCIENTIFIC_CONFIG_KEYS = {
    "candidate_scope",
    "diagnostic_acquisition_plan",
    "diagnostic_caveat",
    "diagnostic_protocol",
    "hpo_stage",
    "ladder_role",
    "ladder_slot",
    "mfu_measurement_protocol",
    "mfu_preflight_certificate",
    "monitoring_policy",
    "operator_override",
    "out_dir",
    "paired_confirmatory_registration_parent_commit",
    "registered_plan",
    "registered_plan_sha256",
    "selection_endpoint",
    "trajectory_acquisition_provenance",
}


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def prediction_intervals() -> dict[str, dict[str, Any]]:
    result = {}
    for step, values in HISTORICAL_CURVES.items():
        mean = statistics.mean(values)
        sample_sd = statistics.stdev(values)
        half_width = T_CRITICAL_DF3_975 * sample_sd * math.sqrt(1 + 1 / len(values))
        result[str(step)] = {
            "historical_values": values,
            "n": len(values),
            "mean": mean,
            "sample_sd": sample_sd,
            "t_critical_df3_two_sided_95": T_CRITICAL_DF3_975,
            "prediction_half_width": half_width,
            "lower": mean - half_width,
            "upper": mean + half_width,
        }
    return result


def changed_config_keys(source: dict[str, Any], candidate: dict[str, Any]) -> set[str]:
    return {
        key
        for key in set(source) | set(candidate)
        if source.get(key) != candidate.get(key)
    }


def make_config(source: dict[str, Any]) -> dict[str, Any]:
    pinned = {
        SOURCE_CONFIG: SOURCE_CONFIG_SHA256,
        SOURCE_RESULT: SOURCE_RESULT_SHA256,
        FULL_STATE_RESULT: FULL_STATE_RESULT_SHA256,
        OPTIMIZER_STATE_RESULT: OPTIMIZER_STATE_RESULT_SHA256,
        PAIRED_REJECTION_RESULT: PAIRED_REJECTION_RESULT_SHA256,
    }
    for path, expected in pinned.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"calibration input drifted: {path}")
    config = copy.deepcopy(source)
    config.pop("registered_plan_sha256", None)
    config.update(
        {
            "out_dir": str(SCIENTIFIC_OUT),
            "mfu_preflight_certificate": str(CERTIFICATE),
            "hpo_stage": "late_cproj_paired_state_confirmatory_replicate",
            "ladder_slot": "124m_5tpp_cproj_layers8_11_paired_state_replicate2",
            "ladder_role": "prospective_replicate_equivalence_confirmation",
            "candidate_scope": (
                "One fresh paired-state replay confirms a prospectively "
                "calibrated run-to-run equivalence gate; it is not a search "
                "for a favorable replay."
            ),
            "registered_plan": str(OUTPUT_PLAN.relative_to(ROOT)),
            "diagnostic_acquisition_plan": str(OUTPUT_PLAN.relative_to(ROOT)),
            "paired_confirmatory_registration_parent_commit": git_head(),
            "trajectory_acquisition_provenance": {
                "classification": "paired_state_confirmatory_replicate",
                "source_config": str(SOURCE_CONFIG.relative_to(ROOT)),
                "source_config_sha256": SOURCE_CONFIG_SHA256,
                "paired_rejection_result": str(
                    PAIRED_REJECTION_RESULT.relative_to(ROOT)
                ),
                "paired_rejection_result_sha256": PAIRED_REJECTION_RESULT_SHA256,
                "scientific_settings_changed": False,
                "replicate_rule": (
                    "one new run only; accept against prediction intervals "
                    "calibrated from all four pre-existing curves"
                ),
            },
            "diagnostic_protocol": (
                "Repeat the exact paired snapshot/probe acquisition once and "
                "score its four fixed validation checkpoints against the "
                "preregistered historical 95% prediction intervals."
            ),
            "diagnostic_caveat": (
                "The rejected paired replay remains rejected under its old "
                "0.005 gate and is not retroactively accepted."
            ),
            "selection_endpoint": (
                "bitwise same-run pairing plus prospective replicate prediction intervals"
            ),
            "mfu_measurement_protocol": (
                "fresh foreground exact-config preflight with update-0 targeted "
                "snapshot and optimizer-probe I/O charged; native BlockFHT and "
                "MFU >=0.20 required"
            ),
            "monitoring_policy": (
                "80-100 minutes: one idempotent terminal-only watchdog; callback "
                "only on completion, error, or actionable stall"
            ),
            "operator_override": (
                "2026-08-07: replace the single-path 0.005 replay gate only for "
                "one future confirmatory run, using all prior replicates to "
                "freeze a prediction interval before launch."
            ),
        }
    )
    unexpected = changed_config_keys(source, config) - NONSCIENTIFIC_CONFIG_KEYS
    if unexpected:
        raise RuntimeError(f"scientific config changed: {sorted(unexpected)}")
    return config


def make_plan(config_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": (
            "mai_124m_repaired_attention_cfc_late_cproj_"
            "paired_state_confirmatory_replicate_plan_v1"
        ),
        "status": "registered_before_preflight_or_execution",
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scientific_question": (
            "Can one fresh exact paired-state replay satisfy a prospective "
            "run-to-run equivalence interval and provide an admissible causal trajectory?"
        ),
        "causal_basis": [
            "The rejected paired replay has complete bitwise same-run parameter/optimizer pairing.",
            "It failed only the frozen single-reference 0.005 CE gate at three later checkpoints.",
            "Four pre-existing identical-science runs span up to 0.0086 CE, so the old gate is narrower than observed CUDA run-to-run variation.",
            "A new run, not the observed rejected run, must satisfy the interval calibrated from all four prior curves.",
            "Only one confirmatory replicate is authorized; repeated attempts until a pass are forbidden.",
        ],
        "identity": {
            "source_config": str(SOURCE_CONFIG.relative_to(ROOT)),
            "source_config_sha256": SOURCE_CONFIG_SHA256,
            "source_result": str(SOURCE_RESULT.relative_to(ROOT)),
            "source_result_sha256": SOURCE_RESULT_SHA256,
            "full_state_result": str(FULL_STATE_RESULT.relative_to(ROOT)),
            "full_state_result_sha256": FULL_STATE_RESULT_SHA256,
            "optimizer_state_result": str(OPTIMIZER_STATE_RESULT.relative_to(ROOT)),
            "optimizer_state_result_sha256": OPTIMIZER_STATE_RESULT_SHA256,
            "paired_rejection_result": str(PAIRED_REJECTION_RESULT.relative_to(ROOT)),
            "paired_rejection_result_sha256": PAIRED_REJECTION_RESULT_SHA256,
            "candidate_config": str(OUTPUT_CONFIG.relative_to(ROOT)),
            "candidate_config_sha256": config_sha256,
            "verifier": str(VERIFIER.relative_to(ROOT)),
            "verifier_sha256": sha256_file(VERIFIER),
            "dataset_manifest_sha256": "1e1de075c504906a93637bd79450d30da2243797d2e1d3e33f2392d9492ddf8b",
            "fixed_eval_indices_sha256": "5ca31b59768e43de808ad5e206ed152a4a0a3515ad68d29a0b2338c4db140747",
            "execution_commit_rule": (
                "one clean pushed commit containing maker, verifier, tests, "
                "immutable config, plan, and rejection audit"
            ),
        },
        "calibration": {
            "method": (
                "two-sided 95% prediction interval for one future normal "
                "observation: mean +/- t(df=3,0.975)*s*sqrt(1+1/n)"
            ),
            "historical_run_order": [
                "accepted_late_cproj_lwt",
                "full_state_replay",
                "optimizer_state_replay",
                "paired_state_replay_rejected_by_old_gate",
            ],
            "intervals_by_step": prediction_intervals(),
            "calibration_frozen_before_confirmatory_run": True,
            "retroactive_acceptance_of_prior_rejection": False,
        },
        "targeted_snapshot_contract": {
            "trajectory_schema_version": "nanogpt_parameter_trajectory_v1",
            "steps": SNAPSHOT_STEPS,
            "layers": PROBE_LAYERS,
            "target": "mlp.c_proj",
            "storage_dtype": "float32",
            "terminal_step": 2373,
            "trainable_structured_buffers_exported_as_parameters": True,
            "all_parameters": False,
            "all_buffers": False,
        },
        "optimizer_state_contract": {
            "trajectory_schema_version": "nanogpt_optimizer_probe_v2",
            "pre_step_probe_steps": PROBE_STEPS,
            "post_step_snapshot_steps": POST_STEP_SNAPSHOTS,
            "layers": PROBE_LAYERS,
            "target": "mlp.c_proj",
            "storage_dtype": "float32",
            "optimizer_kind": "MuonMatchedGivens",
            "required_tensor_fields": [
                "weight_before_step",
                "gradient_after_clip",
                "momentum_buffer_before_step",
                "compression_residual_before_step",
                "weight_after_step",
                "combined_momentum_update",
                "applied_direction_per_lr",
                "momentum_buffer_after_step",
                "compression_residual_after_step",
            ],
            "error_feedback": True,
            "error_feedback_decay": 0.5,
        },
        "performance_gate": {
            "minimum_mfu_fraction": 0.2,
            "native_extension_required": True,
            "all_losses_finite": True,
            "include_diagnostic_io": True,
            "required_snapshot_step_in_preflight": 0,
            "required_probe_step_in_preflight": 0,
            "mode": "direct foreground polling",
            "watchdog": False,
            "callback": False,
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
        "acquisition": {
            "host": "PRO6",
            "gpu": 0,
            "expected_duration_minutes": [80, 100],
            "out_dir": str(SCIENTIFIC_OUT),
            "command": [
                str(REMOTE_REPO / "examples/nanogpt/launch_y400_ladder_detached.sh"),
                str(REMOTE_REPO / OUTPUT_CONFIG.relative_to(ROOT)),
                "0",
                RUN_NAME,
                str(WORKSPACE),
            ],
        },
        "acceptance": {
            "replicate_prediction_intervals_by_step": prediction_intervals(),
            "all_four_checkpoints_must_pass": True,
            "required": [
                "fresh exact-config MFU >=0.20 with both update-0 writes charged and native BlockFHT loaded",
                "clean terminal 2373/2373 and exact-resume checkpoint",
                "25 targeted c_proj snapshots and ten optimizer probes under one run identity",
                "all ten phase pairings and terminal checkpoint compare bitwise",
                "all four fixed validation checkpoints lie inside the preregistered prediction intervals",
            ],
            "threshold_changes_after_measurement": False,
        },
        "preregistered_zero_update_analysis": {
            "same_run_only": True,
            "components": [
                "requested", "feedback", "corrected", "realized", "unrepresented"
            ],
            "targets": ["next phase chord", "terminal exact-minus-scheduled residual"],
            "metrics": [
                "raw weight", "terminal post-GELU functional", "output-additive projection"
            ],
            "heldout_probe_steps": [1781, 2078, 2372],
            "authorization_threshold": 0.80,
        },
        "authorization": {
            "one_confirmatory_replicate_after_fresh_exact_config_mfu": True,
            "additional_replicates_after_this_one": False,
            "zero_update_state_transport_analysis_after_acceptance": True,
            "candidate_structure_implementation": False,
            "candidate_language_model_training": False,
            "larger_rung": False,
        },
        "monitoring": {
            "policy": "one idempotent terminal-only watchdog",
            "callbacks": ["100% clean completion", "error or actionable stall"],
            "milestones": False,
            "heartbeats": False,
            "callback_endpoint": "http://127.0.0.1:8766/send-opencode-test",
            "callback_prompt": (
                "Verify the confirmatory replicate against its prospective "
                "prediction intervals and bitwise pairing, seal the result, "
                "run the zero-update analysis only if accepted, and continue."
            ),
        },
    }


def main() -> None:
    source = json.loads(SOURCE_CONFIG.read_text())
    config = make_config(source)
    config_raw = json_bytes(config)
    plan = make_plan(sha256_bytes(config_raw))
    OUTPUT_CONFIG.write_bytes(config_raw)
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
