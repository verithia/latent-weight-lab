#!/usr/bin/env python3
"""Register a same-run c_proj parameter/optimizer-state trajectory replay."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONFIG = ROOT / (
    "examples/nanogpt/configs/pro6_mai_v3_124m_repairedattn_cfc_"
    "latecproj_fullstate_trajectory_5tpp.json"
)
SOURCE_CONFIG_SHA256 = "1e8e6aba162ec94a0a8b469b4e5620190f096035585835c8c434126d7e53b20a"
SOURCE_VERIFICATION = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/124m_repaired_attention_"
    "cfc_late_cproj_full_state_trajectory_verification_result.json"
)
SOURCE_VERIFICATION_SHA256 = "d34a06734733587db069c2de0424e93e16ce952bd71f5c854de91d803ca98489"
INVALID_CROSS_RUN_RESULT = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/124m_repaired_attention_"
    "cfc_late_cproj_optimizer_state_cross_run_invalid_result.json"
)
INVALID_CROSS_RUN_RESULT_SHA256 = "379c4a114a7246261932fcc5182f150880776bf947d45a87a42a0142c897a868"
OUTPUT_CONFIG = ROOT / (
    "examples/nanogpt/configs/pro6_mai_v3_124m_repairedattn_cfc_"
    "latecproj_pairedstate_trajectory_5tpp.json"
)
OUTPUT_PLAN = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/124m_repaired_attention_"
    "cfc_late_cproj_paired_state_trajectory_acquisition_plan.json"
)
VERIFIER = ROOT / "examples/nanogpt/verify_late_cproj_paired_state_trajectory.py"
WORKSPACE = Path("/mnt/ssd-data/orj/MappingNetworks")
REMOTE_REPO = WORKSPACE / "latent-weight-lab-symmetric-cproj-5tpp"
RUN_NAME = (
    "pro6_mai_v3_124m_repairedattn_cfc_latecproj_"
    "pairedstate_trajectory_5tpp"
)
OUTPUT_ROOT = WORKSPACE / "outputs/pro6_mai_v3_mlp_manifold" / RUN_NAME
SCIENTIFIC_OUT = OUTPUT_ROOT / "scientific"
CERTIFICATE = OUTPUT_ROOT / "performance_preflight.json"
PROBE_STEPS = [0, 98, 296, 593, 890, 1187, 1484, 1781, 2078, 2372]
POST_STEP_SNAPSHOTS = [99, 297, 594, 891, 1188, 1485, 1782, 2079, 2373]
SNAPSHOT_STEPS = [0, *range(99, 2373, 99), 2373]
PROBE_LAYERS = [8, 9, 10, 11]
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
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def make_config(source: dict[str, Any]) -> dict[str, Any]:
    if sha256_file(SOURCE_CONFIG) != SOURCE_CONFIG_SHA256:
        raise RuntimeError("accepted full-state config drifted")
    if sha256_file(SOURCE_VERIFICATION) != SOURCE_VERIFICATION_SHA256:
        raise RuntimeError("accepted full-state verification drifted")
    if sha256_file(INVALID_CROSS_RUN_RESULT) != INVALID_CROSS_RUN_RESULT_SHA256:
        raise RuntimeError("cross-run rejection audit drifted")
    config = copy.deepcopy(source)
    config.pop("registered_plan_sha256", None)
    config.update(
        {
            "out_dir": str(SCIENTIFIC_OUT),
            "mfu_preflight_certificate": str(CERTIFICATE),
            "hpo_stage": "late_cproj_paired_state_trajectory_acquisition",
            "ladder_slot": "124m_5tpp_cproj_layers8_11_paired_state",
            "ladder_role": "same_run_parameter_optimizer_state_acquisition",
            "candidate_scope": (
                "Replay the accepted co-adapted model while capturing its "
                "targeted c_proj path and structured-Muon state in one run."
            ),
            "registered_plan": str(OUTPUT_PLAN.relative_to(ROOT)),
            "implementation_source_hashes": {
                path: sha256_file(ROOT / path) for path in SOURCE_PATHS
            },
            "paired_state_registration_parent_commit": git_head(),
            "trajectory_snapshot_interval": 99,
            "trajectory_snapshot_targets": ["mlp.c_proj"],
            "trajectory_snapshot_dtype": "float32",
            "trajectory_snapshot_layers": PROBE_LAYERS,
            "trajectory_snapshot_all_parameters": False,
            "trajectory_snapshot_all_buffers": False,
            "optimizer_probe_steps": PROBE_STEPS,
            "optimizer_probe_targets": ["mlp.c_proj"],
            "optimizer_probe_layers": PROBE_LAYERS,
            "optimizer_probe_dtype": "float32",
            "estimated_trajectory_payload_bytes": 4800000000,
            "diagnostic_acquisition_plan": str(OUTPUT_PLAN.relative_to(ROOT)),
            "diagnostic_protocol": (
                "Capture four targeted trainable c_proj buffers every 99 "
                "updates and ten phase-aligned structured-Muon probes in the "
                "same run. Both update-0 writes are charged in preflight."
            ),
            "diagnostic_caveat": (
                "Observational only. Cross-run tensor paths are invalid; all "
                "causal targets must be paired by this run identity."
            ),
            "trajectory_acquisition_provenance": {
                "classification": "same_run_cproj_parameter_optimizer_state_replay",
                "source_config": str(SOURCE_CONFIG.relative_to(ROOT)),
                "source_config_sha256": SOURCE_CONFIG_SHA256,
                "source_verification": str(SOURCE_VERIFICATION.relative_to(ROOT)),
                "source_verification_sha256": SOURCE_VERIFICATION_SHA256,
                "invalid_cross_run_result": str(
                    INVALID_CROSS_RUN_RESULT.relative_to(ROOT)
                ),
                "invalid_cross_run_result_sha256": INVALID_CROSS_RUN_RESULT_SHA256,
                "scientific_settings_changed": False,
                "state_capture_change": (
                    "pair targeted c_proj trajectory snapshots and optimizer "
                    "state probes inside one independent replay"
                ),
            },
            "mfu_measurement_protocol": (
                "foreground exact-config preflight with one warmup and eight "
                "timed updates; update-0 targeted snapshot plus optimizer "
                "probe I/O inside the measured loop; native BlockFHT and "
                "MFU >=0.20 required"
            ),
            "monitoring_policy": (
                "approximately 80-100 minutes: one idempotent terminal-only "
                "watchdog; callback only on completion, error, or actionable stall"
            ),
            "selection_endpoint": (
                "same-run bitwise parameter/optimizer pairing and fixed-curve reproduction"
            ),
            "operator_override": (
                "2026-08-07: independent CUDA replays share initialization and "
                "loss but not tensor paths; acquire causal state and target in one run."
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
        "schema_version": (
            "mai_124m_repaired_attention_cfc_late_cproj_"
            "paired_state_acquisition_plan_v1"
        ),
        "status": "registered_before_preflight_or_execution",
        "recorded_at": "2026-08-07",
        "scientific_question": (
            "Within one exact training path, does structured-Muon state "
            "transport the function-critical c_proj residual?"
        ),
        "causal_basis": [
            "The first optimizer-state replay has valid probes, identical initialization, matching CE, and an exact terminal checkpoint.",
            "Its independently replayed parameter path differs from the accepted trajectory by 0.069 to 0.618 relative Frobenius norm, invalidating cross-run pairing.",
            "The only valid repair is to capture parameter targets and optimizer state under one run identity.",
            "Targeted trainable-buffer snapshots retain every required c_proj tensor without duplicating full-model state.",
        ],
        "identity": {
            "source_config": str(SOURCE_CONFIG.relative_to(ROOT)),
            "source_config_sha256": SOURCE_CONFIG_SHA256,
            "source_verification": str(SOURCE_VERIFICATION.relative_to(ROOT)),
            "source_verification_sha256": SOURCE_VERIFICATION_SHA256,
            "invalid_cross_run_result": str(INVALID_CROSS_RUN_RESULT.relative_to(ROOT)),
            "invalid_cross_run_result_sha256": INVALID_CROSS_RUN_RESULT_SHA256,
            "candidate_config": str(OUTPUT_CONFIG.relative_to(ROOT)),
            "candidate_config_sha256": config_sha256,
            "verifier": str(VERIFIER.relative_to(ROOT)),
            "verifier_sha256": sha256_file(VERIFIER),
            "dataset_manifest_sha256": "1e1de075c504906a93637bd79450d30da2243797d2e1d3e33f2392d9492ddf8b",
            "fixed_eval_indices_sha256": "5ca31b59768e43de808ad5e206ed152a4a0a3515ad68d29a0b2338c4db140747",
            "execution_commit_rule": (
                "one clean pushed commit containing targeted-buffer capture, "
                "maker, verifier, tests, config, plan, and invalid audit"
            ),
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
        "nonintervention_gate": {
            "scientific_config_changes": False,
            "same_run_identity_required": True,
            "snapshot_probe_comparison": "bitwise float32 equality",
            "terminal_snapshot_checkpoint_comparison": "bitwise float32 equality",
            "fixed_curve_absolute_tolerance_ce": 0.005,
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
            "accepted_validation_ce_by_step": {
                "594": 4.2239,
                "1188": 3.851,
                "1782": 3.6998,
                "2373": 3.636904239654541,
            },
            "curve_absolute_tolerance_ce": 0.005,
            "required": [
                "exact-config MFU >=0.20 with both update-0 writes charged and native BlockFHT loaded",
                "clean terminal 2373/2373 and exact-resume checkpoint",
                "25 targeted c_proj snapshots and ten optimizer probes under one run identity",
                "all ten phase pairings and the terminal checkpoint compare bitwise",
                "four fixed validation checkpoints reproduce within 0.005 CE",
            ],
            "threshold_changes_after_measurement": False,
        },
        "preregistered_zero_update_analysis": {
            "same_run_only": True,
            "components": [
                "requested", "feedback", "corrected", "realized", "unrepresented"
            ],
            "targets": ["next phase chord", "terminal exact-minus-scheduled residual"],
            "metrics": ["raw weight", "terminal post-GELU functional", "output-additive projection"],
            "heldout_probe_steps": [1781, 2078, 2372],
            "authorization_threshold": 0.80,
        },
        "authorization": {
            "one_paired_state_acquisition_after_exact_config_mfu": True,
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
                "Verify all same-run snapshots/probes/checkpoint bitwise, seal "
                "the result, run the preregistered zero-update transport "
                "analysis, and continue causally. Do not merely acknowledge."
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
