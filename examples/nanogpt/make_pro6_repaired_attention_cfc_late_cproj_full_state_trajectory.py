#!/usr/bin/env python3
"""Register a full-state replay of the accepted late c_proj LWT path."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONFIG = ROOT / (
    "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_repairedfullattn_plus_cfc_"
    "latecproj_lwt_5tpp_lr24e4.json"
)
SOURCE_CONFIG_SHA256 = (
    "684f24b4aa6482aef98ee776d9e71f101ff9a56d9f72f6a783f9997ea91d717c"
)
SOURCE_RESULT = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_repaired_attention_cfc_late_cproj_lwt_5tpp_result.json"
)
SOURCE_RESULT_SHA256 = (
    "f3a436af76d7bc85bee8301fae8277f878f4b5fad5c7bf24aacefb1c2b288e4f"
)
PHASE_CONTROL_RESULT = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_mlp_cproj_late_band_phase_validity_result.json"
)
PHASE_CONTROL_RESULT_SHA256 = (
    "9e1754d5270a37ed6849c4298f711f8202d5c0ae1c6a3dbb24208382d30e4b76"
)
OUTPUT_CONFIG = ROOT / (
    "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_repairedattn_cfc_latecproj_"
    "fullstate_trajectory_5tpp.json"
)
OUTPUT_PLAN = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_repaired_attention_cfc_late_cproj_"
    "full_state_trajectory_acquisition_plan.json"
)
VERIFIER = ROOT / "examples/nanogpt/verify_late_cproj_full_state_trajectory.py"
WORKSPACE = Path("/mnt/ssd-data/orj/MappingNetworks")
REMOTE_REPO = WORKSPACE / "latent-weight-lab-symmetric-cproj-5tpp"
RUN_NAME = (
    "pro6_mai_v3_124m_repairedattn_cfc_latecproj_"
    "fullstate_trajectory_5tpp"
)
OUTPUT_ROOT = WORKSPACE / "outputs/pro6_mai_v3_mlp_manifold" / RUN_NAME
SCIENTIFIC_OUT = OUTPUT_ROOT / "scientific"
CERTIFICATE = OUTPUT_ROOT / "performance_preflight.json"
CAPTURE_INTERVAL = 99
PHASE_BOUNDARIES = [0, 594, 1188, 1782, 2373]
EXPECTED_SNAPSHOT_STEPS = [
    *range(0, 2277 + CAPTURE_INTERVAL, CAPTURE_INTERVAL),
    2373,
]
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
        raise RuntimeError("accepted late-band source config drifted")
    if sha256_file(SOURCE_RESULT) != SOURCE_RESULT_SHA256:
        raise RuntimeError("accepted late-band result drifted")
    if sha256_file(PHASE_CONTROL_RESULT) != PHASE_CONTROL_RESULT_SHA256:
        raise RuntimeError("phase-proxy control result drifted")
    config = copy.deepcopy(source)
    config.pop("registered_plan_sha256", None)
    config.update(
        {
            "out_dir": str(SCIENTIFIC_OUT),
            "mfu_preflight_certificate": str(CERTIFICATE),
            "hpo_stage": "late_cproj_lwt_full_state_trajectory_acquisition",
            "ladder_slot": "124m_5tpp_cfc_all_cproj_layers8_11_fullstate",
            "ladder_role": "coadapted_late_cproj_manifold_acquisition",
            "candidate_scope": (
                "Observational replay of the accepted repaired-attention, "
                "procedural-c_fc, and procedural c_proj layers 8-11 model. "
                "Every scientific optimizer, model, data, and schedule field "
                "is inherited unchanged; only full-state snapshot capture is "
                "enabled."
            ),
            "registered_plan": str(OUTPUT_PLAN.relative_to(ROOT)),
            "implementation_source_hashes": {
                path: sha256_file(ROOT / path) for path in SOURCE_PATHS
            },
            "state_capture_registration_parent_commit": git_head(),
            "trajectory_snapshot_interval": CAPTURE_INTERVAL,
            "trajectory_snapshot_targets": [],
            "trajectory_snapshot_dtype": "float32",
            "trajectory_snapshot_layers": None,
            "trajectory_snapshot_all_parameters": True,
            "trajectory_snapshot_all_buffers": True,
            "estimated_trajectory_payload_bytes": 10500000000,
            "diagnostic_acquisition_plan": str(OUTPUT_PLAN.relative_to(ROOT)),
            "diagnostic_protocol": (
                "Capture 25 full-state snapshots at 99-step cadence plus the "
                "terminal boundary. Required fixed-eval boundaries are 0, "
                "594, 1188, 1782, and 2373. No optimizer probe or extra loss "
                "is enabled."
            ),
            "diagnostic_caveat": (
                "This rerun is accepted only if its own fixed curve reproduces "
                "the sealed late-band run within 0.005 CE and every required "
                "snapshot functionally replays its same-step loss within "
                "0.005 CE."
            ),
            "trajectory_acquisition_provenance": {
                "classification": "coadapted_late_band_full_state_replay",
                "source_config": str(SOURCE_CONFIG.relative_to(ROOT)),
                "source_config_sha256": SOURCE_CONFIG_SHA256,
                "source_result": str(SOURCE_RESULT.relative_to(ROOT)),
                "source_result_sha256": SOURCE_RESULT_SHA256,
                "phase_control_result": str(
                    PHASE_CONTROL_RESULT.relative_to(ROOT)
                ),
                "phase_control_result_sha256": PHASE_CONTROL_RESULT_SHA256,
                "scientific_settings_changed": False,
                "state_capture_change": (
                    "enable all-parameter and all-persistent-buffer snapshots"
                ),
            },
            "mfu_measurement_protocol": (
                "foreground exact-config preflight with one warmup and eight "
                "timed real updates; charge diagnostic snapshot I/O; require "
                "native BlockFHT and MFU >=0.20"
            ),
            "monitoring_policy": (
                "approximately 85-100 minutes: one idempotent terminal-only "
                "watchdog; callback only on clean completion, actionable "
                "error, or stall; no milestones or heartbeat"
            ),
            "selection_endpoint": (
                "observational acquisition only: reproduce all four sealed "
                "validation checkpoints and exact functional replay gates"
            ),
            "operator_override": (
                "2026-08-07: dense-parent phase replay is invalid for the "
                "co-adapted late c_proj path. Re-run the accepted configuration "
                "once with nonintervening full-state trajectory capture."
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
            "full_state_trajectory_acquisition_plan_v1"
        ),
        "status": "registered_before_preflight_or_execution",
        "recorded_at": "2026-08-07",
        "scientific_question": (
            "What manifold and phase geometry does the accepted co-adapted "
            "late c_proj LWT model follow in its own gauge?"
        ),
        "causal_basis": [
            "The terminal layers-4-11 capacity mask is replicated, minimal, and within 0.00461 CE of its same-gauge parent.",
            "Dense-parent phase replay fails even for the independently accepted late layers 8-11, so it is not a valid proxy for a jointly trained procedural trajectory.",
            "The accepted late-band scientific implementation is unchanged; the only source delta since its execution is nonintervening full-state snapshot and post-step probe capture in train.py.",
            "The next valid object is the co-adapted model's own path, not another chart or mask guess.",
        ],
        "identity": {
            "source_config": str(SOURCE_CONFIG.relative_to(ROOT)),
            "source_config_sha256": SOURCE_CONFIG_SHA256,
            "source_result": str(SOURCE_RESULT.relative_to(ROOT)),
            "source_result_sha256": SOURCE_RESULT_SHA256,
            "phase_control_result": str(
                PHASE_CONTROL_RESULT.relative_to(ROOT)
            ),
            "phase_control_result_sha256": PHASE_CONTROL_RESULT_SHA256,
            "candidate_config": str(OUTPUT_CONFIG.relative_to(ROOT)),
            "candidate_config_sha256": config_sha256,
            "verifier": str(VERIFIER.relative_to(ROOT)),
            "verifier_sha256": sha256_file(VERIFIER),
            "dataset_manifest_sha256": (
                "1e1de075c504906a93637bd79450d30da2243797d2e1d3e33f2392d9492ddf8b"
            ),
            "fixed_eval_indices_sha256": (
                "5ca31b59768e43de808ad5e206ed152a4a0a3515ad68d29a0b2338c4db140747"
            ),
            "execution_commit_rule": (
                "use one clean pushed commit containing the maker, verifier, "
                "tests, immutable config, and immutable plan"
            ),
        },
        "full_state_contract": {
            "trajectory_schema_version": "nanogpt_parameter_trajectory_v2",
            "capture_interval": CAPTURE_INTERVAL,
            "expected_snapshot_steps": EXPECTED_SNAPSHOT_STEPS,
            "required_functional_replay_steps": PHASE_BOUNDARIES,
            "all_named_parameters": True,
            "all_persistent_named_buffers": True,
            "storage_dtype": "float32",
            "inventory_policy": (
                "the first snapshot freezes the exact parameter and buffer "
                "name sets; all 25 snapshots must match it"
            ),
            "optimizer_probes": False,
        },
        "nonintervention_gate": {
            "accepted_execution_commit": (
                "7b6cc4986581c37fca05d712f1c7b067f26a44f7"
            ),
            "source_diff": (
                "model.py, muon.py, muon_matched_givens.py, and block_fht.py "
                "are byte-identical; train.py differs only by all-buffer "
                "snapshot forwarding and nonintervening post-step probe write"
            ),
            "scientific_config_changes": False,
        },
        "performance_gate": {
            "minimum_mfu_fraction": 0.2,
            "native_extension_required": True,
            "all_losses_finite": True,
            "include_diagnostic_io": True,
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
            "expected_duration_minutes": [85, 100],
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
            "functional_replay_absolute_tolerance_ce": 0.005,
            "required": [
                "exact-config MFU >=0.20 with diagnostic I/O charged and native BlockFHT loaded",
                "clean terminal 2373/2373 and complete exact-resume checkpoint",
                "exactly the 25 registered full-state snapshots exist and share one run identity and one exact inventory",
                "all snapshot tensors are finite",
                "the four nonzero fixed validation checkpoints reproduce the accepted run within 0.005 CE",
                "the five required snapshots replay their own same-step validation CE within 0.005 CE",
                "the terminal snapshot parameter and buffer values equal the terminal exact-resume checkpoint",
            ],
            "threshold_changes_after_measurement": False,
        },
        "authorization": {
            "one_full_state_acquisition_after_exact_config_mfu": True,
            "zero_update_coadapted_manifold_analysis_after_acceptance": True,
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
                "Verify terminal status, all 25 snapshots, fixed-curve "
                "reproduction, exact functional replay, checkpoint equality, "
                "and artifact hashes; seal the acquisition and continue with "
                "the preregistered co-adapted manifold analysis."
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
