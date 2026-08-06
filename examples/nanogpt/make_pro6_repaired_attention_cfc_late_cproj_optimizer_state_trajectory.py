#!/usr/bin/env python3
"""Register the nonintervening optimizer-state replay of accepted late c_proj."""

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
    "pro6_mai_v3_124m_repairedattn_cfc_latecproj_"
    "fullstate_trajectory_5tpp.json"
)
SOURCE_CONFIG_SHA256 = (
    "1e8e6aba162ec94a0a8b469b4e5620190f096035585835c8c434126d7e53b20a"
)
SOURCE_VERIFICATION = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_repaired_attention_cfc_late_cproj_"
    "full_state_trajectory_verification_result.json"
)
SOURCE_VERIFICATION_SHA256 = (
    "d34a06734733587db069c2de0424e93e16ce952bd71f5c854de91d803ca98489"
)
OUTPUT_CONFIG = ROOT / (
    "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_repairedattn_cfc_latecproj_"
    "optimizerstate_trajectory_5tpp.json"
)
OUTPUT_PLAN = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_repaired_attention_cfc_late_cproj_"
    "optimizer_state_trajectory_acquisition_plan.json"
)
VERIFIER = ROOT / "examples/nanogpt/verify_late_cproj_optimizer_state_trajectory.py"
WORKSPACE = Path("/mnt/ssd-data/orj/MappingNetworks")
REMOTE_REPO = WORKSPACE / "latent-weight-lab-symmetric-cproj-5tpp"
RUN_NAME = (
    "pro6_mai_v3_124m_repairedattn_cfc_latecproj_"
    "optimizerstate_trajectory_5tpp"
)
OUTPUT_ROOT = WORKSPACE / "outputs/pro6_mai_v3_mlp_manifold" / RUN_NAME
SCIENTIFIC_OUT = OUTPUT_ROOT / "scientific"
CERTIFICATE = OUTPUT_ROOT / "performance_preflight.json"
REFERENCE_SNAPSHOT_DIR = Path(
    "/mnt/ssd-data/orj/MappingNetworks/outputs/pro6_mai_v3_mlp_manifold/"
    "pro6_mai_v3_124m_repairedattn_cfc_latecproj_"
    "fullstate_trajectory_5tpp/scientific/parameter_trajectory"
)
# Step zero forces the exact-config MFU gate to charge one complete probe.
# The remaining pre-step captures end at phase-aligned old snapshot states.
PROBE_STEPS = [0, 98, 296, 593, 890, 1187, 1484, 1781, 2078, 2372]
REFERENCE_POST_STEPS = [99, 297, 594, 891, 1188, 1485, 1782, 2079, 2373]
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
    config = copy.deepcopy(source)
    config.pop("registered_plan_sha256", None)
    config.update(
        {
            "out_dir": str(SCIENTIFIC_OUT),
            "mfu_preflight_certificate": str(CERTIFICATE),
            "hpo_stage": "late_cproj_optimizer_state_trajectory_acquisition",
            "ladder_slot": "124m_5tpp_cproj_layers8_11_optimizer_state",
            "ladder_role": "nonintervening_structured_muon_state_acquisition",
            "candidate_scope": (
                "Exact replay of the accepted co-adapted model with only "
                "structured Muon gradient, momentum, compression-residual, "
                "and realized-step probes enabled for c_proj layers 8-11."
            ),
            "registered_plan": str(OUTPUT_PLAN.relative_to(ROOT)),
            "implementation_source_hashes": {
                path: sha256_file(ROOT / path) for path in SOURCE_PATHS
            },
            "optimizer_state_registration_parent_commit": git_head(),
            "trajectory_snapshot_interval": 0,
            "trajectory_snapshot_targets": [],
            "trajectory_snapshot_dtype": "float32",
            "trajectory_snapshot_layers": None,
            "trajectory_snapshot_all_parameters": False,
            "trajectory_snapshot_all_buffers": False,
            "optimizer_probe_steps": PROBE_STEPS,
            "optimizer_probe_targets": ["mlp.c_proj"],
            "optimizer_probe_layers": PROBE_LAYERS,
            "optimizer_probe_dtype": "float32",
            "estimated_trajectory_payload_bytes": 3800000000,
            "diagnostic_acquisition_plan": str(OUTPUT_PLAN.relative_to(ROOT)),
            "diagnostic_protocol": (
                "Capture ten pre/post structured-Muon probes. Step zero "
                "charges exact probe I/O in preflight; steps 98..2372 map "
                "to the nine registered post-step full-state snapshots."
            ),
            "diagnostic_caveat": (
                "Observational only. Acceptance requires deterministic "
                "post-step equality to the accepted full-state trajectory; "
                "the data cannot by itself authorize a candidate mapper."
            ),
            "trajectory_acquisition_provenance": {
                "classification": "coadapted_late_band_optimizer_state_replay",
                "source_config": str(SOURCE_CONFIG.relative_to(ROOT)),
                "source_config_sha256": SOURCE_CONFIG_SHA256,
                "source_verification": str(SOURCE_VERIFICATION.relative_to(ROOT)),
                "source_verification_sha256": SOURCE_VERIFICATION_SHA256,
                "scientific_settings_changed": False,
                "state_capture_change": (
                    "replace full-state snapshots with phase-aligned "
                    "structured-Muon optimizer probes"
                ),
            },
            "mfu_measurement_protocol": (
                "foreground exact-config preflight with one warmup and eight "
                "timed updates; update-0 probe serialization is inside the "
                "measured loop; require native BlockFHT and MFU >=0.20"
            ),
            "monitoring_policy": (
                "approximately 80-100 minutes: one idempotent terminal-only "
                "watchdog; callback only on clean completion, actionable "
                "error, or stall; no milestones or heartbeat"
            ),
            "selection_endpoint": (
                "observational acquisition only: exact path reproduction, "
                "complete state inventory, and post-step snapshot equality"
            ),
            "operator_override": (
                "2026-08-07: static modulation closes only 53.5% of the "
                "scheduled residual CE gap; acquire the omitted adaptive "
                "optimizer state before designing another structure."
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
            "optimizer_state_acquisition_plan_v1"
        ),
        "status": "registered_before_preflight_or_execution",
        "recorded_at": "2026-08-07",
        "scientific_question": (
            "Does the accepted c_proj path's function-critical residual live "
            "in the persistent Muon compression state and its task-conditioned "
            "transport into the structured chart?"
        ),
        "causal_basis": [
            "The co-adapted endpoint path is nearly one-dimensional but its static polynomial endpoint costs about 0.044 CE.",
            "Literal paper modulation is null; output-additive modulation recovers about 53.5% but leaves +0.02035 CE.",
            "The accepted path already uses fresh task-selected Givens and error feedback decay 0.5, but prior full-state snapshots omit optimizer momentum and compression residual.",
            "Therefore the only justified next intervention is a nonintervening acquisition of the omitted adaptive state.",
        ],
        "identity": {
            "source_config": str(SOURCE_CONFIG.relative_to(ROOT)),
            "source_config_sha256": SOURCE_CONFIG_SHA256,
            "source_verification": str(SOURCE_VERIFICATION.relative_to(ROOT)),
            "source_verification_sha256": SOURCE_VERIFICATION_SHA256,
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
            "reference_snapshot_dir": str(REFERENCE_SNAPSHOT_DIR),
            "execution_commit_rule": (
                "one clean pushed commit containing instrumentation, maker, "
                "verifier, tests, config, and plan"
            ),
        },
        "optimizer_state_contract": {
            "trajectory_schema_version": "nanogpt_optimizer_probe_v2",
            "pre_step_probe_steps": PROBE_STEPS,
            "post_step_reference_steps": REFERENCE_POST_STEPS,
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
            "scientific_full_state_snapshots_duplicated": False,
        },
        "nonintervention_gate": {
            "scientific_config_changes": False,
            "post_step_reference_comparison": "bitwise float32 equality",
            "fixed_curve_absolute_tolerance_ce": 0.005,
            "terminal_step": 2373,
            "terminal_validation_ce": 3.6411821842193604,
        },
        "performance_gate": {
            "minimum_mfu_fraction": 0.2,
            "native_extension_required": True,
            "all_losses_finite": True,
            "include_diagnostic_io": True,
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
                "exact-config MFU >=0.20 with update-0 optimizer-probe I/O charged and native BlockFHT loaded",
                "clean terminal 2373/2373 and complete exact-resume checkpoint",
                "exactly ten registered probes with four c_proj tensors and complete finite state inventory",
                "all nine phase-aligned post-step weights equal accepted same-step full-state tensors bitwise",
                "the four fixed validation checkpoints reproduce the accepted curve within 0.005 CE",
            ],
            "threshold_changes_after_measurement": False,
        },
        "preregistered_zero_update_analysis": {
            "requested_update": "lr*(-polar(combined_momentum)*polar_scale - weight_decay*weight_before)",
            "corrected_target": "requested_update + 0.5*compression_residual_before",
            "realized_update": "weight_after - weight_before",
            "unrepresented_residual": "corrected_target - realized_update",
            "tests": [
                "raw and post-GELU functional energy/cosine of requested, feedback, corrected, realized, and unrepresented directions",
                "output-additive projection of each component under the fixed terminal activation metric",
                "whether compression state predicts the next phase's function-critical exact-minus-scheduled residual",
                "whether transport is stable across discovery and held-out phases rather than fitted at the endpoint",
            ],
            "decision_rule": (
                "Only a causal state component with held-out functional "
                "recovery >=0.80 may authorize a compressed state-conditioned "
                "mapper; otherwise reject this mechanism and revisit the map."
            ),
        },
        "authorization": {
            "one_optimizer_state_acquisition_after_exact_config_mfu": True,
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
                "Verify terminal state, all ten optimizer probes, fixed-curve "
                "reproduction, phase-aligned snapshot equality, and hashes; "
                "seal the acquisition and execute the preregistered zero-update "
                "state-transport analysis. Do not merely acknowledge."
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
