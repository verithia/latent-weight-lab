#!/usr/bin/env python3
"""Register the corrected non-intervening 124M/5TPP metric acquisition."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONFIG = ROOT / (
    "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_repairedfullattn_plus_cfconly_decay1_5tpp_lr24e4.json"
)
OUTPUT_CONFIG = ROOT / (
    "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_repairedattn_cfconly_"
    "nonintervening_metric_probe_5tpp.json"
)
OUTPUT_PLAN = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_repaired_attention_cfc_parent_"
    "functional_metric_acquisition_v2_plan.json"
)

SOURCE_CONFIG_SHA256 = (
    "baee2a5148f8e66bcd955680b39b7b6ccc7b3a7be00440a535a0dabf60a6c857"
)
SOURCE_RESULT_SHA256 = (
    "309556d0290c4a51fc8535c42b41f4e2982dcb7e3c4a048ceed5dfd3bd734b5f"
)
REJECTED_ACQUISITION_RESULT_SHA256 = (
    "ac76087b7782635aad09c3ec77c39849e281b14148818b2a77c866f25de086ad"
)
DATASET_MANIFEST_SHA256 = (
    "1e1de075c504906a93637bd79450d30da2243797d2e1d3e33f2392d9492ddf8b"
)
FIXED_EVAL_INDICES_SHA256 = (
    "5ca31b59768e43de808ad5e206ed152a4a0a3515ad68d29a0b2338c4db140747"
)
PROBE_IMPLEMENTATION_COMMIT = "9a95d4e"
CUDA_IDENTITY_TEST_COMMIT = "7a85cc1"

WORKSPACE = Path("/home/pro6000-9980x/MappingNetworks")
REMOTE_REPO = Path(
    "/mnt/ssd-data/orj/MappingNetworks/latent-weight-lab-symmetric-cproj-5tpp"
)
RUN_NAME = (
    "pro6_mai_v3_124m_repairedattn_cfconly_"
    "nonintervening_metric_probe_5tpp"
)
OUTPUT_ROOT = WORKSPACE / "outputs/pro6_mai_v3_mlp_manifold" / RUN_NAME
SCIENTIFIC_OUT = OUTPUT_ROOT / "scientific"
CERTIFICATE = OUTPUT_ROOT / "performance_preflight.json"
PHASE_BOUNDARIES = [0, 594, 1188, 1782, 2373]
PROBE_STEPS = PHASE_BOUNDARIES[:-1]
PROBE_LAYERS = list(range(8))


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
        + b"\n"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def make_config(source: dict[str, Any]) -> dict[str, Any]:
    if sha256_file(SOURCE_CONFIG) != SOURCE_CONFIG_SHA256:
        raise RuntimeError("immutable accepted c_fc-only config drifted")
    config = copy.deepcopy(source)
    config.update(
        {
            "out_dir": str(SCIENTIFIC_OUT),
            "mfu_preflight_certificate": str(CERTIFICATE),
            "trajectory_snapshot_interval": 594,
            "trajectory_snapshot_targets": [],
            "trajectory_snapshot_layers": None,
            "trajectory_snapshot_all_parameters": True,
            "trajectory_snapshot_dtype": "float32",
            "optimizer_probe_steps": PROBE_STEPS,
            "optimizer_probe_targets": ["mlp.c_proj"],
            "optimizer_probe_layers": PROBE_LAYERS,
            "optimizer_probe_dtype": "float32",
            "diagnostic_acquisition_plan": str(OUTPUT_PLAN.relative_to(ROOT)),
            "diagnostic_protocol": (
                "Replay the accepted repaired-attention plus procedural-c_fc "
                "parent unchanged. Snapshot all parameters at phase boundaries. "
                "At probe steps, copy only raw pre-step c_proj weight, clipped "
                "gradient, momentum, and hyperparameters to CPU; execute the one "
                "real Muon step; then copy post-step weight and derive the realized "
                "direction using CPU arithmetic. No alternate pre-step GPU matrix "
                "kernel is permitted."
            ),
            "diagnostic_caveat": (
                "This corrected acquisition authorizes only parent-equivalence "
                "verification and, after acceptance, zero-update functional-metric "
                "calibration. It authorizes no structural candidate or LM run."
            ),
            "estimated_trajectory_payload_bytes": 1500000000,
            "estimated_optimizer_probe_payload_bytes": 2113929216,
            "functional_metric_acquisition_provenance": {
                "classification": "nonintervening_observer_effect_repair_replay",
                "source_config": str(SOURCE_CONFIG.relative_to(ROOT)),
                "source_config_sha256": SOURCE_CONFIG_SHA256,
                "source_result_sha256": SOURCE_RESULT_SHA256,
                "rejected_acquisition_result_sha256": (
                    REJECTED_ACQUISITION_RESULT_SHA256
                ),
                "probe_implementation_commit": PROBE_IMPLEMENTATION_COMMIT,
                "cuda_identity_test_commit": CUDA_IDENTITY_TEST_COMMIT,
                "scientific_settings_changed": False,
                "observer_effect_removed": (
                    "no pre-step Newton-Schulz or other matrix multiplication is "
                    "performed by the recorder on the training accelerator"
                ),
            },
            "monitoring_policy": (
                "Approximately 80-90 minutes: one idempotent terminal-only "
                "watchdog; callback only on clean completion, actionable error, "
                "or stall. No milestone or heartbeat callbacks."
            ),
        }
    )
    allowed = {
        "out_dir",
        "mfu_preflight_certificate",
        "trajectory_snapshot_interval",
        "trajectory_snapshot_targets",
        "trajectory_snapshot_layers",
        "trajectory_snapshot_all_parameters",
        "trajectory_snapshot_dtype",
        "optimizer_probe_steps",
        "optimizer_probe_targets",
        "optimizer_probe_layers",
        "optimizer_probe_dtype",
        "diagnostic_acquisition_plan",
        "diagnostic_protocol",
        "diagnostic_caveat",
        "estimated_trajectory_payload_bytes",
        "estimated_optimizer_probe_payload_bytes",
        "functional_metric_acquisition_provenance",
        "monitoring_policy",
    }
    changed = {
        key
        for key in set(source) | set(config)
        if source.get(key) != config.get(key)
    }
    if changed - allowed:
        raise RuntimeError(
            f"scientific config changed unexpectedly: {changed - allowed}"
        )
    if config["max_iters"] != 2373 or config["eval_interval"] != 594:
        raise RuntimeError("accepted 5TPP schedule drifted")
    return config


def make_plan(config_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": (
            "mai_124m_repaired_attention_cfc_parent_"
            "functional_metric_acquisition_v2_plan_v1"
        ),
        "status": "registered_before_preflight_or_execution",
        "recorded_at": "2026-08-06",
        "scientific_question": (
            "Can a genuinely non-intervening replay acquire parent-equivalent "
            "same-gauge 5TPP c_proj phase states and realized Muon actions?"
        ),
        "causal_basis": [
            "The first acquisition produced complete artifacts but failed the frozen step-594 parent-equivalence tolerance by 0.0045 CE beyond its limit.",
            "Its recorder ran eight additional dense Newton-Schulz decompositions on the live GPU before the real optimizer step, violating non-intervention.",
            "The replacement recorder performs only immutable CPU copies before the real step and derives the realized direction from post-step weights afterward.",
            "CPU and CUDA tests require exact equality of probed and unprobed Muon weights and momentum after the step.",
        ],
        "identity": {
            "source_config": str(SOURCE_CONFIG.relative_to(ROOT)),
            "source_config_sha256": SOURCE_CONFIG_SHA256,
            "source_result_sha256": SOURCE_RESULT_SHA256,
            "rejected_acquisition_result_sha256": (
                REJECTED_ACQUISITION_RESULT_SHA256
            ),
            "candidate_config": str(OUTPUT_CONFIG.relative_to(ROOT)),
            "candidate_config_sha256": config_sha256,
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "fixed_eval_indices_sha256": FIXED_EVAL_INDICES_SHA256,
            "probe_implementation_commit": PROBE_IMPLEMENTATION_COMMIT,
            "cuda_identity_test_commit": CUDA_IDENTITY_TEST_COMMIT,
            "execution_commit_rule": (
                "use one clean pushed commit containing this immutable plan and config"
            ),
        },
        "nonintervention_gate": {
            "required": [
                "prepare phase performs no Newton-Schulz, GEMM, or mapper evaluation on the training accelerator",
                "probed and unprobed CPU Muon trajectories are bitwise identical",
                "probed and unprobed CUDA Muon trajectories are bitwise identical",
                "realized applied direction equals (post_step_weight - pre_step_weight) / learning_rate",
                "all focused tests pass on PRO6 before preflight",
            ],
            "schema_version": "nanogpt_optimizer_probe_v2",
            "capture_protocol": (
                "pre_step_cpu_state_post_step_realized_direction_v2"
            ),
        },
        "acquisition": {
            "host": "PRO6",
            "gpu": 0,
            "phase_boundaries": PHASE_BOUNDARIES,
            "optimizer_probe_steps": PROBE_STEPS,
            "optimizer_probe_layers": PROBE_LAYERS,
            "snapshot_scope": "all named model parameters in float32",
            "optimizer_probe_scope": (
                "pre/post weight, clipped gradient, old momentum, CPU-derived "
                "combined momentum and polar residual, and exact realized "
                "applied direction per LR for dense mlp.c_proj"
            ),
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
                "all registered snapshots and v2 probes exist and are finite",
                "one common run identity across snapshots, probes, and checkpoint",
                "dataset and fixed-evaluation digests match the accepted parent",
                "absolute validation CE difference from the accepted parent is <=0.005 at every fixed checkpoint",
            ],
            "accepted_parent_validation_ce": [
                4.2028,
                3.8395,
                3.6888,
                3.625838041305542,
            ],
            "threshold_changes_after_measurement": False,
        },
        "authorization": {
            "blind_rerun": False,
            "one_corrected_acquisition_after_all_gates_pass": True,
            "zero_update_metric_calibration_after_acquisition_pass": True,
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
                "verify parent equivalence and every artifact; only on full "
                "acceptance run the preregistered zero-update calibration"
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
