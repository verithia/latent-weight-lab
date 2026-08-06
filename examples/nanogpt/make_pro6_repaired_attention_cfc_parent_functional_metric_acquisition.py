#!/usr/bin/env python3
"""Register the 124M/5TPP same-gauge c_proj metric acquisition replay."""

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
    "pro6_mai_v3_124m_repairedattn_cfconly_functional_metric_probe_5tpp.json"
)
OUTPUT_PLAN = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_repaired_attention_cfc_parent_functional_metric_acquisition_plan.json"
)

SOURCE_CONFIG_SHA256 = (
    "baee2a5148f8e66bcd955680b39b7b6ccc7b3a7be00440a535a0dabf60a6c857"
)
SOURCE_RESULT_SHA256 = (
    "309556d0290c4a51fc8535c42b41f4e2982dcb7e3c4a048ceed5dfd3bd734b5f"
)
LATE_LWT_RESULT_SHA256 = (
    "f3a436af76d7bc85bee8301fae8277f878f4b5fad5c7bf24aacefb1c2b288e4f"
)
DATASET_MANIFEST_SHA256 = (
    "1e1de075c504906a93637bd79450d30da2243797d2e1d3e33f2392d9492ddf8b"
)
FIXED_EVAL_INDICES_SHA256 = (
    "5ca31b59768e43de808ad5e206ed152a4a0a3515ad68d29a0b2338c4db140747"
)

WORKSPACE = Path("/home/pro6000-9980x/MappingNetworks")
REMOTE_REPO = Path(
    "/mnt/ssd-data/orj/MappingNetworks/latent-weight-lab-symmetric-cproj-5tpp"
)
RUN_NAME = "pro6_mai_v3_124m_repairedattn_cfconly_functional_metric_probe_5tpp"
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
                "parent unchanged for 2373 updates. Save all named parameters "
                "at steps 0,594,1188,1782,2373 and exact pre-step dense c_proj "
                "Muon state/directions at steps 0,594,1188,1782 for layers "
                "0-7. These are observational side effects only."
            ),
            "diagnostic_caveat": (
                "This acquisition authorizes only same-gauge functional-metric "
                "calibration and residual decomposition. It does not authorize "
                "a candidate chart, MFU gate for a candidate, or candidate LM run."
            ),
            "estimated_trajectory_payload_bytes": 1500000000,
            "estimated_optimizer_probe_payload_bytes": 1812162208,
            "functional_metric_acquisition_provenance": {
                "classification": "deterministic_same_host_replay_not_resume",
                "source_config": str(SOURCE_CONFIG.relative_to(ROOT)),
                "source_config_sha256": SOURCE_CONFIG_SHA256,
                "source_result_sha256": SOURCE_RESULT_SHA256,
                "late_lwt_result_sha256": LATE_LWT_RESULT_SHA256,
                "scientific_settings_changed": False,
                "allowed_changes": [
                    "output and MFU-certificate paths",
                    "sparse all-parameter phase snapshots",
                    "phase-start dense c_proj Muon probes for layers 0-7",
                    "diagnostic provenance, storage estimate, and monitoring prose",
                ],
            },
            "monitoring_policy": (
                "Approximately 80-90 minute diagnostic acquisition: one "
                "idempotent terminal-only watchdog; callback only on clean "
                "completion, actionable error, or stall. No milestone or heartbeat callbacks."
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
    unexpected = changed - allowed
    if unexpected:
        raise RuntimeError(f"scientific config changed unexpectedly: {unexpected}")
    if config["max_iters"] != 2373 or config["eval_interval"] != 594:
        raise RuntimeError("accepted 5TPP schedule drifted")
    return config


def make_plan(config_sha256: str) -> dict[str, Any]:
    python = WORKSPACE / ".venv/bin/python"
    return {
        "schema_version": (
            "mai_124m_repaired_attention_cfc_parent_"
            "functional_metric_acquisition_plan_v1"
        ),
        "status": "registered_before_preflight_or_execution",
        "recorded_at": "2026-08-06",
        "scientific_question": (
            "What same-gauge task-weighted dense-Muon c_proj action remains "
            "uncaptured in early/middle layers 0-7 across the 5TPP trajectory?"
        ),
        "causal_basis": [
            "The late-band LWT run recovered about 67.8% of the all-layer c_proj penalty by keeping layers 0-7 dense.",
            "The accepted c_fc-only parent is the exact same-attention, same-c_fc gauge needed to study those dense c_proj layers.",
            "Only terminal checkpoints exist; prior 0.5TPP or independently trained trajectories cannot be spliced into this 5TPP state.",
            "The replay changes no model, optimizer, schedule, seed, data, or evaluation setting; it adds observational snapshots and probes only.",
        ],
        "paper_link": {
            "mechanism": "LWT and layer-specific manifolds",
            "boundary": (
                "This acquisition tests the layer-local premise. It does not "
                "assume that paper modulation, Mapping Loss, or a random "
                "orthogonal chart spans the measured residual."
            ),
        },
        "identity": {
            "source_config": str(SOURCE_CONFIG.relative_to(ROOT)),
            "source_config_sha256": SOURCE_CONFIG_SHA256,
            "source_result_sha256": SOURCE_RESULT_SHA256,
            "late_lwt_result_sha256": LATE_LWT_RESULT_SHA256,
            "candidate_config": str(OUTPUT_CONFIG.relative_to(ROOT)),
            "candidate_config_sha256": config_sha256,
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "fixed_eval_indices_sha256": FIXED_EVAL_INDICES_SHA256,
            "execution_commit_rule": (
                "use one clean pushed commit containing this immutable plan and config"
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
                "weight, clipped gradient, old and combined momentum, polar "
                "update, and exact applied direction per LR for dense mlp.c_proj"
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
                str(python),
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
                "all-parameter snapshots exactly at 0,594,1188,1782,2373",
                "dense c_proj optimizer probes exactly at 0,594,1188,1782 for layers 0-7",
                "one common run identity across snapshots, probes, and checkpoint",
                "dataset and fixed-evaluation digests match the accepted parent",
                "all losses and floating model, optimizer, snapshot, and probe tensors are finite",
                "absolute validation CE difference from the accepted parent is <=0.005 at every fixed checkpoint",
            ],
            "threshold_changes_after_measurement": False,
        },
        "post_acquisition_analysis_only": {
            "fit_window": {
                "split": "validation",
                "seed": 20260806,
                "batch_size": 2,
                "block_size": 256,
                "batches": 4,
                "participates_in_metric_fitting": True,
            },
            "confirmation_window": {
                "split": "validation",
                "seed": 20260807,
                "batch_size": 2,
                "block_size": 256,
                "batches": 4,
                "participates_in_metric_fitting": False,
            },
            "measurements": [
                "exact post-GELU output action ||H deltaW|| and task-gradient inner product per layer and phase",
                "exact finite-CE effect of bounded dense-Muon, current hidden64+24, and equal-coordinate Haar controls",
                "fit-to-confirmation sign and ordering calibration for identity, activation-covariance, and downstream-gradient block metrics",
                "uncaptured residual decomposition by layer, phase, hidden channel, output side, and effective rank",
                "short teacher-forced multi-step rotation of the residual before any candidate chart is proposed",
            ],
            "metric_gate": (
                "Freeze thresholds from duplicate-window noise before inspecting "
                "any new chart. A metric must predict held-out finite-CE sign and "
                "ordering across phases; Frobenius recovery alone cannot pass."
            ),
        },
        "monitoring": {
            "scientific_run_policy": "one idempotent terminal-only watchdog",
            "callbacks": ["100% clean completion", "error or actionable stall"],
            "milestones": False,
            "heartbeats": False,
            "callback_action": (
                "verify terminal identity, fixed losses, snapshots, probes, "
                "checkpoint, hashes, GPU and storage; seal acquisition; then "
                "run only the preregistered zero-update metric calibration"
            ),
        },
        "authorization": {
            "generate_config_and_run_focused_tests": True,
            "one_exact_config_mfu_preflight": True,
            "one_diagnostic_acquisition_after_mfu_pass": True,
            "zero_update_metric_calibration_after_acquisition_pass": True,
            "candidate_structure_implementation": False,
            "candidate_language_model_training": False,
            "automatic_rerun": False,
            "larger_rung": False,
        },
    }


def main() -> None:
    source = json.loads(SOURCE_CONFIG.read_text())
    config = make_config(source)
    config_bytes = json_bytes(config)
    plan = make_plan(sha256_bytes(config_bytes))
    OUTPUT_CONFIG.write_bytes(config_bytes)
    OUTPUT_PLAN.write_bytes(json_bytes(plan))
    print(OUTPUT_CONFIG.relative_to(ROOT), sha256_file(OUTPUT_CONFIG))
    print(OUTPUT_PLAN.relative_to(ROOT), sha256_file(OUTPUT_PLAN))


if __name__ == "__main__":
    main()
