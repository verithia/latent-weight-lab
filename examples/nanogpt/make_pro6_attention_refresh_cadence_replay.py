#!/usr/bin/env python3
"""Register a cadence-15 dense replay for attention refresh analysis."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONFIG = ROOT / (
    "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_muon_0p5tpp_attention_trajectory_replay_lr24e4.json"
)
OUTPUT_CONFIG = ROOT / (
    "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_muon_0p5tpp_attention_trajectory_cadence15_lr24e4.json"
)
OUTPUT_PLAN = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_attention_refresh_cadence_gate_plan.json"
)
EXPECTED_SOURCE_SHA256 = (
    "f5ffdc737b6358b0f9c5b85d01f2f78b45182b3295eabd8d70ee65bbeb726c6d"
)
DATASET_MANIFEST_SHA256 = (
    "1e1de075c504906a93637bd79450d30da2243797d2e1d3e33f2392d9492ddf8b"
)
FIXED_EVAL_INDICES_SHA256 = (
    "5ca31b59768e43de808ad5e206ed152a4a0a3515ad68d29a0b2338c4db140747"
)
WORKSPACE = Path("/home/pro6000-9980x/MappingNetworks")
REMOTE_REPO = Path("/mnt/ssd-data/orj/MappingNetworks/latent-weight-lab")
RUN_NAME = "pro6_mai_v3_124m_dense_attention_refresh_cadence15_0p5tpp"
OUTPUT_ROOT = (
    Path("/mnt/ssd-data/orj/MappingNetworks/outputs")
    / "pro6_mai_v3_attention_refresh_cadence"
    / RUN_NAME
)
SCIENTIFIC_OUT = OUTPUT_ROOT / "scientific"
CERTIFICATE = OUTPUT_ROOT / "performance_preflight.json"
SNAPSHOT_INTERVAL = 15
PHASE_STARTS = [0, 60, 120, 180]
PROBE_LAYERS = [0, 3, 6, 9, 11]


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
    if sha256_file(SOURCE_CONFIG) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("immutable dense replay config drifted")
    config = copy.deepcopy(source)
    config.pop("product_fht_residual_gate_provenance", None)
    config.update(
        {
            "out_dir": str(SCIENTIFIC_OUT),
            "mfu_preflight_certificate": str(CERTIFICATE),
            "trajectory_snapshot_interval": SNAPSHOT_INTERVAL,
            "optimizer_probe_steps": PHASE_STARTS,
            "estimated_trajectory_payload_bytes": 1925185536,
            "estimated_optimizer_probe_payload_bytes": 1132462080,
            "candidate_scope": (
                "dense_baseline_attention_refresh_staleness_15_30_60_only"
            ),
            "diagnostic_protocol": (
                "Replay the exact 124M dense Muon 0.5-TPP control on PRO6; "
                "save five-layer attention weights every 15 updates and exact "
                "Muon probes at 0/60/120/180 for causal 15/30/60-update "
                "connectivity-staleness measurement."
            ),
            "attention_refresh_cadence_provenance": {
                "classification": "new_exact_replay_not_resume",
                "source_config": str(SOURCE_CONFIG.relative_to(ROOT)),
                "source_config_sha256": EXPECTED_SOURCE_SHA256,
                "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
                "fixed_eval_indices_sha256": FIXED_EVAL_INDICES_SHA256,
                "scientific_settings_changed": False,
                "only_diagnostic_change": (
                    "trajectory snapshot interval 60 -> 15; optimizer probe "
                    "steps remain 0/60/120/180"
                ),
            },
            "monitoring_policy": (
                "Short 124M replay: direct foreground polling for preflight "
                "and acquisition; no watchdog, callbacks, or heartbeat."
            ),
        }
    )
    scientific_keys = {
        "batch_size",
        "block_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "max_iters",
        "method",
        "model_seed",
        "muon_adamw_lr_scale",
        "muon_momentum",
        "muon_ns_steps",
        "n_embd",
        "n_head",
        "n_layer",
        "optimizer",
        "train_data_seed",
        "warmup_iters",
        "weight_decay",
    }
    if any(config[key] != source[key] for key in scientific_keys):
        raise RuntimeError("scientific training settings changed")
    return config


def make_plan(config_sha256: str) -> dict[str, Any]:
    python = WORKSPACE / ".venv/bin/python"
    return {
        "schema_version": "mai_124m_attention_refresh_cadence_gate_plan_v1",
        "status": "registered_before_preflight_and_replay",
        "recorded_at": "2026-08-04",
        "scientific_question": (
            "At what 15/30/60-update horizon does causally selected sparse "
            "attention connectivity cease to retain at least 2x random "
            "future-chord recovery?"
        ),
        "authorization": {
            "dense_replay": "one exact 238-update PRO6 replay after MFU pass",
            "zero_update_oracle": True,
            "candidate_training": (
                "none until one refresh horizon passes and a compact "
                "implementation separately clears exact-config MFU >= 0.20"
            ),
            "watchdog": False,
            "callbacks": False,
        },
        "identity": {
            "source_config": str(SOURCE_CONFIG.relative_to(ROOT)),
            "source_config_sha256": EXPECTED_SOURCE_SHA256,
            "candidate_config": str(OUTPUT_CONFIG.relative_to(ROOT)),
            "candidate_config_sha256": config_sha256,
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "fixed_eval_indices_sha256": FIXED_EVAL_INDICES_SHA256,
        },
        "replay": {
            "host": "PRO6",
            "gpu": 0,
            "snapshot_interval": SNAPSHOT_INTERVAL,
            "probe_steps": PHASE_STARTS,
            "layers": PROBE_LAYERS,
            "targets": ["attn.c_attn", "attn.c_proj"],
            "out_dir": str(SCIENTIFIC_OUT),
            "direct_foreground_polling": True,
            "command": [
                str(
                    REMOTE_REPO
                    / "examples/nanogpt/launch_y400_ladder_detached.sh"
                ),
                "--foreground",
                str(REMOTE_REPO / OUTPUT_CONFIG.relative_to(ROOT)),
                "0",
                RUN_NAME,
                str(WORKSPACE),
            ],
        },
        "performance_gate": {
            "minimum_mfu_fraction": 0.2,
            "include_diagnostic_io": True,
            "warmup_updates": 1,
            "timed_updates": 16,
            "reason": (
                "the timed window crosses update 15 and therefore includes "
                "the new periodic snapshot cost"
            ),
            "direct_foreground_polling": True,
            "command": [
                "env",
                "CUDA_VISIBLE_DEVICES=0",
                f"CUDA_HOME={WORKSPACE / '.cuda-12.8'}",
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
                "16",
                "--include-diagnostic-io",
            ],
        },
        "oracle": {
            "stage_count": 64,
            "neighbors": 128,
            "matching_seed": 161803,
            "random_seed": 271828,
            "phase_seed_stride": 8192,
            "cg_iterations": 160,
            "cg_tolerance": 1e-5,
            "ridge": 0.0,
            "phase_starts": PHASE_STARTS,
            "horizons": [15, 30, 60],
            "selection_policy": (
                "select integer Givens pairs only from the exact current "
                "dense-Muon direction; future weights are scoring-only"
            ),
            "random_control": "equal-coordinate unique random matchings",
            "learned_dense_basis": False,
            "learned_additive_adapter": False,
            "parameter_updates": 0,
        },
        "decision_rule": {
            "choose": "the longest registered horizon passing every threshold",
            "thresholds": {
                "current_dense_recovery_minimum": 0.20,
                "current_dense_enrichment_minimum": 3.0,
                "future_chord_recovery_minimum": 0.05,
                "future_chord_over_random_minimum": 2.0,
                "per_target_chord_recovery_minimum": 0.025,
                "maximum_projection_error": 0.0001,
                "maximum_normal_residual": 0.0001,
            },
            "no_posthoc_threshold_changes": True,
        },
    }


def main() -> None:
    source = json.loads(SOURCE_CONFIG.read_text())
    config = make_config(source)
    config_bytes = json_bytes(config)
    OUTPUT_CONFIG.write_bytes(config_bytes)
    OUTPUT_PLAN.write_bytes(json_bytes(make_plan(sha256_bytes(config_bytes))))
    print(OUTPUT_CONFIG)
    print(OUTPUT_PLAN)


if __name__ == "__main__":
    main()
