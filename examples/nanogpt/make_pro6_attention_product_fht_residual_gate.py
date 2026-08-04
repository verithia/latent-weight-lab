#!/usr/bin/env python3
"""Register the PRO6 dense-attention replay for a product-FHT residual gate."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONFIG = ROOT / (
    "examples/nanogpt/configs/"
    "y400_mai_v3_124m_muon_0p5tpp_attention_trajectory_lr24e4.json"
)
OUTPUT_CONFIG = ROOT / (
    "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_muon_0p5tpp_attention_trajectory_replay_lr24e4.json"
)
OUTPUT_PLAN = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_attention_product_fht_residual_gate_plan.json"
)
EXPECTED_SOURCE_CONFIG_SHA256 = (
    "3c712e68108fec9a71064ed8b8a5fb836992d266f0d56b86fbe413c5ebd4a892"
)
DATASET_MANIFEST_SHA256 = (
    "1e1de075c504906a93637bd79450d30da2243797d2e1d3e33f2392d9492ddf8b"
)
FIXED_EVAL_INDICES_SHA256 = (
    "5ca31b59768e43de808ad5e206ed152a4a0a3515ad68d29a0b2338c4db140747"
)
WORKSPACE = Path("/home/pro6000-9980x/MappingNetworks")
REMOTE_REPO = Path("/mnt/ssd-data/orj/MappingNetworks/latent-weight-lab")
RUN_NAME = "pro6_mai_v3_124m_dense_attention_productfht_gate_0p5tpp"
OUTPUT_ROOT = (
    Path("/mnt/ssd-data/orj/MappingNetworks/outputs")
    / "pro6_mai_v3_attention_product_fht_gate"
    / RUN_NAME
)
SCIENTIFIC_OUT = OUTPUT_ROOT / "scientific"
CERTIFICATE = OUTPUT_ROOT / "performance_preflight.json"
PHASE_BOUNDARIES = [0, 60, 120, 180, 238]
PROBE_STEPS = PHASE_BOUNDARIES[:-1]
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
    if sha256_file(SOURCE_CONFIG) != EXPECTED_SOURCE_CONFIG_SHA256:
        raise RuntimeError("immutable dense-attention source config drifted")
    config = copy.deepcopy(source)
    config.update(
        {
            "data_dir": str(WORKSPACE / "data/finewebedu_20b"),
            "out_dir": str(SCIENTIFIC_OUT),
            "mfu_preflight_certificate": str(CERTIFICATE),
            "trajectory_snapshot_interval": 60,
            "trajectory_snapshot_targets": ["attn.c_attn", "attn.c_proj"],
            "trajectory_snapshot_layers": PROBE_LAYERS,
            "trajectory_snapshot_dtype": "float32",
            "optimizer_probe_steps": PROBE_STEPS,
            "optimizer_probe_targets": ["attn.c_attn", "attn.c_proj"],
            "optimizer_probe_layers": PROBE_LAYERS,
            "optimizer_probe_dtype": "float32",
            "diagnostic_protocol": (
                "Replay the exact 124M dense Muon 0.5-TPP control on PRO6; "
                "save attention phase-boundary weights and exact pre-step Muon "
                "directions for layers 0,3,6,9,11. Diagnostics do not alter "
                "optimization, data order, schedule, or fixed evaluation."
            ),
            "diagnostic_caveat": (
                "This replay measures local path and update alignment. One "
                "optimizer trajectory is not the global solution manifold."
            ),
            "estimated_trajectory_payload_bytes": 566231040,
            "estimated_optimizer_probe_payload_bytes": 1132462080,
            "product_fht_residual_gate_provenance": {
                "classification": "new_cross_host_replay_not_resume",
                "source_config": str(SOURCE_CONFIG.relative_to(ROOT)),
                "source_config_sha256": EXPECTED_SOURCE_CONFIG_SHA256,
                "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
                "fixed_eval_indices_sha256": FIXED_EVAL_INDICES_SHA256,
                "scientific_settings_changed": False,
                "allowed_changes": [
                    "PRO6 data/output/certificate paths",
                    "five-layer phase-boundary attention snapshots",
                    "five-layer phase-start attention Muon probes",
                    "diagnostic and replay provenance fields",
                ],
            },
            "monitoring_policy": (
                "Short 124M diagnostic: direct foreground polling for preflight "
                "and acquisition; no watchdog, callbacks, or heartbeat."
            ),
        }
    )
    allowed = {
        "data_dir",
        "out_dir",
        "mfu_preflight_certificate",
        "trajectory_snapshot_interval",
        "trajectory_snapshot_targets",
        "trajectory_snapshot_layers",
        "trajectory_snapshot_dtype",
        "optimizer_probe_steps",
        "optimizer_probe_targets",
        "optimizer_probe_layers",
        "optimizer_probe_dtype",
        "diagnostic_protocol",
        "diagnostic_caveat",
        "estimated_trajectory_payload_bytes",
        "estimated_optimizer_probe_payload_bytes",
        "product_fht_residual_gate_provenance",
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
    return config


def make_plan(config_sha256: str) -> dict[str, Any]:
    python = WORKSPACE / ".venv/bin/python"
    return {
        "schema_version": "mai_124m_attention_product_fht_residual_gate_plan_v1",
        "status": "registered_before_replay_preflight_or_oracle_implementation",
        "recorded_at": "2026-08-04",
        "scientific_question": (
            "Can a zero-preserving product of fixed global FHT mixers and "
            "learned diagonals supply an attention residual tangent aligned "
            "with dense Muon without a learned dense/LoRA-like basis?"
        ),
        "prior_evidence": {
            "best_attention_5tpp_validation_ce": 3.6151,
            "dense_5tpp_validation_ce": 3.5401,
            "remaining_gap_ce": 0.0750,
            "fixed_right_best_recovery": 0.0536304,
            "deployed_blockfht_recovery": 0.01002095,
            "global_fht_locality_control_delta_ce": 0.0053,
            "interpretation": (
                "The residual is an orientation problem, not isolated repeated-"
                "BlockFHT locality or scalar radial capacity."
            ),
        },
        "authorization": {
            "dense_replay": "one exact 238-update PRO6 replay after MFU pass",
            "zero_update_oracle": True,
            "candidate_training": (
                "at most one 124M/0.5-TPP run only after the oracle passes"
            ),
            "watchdog": False,
            "callbacks": False,
        },
        "identity": {
            "source_config": str(SOURCE_CONFIG.relative_to(ROOT)),
            "source_config_sha256": EXPECTED_SOURCE_CONFIG_SHA256,
            "candidate_config": str(OUTPUT_CONFIG.relative_to(ROOT)),
            "candidate_config_sha256": config_sha256,
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "fixed_eval_indices_sha256": FIXED_EVAL_INDICES_SHA256,
            "execution_commit_rule": (
                "use a clean pushed commit containing this immutable plan"
            ),
        },
        "replay": {
            "host": "PRO6",
            "gpu": 0,
            "phase_boundaries": PHASE_BOUNDARIES,
            "optimizer_probe_steps": PROBE_STEPS,
            "layers": PROBE_LAYERS,
            "snapshot_targets": ["attn.c_attn", "attn.c_proj"],
            "out_dir": str(SCIENTIFIC_OUT),
            "direct_foreground_polling": True,
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
            "include_diagnostic_io": True,
            "warmup_updates": 1,
            "timed_updates": 8,
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
                "8",
                "--include-diagnostic-io",
            ],
        },
        "oracle": {
            "candidate_family": (
                "zero-preserving additive residual W(D)-W(0), where W is a "
                "product of fixed seeded global FHT/sign mixers and learned "
                "per-channel diagonals"
            ),
            "factors": [6, 12],
            "targets": ["attn.c_attn", "attn.c_proj"],
            "selection_data": "only the registered dense replay probes",
            "learned_dense_basis": False,
            "parameter_updates": 0,
            "primary_metric": "dense-Muon applied-direction energy recovery",
            "normalization": (
                "compare recovery with coordinate_fraction and report "
                "recovery/coordinate_fraction enrichment"
            ),
        },
        "decision_rule": {
            "promote_at_most_one": True,
            "promote": (
                "aggregate recovery >= 0.10, enrichment >= 2.0, every target "
                "recovery >= 0.02, exact projection residual checks pass, and "
                "a later real-training preflight reaches MFU >= 0.20"
            ),
            "reject": (
                "no factor count meets every registered threshold; close this "
                "fixed-basis nonlinear residual family without a training run"
            ),
            "no_posthoc_threshold_changes": True,
        },
    }


def main() -> None:
    source = json.loads(SOURCE_CONFIG.read_text())
    config = make_config(source)
    config_bytes = json_bytes(config)
    plan = make_plan(sha256_bytes(config_bytes))
    OUTPUT_CONFIG.write_bytes(config_bytes)
    OUTPUT_PLAN.write_bytes(json_bytes(plan))
    print(OUTPUT_CONFIG)
    print(OUTPUT_PLAN)


if __name__ == "__main__":
    main()
