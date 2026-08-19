#!/usr/bin/env python3
"""Register the sensitivity-matched compact-state full-MLP verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "examples/nanogpt/configs"
ARTIFACTS = CONFIGS / "selection_artifacts"
BASE = CONFIGS / "pro6_mai_v3_124m_fullattn_plus_fullmlp_errorfeedback_0p5tpp.json"
ORACLE = ARTIFACTS / "124m_full_mlp_temporal_state_hybrid_precision_result.json"
OUTPUT_CONFIG = CONFIGS / "pro6_mai_v3_124m_fullattn_fullmlp_hybridstate_0p5tpp.json"
OUTPUT_PLAN = ARTIFACTS / "124m_full_mlp_hybrid_state_0p5tpp_plan.json"

BASE_SHA256 = "d361e03ef4607bbbd7d8982e11da52fdaa8fcb7666ea16fe3d4fd06ba8bd8712"
ORACLE_SHA256 = "f6fabd14f604b6f5bd4423a601a57ac8400010066fd7eae295926d2af8552014"
IMPLEMENTATION_COMMIT = "bdf19a0b59b1baee8e7ac1863956864840be5fae"
DATASET_MANIFEST_SHA256 = "1e1de075c504906a93637bd79450d30da2243797d2e1d3e33f2392d9492ddf8b"
FIXED_EVAL_INDICES_SHA256 = "5ca31b59768e43de808ad5e206ed152a4a0a3515ad68d29a0b2338c4db140747"
SOURCE_HASHES = {
    "examples/nanogpt/mfu_preflight.py": "4f244e23be072602ef959694095a743306b35d1d2dcb27b7462fdbc002a28303",
    "examples/nanogpt/model.py": "c7c6d4a4356ededf717ed5010bf87d16df1dd190f436fca9b7291e98dd38ae14",
    "examples/nanogpt/muon.py": "4702dbee85408ab43112acf8c11c9f3e09fecdaf46345c1012a765c516ef44a1",
    "examples/nanogpt/muon_matched_givens.py": "11af80c72c54c34492cfaf5c9ef7a93e2f36af0a88a5caf51aa67bfe593b9032",
    "examples/nanogpt/train.py": "f37a02eb3d84a654f03907295823b64912da7311398e2fb447f4a75447901884",
    "latent_weight_lab/block_fht.py": "e0b692156130d11d55a240d192e9dde2077a77d0cfc4412356c5ddab49e80f36",
}

REMOTE_WORKTREE = "/mnt/ssd-data/orj/MappingNetworks/latent-weight-lab-paired-ambient-20260819"
PYTHON = "/mnt/ssd-data/orj/MappingNetworks/.venv/bin/python"
OUTPUT_ROOT = "/home/pro6000-9980x/MappingNetworks/outputs/pro6_mai_v3_full_mlp_hybridstate_0p5tpp"
RUN_DIR = f"{OUTPUT_ROOT}/scientific"
CERTIFICATE = f"{OUTPUT_ROOT}/performance_preflight.json"
PREFLIGHT_LOG = f"{OUTPUT_ROOT}/performance_preflight.log"
RUN_METADATA = f"{OUTPUT_ROOT}/prelaunch_run_metadata.json"
PERSISTENT_STATE_BYTES = 169_896_960
DENSE_FP32_STATE_BYTES = 452_984_832


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def validate_inputs() -> None:
    if sha256_file(BASE) != BASE_SHA256:
        raise RuntimeError("accepted full-MLP parent config hash drifted")
    if sha256_file(ORACLE) != ORACLE_SHA256:
        raise RuntimeError("hybrid-state oracle hash drifted")
    oracle = json.loads(ORACLE.read_text())
    if oracle["classification"] != "HYBRID_PRECISION_AMBIENT_STATE_PLAUSIBLE":
        raise RuntimeError("hybrid-state oracle did not pass")
    if not oracle["decision"]["codec_implementation_authorized"]:
        raise RuntimeError("hybrid-state oracle did not authorize implementation")
    if oracle["selected"]["persistent_storage_bytes"] != PERSISTENT_STATE_BYTES:
        raise RuntimeError("hybrid-state byte accounting drifted")
    for relative, digest in SOURCE_HASHES.items():
        if sha256_file(ROOT / relative) != digest:
            raise RuntimeError(f"training source hash drifted: {relative}")


def make_config() -> dict[str, Any]:
    config = json.loads(BASE.read_text())
    config.update(
        {
            "block_fht_mlp_muon_momentum_state_dtype": "float16",
            "block_fht_mlp_error_feedback_state_codec": "int8_blockwise",
            "block_fht_mlp_error_feedback_state_block_size": 4096,
            "candidate_scope": (
                "accepted full-attention plus full-MLP error-feedback structure; "
                "only persistent MLP Muon state changes to sensitivity-matched "
                "FP16 momentum and block-4096 int8 residuals"
            ),
            "hpo_stage": "full_mlp_hybrid_precision_state_0p5tpp_verification",
            "implementation_commit": IMPLEMENTATION_COMMIT,
            "implementation_source_hashes": SOURCE_HASHES,
            "implementation_test_evidence": {
                "remote": "44 passed on PRO6",
                "coverage": [
                    "deterministic int8 encode/decode",
                    "default FP32 update equivalence",
                    "exact compact-state checkpoint resume",
                    "legacy checkpoint state migration",
                    "compact codec limited to MLP c_fc and c_proj optimizers",
                ],
            },
            "mfu_measurement_protocol": (
                "foreground exact-config real-training preflight with one warmup "
                "and eight timed updates; includes FP16 momentum decode/store and "
                "block-4096 int8 feedback decode/re-encode on both MLP optimizers"
            ),
            "mfu_preflight_certificate": CERTIFICATE,
            "monitoring_policy": (
                "MFU/VRAM preflight is foreground-polled; the 238-update run uses "
                "one terminal/error-only watchdog because expected duration may "
                "exceed five minutes"
            ),
            "out_dir": RUN_DIR,
            "prelaunch_provenance_requirements": (
                "record implementation commit, source/config/dataset/fixed-eval "
                "hashes, literal command, exact-host MFU certificate, peak VRAM, "
                "status/log/checkpoint, and compact optimizer-state dtypes"
            ),
            "registered_resume_protocol": (
                "atomic latest checkpoint with full RNG state, folded c_fc/c_proj "
                "weights, FP16 Muon momentum, int8 residual codes, and FP16 "
                "block-4096 scales; exact continuation required"
            ),
            "run_metadata_path": RUN_METADATA,
            "selected_state_oracle": str(ORACLE.relative_to(ROOT)),
            "selected_state_oracle_sha256": ORACLE_SHA256,
            "temporal_state_representation": {
                "momentum": "FP16 for both full-MLP custom optimizers",
                "feedback": "signed int8 plus one FP16 max-absolute scale per 4096 elements",
                "persistent_storage_bytes": PERSISTENT_STATE_BYTES,
                "dense_fp32_storage_bytes": DENSE_FP32_STATE_BYTES,
                "storage_ratio": PERSISTENT_STATE_BYTES / DENSE_FP32_STATE_BYTES,
                "storage_reduction_factor": DENSE_FP32_STATE_BYTES / PERSISTENT_STATE_BYTES,
                "additional_trainable_parameters": 0,
            },
        }
    )
    return config


def make_plan(config_sha256: str) -> dict[str, Any]:
    remote_config = f"{REMOTE_WORKTREE}/examples/nanogpt/configs/{OUTPUT_CONFIG.name}"
    return {
        "schema_version": "mai_124m_full_mlp_hybrid_state_0p5tpp_plan_v1",
        "created_at": "2026-08-19",
        "status": "registered_before_exact_config_mfu_and_training",
        "question": (
            "Does the sensitivity-matched 0.375x persistent-state codec preserve "
            "the accepted full-replacement CE while reducing optimizer VRAM?"
        ),
        "candidate": {
            "config": str(OUTPUT_CONFIG.relative_to(ROOT)),
            "config_sha256": config_sha256,
            "parent_terminal_validation_ce": 5.529316425323486,
            "attention_only_validation_ce": 5.4918,
            "persistent_state_bytes": PERSISTENT_STATE_BYTES,
            "dense_fp32_state_bytes": DENSE_FP32_STATE_BYTES,
            "maximum_validation_ce": 5.5918,
        },
        "identity": {
            "base_config_sha256": BASE_SHA256,
            "oracle_sha256": ORACLE_SHA256,
            "implementation_commit": IMPLEMENTATION_COMMIT,
            "implementation_source_hashes": SOURCE_HASHES,
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "fixed_eval_indices_sha256": FIXED_EVAL_INDICES_SHA256,
        },
        "decision_rule": {
            "mfu_pass": "measured exact-config MFU >= 0.20",
            "memory_pass": "peak VRAM is below the 97,887 MiB device limit and below the 44,410 MiB FP32-state parent peak",
            "scientific_success": "finite terminal fixed-window validation CE <= 5.5918",
            "preferred": "terminal CE <= 5.529316425323486 with lower peak VRAM",
            "larger_rung": "not authorized by this plan",
        },
        "protocol": {
            "host": "PRO6",
            "gpu": 0,
            "python": PYTHON,
            "working_directory": REMOTE_WORKTREE,
            "run_directory": RUN_DIR,
            "certificate": CERTIFICATE,
            "preflight_log": PREFLIGHT_LOG,
            "preflight_command": [
                "env", "CUDA_VISIBLE_DEVICES=0",
                "CUDA_HOME=/mnt/ssd-data/orj/MappingNetworks/.cuda-12.8",
                "PYTHONPATH=.", PYTHON, "-u", "-m",
                "examples.nanogpt.mfu_preflight", "--config", remote_config,
                "--output", CERTIFICATE, "--log-output", PREFLIGHT_LOG,
                "--min-fraction", "0.2", "--warmup-updates", "1",
                "--timed-updates", "8",
            ],
            "training_command": [
                "env", "CUDA_VISIBLE_DEVICES=0",
                "CUDA_HOME=/mnt/ssd-data/orj/MappingNetworks/.cuda-12.8",
                "PYTHONPATH=.", PYTHON, "-u", "-m", "examples.nanogpt.train",
                "--config", remote_config,
            ],
            "preflight_monitor": "foreground polling",
            "training_monitor": "single terminal/error-only watchdog with @Codex action prompt",
        },
        "authorization": {
            "exact_config_preflight": True,
            "training_before_preflight_pass": False,
            "one_124m_training_after_preflight_pass": True,
            "automatic_rerun": False,
            "larger_rung": False,
        },
    }


def main() -> None:
    validate_inputs()
    config = make_config()
    OUTPUT_CONFIG.write_bytes(json_bytes(config))
    OUTPUT_PLAN.write_bytes(json_bytes(make_plan(hashlib.sha256(json_bytes(config)).hexdigest())))
    print(OUTPUT_CONFIG.relative_to(ROOT))
    print(OUTPUT_PLAN.relative_to(ROOT))


if __name__ == "__main__":
    main()
