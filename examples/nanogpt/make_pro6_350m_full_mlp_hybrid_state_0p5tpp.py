#!/usr/bin/env python3
"""Register the 350M transfer of the accepted hybrid temporal-state codec."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "examples/nanogpt/configs"
ARTIFACTS = CONFIGS / "selection_artifacts"
BASE = CONFIGS / "pro6_mai_v3_350m_fullattn_plus_fullmlp_cfcdecay1_cprojdecay0p5_0p5tpp.json"
PARENT_RESULT = ARTIFACTS / "350m_full_mlp_cfc_on_bounded_cproj_0p5tpp_result.json"
HYBRID_ORACLE = ARTIFACTS / "124m_full_mlp_temporal_state_hybrid_precision_result.json"
SMALLEST_RESULT = ARTIFACTS / "124m_full_mlp_hybrid_state_0p5tpp_result.json"
OUTPUT_CONFIG = CONFIGS / "pro6_mai_v3_350m_fullattn_fullmlp_hybridstate_0p5tpp.json"
OUTPUT_PLAN = ARTIFACTS / "350m_full_mlp_hybrid_state_0p5tpp_plan.json"

BASE_SHA256 = "a316a887a84c8ebe7b2037f3d0fe3b9d3ac1cbad1695b305b550f0e6fe9db250"
PARENT_RESULT_SHA256 = "40cf563a9cdc7395b37957c70956613b4b148d0188ba3bcbd5ae5ac6d506a182"
HYBRID_ORACLE_SHA256 = "f6fabd14f604b6f5bd4423a601a57ac8400010066fd7eae295926d2af8552014"
SMALLEST_RESULT_SHA256 = "c89118a58bb8e9bd75269f931bf0ef199800c2921bbe45b60e3e8bdd87f72ac5"
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
OUTPUT_ROOT = "/home/pro6000-9980x/MappingNetworks/outputs/pro6_mai_v3_350m_full_mlp_hybridstate_0p5tpp"
RUN_DIR = f"{OUTPUT_ROOT}/scientific"
CERTIFICATE = f"{OUTPUT_ROOT}/performance_preflight.json"
PREFLIGHT_LOG = f"{OUTPUT_ROOT}/performance_preflight.log"
RUN_METADATA = f"{OUTPUT_ROOT}/prelaunch_run_metadata.json"

MATRIX_ELEMENTS = 24 * 4096 * 1024 * 2
DENSE_FP32_STATE_BYTES = MATRIX_ELEMENTS * 2 * 4
MOMENTUM_FP16_BYTES = MATRIX_ELEMENTS * 2
FEEDBACK_INT8_BYTES = MATRIX_ELEMENTS
SCALE_COUNT = 24 * 2 * (4096 * 1024 // 4096)
SCALE_FP16_BYTES = SCALE_COUNT * 2
PERSISTENT_STATE_BYTES = MOMENTUM_FP16_BYTES + FEEDBACK_INT8_BYTES + SCALE_FP16_BYTES


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def validate_inputs() -> None:
    expected = {
        BASE: BASE_SHA256,
        PARENT_RESULT: PARENT_RESULT_SHA256,
        HYBRID_ORACLE: HYBRID_ORACLE_SHA256,
        SMALLEST_RESULT: SMALLEST_RESULT_SHA256,
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise RuntimeError(f"registered input hash drifted: {path.relative_to(ROOT)}")
    parent = json.loads(PARENT_RESULT.read_text())
    if parent["classification"] != "PASS_FULL_MLP_CFC_ON_BOUNDED_CPROJ_350M_CLOSES_ATTENTION_GAP":
        raise RuntimeError("350M full-MLP parent did not pass")
    smallest = json.loads(SMALLEST_RESULT.read_text())
    if smallest["classification"] != "PASS_124M_FULL_REPLACEMENT_HYBRID_STATE_PRESERVES_PARENT":
        raise RuntimeError("124M compact-state transfer did not pass")
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
                "accepted 350M full-attention plus full-MLP recipe with c_fc decay 1 "
                "and bounded c_proj decay 0.5; only persistent MLP Muon state changes "
                "to FP16 momentum and block-4096 int8 feedback"
            ),
            "failed_mfu_preflight": None,
            "hpo_stage": "full_mlp_hybrid_precision_state_transfer_350m_0p5tpp",
            "implementation_commit": IMPLEMENTATION_COMMIT,
            "implementation_source_hashes": SOURCE_HASHES,
            "implementation_test_evidence": {
                "remote": "46 passed on PRO6",
                "coverage": [
                    "deterministic int8 encode/decode",
                    "default FP32 update equivalence",
                    "exact compact-state checkpoint resume for c_fc and c_proj",
                    "compact-state dtype persistence",
                    "compact codec limited to MLP c_fc and c_proj optimizers",
                    "124M terminal checkpoint dtype audit",
                ],
            },
            "launch_block_reason": None,
            "launch_ready": True,
            "matched_full_mlp_parent_result": str(PARENT_RESULT.relative_to(ROOT)),
            "matched_full_mlp_parent_result_sha256": PARENT_RESULT_SHA256,
            "matched_full_mlp_parent_terminal_validation_loss": 4.425156116485596,
            "mfu_measurement_protocol": (
                "foreground exact-config real-training preflight with one warmup and "
                "eight timed updates; includes FP16 momentum decode/store and "
                "block-4096 int8 feedback decode/re-encode on both 350M MLP optimizers"
            ),
            "mfu_preflight_certificate": CERTIFICATE,
            "monitoring_policy": (
                "foreground-poll the exact-config MFU/VRAM gate; after a pass, the "
                "677-update run uses one terminal/error-only watchdog because its "
                "expected duration is within one to two hours"
            ),
            "out_dir": RUN_DIR,
            "prelaunch_provenance_requirements": (
                "record implementation commit, source/config/dataset/fixed-eval hashes, "
                "literal command, exact-host MFU certificate, peak VRAM, status, log, "
                "checkpoint, and compact optimizer-state dtypes"
            ),
            "preregistered_decision_rule": {
                "accepted_attention_gap": 0.1,
                "attention_only_terminal_validation_ce": 4.3629,
                "fp32_state_parent_terminal_validation_ce": 4.425156116485596,
                "nonbinding_parent_preservation_ce_maximum": 4.430156116485596,
                "pass_validation_ce_maximum": 4.4629,
                "primary_metric": "fixed-window validation cross entropy at update 677",
                "reject": (
                    "candidate is unstable, fails the 20 percent MFU gate, or finishes "
                    "above 4.4629 CE; do not promote the compact state codec"
                ),
                "success": (
                    "finite terminal validation CE at or below 4.4629, preserving the "
                    "accepted full attention plus full MLP replacement at 350M"
                ),
                "threshold_changes_after_measurement": False,
            },
            "registered_resume_protocol": (
                "atomic latest checkpoint with full RNG state, folded c_fc/c_proj "
                "weights, FP16 Muon momentum, int8 feedback codes, and FP16 "
                "block-4096 scales; exact continuation required"
            ),
            "run_metadata_path": RUN_METADATA,
            "selected_124m_hybrid_result": str(SMALLEST_RESULT.relative_to(ROOT)),
            "selected_124m_hybrid_result_sha256": SMALLEST_RESULT_SHA256,
            "selected_state_oracle": str(HYBRID_ORACLE.relative_to(ROOT)),
            "selected_state_oracle_sha256": HYBRID_ORACLE_SHA256,
            "selection_endpoint": (
                "terminal step-677 fixed-window validation CE versus the immutable "
                "350M attention-only and FP32-state full-MLP controls"
            ),
            "temporal_state_representation": {
                "momentum": "FP16 for both 350M full-MLP custom optimizers",
                "feedback": "signed int8 plus one FP16 max-absolute scale per 4096 elements",
                "momentum_bytes": MOMENTUM_FP16_BYTES,
                "feedback_code_bytes": FEEDBACK_INT8_BYTES,
                "feedback_scale_bytes": SCALE_FP16_BYTES,
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
        "schema_version": "mai_350m_full_mlp_hybrid_state_0p5tpp_plan_v1",
        "created_at": "2026-08-19",
        "status": "registered_before_exact_config_mfu_and_training",
        "question": (
            "Does the 124M-validated sensitivity-matched temporal-state codec preserve "
            "the accepted 350M full-replacement loss while reducing optimizer VRAM?"
        ),
        "candidate": {
            "config": str(OUTPUT_CONFIG.relative_to(ROOT)),
            "config_sha256": config_sha256,
            "only_scientific_change_from_parent": (
                "FP32 MLP momentum becomes FP16 and FP32 error feedback becomes "
                "signed int8 with one FP16 scale per 4096 elements"
            ),
            "parent_terminal_validation_ce": 4.425156116485596,
            "attention_only_validation_ce": 4.3629,
            "maximum_validation_ce": 4.4629,
            "persistent_state_bytes": PERSISTENT_STATE_BYTES,
            "dense_fp32_state_bytes": DENSE_FP32_STATE_BYTES,
        },
        "identity": {
            "base_config_sha256": BASE_SHA256,
            "parent_result_sha256": PARENT_RESULT_SHA256,
            "hybrid_oracle_sha256": HYBRID_ORACLE_SHA256,
            "smallest_hybrid_result_sha256": SMALLEST_RESULT_SHA256,
            "implementation_commit": IMPLEMENTATION_COMMIT,
            "implementation_source_hashes": SOURCE_HASHES,
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "fixed_eval_indices_sha256": FIXED_EVAL_INDICES_SHA256,
        },
        "decision_rule": {
            "mfu_pass": "measured exact-config MFU >= 0.20",
            "memory_pass": (
                "peak VRAM is below the 97,887 MiB device limit and below the "
                "53,317.05 MiB FP32-state 350M parent peak"
            ),
            "scientific_success": "finite terminal fixed-window validation CE <= 4.4629",
            "parent_preservation": "nonbinding terminal CE <= 4.430156116485596",
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
            "one_350m_training_after_preflight_pass": True,
            "automatic_rerun": False,
            "larger_rung": False,
        },
    }


def main() -> None:
    validate_inputs()
    config = make_config()
    encoded = json_bytes(config)
    OUTPUT_CONFIG.write_bytes(encoded)
    OUTPUT_PLAN.write_bytes(json_bytes(make_plan(hashlib.sha256(encoded).hexdigest())))
    print(OUTPUT_CONFIG.relative_to(ROOT))
    print(OUTPUT_PLAN.relative_to(ROOT))


if __name__ == "__main__":
    main()
