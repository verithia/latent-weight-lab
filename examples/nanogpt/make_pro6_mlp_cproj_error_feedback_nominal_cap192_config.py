#!/usr/bin/env python3
"""Bind the preregistered 124M c_proj nominal-step cap to an exact run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "examples/nanogpt/configs"
ARTIFACTS = CONFIGS / "selection_artifacts"
BASE = (
    CONFIGS
    / "pro6_mai_v3_124m_fullattn_plus_mlp_cproj_"
    "twopassfresh88_errorfeedback_0p5tpp.json"
)
PLAN = ARTIFACTS / "124m_mlp_cproj_error_feedback_nominal_cap192_plan.json"
OUTPUT_CONFIG = (
    CONFIGS
    / "pro6_mai_v3_124m_fullattn_plus_mlp_cproj_"
    "twopassfresh88_errorfeedback_cap192_0p5tpp.json"
)
RESOLUTION = (
    ARTIFACTS
    / "124m_mlp_cproj_error_feedback_nominal_cap192_resolution.json"
)

BASE_SHA256 = "fb7d8f5b4e5f8a98a30fa6216080146622f018403a5b7f8dddc1875803a81cd9"
PLAN_SHA256 = "b1c9fbd15c1483d99d8677300a372c31c8414efbfba83ba7a5fd1c7ab0973511"
IMPLEMENTATION_COMMIT = "5e19cf7d5a89d5360adeb2f13d41ffe009c7b0db"
DATASET_MANIFEST_SHA256 = "1e1de075c504906a93637bd79450d30da2243797d2e1d3e33f2392d9492ddf8b"
FIXED_EVAL_INDICES_SHA256 = "5ca31b59768e43de808ad5e206ed152a4a0a3515ad68d29a0b2338c4db140747"
SOURCE_HASHES = {
    "examples/nanogpt/mfu_preflight.py": "eb2312801e4b532d10540224aa2027ab70d1e5c68845febced341412eea1e985",
    "examples/nanogpt/model.py": "86bcae18fcf86e75af4e3929a239897ed18129e19e6fc79c92d4cf2eb7a54666",
    "examples/nanogpt/muon.py": "532e172d91306d12284507c96aa3176792b33eb657f568512ce278bb5a9874ff",
    "examples/nanogpt/muon_matched_givens.py": "b73ed8ed213b17f2d63046b7a75451c3f17b3516a9b77c0809d3b0a031461cf7",
    "examples/nanogpt/test_cproj_error_feedback_schedule.py": "68af908631af05e18ba5c8655ce8f68700b7a588fad98143db76c3bdc0805b21",
    "examples/nanogpt/test_muon_matched_givens.py": "30db3afa48881f8a294c6958844a33214e5922943ac178e0fcb8aa115b849b55",
    "examples/nanogpt/train.py": "56ee43523e124e785163fa3ae1a2a7c54a02c8c593042ee4367934d094df4000",
    "latent_weight_lab/block_fht.py": "864ba9a79664cba2f830c06b11214538b7817685e1ba990f6e103feefb49b561",
}

CAP_NOMINAL_STEPS = 192.0
PARENT_CE = 5.527365207672119
PASS_CE = 5.522365207672119
NEAR_CLOSE_CE = 5.5118
ATTENTION_ONLY_CE = 5.4918

WORKTREE = "/mnt/ssd-data/orj/MappingNetworks/latent-weight-lab"
PYTHON = "/home/pro6000-9980x/MappingNetworks/.venv/bin/python"
OUTPUT_ROOT = (
    "/home/pro6000-9980x/MappingNetworks/outputs/"
    "pro6_mai_v3_mlp_cproj_errorfeedback_cap192_0p5tpp"
)
RUN_DIR = f"{OUTPUT_ROOT}/scientific"
CERTIFICATE = f"{OUTPUT_ROOT}/performance_preflight.json"
PREFLIGHT_LOG = f"{OUTPUT_ROOT}/performance_preflight.log"
RUN_METADATA = f"{OUTPUT_ROOT}/prelaunch_run_metadata.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def validate_inputs() -> None:
    if sha256_file(BASE) != BASE_SHA256:
        raise RuntimeError("full-carry c_proj parent config hash drifted")
    if sha256_file(PLAN) != PLAN_SHA256:
        raise RuntimeError("nominal-cap preregistration hash drifted")
    base = json.loads(BASE.read_text())
    if base["data_manifest_sha256"] != DATASET_MANIFEST_SHA256:
        raise RuntimeError("dataset manifest identity drifted")
    for relative, digest in SOURCE_HASHES.items():
        if sha256_file(ROOT / relative) != digest:
            raise RuntimeError(f"production source hash drifted: {relative}")


def make_config() -> dict[str, Any]:
    config = json.loads(BASE.read_text())
    for stale in (
        "mfu_preflight_result",
        "mfu_preflight_result_sha256",
        "mfu_result_commit",
    ):
        config.pop(stale, None)
    config.update(
        {
            "block_fht_mlp_cproj_muon_matched_givens_error_feedback_max_nominal_steps": CAP_NOMINAL_STEPS,
            "candidate_scope": (
                "accepted full-attention replacement plus isolated mlp.c_proj "
                "hidden64+24 chart with decay-1 error feedback whose newly "
                "formed state is direction-preservingly capped at 192 nominal "
                "Muon steps; mlp.c_fc remains dense"
            ),
            "failed_mfu_preflight": None,
            "hpo_stage": "cproj_temporal_feedback_nominal_cap192_gate",
            "implementation_commit": IMPLEMENTATION_COMMIT,
            "implementation_source_hashes": SOURCE_HASHES,
            "implementation_test_evidence": {
                "focused_result": "40 passed",
                "full_suite_result": (
                    "568 passed, 57 subtests passed, 9 expected failures from "
                    "historical immutable source-hash fixtures"
                ),
                "coverage": [
                    "default-off and inactive bitwise identity",
                    "exact bound and direction preservation",
                    "diagnostic finiteness and active flag",
                    "configuration validation and c_proj-only group wiring",
                    "interrupted exact-resume identity",
                ],
            },
            "ladder_role": "mlp_cproj_nominal_cap192_smallest_rung_candidate",
            "mfu_measurement_protocol": (
                "foreground real-training preflight with 1 warmup and 8 timed "
                "updates; qualification additionally requires at least one "
                "logged active cap event in the scratch horizon"
            ),
            "mfu_preflight_certificate": CERTIFICATE,
            "monitoring_policy": (
                "direct foreground polling; no watchdog, callback, queue "
                "worker, or heartbeat for this short preflight and 238-update run"
            ),
            "out_dir": RUN_DIR,
            "preregistered_decision_rule": {
                "primary_metric": "terminal fixed-window validation CE at update 238",
                "parent_validation_ce": PARENT_CE,
                "minimum_improvement_ce": 0.005,
                "pass_validation_ce_maximum": PASS_CE,
                "near_close_validation_ce_maximum": NEAR_CLOSE_CE,
                "attention_only_validation_ce": ATTENTION_ONLY_CE,
                "required_cap_events": 1,
                "reject": (
                    "terminal CE above 5.522365207672119, no active cap event, "
                    "instability, incomplete terminal evaluation, or identity mismatch"
                ),
            },
            "run_metadata_path": RUN_METADATA,
        }
    )
    representation = dict(config["muon_matched_givens_representation"])
    representation.update(
        {
            "error_feedback_decay": 1.0,
            "error_feedback_max_nominal_steps": CAP_NOMINAL_STEPS,
            "feedback_cap_rule": (
                "post-compression direction-preserving Frobenius rescale to "
                "192 * abs(current_lr) * sqrt(output_rows) when exceeded"
            ),
        }
    )
    config["muon_matched_givens_representation"] = representation
    return config


def make_resolution(config_sha256: str) -> dict[str, Any]:
    remote_config = f"{WORKTREE}/examples/nanogpt/configs/{OUTPUT_CONFIG.name}"
    mfu_command = [
        "env",
        "CUDA_VISIBLE_DEVICES=0",
        "CUDA_HOME=/mnt/ssd-data/orj/MappingNetworks/.cuda-12.8",
        "MAPPING_NETWORKS_NATIVE_CACHE=/mnt/ssd-data/orj/MappingNetworks/outputs/native_cache",
        "PYTHONPATH=.",
        PYTHON,
        "-u",
        "-m",
        "examples.nanogpt.mfu_preflight",
        "--config",
        remote_config,
        "--output",
        CERTIFICATE,
        "--log-output",
        PREFLIGHT_LOG,
        "--min-fraction",
        "0.2",
        "--warmup-updates",
        "1",
        "--timed-updates",
        "8",
    ]
    scientific_command = [
        "env",
        "CUDA_VISIBLE_DEVICES=0",
        "CUDA_HOME=/mnt/ssd-data/orj/MappingNetworks/.cuda-12.8",
        "MAPPING_NETWORKS_NATIVE_CACHE=/mnt/ssd-data/orj/MappingNetworks/outputs/native_cache",
        "PYTHONPATH=.",
        PYTHON,
        "-u",
        "-m",
        "examples.nanogpt.train",
        "--config",
        remote_config,
    ]
    return {
        "schema_version": "mai_124m_mlp_cproj_nominal_cap192_resolution_v1",
        "recorded_at": "2026-08-05",
        "status": "bound_before_performance_or_scientific_execution",
        "plan": {
            "path": str(PLAN.relative_to(ROOT)),
            "sha256": PLAN_SHA256,
        },
        "config": {
            "path": str(OUTPUT_CONFIG.relative_to(ROOT)),
            "sha256": config_sha256,
        },
        "implementation": {
            "commit": IMPLEMENTATION_COMMIT,
            "source_hashes": SOURCE_HASHES,
        },
        "scientific_invariants": {
            "only_scientific_change_from_parent": (
                "error_feedback_max_nominal_steps: null -> 192.0"
            ),
            "parent_config_sha256": BASE_SHA256,
            "decision_threshold_changed": False,
            "mlp_cfc_dense": True,
        },
        "execution": {
            "host": "PRO6",
            "gpu": 0,
            "working_directory": WORKTREE,
            "python": PYTHON,
            "mfu_command": mfu_command,
            "scientific_command": scientific_command,
            "certificate": CERTIFICATE,
            "preflight_log": PREFLIGHT_LOG,
            "prelaunch_metadata": RUN_METADATA,
            "direct_foreground_polling": True,
            "watchdog": False,
            "callback": False,
        },
        "gates": {
            "minimum_mfu_fraction": 0.20,
            "required_timed_updates": 8,
            "required_scratch_cap_events": 1,
            "scientific_promotion_ceiling": PASS_CE,
            "required_scientific_cap_events": 1,
        },
        "authorization": {
            "single_mfu_preflight": True,
            "one_scientific_run_only_after_full_gate": True,
            "automatic_rerun_authorized": False,
            "cap_sweep_authorized": False,
            "larger_model_or_token_rung_authorized": False,
        },
    }


def main() -> None:
    validate_inputs()
    config = make_config()
    OUTPUT_CONFIG.write_bytes(json_bytes(config))
    config_sha256 = sha256_file(OUTPUT_CONFIG)
    RESOLUTION.write_bytes(json_bytes(make_resolution(config_sha256)))
    print(f"config={OUTPUT_CONFIG} sha256={config_sha256}")
    print(f"resolution={RESOLUTION} sha256={sha256_file(RESOLUTION)}")


if __name__ == "__main__":
    main()
