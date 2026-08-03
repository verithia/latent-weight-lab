#!/usr/bin/env python3
"""Preregister the native-extension-qualified directed-product MFU retry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "examples/nanogpt/configs"
ARTIFACTS = CONFIGS / "selection_artifacts"
PARENT_CONFIG = (
    CONFIGS
    / "pro6_mai_v3_124m_fullattn_plus_mlp_cfc_"
    "directedproduct30_29_29_mfu.json"
)
OUTPUT_CONFIG = (
    CONFIGS
    / "pro6_mai_v3_124m_fullattn_plus_mlp_cfc_"
    "directedproduct30_29_29_mfu_retry1.json"
)
FAILED_RESULT = (
    ARTIFACTS / "124m_mlp_cfc_directed_product_mfu_result.json"
)
OUTPUT_PLAN = (
    ARTIFACTS / "124m_mlp_cfc_directed_product_mfu_retry1_plan.json"
)
IMPLEMENTATION_COMMIT = "29d5e90ff419cde1b13fc5541aff79b12ec49f27"
PARENT_CONFIG_SHA256 = (
    "bd9b1d044ad7d1dfb6104261bdaed980e2717bb61cb816b756f8263edf9458e4"
)
FAILED_RESULT_SHA256 = (
    "f914c41b4450ef8599fdc7167e32ca99d419d22131a35c8ecdb24a79f6bbc611"
)
DATASET_MANIFEST_SHA256 = (
    "1e1de075c504906a93637bd79450d30da2243797d2e1d3e33f2392d9492ddf8b"
)
SOURCE_HASHES = {
    "examples/nanogpt/model.py": (
        "07602c5045077a848ab1e0a176431dde6b15c07be08b271a56276e91ad13ceae"
    ),
    "examples/nanogpt/muon_matched_givens.py": (
        "b2973183268673859272837c80a13a8ddeec6b2bd5a43cef1703bcc9a039c641"
    ),
    "examples/nanogpt/train.py": (
        "bc43b09497dd396025f1335c40889698ba1e7f4d5ad7ca76809e6e8d388cda44"
    ),
    "examples/nanogpt/test_muon_directed_product.py": (
        "681ed76c96e1521bea019462b2d3f95af4960ce746e06204458a7a6f4ebf8694"
    ),
    "examples/nanogpt/mfu_preflight.py": (
        "b9454210c6f6aec59aa39da7a9f1b36f111eba5f13127c39fcf47758a94179a3"
    ),
    "examples/nanogpt/muon.py": (
        "532e172d91306d12284507c96aa3176792b33eb657f568512ce278bb5a9874ff"
    ),
    "latent_weight_lab/block_fht.py": (
        "864ba9a79664cba2f830c06b11214538b7817685e1ba990f6e103feefb49b561"
    ),
}
REMOTE_WORKTREE = (
    "/home/pro6000-9980x/MappingNetworks/"
    "latent-weight-lab-cfc-midpoint-replay"
)
PYTHON = "/mnt/ssd-data/orj/MappingNetworks/.venv/bin/python"
OUTPUT_ROOT = (
    "/home/pro6000-9980x/MappingNetworks/outputs/"
    "pro6_mai_v3_mlp_cfc_directed_product_mfu_retry1"
)
CERTIFICATE = f"{OUTPUT_ROOT}/performance_preflight.json"
PREFLIGHT_LOG = f"{OUTPUT_ROOT}/preflight.log"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def validate_inputs() -> None:
    if sha256_file(PARENT_CONFIG) != PARENT_CONFIG_SHA256:
        raise RuntimeError("parent MFU config hash drifted")
    if sha256_file(FAILED_RESULT) != FAILED_RESULT_SHA256:
        raise RuntimeError("failed MFU result hash drifted")
    for relative, expected in SOURCE_HASHES.items():
        if sha256_file(ROOT / relative) != expected:
            raise RuntimeError(f"runtime source hash drifted: {relative}")


def make_config() -> dict[str, Any]:
    config = json.loads(PARENT_CONFIG.read_text())
    config.update(
        {
            "block_fht_native_extension_required": True,
            "failed_mfu_preflight": {
                "result": str(FAILED_RESULT.relative_to(ROOT)),
                "result_sha256": FAILED_RESULT_SHA256,
                "classification": (
                    "MFU_RUNTIME_REJECTED_NATIVE_EXTENSION_UNAVAILABLE"
                ),
                "mfu_fraction": 0.008357316546691727,
            },
            "hpo_stage": (
                "directed_product_cfc_native_runtime_repair_mfu_retry1"
            ),
            "implementation_commit": IMPLEMENTATION_COMMIT,
            "implementation_source_hashes": SOURCE_HASHES,
            "mfu_preflight_certificate": CERTIFICATE,
            "out_dir": f"{OUTPUT_ROOT}/preflight_scratch",
            "runtime_environment": {
                "python": PYTHON,
                "torch": "2.7.0+cu128",
                "cuda": "12.8",
                "ninja_present": True,
                "native_extension_precheck": {
                    "extension_loaded": True,
                    "extension_error": None,
                    "cuda_home": (
                        "/mnt/ssd-data/orj/MappingNetworks/.cuda-12.8"
                    ),
                },
            },
            "screen_only_resolution": (
                "one native-extension-qualified directly polled MFU retry; "
                "scientific training and larger rungs remain unauthorized"
            ),
        }
    )
    return config


def make_plan(config_sha256: str) -> dict[str, Any]:
    remote_config = (
        f"{REMOTE_WORKTREE}/examples/nanogpt/configs/{OUTPUT_CONFIG.name}"
    )
    command = [
        "env",
        "CUDA_VISIBLE_DEVICES=0",
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
    return {
        "schema_version": (
            "mai_124m_mlp_cfc_directed_product_mfu_retry1_plan_v1"
        ),
        "created_at": "2026-08-03",
        "status": "registered_after_runtime_failure_before_retry",
        "question": (
            "Does the unchanged selected 30+29+29 c_fc path pass 20% MFU "
            "when the BlockFHT native CUDA extension is required and loaded?"
        ),
        "candidate": {
            "config": str(OUTPUT_CONFIG.relative_to(ROOT)),
            "config_sha256": config_sha256,
            "implementation_commit": IMPLEMENTATION_COMMIT,
            "incoming_schedule": [30, 29, 29],
            "coordinates_per_layer": 88 * 3072,
            "scientific_structure_changed_from_failed_attempt": False,
        },
        "failed_attempt": {
            "result": str(FAILED_RESULT.relative_to(ROOT)),
            "result_sha256": FAILED_RESULT_SHA256,
            "failure": "native extension unavailable because ninja was absent",
            "directed_product_optimizer_ms": 372.8275,
            "prepare_plus_flush_ms": 54880.41625,
        },
        "identity": {
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "implementation_source_hashes": SOURCE_HASHES,
        },
        "runtime_preconditions": {
            "python": PYTHON,
            "torch": "2.7.0+cu128",
            "cuda": "12.8",
            "ninja_present": True,
            "block_fht_native_extension_required": True,
            "precheck_observed_extension_loaded": True,
        },
        "protocol": {
            "host": "PRO6",
            "gpu": 0,
            "warmup_updates": 1,
            "timed_updates": 8,
            "minimum_mfu_fraction": 0.2,
            "denominator": (
                "same-host empirical BF16 8192-square tensor-core GEMM peak"
            ),
            "execution": "direct foreground polling through terminal exit",
            "watchdog": False,
            "callback": False,
            "queue_worker": False,
            "heartbeat": False,
            "command": command,
            "certificate": CERTIFICATE,
            "log": PREFLIGHT_LOG,
        },
        "decision_rule": {
            "pass": (
                "native extension guard passes, exit zero, finite complete "
                "certificate has all eight timed updates, and MFU >= 0.20"
            ),
            "reject": (
                "native extension guard fails, MFU < 0.20, nonfinite path, "
                "incomplete timing, provenance mismatch, or nonzero exit"
            ),
            "threshold_changes_after_measurement": False,
        },
        "authorization": {
            "scope": "exactly one directly polled repaired-runtime MFU retry",
            "scientific_training_authorized": False,
            "larger_rung_authorized": False,
            "additional_structure_authorized": False,
        },
    }


def main() -> None:
    validate_inputs()
    config = make_config()
    config_payload = json_bytes(config)
    config_sha256 = hashlib.sha256(config_payload).hexdigest()
    plan = make_plan(config_sha256)
    OUTPUT_CONFIG.write_bytes(config_payload)
    OUTPUT_PLAN.write_bytes(json_bytes(plan))
    print(
        json.dumps(
            {
                "config": str(OUTPUT_CONFIG),
                "config_sha256": config_sha256,
                "plan": str(OUTPUT_PLAN),
                "plan_sha256": sha256_file(OUTPUT_PLAN),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
