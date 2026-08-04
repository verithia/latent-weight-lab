#!/usr/bin/env python3
"""Preregister the 124M c_proj halfway-bounded carry MFU gate."""

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
SELECTION = (
    ARTIFACTS
    / "124m_mlp_cproj_teacher_forced_carry_schedule_result.json"
)
OUTPUT_CONFIG = (
    CONFIGS
    / "pro6_mai_v3_124m_fullattn_plus_mlp_cproj_"
    "twopassfresh88_errorfeedback_halfswitch_0p5tpp.json"
)
OUTPUT_PLAN = (
    ARTIFACTS
    / "124m_mlp_cproj_error_feedback_half_switch_0p5tpp_mfu_plan.json"
)

BASE_SHA256 = "fb7d8f5b4e5f8a98a30fa6216080146622f018403a5b7f8dddc1875803a81cd9"
SELECTION_SHA256 = "51beef439a8bc9fb765357d617883ae017dbba55369c3441bd0e89b605cb2152"
SELECTION_COMMIT = "13d397f62ca939ce42d0ecb90d41d09a4e0800de"
IMPLEMENTATION_COMMIT = "f11c020e669175906a56beba9a159f869259a627"
DATASET_MANIFEST_SHA256 = "1e1de075c504906a93637bd79450d30da2243797d2e1d3e33f2392d9492ddf8b"
FIXED_EVAL_INDICES_SHA256 = "5ca31b59768e43de808ad5e206ed152a4a0a3515ad68d29a0b2338c4db140747"
SOURCE_HASHES = {
    "examples/nanogpt/mfu_preflight.py": "4a89b1b68d901072b39773ff2c461a1863765ccd4722ff7d6ddbb396d5a0e6aa",
    "examples/nanogpt/model.py": "4f93fd50141c00858b9ff92e2149e29da04b40e52150111b9a77a29e20357e51",
    "examples/nanogpt/muon.py": "532e172d91306d12284507c96aa3176792b33eb657f568512ce278bb5a9874ff",
    "examples/nanogpt/muon_matched_givens.py": "9532eabdf3538eca2c0774ab6f7e35cd05fd28a59c451cf9c41b64d0376cf0fc",
    "examples/nanogpt/test_cproj_error_feedback_schedule.py": "8e87d6665b67f53a56397a263d57741b5a3b8f7cadaa0f32d974f67e63641ec5",
    "examples/nanogpt/test_muon_matched_givens.py": "848c00e907ee9fb683ce5c4aea688af6542bf3d7b883a4775bd6837409e51775",
    "examples/nanogpt/train.py": "1f81dc03a37b49fad8ef8348bbafffcde81a9af919420ba22b52b158d51764a5",
    "latent_weight_lab/block_fht.py": "864ba9a79664cba2f830c06b11214538b7817685e1ba990f6e103feefb49b561",
}

DECAY_BEFORE = 1.0
DECAY_AFTER = 0.5
SWITCH_ITER = 120
MAX_ITERS = 238
SWITCH_FRACTION = SWITCH_ITER / MAX_ITERS
ATTENTION_ONLY_CE = 5.4918
CONSTANT_DECAY1_CPROJ_CE = 5.527365207672119
STRICT_CPROJ_GAP = 0.02

WORKTREE = "/mnt/ssd-data/orj/MappingNetworks/latent-weight-lab"
PYTHON = "/home/pro6000-9980x/MappingNetworks/.venv/bin/python"
OUTPUT_ROOT = (
    "/home/pro6000-9980x/MappingNetworks/outputs/"
    "pro6_mai_v3_mlp_cproj_errorfeedback_halfswitch_0p5tpp"
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
        raise RuntimeError("constant-decay c_proj config hash drifted")
    if sha256_file(SELECTION) != SELECTION_SHA256:
        raise RuntimeError("carry-schedule selection result hash drifted")
    selection = json.loads(SELECTION.read_text())
    if selection["classification"] != "SELECT_HALF_SWITCH_FOR_PRODUCTION_PREFLIGHT":
        raise RuntimeError("selection result does not select half-switch")
    decision = selection["decision"]
    if not decision["one_exact_124m_mfu_preflight_authorized_after_implementation"]:
        raise RuntimeError("selection result does not authorize MFU preflight")
    if decision["language_model_training_authorized"]:
        raise RuntimeError("selection prematurely authorizes training")
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
            "block_fht_mlp_cproj_muon_matched_givens_error_feedback_decay": DECAY_BEFORE,
            "block_fht_mlp_cproj_muon_matched_givens_error_feedback_decay_after": DECAY_AFTER,
            "block_fht_mlp_cproj_muon_matched_givens_error_feedback_switch_fraction": SWITCH_FRACTION,
            "candidate_scope": (
                "accepted full-attention replacement plus isolated mlp.c_proj "
                "hidden64+24 chart with decay 1.0 through update 119 and "
                "decay 0.5 from update 120; mlp.c_fc remains dense"
            ),
            "failed_mfu_preflight": None,
            "hpo_stage": "cproj_temporal_carry_half_switch_mfu_gate",
            "implementation_commit": IMPLEMENTATION_COMMIT,
            "implementation_source_hashes": SOURCE_HASHES,
            "implementation_test_evidence": {
                "host": "PRO6",
                "result": "53 passed, 40 subtests passed",
                "coverage": [
                    "exact 120/238 boundary",
                    "normalized short-horizon scaling",
                    "resume-time schedule recomputation",
                    "c_proj-only optimizer-group isolation",
                    "existing custom optimizer and RNG regressions",
                ],
            },
            "ladder_role": "mlp_cproj_half_switch_smallest_rung_candidate",
            "mfu_measurement_protocol": (
                "foreground real-training preflight with 1 warmup and 8 timed "
                "updates; the scratch horizon is normalized so both carry "
                "phases execute during the gate"
            ),
            "mfu_preflight_certificate": CERTIFICATE,
            "mfu_preflight_result_planned": str(
                (
                    ARTIFACTS
                    / "124m_mlp_cproj_error_feedback_half_switch_0p5tpp_mfu_result.json"
                ).relative_to(ROOT)
            ),
            "monitoring_policy": (
                "direct foreground polling; no watchdog, callback, queue "
                "worker, or heartbeat for the short preflight"
            ),
            "out_dir": RUN_DIR,
            "parent_selection_result": str(SELECTION.relative_to(ROOT)),
            "parent_selection_result_sha256": SELECTION_SHA256,
            "preregistered_decision_rule": {
                "primary_metric": "terminal fixed-window validation CE at update 238",
                "strict_close": (
                    "finite terminal CE <= 5.5118, i.e. c_proj contributes "
                    "at most +0.02 CE over attention-only"
                ),
                "partial_improvement": (
                    "5.5118 < terminal CE < 5.527365207672119"
                ),
                "reject": (
                    "terminal CE >= 5.527365207672119, instability, incomplete "
                    "terminal evaluation, or identity mismatch"
                ),
                "attention_only_validation_ce": ATTENTION_ONLY_CE,
                "strict_validation_ce_maximum": ATTENTION_ONLY_CE + STRICT_CPROJ_GAP,
                "constant_decay1_cproj_validation_ce": CONSTANT_DECAY1_CPROJ_CE,
            },
            "run_metadata_path": RUN_METADATA,
            "selection_result_commit": SELECTION_COMMIT,
        }
    )
    representation = dict(config["muon_matched_givens_representation"])
    representation.update(
        {
            "error_feedback_decay": DECAY_BEFORE,
            "error_feedback_decay_after": DECAY_AFTER,
            "error_feedback_switch_fraction": SWITCH_FRACTION,
            "error_feedback_switch_iter_at_124m_0p5tpp": SWITCH_ITER,
            "temporal_rule": (
                "full carry for updates 0..119, bounded half carry for "
                "updates 120..237; recompute phase from resumed iter_num"
            ),
        }
    )
    config["muon_matched_givens_representation"] = representation
    return config


def make_plan(config_sha256: str) -> dict[str, Any]:
    remote_config = (
        f"{WORKTREE}/examples/nanogpt/configs/{OUTPUT_CONFIG.name}"
    )
    command = [
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
    return {
        "schema_version": "mai_124m_mlp_cproj_half_switch_mfu_plan_v1",
        "recorded_at": "2026-08-04",
        "status": "registered_before_exact_config_mfu_measurement",
        "scientific_question": (
            "Can the replay-selected halfway-bounded c_proj carry execute the "
            "exact real training path at >=20% MFU before loss validation?"
        ),
        "candidate": {
            "config": str(OUTPUT_CONFIG.relative_to(ROOT)),
            "config_sha256": config_sha256,
            "implementation_commit": IMPLEMENTATION_COMMIT,
            "decay_before": DECAY_BEFORE,
            "decay_after": DECAY_AFTER,
            "switch_iter": SWITCH_ITER,
            "max_iters": MAX_ITERS,
            "switch_fraction": SWITCH_FRACTION,
            "parent_stages": 64,
            "residual_stages": 24,
        },
        "identity": {
            "base_config": str(BASE.relative_to(ROOT)),
            "base_config_sha256": BASE_SHA256,
            "selection_result": str(SELECTION.relative_to(ROOT)),
            "selection_result_sha256": SELECTION_SHA256,
            "selection_result_commit": SELECTION_COMMIT,
            "source_hashes": SOURCE_HASHES,
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "fixed_eval_indices_sha256": FIXED_EVAL_INDICES_SHA256,
        },
        "decision_rule": {
            "pass": (
                "exit zero, finite complete certificate bound to the exact "
                "config hash, all 8 timed updates present, both carry phases "
                "executed, native matcher validated, and MFU >= 0.20"
            ),
            "reject": (
                "MFU < 0.20, nonfinite path, incomplete timing, native "
                "validation failure, provenance mismatch, or nonzero exit"
            ),
            "threshold_changes_after_measurement": False,
        },
        "protocol": {
            "host": "PRO6",
            "gpu": 0,
            "working_directory": WORKTREE,
            "python": PYTHON,
            "command": command,
            "certificate": CERTIFICATE,
            "log": PREFLIGHT_LOG,
            "warmup_updates": 1,
            "timed_updates": 8,
            "denominator": "same-host empirical BF16 8192-square GEMM peak",
            "execution": "direct foreground process polled through exit",
            "watchdog": False,
            "callback": False,
            "heartbeat": False,
            "queue_worker": False,
        },
        "authorization": {
            "parameter_updates_persisted": 0,
            "single_exact_mfu_preflight": True,
            "scientific_training": False,
            "automatic_rerun": False,
            "larger_rung": False,
        },
    }


def main() -> None:
    validate_inputs()
    config = make_config()
    OUTPUT_CONFIG.write_bytes(json_bytes(config))
    config_sha256 = sha256_file(OUTPUT_CONFIG)
    OUTPUT_PLAN.write_bytes(json_bytes(make_plan(config_sha256)))
    print(f"config={OUTPUT_CONFIG} sha256={config_sha256}")
    print(f"plan={OUTPUT_PLAN} sha256={sha256_file(OUTPUT_PLAN)}")


if __name__ == "__main__":
    main()
