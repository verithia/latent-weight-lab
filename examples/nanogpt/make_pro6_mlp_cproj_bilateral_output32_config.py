#!/usr/bin/env python3
"""Register the smallest endpoint-selected bilateral c_proj MFU candidate."""

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
BASE_SHA256 = "fb7d8f5b4e5f8a98a30fa6216080146622f018403a5b7f8dddc1875803a81cd9"
SELECTION = (
    ARTIFACTS
    / "124m_mlp_cproj_bilateral_endpoint_fixed_eval_pro6_replay_result.json"
)
SELECTION_SHA256 = "c7a181dd41405da424e215806ebe92d75b939700bf93d7d964b850736aa11a35"
CONTROL_RESULT = ARTIFACTS / "124m_mlp_cproj_error_feedback_0p5tpp_result.json"
CONTROL_RESULT_SHA256 = (
    "272f8709ddc805175542b2f163398d9823141aca0e1f6026f699a51bc5af87df"
)
OUTPUT_CONFIG = (
    CONFIGS
    / "pro6_mai_v3_124m_fullattn_plus_mlp_cproj_"
    "bilateral_output32_errorfeedback_0p5tpp.json"
)
OUTPUT_PLAN = (
    ARTIFACTS / "124m_mlp_cproj_bilateral_output32_mfu_plan.json"
)
IMPLEMENTATION_COMMIT = "6cb178d7ddc55c4157bb4ba9ec6c54f628fad6f0"
SOURCE_HASHES = {
    "examples/nanogpt/csrc/task_edge_coloring.cpp": (
        "988e1ae64f13061a5ae02eb1227523cea6a5e84b084121ba8f877b391dd8a53f"
    ),
    "examples/nanogpt/fast_task_matching.py": (
        "ec922fd31e136cf7a5c0ddff87f2f20db0da141ea9411a71ff7a25ba6b25c7c2"
    ),
    "examples/nanogpt/mfu_preflight.py": (
        "4a89b1b68d901072b39773ff2c461a1863765ccd4722ff7d6ddbb396d5a0e6aa"
    ),
    "examples/nanogpt/model.py": (
        "f2c22ca42a9acef7da34b22952d1cbac5c1b7a8fb6707bd7648174a523f62cf0"
    ),
    "examples/nanogpt/muon.py": (
        "532e172d91306d12284507c96aa3176792b33eb657f568512ce278bb5a9874ff"
    ),
    "examples/nanogpt/muon_matched_givens.py": (
        "98b6c9958ae2a38fd6ce25f269d9427fe8e2c8ce4ebfb78c082d127d4073ca90"
    ),
    "examples/nanogpt/test_muon_matched_givens.py": (
        "b5daa6acdb2c7844da79b525cb56ee37e6589bebc65b01a42d64747b2430acfb"
    ),
    "examples/nanogpt/train.py": (
        "a29473d8bc16f27990de2ea959640104e8a61ca29160492f95c5f162817cd59e"
    ),
    "latent_weight_lab/block_fht.py": (
        "864ba9a79664cba2f830c06b11214538b7817685e1ba990f6e103feefb49b561"
    ),
}

WORKTREE = "/home/pro6000-9980x/MappingNetworks/latent-weight-lab"
PYTHON = "/home/pro6000-9980x/MappingNetworks/.venv/bin/python"
OUTPUT_ROOT = (
    "/home/pro6000-9980x/MappingNetworks/outputs/"
    "pro6_mai_v3_mlp_cproj_bilateral_output32_0p5tpp"
)
CERTIFICATE = f"{OUTPUT_ROOT}/performance_preflight.json"
PREFLIGHT_LOG = f"{OUTPUT_ROOT}/performance_preflight.log"
RUN_DIR = f"{OUTPUT_ROOT}/scientific"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def validate_inputs() -> None:
    for path, expected in (
        (BASE, BASE_SHA256),
        (SELECTION, SELECTION_SHA256),
        (CONTROL_RESULT, CONTROL_RESULT_SHA256),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"registered input drifted: {path}")
    selected = json.loads(SELECTION.read_text())
    if selected["decision"]["selected_variant"] != (
        "hidden88_output32_full_carry"
    ):
        raise RuntimeError("endpoint evidence does not select output32")
    if not selected["decision"]["production_implementation_authorized"]:
        raise RuntimeError("endpoint evidence does not authorize implementation")
    if selected["decision"]["language_model_training_authorized"]:
        raise RuntimeError("endpoint evidence prematurely authorizes training")
    for relative, expected in SOURCE_HASHES.items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"implementation source drifted: {relative}")


def make_config() -> dict[str, Any]:
    config = json.loads(BASE.read_text())
    config.update(
        {
            "block_fht_mlp_cproj_muon_matched_givens_output_stages": 32,
            "block_fht_native_extension_required": True,
            "candidate_scope": (
                "accepted full-attention replacement plus materialized "
                "mlp.c_proj updated by fresh hidden64, hidden24 residual, "
                "and output32 fits to the remaining transposed residual; "
                "constant full compression-error carry; mlp.c_fc stays dense"
            ),
            "failed_mfu_preflight": None,
            "hpo_stage": "cproj_bilateral_output32_exact_path_mfu_gate",
            "implementation_commit": IMPLEMENTATION_COMMIT,
            "implementation_source_hashes": SOURCE_HASHES,
            "implementation_test_evidence": {
                "host": "PRO6",
                "result": "46 passed, 40 subtests passed",
                "coverage": [
                    "output pass receives the transposed post-hidden residual",
                    "output matching seed and stage count",
                    "combined coordinate accounting",
                    "materialized model and optimizer exact state round trip",
                    "existing custom optimizer and RNG regressions",
                ],
            },
            "ladder_role": "mlp_cproj_bilateral_output32_smallest_rung_candidate",
            "mfu_measurement_protocol": (
                "directly polled real-training preflight with one warmup and "
                "eight timed updates; exact Muon direction, hidden64 parent, "
                "hidden24 residual, output32 transposed-residual fit, full "
                "error feedback, native validation, and folded update all run"
            ),
            "mfu_preflight_certificate": CERTIFICATE,
            "monitoring_policy": (
                "direct foreground polling; no watchdog, callback, queue "
                "worker, or heartbeat for the short MFU gate"
            ),
            "out_dir": RUN_DIR,
            "parent_bilateral_endpoint_result": str(SELECTION.relative_to(ROOT)),
            "parent_bilateral_endpoint_result_sha256": SELECTION_SHA256,
            "matched_right_only_control_result": str(
                CONTROL_RESULT.relative_to(ROOT)
            ),
            "matched_right_only_control_result_sha256": CONTROL_RESULT_SHA256,
            "preregistered_decision_rule": {
                "primary_metric": (
                    "terminal fixed-window validation cross entropy at update 238"
                ),
                "attention_only_validation_ce": 5.4918,
                "right_only_full_carry_validation_ce": 5.527365207672119,
                "strict_close_validation_ce_maximum": 5.5118,
                "minimum_directional_gain_ce": 0.002,
                "directional_ce_maximum": 5.525365207672119,
                "strict_close": (
                    "stable terminal validation CE <= 5.5118, leaving at "
                    "most +0.0200 CE over attention-only"
                ),
                "directional_only": (
                    "5.5118 < CE <= 5.525365207672119, at least 0.002 "
                    "better than the paired right-only full-carry control"
                ),
                "reject": (
                    "CE > 5.525365207672119, nonfinite state, incomplete "
                    "terminal evaluation, identity mismatch, or failed MFU"
                ),
                "threshold_changes_after_measurement": False,
            },
            "registered_resume_protocol": (
                "atomic_latest_checkpoint_v2 with full RNG state, folded "
                "parent/residual/output Givens buffers, exact Muon momentum, "
                "and dense compression residual"
            ),
            "screen_only": True,
            "screen_only_resolution": (
                "this exact config is authorized only for the directly "
                "polled MFU gate until a passing certificate and separate "
                "training plan are recorded"
            ),
        }
    )
    representation = dict(config["muon_matched_givens_representation"])
    representation.update(
        {
            "coordinates_per_layer": 147456,
            "coordinate_fraction_per_cproj": 0.0625,
            "output_matching_stages": 32,
            "total_matching_stages": 120,
            "matching_policy": (
                "fresh hidden64 on the corrected current Muon request, fresh "
                "hidden24 on its residual, then fresh output32 on the "
                "transpose of the remaining residual; fold all three before "
                "decoupled weight decay and carry corrected minus actual"
            ),
            "persistent_resume_state_addition": [
                "output selected and inverse permutations",
                "output last angles",
            ],
        }
    )
    config["muon_matched_givens_representation"] = representation
    return config


def make_plan(config_hash: str) -> dict[str, Any]:
    remote_config = f"{WORKTREE}/examples/nanogpt/configs/{OUTPUT_CONFIG.name}"
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
        "schema_version": "mai_124m_mlp_cproj_bilateral_output32_mfu_plan_v1",
        "recorded_at": "2026-08-04",
        "status": "registered_before_exact_config_mfu_measurement",
        "scientific_question": (
            "Can the endpoint-selected bilateral output32 c_proj optimizer "
            "execute its complete causal real-training path at >=20% MFU?"
        ),
        "candidate": {
            "config": str(OUTPUT_CONFIG.relative_to(ROOT)),
            "config_sha256": config_hash,
            "implementation_commit": IMPLEMENTATION_COMMIT,
            "parent_stages": 64,
            "residual_stages": 24,
            "output_stages": 32,
            "neighbors": 64,
            "error_feedback": True,
            "error_feedback_decay": 1.0,
            "coordinates_per_layer": 147456,
            "coordinate_fraction_per_cproj": 0.0625,
            "learned_basis": False,
            "dense_residual": False,
        },
        "identity": {
            "base_config": str(BASE.relative_to(ROOT)),
            "base_config_sha256": BASE_SHA256,
            "endpoint_selection_result": str(SELECTION.relative_to(ROOT)),
            "endpoint_selection_result_sha256": SELECTION_SHA256,
            "right_only_control_result": str(CONTROL_RESULT.relative_to(ROOT)),
            "right_only_control_result_sha256": CONTROL_RESULT_SHA256,
            "source_hashes": SOURCE_HASHES,
            "dataset_manifest_sha256": (
                "1e1de075c504906a93637bd79450d30da2243797d2e1d3e33f2392d9492ddf8b"
            ),
            "fixed_eval_indices_sha256": (
                "5ca31b59768e43de808ad5e206ed152a4a0a3515ad68d29a0b2338c4db140747"
            ),
        },
        "decision_rule": {
            "pass": (
                "exit zero, finite complete certificate bound to the exact "
                "config hash, all eight timed real-training updates present, "
                "native parent/residual/output matcher outputs validated, "
                "and measured MFU >=0.20"
            ),
            "reject": (
                "MFU <0.20, nonfinite path, incomplete timing, missing output "
                "diagnostics, native validation failure, provenance mismatch, "
                "or nonzero exit"
            ),
            "threshold_changes_after_measurement": False,
        },
        "execution": {
            "host": "PRO6",
            "gpu": 0,
            "working_directory": WORKTREE,
            "mfu_command": command,
            "mfu_certificate": CERTIFICATE,
            "mfu_log": PREFLIGHT_LOG,
            "mfu_polling": "foreground direct polling through terminal exit",
            "minimum_mfu_fraction": 0.2,
            "warmup_updates": 1,
            "timed_updates": 8,
            "watchdog": False,
            "callback": False,
            "heartbeat": False,
            "queue_worker": False,
        },
        "authorization": {
            "one_exact_config_mfu_preflight_authorized": True,
            "scientific_training_authorized": False,
            "automatic_retry_authorized": False,
            "larger_rung_authorized": False,
            "on_pass": (
                "record the certificate and register one 124M 0.5-TPP "
                "scientific training plan without changing structure or CE gates"
            ),
        },
    }


def main() -> None:
    validate_inputs()
    config = make_config()
    OUTPUT_CONFIG.write_bytes(dump(config))
    plan = make_plan(sha256(OUTPUT_CONFIG))
    OUTPUT_PLAN.write_bytes(dump(plan))
    print(f"config={OUTPUT_CONFIG} sha256={sha256(OUTPUT_CONFIG)}")
    print(f"plan={OUTPUT_PLAN} sha256={sha256(OUTPUT_PLAN)}")


if __name__ == "__main__":
    main()
