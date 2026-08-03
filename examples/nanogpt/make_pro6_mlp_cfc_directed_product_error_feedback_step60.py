#!/usr/bin/env python3
"""Preregister the full-radius 22x6 c_fc error-feedback screen."""

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
    / "pro6_mai_v3_124m_fullattn_plus_mlp_cfc_"
    "directedproduct22x6_radius1p00_step60.json"
)
PARENT_RESULT = (
    ARTIFACTS
    / "124m_mlp_cfc_directed_product_2x_radius1p00_step60_result.json"
)
OUTPUT_CONFIG = (
    CONFIGS
    / "pro6_mai_v3_124m_fullattn_plus_mlp_cfc_"
    "directedproduct22x6_radius1p00_errorfeedback_step60.json"
)
OUTPUT_PLAN = (
    ARTIFACTS
    / "124m_mlp_cfc_directed_product_error_feedback_step60_plan.json"
)

BASE_SHA256 = (
    "9337216260b6d6b8bf42895f5dd1b179714fa396f215da0681194f822e3f1619"
)
PARENT_RESULT_SHA256 = (
    "f8da32f130ef8a4c77f7e7ca6a7a26cd2abc411ad8426eda9337b3a643155552"
)
IMPLEMENTATION_COMMIT = "34b07c80dfa4d45bd11ec449afa36eb756ceb48a"
SOURCE_HASHES = {
    "examples/nanogpt/mfu_preflight.py": (
        "b9454210c6f6aec59aa39da7a9f1b36f111eba5f13127c39fcf47758a94179a3"
    ),
    "examples/nanogpt/model.py": (
        "c472a236ad61e529fe7a8939adc09529aea1fead0f8c937be02f77c3c7d53d5f"
    ),
    "examples/nanogpt/muon.py": (
        "532e172d91306d12284507c96aa3176792b33eb657f568512ce278bb5a9874ff"
    ),
    "examples/nanogpt/muon_matched_givens.py": (
        "d7780984879999c509972c87fa9c1e7dc4fa6634c661642f0a3df013813f99b0"
    ),
    "examples/nanogpt/test_muon_directed_product.py": (
        "17a0310c1c46147f65ef21968d7bc29397d14b86371235fde64ebd2674258885"
    ),
    "examples/nanogpt/train.py": (
        "b6b82e2ca74f80a085cfddbd6846de90d9e301b7431b91eac704ff5de6990886"
    ),
    "latent_weight_lab/block_fht.py": (
        "864ba9a79664cba2f830c06b11214538b7817685e1ba990f6e103feefb49b561"
    ),
}
DATASET_MANIFEST_SHA256 = (
    "1e1de075c504906a93637bd79450d30da2243797d2e1d3e33f2392d9492ddf8b"
)
FIXED_EVAL_INDICES_SHA256 = (
    "5ca31b59768e43de808ad5e206ed152a4a0a3515ad68d29a0b2338c4db140747"
)
SCHEDULE = [22, 22, 22, 22, 22, 22]
FAMILY_RADIUS_RATIO = 1.0
ERROR_FEEDBACK_DECAY = 1.0
ADDITIONAL_OPTIMIZER_STATE_BYTES = 113_246_208
CONTROL_STEP60_VALIDATION_CE = 6.347743988037109
DENSE_CFC_STEP60_VALIDATION_CE = 6.3141
PASS_VALIDATION_CE = 6.330921994018555
MAX_ITERS = 60
LR_DECAY_ITERS = 238

REMOTE_WORKTREE = (
    "/home/pro6000-9980x/MappingNetworks/"
    "latent-weight-lab-cfc-midpoint-replay"
)
PYTHON = "/mnt/ssd-data/orj/MappingNetworks/.venv/bin/python"
OUTPUT_ROOT = (
    "/home/pro6000-9980x/MappingNetworks/outputs/"
    "pro6_mai_v3_mlp_cfc_directed_product_error_feedback_step60"
)
RUN_NAME = (
    "pro6_mai_v3_124m_cfc_directed_product_22x6_"
    "radius1p00_errorfeedback_step60"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def validate_inputs() -> None:
    if sha256_file(BASE) != BASE_SHA256:
        raise RuntimeError("full-radius step-60 control config hash drifted")
    if sha256_file(PARENT_RESULT) != PARENT_RESULT_SHA256:
        raise RuntimeError("2x-coordinate rejection result hash drifted")
    parent = json.loads(PARENT_RESULT.read_text())
    if parent["classification"] != "STABLE_SAME_FAMILY_COORDINATE_SCREEN_REJECTED":
        raise RuntimeError("parent result does not authorize temporal repair")
    if parent["decision"]["full_238_update_run_authorized"] is not False:
        raise RuntimeError("parent result authorization drifted")
    for relative, digest in SOURCE_HASHES.items():
        if sha256_file(ROOT / relative) != digest:
            raise RuntimeError(f"error-feedback source hash drifted: {relative}")


def make_config() -> dict[str, Any]:
    config = json.loads(BASE.read_text())
    config.update(
        {
            "block_fht_mlp_cfc_directed_product_schedule": SCHEDULE,
            "block_fht_mlp_cfc_directed_product_family_radius_ratio": (
                FAMILY_RADIUS_RATIO
            ),
            "block_fht_mlp_cfc_directed_product_error_feedback": True,
            "block_fht_mlp_cfc_directed_product_error_feedback_decay": (
                ERROR_FEEDBACK_DECAY
            ),
            "candidate_scope": (
                "single causal step-60 screen of dense optimizer-state error "
                "feedback around the full-radius 22x6 directed-product c_fc "
                "control; the compressor, initialization, batches, fixed "
                "evaluation, and 238-update LR schedule are held fixed"
            ),
            "hpo_stage": "directed_product_cfc_error_feedback_step60_screen",
            "ladder_role": "mlp_cfc_temporal_error_feedback_step60",
            "implementation_commit": IMPLEMENTATION_COMMIT,
            "implementation_source_hashes": SOURCE_HASHES,
            "implementation_test_evidence": {
                "local": "36 passed, 40 subtests passed before config registration",
                "coverage": [
                    "uncompensated first-step identity",
                    "exact compression-residual carry",
                    "exact optimizer-state resume",
                    "checkpoint RNG and resume-envelope regression",
                    "existing directed-product solver and wiring"
                ]
            },
            "lr_decay_iters": LR_DECAY_ITERS,
            "max_iters": MAX_ITERS,
            "mfu_preflight_certificate": (
                f"{OUTPUT_ROOT}/performance_preflight.json"
            ),
            "monitoring_policy": (
                "short 60-update run is directly foreground-polled; no "
                "watchdog, callback, queue worker, or heartbeat"
            ),
            "out_dir": f"{OUTPUT_ROOT}/{RUN_NAME}",
            "parent_selection_result": str(PARENT_RESULT.relative_to(ROOT)),
            "parent_selection_result_sha256": PARENT_RESULT_SHA256,
            "planned_tokens": 15_728_640,
            "planned_tpp": 0.126,
            "preregistered_decision_rule": {
                "primary_metric": "fixed-window validation cross entropy at update 60",
                "control_validation_ce": CONTROL_STEP60_VALIDATION_CE,
                "dense_cfc_validation_ce": DENSE_CFC_STEP60_VALIDATION_CE,
                "minimum_remaining_gap_fraction_to_recover": 0.5,
                "pass_validation_ce_maximum": PASS_VALIDATION_CE,
                "pass": (
                    "finite complete step-60 validation CE <= 6.330922 with "
                    "exact registered source, data, and evaluation identity"
                ),
                "reject": (
                    "validation CE > 6.330922, instability, incomplete "
                    "step-60 evaluation, or identity mismatch"
                )
            },
            "run_metadata_path": f"{OUTPUT_ROOT}/prelaunch_run_metadata.json",
            "screen_only_resolution": (
                "a pass authorizes registration and exact MFU measurement of "
                "one 238-update error-feedback run; it does not authorize that "
                "run before a new immutable plan"
            )
        }
    )
    representation = dict(config["directed_product_representation"])
    representation.update(
        {
            "incoming_schedule": SCHEDULE,
            "coordinates_per_layer": 405_504,
            "coordinate_fraction_per_cfc": 0.04296875,
            "family_radius_ratio": FAMILY_RADIUS_RATIO,
            "temporal_error_feedback": True,
            "temporal_error_feedback_decay": ERROR_FEEDBACK_DECAY,
            "temporal_error_feedback_rule": (
                "compress requested_dense_update + prior_compression_residual; "
                "store corrected_target - applied_update for the next step"
            ),
            "additional_dense_optimizer_state_bytes": (
                ADDITIONAL_OPTIMIZER_STATE_BYTES
            ),
            "additional_trainable_parameters": 0,
            "scope_limit": (
                "causal optimizer-state diagnostic; dense residual state is not "
                "claimed as a final optimizer-memory compression solution"
            )
        }
    )
    config["directed_product_representation"] = representation
    return config


def make_plan(config_sha256: str) -> dict[str, Any]:
    remote_config = (
        f"{REMOTE_WORKTREE}/examples/nanogpt/configs/{OUTPUT_CONFIG.name}"
    )
    return {
        "schema_version": "mai_124m_mlp_cfc_error_feedback_step60_plan_v1",
        "created_at": "2026-08-03",
        "status": "registered_before_exact_config_mfu_and_training",
        "question": (
            "Is the remaining c_fc trajectory gap caused by repeatedly "
            "discarding the directed-product compression residual?"
        ),
        "identity": {
            "config": str(OUTPUT_CONFIG.relative_to(ROOT)),
            "config_sha256": config_sha256,
            "base_config": str(BASE.relative_to(ROOT)),
            "base_config_sha256": BASE_SHA256,
            "parent_result": str(PARENT_RESULT.relative_to(ROOT)),
            "parent_result_sha256": PARENT_RESULT_SHA256,
            "implementation_commit": IMPLEMENTATION_COMMIT,
            "implementation_source_hashes": SOURCE_HASHES,
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "fixed_eval_indices_sha256": FIXED_EVAL_INDICES_SHA256
        },
        "candidate": {
            "schedule": SCHEDULE,
            "family_radius_ratio": FAMILY_RADIUS_RATIO,
            "error_feedback": True,
            "error_feedback_decay": ERROR_FEEDBACK_DECAY,
            "algorithm": [
                "form exact current dense Muon update including weight decay",
                "add the prior full-precision compression residual",
                "fit and full-radius scale the unchanged 22x6 directed product",
                "apply the structured update",
                "store corrected target minus applied update in optimizer state"
            ],
            "additional_dense_optimizer_state_bytes": (
                ADDITIONAL_OPTIMIZER_STATE_BYTES
            ),
            "additional_trainable_parameters": 0,
            "forward_structure_changed": False
        },
        "control": {
            "error_feedback": False,
            "step60_validation_ce": CONTROL_STEP60_VALIDATION_CE,
            "dense_cfc_step60_validation_ce": DENSE_CFC_STEP60_VALIDATION_CE,
            "remaining_gap": (
                CONTROL_STEP60_VALIDATION_CE
                - DENSE_CFC_STEP60_VALIDATION_CE
            )
        },
        "decision_rule": {
            "primary_metric": "fixed-window step-60 validation cross entropy",
            "minimum_remaining_gap_fraction_to_recover": 0.5,
            "pass_validation_ce_maximum": PASS_VALIDATION_CE,
            "pass": (
                "register one full 238-update error-feedback config and measure "
                "its exact-config MFU before training"
            ),
            "failure": (
                "reject accumulated compression residual as the main cause and "
                "move to a different static action family"
            ),
            "threshold_changes_after_measurement": False
        },
        "execution": {
            "host": "PRO6",
            "gpu": 0,
            "working_directory": REMOTE_WORKTREE,
            "python": PYTHON,
            "max_iters": MAX_ITERS,
            "lr_decay_iters": LR_DECAY_ITERS,
            "exact_config_mfu_minimum": 0.2,
            "watchdog": False,
            "callback": False,
            "queue_worker": False,
            "mfu_certificate": f"{OUTPUT_ROOT}/performance_preflight.json",
            "mfu_log": f"{OUTPUT_ROOT}/performance_preflight.log",
            "mfu_command": [
                "env",
                "CUDA_VISIBLE_DEVICES=0",
                "CUDA_HOME=/mnt/ssd-data/orj/MappingNetworks/.cuda-12.8",
                "PYTHONPATH=.",
                PYTHON,
                "-u",
                "-m",
                "examples.nanogpt.mfu_preflight",
                "--config",
                remote_config,
                "--output",
                f"{OUTPUT_ROOT}/performance_preflight.json",
                "--log-output",
                f"{OUTPUT_ROOT}/performance_preflight.log",
                "--min-fraction",
                "0.2",
                "--warmup-updates",
                "1",
                "--timed-updates",
                "8"
            ],
            "training_command": [
                "env",
                "CUDA_VISIBLE_DEVICES=0",
                "CUDA_HOME=/mnt/ssd-data/orj/MappingNetworks/.cuda-12.8",
                "PYTHONPATH=.",
                PYTHON,
                "-u",
                "-m",
                "examples.nanogpt.train",
                "--config",
                remote_config
            ]
        },
        "authorization": {
            "scientific_step60_run_authorized_after_exact_mfu_pass": True,
            "automatic_rerun_authorized": False,
            "full_238_update_run_authorized": False,
            "larger_rung_authorized": False,
            "additional_structure_authorized": False
        }
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
