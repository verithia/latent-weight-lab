#!/usr/bin/env python3
"""Preregister the 2x-coordinate c_fc error-feedback step-60 screen."""

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
    "directedproduct22x6_radius1p00_errorfeedback_step60.json"
)
PARENT_RESULT = (
    ARTIFACTS
    / "124m_mlp_cfc_directed_product_error_feedback_0p5tpp_result.json"
)
OUTPUT_CONFIG = (
    CONFIGS
    / "pro6_mai_v3_124m_fullattn_plus_mlp_cfc_"
    "directedproduct2x_radius1p00_errorfeedback_step60.json"
)
OUTPUT_PLAN = (
    ARTIFACTS
    / "124m_mlp_cfc_directed_product_2x_error_feedback_step60_plan.json"
)

BASE_SHA256 = (
    "ae556855c64458a8151d2072da8363d147358428e1e3afd2789f28344a55c832"
)
PARENT_RESULT_SHA256 = (
    "9b3770e89638700c297de9a0ca9ff04b70ed3cafb772ce6030609e2c10d037e6"
)
PARENT_RESULT_COMMIT = "adb88ff9d5b7245160d15d669b1bee3e4168b635"
DATASET_MANIFEST_SHA256 = (
    "1e1de075c504906a93637bd79450d30da2243797d2e1d3e33f2392d9492ddf8b"
)
FIXED_EVAL_INDICES_SHA256 = (
    "5ca31b59768e43de808ad5e206ed152a4a0a3515ad68d29a0b2338c4db140747"
)
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
SCHEDULE = [30, 30, 29, 29, 29, 29]
COORDINATES_PER_LAYER = 540_672
FAMILY_RADIUS_RATIO = 1.0
ERROR_FEEDBACK_DECAY = 1.0
ADDITIONAL_OPTIMIZER_STATE_BYTES = 113_246_208
MAX_ITERS = 60
LR_DECAY_ITERS = 238
CONTROL_VALIDATION_CE = 6.319543838500977
DENSE_CFC_VALIDATION_CE = 6.3141
PASS_VALIDATION_CE = 6.316821919250488

REMOTE_WORKTREE = (
    "/home/pro6000-9980x/MappingNetworks/"
    "latent-weight-lab-cfc-midpoint-replay"
)
PYTHON = "/mnt/ssd-data/orj/MappingNetworks/.venv/bin/python"
OUTPUT_ROOT = (
    "/home/pro6000-9980x/MappingNetworks/outputs/"
    "pro6_mai_v3_mlp_cfc_directed_product_2x_error_feedback_step60"
)
RUN_NAME = (
    "pro6_mai_v3_124m_cfc_directed_product_2x_"
    "radius1p00_errorfeedback_step60"
)
RUN_DIR = f"{OUTPUT_ROOT}/{RUN_NAME}"
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
        raise RuntimeError("22x6 error-feedback step-60 config hash drifted")
    if sha256_file(PARENT_RESULT) != PARENT_RESULT_SHA256:
        raise RuntimeError("full error-feedback result hash drifted")
    result = json.loads(PARENT_RESULT.read_text())
    if result["classification"] != "STABLE_IMPROVEMENT_WITHOUT_STRICT_CLOSURE":
        raise RuntimeError("full result does not authorize 2x screen")
    if not result["decision"][
        "two_x_error_feedback_step60_config_registration_authorized"
    ]:
        raise RuntimeError("full result did not authorize 2x screen registration")
    if result["decision"]["two_x_error_feedback_full_run_authorized"]:
        raise RuntimeError("2x full run was authorized before screening")
    for relative, digest in SOURCE_HASHES.items():
        if sha256_file(ROOT / relative) != digest:
            raise RuntimeError(f"error-feedback source hash drifted: {relative}")


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
            "block_fht_mlp_cfc_directed_product_schedule": SCHEDULE,
            "block_fht_mlp_cfc_directed_product_family_radius_ratio": (
                FAMILY_RADIUS_RATIO
            ),
            "block_fht_mlp_cfc_directed_product_error_feedback": True,
            "block_fht_mlp_cfc_directed_product_error_feedback_decay": (
                ERROR_FEEDBACK_DECAY
            ),
            "candidate_scope": (
                "single causal step-60 screen combining the prequalified 2x "
                "directed-product coordinate schedule with full-radius dense "
                "optimizer-side error feedback; all data, seeds, evaluation, "
                "and the 238-update LR schedule are held fixed"
            ),
            "hpo_stage": "directed_product_cfc_2x_error_feedback_step60_screen",
            "ladder_role": "mlp_cfc_instantaneous_plus_temporal_step60",
            "implementation_source_hashes": SOURCE_HASHES,
            "implementation_test_evidence": {
                "local": "42 passed, 40 subtests passed before registration",
                "coverage": [
                    "2x schedule and coordinate accounting",
                    "full error-feedback carry and exact resume",
                    "fixed evaluation and data identity",
                    "existing directed-product solver and wiring",
                ],
            },
            "lr_decay_iters": LR_DECAY_ITERS,
            "max_iters": MAX_ITERS,
            "mfu_measurement_protocol": (
                "direct foreground real CUDA BF16 training-path preflight with "
                "one warmup and eight timed updates; the exact 2x-coordinate "
                "c_fc support/refit, full-radius projection, error-feedback "
                "state update, qualified c_proj update, and folded "
                "materialization execute on every measured update"
            ),
            "mfu_preflight_certificate": CERTIFICATE,
            "mfu_preflight_result_planned": str(
                (
                    ARTIFACTS
                    / "124m_mlp_cfc_directed_product_2x_error_feedback_step60_mfu_result.json"
                ).relative_to(ROOT)
            ),
            "monitoring_policy": (
                "short 60-update run is directly foreground-polled; no "
                "watchdog, callback, queue worker, or heartbeat"
            ),
            "out_dir": RUN_DIR,
            "parent_selection_result": str(PARENT_RESULT.relative_to(ROOT)),
            "parent_selection_result_sha256": PARENT_RESULT_SHA256,
            "planned_tokens": 15_728_640,
            "planned_tpp": 0.126,
            "preregistered_decision_rule": {
                "primary_metric": (
                    "fixed-window validation cross entropy at update 60"
                ),
                "control_validation_ce": CONTROL_VALIDATION_CE,
                "dense_cfc_validation_ce": DENSE_CFC_VALIDATION_CE,
                "minimum_remaining_gap_fraction_to_recover": 0.5,
                "pass_validation_ce_maximum": PASS_VALIDATION_CE,
                "pass": (
                    "finite complete step-60 validation CE <= "
                    "6.316821919250488 with exact registered identity"
                ),
                "reject": (
                    "validation CE > 6.316821919250488, instability, incomplete "
                    "evaluation, or identity mismatch"
                ),
            },
            "run_metadata_path": RUN_METADATA,
            "screen_only": True,
            "screen_only_resolution": (
                "a pass authorizes registration and exact MFU measurement of "
                "one 238-update 2x-plus-error-feedback run; no full run, rerun, "
                "or larger rung is authorized before a new immutable plan"
            ),
            "selection_result_commit": PARENT_RESULT_COMMIT,
        }
    )
    representation = dict(config["directed_product_representation"])
    representation.update(
        {
            "incoming_schedule": SCHEDULE,
            "coordinates_per_layer": COORDINATES_PER_LAYER,
            "coordinate_fraction_per_cfc": 0.057291666666666664,
            "coordinate_multiplier_vs_22x6": 1.3333333333333333,
            "family_radius_ratio": FAMILY_RADIUS_RATIO,
            "temporal_error_feedback": True,
            "temporal_error_feedback_decay": ERROR_FEEDBACK_DECAY,
            "additional_dense_optimizer_state_bytes": (
                ADDITIONAL_OPTIMIZER_STATE_BYTES
            ),
            "additional_trainable_parameters": 0,
            "scope_limit": (
                "causal optimizer-state diagnostic; dense residual state is not "
                "claimed as a final optimizer-memory compression solution"
            ),
        }
    )
    config["directed_product_representation"] = representation
    return config


def make_plan(config_sha256: str) -> dict[str, Any]:
    remote_config = (
        f"{REMOTE_WORKTREE}/examples/nanogpt/configs/{OUTPUT_CONFIG.name}"
    )
    return {
        "schema_version": "mai_124m_mlp_cfc_2x_error_feedback_step60_plan_v1",
        "created_at": "2026-08-03",
        "status": "registered_before_exact_config_mfu_and_training",
        "question": (
            "After temporal projection error is controlled, does the "
            "prequalified 2x chart close at least half the remaining "
            "instantaneous c_fc midpoint gap?"
        ),
        "candidate": {
            "config": str(OUTPUT_CONFIG.relative_to(ROOT)),
            "config_sha256": config_sha256,
            "incoming_schedule": SCHEDULE,
            "coordinates_per_layer": COORDINATES_PER_LAYER,
            "coordinate_multiplier_vs_22x6": 1.3333333333333333,
            "family_radius_ratio": FAMILY_RADIUS_RATIO,
            "error_feedback": True,
            "error_feedback_decay": ERROR_FEEDBACK_DECAY,
            "additional_dense_optimizer_state_bytes": (
                ADDITIONAL_OPTIMIZER_STATE_BYTES
            ),
            "additional_trainable_parameters": 0,
        },
        "control": {
            "schedule": [22, 22, 22, 22, 22, 22],
            "error_feedback": True,
            "step60_validation_ce": CONTROL_VALIDATION_CE,
            "dense_cfc_step60_validation_ce": DENSE_CFC_VALIDATION_CE,
            "remaining_gap": CONTROL_VALIDATION_CE - DENSE_CFC_VALIDATION_CE,
        },
        "identity": {
            "base_config": str(BASE.relative_to(ROOT)),
            "base_config_sha256": BASE_SHA256,
            "parent_result": str(PARENT_RESULT.relative_to(ROOT)),
            "parent_result_sha256": PARENT_RESULT_SHA256,
            "parent_result_commit": PARENT_RESULT_COMMIT,
            "implementation_source_hashes": SOURCE_HASHES,
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "fixed_eval_indices_sha256": FIXED_EVAL_INDICES_SHA256,
        },
        "decision_rule": {
            "primary_metric": "fixed-window step-60 validation cross entropy",
            "minimum_remaining_gap_fraction_to_recover": 0.5,
            "pass_validation_ce_maximum": PASS_VALIDATION_CE,
            "pass": (
                "register one full 238-update 2x error-feedback config and "
                "measure its exact-config MFU before training"
            ),
            "failure": (
                "reject more instantaneous coordinates as the remaining repair "
                "and preserve the 22x6 error-feedback result"
            ),
            "threshold_changes_after_measurement": False,
        },
        "execution": {
            "host": "PRO6",
            "gpu": 0,
            "working_directory": REMOTE_WORKTREE,
            "python": PYTHON,
            "max_iters": MAX_ITERS,
            "lr_decay_iters": LR_DECAY_ITERS,
            "exact_config_mfu_minimum": 0.2,
            "mfu_certificate": CERTIFICATE,
            "mfu_log": PREFLIGHT_LOG,
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
                CERTIFICATE,
                "--log-output",
                PREFLIGHT_LOG,
                "--min-fraction",
                "0.2",
                "--warmup-updates",
                "1",
                "--timed-updates",
                "8",
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
                remote_config,
            ],
            "watchdog": False,
            "callback": False,
            "queue_worker": False,
        },
        "authorization": {
            "scientific_step60_run_authorized_after_exact_mfu_pass": True,
            "automatic_rerun_authorized": False,
            "full_238_update_run_authorized": False,
            "larger_rung_authorized": False,
            "additional_structure_authorized": False,
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
