#!/usr/bin/env python3
"""Preregister the selected c_fc error-feedback 124M/0.5TPP run."""

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
SCREEN_RESULT = (
    ARTIFACTS
    / "124m_mlp_cfc_directed_product_error_feedback_step60_result.json"
)
OUTPUT_CONFIG = (
    CONFIGS
    / "pro6_mai_v3_124m_fullattn_plus_mlp_cfc_"
    "directedproduct22x6_radius1p00_errorfeedback_0p5tpp.json"
)
OUTPUT_PLAN = (
    ARTIFACTS
    / "124m_mlp_cfc_directed_product_error_feedback_0p5tpp_plan.json"
)

BASE_SHA256 = (
    "ae556855c64458a8151d2072da8363d147358428e1e3afd2789f28344a55c832"
)
SCREEN_RESULT_SHA256 = (
    "3cc70a2f21264432b496baf1f022ae8bac6064010bf045b1203ec3c3b3c4426a"
)
SCREEN_RESULT_COMMIT = "a70f4b05a8708295efacbf60e978ddcc9f64dd04"
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

MAX_ITERS = 238
SCHEDULE = [22, 22, 22, 22, 22, 22]
FAMILY_RADIUS_RATIO = 1.0
ERROR_FEEDBACK_DECAY = 1.0
ADDITIONAL_OPTIMIZER_STATE_BYTES = 113_246_208
ATTENTION_ONLY_VALIDATION_CE = 5.4918
SUCCESS_CE = 5.5918
DENSE_CFC_VALIDATION_CE = 5.592058181762695
UNCOMPENSATED_VALIDATION_CE = 5.617342948913574

REMOTE_WORKTREE = (
    "/home/pro6000-9980x/MappingNetworks/"
    "latent-weight-lab-cfc-midpoint-replay"
)
PYTHON = "/mnt/ssd-data/orj/MappingNetworks/.venv/bin/python"
OUTPUT_ROOT = (
    "/home/pro6000-9980x/MappingNetworks/outputs/"
    "pro6_mai_v3_mlp_cfc_directed_product_error_feedback_0p5tpp"
)
RUN_NAME = (
    "pro6_mai_v3_124m_cfc_directed_product_22x6_"
    "radius1p00_errorfeedback_0p5tpp"
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
        raise RuntimeError("error-feedback step-60 config hash drifted")
    if sha256_file(SCREEN_RESULT) != SCREEN_RESULT_SHA256:
        raise RuntimeError("error-feedback step-60 result hash drifted")
    result = json.loads(SCREEN_RESULT.read_text())
    if result["classification"] != "STABLE_TEMPORAL_ERROR_FEEDBACK_SCREEN_PASSED":
        raise RuntimeError("step-60 result does not authorize full config")
    if not result["decision"]["full_238_update_config_registration_authorized"]:
        raise RuntimeError("step-60 result did not authorize registration")
    if result["decision"]["full_238_update_run_authorized"]:
        raise RuntimeError("full run was authorized before exact MFU")
    base = json.loads(BASE.read_text())
    if base["data_manifest_sha256"] != DATASET_MANIFEST_SHA256:
        raise RuntimeError("dataset manifest identity drifted")
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
                "one selected 124M/0.5TPP verification of the full-attention, "
                "qualified c_proj, and 22x6 full-radius c_fc stack with dense "
                "optimizer-side compression error feedback"
            ),
            "hpo_stage": "directed_product_cfc_error_feedback_0p5tpp_validation",
            "ladder_role": "mlp_full_replacement_error_feedback_smallest_rung",
            "implementation_source_hashes": SOURCE_HASHES,
            "implementation_test_evidence": {
                "local": "40 passed, 40 subtests passed before full registration",
                "coverage": [
                    "uncompensated first-step identity",
                    "exact compression-residual carry",
                    "exact optimizer-state resume",
                    "checkpoint RNG and resume-envelope regression",
                    "existing directed-product solver and wiring",
                ],
            },
            "lr_decay_iters": MAX_ITERS,
            "max_iters": MAX_ITERS,
            "mfu_preflight_certificate": CERTIFICATE,
            "mfu_preflight_result_planned": str(
                (
                    ARTIFACTS
                    / "124m_mlp_cfc_directed_product_error_feedback_0p5tpp_mfu_result.json"
                ).relative_to(ROOT)
            ),
            "monitoring_policy": (
                "short 238-update run is directly foreground-polled; no "
                "watchdog, callback, queue worker, or heartbeat"
            ),
            "out_dir": RUN_DIR,
            "parent_selection_result": str(SCREEN_RESULT.relative_to(ROOT)),
            "parent_selection_result_sha256": SCREEN_RESULT_SHA256,
            "planned_tokens": 62_186_880,
            "planned_tpp": 0.5,
            "preregistered_decision_rule": {
                "primary_metric": (
                    "terminal fixed-window validation cross entropy at update 238"
                ),
                "attention_only_validation_ce": ATTENTION_ONLY_VALIDATION_CE,
                "accepted_attention_gap": 0.1,
                "success_ce_maximum": SUCCESS_CE,
                "dense_cfc_validation_ce": DENSE_CFC_VALIDATION_CE,
                "uncompensated_full_radius_validation_ce": (
                    UNCOMPENSATED_VALIDATION_CE
                ),
                "success": (
                    "finite complete terminal validation CE <= 5.5918, closing "
                    "full replacement to at most +0.10 versus attention-only"
                ),
                "improvement_without_closure": (
                    "terminal validation CE > 5.5918 and < 5.617342948913574"
                ),
                "reject": (
                    "terminal validation CE >= 5.617342948913574, instability, "
                    "incomplete terminal evaluation, or identity mismatch"
                ),
            },
            "registered_resume_protocol": (
                "atomic latest checkpoint with full RNG state, folded c_fc and "
                "c_proj weights, exact Muon momentum, and compression residuals"
            ),
            "run_metadata_path": RUN_METADATA,
            "screen_only": False,
            "screen_only_resolution": (
                "exactly one 124M/0.5TPP error-feedback verification is "
                "authorized after its exact config passes MFU; no automatic "
                "rerun or larger rung"
            ),
            "selection_result_commit": SCREEN_RESULT_COMMIT,
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
        "schema_version": "mai_124m_mlp_cfc_error_feedback_0p5tpp_plan_v1",
        "created_at": "2026-08-03",
        "status": "registered_before_exact_config_mfu_and_training",
        "question": (
            "Does optimizer-side error feedback close full MLP replacement to "
            "within +0.10 CE of attention-only at 124M/0.5TPP?"
        ),
        "candidate": {
            "config": str(OUTPUT_CONFIG.relative_to(ROOT)),
            "config_sha256": config_sha256,
            "incoming_schedule": SCHEDULE,
            "coordinates_per_layer": 405_504,
            "family_radius_ratio": FAMILY_RADIUS_RATIO,
            "error_feedback": True,
            "error_feedback_decay": ERROR_FEEDBACK_DECAY,
            "additional_dense_optimizer_state_bytes": (
                ADDITIONAL_OPTIMIZER_STATE_BYTES
            ),
            "additional_trainable_parameters": 0,
            "max_iters": MAX_ITERS,
            "planned_tpp": 0.5,
        },
        "identity": {
            "base_config": str(BASE.relative_to(ROOT)),
            "base_config_sha256": BASE_SHA256,
            "screen_result": str(SCREEN_RESULT.relative_to(ROOT)),
            "screen_result_sha256": SCREEN_RESULT_SHA256,
            "screen_result_commit": SCREEN_RESULT_COMMIT,
            "implementation_source_hashes": SOURCE_HASHES,
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "fixed_eval_indices_sha256": FIXED_EVAL_INDICES_SHA256,
        },
        "decision_rule": {
            "success": "finite complete terminal validation CE <= 5.5918",
            "improvement_without_closure": (
                "terminal validation CE > 5.5918 and < 5.617342948913574"
            ),
            "reject": (
                "terminal validation CE >= 5.617342948913574, nonfinite path, "
                "incomplete evaluation, or identity mismatch"
            ),
            "threshold_changes_after_measurement": False,
        },
        "protocol": {
            "host": "PRO6",
            "gpu": 0,
            "python": PYTHON,
            "working_directory": REMOTE_WORKTREE,
            "run_directory": RUN_DIR,
            "prelaunch_run_metadata": RUN_METADATA,
            "exact_config_certificate": CERTIFICATE,
            "exact_config_preflight_log": PREFLIGHT_LOG,
            "exact_config_mfu_minimum": 0.2,
            "exact_config_preflight_command": [
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
            "execution": "direct foreground polling through terminal exit",
            "watchdog": False,
            "callback": False,
            "heartbeat": False,
            "queue_worker": False,
        },
        "authorization": {
            "scope": "exactly one 124M/0.5TPP error-feedback run",
            "training_requires_exact_config_mfu_pass": True,
            "training_authorized_before_exact_mfu": False,
            "automatic_rerun_authorized": False,
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
