#!/usr/bin/env python3
"""Preregister the 2x-coordinate, full-radius c_fc step-60 screen."""

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
FULL_RADIUS_RESULT = (
    ARTIFACTS
    / "124m_mlp_cfc_directed_product_22x6_radius1p00_0p5tpp_result.json"
)
CAPACITY_RESULT = (
    ARTIFACTS
    / "124m_mlp_cfc_directed_product_terminal_capacity_result.json"
)
OUTPUT_CONFIG = (
    CONFIGS
    / "pro6_mai_v3_124m_fullattn_plus_mlp_cfc_"
    "directedproduct2x_radius1p00_step60.json"
)
OUTPUT_PLAN = (
    ARTIFACTS
    / "124m_mlp_cfc_directed_product_2x_radius1p00_step60_plan.json"
)

BASE_SHA256 = (
    "9337216260b6d6b8bf42895f5dd1b179714fa396f215da0681194f822e3f1619"
)
FULL_RADIUS_RESULT_SHA256 = (
    "e9da5f7e34a6e8224b98cb707c4f6c544bccebcf65e131c06551f4c9c67c774f"
)
CAPACITY_RESULT_SHA256 = (
    "62836f9c95ae28e129cbbe5b6c9a8450d31f2636f1570c2382cb5c84fb64a59b"
)
DATASET_MANIFEST_SHA256 = (
    "1e1de075c504906a93637bd79450d30da2243797d2e1d3e33f2392d9492ddf8b"
)
FIXED_EVAL_INDICES_SHA256 = (
    "5ca31b59768e43de808ad5e206ed152a4a0a3515ad68d29a0b2338c4db140747"
)
SCHEDULE = [30, 30, 29, 29, 29, 29]
COORDINATES_PER_LAYER = 540_672
COORDINATE_FRACTION_PER_CFC = 0.057291666666666664
FAMILY_RADIUS_RATIO = 1.0
CONTROL_STEP60_VALIDATION_CE = 6.347743988037109
DENSE_CFC_STEP60_VALIDATION_CE = 6.3141
PASS_VALIDATION_CE = 6.330921994018555
MAX_ITERS = 60
LR_DECAY_ITERS = 238
PLANNED_TOKENS = 15_728_640
PLANNED_TPP = 0.126

REMOTE_WORKTREE = (
    "/home/pro6000-9980x/MappingNetworks/"
    "latent-weight-lab-cfc-midpoint-replay"
)
PYTHON = "/mnt/ssd-data/orj/MappingNetworks/.venv/bin/python"
OUTPUT_ROOT = (
    "/home/pro6000-9980x/MappingNetworks/outputs/"
    "pro6_mai_v3_mlp_cfc_directed_product_2x_radius1p00_step60"
)
RUN_NAME = "pro6_mai_v3_124m_cfc_directed_product_2x_radius1p00_step60"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def validate_inputs() -> None:
    expected = {
        BASE: BASE_SHA256,
        FULL_RADIUS_RESULT: FULL_RADIUS_RESULT_SHA256,
        CAPACITY_RESULT: CAPACITY_RESULT_SHA256,
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise RuntimeError(f"preregistered input hash drifted: {path}")

    base = json.loads(BASE.read_text())
    if base["data_manifest_sha256"] != DATASET_MANIFEST_SHA256:
        raise RuntimeError("dataset manifest identity drifted")
    if base["block_fht_mlp_cfc_directed_product_family_radius_ratio"] != 1.0:
        raise RuntimeError("base is not the selected full-radius screen config")
    for relative, digest in base["implementation_source_hashes"].items():
        if sha256_file(ROOT / relative) != digest:
            raise RuntimeError(f"production source hash drifted: {relative}")

    full_radius = json.loads(FULL_RADIUS_RESULT.read_text())
    if full_radius["classification"] != "STABLE_DIRECTIONAL_ONLY_TERMINAL_CE":
        raise RuntimeError("full-radius result does not authorize coordinate test")
    fixed = {row["step"]: row["validation"] for row in full_radius["loss"]["fixed_evaluations"]}
    if fixed[60] != 6.3479:
        raise RuntimeError("rounded full-radius step-60 metric drifted")

    capacity = json.loads(CAPACITY_RESULT.read_text())
    selected = capacity["candidate_summaries"]["depth6_total176"]
    if selected["schedule"] != SCHEDULE:
        raise RuntimeError("offline-qualified 2x schedule drifted")
    if selected["coordinates_per_layer"] != COORDINATES_PER_LAYER:
        raise RuntimeError("offline-qualified coordinate count drifted")
    if selected["minimum_positive_line_recovery"] < 0.75:
        raise RuntimeError("offline 2x candidate no longer meets capacity gate")


def make_config() -> dict[str, Any]:
    config = json.loads(BASE.read_text())
    config.update(
        {
            "block_fht_mlp_cfc_directed_product_schedule": SCHEDULE,
            "block_fht_mlp_cfc_directed_product_family_radius_ratio": (
                FAMILY_RADIUS_RATIO
            ),
            "candidate_scope": (
                "single causal step-60 screen of the offline-qualified "
                "six-stage 2x-coordinate c_fc chart at full radius; exact "
                "initialization, data order, evaluation indices, and 238-update "
                "learning-rate schedule are held fixed"
            ),
            "hpo_stage": "directed_product_cfc_2x_radius1_step60_screen",
            "ladder_role": "mlp_cfc_2x_coordinate_step60",
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
            "parent_selection_result": str(
                FULL_RADIUS_RESULT.relative_to(ROOT)
            ),
            "parent_selection_result_sha256": FULL_RADIUS_RESULT_SHA256,
            "capacity_diagnostic_result": str(CAPACITY_RESULT.relative_to(ROOT)),
            "capacity_diagnostic_result_sha256": CAPACITY_RESULT_SHA256,
            "planned_tokens": PLANNED_TOKENS,
            "planned_tpp": PLANNED_TPP,
            "preregistered_decision_rule": {
                "primary_metric": "fixed-window validation cross entropy at update 60",
                "control_schedule": [22, 22, 22, 22, 22, 22],
                "control_coordinates_per_layer": 405_504,
                "control_validation_ce": CONTROL_STEP60_VALIDATION_CE,
                "dense_cfc_validation_ce": DENSE_CFC_STEP60_VALIDATION_CE,
                "remaining_control_gap": (
                    CONTROL_STEP60_VALIDATION_CE
                    - DENSE_CFC_STEP60_VALIDATION_CE
                ),
                "minimum_remaining_gap_fraction_to_recover": 0.5,
                "pass_validation_ce_maximum": PASS_VALIDATION_CE,
                "pass": (
                    "finite complete step-60 validation CE <= 6.330922 with "
                    "exact registered identity"
                ),
                "reject": (
                    "validation CE > 6.330922, instability, incomplete "
                    "step-60 evaluation, or identity mismatch"
                ),
            },
            "run_metadata_path": f"{OUTPUT_ROOT}/prelaunch_run_metadata.json",
            "screen_only_resolution": (
                "a passing screen authorizes registration and exact MFU "
                "measurement of one full 238-update 2x-coordinate run; it does "
                "not authorize that full run before a new immutable plan"
            ),
        }
    )
    representation = dict(config["directed_product_representation"])
    representation.update(
        {
            "incoming_schedule": SCHEDULE,
            "coordinates_per_layer": COORDINATES_PER_LAYER,
            "coordinate_fraction_per_cfc": COORDINATE_FRACTION_PER_CFC,
            "family_radius_ratio": FAMILY_RADIUS_RATIO,
            "family_radius_calibration": (
                "full radius selected by the prior causal radius screen"
            ),
            "schedule_selection": (
                "offline terminal capacity candidate depth6_total176; only "
                "coordinate reach changes relative to the 22x6 control"
            ),
        }
    )
    config["directed_product_representation"] = representation
    return config


def make_plan(config_sha256: str) -> dict[str, Any]:
    base = json.loads(BASE.read_text())
    remote_config = (
        f"{REMOTE_WORKTREE}/examples/nanogpt/configs/{OUTPUT_CONFIG.name}"
    )
    return {
        "schema_version": "mai_124m_mlp_cfc_2x_radius1_step60_plan_v1",
        "created_at": "2026-08-03",
        "status": "registered_before_exact_config_mfu_and_training",
        "question": (
            "Does the offline-qualified 2x-coordinate six-stage chart close "
            "at least half of the remaining online step-60 directional gap?"
        ),
        "identity": {
            "config": str(OUTPUT_CONFIG.relative_to(ROOT)),
            "config_sha256": config_sha256,
            "base_config": str(BASE.relative_to(ROOT)),
            "base_config_sha256": BASE_SHA256,
            "full_radius_result": str(FULL_RADIUS_RESULT.relative_to(ROOT)),
            "full_radius_result_sha256": FULL_RADIUS_RESULT_SHA256,
            "capacity_result": str(CAPACITY_RESULT.relative_to(ROOT)),
            "capacity_result_sha256": CAPACITY_RESULT_SHA256,
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "fixed_eval_indices_sha256": FIXED_EVAL_INDICES_SHA256,
            "implementation_source_hashes": base["implementation_source_hashes"],
        },
        "candidate": {
            "schedule": SCHEDULE,
            "coordinates_per_layer": COORDINATES_PER_LAYER,
            "coordinate_multiplier_vs_original": 2.0,
            "coordinate_multiplier_vs_22x6_control": 4.0 / 3.0,
            "family_radius_ratio": FAMILY_RADIUS_RATIO,
            "offline_minimum_positive_line_recovery": 0.7503036425412035,
        },
        "control": {
            "schedule": [22, 22, 22, 22, 22, 22],
            "coordinates_per_layer": 405_504,
            "step60_validation_ce": CONTROL_STEP60_VALIDATION_CE,
            "dense_cfc_step60_validation_ce": DENSE_CFC_STEP60_VALIDATION_CE,
            "remaining_gap": (
                CONTROL_STEP60_VALIDATION_CE
                - DENSE_CFC_STEP60_VALIDATION_CE
            ),
        },
        "decision_rule": {
            "primary_metric": "fixed-window step-60 validation cross entropy",
            "minimum_remaining_gap_fraction_to_recover": 0.5,
            "pass_validation_ce_maximum": PASS_VALIDATION_CE,
            "pass": (
                "register exactly one full 238-update 2x-coordinate config and "
                "measure its exact-config MFU before training"
            ),
            "failure": (
                "reject additional same-family coordinate expansion as the main "
                "repair and move to a qualitatively different direction family"
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
