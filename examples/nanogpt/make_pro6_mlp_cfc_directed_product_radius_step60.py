#!/usr/bin/env python3
"""Preregister the 22x6 directed-product c_fc step-60 radius bracket."""

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
    "directedproduct22x6_0p5tpp.json"
)
CONTROL_RESULT = (
    ARTIFACTS
    / "124m_mlp_cfc_directed_product_22x6_0p5tpp_result.json"
)
OUTPUT_PLAN = (
    ARTIFACTS
    / "124m_mlp_cfc_directed_product_22x6_radius_step60_plan.json"
)

BASE_SHA256 = (
    "0405340d2ea327331f983662c2da3db48c803271cec4861779948884abc1d21f"
)
CONTROL_RESULT_SHA256 = (
    "9758b4a4bd9ef2929f3a052fed6b2565fd1b3f2e15f3c664ee0ec1563e23c2c9"
)
DATASET_MANIFEST_SHA256 = (
    "1e1de075c504906a93637bd79450d30da2243797d2e1d3e33f2392d9492ddf8b"
)
FIXED_EVAL_INDICES_SHA256 = (
    "5ca31b59768e43de808ad5e206ed152a4a0a3515ad68d29a0b2338c4db140747"
)
CONTROL_RADIUS = 0.6589686140591383
CONTROL_STEP60_VALIDATION_CE = 6.3829
DENSE_CFC_STEP60_VALIDATION_CE = 6.3141
PASS_VALIDATION_CE = 6.3485
MAX_ITERS = 60
LR_DECAY_ITERS = 238
PLANNED_TOKENS = 15728640
PLANNED_TPP = 0.126

REMOTE_WORKTREE = (
    "/home/pro6000-9980x/MappingNetworks/"
    "latent-weight-lab-cfc-midpoint-replay"
)
PYTHON = "/mnt/ssd-data/orj/MappingNetworks/.venv/bin/python"
OUTPUT_BASE = (
    "/home/pro6000-9980x/MappingNetworks/outputs/"
    "pro6_mai_v3_mlp_cfc_directed_product_22x6_radius_step60"
)
ARMS = {
    "radius0p82": {
        "radius": 0.82,
        "config": (
            CONFIGS
            / "pro6_mai_v3_124m_fullattn_plus_mlp_cfc_"
            "directedproduct22x6_radius0p82_step60.json"
        ),
    },
    "radius1p00": {
        "radius": 1.0,
        "config": (
            CONFIGS
            / "pro6_mai_v3_124m_fullattn_plus_mlp_cfc_"
            "directedproduct22x6_radius1p00_step60.json"
        ),
    },
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def validate_inputs() -> None:
    if sha256_file(BASE) != BASE_SHA256:
        raise RuntimeError("six-stage base config hash drifted")
    if sha256_file(CONTROL_RESULT) != CONTROL_RESULT_SHA256:
        raise RuntimeError("six-stage control result hash drifted")
    result = json.loads(CONTROL_RESULT.read_text())
    if result.get("classification") != "STABLE_DIRECTIONAL_ONLY_TERMINAL_CE":
        raise RuntimeError("control result does not authorize radius diagnosis")
    if result["loss"]["fixed_evaluations"][1]["validation"] != (
        CONTROL_STEP60_VALIDATION_CE
    ):
        raise RuntimeError("control step-60 metric drifted")
    base = json.loads(BASE.read_text())
    if base["data_manifest_sha256"] != DATASET_MANIFEST_SHA256:
        raise RuntimeError("dataset manifest identity drifted")
    for relative, digest in base["implementation_source_hashes"].items():
        if sha256_file(ROOT / relative) != digest:
            raise RuntimeError(f"production source hash drifted: {relative}")


def make_config(arm: str) -> dict[str, Any]:
    spec = ARMS[arm]
    radius = float(spec["radius"])
    output_root = f"{OUTPUT_BASE}/{arm}"
    run_name = f"pro6_mai_v3_124m_cfc_directed_product_22x6_{arm}_step60"
    config = json.loads(BASE.read_text())
    config.update(
        {
            "block_fht_mlp_cfc_directed_product_family_radius_ratio": radius,
            "candidate_scope": (
                "six-stage directed-product c_fc step-60 causal radius "
                f"screen at family radius {radius}; identical initialization, "
                "data order, evaluation indices, and 238-update LR schedule"
            ),
            "hpo_stage": "directed_product_cfc_22x6_radius_step60_screen",
            "ladder_role": f"mlp_cfc_radius_step60_{arm}",
            "lr_decay_iters": LR_DECAY_ITERS,
            "max_iters": MAX_ITERS,
            "mfu_preflight_certificate": (
                f"{output_root}/performance_preflight.json"
            ),
            "monitoring_policy": (
                "short 60-update run is directly foreground-polled; no "
                "watchdog, callback, queue worker, or heartbeat"
            ),
            "out_dir": f"{output_root}/{run_name}",
            "parent_selection_result": str(CONTROL_RESULT.relative_to(ROOT)),
            "parent_selection_result_sha256": CONTROL_RESULT_SHA256,
            "planned_tokens": PLANNED_TOKENS,
            "planned_tpp": PLANNED_TPP,
            "preregistered_decision_rule": {
                "primary_metric": (
                    "fixed-window validation cross entropy at update 60"
                ),
                "control_radius": CONTROL_RADIUS,
                "control_validation_ce": CONTROL_STEP60_VALIDATION_CE,
                "dense_cfc_validation_ce": DENSE_CFC_STEP60_VALIDATION_CE,
                "minimum_gap_fraction_to_recover": 0.5,
                "pass_validation_ce_maximum": PASS_VALIDATION_CE,
                "pass": (
                    "finite complete step-60 validation CE <= 6.3485 with "
                    "exact initialization, data, evaluation, and source identity"
                ),
                "reject": (
                    "validation CE > 6.3485, instability, incomplete step-60 "
                    "evaluation, or identity mismatch"
                ),
            },
            "run_metadata_path": f"{output_root}/prelaunch_run_metadata.json",
            "screen_only_resolution": (
                "this arm authorizes no full run by itself; the frozen bracket "
                "selects at most one radius for separate full-config registration"
            ),
        }
    )
    representation = dict(config["directed_product_representation"])
    representation.update(
        {
            "family_radius_ratio": radius,
            "family_radius_calibration": (
                "causal step-60 bracket; structure, supports, refit, "
                "initialization, data order, and LR schedule held fixed"
            ),
        }
    )
    config["directed_product_representation"] = representation
    return config


def arm_plan(arm: str, config_sha256: str) -> dict[str, Any]:
    spec = ARMS[arm]
    config_path = spec["config"]
    assert isinstance(config_path, Path)
    remote_config = (
        f"{REMOTE_WORKTREE}/examples/nanogpt/configs/{config_path.name}"
    )
    output_root = f"{OUTPUT_BASE}/{arm}"
    return {
        "radius": spec["radius"],
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": config_sha256,
        "max_iters": MAX_ITERS,
        "lr_decay_iters": LR_DECAY_ITERS,
        "mfu_certificate": f"{output_root}/performance_preflight.json",
        "mfu_log": f"{output_root}/performance_preflight.log",
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
            f"{output_root}/performance_preflight.json",
            "--log-output",
            f"{output_root}/performance_preflight.log",
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
    }


def make_plan(config_hashes: dict[str, str]) -> dict[str, Any]:
    base = json.loads(BASE.read_text())
    return {
        "schema_version": "mai_124m_mlp_cfc_22x6_radius_step60_plan_v1",
        "created_at": "2026-08-03",
        "status": "registered_before_exact_config_mfu_and_training",
        "question": (
            "Is the six-stage online miss caused by the fixed 0.6589686 "
            "c_fc family radius understepping useful early Muon directions?"
        ),
        "identity": {
            "base_config": str(BASE.relative_to(ROOT)),
            "base_config_sha256": BASE_SHA256,
            "control_result": str(CONTROL_RESULT.relative_to(ROOT)),
            "control_result_sha256": CONTROL_RESULT_SHA256,
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "fixed_eval_indices_sha256": FIXED_EVAL_INDICES_SHA256,
            "implementation_source_hashes": base[
                "implementation_source_hashes"
            ],
        },
        "control": {
            "radius": CONTROL_RADIUS,
            "step60_validation_ce": CONTROL_STEP60_VALIDATION_CE,
            "dense_cfc_step60_validation_ce": DENSE_CFC_STEP60_VALIDATION_CE,
            "gap_to_dense_cfc": 0.0688,
        },
        "arms": {
            arm: arm_plan(arm, config_hashes[arm]) for arm in ARMS
        },
        "decision_rule": {
            "primary_metric": "fixed-window step-60 validation cross entropy",
            "minimum_gap_fraction_to_recover": 0.5,
            "pass_validation_ce_maximum": PASS_VALIDATION_CE,
            "selection": (
                "select the lowest-CE passing arm; if arm CEs differ by less "
                "than 0.005, select radius 0.82 as the smaller trust radius"
            ),
            "failure": (
                "if neither arm reaches CE <= 6.3485, reject scalar-radius "
                "recalibration as the main repair"
            ),
            "threshold_changes_after_measurement": False,
        },
        "protocol": {
            "host": "PRO6",
            "gpu": 0,
            "working_directory": REMOTE_WORKTREE,
            "python": PYTHON,
            "identical_model_seed": base["model_seed"],
            "identical_train_seed": base["train_data_seed"],
            "identical_lr_decay_iters": LR_DECAY_ITERS,
            "max_iters": MAX_ITERS,
            "exact_config_mfu_minimum": 0.2,
            "execution": "sequential direct foreground polling",
            "watchdog": False,
            "callback": False,
            "heartbeat": False,
            "queue_worker": False,
        },
        "authorization": {
            "scientific_arms_authorized_after_each_exact_mfu_pass": list(ARMS),
            "automatic_rerun_authorized": False,
            "full_238_update_run_authorized": False,
            "larger_rung_authorized": False,
            "additional_structure_authorized": False,
        },
    }


def main() -> None:
    validate_inputs()
    config_hashes: dict[str, str] = {}
    for arm, spec in ARMS.items():
        config = make_config(arm)
        config_path = spec["config"]
        assert isinstance(config_path, Path)
        config_path.write_bytes(json_bytes(config))
        config_hashes[arm] = sha256_file(config_path)
        print(f"config={config_path} sha256={config_hashes[arm]}")
    OUTPUT_PLAN.write_bytes(json_bytes(make_plan(config_hashes)))
    print(f"plan={OUTPUT_PLAN} sha256={sha256_file(OUTPUT_PLAN)}")


if __name__ == "__main__":
    main()
