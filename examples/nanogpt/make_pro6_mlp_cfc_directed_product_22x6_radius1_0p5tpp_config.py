#!/usr/bin/env python3
"""Preregister the selected full-radius 22x6 c_fc 124M/0.5TPP run."""

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
SCREEN_RESULT = (
    ARTIFACTS
    / "124m_mlp_cfc_directed_product_22x6_radius_step60_result.json"
)
OUTPUT_CONFIG = (
    CONFIGS
    / "pro6_mai_v3_124m_fullattn_plus_mlp_cfc_"
    "directedproduct22x6_radius1p00_0p5tpp.json"
)
OUTPUT_PLAN = (
    ARTIFACTS
    / "124m_mlp_cfc_directed_product_22x6_radius1p00_0p5tpp_plan.json"
)

BASE_SHA256 = (
    "0405340d2ea327331f983662c2da3db48c803271cec4861779948884abc1d21f"
)
SCREEN_RESULT_SHA256 = (
    "2a04dc097413d427dd806943fc86299f7f4dcd0f54bfdd3a4e69a025cbe7892c"
)
DATASET_MANIFEST_SHA256 = (
    "1e1de075c504906a93637bd79450d30da2243797d2e1d3e33f2392d9492ddf8b"
)
FIXED_EVAL_INDICES_SHA256 = (
    "5ca31b59768e43de808ad5e206ed152a4a0a3515ad68d29a0b2338c4db140747"
)
RADIUS_SCREEN_RESULT_COMMIT = "d2d7dbd09086074d78ad9e36665b9aa73c37a923"
RADIUS = 1.0
MAX_ITERS = 238
SUCCESS_CE = 5.5918
CONTROL_CE = 5.649507999420166

REMOTE_WORKTREE = (
    "/home/pro6000-9980x/MappingNetworks/"
    "latent-weight-lab-cfc-midpoint-replay"
)
PYTHON = "/mnt/ssd-data/orj/MappingNetworks/.venv/bin/python"
OUTPUT_ROOT = (
    "/home/pro6000-9980x/MappingNetworks/outputs/"
    "pro6_mai_v3_mlp_cfc_directed_product_22x6_radius1p00_0p5tpp"
)
RUN_DIR = (
    f"{OUTPUT_ROOT}/"
    "pro6_mai_v3_124m_cfc_directed_product_22x6_radius1p00_0p5tpp"
)
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
        raise RuntimeError("six-stage base config hash drifted")
    if sha256_file(SCREEN_RESULT) != SCREEN_RESULT_SHA256:
        raise RuntimeError("radius-screen result hash drifted")
    result = json.loads(SCREEN_RESULT.read_text())
    if result.get("classification") != "FULL_RADIUS_PASSES_EARLY_CAUSAL_GATE":
        raise RuntimeError("radius screen does not authorize full config")
    if result["selection"]["selected_radius"] != RADIUS:
        raise RuntimeError("radius screen selected a different arm")
    base = json.loads(BASE.read_text())
    if base["data_manifest_sha256"] != DATASET_MANIFEST_SHA256:
        raise RuntimeError("dataset manifest identity drifted")
    for relative, digest in base["implementation_source_hashes"].items():
        if sha256_file(ROOT / relative) != digest:
            raise RuntimeError(f"production source hash drifted: {relative}")


def make_config() -> dict[str, Any]:
    config = json.loads(BASE.read_text())
    config.update(
        {
            "block_fht_mlp_cfc_directed_product_family_radius_ratio": RADIUS,
            "candidate_scope": (
                "held-out-selected full-attention replacement plus qualified "
                "two-pass c_proj and six-stage directed-product c_fc at the "
                "step-60-selected full dense-family radius"
            ),
            "hpo_stage": (
                "directed_product_cfc_22x6_radius1p00_smallest_rung_validation"
            ),
            "ladder_role": (
                "mlp_full_replacement_22x6_radius1p00_smallest_rung"
            ),
            "max_iters": MAX_ITERS,
            "lr_decay_iters": MAX_ITERS,
            "mfu_preflight_certificate": CERTIFICATE,
            "monitoring_policy": (
                "short 238-update run is directly foreground-polled; no "
                "watchdog, callback, queue worker, or heartbeat"
            ),
            "out_dir": RUN_DIR,
            "parent_selection_result": str(SCREEN_RESULT.relative_to(ROOT)),
            "parent_selection_result_sha256": SCREEN_RESULT_SHA256,
            "preregistered_decision_rule": {
                "primary_metric": (
                    "terminal fixed-window validation cross entropy at update 238"
                ),
                "attention_only_validation_ce": 5.4918,
                "accepted_attention_gap": 0.1,
                "success_ce_maximum": SUCCESS_CE,
                "qualified_cproj_only_validation_ce": 5.592058181762695,
                "registered_radius_control_validation_ce": CONTROL_CE,
                "success": (
                    "stable terminal validation CE <= 5.5918, closing full "
                    "replacement to at most +0.10 versus attention-only"
                ),
                "directional_only": (
                    "stable terminal validation CE > 5.5918 and < "
                    "5.649507999420166"
                ),
                "reject": (
                    "terminal validation CE >= 5.649507999420166, "
                    "instability, incomplete run, or provenance failure"
                ),
            },
            "radius_screen_result_commit": RADIUS_SCREEN_RESULT_COMMIT,
            "run_metadata_path": RUN_METADATA,
            "screen_only_resolution": (
                "only this selected full-radius 124M/0.5TPP run is authorized "
                "after its exact config passes MFU; no rerun or larger rung"
            ),
        }
    )
    representation = dict(config["directed_product_representation"])
    representation.update(
        {
            "family_radius_ratio": RADIUS,
            "family_radius_calibration": (
                "full dense-family norm selected by the preregistered paired "
                "step-60 radius screen"
            ),
            "selection": (
                "six-stage supports and refit held fixed; full radius selected "
                "after recovering at least half the early dense-c_fc gap"
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
        "schema_version": (
            "mai_124m_mlp_cfc_22x6_radius1p00_0p5tpp_plan_v1"
        ),
        "created_at": "2026-08-03",
        "status": "registered_before_exact_config_mfu_and_training",
        "question": (
            "Does correcting the six-stage c_fc family understep close full "
            "MLP replacement to within +0.10 CE of attention-only?"
        ),
        "candidate": {
            "config": str(OUTPUT_CONFIG.relative_to(ROOT)),
            "config_sha256": config_sha256,
            "radius": RADIUS,
            "incoming_schedule": [22, 22, 22, 22, 22, 22],
            "coordinates_per_layer": 405504,
            "max_iters": MAX_ITERS,
            "planned_tpp": 0.5,
            "radius_screen_result_commit": RADIUS_SCREEN_RESULT_COMMIT,
        },
        "identity": {
            "base_config": str(BASE.relative_to(ROOT)),
            "base_config_sha256": BASE_SHA256,
            "radius_screen_result": str(SCREEN_RESULT.relative_to(ROOT)),
            "radius_screen_result_sha256": SCREEN_RESULT_SHA256,
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "fixed_eval_indices_sha256": FIXED_EVAL_INDICES_SHA256,
            "implementation_source_hashes": base[
                "implementation_source_hashes"
            ],
        },
        "decision_rule": {
            "success": "finite complete terminal validation CE <= 5.5918",
            "directional_only": (
                "terminal validation CE > 5.5918 and < 5.649507999420166"
            ),
            "reject": (
                "terminal validation CE >= 5.649507999420166, nonfinite "
                "path, incomplete evaluation, or identity mismatch"
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
            "scope": "exactly one 124M/0.5TPP six-stage full-radius run",
            "training_requires_exact_config_mfu_pass": True,
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
