#!/usr/bin/env python3
"""Register the selected six-stage c_fc production MFU preflight."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "examples/nanogpt/configs"
ARTIFACTS = CONFIGS / "selection_artifacts"
PARENT = (
    CONFIGS
    / "pro6_mai_v3_124m_fullattn_plus_mlp_cfc_"
    "directedproduct30_29_29_mfu_retry1.json"
)
SELECTION_RESULT = (
    ARTIFACTS
    / "124m_mlp_cfc_directed_product_terminal_capacity_result.json"
)
OUTPUT_CONFIG = (
    CONFIGS
    / "pro6_mai_v3_124m_fullattn_plus_mlp_cfc_"
    "directedproduct22x6_mfu.json"
)
OUTPUT_PLAN = (
    ARTIFACTS / "124m_mlp_cfc_directed_product_22x6_mfu_plan.json"
)

PARENT_SHA256 = (
    "cbdb67234128a8f99b8735ea0964d15159e821fc27f3c82ff44bda4355aae6f2"
)
SELECTION_RESULT_SHA256 = (
    "62836f9c95ae28e129cbbe5b6c9a8450d31f2636f1570c2382cb5c84fb64a59b"
)
DATASET_MANIFEST_SHA256 = (
    "1e1de075c504906a93637bd79450d30da2243797d2e1d3e33f2392d9492ddf8b"
)
PRODUCTION_IMPLEMENTATION_COMMIT = (
    "29d5e90ff419cde1b13fc5541aff79b12ec49f27"
)
SELECTION_RESULT_COMMIT = "2fb83ef164a404a905298111476e17ee279e7c1e"
SCHEDULE = [22, 22, 22, 22, 22, 22]
COORDINATES_PER_LAYER = 405504

REMOTE_ROOT = (
    "/home/pro6000-9980x/MappingNetworks/"
    "latent-weight-lab-cfc-midpoint-replay"
)
PYTHON = "/mnt/ssd-data/orj/MappingNetworks/.venv/bin/python"
OUTPUT_ROOT = (
    "/home/pro6000-9980x/MappingNetworks/outputs/"
    "pro6_mai_v3_mlp_cfc_directed_product_22x6_mfu"
)
CERTIFICATE = f"{OUTPUT_ROOT}/performance_preflight.json"
LOG = f"{OUTPUT_ROOT}/preflight.log"
SCRATCH = f"{OUTPUT_ROOT}/preflight_scratch"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_inputs() -> None:
    if sha256_file(PARENT) != PARENT_SHA256:
        raise RuntimeError("qualified production MFU parent hash drifted")
    if sha256_file(SELECTION_RESULT) != SELECTION_RESULT_SHA256:
        raise RuntimeError("terminal capacity selection result hash drifted")
    result = json.loads(SELECTION_RESULT.read_text())
    if (
        result["classification"]
        != "TERMINAL_COMPOSITIONAL_CAPACITY_PASSES"
        or result["selected_candidate"] != "depth6_total132"
    ):
        raise RuntimeError("selection result does not authorize six-stage MFU")


def make_config() -> dict[str, Any]:
    config = json.loads(PARENT.read_text())
    config.update(
        {
            "block_fht_mlp_cfc_directed_product_schedule": SCHEDULE,
            "candidate_scope": (
                "held-out-selected full-attention plus qualified two-pass "
                "c_proj and terminal-selected six-stage directed sparse c_fc "
                "product with 22 incoming coordinates per stage"
            ),
            "hpo_stage": "directed_product_cfc_22x6_production_mfu_gate",
            "ladder_role": "mlp_cfc_22x6_terminal_capacity_mfu_gate",
            "mfu_measurement_protocol": (
                "direct foreground real CUDA BF16 training-path preflight "
                "with one warmup and eight timed updates; exact Muon target, "
                "all six support/refit stages, family-radius projection, "
                "qualified c_proj update, and folded materialization execute "
                "on every measured update"
            ),
            "mfu_preflight_certificate": CERTIFICATE,
            "out_dir": SCRATCH,
            "parent_selection_result": str(
                SELECTION_RESULT.relative_to(ROOT)
            ),
            "parent_selection_result_sha256": SELECTION_RESULT_SHA256,
            "screen_only_resolution": (
                "one directly polled six-stage production MFU measurement; "
                "scientific training and larger rungs remain unauthorized"
            ),
            "selection_result_commit": SELECTION_RESULT_COMMIT,
        }
    )
    representation = dict(config["directed_product_representation"])
    representation.update(
        {
            "coordinate_fraction_per_cfc": 0.04296875,
            "coordinates_per_layer": COORDINATES_PER_LAYER,
            "incoming_schedule": SCHEDULE,
            "selection": (
                "terminal-selected six-stage per-update task-conditioned "
                "supports with joint per-target ridge refit"
            ),
        }
    )
    config["directed_product_representation"] = representation
    return config


def make_plan(config_sha256: str) -> dict[str, Any]:
    remote_config = f"{REMOTE_ROOT}/{OUTPUT_CONFIG.relative_to(ROOT)}"
    command = [
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
        LOG,
        "--min-fraction",
        "0.2",
        "--warmup-updates",
        "1",
        "--timed-updates",
        "8",
    ]
    return {
        "schema_version": "mai_124m_mlp_cfc_22x6_mfu_plan_v1",
        "created_at": "2026-08-03",
        "status": "registered_after_terminal_capacity_pass_before_mfu",
        "question": (
            "Does the selected six-stage 22x6 directed-product c_fc chart "
            "sustain at least 20 percent measured MFU on the exact production "
            "training path?"
        ),
        "candidate": {
            "config": str(OUTPUT_CONFIG.relative_to(ROOT)),
            "config_sha256": config_sha256,
            "incoming_schedule": SCHEDULE,
            "coordinates_per_layer": COORDINATES_PER_LAYER,
            "coordinate_multiplier_over_30_29_29": 1.5,
            "production_implementation_commit": (
                PRODUCTION_IMPLEMENTATION_COMMIT
            ),
            "selection_result_commit": SELECTION_RESULT_COMMIT,
        },
        "identity": {
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "parent_config": str(PARENT.relative_to(ROOT)),
            "parent_config_sha256": PARENT_SHA256,
            "selection_result": str(SELECTION_RESULT.relative_to(ROOT)),
            "selection_result_sha256": SELECTION_RESULT_SHA256,
            "implementation_source_hashes": json.loads(PARENT.read_text())[
                "implementation_source_hashes"
            ],
        },
        "protocol": {
            "host": "PRO6",
            "gpu": 0,
            "python": PYTHON,
            "working_directory": REMOTE_ROOT,
            "certificate": CERTIFICATE,
            "log": LOG,
            "warmup_updates": 1,
            "timed_updates": 8,
            "minimum_mfu_fraction": 0.2,
            "denominator": (
                "same-host empirical BF16 8192-square tensor-core GEMM peak"
            ),
            "execution": "direct foreground polling through terminal exit",
            "watchdog": False,
            "callback": False,
            "heartbeat": False,
            "queue_worker": False,
            "command": command,
        },
        "decision_rule": {
            "pass": (
                "native extension guard passes, exit zero, all losses and "
                "timings are finite, all eight timed updates complete, and "
                "MFU is at least 0.20"
            ),
            "reject": (
                "MFU below 0.20, native extension failure, nonfinite path, "
                "incomplete timing, identity mismatch, or nonzero exit"
            ),
            "threshold_changes_after_measurement": False,
        },
        "authorization": {
            "scope": "exactly one directly polled six-stage MFU preflight",
            "scientific_training_authorized": False,
            "larger_rung_authorized": False,
            "automatic_rerun_authorized": False,
            "pass_authorizes": (
                "a separately preregistered 124M/0.5TPP scientific run only"
            ),
        },
    }


def main() -> None:
    validate_inputs()
    config = make_config()
    OUTPUT_CONFIG.write_text(
        json.dumps(config, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    config_sha256 = sha256_file(OUTPUT_CONFIG)
    OUTPUT_PLAN.write_text(
        json.dumps(
            make_plan(config_sha256),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    print(f"config={OUTPUT_CONFIG} sha256={config_sha256}")
    print(f"plan={OUTPUT_PLAN} sha256={sha256_file(OUTPUT_PLAN)}")


if __name__ == "__main__":
    main()
