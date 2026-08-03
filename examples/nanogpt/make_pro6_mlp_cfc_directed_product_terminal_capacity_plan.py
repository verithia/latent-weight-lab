#!/usr/bin/env python3
"""Preregister terminal directed-product depth/capacity geometry bracket."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_fullattn_plus_mlp_cfc_"
    "directedproduct30_29_29_0p5tpp.json"
)
ENTRYPOINT = (
    ROOT
    / "examples/nanogpt/"
    "analyze_mlp_cfc_directed_product_terminal_capacity.py"
)
PARENT_RESULT = (
    ROOT
    / "examples/nanogpt/configs/selection_artifacts/"
    "124m_mlp_cfc_directed_product_terminal_result.json"
)
OUTPUT = (
    ROOT
    / "examples/nanogpt/configs/selection_artifacts/"
    "124m_mlp_cfc_directed_product_terminal_capacity_plan.json"
)

REMOTE_ROOT = (
    "/home/pro6000-9980x/MappingNetworks/"
    "latent-weight-lab-cfc-midpoint-replay"
)
PYTHON = "/mnt/ssd-data/orj/MappingNetworks/.venv/bin/python"
DATA_DIR = "/home/pro6000-9980x/MappingNetworks/data/finewebedu_20b"
CHECKPOINT = (
    "/home/pro6000-9980x/MappingNetworks/outputs/"
    "pro6_mai_v3_mlp_cfc_directed_product_0p5tpp/"
    "pro6_mai_v3_124m_cfc_directed_product_0p5tpp/ckpt.pt"
)
REMOTE_CONFIG = f"{REMOTE_ROOT}/{CONFIG.relative_to(ROOT)}"
REMOTE_PLAN = f"{REMOTE_ROOT}/{OUTPUT.relative_to(ROOT)}"
REMOTE_ENTRYPOINT = f"{REMOTE_ROOT}/{ENTRYPOINT.relative_to(ROOT)}"
OUTPUT_DIR = (
    "/home/pro6000-9980x/MappingNetworks/outputs/"
    "pro6_mai_v3_mlp_cfc_directed_product_terminal_capacity"
)

CONFIG_SHA256 = (
    "80c3cd31d1494799c19c1083504231819caefda33f917d6e48fa96c73257ed7b"
)
CHECKPOINT_SHA256 = (
    "aaf00e4b36489fb4eb0d720cc821e401579c3d00c917489ea86a4eb61b278f1a"
)
DATASET_MANIFEST_SHA256 = (
    "1e1de075c504906a93637bd79450d30da2243797d2e1d3e33f2392d9492ddf8b"
)
ENTRYPOINT_SHA256 = (
    "2287ab327fccbfeafce1264ade592a770918419265063c1d6553abdf07a445fc"
)
PARENT_RESULT_SHA256 = (
    "3edbae656215de35a62846c7dfc67d5be12901764ff0177ac9969cecde183748"
)
IMPLEMENTATION_COMMIT = "a87440582d9a26a90ea13d419b0a46e34b36019f"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_inputs() -> None:
    expected = {
        CONFIG: CONFIG_SHA256,
        ENTRYPOINT: ENTRYPOINT_SHA256,
        PARENT_RESULT: PARENT_RESULT_SHA256,
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise RuntimeError(f"registered input hash drifted: {path}")


def make_plan() -> dict[str, Any]:
    command = [
        "env",
        "CUDA_VISIBLE_DEVICES=0",
        "CUDA_HOME=/mnt/ssd-data/orj/MappingNetworks/.cuda-12.8",
        "PYTHONPATH=.",
        PYTHON,
        "-u",
        REMOTE_ENTRYPOINT,
        "--checkpoint",
        CHECKPOINT,
        "--config",
        REMOTE_CONFIG,
        "--data-dir",
        DATA_DIR,
        "--plan",
        REMOTE_PLAN,
        "--output",
        OUTPUT_DIR,
        "--device",
        "cuda",
    ]
    return {
        "schema_version": "mai_124m_mlp_cfc_terminal_capacity_plan_v1",
        "created_at": "2026-08-03",
        "status": "registered_before_terminal_capacity_measurement",
        "question": (
            "Can additional directed-product depth or at most 2x coordinates "
            "materially raise c_fc dense-direction recovery across multiple "
            "fresh terminal gradients?"
        ),
        "identity": {
            "checkpoint": CHECKPOINT,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "checkpoint_next_iter": 238,
            "config": str(CONFIG.relative_to(ROOT)),
            "config_sha256": CONFIG_SHA256,
            "dataset_manifest": f"{DATA_DIR}/manifest.json",
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "entrypoint": str(ENTRYPOINT.relative_to(ROOT)),
            "entrypoint_sha256": ENTRYPOINT_SHA256,
            "implementation_commit": IMPLEMENTATION_COMMIT,
            "parent_result": str(PARENT_RESULT.relative_to(ROOT)),
            "parent_result_sha256": PARENT_RESULT_SHA256,
        },
        "protocol": {
            "parameter_updates_to_checkpoint": 0,
            "gradient_split": "train",
            "gradient_seeds": [2026080401, 2026080407, 2026080413],
            "gradient_batch_size": 4,
            "gradient_accumulation_steps": 8,
            "registered_radius_ratio": 0.6589686140591383,
            "current_candidate": "current_depth3_total88",
            "candidate_schedules": {
                "current_depth3_total88": [30, 29, 29],
                "depth4_total88": [22, 22, 22, 22],
                "depth6_total88": [15, 15, 15, 15, 14, 14],
                "depth4_total132": [33, 33, 33, 33],
                "depth6_total132": [22, 22, 22, 22, 22, 22],
                "depth4_total176": [44, 44, 44, 44],
                "depth6_total176": [30, 30, 29, 29, 29, 29]
            },
            "execution": "direct foreground polling through process exit",
            "host": "PRO6",
            "gpu": 0,
            "watchdog": False,
            "callback": False,
            "heartbeat": False,
            "queue_worker": False,
            "working_directory": REMOTE_ROOT,
            "output_directory": OUTPUT_DIR,
            "command": command,
        },
        "decision_rule": {
            "primary_metric": (
                "minimum family positive-line recovery across all three "
                "fresh terminal gradient windows"
            ),
            "maximum_coordinates_per_layer": 540672,
            "minimum_positive_line_recovery": 0.65,
            "minimum_layer_positive_line_recovery": 0.62,
            "minimum_improvement_over_current": 0.12,
            "maximum_radius_error": 1e-7,
            "selection": (
                "among candidates passing every threshold on every gradient, "
                "select the fewest coordinates, then the highest worst-window "
                "recovery"
            ),
            "reject": (
                "if none passes, reject additional depth/capacity in this "
                "directed sparse product topology"
            ),
            "threshold_changes_after_measurement": False,
        },
        "authorization": {
            "training_authorized": False,
            "checkpoint_mutation_authorized": False,
            "larger_rung_authorized": False,
            "automatic_rerun_authorized": False,
            "pass_authorizes": (
                "production implementation and a separately preregistered "
                "real-training MFU gate only"
            ),
        },
    }


def main() -> None:
    validate_inputs()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(make_plan(), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    print(f"wrote {OUTPUT}")
    print(f"sha256={sha256_file(OUTPUT)}")


if __name__ == "__main__":
    main()
