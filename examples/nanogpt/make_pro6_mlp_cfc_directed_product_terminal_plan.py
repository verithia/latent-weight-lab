#!/usr/bin/env python3
"""Preregister the terminal c_fc radius-versus-direction discriminator."""

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
    ROOT / "examples/nanogpt/analyze_mlp_cfc_directed_product_terminal.py"
)
TRAINING_RESULT = (
    ROOT
    / "examples/nanogpt/configs/selection_artifacts/"
    "124m_mlp_cfc_directed_product_0p5tpp_result.json"
)
OUTPUT = (
    ROOT
    / "examples/nanogpt/configs/selection_artifacts/"
    "124m_mlp_cfc_directed_product_terminal_plan_v2.json"
)
FAILED_PLAN = (
    ROOT
    / "examples/nanogpt/configs/selection_artifacts/"
    "124m_mlp_cfc_directed_product_terminal_plan.json"
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
    "pro6_mai_v3_mlp_cfc_directed_product_terminal_diag_v2"
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
    "38d79db3b5aedb4484811abe10b3983a9d0255c3b78da99c73764b52225b4e15"
)
TRAINING_RESULT_SHA256 = (
    "411e7081502bdb2b22f157d658733955fe7ea1ec9d8e565f73ffd0f877f2f17b"
)
IMPLEMENTATION_COMMIT = "5b25516e6c2e418a3086c8f821229bd0e4f6cc81"
FAILED_PLAN_SHA256 = (
    "aa2e64ac2e310e61709f84bcf131b241f43a6467ca7ec5306350519c50ee3ae9"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_inputs() -> None:
    expected = {
        CONFIG: CONFIG_SHA256,
        ENTRYPOINT: ENTRYPOINT_SHA256,
        TRAINING_RESULT: TRAINING_RESULT_SHA256,
        FAILED_PLAN: FAILED_PLAN_SHA256,
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
        "schema_version": "mai_124m_mlp_cfc_terminal_discriminator_plan_v2",
        "created_at": "2026-08-03",
        "status": "registered_after_pre_evaluation_adapter_failure",
        "question": (
            "At the rejected 124M terminal checkpoint, is the next c_fc step "
            "limited primarily by the directed-product trust radius or by its "
            "unrepresented dense residual direction?"
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
            "training_result": str(TRAINING_RESULT.relative_to(ROOT)),
            "training_result_sha256": TRAINING_RESULT_SHA256,
            "superseded_plan": str(FAILED_PLAN.relative_to(ROOT)),
            "superseded_plan_sha256": FAILED_PLAN_SHA256,
            "superseded_execution_observation": (
                "failed before candidate evaluation because the V1 entrypoint "
                "used a functional-shear-only weight adapter; zero checkpoint "
                "updates and zero candidate CE measurements occurred"
            ),
        },
        "protocol": {
            "parameter_updates_to_checkpoint": 0,
            "gradient_split": "train",
            "gradient_seed": 2026080301,
            "gradient_batch_size": 4,
            "gradient_accumulation_steps": 8,
            "evaluation_split": "val",
            "evaluation_batch_size": 4,
            "validation_batches_per_window": 8,
            "validation_seeds": [2026080311, 2026080329],
            "registered_radius_ratio": 0.6589686140591383,
            "radius_ratios": [0.5, 0.6589686140591383, 0.75, 1.0],
            "residual_fractions": [0.25, 0.5, 1.0],
            "candidate_controls": [
                "unchanged_terminal_checkpoint",
                "registered_product_update",
                "same_dense_direction_at_registered_radius",
                "full_dense_direction",
                "duplicate_registered_radius_idempotence_control",
            ],
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
            "confidence_z": 2.5758293035489004,
            "paired_unit": "same validation window and batch index",
            "residual_direction_limited": (
                "same-radius dense direction has a strictly negative 99% "
                "paired CE-difference upper bound while no alternative "
                "product radius does"
            ),
            "trust_radius_limited": (
                "at least one alternative product radius has a strictly "
                "negative 99% paired CE-difference upper bound while the "
                "same-radius dense direction does not"
            ),
            "mixed": (
                "both same-radius dense direction and an alternative product "
                "radius have strictly negative 99% paired CE-difference "
                "upper bounds"
            ),
            "not_discriminating": (
                "neither comparison is reliably better on the frozen windows"
            ),
            "threshold_changes_after_measurement": False,
        },
        "authorization": {
            "training_authorized": False,
            "checkpoint_mutation_authorized": False,
            "larger_rung_authorized": False,
            "automatic_rerun_authorized": False,
            "follow_up": (
                "freeze the diagnostic result and use it to preregister one "
                "structural smallest-rung repair; do not train from this plan"
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
