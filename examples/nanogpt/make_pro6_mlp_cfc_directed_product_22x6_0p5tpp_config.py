#!/usr/bin/env python3
"""Preregister the six-stage c_fc 124M/0.5TPP scientific run."""

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
    "directedproduct30_29_29_0p5tpp.json"
)
CAPACITY_RESULT = (
    ARTIFACTS
    / "124m_mlp_cfc_directed_product_terminal_capacity_result.json"
)
MFU_RESULT = (
    ARTIFACTS / "124m_mlp_cfc_directed_product_22x6_mfu_result.json"
)
OUTPUT_CONFIG = (
    CONFIGS
    / "pro6_mai_v3_124m_fullattn_plus_mlp_cfc_"
    "directedproduct22x6_0p5tpp.json"
)
OUTPUT_PLAN = (
    ARTIFACTS / "124m_mlp_cfc_directed_product_22x6_0p5tpp_plan.json"
)

PARENT_SHA256 = (
    "80c3cd31d1494799c19c1083504231819caefda33f917d6e48fa96c73257ed7b"
)
CAPACITY_RESULT_SHA256 = (
    "62836f9c95ae28e129cbbe5b6c9a8450d31f2636f1570c2382cb5c84fb64a59b"
)
MFU_RESULT_SHA256 = (
    "5fcb814f841754c7c4a8a1111353d23c9a262062d6b0f52f6b205075ff2919fd"
)
DATASET_MANIFEST_SHA256 = (
    "1e1de075c504906a93637bd79450d30da2243797d2e1d3e33f2392d9492ddf8b"
)
FIXED_EVAL_SPEC_SHA256 = (
    "7cb7f4a2e14eca6229bcca0bde184ab21acaab01cc041cf39d0c87837b90cb52"
)
PRODUCTION_IMPLEMENTATION_COMMIT = (
    "29d5e90ff419cde1b13fc5541aff79b12ec49f27"
)
SELECTION_RESULT_COMMIT = "2fb83ef164a404a905298111476e17ee279e7c1e"
MFU_RESULT_COMMIT = "7751ea42297e31acf61d42a943fec94a3ed089ce"
SCHEDULE = [22, 22, 22, 22, 22, 22]
COORDINATES_PER_LAYER = 405504

REMOTE_WORKTREE = (
    "/home/pro6000-9980x/MappingNetworks/"
    "latent-weight-lab-cfc-midpoint-replay"
)
PYTHON = "/mnt/ssd-data/orj/MappingNetworks/.venv/bin/python"
OUTPUT_ROOT = (
    "/home/pro6000-9980x/MappingNetworks/outputs/"
    "pro6_mai_v3_mlp_cfc_directed_product_22x6_0p5tpp"
)
RUN_DIR = f"{OUTPUT_ROOT}/pro6_mai_v3_124m_cfc_directed_product_22x6_0p5tpp"
LOG = f"{OUTPUT_ROOT}/train.log"
RUN_METADATA = f"{OUTPUT_ROOT}/prelaunch_run_metadata.json"
ENGINEERING_CERTIFICATE = (
    "/home/pro6000-9980x/MappingNetworks/outputs/"
    "pro6_mai_v3_mlp_cfc_directed_product_22x6_mfu/"
    "performance_preflight.json"
)
ENGINEERING_CERTIFICATE_SHA256 = (
    "267f7c4dd7009ba6958c7f84f8d5028f375fa15bdca06149c3f3591b5090ee17"
)
CERTIFICATE = f"{OUTPUT_ROOT}/performance_preflight.json"
PREFLIGHT_LOG = f"{OUTPUT_ROOT}/performance_preflight.log"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_inputs() -> None:
    expected = {
        PARENT: PARENT_SHA256,
        CAPACITY_RESULT: CAPACITY_RESULT_SHA256,
        MFU_RESULT: MFU_RESULT_SHA256,
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise RuntimeError(f"registered input hash drifted: {path}")
    capacity = json.loads(CAPACITY_RESULT.read_text())
    if capacity.get("selected_candidate") != "depth6_total132":
        raise RuntimeError("capacity result does not select six-stage chart")
    mfu = json.loads(MFU_RESULT.read_text())
    if not mfu.get("passed") or mfu["measurement"]["mfu_fraction"] < 0.2:
        raise RuntimeError("engineering MFU result does not authorize config")
    source_hashes = json.loads(PARENT.read_text())[
        "implementation_source_hashes"
    ]
    for relative, digest in source_hashes.items():
        if sha256_file(ROOT / relative) != digest:
            raise RuntimeError(f"production source hash drifted: {relative}")


def make_config() -> dict[str, Any]:
    config = json.loads(PARENT.read_text())
    config.update(
        {
            "block_fht_mlp_cfc_directed_product_schedule": SCHEDULE,
            "candidate_scope": (
                "held-out-selected full-attention replacement plus qualified "
                "two-pass c_proj and terminal-selected six-stage 22x6 "
                "directed-product c_fc; second scientific 124M/0.5TPP test"
            ),
            "engineering_mfu_preflight_certificate": ENGINEERING_CERTIFICATE,
            "engineering_mfu_preflight_certificate_sha256": (
                ENGINEERING_CERTIFICATE_SHA256
            ),
            "hpo_stage": "directed_product_cfc_22x6_smallest_rung_validation",
            "ladder_role": "mlp_full_replacement_22x6_smallest_rung",
            "mfu_preflight_certificate": CERTIFICATE,
            "mfu_preflight_result": str(MFU_RESULT.relative_to(ROOT)),
            "mfu_preflight_result_sha256": MFU_RESULT_SHA256,
            "out_dir": RUN_DIR,
            "parent_selection_result": str(CAPACITY_RESULT.relative_to(ROOT)),
            "parent_selection_result_sha256": CAPACITY_RESULT_SHA256,
            "preregistered_decision_rule": {
                "primary_metric": (
                    "terminal fixed-window validation cross entropy at update 238"
                ),
                "attention_only_validation_ce": 5.4918,
                "accepted_attention_gap": 0.1,
                "success_ce_maximum": 5.5918,
                "qualified_cproj_only_validation_ce": 5.592058181762695,
                "rejected_three_stage_validation_ce": 5.658790111541748,
                "success": (
                    "stable terminal validation CE <= 5.5918, closing full "
                    "replacement to at most +0.10 versus attention-only"
                ),
                "directional_only": (
                    "stable terminal validation CE > 5.5918 and < "
                    "5.658790111541748"
                ),
                "reject": (
                    "terminal validation CE >= 5.658790111541748, "
                    "instability, incomplete run, or provenance failure"
                ),
            },
            "run_metadata_path": RUN_METADATA,
            "screen_only_resolution": (
                "only this preregistered six-stage 124M/0.5TPP run is "
                "authorized after its exact config passes MFU; no larger rung"
            ),
            "selection_result_commit": SELECTION_RESULT_COMMIT,
            "mfu_result_commit": MFU_RESULT_COMMIT,
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
    remote_config = (
        f"{REMOTE_WORKTREE}/examples/nanogpt/configs/{OUTPUT_CONFIG.name}"
    )
    preflight_command = [
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
    ]
    train_command = [
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
    ]
    return {
        "schema_version": "mai_124m_mlp_cfc_22x6_0p5tpp_plan_v1",
        "created_at": "2026-08-03",
        "status": "registered_before_exact_config_mfu_and_training",
        "question": (
            "Does the terminal-selected six-stage c_fc chart close full MLP "
            "replacement to within +0.10 CE of attention-only at 124M/0.5TPP?"
        ),
        "candidate": {
            "config": str(OUTPUT_CONFIG.relative_to(ROOT)),
            "config_sha256": config_sha256,
            "incoming_schedule": SCHEDULE,
            "coordinates_per_layer": COORDINATES_PER_LAYER,
            "max_iters": 238,
            "planned_tpp": 0.5,
            "family_radius_ratio": 0.6589686140591383,
            "production_implementation_commit": (
                PRODUCTION_IMPLEMENTATION_COMMIT
            ),
            "selection_result_commit": SELECTION_RESULT_COMMIT,
            "mfu_result_commit": MFU_RESULT_COMMIT,
        },
        "identity": {
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "fixed_eval_spec_sha256": FIXED_EVAL_SPEC_SHA256,
            "capacity_result": str(CAPACITY_RESULT.relative_to(ROOT)),
            "capacity_result_sha256": CAPACITY_RESULT_SHA256,
            "engineering_mfu_result": str(MFU_RESULT.relative_to(ROOT)),
            "engineering_mfu_result_sha256": MFU_RESULT_SHA256,
            "implementation_source_hashes": json.loads(PARENT.read_text())[
                "implementation_source_hashes"
            ],
        },
        "controls": {
            "attention_only_validation_ce": 5.4918,
            "qualified_cproj_only_validation_ce": 5.592058181762695,
            "rejected_three_stage_validation_ce": 5.658790111541748,
        },
        "decision_rule": {
            "success": "finite complete run with terminal validation CE <= 5.5918",
            "directional_only": (
                "terminal validation CE > 5.5918 and < 5.658790111541748"
            ),
            "reject": (
                "terminal validation CE >= 5.658790111541748, nonfinite "
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
            "log": LOG,
            "prelaunch_run_metadata": RUN_METADATA,
            "exact_config_certificate": CERTIFICATE,
            "exact_config_preflight_log": PREFLIGHT_LOG,
            "exact_config_preflight_command": preflight_command,
            "training_command": train_command,
            "execution": "direct foreground polling through terminal exit",
            "watchdog": False,
            "callback": False,
            "heartbeat": False,
            "queue_worker": False,
        },
        "authorization": {
            "scope": "exactly one 124M/0.5TPP six-stage scientific run",
            "training_requires_exact_config_mfu_pass": True,
            "automatic_rerun_authorized": False,
            "larger_rung_authorized": False,
            "additional_structure_authorized": False,
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
