#!/usr/bin/env python3
"""Freeze the selected 30+29+29 c_fc production MFU preflight."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "examples/nanogpt/configs"
ARTIFACTS = CONFIGS / "selection_artifacts"
PARENT_CONFIG = (
    CONFIGS
    / "pro6_mai_v3_124m_fullattn_plus_mlp_functionmix50_mfu.json"
)
OUTPUT_CONFIG = (
    CONFIGS
    / "pro6_mai_v3_124m_fullattn_plus_mlp_cfc_"
    "directedproduct30_29_29_mfu.json"
)
OUTPUT_PLAN = (
    ARTIFACTS / "124m_mlp_cfc_directed_product_mfu_plan.json"
)
IMPLEMENTATION_COMMIT = "6804c654919518b9e0c0cd63ee8eb3061d614368"
PARENT_CONFIG_SHA256 = (
    "ed987e69c6a3a751d385f0235b5f8f78dfb47b235595fda519bda9e532827417"
)
SELECTION_RESULT = (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_mlp_cfc_multistage_directed_result.json"
)
SELECTION_RESULT_SHA256 = (
    "b3f59359eb9240c29ba83f5b7dcbf0a546e752bab10a625d5e7120f3c8b0b3b7"
)
DATASET_MANIFEST_SHA256 = (
    "1e1de075c504906a93637bd79450d30da2243797d2e1d3e33f2392d9492ddf8b"
)
SOURCE_HASHES = {
    "examples/nanogpt/model.py": (
        "07602c5045077a848ab1e0a176431dde6b15c07be08b271a56276e91ad13ceae"
    ),
    "examples/nanogpt/muon_matched_givens.py": (
        "b2973183268673859272837c80a13a8ddeec6b2bd5a43cef1703bcc9a039c641"
    ),
    "examples/nanogpt/train.py": (
        "c6b8e08221ba4bb0dc76bd20a522e1f079eb60dce3363ada54171291788eddd5"
    ),
    "examples/nanogpt/test_muon_directed_product.py": (
        "5ee110e99d61cff37407f75da9496ee649d22d7aec1acef45f91d8983d6aea89"
    ),
    "examples/nanogpt/mfu_preflight.py": (
        "b9454210c6f6aec59aa39da7a9f1b36f111eba5f13127c39fcf47758a94179a3"
    ),
    "examples/nanogpt/muon.py": (
        "532e172d91306d12284507c96aa3176792b33eb657f568512ce278bb5a9874ff"
    ),
    "latent_weight_lab/block_fht.py": (
        "864ba9a79664cba2f830c06b11214538b7817685e1ba990f6e103feefb49b561"
    ),
}
REMOTE_WORKTREE = (
    "/home/pro6000-9980x/MappingNetworks/"
    "latent-weight-lab-cfc-midpoint-replay"
)
OUTPUT_ROOT = (
    "/home/pro6000-9980x/MappingNetworks/outputs/"
    "pro6_mai_v3_mlp_cfc_directed_product_mfu"
)
CERTIFICATE = f"{OUTPUT_ROOT}/performance_preflight.json"
PREFLIGHT_LOG = f"{OUTPUT_ROOT}/preflight.log"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def validate_inputs() -> None:
    if sha256_file(PARENT_CONFIG) != PARENT_CONFIG_SHA256:
        raise RuntimeError("parent MFU config hash drifted")
    if sha256_file(ROOT / SELECTION_RESULT) != SELECTION_RESULT_SHA256:
        raise RuntimeError("three-stage selection result hash drifted")
    for relative, expected in SOURCE_HASHES.items():
        if sha256_file(ROOT / relative) != expected:
            raise RuntimeError(f"runtime source hash drifted: {relative}")


def make_config() -> dict[str, Any]:
    config = json.loads(PARENT_CONFIG.read_text())
    config.update(
        {
            "block_fht_mlp_cfc_functional_shear": False,
            "block_fht_mlp_cfc_directed_product": True,
            "block_fht_mlp_cfc_directed_product_schedule": [30, 29, 29],
            "block_fht_mlp_cfc_directed_product_ridge_ratio": 1e-6,
            "block_fht_mlp_cfc_directed_product_chunk_size": 256,
            "block_fht_mlp_cfc_directed_product_family_radius_ratio": (
                0.6589686140591383
            ),
            "candidate_scope": (
                "held-out-selected full-attention plus qualified two-pass "
                "c_proj and task-selected three-stage 30+29+29 directed "
                "sparse c_fc product; the folded c_fc base is materialized "
                "but not an optimizer-visible dense model parameter"
            ),
            "failed_mfu_preflight": None,
            "hpo_stage": (
                "directed_product_cfc_production_integration_and_mfu_gate"
            ),
            "implementation_commit": IMPLEMENTATION_COMMIT,
            "implementation_source_hashes": SOURCE_HASHES,
            "implementation_test_evidence": {
                "local": "24 passed",
                "pro6": "50 passed, 40 subtests passed",
                "coverage": [
                    "batched equality to preregistered scalar solver",
                    "finite gradient and family-radius enforcement",
                    "exact optimizer resume",
                    "paired-seed initialization",
                    "coordinate accounting and no persistent dense basis",
                    "full checkpoint RNG and envelope regression suites",
                ],
            },
            "directed_product_representation": {
                "incoming_schedule": [30, 29, 29],
                "coordinates_per_layer": 88 * 3072,
                "coordinate_fraction_per_cfc": 88 / 3072,
                "selection": (
                    "per update and stage, sparsify the exact minimum-norm "
                    "full output action and jointly ridge-refit each target "
                    "channel from its selected current source columns"
                ),
                "product": (
                    "each residual stage acts on the source transformed by "
                    "all preceding stages"
                ),
                "ridge_ratio": 1e-6,
                "solver_chunk_size": 256,
                "family_radius_ratio": 0.6589686140591383,
                "family_radius_calibration": (
                    "deployed production c_fc family Frobenius radius divided "
                    "by exact dense scheduled Muon family radius at the "
                    "registered step-120 midpoint"
                ),
                "dense_folded_base_buffer": True,
                "persistent_dense_basis": False,
                "learned_dense_basis": False,
                "dense_residual": False,
                "lora_adapter": False,
                "exact_muon_momentum": True,
                "matching_refresh_updates": 1,
                "scope_limit": (
                    "compresses update coordinates, not materialized c_fc "
                    "checkpoint size or inference FLOPs"
                ),
            },
            "mfu_measurement_protocol": (
                "direct foreground real CUDA BF16 training-path preflight "
                "with one warmup and eight timed updates; the exact Muon "
                "direction, all three batched support/refit stages, global "
                "family-radius projection, qualified c_proj update, and "
                "folded materialization execute on every measured update"
            ),
            "mfu_preflight_certificate": CERTIFICATE,
            "mfu_preflight_required": True,
            "mfu_min_fraction": 0.2,
            "out_dir": f"{OUTPUT_ROOT}/preflight_scratch",
            "parent_selection_result": SELECTION_RESULT,
            "parent_selection_result_sha256": SELECTION_RESULT_SHA256,
            "screen_only_resolution": (
                "only this registered directly polled MFU preflight is "
                "authorized; no scientific training or larger rung is "
                "authorized by the geometry result"
            ),
        }
    )
    return config


def make_plan(config_sha256: str) -> dict[str, Any]:
    remote_config = (
        f"{REMOTE_WORKTREE}/examples/nanogpt/configs/{OUTPUT_CONFIG.name}"
    )
    command = [
        "env",
        "CUDA_VISIBLE_DEVICES=0",
        "/mnt/ssd-data/orj/MappingNetworks/.venv-pro6-diagnostics/bin/python",
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
    return {
        "schema_version": "mai_124m_mlp_cfc_directed_product_mfu_plan_v1",
        "created_at": "2026-08-03",
        "status": "registered_before_real_training_path_mfu_measurement",
        "scientific_question": (
            "Can the held-out-selected exact 30+29+29 directed-product c_fc "
            "optimizer execute the complete 124M training path at or above "
            "the mandatory 20% model-FLOP utilization floor?"
        ),
        "candidate": {
            "config": str(OUTPUT_CONFIG.relative_to(ROOT)),
            "config_sha256": config_sha256,
            "implementation_commit": IMPLEMENTATION_COMMIT,
            "model_tier": "124m",
            "incoming_schedule": [30, 29, 29],
            "coordinates_per_layer": 88 * 3072,
            "family_radius_ratio": 0.6589686140591383,
        },
        "parent_evidence": {
            "result": SELECTION_RESULT,
            "result_sha256": SELECTION_RESULT_SHA256,
            "classification": "THREE_STAGE_DIRECTED_CFC_PASSES",
            "selected_incoming_total_per_target": 88,
            "scope": (
                "authorizes production implementation and this MFU preflight "
                "only, not scientific training"
            ),
        },
        "identity": {
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "implementation_source_hashes": SOURCE_HASHES,
        },
        "protocol": {
            "host": "PRO6",
            "gpu": 0,
            "warmup_updates": 1,
            "timed_updates": 8,
            "minimum_mfu_fraction": 0.2,
            "denominator": (
                "same-host empirical BF16 8192-square tensor-core GEMM peak"
            ),
            "execution": "direct foreground polling through terminal exit",
            "watchdog": False,
            "callback": False,
            "queue_worker": False,
            "heartbeat": False,
            "command": command,
            "certificate": CERTIFICATE,
            "log": PREFLIGHT_LOG,
        },
        "decision_rule": {
            "pass": (
                "exit zero, finite complete certificate bound to the exact "
                "config hash, all eight timed updates present, and measured "
                "MFU >= 0.20"
            ),
            "reject": (
                "MFU < 0.20, nonfinite training path, incomplete timing, "
                "provenance mismatch, or nonzero exit"
            ),
            "threshold_changes_after_measurement": False,
        },
        "authorization": {
            "scope": "exactly one directly polled MFU preflight",
            "scientific_training_authorized": False,
            "larger_rung_authorized": False,
            "additional_structure_authorized": False,
        },
    }


def main() -> None:
    validate_inputs()
    config = make_config()
    config_payload = json_bytes(config)
    config_sha256 = hashlib.sha256(config_payload).hexdigest()
    plan = make_plan(config_sha256)
    OUTPUT_CONFIG.write_bytes(config_payload)
    OUTPUT_PLAN.write_bytes(json_bytes(plan))
    print(
        json.dumps(
            {
                "config": str(OUTPUT_CONFIG),
                "config_sha256": config_sha256,
                "plan": str(OUTPUT_PLAN),
                "plan_sha256": sha256_file(OUTPUT_PLAN),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
