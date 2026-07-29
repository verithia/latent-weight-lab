"""Generate 12/24-factor one-step product-FHT direction calibrations."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "examples" / "nanogpt" / "configs"
ARTIFACT_DIR = CONFIG_DIR / "selection_artifacts"
PARENT = (
    CONFIG_DIR
    / "y400_mai_v3_124m_fullattn_plus_mlp_cproj_"
    "productfht6_wsmuon_0p5tpp_lr24e4.json"
)
OUTPUT_ROOT = (
    "/root/userdata/MappingNetworks/outputs/"
    "y400_mai_v3_mlp_product_fht_depth_calibration"
)
SOURCE_PATHS = (
    "examples/nanogpt/model.py",
    "examples/nanogpt/muon.py",
    "examples/nanogpt/train.py",
    "latent_weight_lab/block_fht.py",
)
DEPTHS = (12, 24)
PADDED_WIDTH = 4096
OUTPUT_WIDTH = 768
MATERIALIZED_WEIGHTS = 768 * 3072
SIX_FACTOR_COSINE = 0.10021041395763557
SIX_FACTOR_FRACTION = (6 * PADDED_WIDTH + OUTPUT_WIDTH) / (
    MATERIALIZED_WEIGHTS
)
SIX_FACTOR_ENRICHMENT = SIX_FACTOR_COSINE / math.sqrt(
    SIX_FACTOR_FRACTION
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def main() -> None:
    implementation_commit = git_head()
    source_hashes = {
        path: sha256(ROOT / path) for path in SOURCE_PATHS
    }
    parent = json.loads(PARENT.read_text())
    candidates = []
    for factors in DEPTHS:
        stem = (
            "y400_mai_v3_124m_fullattn_plus_mlp_cproj_"
            f"productfht{factors}_wsmuon_jvpnorm_depth_calibration"
        )
        probe_output = (
            f"{OUTPUT_ROOT}/productfht{factors}_real_batch_probe.json"
        )
        trainable_scalars = factors * PADDED_WIDTH + OUTPUT_WIDTH
        trainable_fraction = trainable_scalars / MATERIALIZED_WEIGHTS
        ambient_cosine = math.sqrt(trainable_fraction)
        required_cosine = (
            1.15 * SIX_FACTOR_ENRICHMENT * ambient_cosine
        )
        config = dict(parent)
        config.update(
            {
                "out_dir": f"{OUTPUT_ROOT}/{stem}",
                "hpo_stage": (
                    "cproj_product_fht_depth_direction_calibration"
                ),
                "ladder_slot": (
                    f"diagnostic_productfht{factors}_not_scientific_screen"
                ),
                "candidate_scope": (
                    f"one real FineWeb-Edu update for a {factors}-factor "
                    "diagonal-FHT c_proj chart; measure whether tangent "
                    "direction enrichment rises beyond capacity-only ambient "
                    "scaling"
                ),
                "block_fht_cproj_product_fht_factors": factors,
                "block_fht_cproj_product_fht_pullback_normalize": True,
                "block_fht_cproj_product_fht_pullback_max_coordinate_update": (
                    0.02
                ),
                "block_fht_cproj_product_fht_pullback_refresh_interval": 4,
                "block_fht_cproj_product_fht_pullback_probe": True,
                "block_fht_cproj_product_fht_pullback_probe_output": (
                    probe_output
                ),
                "implementation_commit": implementation_commit,
                "implementation_source_hashes": source_hashes,
                "mfu_preflight_certificate": (
                    f"{OUTPUT_ROOT}/performance_preflight_{stem}.json"
                ),
                "failed_mfu_preflight": None,
                "max_iters": 1,
                "warmup_iters": 0,
                "lr_decay_iters": 1,
                "eval_interval": 1,
                "eval_iters": 1,
                "save_checkpoint": False,
                "checkpoint_history": False,
                "registered_resume_determinism_required": False,
                "registered_resume_protocol": (
                    "not_applicable_single_update_diagnostic"
                ),
                "launch_ready": True,
                "screen_only": False,
                "diagnostic_only": True,
                "monitoring_policy": (
                    "run the exact performance gate and one real update "
                    "synchronously on an idle Y400 GPU; poll directly with "
                    "no watchdog, callback, queue worker, or PRO6 execution"
                ),
                "depth_direction_gate": {
                    "trainable_scalars_per_layer": trainable_scalars,
                    "trainable_fraction_per_cproj": trainable_fraction,
                    "square_root_ambient_cosine": ambient_cosine,
                    "six_factor_normalized_enrichment": (
                        SIX_FACTOR_ENRICHMENT
                    ),
                    "required_normalized_enrichment_multiplier": 1.15,
                    "required_mean_task_direction_cosine": (
                        required_cosine
                    ),
                    "actual_to_target_update_norm_ratio_per_layer": [
                        0.8,
                        1.2
                    ],
                    "maximum_coordinate_update": 0.02,
                    "minimum_mfu_fraction": 0.2
                },
                "parent_result": (
                    "examples/nanogpt/configs/selection_artifacts/"
                    "124m_mlp_product_fht_jvpnorm_screen_result.json"
                )
            }
        )
        config_path = CONFIG_DIR / f"{stem}.json"
        config_path.write_text(
            json.dumps(
                config,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        candidates.append(
            {
                "factors": factors,
                "config": str(config_path.relative_to(ROOT)),
                "config_sha256": sha256(config_path),
                "probe_output": probe_output,
                "gate": config["depth_direction_gate"],
                "mfu_preflight_certificate": config[
                    "mfu_preflight_certificate"
                ]
            }
        )
    plan = {
        "schema_version": (
            "mai_124m_mlp_product_fht_depth_calibration_plan_v1"
        ),
        "status": "registered_before_calibration",
        "implementation_commit": implementation_commit,
        "implementation_source_hashes": source_hashes,
        "parent_result": (
            "examples/nanogpt/configs/selection_artifacts/"
            "124m_mlp_product_fht_jvpnorm_screen_result.json"
        ),
        "hypothesis": (
            "if deeper nonlinear diagonal-FHT products learn task-aligned "
            "moving axes rather than merely buying proportional random "
            "coordinates, their one-step tangent cosine normalized by the "
            "square root of trainable fraction should exceed the six-factor "
            "chart by at least 15 percent"
        ),
        "candidates": candidates,
        "decision_rule": {
            "promote_at_most_one": True,
            "promote": (
                "candidate passes exact movement and 20-percent MFU gates "
                "and its mean task-direction cosine clears its registered "
                "15-percent enrichment threshold"
            ),
            "reject": (
                "candidate fails MFU/movement, or cosine growth is explained "
                "by square-root ambient dimension alone"
            )
        },
        "launch_rule": (
            "poll each performance test and one-step diagnostic directly; "
            "do not attach short calibrations to a watchdog"
        )
    }
    plan_path = (
        ARTIFACT_DIR / "124m_mlp_product_fht_depth_calibration_plan.json"
    )
    plan_path.write_text(
        json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "plan_path": str(plan_path.relative_to(ROOT)),
                "plan_sha256": sha256(plan_path),
                "candidates": candidates
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
