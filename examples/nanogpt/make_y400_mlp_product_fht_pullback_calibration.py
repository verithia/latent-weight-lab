"""Generate the one-step real-batch product-FHT pullback calibration."""

from __future__ import annotations

import hashlib
import json
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
STEM = (
    "y400_mai_v3_124m_fullattn_plus_mlp_cproj_"
    "productfht6_wsmuon_jvpnorm_calibration"
)
OUTPUT_ROOT = (
    "/root/userdata/MappingNetworks/outputs/"
    "y400_mai_v3_mlp_product_fht_pullback_calibration"
)
SOURCE_PATHS = (
    "examples/nanogpt/model.py",
    "examples/nanogpt/muon.py",
    "examples/nanogpt/train.py",
    "latent_weight_lab/block_fht.py",
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
    probe_output = f"{OUTPUT_ROOT}/real_batch_pullback_probe_v1.json"
    config = json.loads(PARENT.read_text())
    config.update(
        {
            "out_dir": f"{OUTPUT_ROOT}/{STEM}",
            "hpo_stage": "cproj_product_fht_pullback_calibration",
            "ladder_slot": "diagnostic_only_not_a_scientific_screen",
            "candidate_scope": (
                "one real FineWeb-Edu outer update to calibrate exact "
                "materialized JVP motion for the six-factor product-FHT "
                "c_proj chart"
            ),
            "implementation_commit": implementation_commit,
            "implementation_source_hashes": source_hashes,
            "block_fht_cproj_product_fht_pullback_normalize": True,
            "block_fht_cproj_product_fht_pullback_max_coordinate_update": (
                0.02
            ),
            "block_fht_cproj_product_fht_pullback_probe": True,
            "block_fht_cproj_product_fht_pullback_probe_output": (
                probe_output
            ),
            "mfu_preflight_certificate": (
                f"{OUTPUT_ROOT}/performance_preflight_{STEM}.json"
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
                "run synchronously on one idle Y400 GPU and poll the process "
                "directly; do not attach a watchdog, callback, queue worker, "
                "or PRO6 execution"
            ),
            "calibration_gate": {
                "actual_to_target_update_norm_ratio_per_layer": [
                    0.8,
                    1.2
                ],
                "actual_target_update_cosine_minimum": 0.0,
                "coordinate_update_cap": 0.02,
                "scientific_run_blocked_until_pass": True
            },
            "parent_rejected_result": (
                "examples/nanogpt/configs/selection_artifacts/"
                "124m_mlp_product_fht_screen_result.json"
            ),
            "pullback_repair": {
                "normalization": (
                    "multiply the preconditioned factor direction by "
                    "||d_muon||_F / ||J q||_F using an exact JVP"
                ),
                "trust_region": (
                    "cap the maximum absolute factor-coordinate update at "
                    "0.02 per outer step"
                ),
                "global_clip": (
                    "exclude exactly normalized product factors while "
                    "retaining the registered clip for all other parameters"
                ),
                "probe_output": probe_output
            }
        }
    )
    config_path = CONFIG_DIR / f"{STEM}.json"
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    config_hash = sha256(config_path)
    plan = {
        "schema_version": (
            "mai_124m_mlp_product_fht_pullback_calibration_plan_v1"
        ),
        "status": "registered_before_calibration",
        "candidate_config": str(config_path.relative_to(ROOT)),
        "candidate_config_sha256": config_hash,
        "implementation_commit": implementation_commit,
        "implementation_source_hashes": source_hashes,
        "failed_run_diagnosis": {
            "product_relative_displacement_mean": 0.00231545,
            "dense_relative_displacement_mean": 0.492487,
            "endpoint_diagnosis_sha256": (
                "eb7fcf48407b4c86627f9b83a65b5af75938fbc599b0c02fbeeb485efef85ddd"
            )
        },
        "calibration_gate": config["calibration_gate"],
        "launch_rule": (
            "run only this exact committed diagnostic configuration on an "
            "idle Y400 GPU; inspect its one-step JSON synchronously before "
            "preregistering any further 238-step candidate"
        )
    }
    plan_path = (
        ARTIFACT_DIR
        / "124m_mlp_product_fht_pullback_calibration_plan.json"
    )
    plan_path.write_text(
        json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "config_path": str(config_path.relative_to(ROOT)),
                "config_sha256": config_hash,
                "plan_path": str(plan_path.relative_to(ROOT)),
                "plan_sha256": sha256(plan_path),
                "probe_output": probe_output
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
