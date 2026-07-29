"""Generate the 124M frozen-base plus zero-init BlockFHT c_proj delta screen."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "examples" / "nanogpt" / "configs"
PARENT = (
    CONFIG_DIR
    / "y400_mai_v3_124m_fullattn_plus_mlp_cproj_"
    "muonchart154_0p5tpp_lr24e4.json"
)
STEM = (
    "y400_mai_v3_124m_fullattn_plus_mlp_cproj_"
    "affinedelta1_muonchart154_0p5tpp"
)
OUTPUT_ROOT = (
    "/root/userdata/MappingNetworks/outputs/"
    "y400_mai_v3_mlp_affine_delta_screen"
)
SOURCE_PATHS = (
    "examples/nanogpt/mai_selection_artifacts.py",
    "examples/nanogpt/model.py",
    "examples/nanogpt/muon.py",
    "examples/nanogpt/parameter_trajectory.py",
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
    config = json.loads(PARENT.read_text())
    config.update(
        {
            "out_dir": f"{OUTPUT_ROOT}/{STEM}",
            "hpo_stage": "cproj_affine_delta_smallest_rung_0p5tpp",
            "ladder_slot": "cproj_frozen_dense_base_plus_blockfht_delta",
            "ladder_role": "mlp_manifold_affine_subspace_causal_screen",
            "candidate_scope": (
                "accepted full-attention replacement plus mlp.c_proj "
                "parameterized as a frozen GPT-initialized dense base and "
                "a zero-initialized fixed BlockFHT delta; no learned basis, "
                "LoRA adapter, or trainable dense residual"
            ),
            "block_fht_residual_base_scale": 0.0,
            "block_fht_affine_delta_targets": ["mlp.c_proj"],
            "block_fht_affine_delta_scale": 1.0,
            "implementation_commit": git_head(),
            "implementation_source_hashes": {
                path: sha256(ROOT / path) for path in SOURCE_PATHS
            },
            "mfu_preflight_certificate": (
                f"{OUTPUT_ROOT}/"
                "performance_preflight_affinedelta1_muonchart154.json"
            ),
            "failed_mfu_preflight": None,
            "mfu_preflight_required": True,
            "mfu_min_fraction": 0.2,
            "launch_ready": True,
            "recipe_resolution_required": False,
            "recipe_resolution_stage": (
                "phase_drift_affine_subspace_smallest_rung"
            ),
            "recipe_resolution_dependency": (
                "exact 41-snapshot dense c_fc/c_proj tangent analysis found "
                "high local low-rank energy but severe phase-to-phase "
                "rotation and weak capture by one origin-anchored fixed "
                "subspace"
            ),
            "selection_endpoint": (
                "terminal fixed-window validation CE versus the exact "
                "attention-only and Muon-chart generated-c_proj controls"
            ),
            "preregistered_decision_rule": {
                "primary_metric": (
                    "terminal fixed-window validation cross entropy at "
                    "update 238"
                ),
                "attention_only_validation_ce": 5.4918,
                "muonchart_cproj_validation_ce": 5.7395,
                "acceptable_attention_gap_ce": 0.1,
                "success": (
                    "stable candidate finishes at or below 5.5918 CE, "
                    "within +0.1000 of the accepted attention-only control"
                ),
                "directional_only": (
                    "stable candidate improves on 5.7395 but remains above "
                    "the +0.1000 attention-control gap"
                ),
                "reject": (
                    "candidate ties or regresses versus 5.7395, is unstable, "
                    "or does not complete"
                ),
            },
            "optimizer_assignment_expected": (
                "matrix-shaped zero-initialized c_proj BlockFHT latents are "
                "Muon-owned; attention latents use the registered AdamW "
                "fallback; frozen dense base buffers are not optimized"
            ),
            "parent_muonchart_config": str(PARENT.relative_to(ROOT)),
            "parent_muonchart_config_sha256": sha256(PARENT),
            "affine_subspace_parameterization": {
                "formula": "W_cproj(z) = W0_gpt + A_blockfht z",
                "base_distribution": (
                    "frozen Gaussian residual-projection initialization "
                    "with std 0.02/sqrt(2*n_layer)"
                ),
                "base_trainable": False,
                "delta_initialization": "exact zero",
                "delta_scale": 1.0,
                "learned_dense_basis": False,
                "lora_adapter": False,
                "additional_trainable_parameters": 0,
                "fixed_storage_caveat": (
                    "preserves trainable-coordinate compression but stores "
                    "one frozen dense c_proj base per layer"
                ),
            },
            "phase_drift_evidence": {
                "analysis_commit": (
                    "6f54c120ff6d28615146d2bc5610aa05a943fafd"
                ),
                "analysis_entrypoint_sha256": (
                    "14544cd184f6ebb4e4256e5056cfca43554e0d090268cc380212d2ccd6194507"
                ),
                "analysis_metadata_sha256": (
                    "a5215ac291fd2bb57fa9e3c1b3b7f7a56888fe68d51cda1f96efa69fa579a96b"
                ),
                "cproj_rank2_adjacent_overlap_mean": 0.0858642,
                "cproj_rank2_prior_increment_capture_mean": 0.1009809,
                "cproj_rank2_prior_chord_capture_mean": 0.1440471,
                "cproj_rank2_max_principal_angle_mean_degrees": 87.8921,
                "interpretation": (
                    "retain a full-distribution initialization point while "
                    "testing whether the compact fixed chart can express "
                    "the useful local displacement"
                ),
            },
            "screen_only_resolution": (
                "one 124M/0.5TPP causal run after a foreground-polled "
                "20-percent-MFU qualification"
            ),
            "monitoring_policy": (
                "short preflight and training run are polled directly; no "
                "watchdog, callback, queue heartbeat, or PRO6 execution"
            ),
            "prelaunch_provenance_requirements": (
                "record commit, source/config/dataset/fixed-eval hashes, "
                "literal command, exact host-local MFU certificate, status, "
                "log, and terminal checkpoint"
            ),
        }
    )
    path = CONFIG_DIR / f"{STEM}_lr24e4.json"
    path.write_text(
        json.dumps(
            config,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "out_dir": config["out_dir"],
                "mfu_preflight_certificate": config[
                    "mfu_preflight_certificate"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
