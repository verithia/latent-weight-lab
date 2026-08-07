"""Generate the preregistered tail-two c_proj LWT 124M/5TPP config."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / (
    "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_repairedfullattn_plus_cfc_latecproj_lwt_5tpp_lr24e4.json"
)
PLAN = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_repaired_attention_cfc_tail2_cproj_lwt_5tpp_plan.json"
)
GEOMETRY_CORRECTION = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_repaired_attention_cfc_tail2_cproj_lwt_5tpp_geometry_correction.json"
)
OUTPUT = ROOT / (
    "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_repairedfullattn_plus_cfc_tail2cproj_lwt_5tpp_lr24e4.json"
)
SOURCE_PATHS = (
    "examples/nanogpt/model.py",
    "examples/nanogpt/muon.py",
    "examples/nanogpt/muon_matched_givens.py",
    "examples/nanogpt/train.py",
    "examples/nanogpt/mfu_preflight.py",
    "examples/nanogpt/test_mlp_cproj_layer_allocation.py",
    "latent_weight_lab/block_fht.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def main() -> None:
    config = json.loads(PARENT.read_text())
    root = (
        "/mnt/ssd-data/orj/MappingNetworks/outputs/"
        "pro6_mai_v3_124m_repairedfullattn_plus_cfc_tail2cproj_lwt_5tpp"
    )
    config.update(
        {
            "out_dir": f"{root}/scientific",
            "hpo_stage": "repaired_attention_cfc_tail2_cproj_lwt_124m_5tpp",
            "ladder_slot": "124m_5tpp_cfc_all_cproj_layers10_11",
            "ladder_role": "strict_loss_tail2_layerwise_mlp_allocation",
            "ladder_interpretation": (
                "single preregistered LWT Pareto-boundary test of the final "
                "two c_proj layers; not a fitted mask sweep or scaling ladder"
            ),
            "candidate_scope": (
                "Repaired attention and accepted c_fc in all layers; ordinary "
                "dense c_proj in layers 0-9 and the accepted hidden64+residual24 "
                "decay-0.5 procedural c_proj only in layers 10-11, all trained "
                "jointly from initialization."
            ),
            "block_fht_mlp_cproj_muon_matched_givens_layers": [10, 11],
            "candidate_cproj_target_elements": 2 * 768 * 3072,
            "candidate_cproj_procedural_coordinates_per_update": 2 * 135168,
            "candidate_cproj_coordinate_ratio": 0.057291666666666664,
            "realized_pytorch_trainable_parameter_reduction": 0,
            "checkpoint_wall_clock_seconds": 7200,
            "registered_plan": str(PLAN.relative_to(ROOT)),
            "registered_plan_sha256": sha256(PLAN),
            "registered_plan_geometry_correction": str(
                GEOMETRY_CORRECTION.relative_to(ROOT)
            ),
            "registered_plan_geometry_correction_sha256": sha256(
                GEOMETRY_CORRECTION
            ),
            "implementation_commit": git_head(),
            "implementation_source_hashes": {
                path: sha256(ROOT / path) for path in SOURCE_PATHS
            },
            "mfu_preflight_certificate": f"{root}/performance_preflight.json",
            "mfu_measurement_protocol": (
                "foreground exact scientific config, one warmup plus eight "
                "timed real updates on PRO6 GPU0; verify dense c_proj layers "
                "0-9, procedural c_proj layers 10-11, native BlockFHT, and "
                "finite losses; no watchdog"
            ),
            "monitoring_policy": (
                "one idempotent terminal-only watchdog; completion or "
                "actionable error/stall callbacks only; no milestones, "
                "heartbeats, or duplicate terminal messages"
            ),
            "selection_endpoint": (
                "terminal fixed-window validation CE <=3.630838041305542 and "
                "no fixed curve point more than 0.010 worse than the accepted "
                "c_fc-only parent"
            ),
            "practical_equivalence_nll": 0.005,
            "practical_equivalence_policy": (
                "primary incremental MLP gate: terminal fixed-window "
                "validation CE <=3.630838041305542, at most +0.005 versus "
                "the accepted c_fc-only parent 3.625838041305542; every fixed "
                "curve point must be finite and at most +0.010 versus that "
                "parent; thresholds never change after observation"
            ),
            "operator_override": (
                "2026-08-07: one preregistered tail-two LWT Pareto-boundary "
                "test; no layer sweep or automatic rerun. A 1-2h run emits "
                "only terminal or actionable-error callbacks."
            ),
            "launch_ready": True,
            "launch_block_reason": None,
            "recipe_resolution_required": False,
            "recipe_resolution_stage": "same_gauge_tail2_cproj_lwt",
            "screen_only": False,
            "terminal_eval_required": True,
        }
    )
    OUTPUT.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "path": str(OUTPUT.relative_to(ROOT)),
                "sha256": sha256(OUTPUT),
                "implementation_commit": config["implementation_commit"],
                "plan_sha256": config["registered_plan_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
