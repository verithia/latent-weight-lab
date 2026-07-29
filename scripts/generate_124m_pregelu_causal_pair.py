#!/usr/bin/env python3
"""Generate the preregistered matched 124M pre-GELU causal pair."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "examples" / "nanogpt" / "configs"
SOURCE = CONFIG_DIR / (
    "y400_mai_v3_124m_fullattn_plus_mlp_cproj_"
    "hiddenblock32_s2_c4_g4_outblock32_s4_c4_g4_"
    "cachevjp_muonchart154_0p5tpp_lr24e4.json"
)
PLAN = CONFIG_DIR / "selection_artifacts" / "124m_mlp_pregelu_causal_pair_plan.json"
IMPLEMENTATION_COMMIT = "ce30dccc1114a466d73e0950b30036961d34717c"
SOURCE_FILES = (
    "examples/nanogpt/model.py",
    "examples/nanogpt/train.py",
    "examples/nanogpt/muon.py",
    "examples/nanogpt/parameter_trajectory.py",
    "latent_weight_lab/block_fht.py",
)
OUTPUT_ROOT = (
    "/root/userdata/MappingNetworks/outputs/"
    "y400_mai_v3_mlp_pregelu_causal_pair"
)
PARENT_NAME = (
    "y400_mai_v3_124m_fullattn_plus_mlp_cproj_"
    "postgelu_rotations_matched_0p5tpp_lr24e4"
)
CANDIDATE_NAME = (
    "y400_mai_v3_124m_fullattn_plus_mlp_cproj_"
    "pregelu_s2_postgelu_rotations_matched_0p5tpp_lr24e4"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def make_arm(source: dict, *, name: str, pregelu_stages: int) -> dict:
    config = copy.deepcopy(source)
    for stale_key in (
        "cache_vjp_evidence",
        "four_stage_metric_followup_evidence",
        "manifold_structure",
        "optimizer_assignment_expected",
        "parent_output_chart_config",
        "parent_output_chart_config_sha256",
        "residual_metric_evidence",
    ):
        config.pop(stale_key, None)
    config.update(
        {
            "bilateral_manifold_structure": {
                "cached_weight_formula": (
                    "R_out W_generated R_hidden^T; no scalar gains"
                ),
                "hidden_features": 3072,
                "hidden_rotation_coordinates": 95232,
                "hidden_stages": 2,
                "output_features": 768,
                "output_rotation_coordinates": 47616,
                "output_stages": 4,
                "identity_initialization": True,
                "learned_dense_basis": False,
                "lora_adapter": False,
                "scalar_gains": False,
            },
            "block_fht_mlp_hidden_gain": False,
            "block_fht_mlp_hidden_log_gain_init": 0.0,
            "block_fht_mlp_residual_output_gain": False,
            "block_fht_mlp_residual_output_log_gain_init": 0.0,
            "block_fht_mlp_chart_lr_scale": 1.0,
            "block_fht_mlp_pregelu_block_rotation_stages": pregelu_stages,
            "block_fht_mlp_pregelu_block_rotation_size": 32,
            "block_fht_mlp_pregelu_block_rotation_basis_size": 256,
            "block_fht_mlp_pregelu_block_rotation_coordinate_scale": 4.0,
            "block_fht_mlp_pregelu_block_rotation_seed": 161803,
            "block_fht_mlp_pregelu_chart_lr_scale": 0.1,
            "checkpoint_history": False,
            "checkpoint_wall_clock_seconds": 7200,
            "candidate_scope": (
                "matched post-GELU rotations without scalar gains"
                + (
                    " plus an independent two-stage folded pre-GELU frame"
                    if pregelu_stages
                    else ""
                )
            ),
            "failed_mfu_preflight": None,
            "hpo_stage": "pregelu_causal_pair_preregistered_0p5tpp",
            "implementation_commit": IMPLEMENTATION_COMMIT,
            "implementation_source_hashes": {
                relative: sha256(ROOT / relative) for relative in SOURCE_FILES
            },
            "ladder_interpretation": (
                "single preregistered causal matched pair; not a fitted "
                "MLP scaling ladder"
            ),
            "ladder_role": "mlp_pregelu_causal_pair_preregistered",
            "ladder_slot": (
                "candidate_pregelu_s2"
                if pregelu_stages
                else "parent_postgelu_rotations_only"
            ),
            "launch_block_reason": None,
            "launch_ready": True,
            "mfu_min_fraction": 0.2,
            "mfu_preflight_certificate": (
                f"{OUTPUT_ROOT}/performance_preflight_{name}.json"
            ),
            "mfu_preflight_required": True,
            "optimizer_assignment_expected": (
                "matrix-shaped c_proj BlockFHT latents are Muon-owned; "
                "post-GELU Cayley coordinates use the registered AdamW "
                "fallback; candidate pre-GELU coordinates use their separate "
                "0.1-scaled AdamW group"
            ),
            "out_dir": f"{OUTPUT_ROOT}/{name}",
            "practical_equivalence_nll": 0.02,
            "practical_equivalence_policy": (
                "candidate must beat the exact matched parent by at least "
                "0.020 terminal fixed-window validation CE to promote"
            ),
            "pre_gelu_causal_pair": {
                "arm": "candidate" if pregelu_stages else "parent",
                "matched_parent": PARENT_NAME,
                "candidate": CANDIDATE_NAME,
                "only_arm_difference": (
                    "candidate enables the two-stage pre-GELU frame and its "
                    "separate 0.1 chart LR scale"
                ),
                "plan": str(PLAN.relative_to(ROOT)),
                "plan_sha256": sha256(PLAN),
                "postgelu_hidden_frame": "block32 stages2 basis256 scale4",
                "postgelu_output_frame": "block32 stages4 basis256 scale4",
                "scalar_gains": False,
                "residual_hyperconnection": False,
                "terminal_promotion_margin_ce": 0.02,
            },
            "recipe_resolution_dependency": (
                "positive three-seed task-conditioned pre-GELU endpoint "
                "capacity result"
            ),
            "recipe_resolution_required": False,
            "trajectory_snapshot_interval": 0,
        }
    )
    return config


def main() -> None:
    source = json.loads(SOURCE.read_text())
    outputs = (
        (PARENT_NAME, 0),
        (CANDIDATE_NAME, 2),
    )
    for name, stages in outputs:
        destination = CONFIG_DIR / f"{name}.json"
        write_json(
            destination,
            make_arm(source, name=name, pregelu_stages=stages),
        )
        print(f"{destination.relative_to(ROOT)} {sha256(destination)}")


if __name__ == "__main__":
    main()
