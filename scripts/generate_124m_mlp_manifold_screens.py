#!/usr/bin/env python3
"""Generate the registered 124M MLP manifold screen configurations."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "examples/nanogpt/configs"
BASE = CONFIG_DIR / "y400_mai_v3_124m_fullattn_plus_mlp_cproj_0p5tpp_lr24e4_provisional.json"
MANIFOLD_METADATA_SHA256 = "335e380ef96f1c59e15d21d33057f64c46161415d500671b182e6638b228237a"
ATTENTION_TARGETS = ["attn.c_attn.qk_headwise", "attn.c_attn.v", "attn.c_proj"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def common(base: dict, *, run_name: str, stage: str, slot: str) -> dict:
    config = copy.deepcopy(base)
    config.update(
        {
            "block_fht_muon_latent_rows": 154,
            "hpo_stage": stage,
            "ladder_role": "mlp_manifold_structure_screen_provisional",
            "ladder_slot": slot,
            "out_dir": (
                "/root/userdata/MappingNetworks/outputs/"
                f"y400_mai_v3_mlp_manifold_screens/{run_name}"
            ),
            "manifold_study": {
                "dense_run": "y400_mai_v3_124m_dense_mlp_trajectory_0p5tpp_lr24e4",
                "structure_analysis_metadata_sha256": MANIFOLD_METADATA_SHA256,
                "evidence": (
                    "Each c_fc/c_proj path is locally two-PC at 95 percent energy, "
                    "but terminal displacements and temporal PC directions are "
                    "high matrix rank; layer tangents are raw-coordinate orthogonal, "
                    "while paired hidden-channel displacement norms are strongly "
                    "correlated."
                ),
                "scope_limit": (
                    "One Muon optimizer trajectory is not the intrinsic dimension "
                    "of the global solution manifold."
                ),
            },
            "muon_latent_chart": {
                "shape": [154, 154],
                "scalars_per_mlp_matrix": 23716,
                "unrounded_one_percent_scalars": 23593,
                "capacity_overhead_fraction": (23716 - 23593) / 23593,
                "generator_forward": (
                    "unchanged BlockFHT map after row-major flattening; the 2D "
                    "shape changes optimizer geometry only"
                ),
                "learned_basis": False,
            },
            "recipe_resolution_stage": "post_manifold_smallest_rung_structure_screen",
            "recipe_resolution_dependency": (
                "quantitative 124M dense Muon c_fc/c_proj trajectory analysis; "
                "do not promote before matched terminal evidence"
            ),
            "screen_only": True,
            "screen_only_resolution": (
                "124M/0.5TPP causal structure screen on fixed evaluation windows"
            ),
            "selection_endpoint": (
                "terminal held-out NLL versus matched attention-only and plain "
                "generated-c_proj controls"
            ),
            "practical_equivalence_nll": 0.02,
            "launch_ready": True,
            "launch_block_reason": None,
        }
    )
    return config


def build() -> dict[Path, dict]:
    base = json.loads(BASE.read_text())
    base_sha = sha256(BASE)

    cproj = common(
        base,
        run_name="y400_mai_v3_124m_fullattn_plus_mlp_cproj_muonchart154_0p5tpp",
        stage="post_manifold_cproj_muon_chart_0p5tpp",
        slot="cproj_muon_chart_optimizer_control",
    )
    cproj.update(
        {
            "block_fht_targets": ATTENTION_TARGETS + ["mlp.c_proj"],
            "block_fht_muon_latent_targets": ["mlp.c_proj"],
            "block_fht_mlp_shared_hidden_gain": False,
            "candidate_scope": (
                "accepted full-attention structure plus generated mlp.c_proj; "
                "only c_proj latent optimizer ownership changes to Muon"
            ),
            "optimizer_assignment_expected": (
                "dense c_fc and matrix-shaped c_proj latents are Muon-owned; "
                "attention latents remain on the registered AdamW fallback"
            ),
            "parent_plain_cproj_config": str(BASE.relative_to(ROOT)),
            "parent_plain_cproj_config_sha256": base_sha,
        }
    )

    paired = copy.deepcopy(cproj)
    paired.update(
        {
            "hpo_stage": "post_manifold_cproj_muon_chart_shared_hidden_gain_0p5tpp",
            "ladder_slot": "cproj_muon_chart_plus_paired_hidden_channel_gain",
            "out_dir": (
                "/root/userdata/MappingNetworks/outputs/"
                "y400_mai_v3_mlp_manifold_screens/"
                "y400_mai_v3_124m_fullattn_plus_mlp_cproj_muonchart154_sharedhidden_0p5tpp"
            ),
            "block_fht_mlp_shared_hidden_gain": True,
            "candidate_scope": (
                "c_proj Muon-chart control plus one shared log-gain over expansion "
                "channels, applied before GELU and again before c_proj"
            ),
            "paired_hidden_channel_chart": {
                "parameters_per_layer": 3072,
                "initialization": "log-gain zero; exact identity at step zero",
                "application": (
                    "multiply c_fc output before GELU and post-GELU c_proj input "
                    "by the same positive channel gain"
                ),
                "basis": "none",
            },
        }
    )

    joint = copy.deepcopy(paired)
    joint.update(
        {
            "hpo_stage": "post_manifold_joint_mlp_muon_chart_shared_hidden_gain_0p5tpp",
            "ladder_slot": "joint_mlp_muon_chart_plus_paired_hidden_channel_gain",
            "out_dir": (
                "/root/userdata/MappingNetworks/outputs/"
                "y400_mai_v3_mlp_manifold_screens/"
                "y400_mai_v3_124m_fullattn_plus_jointmlp_muonchart154_sharedhidden_0p5tpp"
            ),
            "block_fht_targets": ATTENTION_TARGETS + ["mlp.c_fc", "mlp.c_proj"],
            "block_fht_muon_latent_targets": ["mlp.c_fc", "mlp.c_proj"],
            "candidate_scope": (
                "accepted full-attention structure plus jointly generated c_fc "
                "and c_proj, separate layer/matrix latents, Muon chart ownership, "
                "and a shared hidden-channel log-gain"
            ),
            "optimizer_assignment_expected": (
                "separate matrix-shaped c_fc and c_proj latents are Muon-owned; "
                "attention latents and shared 1D hidden gains use AdamW fallback"
            ),
        }
    )

    return {
        CONFIG_DIR
        / "y400_mai_v3_124m_fullattn_plus_mlp_cproj_muonchart154_0p5tpp_lr24e4.json": cproj,
        CONFIG_DIR
        / "y400_mai_v3_124m_fullattn_plus_mlp_cproj_muonchart154_sharedhidden_0p5tpp_lr24e4.json": paired,
        CONFIG_DIR
        / "y400_mai_v3_124m_fullattn_plus_jointmlp_muonchart154_sharedhidden_0p5tpp_lr24e4.json": joint,
    }


def main() -> None:
    for path, payload in build().items():
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"{sha256(path)}  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
