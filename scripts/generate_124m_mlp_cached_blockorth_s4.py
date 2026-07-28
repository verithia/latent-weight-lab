#!/usr/bin/env python3
"""Generate the cache-optimized four-stage MLP output-chart screen."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "examples/nanogpt/configs"
BASE = (
    CONFIG_DIR
    / "y400_mai_v3_124m_fullattn_plus_mlp_cproj_"
    "blockorth32_s2_c16_g4_init0p125_muonchart154_0p5tpp_lr24e4.json"
)
LABEL = "blockorth32_s4_c16_g4_init0p125_cachevjp"
RUN_NAME = (
    "y400_mai_v3_124m_fullattn_plus_mlp_cproj_"
    f"{LABEL}_muonchart154_0p5tpp"
)
IMPLEMENTATION_COMMIT = "2357c77083e26b383fa40a679e8f8d17948daa13"
SOURCE_HASHES = {
    "examples/nanogpt/model.py": (
        "f8f93760043bc3aa74543cfe2a7f271166112a05e82a740d4fe2f5a25f17972f"
    ),
    "examples/nanogpt/train.py": (
        "3a28ed4975b01cc2ae7677f424b1cbcae7a2df71ea33ab40abd54e49af19ca5a"
    ),
    "examples/nanogpt/parameter_trajectory.py": (
        "897e4aa4007d8f8b1cf56891f0cfe2a3a4dc0807eaa456ba70b096429ec022b6"
    ),
    "latent_weight_lab/block_fht.py": (
        "003d041159cf6b57575ac2ffcee06db9fddbb7b13cb80788f0c21e432b0bf393"
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> tuple[Path, dict]:
    config = copy.deepcopy(json.loads(BASE.read_text()))
    config.update(
        {
            "block_fht_mlp_output_block_rotation_stages": 4,
            "block_fht_mlp_output_block_rotation_coordinate_scale": 16.0,
            "block_fht_mlp_residual_output_gain_scale": 4.0,
            "block_fht_mlp_residual_output_log_gain_init": 0.125,
            "hpo_stage": f"post_manifold_cproj_{LABEL}_0p5tpp",
            "ladder_role": "mlp_manifold_cached_composition_screen_provisional",
            "ladder_slot": f"cproj_{LABEL}",
            "out_dir": (
                "/root/userdata/MappingNetworks/outputs/"
                f"y400_mai_v3_mlp_blockorth_screens/{RUN_NAME}"
            ),
            "candidate_scope": (
                "four-stage fixed-basis block-orthogonal c_proj chart using "
                "the terminal-winning coordinate metric and measured "
                "residual-scale initialization; the complete charted weight "
                "is materialized once per optimizer step and its accumulated "
                "gradient is projected by an exact VJP"
            ),
            "implementation_commit": IMPLEMENTATION_COMMIT,
            "implementation_source_hashes": SOURCE_HASHES,
            "mfu_preflight_certificate": (
                "/root/userdata/MappingNetworks/outputs/"
                "y400_mai_v3_mlp_blockorth_screens/"
                f"performance_preflight_{LABEL}.json"
            ),
            "recipe_resolution_stage": (
                "post_terminal_blockorth_cache_optimized_composition"
            ),
            "recipe_resolution_dependency": (
                "four-stage held-out functional oracle, terminal two-stage "
                "metric bracket, exact cache/live gradient equivalence, and "
                "exact host-local >=20% MFU gate"
            ),
            "selection_endpoint": (
                "terminal held-out NLL, residual cosine/parallel energy, "
                "update-to-residual RMS, and learned chart travel"
            ),
            "launch_ready": True,
            "launch_block_reason": None,
            "failed_mfu_preflight": None,
            "cache_vjp_evidence": {
                "implementation_commit": IMPLEMENTATION_COMMIT,
                "two_stage_cached_mfu_fraction": 0.28242363441025464,
                "two_stage_uncached_mfu_fraction": 0.2133811,
                "four_stage_oracle_holdout_explained_target_energy": (
                    0.25306518077850343
                ),
                "two_stage_oracle_holdout_explained_target_energy": (
                    0.15844939947128295
                ),
                "two_stage_metric16_terminal_validation_loss": 5.6957,
                "attention_only_terminal_validation_loss": 5.4918,
            },
        }
    )
    structure = dict(config["manifold_structure"])
    structure.update(
        {
            "learned_rotation": (
                "4 stages of 24 independent 32x32 Cayley rotations "
                "conjugated by distinct fixed bases"
            ),
            "parameters_per_layer": 48384,
            "fraction_of_dense_cproj": 0.0205078125,
            "rotation_coordinate_scale": 16.0,
            "residual_output_log_gain_scale": 4.0,
            "residual_output_effective_log_gain_init": 0.125,
            "cache_semantics": (
                "one charted-weight materialization per optimizer step; "
                "exact VJP to base latent, Cayley coordinates, and log-gain"
            ),
        }
    )
    config["manifold_structure"] = structure
    path = (
        CONFIG_DIR
        / "y400_mai_v3_124m_fullattn_plus_mlp_cproj_"
        f"{LABEL}_muonchart154_0p5tpp_lr24e4.json"
    )
    return path, config


def main() -> None:
    path, payload = build()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"{sha256(path)}  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
