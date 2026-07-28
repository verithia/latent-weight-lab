#!/usr/bin/env python3
"""Generate manifold- and residual-aligned 124M MLP structure screens."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "examples/nanogpt/configs"
BASE = (
    CONFIG_DIR
    / "y400_mai_v3_124m_fullattn_plus_mlp_cproj_muonchart154_0p5tpp_lr24e4.json"
)
IMPLEMENTATION_COMMIT = "fcf14feae3eff0990948738abd85499ba4609ffe"
ANALYSIS_COMMIT = "8d72e96a5a1fe254965acf2a8998382811ec62f6"
SOURCE_HASHES = {
    "examples/nanogpt/model.py": (
        "caf930afd9d0f92c103dea850e1c7b1301ecd2118a66b7860f6231286588c9fa"
    ),
    "examples/nanogpt/train.py": (
        "daebbbb7ed1080a172b4dae37169a9fa7458dfb8d1ae891cd9c6d1746f5ad63c"
    ),
    "examples/nanogpt/parameter_trajectory.py": (
        "897e4aa4007d8f8b1cf56891f0cfe2a3a4dc0807eaa456ba70b096429ec022b6"
    ),
    "latent_weight_lab/block_fht.py": (
        "003d041159cf6b57575ac2ffcee06db9fddbb7b13cb80788f0c21e432b0bf393"
    ),
}
ANALYSIS_ARTIFACTS = {
    "paper_tsne_continuity_summary_sha256": (
        "9065cb52573868cbeb1b00b988c286f386cdf80d3a1ecd7582f073b07ce77d92"
    ),
    "block_fht_tangent_overlap_sha256": (
        "f4f56e3ff99216140ce22eca32f947789219dc5fe0b9e653718da2aadb9ce095"
    ),
    "shared_hidden_radial_overlap_sha256": (
        "26f92578de7ca236f888e0560f2965ef68138ca7182f0fca3d7648f6cc61cf85"
    ),
    "residual_compatibility_sha256": (
        "c05497a623d44d91abd986cd44c4490e6714058882b39c87aff4c0672ada7021"
    ),
}
QUADRATIC_RESULT_SHA256 = (
    "788f5ebdfb5bbfd24d78763e7a4b0bc901b8b67715b7d1621f2185a634b2dff6"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict[Path, dict]:
    base = json.loads(BASE.read_text())
    variants = (
        {
            "label": "radial4",
            "shared_gain": True,
            "gain_scale": 4.0,
            "rotation_stages": 0,
            "hypothesis": (
                "metric-correct the paired hidden radial coordinate so its effective "
                "early displacement matches the dense trajectory scale"
            ),
        },
        {
            "label": "givens1",
            "shared_gain": False,
            "gain_scale": 1.0,
            "rotation_stages": 1,
            "hypothesis": (
                "repair c_proj-to-residual directional alignment with an exactly "
                "norm-preserving identity-initialized minimal output rotation"
            ),
        },
        {
            "label": "radial4_givens1",
            "shared_gain": True,
            "gain_scale": 4.0,
            "rotation_stages": 1,
            "hypothesis": (
                "combine the measured paired hidden radial tangent with the "
                "minimal residual-alignment output rotation"
            ),
        },
    )
    outputs: dict[Path, dict] = {}
    for variant in variants:
        label = variant["label"]
        run_name = (
            "y400_mai_v3_124m_fullattn_plus_mlp_cproj_"
            f"{label}_muonchart154_0p5tpp"
        )
        config = copy.deepcopy(base)
        config.update(
            {
                "block_fht_quadratic_targets": [],
                "block_fht_quadratic_scale": 0.0,
                "block_fht_quadratic_seed_offset": 104729,
                "block_fht_mlp_shared_hidden_gain": variant["shared_gain"],
                "block_fht_mlp_shared_hidden_gain_scale": variant["gain_scale"],
                "block_fht_mlp_output_rotation_stages": variant["rotation_stages"],
                "block_fht_mlp_output_rotation_seed": 271828,
                "hpo_stage": f"post_manifold_cproj_{label}_0p5tpp",
                "ladder_role": "mlp_manifold_residual_geometry_screen_provisional",
                "ladder_slot": f"cproj_{label}",
                "out_dir": (
                    "/root/userdata/MappingNetworks/outputs/"
                    f"y400_mai_v3_mlp_geometry_screens/{run_name}"
                ),
                "candidate_scope": (
                    "accepted full-attention structure plus generated mlp.c_proj "
                    f"with {variant['hypothesis']}"
                ),
                "manifold_structure": {
                    "hypothesis": variant["hypothesis"],
                    "shared_hidden_gain": variant["shared_gain"],
                    "shared_hidden_gain_coordinate_scale": variant["gain_scale"],
                    "output_rotation_stages": variant["rotation_stages"],
                    "output_rotation_angles_per_layer": (
                        variant["rotation_stages"] * 768 // 2
                    ),
                    "output_rotation_initialization": "exact identity",
                    "output_rotation_invariant": (
                        "per-token Euclidean norm and activation covariance spectrum"
                    ),
                    "learned_dense_basis": False,
                    "lora_adapter": False,
                },
                "manifold_evidence": {
                    **ANALYSIS_ARTIFACTS,
                    "analysis_commit": ANALYSIS_COMMIT,
                    "paper_style_tsne": (
                        "layer-separated continuous traces; 5-NN same-layer fraction "
                        "0.945/0.947 for c_fc/c_proj; visualization only"
                    ),
                    "random_tangent": (
                        "1% linear BlockFHT captures 1.005% mean dense displacement "
                        "energy; 2/4/8% capture 2.004/4.001/8.002%, showing dimensional "
                        "but no special orientation alignment"
                    ),
                    "paired_radial": (
                        "shared hidden radial tangent captures 7.28% mean early-window "
                        "energy from 3,072 coordinates; prior learned log gains moved "
                        "only about 0.006-0.008 versus dense early RMS 0.036"
                    ),
                    "residual_interface": (
                        "relative to attention-only, generated c_proj changes "
                        "update/residual RMS 0.8121 to 0.7116 and parallel energy "
                        "0.0570 to 0.0256 while activation/update ranks increase"
                    ),
                },
                "rejected_quadratic_result": (
                    "examples/nanogpt/configs/selection_artifacts/"
                    "124m_mlp_quadratic_screens_result.json"
                ),
                "rejected_quadratic_result_sha256": QUADRATIC_RESULT_SHA256,
                "implementation_commit": IMPLEMENTATION_COMMIT,
                "implementation_source_hashes": SOURCE_HASHES,
                "mfu_preflight_certificate": (
                    "/root/userdata/MappingNetworks/outputs/"
                    "y400_mai_v3_mlp_geometry_screens/"
                    f"performance_preflight_{label}.json"
                ),
                "recipe_resolution_stage": (
                    "post_manifold_smallest_rung_aligned_coordinate_screen"
                ),
                "recipe_resolution_dependency": (
                    "exact dense-trajectory tangent overlap and fixed-token residual "
                    "compatibility; no 5TPP promotion without matched terminal evidence"
                ),
                "selection_endpoint": (
                    "terminal held-out NLL versus matched attention-only and linear "
                    "generated-c_proj controls, followed by residual compatibility"
                ),
                "screen_only": True,
                "screen_only_resolution": (
                    "124M/0.5TPP causal structure screen on fixed evaluation windows"
                ),
                "practical_equivalence_nll": 0.02,
                "launch_ready": True,
                "launch_block_reason": None,
            }
        )
        path = (
            CONFIG_DIR
            / f"y400_mai_v3_124m_fullattn_plus_mlp_cproj_{label}_muonchart154_0p5tpp_lr24e4.json"
        )
        outputs[path] = config
    return outputs


def main() -> None:
    for path, payload in build().items():
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"{sha256(path)}  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
