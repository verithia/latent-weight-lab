#!/usr/bin/env python3
"""Generate the 124M nonlinear FHT-chart MLP structure screens."""

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
IMPLEMENTATION_COMMIT = "5d8686ecd9219226d923e457a33dc11924084b46"
MANIFOLD_METADATA_SHA256 = (
    "335e380ef96f1c59e15d21d33057f64c46161415d500671b182e6638b228237a"
)
CONTROL_RESULT_SHA256 = (
    "937a5fb824c77e4fddacd4cd9bcdf5ca90c7edc8787ec2a73fdd8f3bdc600c13"
)
SOURCE_HASHES = {
    "examples/nanogpt/model.py": (
        "fcc552c0f33971f10ef9722fb33c4c44148f266f294f6d9607007e698446f4df"
    ),
    "examples/nanogpt/train.py": (
        "e20108cdeb86f9c5a91ba5f69c150acc8f70db52a03f0d6e0938058830e7549c"
    ),
    "latent_weight_lab/block_fht.py": (
        "003d041159cf6b57575ac2ffcee06db9fddbb7b13cb80788f0c21e432b0bf393"
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scale_label(scale: float) -> str:
    return f"{scale:.2f}".replace(".", "p")


def build() -> dict[Path, dict]:
    base = json.loads(BASE.read_text())
    outputs: dict[Path, dict] = {}
    for scale in (0.25, 0.50, 1.00):
        label = scale_label(scale)
        run_name = (
            "y400_mai_v3_124m_fullattn_plus_mlp_cproj_"
            f"quadraticfht_q{label}_muonchart154_0p5tpp"
        )
        config = copy.deepcopy(base)
        config.update(
            {
                "block_fht_quadratic_targets": ["mlp.c_proj"],
                "block_fht_quadratic_scale": scale,
                "block_fht_quadratic_seed_offset": 104729,
                "block_fht_mlp_shared_hidden_gain": False,
                "hpo_stage": f"post_manifold_cproj_quadratic_fht_q{label}_0p5tpp",
                "ladder_role": "mlp_nonlinear_manifold_structure_screen_provisional",
                "ladder_slot": f"cproj_quadratic_fht_q{label}",
                "out_dir": (
                    "/root/userdata/MappingNetworks/outputs/"
                    f"y400_mai_v3_mlp_quadratic_screens/{run_name}"
                ),
                "candidate_scope": (
                    "accepted full-attention structure plus generated mlp.c_proj; "
                    "replace the fixed affine BlockFHT chart Az with the centered "
                    f"variance-preserving fixed quadratic chart at q={scale:.2f}"
                ),
                "quadratic_fht_chart": {
                    "formula": (
                        "theta(z) = [A z + q * center((B z) odot (C z)) / "
                        "latent_init_std] / sqrt(1 + q^2 * latent_dim / "
                        "next_power_of_two(latent_dim))"
                    ),
                    "scale": scale,
                    "fixed_maps": 3,
                    "seed_offset": 104729,
                    "trainable_parameter_overhead": 0,
                    "learned_basis": False,
                    "geometry": (
                        "non-affine quadratic immersion with a latent-dependent "
                        "tangent; output is no longer confined to one fixed "
                        "linear generator subspace"
                    ),
                    "initialization_control": (
                        "centered product and analytic variance normalization "
                        "preserve the matched GPT weight standard deviation"
                    ),
                },
                "manifold_study": {
                    "dense_run": (
                        "y400_mai_v3_124m_dense_mlp_trajectory_0p5tpp_lr24e4"
                    ),
                    "structure_analysis_metadata_sha256": (
                        MANIFOLD_METADATA_SHA256
                    ),
                    "evidence": (
                        "Every local window is two-PC at 95 percent energy; PC2 "
                        "is a quadratic function of PC1 with mean R2 above 0.997, "
                        "while the spatial directions are high matrix rank."
                    ),
                    "control_result_sha256": CONTROL_RESULT_SHA256,
                    "rejected_controls": [
                        "native c_proj latent reassigned from AdamW to Muon",
                        "direct shared hidden-channel diagonal gain",
                        "joint c_fc+c_proj Muon charts",
                    ],
                    "scope_limit": (
                        "A single optimizer trajectory motivates the chart order "
                        "but does not establish the global solution manifold."
                    ),
                },
                "implementation_commit": IMPLEMENTATION_COMMIT,
                "implementation_source_hashes": SOURCE_HASHES,
                "mfu_preflight_certificate": (
                    "/root/userdata/MappingNetworks/outputs/"
                    "y400_mai_v3_mlp_quadratic_screens/"
                    f"performance_preflight_{label}.json"
                ),
                "recipe_resolution_stage": (
                    "post_manifold_smallest_rung_nonlinear_chart_screen"
                ),
                "recipe_resolution_dependency": (
                    "paper-faithful per-layer trajectory PCA plus local curve and "
                    "spatial-rank diagnostics; no 5TPP promotion without matched "
                    "terminal evidence"
                ),
                "selection_endpoint": (
                    "terminal held-out NLL versus matched attention-only, plain "
                    "generated-c_proj, and linear Muon-chart controls"
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
            / f"y400_mai_v3_124m_fullattn_plus_mlp_cproj_quadraticfht_q{label}_muonchart154_0p5tpp_lr24e4.json"
        )
        outputs[path] = config
    return outputs


def main() -> None:
    for path, payload in build().items():
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"{sha256(path)}  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
