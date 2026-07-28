#!/usr/bin/env python3
"""Generate oracle-selected fixed-basis block-orthogonal MLP screens."""

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
IMPLEMENTATION_COMMIT = "e316c509144be877c4351ab50e46aa4bd8cc269c"
ANALYSIS_COMMIT = "2a2816df99d46909c9cac8a8d127175ce2402af5"
SOURCE_HASHES = {
    "examples/nanogpt/model.py": (
        "93bdbdfa8470f92d43b572c5968ca30bfcbb1aefc8587bb5d321846bbda58eb8"
    ),
    "examples/nanogpt/train.py": (
        "7cb2708956869def4afeada74b415c91bf4e4085898a063c5389de4b92a6d69e"
    ),
    "examples/nanogpt/parameter_trajectory.py": (
        "897e4aa4007d8f8b1cf56891f0cfe2a3a4dc0807eaa456ba70b096429ec022b6"
    ),
    "latent_weight_lab/block_fht.py": (
        "003d041159cf6b57575ac2ffcee06db9fddbb7b13cb80788f0c21e432b0bf393"
    ),
}
ORACLE_ARTIFACTS = {
    "summary_sha256": (
        "5b5be710c1aed4420f6531b4d6a3400320530d8f0cead77ce1494cf9aaae3e93"
    ),
    "functional_csv_sha256": (
        "f5dfc8409433cba948aaa69702768186d8d582e03dfa08415606e13e95d796e0"
    ),
    "weight_csv_sha256": (
        "1ba0b21f18a0d298fdc082dffc7534a6d577fb835d347d4da267dcf6caa2d807"
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict[Path, dict]:
    base = json.loads(BASE.read_text())
    variants = (
        {
            "label": "blockorth32_s2_gain4",
            "stages": 2,
            "parameters_per_layer": 24576,
            "heldout_energy": 0.15844939947128295,
            "heldout_cosine": 0.3847886919975281,
            "oracle_coordinate_rms": 0.4779899179935455,
            "oracle_log_gain_rms": 1.9792399406433105,
            "launch_ready": True,
        },
        {
            "label": "blockorth32_s4_gain4",
            "stages": 4,
            "parameters_per_layer": 48384,
            "heldout_energy": 0.25306518077850343,
            "heldout_cosine": 0.49673683047294614,
            "oracle_coordinate_rms": 0.33420037627220156,
            "oracle_log_gain_rms": 1.550129246711731,
            "launch_ready": False,
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
                "block_fht_mlp_shared_hidden_gain": False,
                "block_fht_mlp_shared_hidden_gain_scale": 1.0,
                "block_fht_mlp_output_rotation_stages": 0,
                "block_fht_mlp_output_rotation_seed": 271828,
                "block_fht_mlp_output_block_rotation_stages": variant["stages"],
                "block_fht_mlp_output_block_rotation_size": 32,
                "block_fht_mlp_output_block_rotation_basis_size": 256,
                "block_fht_mlp_output_block_rotation_coordinate_scale": 4.0,
                "block_fht_mlp_residual_output_gain": True,
                "block_fht_mlp_residual_output_gain_scale": 4.0,
                "hpo_stage": f"post_manifold_cproj_{label}_0p5tpp",
                "ladder_role": (
                    "mlp_manifold_functional_oracle_screen_provisional"
                ),
                "ladder_slot": f"cproj_{label}",
                "out_dir": (
                    "/root/userdata/MappingNetworks/outputs/"
                    f"y400_mai_v3_mlp_blockorth_screens/{run_name}"
                ),
                "candidate_scope": (
                    "accepted full-attention structure plus generated mlp.c_proj "
                    "with a fixed-global-basis block-orthogonal output chart and "
                    "residual-channel log gain"
                ),
                "manifold_structure": {
                    "hypothesis": (
                        "repair the measured c_proj-to-residual directional "
                        "misalignment with a globally mixed compact orthogonal "
                        "chart and the smaller oracle-indicated diagonal scale"
                    ),
                    "fixed_basis": (
                        "per-stage signed permutation and independent 256-wide "
                        "normalized FHT blocks"
                    ),
                    "learned_rotation": (
                        f"{variant['stages']} stages of 24 independent 32x32 "
                        "Cayley rotations conjugated by the fixed bases"
                    ),
                    "rotation_coordinate_scale": 4.0,
                    "residual_output_log_gain_scale": 4.0,
                    "parameters_per_layer": variant["parameters_per_layer"],
                    "fraction_of_dense_cproj": (
                        variant["parameters_per_layer"] / (768 * 3072)
                    ),
                    "initialization": "exact identity",
                    "orthogonal_invariant": (
                        "Euclidean norm and covariance spectrum before the "
                        "separate learned residual-channel gain"
                    ),
                    "learned_dense_basis": False,
                    "lora_adapter": False,
                },
                "manifold_evidence": {
                    **ORACLE_ARTIFACTS,
                    "analysis_commit": ANALYSIS_COMMIT,
                    "fit_layers": [0, 3, 6, 9, 11],
                    "fit_samples": 4096,
                    "heldout_sample_seed": 20260717,
                    "full_diagonal_orthogonal_heldout_energy": 0.5285202980041503,
                    "full_linear_heldout_energy": 0.6098914861679077,
                    "candidate_heldout_energy": variant["heldout_energy"],
                    "candidate_heldout_cosine": variant["heldout_cosine"],
                    "candidate_oracle_coordinate_rms": (
                        variant["oracle_coordinate_rms"]
                    ),
                    "candidate_oracle_log_gain_rms": (
                        variant["oracle_log_gain_rms"]
                    ),
                    "interpretation": (
                        "direct shallow Givens is under-capacity; fixed global "
                        "FHT bases make the same compact rotation budget align "
                        "substantially more held-out functional energy"
                    ),
                },
                "implementation_commit": IMPLEMENTATION_COMMIT,
                "implementation_source_hashes": SOURCE_HASHES,
                "mfu_preflight_certificate": (
                    "/root/userdata/MappingNetworks/outputs/"
                    "y400_mai_v3_mlp_blockorth_screens/"
                    f"performance_preflight_{label}.json"
                ),
                "optimizer_assignment_expected": (
                    "matrix-shaped c_proj BlockFHT latents are Muon-owned; "
                    "one-dimensional Cayley coordinates and residual log gains "
                    "use the registered AdamW fallback with fourfold chart metric"
                ),
                "recipe_resolution_stage": (
                    "post_manifold_smallest_rung_functional_oracle_screen"
                ),
                "recipe_resolution_dependency": (
                    "held-out fixed-token output oracle, exact identity and norm "
                    "tests, and a host-local >=20 percent MFU certificate"
                ),
                "selection_endpoint": (
                    "terminal held-out NLL versus matched attention-only and "
                    "plain generated-c_proj controls"
                ),
                "screen_only": True,
                "screen_only_resolution": (
                    "124M/0.5TPP causal structure screen on fixed evaluation windows"
                ),
                "practical_equivalence_nll": 0.02,
                "launch_ready": variant["launch_ready"],
                "launch_block_reason": (
                    None
                    if variant["launch_ready"]
                    else (
                        "host-local foreground gate measured 14.2031% MFU, "
                        "below the mandatory 20% floor"
                    )
                ),
            }
        )
        if not variant["launch_ready"]:
            config["failed_mfu_preflight"] = {
                "measured_fraction": 0.14203104592662785,
                "minimum_fraction": 0.2,
                "certificate_sha256": (
                    "dc31e57fb13b6a554bd572f64898b5245d19b45ceabb8af5c9ba041fa732b750"
                ),
                "decision": "rejected_before_scientific_training",
            }
        path = (
            CONFIG_DIR
            / "y400_mai_v3_124m_fullattn_plus_mlp_cproj_"
            f"{label}_muonchart154_0p5tpp_lr24e4.json"
        )
        outputs[path] = config
    return outputs


def main() -> None:
    for path, payload in build().items():
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"{sha256(path)}  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
