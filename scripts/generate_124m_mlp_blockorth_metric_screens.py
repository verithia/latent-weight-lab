#!/usr/bin/env python3
"""Generate residual-scale and rotation-metric follow-ups for the MLP chart."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "examples/nanogpt/configs"
BASE = (
    CONFIG_DIR
    / "y400_mai_v3_124m_fullattn_plus_mlp_cproj_blockorth32_s2_gain4_muonchart154_0p5tpp_lr24e4.json"
)
IMPLEMENTATION_COMMIT = "9a16773af98950969832d85978d3b77b01bc3d85"
SOURCE_HASHES = {
    "examples/nanogpt/model.py": (
        "5026e92c8dd82d9a668de39e6179ba6459274e15e577009caad23feb9fa0b832"
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
RESIDUAL_ARTIFACTS = {
    "csv_sha256": (
        "78842f6589caf601e6b6322c24218b94b475dc7b2dd92e3559a4fde40ebe7976"
    ),
    "metadata_sha256": (
        "ffed71bec66dffb326c0309955fe12e14b2e01c4ec7cce0fe3c3a3822064ee6a"
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict[Path, dict]:
    base = json.loads(BASE.read_text())
    outputs: dict[Path, dict] = {}
    for coordinate_scale in (4, 16, 32):
        label = f"blockorth32_s2_c{coordinate_scale}_g4_init0p125"
        run_name = (
            "y400_mai_v3_124m_fullattn_plus_mlp_cproj_"
            f"{label}_muonchart154_0p5tpp"
        )
        config = copy.deepcopy(base)
        config.update(
            {
                "block_fht_mlp_output_block_rotation_coordinate_scale": (
                    float(coordinate_scale)
                ),
                "block_fht_mlp_residual_output_gain_scale": 4.0,
                "block_fht_mlp_residual_output_log_gain_init": 0.125,
                "hpo_stage": f"post_manifold_cproj_{label}_0p5tpp",
                "ladder_role": (
                    "mlp_manifold_residual_metric_screen_provisional"
                ),
                "ladder_slot": f"cproj_{label}",
                "out_dir": (
                    "/root/userdata/MappingNetworks/outputs/"
                    f"y400_mai_v3_mlp_blockorth_screens/{run_name}"
                ),
                "candidate_scope": (
                    "two-stage fixed-basis block-orthogonal c_proj chart with "
                    "measured positive residual-scale initialization and "
                    f"rotation coordinate metric {coordinate_scale}"
                ),
                "residual_metric_evidence": {
                    **RESIDUAL_ARTIFACTS,
                    "terminal_validation_loss": 5.7030,
                    "matched_plain_validation_loss": 5.7395,
                    "attention_validation_loss": 5.4918,
                    "plain_update_to_residual_rms": 0.713214,
                    "charted_update_to_residual_rms": 0.670741,
                    "attention_update_to_residual_rms": 0.810567,
                    "plain_residual_update_cosine": 0.125312,
                    "charted_residual_update_cosine": 0.141834,
                    "attention_residual_update_cosine": 0.165604,
                    "initial_effective_log_gain": 0.125,
                    "initial_multiplier": 1.1331484530668263,
                    "terminal_effective_rotation_coordinate_rms": (
                        0.02558867447078228
                    ),
                    "oracle_rotation_coordinate_rms": 0.4779899179935455,
                    "decision": (
                        "preserve the successful orientation family, restore "
                        "the measured branch-scale deficit at initialization, "
                        "and bracket only the under-traveled chart metric"
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
                    "use the registered AdamW fallback"
                ),
                "recipe_resolution_stage": (
                    "post_terminal_blockorth_smallest_rung_metric_bracket"
                ),
                "recipe_resolution_dependency": (
                    "terminal CE gain, residual-interface shift, measured "
                    "coordinate travel, and exact host-local >=20% MFU gate"
                ),
                "selection_endpoint": (
                    "terminal held-out NLL, residual cosine/parallel energy, "
                    "and update-to-residual RMS versus matched controls"
                ),
                "launch_ready": True,
                "launch_block_reason": None,
            }
        )
        structure = dict(config["manifold_structure"])
        structure.update(
            {
                "rotation_coordinate_scale": float(coordinate_scale),
                "residual_output_log_gain_scale": 4.0,
                "residual_output_effective_log_gain_init": 0.125,
                "initialization": (
                    "identity rotation plus measured 1.133x residual output scale"
                ),
            }
        )
        config["manifold_structure"] = structure
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
