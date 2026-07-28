#!/usr/bin/env python3
"""Generate lower-metric follow-ups for the cached four-stage MLP chart."""

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
    "blockorth32_s4_c16_g4_init0p125_cachevjp_"
    "muonchart154_0p5tpp_lr24e4.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict[Path, dict]:
    base = json.loads(BASE.read_text())
    outputs: dict[Path, dict] = {}
    for coordinate_scale in (4, 8):
        label = (
            "blockorth32_s4_"
            f"c{coordinate_scale}_g4_init0p125_cachevjp"
        )
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
                "hpo_stage": f"post_manifold_cproj_{label}_0p5tpp",
                "ladder_slot": f"cproj_{label}",
                "out_dir": (
                    "/root/userdata/MappingNetworks/outputs/"
                    f"y400_mai_v3_mlp_blockorth_screens/{run_name}"
                ),
                "candidate_scope": (
                    "four-stage fixed-basis block-orthogonal c_proj chart "
                    "with exact once-per-step cache VJP, measured residual "
                    "scale initialization, and reduced rotation-coordinate "
                    f"metric {coordinate_scale}"
                ),
                "mfu_preflight_certificate": (
                    "/root/userdata/MappingNetworks/outputs/"
                    "y400_mai_v3_mlp_blockorth_screens/"
                    f"performance_preflight_{label}.json"
                ),
                "recipe_resolution_stage": (
                    "post_four_stage_metric16_early_overrotation_bracket"
                ),
                "recipe_resolution_dependency": (
                    "four-stage metric-16 fixed validation at steps 60/120, "
                    "exact cache/live gradient equivalence, and exact "
                    "host-local >=20% MFU gate"
                ),
                "four_stage_metric_followup_evidence": {
                    "metric16_step60_validation_loss": 6.4090,
                    "metric16_step120_validation_loss": 5.9534,
                    "two_stage_metric16_step60_validation_loss": 6.3740,
                    "two_stage_metric16_step120_validation_loss": 5.9292,
                    "tested_coordinate_scale": float(coordinate_scale),
                    "decision": (
                        "retain the oracle-supported four-stage composition "
                        "and lower only its rotation-coordinate metric"
                    ),
                },
            }
        )
        structure = dict(config["manifold_structure"])
        structure["rotation_coordinate_scale"] = float(coordinate_scale)
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
