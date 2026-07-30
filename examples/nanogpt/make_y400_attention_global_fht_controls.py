"""Build the matched global-FHT controls for attention replacement.

These controls keep the accepted qk-headwise/V/c-proj target split, latent
ratios, optimizer, data, and evaluation protocol unchanged.  The sole model
change is replacing repeated local BlockFHT blocks with one variance-matched
global signed-FHT chart per generated matrix.  This isolates locality without
introducing a learned basis or changing the number of trainable coordinates.
"""

from __future__ import annotations

import json
from pathlib import Path


CONFIG_DIR = Path(__file__).resolve().parent / "configs"

CONTROLS = (
    (
        "y400_mai_v2_124m_fullattn_blockfht_0p5tpp_mult1p00.json",
        "y400_mai_v3_124m_fullattn_globalfht_0p5tpp_lr24e4.json",
        "y400_mai_v3_124m_fullattn_globalfht_0p5tpp_lr24e4",
        "global_fht_locality_control_0p5tpp",
    ),
    (
        "y400_mai_v2_124m_fullattn_blockfht_5tpp_top1.json",
        "y400_mai_v3_124m_fullattn_globalfht_5tpp_lr24e4.json",
        "y400_mai_v3_124m_fullattn_globalfht_5tpp_lr24e4",
        "global_fht_locality_control_5tpp",
    ),
)


def main() -> None:
    for base_name, output_name, run_name, stage in CONTROLS:
        base_path = CONFIG_DIR / base_name
        config = json.loads(base_path.read_text())
        config.update(
            {
                "block_fht_global_output": True,
                "out_dir": (
                    "/root/userdata/MappingNetworks/outputs/"
                    "y400_mai_v3_attention_global_fht/"
                    f"{run_name}"
                ),
                "recipe_resolution_dependency": (
                    "matched repeated-BlockFHT attention control; this "
                    "candidate changes only fixed chart locality"
                ),
                "recipe_resolution_stage": stage,
                "resolved_from_template": base_name,
                "selection_endpoint": (
                    "terminal held-out NLL on the same fixed eval windows; "
                    "compare only against the matched repeated-BlockFHT run"
                ),
            }
        )
        output_path = CONFIG_DIR / output_name
        output_path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n"
        )
        print(output_path)


if __name__ == "__main__":
    main()
