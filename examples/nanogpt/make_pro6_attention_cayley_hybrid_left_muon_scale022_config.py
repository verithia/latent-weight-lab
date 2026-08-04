#!/usr/bin/env python3
"""Register the geometry-calibrated 124M hybrid Cayley screen."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CONFIG_DIR = Path(__file__).resolve().parent / "configs"
SOURCE_NAME = (
    "pro6_mai_v3_124m_fullattn_cayley_horizon_capacity_qk32_v16_"
    "cproj8_targeted_bilateral_hybrid_left_muon_0p5tpp_lr24e4.json"
)
DESTINATION_NAME = (
    "pro6_mai_v3_124m_fullattn_cayley_horizon_capacity_qk32_v16_"
    "cproj8_targeted_bilateral_hybrid_left_muon_scale0p22_0p5tpp_lr24e4.json"
)
PRO6_ROOT = Path("/home/pro6000-9980x/MappingNetworks")


def make_config(source: dict[str, Any]) -> dict[str, Any]:
    config = dict(source)
    stem = Path(DESTINATION_NAME).stem
    config.update(
        {
            "block_fht_attn_cayley_muon_lr_scale": 0.22,
            "out_dir": str(
                PRO6_ROOT
                / "outputs/pro6_mai_v3_attention_hybrid_left_muon_scale022"
                / stem
            ),
            "host_transfer_source_config": (
                f"examples/nanogpt/configs/{SOURCE_NAME}"
            ),
            "host_transfer_policy": (
                "change only the left-factor Muon LR scale from 1.0 to 0.22; "
                "right-factor AdamW and every architecture, rank, seed, decoder, "
                "schedule, data, and non-Cayley optimizer setting remain fixed"
            ),
            "hpo_stage": "attention_cayley_hybrid_left_muon_scale022_124m_0p5tpp",
            "ladder_slot": "qk32_v16_cproj8_hybrid_left_muon_scale022",
            "confirmation_slot": "qk32_v16_cproj8_hybrid_left_muon_scale022",
            "confirmation_source": (
                "unit-scale hybrid controlled right-basis drift to 21.50 degrees "
                "but produced a 166.95-degree Cayley angle; 0.22 is the "
                "precomputed ratio tan(126/2)/9.0603 rounded to two significant "
                "figures, targeting the matched AdamW angle"
            ),
            "candidate_scope": (
                "Keep the left-Muon/right-AdamW split and change only the "
                "left-factor Muon scale to the geometry-calibrated 0.22. No "
                "dense learned basis, additive adapter, LoRA branch, or "
                "post-launch scale selection is admitted."
            ),
            "factor_optimizer_policy": (
                "left factors use Muon with 0.22*sqrt(rank) LR scaling; right "
                "factors retain the exact existing full-Cayley-LR AdamW group"
            ),
            "operator_override": {
                "accepted_as_formal_dense_fit_conditioned_result": False,
                "reason": "single geometry-calibrated saturation correction",
                "recorded_at": "2026-08-04",
                "scope": "124M/0.5TPP attention hybrid scale-0.22 screen"
            },
            "screen_only_resolution": (
                "promote only if stable and terminal val <=5.3924; otherwise "
                "close the Cayley optimizer branch without a scale sweep"
            )
        }
    )
    return config


def main() -> None:
    source = json.loads((CONFIG_DIR / SOURCE_NAME).read_text(encoding="utf-8"))
    path = CONFIG_DIR / DESTINATION_NAME
    path.write_text(
        json.dumps(make_config(source), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(path)


if __name__ == "__main__":
    main()
