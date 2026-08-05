#!/usr/bin/env python3
"""Build the preregistered attention c-proj capacity intervention."""

from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PARENT = (
    ROOT
    / "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_qk_cproj_only_qk64_outputgain_5tpp_lr24e4.json"
)
OUTPUT = (
    ROOT
    / "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_qk_cproj_only_qk64_outputgain_cprojratio10_5tpp_lr24e4.json"
)
QK = "attn.c_attn.qk_headwise"
PROJECTION = "attn.c_proj"


def build(parent: dict[str, object]) -> dict[str, object]:
    """Change only c-proj fixed-chart capacity plus execution metadata."""
    if parent.get("block_fht_targets") != [QK, PROJECTION]:
        raise ValueError("parent must be the registered QK+c_proj/dense-V control")
    if parent.get("block_fht_latent_ratio") != 0.01:
        raise ValueError("parent global latent ratio must be 0.01")
    if parent.get("block_fht_latent_ratios") is not None:
        raise ValueError("parent must not already override target latent ratios")

    config = copy.deepcopy(parent)
    config.update(
        {
            "block_fht_latent_ratios": {QK: 0.01, PROJECTION: 0.10},
            "candidate_scope": (
                "From-scratch QK+c_proj partial attention replacement with "
                "dense V. Change only the fixed BlockFHT c_proj latent ratio "
                "from 0.01 to 0.10; retain QK ratio 0.01, QK64/c_proj8 "
                "Cayley charts, output gains, optimizer, schedule, data, and "
                "all seeds. This is a single preregistered capacity "
                "intervention, not a ratio sweep."
            ),
            "confirmation_slot": "qk_cproj_only_cprojratio10_5tpp",
            "hpo_stage": "attention_cproj_fixed_chart_capacity_124m_5tpp",
            "ladder_role": "attention_cproj_capacity_causal_test",
            "ladder_slot": "qk_cproj_only_cprojratio10",
            "launch_ready": False,
            "launch_block_reason": (
                "Await the terminal QK+c_proj/dense-V factorial result. "
                "Authorize only if c_proj is the dominant positive Shapley "
                "component or the V-by-c_proj interaction is material."
            ),
            "operator_override": {
                "accepted_as_formal_dense_fit_conditioned_result": False,
                "reason": (
                    "One fixed-ratio causal capacity intervention selected "
                    "from the measured tight-frame random-share behavior."
                ),
                "recorded_at": "2026-08-05",
                "scope": "124M/5TPP attention c_proj 10% latent-ratio gate",
            },
            "out_dir": (
                "/mnt/ssd-data/orj/MappingNetworks/outputs/"
                "pro6_mai_v3_attention_cproj_capacity/"
                "pro6_mai_v3_124m_qk_cproj_only_qk64_outputgain_"
                "cprojratio10_5tpp_lr24e4"
            ),
            "practical_equivalence_policy": (
                "Do not launch until the registered factorial gate selects "
                "this branch. Then require a fresh exact-config MFU >=20%, "
                "finite preflight losses, and a verified persistent aggregate "
                "watchdog with 20/50/100 callbacks and a 90-minute heartbeat."
            ),
            "resolved_from_template": str(PARENT.relative_to(ROOT)),
        }
    )
    return config


def main() -> None:
    parent = json.loads(PARENT.read_text())
    OUTPUT.write_text(
        json.dumps(build(parent), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
