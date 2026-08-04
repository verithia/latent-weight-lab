#!/usr/bin/env python3
"""Register the causal 124M attention-capacity closure factorial."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CONFIG_DIR = Path(__file__).resolve().parent / "configs"
BASE_NAME = (
    "y400_mai_v3_124m_fullattn_cayley_horizon_capacity_qk32_v16_"
    "cproj8_targeted_bilateral_fullcayleylr_5tpp_lr24e4.json"
)
OUTPUT_ROOT = (
    "/root/userdata/MappingNetworks/outputs/"
    "y400_mai_v3_attention_capacity_closure"
)
VARIANTS: dict[str, dict[str, Any]] = {
    "qk64": {
        "ranks": {
            "attn.c_attn.qk_headwise": 64,
            "attn.c_attn.v": 16,
            "attn.c_proj": 8,
        },
        "output_gain": False,
        "question": (
            "Does raising only QK from pair rank 32 to 64 close the gap, "
            "as required to exceed 90% of the terminal selected QK orbit?"
        ),
    },
    "outputgain": {
        "ranks": {
            "attn.c_attn.qk_headwise": 32,
            "attn.c_attn.v": 16,
            "attn.c_proj": 8,
        },
        "output_gain": True,
        "question": (
            "Does per-output-channel radial freedom close the gap left by "
            "singular-value-preserving Cayley rotations?"
        ),
    },
    "qk64_outputgain": {
        "ranks": {
            "attn.c_attn.qk_headwise": 64,
            "attn.c_attn.v": 16,
            "attn.c_proj": 8,
        },
        "output_gain": True,
        "question": (
            "Is the remaining gap an interaction between QK orbit capacity "
            "and missing radial/singular-value freedom?"
        ),
    },
}


def destination_name(slot: str) -> str:
    return (
        "y400_mai_v3_124m_fullattn_targeted_bilateral_fullcayleylr_"
        f"{slot}_5tpp_lr24e4.json"
    )


def make_config(slot: str, specification: dict[str, Any]) -> dict[str, Any]:
    source = json.loads((CONFIG_DIR / BASE_NAME).read_text(encoding="utf-8"))
    config = dict(source)
    gains = (
        [
            "attn.c_attn.qk_headwise",
            "attn.c_attn.v",
            "attn.c_proj",
        ]
        if specification["output_gain"]
        else []
    )
    stem = Path(destination_name(slot)).stem
    config.update(
        {
            "block_fht_attn_cayley_ranks": specification["ranks"],
            "block_fht_output_gain_targets": gains,
            "out_dir": f"{OUTPUT_ROOT}/{stem}",
            "execution_host": "Y400",
            "hpo_stage": "attention_practical_gap_capacity_radial_factorial_124m_5tpp",
            "ladder_role": "attention_practical_gap_closure",
            "ladder_slot": slot,
            "confirmation_slot": slot,
            "confirmation_source": (
                "selected targeted-bilateral full-Cayley-LR parent val "
                "3.6278 versus matched dense 3.5401; terminal dense oracle "
                "requires QK pair rank 64 to exceed 90% selected-orbit "
                "recovery, while Cayley rotations cannot alter singular values"
            ),
            "candidate_scope": (
                f"{specification['question']} Hold QK/V/c-proj BlockFHT "
                "targets, targeted bilateral sides, full Cayley learning rate, "
                "data, seeds, schedule, and all MLP matrices fixed. No learned "
                "dense basis, additive residual, LoRA branch, or Mapping Loss."
            ),
            "operator_override": {
                "accepted_as_formal_dense_fit_conditioned_result": False,
                "reason": specification["question"],
                "recorded_at": "2026-08-04",
                "scope": "124M/5TPP attention QK-capacity x radial causal factorial",
            },
            "practical_equivalence_nll": 0.01,
            "practical_equivalence_policy": (
                "poll directly with no long-run watchdog; require exact-config "
                "MFU >=20%, finite four-point fixed evaluation, terminal val "
                "<=3.5501, and dense token-equivalent penalty <=1.10x; rank "
                "candidates by common-loss horizontal ratio before terminal CE"
            ),
            "screen_only": False,
            "screen_only_resolution": None,
            "mfu_preflight_required": True,
            "mfu_min_fraction": 0.20,
            "compute_equivalence_sop": (
                "notes/active/fixed-model-compute-equivalence-sop-20260804.md"
            ),
            "dense_fixed_validation_curve": [
                {"step": 594, "validation_ce": 4.1739},
                {"step": 1188, "validation_ce": 3.7622},
                {"step": 1782, "validation_ce": 3.6038},
                {"step": 2373, "validation_ce": 3.5401},
            ],
            "parent_fixed_validation_curve": [
                {"step": 594, "validation_ce": 4.2151},
                {"step": 1188, "validation_ce": 3.8411},
                {"step": 1782, "validation_ce": 3.6907},
                {"step": 2373, "validation_ce": 3.6278},
            ],
            "parent_dense_token_equivalent_penalty": 1.416,
            "prelaunch_provenance_requirements": (
                "record commit, entrypoint, literal command, archived config "
                "SHA256, source hashes, dataset manifest SHA256, runtime fixed "
                "evaluation digest, and synchronous MFU certificate"
            ),
        }
    )
    return config


def main() -> None:
    for slot, specification in VARIANTS.items():
        path = CONFIG_DIR / destination_name(slot)
        path.write_text(
            json.dumps(make_config(slot, specification), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        print(path)


if __name__ == "__main__":
    main()
