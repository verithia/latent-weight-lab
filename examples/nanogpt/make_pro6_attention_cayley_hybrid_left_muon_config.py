#!/usr/bin/env python3
"""Register the 124M Cayley left-Muon/right-AdamW causal screen."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CONFIG_DIR = Path(__file__).resolve().parent / "configs"
SOURCE_NAME = (
    "y400_mai_v3_124m_fullattn_cayley_horizon_capacity_qk32_v16_"
    "cproj8_targeted_bilateral_fullcayleylr_0p5tpp_lr24e4.json"
)
DESTINATION_NAME = (
    "pro6_mai_v3_124m_fullattn_cayley_horizon_capacity_qk32_v16_"
    "cproj8_targeted_bilateral_hybrid_left_muon_0p5tpp_lr24e4.json"
)
PRO6_ROOT = Path("/home/pro6000-9980x/MappingNetworks")


def make_config(source: dict[str, Any]) -> dict[str, Any]:
    config = dict(source)
    stem = Path(DESTINATION_NAME).stem
    config.update(
        {
            "block_fht_attn_cayley_factor_optimizer": "hybrid_left_muon",
            "block_fht_attn_cayley_muon_lr_scale": 1.0,
            "data_dir": str(PRO6_ROOT / "data/finewebedu_20b"),
            "out_dir": str(
                PRO6_ROOT
                / "outputs/pro6_mai_v3_attention_hybrid_left_muon"
                / stem
            ),
            "execution_host": "PRO6",
            "host_transfer_source_config": (
                f"examples/nanogpt/configs/{SOURCE_NAME}"
            ),
            "host_transfer_policy": (
                "change host paths and split only the existing Cayley-factor "
                "optimizer: left factors use norm-matched Muon, right factors "
                "retain the exact registered AdamW fallback; all architecture, "
                "rank, seed, decoder, schedule, data, and other optimizer "
                "settings remain fixed"
            ),
            "hpo_stage": "attention_cayley_hybrid_left_muon_124m_0p5tpp",
            "ladder_role": "attention_basis_optimizer_causal_screen",
            "ladder_slot": "qk32_v16_cproj8_hybrid_left_muon",
            "confirmation_slot": "qk32_v16_cproj8_hybrid_left_muon",
            "confirmation_source": (
                "full factor Muon improved the identity-chart direction oracle "
                "but regressed CE while rotating the learned right frame to "
                "66.79 degrees versus 19.52 degrees for the AdamW control"
            ),
            "candidate_scope": (
                "Use Muon only for each zero-initialized Cayley left factor and "
                "the exact existing AdamW fallback for each learned right frame. "
                "Keep QK/V/c-proj ranks 32/16/8, decoder, chart seeds/sides, "
                "model/data seeds, schedule, fixed evaluations, and MLP fixed. "
                "No dense learned basis, additive adapter, or LoRA branch."
            ),
            "factor_optimizer_policy": (
                "left factors receive Muon with sqrt(rank) norm matching; right "
                "factors receive the existing full-Cayley-LR AdamW group with "
                "zero weight decay"
            ),
            "operator_override": {
                "accepted_as_formal_dense_fit_conditioned_result": False,
                "reason": (
                    "smallest-rung causal isolation of tangent-direction "
                    "optimization from learned-basis optimization"
                ),
                "recorded_at": "2026-08-04",
                "scope": "124M/0.5TPP attention hybrid Cayley optimizer",
            },
            "practical_equivalence_policy": (
                "poll directly with no watchdog; require exact-config MFU >=20%, "
                "finite fixed evaluations, and terminal val <=5.3924"
            ),
            "screen_only": True,
            "screen_only_resolution": (
                "promote only if stable and terminal val <=5.3924; otherwise "
                "reject split factor optimization and keep the QK64+gain parent"
            ),
            "mfu_preflight_required": True,
            "mfu_min_fraction": 0.20,
            "prelaunch_provenance_requirements": (
                "record commit, entrypoint, literal command, archived config "
                "SHA256, source hashes, dataset manifest SHA256, runtime fixed "
                "evaluation digest, and synchronous MFU certificate"
            ),
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
