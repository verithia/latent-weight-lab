#!/usr/bin/env python3
"""Build preregistered 124M/5TPP partial attention controls for PRO6."""

from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "examples/nanogpt/configs/pro6_mai_v3_124m_fullattn_targeted_bilateral_fullcayleylr_qk64_outputgain_5tpp_lr24e4.json"
OUTPUT = ROOT / "examples/nanogpt/configs/pro6_mai_v3_124m_qk_only_qk64_outputgain_5tpp_lr24e4.json"
QK = "attn.c_attn.qk_headwise"


def build(parent: dict[str, object]) -> dict[str, object]:
    config = copy.deepcopy(parent)
    config.update(
        {
            "block_fht_targets": [QK],
            "block_fht_attn_cayley_targets": [QK],
            "block_fht_attn_cayley_output_targets": [QK],
            "block_fht_attn_cayley_bilateral_targets": [QK],
            "block_fht_attn_cayley_ranks": {QK: 64},
            "block_fht_output_gain_targets": [QK],
            "candidate_scope": (
                "From-scratch QK-only partial attention replacement: keep "
                "V and c_proj dense and Muon-trained while holding QK64, "
                "output gain, schedule, data, seeds, and optimizer recipe "
                "fixed. This is a localization/fallback control, not a full "
                "attention replacement claim."
            ),
            "confirmation_slot": "qk_only_qk64_outputgain_5tpp",
            "confirmation_source": (
                "Dense 5TPP replay val 3.5402 and full-attention QK64 plus "
                "output-gain val 3.6151; endpoint and local teacher swaps "
                "were invalidated by co-adaptation."
            ),
            "hpo_stage": "attention_partial_matrix_localization_124m_5tpp",
            "ladder_role": "attention_partial_replacement_localization",
            "ladder_slot": "qk_only_qk64_outputgain",
            "operator_override": {
                "accepted_as_formal_dense_fit_conditioned_result": False,
                "reason": (
                    "Causal from-scratch localization after full and local "
                    "endpoint swaps proved non-portable."
                ),
                "recorded_at": "2026-08-05",
                "scope": "124M/5TPP QK-only partial attention control",
            },
            "out_dir": (
                "/mnt/ssd-data/orj/MappingNetworks/outputs/"
                "pro6_mai_v3_attention_partial_controls/"
                "pro6_mai_v3_124m_qk_only_qk64_outputgain_5tpp_lr24e4"
            ),
            "parent_dense_token_equivalent_penalty": 1.3707,
            "practical_equivalence_policy": (
                "Direct foreground polling with no watchdog; require fresh "
                "exact-config MFU >=20%, finite fixed evaluations, and "
                "terminal comparison to dense val 3.5402 and full-attention "
                "replacement val 3.6151."
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
