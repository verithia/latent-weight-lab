#!/usr/bin/env python3
"""Build preregistered 124M/5TPP partial attention controls for PRO6."""

from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "examples/nanogpt/configs/pro6_mai_v3_124m_fullattn_targeted_bilateral_fullcayleylr_qk64_outputgain_5tpp_lr24e4.json"
OUTPUTS = {
    "qk_only": ROOT
    / "examples/nanogpt/configs/pro6_mai_v3_124m_qk_only_qk64_outputgain_5tpp_lr24e4.json",
    "qkv_only": ROOT
    / "examples/nanogpt/configs/pro6_mai_v3_124m_qkv_only_qk64_outputgain_5tpp_lr24e4.json",
    "qk_cproj_only": ROOT
    / "examples/nanogpt/configs/pro6_mai_v3_124m_qk_cproj_only_qk64_outputgain_5tpp_lr24e4.json",
}
QK = "attn.c_attn.qk_headwise"
VALUE = "attn.c_attn.v"
PROJECTION = "attn.c_proj"

SCOPES = {
    "qk_only": [QK],
    "qkv_only": [QK, VALUE],
    "qk_cproj_only": [QK, PROJECTION],
}


def build(parent: dict[str, object], scope: str = "qk_only") -> dict[str, object]:
    if scope not in SCOPES:
        raise ValueError(f"unknown partial-attention scope: {scope}")
    targets = SCOPES[scope]
    kept_dense = {
        "qk_only": "V and c_proj",
        "qkv_only": "c_proj",
        "qk_cproj_only": "V",
    }[scope]
    display_scope = {
        "qk_only": "QK-only",
        "qkv_only": "QK+V",
        "qk_cproj_only": "QK+c_proj",
    }[scope]
    config = copy.deepcopy(parent)
    config.update(
        {
            "block_fht_targets": targets,
            "block_fht_attn_cayley_targets": targets,
            "block_fht_attn_cayley_output_targets": [
                target for target in targets if target in (QK, PROJECTION)
            ],
            "block_fht_attn_cayley_bilateral_targets": [
                target for target in targets if target in (QK, VALUE)
            ],
            "block_fht_attn_cayley_ranks": {
                target: {QK: 64, VALUE: 16, PROJECTION: 8}[target]
                for target in targets
            },
            "block_fht_output_gain_targets": targets,
            "candidate_scope": (
                f"From-scratch {display_scope} partial attention "
                f"replacement: keep {kept_dense} dense and Muon-trained "
                "while holding QK64, "
                "output gain, schedule, data, seeds, and optimizer recipe "
                "fixed. This is a localization/fallback control, not a full "
                "attention replacement claim."
            ),
            "confirmation_slot": f"{scope}_qk64_outputgain_5tpp",
            "confirmation_source": (
                "Dense 5TPP replay val 3.5402 and full-attention QK64 plus "
                "output-gain val 3.6151; endpoint and local teacher swaps "
                "were invalidated by co-adaptation."
            ),
            "hpo_stage": "attention_partial_matrix_localization_124m_5tpp",
            "ladder_role": "attention_partial_replacement_localization",
            "ladder_slot": f"{scope}_qk64_outputgain",
            "operator_override": {
                "accepted_as_formal_dense_fit_conditioned_result": False,
                "reason": (
                    "Causal from-scratch localization after full and local "
                    "endpoint swaps proved non-portable."
                ),
                "recorded_at": "2026-08-05",
                "scope": f"124M/5TPP {display_scope} partial attention control",
            },
            "out_dir": (
                "/mnt/ssd-data/orj/MappingNetworks/outputs/"
                "pro6_mai_v3_attention_partial_controls/"
                f"pro6_mai_v3_124m_{scope}_qk64_outputgain_5tpp_lr24e4"
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
    for scope, output in OUTPUTS.items():
        output.write_text(
            json.dumps(build(parent, scope), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
