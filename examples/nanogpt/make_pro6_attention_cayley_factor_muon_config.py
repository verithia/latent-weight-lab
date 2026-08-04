#!/usr/bin/env python3
"""Register the matched 124M/0.5-TPP Cayley-factor Muon screen."""

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
    "cproj8_targeted_bilateral_factor_muon_0p5tpp_lr24e4.json"
)
PRO6_ROOT = Path("/home/pro6000-9980x/MappingNetworks")


def make_config(source: dict[str, Any]) -> dict[str, Any]:
    config = dict(source)
    stem = Path(DESTINATION_NAME).stem
    config.update(
        {
            "block_fht_attn_cayley_factor_optimizer": "muon",
            "block_fht_attn_cayley_muon_lr_scale": 1.0,
            "data_dir": str(PRO6_ROOT / "data/finewebedu_20b"),
            "out_dir": str(
                PRO6_ROOT
                / "outputs/pro6_mai_v3_attention_factor_muon"
                / stem
            ),
            "execution_host": "PRO6",
            "host_transfer_source_config": (
                f"examples/nanogpt/configs/{SOURCE_NAME}"
            ),
            "host_transfer_policy": (
                "host paths plus the preregistered Cayley-factor optimizer "
                "are changed; decoder, ranks, chart sides, seeds, schedule, "
                "data manifest, and all non-Cayley optimizer groups remain fixed"
            ),
            "hpo_stage": "attention_cayley_factor_muon_124m_0p5tpp",
            "ladder_role": "attention_direction_optimizer_screen",
            "ladder_slot": "qk32_v16_cproj8_factor_muon",
            "confirmation_slot": "qk32_v16_cproj8_factor_muon",
            "confirmation_source": (
                "the exact QK32/V16/c-proj8 dense-direction oracle raises "
                "positive-line recovery 2.556x with no target-family regression, "
                "and this rung has an existing optimizer-matched AdamW control"
            ),
            "candidate_scope": (
                "Change only the fixed targeted-bilateral Cayley factors from "
                "full-main-LR AdamW to Frobenius-norm-matched thin-matrix Muon. "
                "Keep QK/V/c-proj ranks 32/16/8, BlockFHT decoder, model/data "
                "seeds, training schedule, fixed evaluations, and all MLP "
                "matrices unchanged. No learned dense basis, additive residual, "
                "or LoRA branch is admitted."
            ),
            "factor_muon_norm_policy": (
                "each thin factor receives Muon's native aspect-ratio scale "
                "times sqrt(rank), matching the first AdamW factor-update "
                "Frobenius norm while changing direction geometry"
            ),
            "operator_override": {
                "accepted_as_formal_dense_fit_conditioned_result": False,
                "reason": (
                    "smallest-rung optimizer-isolated causal screen admitted by "
                    "the exact QK32 dense-direction oracle"
                ),
                "recorded_at": "2026-08-04",
                "scope": "124M/0.5TPP attention Cayley-factor Muon",
            },
            "practical_equivalence_policy": (
                "poll directly with no watchdog; require exact-config MFU >=20%, "
                "finite fixed evaluations, and terminal val <=5.3924 (at least "
                "0.0100 CE better than matched AdamW control 5.4024)"
            ),
            "screen_only": True,
            "screen_only_resolution": (
                "promote only if stable and terminal val <=5.3924; on promotion "
                "combine factor Muon with the already selected QK64 capacity at "
                "5 TPP, otherwise stop this optimizer line"
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
