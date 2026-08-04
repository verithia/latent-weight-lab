#!/usr/bin/env python3
"""Register the causal phase-local attention Cayley atlas candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CONFIG_DIR = Path(__file__).resolve().parent / "configs"
BASE_NAME = (
    "pro6_mai_v3_124m_fullattn_cayley_horizon_capacity_qk32_v16_"
    "cproj8_targeted_bilateral_fullcayleylr_5tpp_lr24e4.json"
)
OUTPUT_ROOT = (
    "/home/pro6000-9980x/MappingNetworks/outputs/"
    "pro6_mai_v3_attention_cayley_atlas"
)
VARIANTS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("phase4", (0, 594, 1188, 1782)),
    ("phase3", (0, 594, 1188)),
    ("phase2", (0, 594)),
)


def destination_name(slot: str) -> str:
    return (
        "pro6_mai_v3_124m_fullattn_targeted_bilateral_fullcayleylr_"
        f"cayley_atlas_{slot}_5tpp_lr24e4.json"
    )


def make_config(slot: str, start_steps: tuple[int, ...]) -> dict[str, Any]:
    source = json.loads((CONFIG_DIR / BASE_NAME).read_text(encoding="utf-8"))
    config = dict(source)
    stem = Path(destination_name(slot)).stem
    config.update(
        {
            "block_fht_attn_cayley_atlas_start_steps": list(start_steps),
            "out_dir": f"{OUTPUT_ROOT}/{stem}",
            "hpo_stage": "attention_phase_local_cayley_atlas_124m_5tpp",
            "ladder_role": "attention_practical_gap_closure",
            "ladder_slot": f"cayley_atlas_{slot}",
            "confirmation_slot": f"cayley_atlas_{slot}",
            "confirmation_source": (
                "the horizon-matched dense trajectory is low-dimensional but "
                "strongly curved (attention c_proj path/chord 5.39 and median "
                "turn 82 degrees); QK64 and output gain leave a late-horizon "
                "gap, so test a causal atlas of phase-local fixed-basis orbits"
            ),
            "candidate_scope": (
                "At each registered phase boundary, freeze the previous "
                "targeted-bilateral Cayley chart and activate a fresh "
                "identity-initialized chart with the same fixed-basis "
                "QK/V/c-proj ranks 32/16/8. The represented function is "
                "continuous. BlockFHT decoder, optimizer, data, seed, schedule, "
                "and all MLP matrices remain fixed; no learned dense basis, "
                "additive residual, or LoRA branch is admitted."
            ),
            "atlas_performance_selection": (
                "gate phase4, then phase3, then phase2 sequentially and train "
                "only the first exact config with measured MFU >=20%; never "
                "use validation CE to choose the stage count"
            ),
            "operator_override": {
                "accepted_as_formal_dense_fit_conditioned_result": False,
                "reason": "causal test of horizon-dependent attention-chart curvature",
                "recorded_at": "2026-08-04",
                "scope": "124M/5TPP attention phase-local Cayley atlas",
            },
            "practical_equivalence_nll": 0.01,
            "practical_equivalence_policy": (
                "poll directly with no watchdog; require exact-config MFU "
                ">=20%, finite four-point fixed evaluation, terminal val "
                "<=3.5501, and direct dense token-equivalent penalty <=1.10x"
            ),
            "compute_equivalence_sop": (
                "notes/active/fixed-model-compute-equivalence-sop-20260804.md"
            ),
            "screen_only": False,
            "screen_only_resolution": None,
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
    for slot, start_steps in VARIANTS:
        path = CONFIG_DIR / destination_name(slot)
        path.write_text(
            json.dumps(make_config(slot, start_steps), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        print(path)


if __name__ == "__main__":
    main()
