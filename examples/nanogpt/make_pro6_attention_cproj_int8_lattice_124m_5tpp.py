#!/usr/bin/env python3
"""Generate the preregistered 124M/5TPP attention c_proj int8-lattice transfer."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "examples/nanogpt/configs/pro6_mai_v3_124m_qkv_cproj_int8_lattice_0p5tpp_lr24e4.json"
OUTPUT = ROOT / "examples/nanogpt/configs/pro6_mai_v3_124m_qkv_cproj_int8_lattice_5tpp_lr24e4.json"


def build_config() -> dict[str, object]:
    config = json.loads(BASE.read_text())
    config.update(
        {
            "candidate_scope": (
                "From-scratch 124M/5TPP horizon transfer of the smallest-rung "
                "full-attention persistent-state result: QK64 and V16 retain "
                "their selected Cayley mappings and attention c_proj retains "
                "the causal block-4096 signed-int8 running-max lattice."
            ),
            "confirmation_slot": "qkv_plus_cproj_int8_lattice_5tpp",
            "confirmation_source": (
                "The sealed 124M/0.5TPP int8-lattice result reached validation "
                "CE 5.2917 versus dense 5.4890 and passed its frozen 5.5890 gate."
            ),
            "eval_interval": 594,
            "hpo_stage": "attention_cproj_int8_lattice_124m_5tpp_transfer",
            "ladder_role": "attention_cproj_int8_lattice_same_size_horizon_transfer",
            "launch_block_reason": None,
            "launch_ready": True,
            "lr_decay_iters": 2373,
            "max_iters": 2373,
            "monitoring_policy": (
                "foreground-poll the <=5 minute exact-config MFU gate; after "
                "a pass, use one idempotent terminal/error-only @Codex "
                "watchdog for the 1-2 hour scientific run"
            ),
            "out_dir": (
                "/home/pro6000-9980x/MappingNetworks/outputs/"
                "pro6_mai_v3_124m_qkv_cproj_int8_lattice_5tpp_lr24e4/scientific"
            ),
            "planned_tokens": 621868800,
            "planned_tpp": 5.0,
            "practical_equivalence_nll": 0.02,
            "practical_equivalence_policy": (
                "At terminal require finite fixed-window validation CE <=3.5602, "
                "i.e. no more than +0.0200 behind ordinary dense 124M/5TPP CE "
                "3.5402. Threshold is frozen before MFU or training."
            ),
            "recipe_resolution_dependency": (
                "sealed 124M/0.5TPP int8-lattice PASS at validation CE 5.2917"
            ),
            "recipe_resolution_stage": "same_size_horizon_transfer_only",
            "scheduled_tokens": 622067712,
            "scheduled_tpp": 5.001599308407175,
            "selection_endpoint": (
                "terminal step-2373 fixed-window validation CE versus ordinary "
                "dense 3.5402, QK+V/dense-c_proj 3.5148, and prior full-attention "
                "continuous-chart 3.6151"
            ),
            "warmup_iters": 23,
        }
    )
    return config


def main() -> None:
    OUTPUT.write_text(json.dumps(build_config(), indent=2, sort_keys=True) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
