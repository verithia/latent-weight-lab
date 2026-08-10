#!/usr/bin/env python3
"""Materialize the preregistered 124M-active sparse-MoE 20TPP winner."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "examples/nanogpt/configs"
BASE = CONFIG_DIR / "pro6_mai_v3_124m_dense_moe8_top2_5tpp_top1_lr24e4.json"
OUTPUT = CONFIG_DIR / "pro6_mai_v3_124m_dense_moe8_top2_20tpp_selected_lr24e4.json"


def make_selected_20tpp(source: dict[str, Any]) -> dict[str, Any]:
    config = dict(source)
    config.update(
        {
            "confirmation_slot": "selected_20tpp",
            "eval_interval": 2374,
            "experiment_role": (
                "124M-active dense complete-expert sparse-MoE selected 20TPP "
                "horizon confirmation"
            ),
            "hpo_stage": "moe_124m_active_selected_20tpp",
            "lr_decay_iters": 9495,
            "max_iters": 9495,
            "mfu_preflight_certificate": (
                "/mnt/ssd-data/orj/MappingNetworks/outputs/"
                "pro6_mai_v3_124m_dense_moe8_top2_mfu_20tpp_selected_lr24e4/"
                "performance_preflight.json"
            ),
            "monitoring_policy": (
                "estimated over two hours; aggregate 20/50/80/100 milestones, "
                "errors/stalls, and 90-minute heartbeat reset by progress callbacks"
            ),
            "out_dir": (
                "/mnt/ssd-data/orj/MappingNetworks/outputs/"
                "pro6_mai_v3_124m_dense_moe_ladder/lr24e4_20tpp_selected"
            ),
            "planned_tokens": 2488949760,
            "planned_tpp_active": 20.0,
            "promotion_source": (
                "examples/nanogpt/configs/selection_artifacts/"
                "124m_sparse_moe_5tpp_ranking.json rank 1"
            ),
            "scheduled_tokens": 2489057280,
            "scheduled_tpp_active": 20.000863978869546,
            "warmup_iters": 94,
        }
    )
    return config


def main() -> None:
    source = json.loads(BASE.read_text())
    OUTPUT.write_text(
        json.dumps(make_selected_20tpp(source), indent=2, sort_keys=True) + "\n"
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
