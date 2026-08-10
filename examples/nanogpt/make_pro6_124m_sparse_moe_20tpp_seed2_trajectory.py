#!/usr/bin/env python3
"""Prepare the frozen second-seed sparse-MoE geometry confirmation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "examples/nanogpt/configs"
BASE = CONFIG_DIR / "pro6_mai_v3_124m_dense_moe8_top2_20tpp_selected_lr24e4.json"
OUTPUT = CONFIG_DIR / "pro6_mai_v3_124m_dense_moe8_top2_20tpp_seed2_trajectory_lr24e4.json"


def make_seed2_trajectory(source: dict[str, Any]) -> dict[str, Any]:
    config = dict(source)
    config.update(
        {
            "confirmation_slot": "structural_replicate_seed2",
            "experiment_role": (
                "124M-active dense complete-expert sparse-MoE second-seed "
                "20TPP geometry confirmation"
            ),
            "hpo_stage": "moe_124m_active_20tpp_seed2_geometry",
            "launch_ready": False,
            "mfu_preflight_certificate": (
                "/mnt/ssd-data/orj/MappingNetworks/outputs/"
                "pro6_mai_v3_124m_dense_moe8_top2_mfu_20tpp_seed2_trajectory_lr24e4/"
                "performance_preflight.json"
            ),
            "model_seed": 1338,
            "monitoring_policy": (
                "aggregate 20/50/80/100 milestones, errors/stalls, and a "
                "90-minute heartbeat reset by progress callbacks"
            ),
            "out_dir": (
                "/mnt/ssd-data/orj/MappingNetworks/outputs/"
                "pro6_mai_v3_124m_dense_moe_ladder/"
                "lr24e4_20tpp_seed2_geometry"
            ),
            "promotion_source": (
                "examples/nanogpt/configs/selection_artifacts/"
                "124m_sparse_moe_paired_neuron_direction_oracle_plan.json"
            ),
            "train_data_seed": 20260715,
            "trajectory_snapshot_dtype": "bfloat16",
            "trajectory_snapshot_interval": 594,
            "trajectory_snapshot_layers": [0, 5, 11],
            "trajectory_snapshot_targets": [
                "mlp.expert_c_fc",
                "mlp.expert_c_proj",
                "mlp.router",
            ],
        }
    )
    return config


def main() -> None:
    source = json.loads(BASE.read_text())
    OUTPUT.write_text(
        json.dumps(make_seed2_trajectory(source), indent=2, sort_keys=True) + "\n"
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
