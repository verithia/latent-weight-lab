from __future__ import annotations

import json

from examples.nanogpt.make_pro6_124m_sparse_moe_20tpp_seed2_trajectory import (
    BASE,
    make_seed2_trajectory,
)


def test_seed2_changes_only_identity_and_geometry_acquisition_fields() -> None:
    source = json.loads(BASE.read_text())
    candidate = make_seed2_trajectory(source)
    mutable = {
        "confirmation_slot",
        "experiment_role",
        "hpo_stage",
        "launch_ready",
        "mfu_preflight_certificate",
        "model_seed",
        "monitoring_policy",
        "out_dir",
        "promotion_source",
        "train_data_seed",
        "trajectory_snapshot_dtype",
        "trajectory_snapshot_interval",
        "trajectory_snapshot_layers",
        "trajectory_snapshot_targets",
    }
    assert {key: value for key, value in candidate.items() if key not in mutable} == {
        key: value for key, value in source.items() if key not in mutable
    }


def test_seed2_preserves_scientific_control_and_stays_blocked() -> None:
    candidate = make_seed2_trajectory(json.loads(BASE.read_text()))
    assert candidate["launch_ready"] is False
    assert candidate["model_seed"] == 1338
    assert candidate["train_data_seed"] == 20260715
    assert candidate["max_iters"] == 9495
    assert candidate["learning_rate"] == 0.0024
    assert candidate["moe_num_experts"] == 8
    assert candidate["moe_top_k"] == 2
    assert candidate["trajectory_snapshot_interval"] == 594
    assert candidate["trajectory_snapshot_dtype"] == "bfloat16"
    assert candidate["trajectory_snapshot_layers"] == [0, 5, 11]
    assert candidate["trajectory_snapshot_targets"] == [
        "mlp.expert_c_fc",
        "mlp.expert_c_proj",
        "mlp.router",
    ]
