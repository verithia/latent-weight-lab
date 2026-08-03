from __future__ import annotations

import json
from pathlib import Path


PLAN = (
    Path(__file__).parent
    / "configs"
    / "selection_artifacts"
    / "124m_mlp_joint_block_output_metric_plan.json"
)


def test_metric_plan_has_disjoint_new_seeds_and_zero_update_authorization() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    protocol = plan["protocol"]
    seeds = [
        protocol["train_seed"],
        *protocol["metric_seeds"],
        *protocol["validation_seeds"],
    ]
    assert len(seeds) == len(set(seeds))
    assert min(seeds) > 20260857
    assert protocol["validation_batches_per_window"] == 128
    assert plan["decision_rule"]["confidence_z"] == 2.576
    assert plan["execution"]["parameter_updates_to_checkpoint"] == 0
    assert "No training run" in plan["authorization"]
