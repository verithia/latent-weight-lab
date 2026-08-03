from __future__ import annotations

import json
from pathlib import Path


PLAN = (
    Path(__file__).parent
    / "configs"
    / "selection_artifacts"
    / "124m_mlp_dense_oracle_gap_plan.json"
)


def test_dense_oracle_plan_uses_new_disjoint_powered_windows() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    protocol = plan["protocol"]
    seeds = [
        protocol["train_seed"],
        *protocol["output_seeds"],
        *protocol["validation_seeds"],
    ]
    assert len(seeds) == len(set(seeds))
    assert min(seeds) > 20260867
    assert protocol["validation_batches_per_window"] == 128
    assert plan["decision_rule"]["confidence_z"] == 2.576
    assert plan["execution"]["parameter_updates_to_checkpoint"] == 0
    assert "No training run" in plan["authorization"]
