from __future__ import annotations

import json
from pathlib import Path


PLAN = Path(__file__).parent / "configs/selection_artifacts/124m_sparse_moe_paired_atom_oracle_plan.json"


def test_paired_atom_budgets_match_preregistered_compression() -> None:
    plan = json.loads(PLAN.read_text())
    families = {item["name"]: item for item in plan["coordinate_families"]}
    target = 18_880_512
    assert families["coupled_four"]["coordinates_per_layer"] == 49_184
    assert families["separate_three_plus_three"]["coordinates_per_layer"] == 73_752
    for family in families.values():
        assert family["target_parameters_per_layer"] == target
        assert abs(target / family["coordinates_per_layer"] - family["compression_ratio"]) < 1e-12
        assert 200.0 <= family["compression_ratio"] <= 500.0


def test_paired_atom_plan_keeps_training_blocked() -> None:
    plan = json.loads(PLAN.read_text())
    assert plan["causal_basis"]["chronological_discovery_transitions"] == 8
    assert plan["causal_basis"]["chronological_heldout_transitions"] == 8
    assert plan["frozen_gates"] == {
        "assignment_overlap_min": 0.75,
        "heldout_exact_recovery_mean_min": 0.9,
        "heldout_exact_recovery_every_layer_min": 0.8,
        "moving_minus_static_mean_min": 0.1,
        "moving_minus_fixed_structured_mean_min": 0.1,
        "all_values_finite": True,
    }
    assert "No generated-expert training" in plan["interpretation"]["authorization"]
