from __future__ import annotations

import math

from examples.nanogpt.make_pro6_attention_paper_activation_oracle_plan import (
    build_plan,
)


def test_plan_freezes_theory_gate_without_training_authority() -> None:
    plan = build_plan()
    protocol = plan["protocol"]
    assert protocol["parameter_updates"] == 0
    assert protocol["heldout_steps"] == [1782, 2372]
    assert protocol["activation_scale_multiplier"] == math.sqrt(10.0 / 9.0)
    assert protocol["targets"]["v"]["seed_stride"] == 8
    assert protocol["targets"]["cproj"]["seed_stride"] == 4
    assert plan["decision_rule"]["thresholds"] == {
        "functional_image_recovery_minimum": 0.8,
        "activated_tangent_recovery_minimum": 0.8,
        "activation_gain_over_identity_minimum": 0.05,
    }
    assert not any(plan["authorization"].values())
    assert plan["execution"]["watchdog"] is False
