from __future__ import annotations

from examples.nanogpt.make_pro6_attention_affine_delta_path_oracle_plan import (
    build_plan,
)


def test_plan_freezes_out_of_sample_path_gate() -> None:
    plan = build_plan()
    protocol = plan["protocol"]
    assert protocol["parameter_updates"] == 0
    assert protocol["trajectory_steps"][-1] == 2373
    assert len(protocol["trajectory_steps"]) == 41
    assert protocol["discovery_max_step"] == 1140
    assert protocol["heldout_min_step"] == 1200
    assert protocol["fit_metric_seed"] != protocol["eval_metric_seed"]
    assert plan["decision_rule"]["thresholds"] == {
        "aggregate_recovery_minimum": 0.8,
        "minimum_layer_recovery_minimum": 0.6,
    }
    assert not any(plan["authorization"].values())
    assert plan["execution"]["watchdog"] is False
