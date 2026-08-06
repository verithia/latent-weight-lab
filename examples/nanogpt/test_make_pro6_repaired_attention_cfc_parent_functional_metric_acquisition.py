from __future__ import annotations

import json

from examples.nanogpt.make_pro6_repaired_attention_cfc_parent_functional_metric_acquisition import (
    OUTPUT_PLAN,
    PHASE_BOUNDARIES,
    PROBE_LAYERS,
    PROBE_STEPS,
    SOURCE_CONFIG,
    make_config,
    make_plan,
)


def test_acquisition_config_changes_only_observational_side_effects() -> None:
    source = json.loads(SOURCE_CONFIG.read_text())
    candidate = make_config(source)
    assert candidate["max_iters"] == source["max_iters"] == 2373
    assert candidate["eval_interval"] == source["eval_interval"] == 594
    assert candidate["learning_rate"] == source["learning_rate"]
    assert candidate["block_fht_targets"] == source["block_fht_targets"]
    assert candidate["trajectory_snapshot_interval"] == 594
    assert candidate["trajectory_snapshot_all_parameters"] is True
    assert candidate["optimizer_probe_steps"] == PROBE_STEPS
    assert candidate["optimizer_probe_layers"] == PROBE_LAYERS
    assert candidate["optimizer_probe_targets"] == ["mlp.c_proj"]
    assert candidate["diagnostic_acquisition_plan"] == str(
        OUTPUT_PLAN.relative_to(SOURCE_CONFIG.parents[3])
    )


def test_plan_authorizes_acquisition_but_no_candidate() -> None:
    plan = make_plan("a" * 64)
    assert plan["acquisition"]["phase_boundaries"] == PHASE_BOUNDARIES
    assert plan["authorization"]["one_diagnostic_acquisition_after_mfu_pass"]
    assert plan["authorization"]["zero_update_metric_calibration_after_acquisition_pass"]
    assert not plan["authorization"]["candidate_structure_implementation"]
    assert not plan["authorization"]["candidate_language_model_training"]
    assert plan["monitoring"]["milestones"] is False
    assert plan["monitoring"]["heartbeats"] is False
