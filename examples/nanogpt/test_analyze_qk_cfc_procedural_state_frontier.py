from examples.nanogpt.analyze_qk_cfc_procedural_state_frontier import (
    classify,
    coordinate_count,
    minimum_stored_byte_ratio,
    summarize_stage,
)


def _row(stage: int, all_recovery: float, late_minimum: float):
    return {
        "stage_count": stage,
        "all": {"energy_weighted_recovery": all_recovery},
        "late": {"minimum_layer_recovery": late_minimum},
    }


def test_coordinate_and_storage_accounting() -> None:
    assert coordinate_count(6) == 405_504
    assert coordinate_count(12) == 811_008
    assert minimum_stored_byte_ratio(6) == 0.2578125
    assert minimum_stored_byte_ratio(12) == 0.515625


def test_stage_summary() -> None:
    result = summarize_stage([0.2, 1.0], [1.0, 3.0], [0, 1])
    assert result["energy_weighted_recovery"] == 0.8
    assert result["minimum_layer_recovery"] == 0.2


def test_classify_moderate_dense_and_unreachable() -> None:
    rule = {
        "minimum_aggregate_recovery": 0.8,
        "minimum_late_layer_recovery": 0.7,
        "maximum_credible_stage_count": 12,
        "threshold_changes_after_measurement": False,
    }
    moderate = {
        name: [_row(6, 0.4, 0.3), _row(12, 0.9, 0.8)]
        for name in ("momentum_buffer", "compression_residual")
    }
    assert classify(moderate, rule)["classification"] == "MODERATE_PROCEDURAL_STATE_BUDGET_PLAUSIBLE"
    dense = {
        name: [_row(12, 0.7, 0.6), _row(18, 0.9, 0.8)]
        for name in ("momentum_buffer", "compression_residual")
    }
    assert classify(dense, rule)["classification"] == "PROCEDURAL_STATE_REQUIRES_DENSE_SCALE_BUDGET"
    unreachable = {
        name: [_row(12, 0.7, 0.6), _row(30, 0.79, 0.9)]
        for name in ("momentum_buffer", "compression_residual")
    }
    assert classify(unreachable, rule)["classification"] == "PROCEDURAL_STATE_UNREACHABLE_AT_MAXIMUM_TESTED_BUDGET"
