import torch

from examples.nanogpt.analyze_qk_cfc_state_side_decomposition import (
    classify,
    full_input_action_projection,
)


def test_input_projection_separates_column_space() -> None:
    source = torch.tensor([[[1.0], [0.0]]])
    inside = torch.tensor([[[3.0], [0.0]]])
    outside = torch.tensor([[[0.0], [2.0]]])
    assert torch.allclose(full_input_action_projection(source, inside), inside)
    assert torch.allclose(
        full_input_action_projection(source, outside), torch.zeros_like(outside)
    )


def _metrics(value: float):
    return {
        "all": {"energy_weighted_recovery": value},
        "late": {"minimum_layer_recovery": value},
    }


def test_classification_prefers_input_then_bilateral() -> None:
    families = (
        "output6",
        "input_full",
        "output6_then_input",
        "input_then_output6",
    )
    rule = {
        "minimum_aggregate_recovery": 0.8,
        "minimum_late_layer_recovery": 0.7,
        "candidate_priority": [
            "input_full",
            "output6_then_input",
            "input_then_output6",
        ],
        "threshold_changes_after_measurement": False,
    }
    aggregate = {
        state: {family: _metrics(0.9 if family != "output6" else 0.2) for family in families}
        for state in ("momentum_buffer", "compression_residual")
    }
    result = classify(aggregate, rule)
    assert result["classification"] == "INPUT_SIDE_TEMPORAL_STATE_PLAUSIBLE"
    for state in aggregate:
        aggregate[state]["input_full"] = _metrics(0.4)
    result = classify(aggregate, rule)
    assert result["classification"] == "BILATERAL_TEMPORAL_STATE_PLAUSIBLE"
    for state in aggregate:
        aggregate[state]["output6_then_input"] = _metrics(0.4)
        aggregate[state]["input_then_output6"] = _metrics(0.4)
    assert classify(aggregate, rule)["classification"] == "WEIGHT_RELATIVE_TEMPORAL_STATE_INSUFFICIENT"
