from examples.nanogpt.analyze_attention_layerwise_local_intervention import (
    decide,
    summarize,
)


def test_summary_and_decision_require_replicated_local_gain() -> None:
    rows = []
    for window in ("primary", "confirmation"):
        for layer in range(2):
            for component in ("score", "value", "projection", "value_projection", "all"):
                delta = -0.004 if component == "value_projection" else 0.001
                rows.append(
                    {
                        "window": window,
                        "layer": layer,
                        "component": component,
                        "ratio": 0.03,
                        "ce_delta": delta,
                    }
                )
    summary = summarize(rows, {"primary": 4.0, "confirmation": 4.0})
    decision = decide(
        summary,
        {
            "selection_ratio": 0.03,
            "minimum_replicated_sum_improvement_ce": 0.005,
            "minimum_improving_layer_fraction": 0.5,
        },
    )
    assert decision["classification"] == "REPLICATED_LOCAL_DENSE_DIRECTION"
    assert decision["selected_component"] == "value_projection"
