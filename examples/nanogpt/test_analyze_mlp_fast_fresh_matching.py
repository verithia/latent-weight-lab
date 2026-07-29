from __future__ import annotations

from examples.nanogpt.analyze_mlp_fast_fresh_matching import (
    aggregate_comparison,
)


def make_rows(candidate: str, scale: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for window in ("fit", "holdout"):
        rows.append(
            {
                "endpoint": 0,
                "layer": 0,
                "window": window,
                "candidate": candidate,
                "requested_update_recovery": 0.2 * scale,
                "requested_update_energy": 1.0,
                "output_positive_line_recovery": 0.3 * scale,
                "output_fixed_scale_recovery": 0.2 * scale,
                "target_output_energy": 1.0,
                "train_gradient_predicted_ce_decrease": 0.4 * scale,
                "validation_gradient_predicted_ce_decrease": (
                    0.5 * scale
                ),
            }
        )
    return rows


def finite_rows(single_loss: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for window in ("fit", "holdout"):
        for candidate, loss in (
            ("baseline", 2.0),
            ("greedy", 1.9),
            ("single_pass", single_loss),
            ("random", 1.99),
        ):
            rows.append(
                {
                    "endpoint": 0,
                    "window": window,
                    "candidate": candidate,
                    "loss": loss,
                }
            )
    return rows


def timing_rows(seconds: float) -> list[dict[str, object]]:
    return [
        {
            "endpoint": 0,
            "layer": layer,
            "candidate": candidate,
            "selection_seconds": (
                seconds if candidate == "single_pass" else 1.0
            ),
            "candidate_edge_fraction": 0.95,
        }
        for layer in range(5)
        for candidate in ("greedy", "single_pass")
    ]


def test_aggregate_promotes_retained_fast_matcher() -> None:
    rows = (
        make_rows("greedy", 1.0)
        + make_rows("single_pass", 0.9)
        + make_rows("random", 0.2)
    )
    result = aggregate_comparison(
        rows,
        finite_rows(1.9001),
        timing_rows(0.02),
        minimum_greedy_retention=0.85,
        minimum_random_enrichment=3.0,
        maximum_median_seconds_per_layer=0.03,
        maximum_p95_seconds_per_layer=0.06,
        maximum_mean_finite_ce_regression=0.00025,
    )
    assert (
        result["decision"]
        == "QUALIFY_STATELESS_FRESH_MATCHER_PRODUCTION_PREFLIGHT"
    )
    assert all(result["passes"].values())


def test_aggregate_rejects_slow_or_misaligned_matcher() -> None:
    rows = (
        make_rows("greedy", 1.0)
        + make_rows("single_pass", 0.7)
        + make_rows("random", 0.3)
    )
    result = aggregate_comparison(
        rows,
        finite_rows(1.902),
        timing_rows(0.08),
        minimum_greedy_retention=0.85,
        minimum_random_enrichment=3.0,
        maximum_median_seconds_per_layer=0.03,
        maximum_p95_seconds_per_layer=0.06,
        maximum_mean_finite_ce_regression=0.00025,
    )
    assert result["decision"] == "REJECT_STATELESS_FRESH_MATCHER"
    assert not all(result["passes"].values())
