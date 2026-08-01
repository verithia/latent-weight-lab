from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_cfc_trust_radius import (
    aggregate_validation,
    choose_trust_scale,
    scaled_updates,
)


def test_scaled_updates_preserve_shape_and_do_not_mutate_source() -> None:
    source = {0: torch.tensor([[1.0, -2.0]])}
    scaled = scaled_updates(source, 0.25)
    torch.testing.assert_close(scaled[0], torch.tensor([[0.25, -0.5]]))
    torch.testing.assert_close(source[0], torch.tensor([[1.0, -2.0]]))


def test_choose_scale_uses_fit_only_and_smallest_tie() -> None:
    rows = []
    for repeat in range(3):
        rows.append(
            {
                "phase": "fit",
                "candidate": "baseline",
                "scale": 0.0,
                "loss": 5.0,
            }
        )
        for scale, loss in ((0.25, 4.9), (0.5, 4.90000000005), (1.0, 5.1)):
            rows.append(
                {
                    "phase": "fit",
                    "candidate": "fresh_expansion88",
                    "scale": scale,
                    "loss": loss,
                }
            )
    result = choose_trust_scale(
        rows, minimum_fit_improvement=1e-4, tie_tolerance=1e-8
    )
    assert result["selected_scale"] == 0.25
    assert result["fit_gate_passed"]


def _validation_rows(fresh: float) -> list[dict[str, object]]:
    rows = []
    for window in ("validation_1", "validation_2"):
        for repeat in range(3):
            for candidate, loss in (
                ("baseline", 5.0),
                ("dense_exact", 4.9),
                ("fresh_expansion88", fresh),
                ("random_expansion88", 4.95),
            ):
                rows.append(
                    {
                        "phase": "validation",
                        "window": window,
                        "candidate": candidate,
                        "loss": loss,
                    }
                )
    return rows


def test_validation_gate_requires_fresh_to_beat_every_control_everywhere() -> None:
    result = aggregate_validation(
        _validation_rows(4.8),
        validation_windows=["validation_1", "validation_2"],
        controls=[
            "dense_exact",
            "fresh_expansion88",
            "random_expansion88",
        ],
        numerical_range_tolerance=1e-7,
        minimum_test_margin=1e-5,
    )
    assert result["gates"]["numerically_stable"]
    assert result["gates"][
        "fresh88_beats_baseline_dense_and_random_on_every_window"
    ]
    result = aggregate_validation(
        _validation_rows(4.92),
        validation_windows=["validation_1", "validation_2"],
        controls=[
            "dense_exact",
            "fresh_expansion88",
            "random_expansion88",
        ],
        numerical_range_tolerance=1e-7,
        minimum_test_margin=1e-5,
    )
    assert not result["gates"][
        "fresh88_beats_baseline_dense_and_random_on_every_window"
    ]
