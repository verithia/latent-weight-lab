from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_phase_leading_update import (
    aggregate_rows,
    leading_direction_metrics,
)


def test_leading_direction_metrics_exact_collinear() -> None:
    chord = torch.tensor(
        [[1.0, -2.0], [3.0, 4.0]],
        dtype=torch.float64,
    )
    metrics = leading_direction_metrics(0.25 * chord, chord)
    assert metrics["cosine"] == 1.0
    assert metrics["one_direction_energy_capture"] == 1.0
    assert metrics["optimal_lead_scale"] == 4.0
    assert metrics["optimal_residual_energy_fraction"] == 0.0


def test_leading_direction_metrics_orthogonal() -> None:
    lead = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    chord = torch.tensor([[0.0, 2.0]], dtype=torch.float64)
    metrics = leading_direction_metrics(lead, chord)
    assert metrics["cosine"] == 0.0
    assert metrics["one_direction_energy_capture"] == 0.0
    assert metrics["optimal_residual_energy_fraction"] == 1.0


def test_aggregate_rows_is_chord_energy_weighted() -> None:
    rows = [
        {
            "lookahead": 6,
            "chord_fro": 1.0,
            "one_direction_energy_capture": 1.0,
            "cosine": 1.0,
        },
        {
            "lookahead": 6,
            "chord_fro": 3.0,
            "one_direction_energy_capture": 0.0,
            "cosine": 0.0,
        },
    ]
    aggregate = aggregate_rows(rows)[0]
    assert aggregate["energy_weighted_capture"] == 0.1
    assert aggregate["energy_weighted_cosine"] == 0.1
