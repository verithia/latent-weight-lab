from __future__ import annotations

import pytest
import torch

from examples.nanogpt.analyze_mlp_orthogonal_transport_oracle import (
    aggregate_rows,
    orthogonal_transport_metrics,
)


def test_right_oracle_recovers_exact_right_rotation() -> None:
    source = torch.tensor(
        [[1.0, 2.0, 0.0], [0.0, 1.0, 3.0]],
        dtype=torch.float64,
    )
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    metrics = orthogonal_transport_metrics(source, source @ rotation)
    assert metrics["right_endpoint_recovery"] == pytest.approx(
        1.0, abs=1e-12
    )
    assert metrics["bilateral_endpoint_recovery"] == pytest.approx(
        1.0, abs=1e-12
    )


def test_left_oracle_recovers_exact_left_rotation() -> None:
    source = torch.tensor(
        [[1.0, 2.0, 0.0], [0.0, 1.0, 3.0]],
        dtype=torch.float64,
    )
    rotation = torch.tensor(
        [[0.0, -1.0], [1.0, 0.0]],
        dtype=torch.float64,
    )
    metrics = orthogonal_transport_metrics(source, rotation @ source)
    assert metrics["left_endpoint_recovery"] == pytest.approx(
        1.0, abs=1e-12
    )
    assert metrics["bilateral_endpoint_recovery"] == pytest.approx(
        1.0, abs=1e-12
    )


def test_aggregate_is_chord_energy_weighted() -> None:
    rows = [
        {
            "chord_fro": 1.0,
            "left_endpoint_recovery": 1.0,
            "right_endpoint_recovery": 1.0,
            "bilateral_endpoint_recovery": 1.0,
            "singular_value_drift_fraction": 0.0,
        },
        {
            "chord_fro": 3.0,
            "left_endpoint_recovery": 0.0,
            "right_endpoint_recovery": 0.0,
            "bilateral_endpoint_recovery": 0.0,
            "singular_value_drift_fraction": 1.0,
        },
    ]
    aggregate = aggregate_rows(rows)
    assert aggregate[
        "energy_weighted_right_endpoint_recovery"
    ] == pytest.approx(0.1)
    assert aggregate[
        "energy_weighted_singular_value_drift_fraction"
    ] == pytest.approx(0.9)
