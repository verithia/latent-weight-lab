from __future__ import annotations

import pytest
import torch

from examples.nanogpt.analyze_mlp_stepwise_direction_history import (
    aggregate_rows,
    direction_history_metrics,
    history_span_metrics,
)


def test_history_span_capture_recovers_target_in_span() -> None:
    history = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        dtype=torch.float64,
    )
    target = torch.tensor([3.0, -4.0, 0.0], dtype=torch.float64)
    metrics = history_span_metrics(history, target)
    assert metrics["span_capture"] == pytest.approx(1.0)
    assert metrics["history_rank"] == 2


def test_direction_history_separates_full_and_future_capture() -> None:
    history = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    full = torch.tensor([1.0, 1.0], dtype=torch.float64)
    future = torch.tensor([0.0, 1.0], dtype=torch.float64)
    metrics = direction_history_metrics(history, full, future)
    assert metrics["cumulative_full_capture"] == pytest.approx(0.5)
    assert metrics["cumulative_future_capture"] == pytest.approx(0.0)
    assert metrics["future_span_capture"] == pytest.approx(0.0)
    assert metrics["first_to_current_update_cosine"] == pytest.approx(1.0)


def test_aggregate_uses_matching_chord_energy() -> None:
    common = {
        "horizon": 1,
        "cumulative_full_cosine": 0.0,
        "full_span_capture": 0.0,
        "cumulative_future_cosine": 0.0,
        "future_span_capture": 0.0,
        "first_to_current_update_cosine": 1.0,
        "previous_to_current_update_cosine": 1.0,
    }
    rows = [
        {
            **common,
            "full_chord_fro": 1.0,
            "future_chord_fro": 3.0,
            "cumulative_full_capture": 1.0,
            "cumulative_future_capture": 1.0,
            "future_span_capture": 1.0,
            "cumulative_future_cosine": 1.0,
        },
        {
            **common,
            "full_chord_fro": 3.0,
            "future_chord_fro": 1.0,
            "cumulative_full_capture": 0.0,
            "cumulative_future_capture": 0.0,
        },
    ]
    aggregate = aggregate_rows(rows)[0]
    assert aggregate[
        "energy_weighted_cumulative_full_capture"
    ] == pytest.approx(0.1)
    assert aggregate[
        "energy_weighted_cumulative_future_capture"
    ] == pytest.approx(0.9)
