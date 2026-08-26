from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_causal_displacement_integrability import (
    best_rank_capture,
    rank_capture_from_singular_energy,
    right_projection_capture,
    summarize,
)


def test_right_projection_capture_is_exact() -> None:
    matrix = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    basis = torch.tensor([[1.0], [0.0], [0.0]])
    expected = (1.0 + 16.0) / float(matrix.square().sum())
    assert abs(right_projection_capture(matrix, basis) - expected) < 1e-7


def test_best_rank_capture_matches_diagonal_energy() -> None:
    matrix = torch.diag(torch.tensor([4.0, 3.0, 0.0]))
    assert abs(best_rank_capture(matrix, 1) - 16.0 / 25.0) < 1e-7
    assert abs(best_rank_capture(matrix, 2) - 1.0) < 1e-7
    energy = torch.tensor([16.0, 9.0, 0.0])
    assert abs(rank_capture_from_singular_energy(energy, 1) - 16.0 / 25.0) < 1e-7


def test_summary_handles_terminal_increment() -> None:
    rows = [
        {
            "parameter": "p",
            "split": "test",
            "union_rank": 6,
            "displacement_capture": 0.8,
            "displacement_oracle_rank_capture": 0.9,
            "next_increment_capture": 0.5,
            "next_increment_oracle_rank_capture": 0.7,
        },
        {
            "parameter": "p",
            "split": "test",
            "union_rank": 6,
            "displacement_capture": 0.7,
            "displacement_oracle_rank_capture": 0.8,
            "next_increment_capture": "",
            "next_increment_oracle_rank_capture": "",
        },
    ]
    result = summarize(rows)[0]
    assert result["sample_count"] == 2
    assert result["increment_sample_count"] == 1
    assert abs(result["displacement_capture_mean"] - 0.75) < 1e-7
    assert result["next_increment_capture_mean"] == 0.5
