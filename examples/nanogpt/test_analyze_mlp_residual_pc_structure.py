from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_residual_pc_structure import (
    matrix_structure_metrics,
    top_fraction_capture,
)


def test_rank_one_metrics_are_exact() -> None:
    matrix = torch.outer(torch.arange(1.0, 6.0), torch.arange(1.0, 8.0))
    metrics = matrix_structure_metrics(matrix)
    assert metrics["rank1_capture"] > 1.0 - 1e-10
    assert metrics["rank_50"] == 1
    assert metrics["rank_99"] == 1
    assert abs(float(metrics["stable_rank"]) - 1.0) < 1e-6


def test_top_fraction_capture_selects_largest_energy() -> None:
    energy = torch.tensor([1.0, 4.0, 9.0, 16.0])
    assert abs(top_fraction_capture(energy, 0.25) - 16.0 / 30.0) < 1e-12


def test_sparse_entry_and_axis_metrics() -> None:
    matrix = torch.zeros(10, 10)
    matrix[2, 7] = 3.0
    metrics = matrix_structure_metrics(matrix)
    assert metrics["entry_p001_capture"] == 1.0
    assert metrics["row_p001_capture"] == 1.0
    assert metrics["column_p001_capture"] == 1.0
