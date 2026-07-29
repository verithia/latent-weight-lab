from __future__ import annotations

from pathlib import Path

import pytest
import torch

from examples.nanogpt.fast_task_matching import (
    color_sorted_edges,
    fast_muon_matched_permutations,
)


def assert_unique_perfect_matchings(permutations: torch.Tensor) -> None:
    stages, width = permutations.shape
    expected = torch.arange(width)
    edges: set[tuple[int, int]] = set()
    for stage in range(stages):
        torch.testing.assert_close(
            torch.sort(permutations[stage]).values,
            expected,
            rtol=0.0,
            atol=0.0,
        )
        for left, right in permutations[stage].view(-1, 2).tolist():
            edge = (min(left, right), max(left, right))
            assert edge not in edges
            edges.add(edge)


def test_color_sorted_edges_is_deterministic_and_complete(
    tmp_path: Path,
) -> None:
    edges = torch.tensor(
        [
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
            [0, 2],
            [1, 3],
            [4, 6],
            [5, 7],
            [0, 3],
            [1, 2],
        ],
        dtype=torch.int32,
    )
    first, first_diagnostics = color_sorted_edges(
        edges,
        width=8,
        stages=3,
        seed=17,
        cache_dir=tmp_path,
    )
    second, second_diagnostics = color_sorted_edges(
        edges,
        width=8,
        stages=3,
        seed=17,
        cache_dir=tmp_path,
    )
    assert_unique_perfect_matchings(first)
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert first_diagnostics["source_sha256"]
    assert first_diagnostics["native_library_sha256"]
    assert (
        first_diagnostics["candidate_edge_fraction"]
        == second_diagnostics["candidate_edge_fraction"]
    )


def test_fast_muon_matching_returns_valid_task_matchings(
    tmp_path: Path,
) -> None:
    generator = torch.Generator().manual_seed(23)
    weight = torch.randn(6, 16, generator=generator)
    direction = torch.randn(6, 16, generator=generator)
    permutations, diagnostics = fast_muon_matched_permutations(
        weight,
        direction,
        stages=4,
        neighbors=6,
        seed=29,
        cache_dir=tmp_path,
    )
    assert_unique_perfect_matchings(permutations)
    assert diagnostics["candidate_edges"] == 16 * 6
    assert 0.0 <= diagnostics["candidate_edge_fraction"] <= 1.0
    assert diagnostics["total_seconds"] > 0.0


@pytest.mark.parametrize(
    "width,stages",
    [(7, 2), (8, 0), (8, 65)],
)
def test_color_sorted_edges_rejects_invalid_shapes(
    tmp_path: Path,
    width: int,
    stages: int,
) -> None:
    with pytest.raises(ValueError):
        color_sorted_edges(
            torch.empty(0, 2, dtype=torch.int32),
            width=width,
            stages=stages,
            seed=0,
            cache_dir=tmp_path,
        )
