from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_block_unfolding_atlas import (
    fold_blocks,
    maximum_rank_for_budget,
    unfold_blocks,
)


def test_block_unfolding_round_trip() -> None:
    matrix = torch.arange(8 * 4).reshape(8, 4)
    unfolded = unfold_blocks(matrix, 2)
    assert unfolded.shape == (8, 4)
    torch.testing.assert_close(fold_blocks(unfolded, 8, 4, 2), matrix)


def test_budget_selects_many_local_coordinates_under_one_percent() -> None:
    assert maximum_rank_for_budget(3072, 768, 32, 0.01) == 7
    assert maximum_rank_for_budget(3072, 768, 64, 0.01) == 5
    assert 7 * (2304 + 1024) / (3072 * 768) < 0.01
    assert 5 * (576 + 4096) / (3072 * 768) < 0.01
