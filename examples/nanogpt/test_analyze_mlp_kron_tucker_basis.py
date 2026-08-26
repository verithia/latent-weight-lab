from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_kron_tucker_basis import (
    direct_kron_apply,
    hooi_fit,
    kron_rearrange,
    kron_unrearrange,
)


def test_kron_rearrangement_roundtrip() -> None:
    matrix = torch.arange(48.0).reshape(6, 8)
    rearranged = kron_rearrange(
        matrix,
        row_outer=2,
        row_inner=3,
        column_outer=2,
        column_inner=4,
    )
    recovered = kron_unrearrange(
        rearranged,
        row_outer=2,
        row_inner=3,
        column_outer=2,
        column_inner=4,
    )
    assert torch.equal(recovered, matrix)


def test_hooi_recovers_shared_rank_one_tensor() -> None:
    left = torch.tensor([1.0, 2.0, -1.0])
    left = left / left.norm()
    right = torch.tensor([2.0, -1.0, 0.5, 1.0])
    right = right / right.norm()
    tensor = torch.stack((torch.outer(left, right), -2 * torch.outer(left, right)))
    fitted_left, fitted_right, core, capture = hooi_fit(
        tensor,
        left_rank=1,
        right_rank=1,
        left_initial=left[:, None],
        right_initial=right[:, None],
        iterations=2,
    )
    assert fitted_left.shape == (3, 1)
    assert fitted_right.shape == (4, 1)
    assert core.shape == (2, 1, 1)
    assert abs(capture - 1.0) < 1e-6


def test_direct_kron_apply_matches_materialized_operator() -> None:
    torch.manual_seed(4)
    p, q = 2, 3
    row_outer, row_inner, column_outer, column_inner = 2, 3, 2, 2
    left = torch.randn(p, row_outer, column_outer)
    right = torch.randn(q, row_inner, column_inner)
    core = torch.randn(p, q)
    inputs = torch.randn(5, column_outer, column_inner)
    direct = direct_kron_apply(inputs, left, right, core).flatten(1)
    dense = sum(
        core[a, b] * torch.kron(left[a], right[b])
        for a in range(p)
        for b in range(q)
    )
    expected = inputs.flatten(1) @ dense.T
    assert torch.allclose(direct, expected, atol=1e-5, rtol=1e-5)
