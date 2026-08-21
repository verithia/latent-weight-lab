from __future__ import annotations

import torch

from examples.nanogpt.analyze_seven_trunk_private_residual_spectrum import (
    cosine,
    private_residuals,
    spectrum,
    subspace_overlap,
    tangent_transfer,
)


def test_private_residuals_sum_to_zero_within_group() -> None:
    gradients = [
        {"c_fc": torch.full((4, 3), float(i)), "c_proj": torch.full((3, 4), float(2 * i))}
        for i in range(4)
    ]
    residuals = private_residuals(gradients, ((0, 1), (2, 3)))
    for group in residuals.values():
        for matrix in ("c_fc", "c_proj"):
            selected = [value for name, value in group.items() if name.endswith(matrix)]
            assert torch.allclose(torch.stack(selected).sum(0), torch.zeros_like(selected[0]))


def test_spectrum_recovers_exact_rank_two() -> None:
    torch.manual_seed(1)
    matrix = torch.randn(8, 2) @ torch.randn(2, 6)
    measured = spectrum(matrix, (1, 2, 4))
    assert measured["rank_recovery"]["1"] < 1.0
    assert abs(measured["rank_recovery"]["2"] - 1.0) < 1e-5


def test_tangent_transfer_and_subspace_overlap_are_exact_for_shared_rank() -> None:
    torch.manual_seed(2)
    left = torch.randn(9, 2)
    right = torch.randn(7, 2)
    source = left @ right.transpose(0, 1)
    target = left @ torch.randn(2, 2) @ right.transpose(0, 1)
    assert tangent_transfer(source, target, 2)["recovery"] > 0.99999
    overlap = subspace_overlap(source, target, 2)
    assert overlap["left"] > 0.99999
    assert overlap["right"] > 0.99999
    assert cosine(source, source) > 0.99999
