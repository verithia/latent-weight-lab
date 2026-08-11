from __future__ import annotations

import torch

from examples.nanogpt.analyze_sparse_moe_depth_grouped_basis import (
    allocate_group_ranks,
)


def test_allocation_preserves_total_and_minimum() -> None:
    spectra = [
        torch.tensor([9.0, 8.0, 7.0, 1.0]),
        torch.tensor([6.0, 5.0, 4.0, 3.0]),
        torch.tensor([2.0, 1.0, 0.5, 0.25]),
    ]
    allocation = allocate_group_ranks(spectra, total_rank=6, minimum_rank=1)
    assert sum(allocation) == 6
    assert min(allocation) >= 1


def test_normalized_allocation_is_scale_invariant() -> None:
    spectra = [torch.tensor([4.0, 2.0, 1.0]), torch.tensor([3.0, 2.0, 1.0])]
    scaled = [spectra[0] * 100.0, spectra[1] * 0.01]
    assert allocate_group_ranks(spectra, total_rank=4, minimum_rank=1) == allocate_group_ranks(
        scaled, total_rank=4, minimum_rank=1
    )


def test_allocation_rejects_infeasible_minimum() -> None:
    spectra = [torch.ones(4), torch.ones(4), torch.ones(4)]
    try:
        allocate_group_ranks(spectra, total_rank=5, minimum_rank=2)
    except ValueError as error:
        assert "minimum" in str(error)
    else:
        raise AssertionError("expected infeasible minimum to fail")
