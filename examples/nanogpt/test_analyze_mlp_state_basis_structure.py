from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_state_basis_structure import (
    analyze_parameter,
    weighted_adaptive_svd_capture,
    weighted_support_capture,
)


def test_weighted_support_selects_shared_coordinate() -> None:
    coefficients = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    eigenvalues = torch.tensor([4.0, 1.0])
    # One shared coordinate captures the dominant PC's 80% variance share.
    assert abs(weighted_support_capture(coefficients, eigenvalues, 1) - 0.8) < 1e-6


def test_weighted_low_rank_basis_capture() -> None:
    matrices = torch.stack((torch.eye(3), torch.ones((3, 3))))
    eigenvalues = torch.tensor([3.0, 1.0])
    weighted, minimum, maximum = weighted_adaptive_svd_capture(
        matrices, eigenvalues, rank=1
    )
    assert 0.0 < minimum <= weighted <= maximum <= 1.0
    assert maximum > 0.999


def test_bilateral_storage_counts_every_temporal_basis_vector() -> None:
    generator = torch.Generator().manual_seed(19)
    positions = torch.randn((5, 4, 4), generator=generator)
    rows = analyze_parameter(
        positions,
        parameter="transformer.h.6.mlp.c_fc.weight",
        ratios=[0.25],
        basis_rank=2,
        block_fht_layers=2,
        block_fht_seed=1000,
    )
    bilateral = next(
        row for row in rows if row["family"] == "w0_bilateral_diagonal_tangent"
    )
    assert bilateral["state_basis_rank"] == 2
    assert bilateral["coordinate_ratio_resolved"] == 0.5
    assert bilateral["total_stored_scalar_fraction_for_all_basis_vectors"] == 1.0
