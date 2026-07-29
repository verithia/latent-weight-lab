from __future__ import annotations

import torch
import pytest

from examples.nanogpt.analyze_mlp_activation_update_alignment import (
    randomized_principal_basis,
    subspace_overlap,
    update_energy_capture,
)


def test_aligned_activation_basis_captures_update() -> None:
    generator = torch.Generator().manual_seed(7)
    directions = torch.linalg.qr(
        torch.randn(16, 4, generator=generator)
    ).Q
    coefficients = torch.randn(64, 4, generator=generator)
    activations = coefficients @ directions.T
    update = torch.randn(6, 4, generator=generator) @ directions.T
    basis, singular, total = randomized_principal_basis(
        activations,
        4,
        center=True,
        seed=11,
        oversample=0,
        power_iterations=2,
    )
    assert singular.shape == (4,)
    assert total > 0.0
    assert update_energy_capture(update, basis) > 0.99999
    assert subspace_overlap(basis, directions) > 0.99999


def test_orthogonal_activation_basis_rejects_update() -> None:
    update = torch.zeros(3, 8)
    update[:, :2] = torch.eye(3, 2)
    basis = torch.eye(8)[:, 2:4]
    assert update_energy_capture(update, basis) == 0.0


def test_subspace_overlap_requires_matched_shapes() -> None:
    with pytest.raises(ValueError, match="same"):
        subspace_overlap(torch.eye(4)[:, :2], torch.eye(4)[:, :3])
