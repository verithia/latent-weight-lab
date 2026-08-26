from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_residual_givens_sparse_basis import (
    RandomMatchingGivens2D,
    captures,
    procedural_support,
)


def test_random_matching_givens_preserves_norm() -> None:
    torch.manual_seed(4)
    module = RandomMatchingGivens2D(8, 6, 3, seed=11, device="cpu")
    with torch.no_grad():
        module.row_angles.normal_(std=0.4)
        module.column_angles.normal_(std=0.4)
    matrices = torch.randn(5, 8, 6)
    transformed = module(matrices)
    assert torch.allclose(
        transformed.flatten(1).norm(dim=1),
        matrices.flatten(1).norm(dim=1),
        atol=1e-5,
        rtol=1e-5,
    )


def test_procedural_support_and_capture_are_deterministic() -> None:
    first = procedural_support(48, 7, seed=13, device="cpu")
    second = procedural_support(48, 7, seed=13, device="cpu")
    assert torch.equal(first, second)
    basis = torch.eye(4).reshape(4, 2, 2)
    weighted, minimum, maximum, _per_pc = captures(
        basis, torch.full((4,), 0.25), torch.arange(4)
    )
    assert weighted == 1.0
    assert minimum == 1.0
    assert maximum == 1.0
