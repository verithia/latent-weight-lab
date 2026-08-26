from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_cross_layer_state_basis import (
    canonical_metrics,
    shared_basis_rows,
    transpose_basis,
    weighted_subspace_capture,
)


def test_weighted_subspace_capture_exact_and_orthogonal() -> None:
    source = torch.eye(4)[:, :2]
    target = torch.eye(4)[:, :2]
    values = torch.tensor([3.0, 1.0])
    assert abs(weighted_subspace_capture(source, target, values) - 1.0) < 1e-7
    assert weighted_subspace_capture(torch.eye(4)[:, 2:], target, values) == 0.0


def test_shared_basis_recovers_common_direction() -> None:
    common = torch.tensor([[1.0], [0.0], [0.0]])
    rows = shared_basis_rows(
        target="x",
        layers=[0, 1],
        bases=[common, common],
        eigenvalues=[torch.ones(1), torch.ones(1)],
        ranks=[1],
        parameter_size=3,
    )
    assert len(rows) == 1
    assert abs(rows[0]["aggregate_state_energy_capture"] - 1.0) < 1e-7
    assert abs(rows[0]["minimum_layer_capture"] - 1.0) < 1e-7


def test_transpose_basis_and_canonical_metrics() -> None:
    matrix = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    basis = (matrix / matrix.norm()).reshape(-1, 1)
    transposed = transpose_basis(basis, 2, 3)
    expected = (matrix.T / matrix.norm()).reshape(-1, 1)
    assert torch.allclose(transposed, expected)
    mean_cos2, minimum, maximum = canonical_metrics(transposed, expected)
    assert abs(mean_cos2 - 1.0) < 1e-6
    assert abs(minimum - 1.0) < 1e-6
    assert abs(maximum - 1.0) < 1e-6
