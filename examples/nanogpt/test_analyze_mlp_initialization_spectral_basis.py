from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_initialization_spectral_basis import (
    common_positions,
    spectral_basis_metrics,
)


def test_common_positions_preserve_w0_and_average_displacements() -> None:
    w0 = torch.eye(3)
    first = [w0, w0 + 2]
    second = [w0.clone(), w0 + 4]
    result = common_positions(first, second)
    torch.testing.assert_close(result[0], w0)
    torch.testing.assert_close(result[1], w0 + 3)


def test_spectral_diagonal_exactly_recovers_diagonal_basis() -> None:
    base = torch.diag(torch.tensor([3.0, 2.0, 1.0]))
    basis = torch.stack(
        (
            torch.diag(torch.tensor([1.0, 0.0, 0.0])),
            torch.diag(torch.tensor([0.0, 1.0, 0.0])),
        )
    )
    overview, rows = spectral_basis_metrics(
        base,
        basis,
        torch.tensor([3.0, 1.0]),
        total_ratios=[0.5],
    )
    assert overview["thin_frame_weighted_capture"] > 0.999999
    assert overview["spectral_diagonal_weighted_capture"] > 0.999999
    assert overview["spectral_diagonal_total_stored_scalar_fraction"] == 2 / 3
    assert all(row["weighted_basis_energy_capture"] > 0.999999 for row in rows)
