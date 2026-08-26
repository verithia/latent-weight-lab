from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_state_common_private import (
    aggregate_capture,
    energy_fraction,
    remove_scalar_orbit,
)


def test_scalar_orbit_removal_is_exact() -> None:
    initial = torch.tensor([1.0, -2.0, 3.0])
    rows = torch.stack((torch.zeros(3), 2.0 * initial, -0.5 * initial))
    residual, coefficients, capture = remove_scalar_orbit(rows, initial)
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        coefficients, torch.tensor([0.0, 2.0, -0.5], dtype=torch.float64)
    )
    assert aggregate_capture(rows[1:], initial) > 1.0 - 1e-12
    assert capture[1:].min() > 1.0 - 1e-12


def test_common_private_energy_identity() -> None:
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[2.0, -1.0], [0.0, 5.0]])
    common = 0.5 * (a + b)
    private = 0.5 * (a - b)
    common_fraction = energy_fraction(common, (common, private))
    private_fraction = energy_fraction(private, (common, private))
    assert abs(common_fraction + private_fraction - 1.0) < 1e-12
