from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_shared_separable_conditioner_loo import (
    conditioner_accounting,
    separable_transfer,
)


def test_conditioner_accounting_is_exact_and_below_one_percent() -> None:
    accounting = conditioner_accounting()
    assert accounting["prompt_scalars"] == 561_408
    assert accounting["shared_row_gain_scalars"] == 3_072
    assert accounting["shared_column_gain_scalars"] == 768
    assert accounting["total_state_scalars"] == 565_272
    assert accounting["state_fraction"] < 0.01


def test_leave_one_out_recovers_a_shared_separable_relation() -> None:
    torch.manual_seed(17)
    shape = (37, 19)
    left = torch.exp(0.2 * torch.randn(shape[0]))
    right = torch.exp(0.2 * torch.randn(shape[1]))
    atoms = tuple(torch.randn(shape) for _ in range(6))
    targets = tuple(
        left[:, None] * atom * right[None, :] + 0.001 * torch.randn_like(atom)
        for atom in atoms
    )
    result = separable_transfer(atoms, targets)
    assert result["minimum_leave_one_out_capture"] > 0.99
    assert result["median_leave_one_out_capture"] > 0.99
    assert len(result["rows"]) == 6
