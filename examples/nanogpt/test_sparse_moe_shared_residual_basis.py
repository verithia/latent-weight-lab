from __future__ import annotations

import torch

from examples.nanogpt.analyze_sparse_moe_shared_residual_basis import (
    basis_from_grams,
    layout_state,
    local_oracle_recovery,
    projection_recovery,
)


def test_registered_layout_state_accounting() -> None:
    dense = 12 * 8 * 768 * 1536
    state = layout_state(
        dense_values=dense,
        matrices=96,
        output_width=768,
        global_compression=200.0,
        per_matrix_coordinates=4194,
    )
    assert state["total_coordinate_budget"] == 566231
    assert state["per_matrix_coordinate_total"] == 402624
    assert state["shared_basis_rank"] == 213
    assert state["total_coordinates_used"] == 566208


def test_basis_from_grams_orders_directions() -> None:
    first = torch.diag(torch.tensor([9.0, 1.0, 0.0]))
    second = torch.diag(torch.tensor([0.0, 4.0, 1.0]))
    values, basis = basis_from_grams([first, second])
    assert torch.allclose(values, torch.tensor([9.0, 5.0, 1.0]))
    assert abs(projection_recovery(first + second, basis, 1) - 9.0 / 15.0) < 1e-7


def test_shared_projection_is_bounded_by_local_oracle() -> None:
    gram = torch.diag(torch.tensor([9.0, 4.0, 1.0]))
    rotated = torch.eye(3)[:, [1, 0, 2]]
    shared = projection_recovery(gram, rotated, 1)
    local = local_oracle_recovery(gram, 1)
    assert abs(shared - 4.0 / 14.0) < 1e-7
    assert abs(local - 9.0 / 14.0) < 1e-7
    assert shared <= local


def test_zero_rank_projection_is_zero() -> None:
    gram = torch.eye(4)
    assert projection_recovery(gram, torch.eye(4), 0) == 0.0
