from __future__ import annotations

import pytest
import torch

from examples.nanogpt.analyze_sparse_moe_paired_atom_oracle import (
    LayerState,
    _state_chord,
    energy_recovery,
    fixed_local_basis,
    project_rows,
    reconstruct_family,
)


def _zero_state(experts: int = 2, hidden: int = 6, width: int = 4) -> LayerState:
    return LayerState(
        torch.zeros(experts, width),
        torch.zeros(experts, hidden, width),
        torch.zeros(experts, width, hidden),
    )


def test_project_rows_recovers_independent_in_span_targets() -> None:
    torch.manual_seed(23)
    basis = torch.randn(3, 2, 5, 7)
    truth = torch.randn(2, 5, 3)
    target = torch.einsum("...k,k...d->...d", truth, basis)
    projected, coordinates = project_rows(target, basis, ridge_ratio=1e-10)
    assert coordinates.shape == truth.shape
    assert energy_recovery(projected, target) > 0.999999


def test_coupled_four_reconstructs_paired_atom_chord() -> None:
    torch.manual_seed(29)
    left = _zero_state()
    history = []
    for _ in range(4):
        history.append(
            (
                torch.randn_like(left.router),
                torch.randn_like(left.c_fc),
                torch.randn_like(left.c_proj).transpose(1, 2),
            )
        )
    pair_coefficients = torch.randn(2, 6, 4)
    router_coefficients = torch.randn(2, 4)
    target_router = torch.einsum("ek,ker->er", router_coefficients, torch.stack([x[0] for x in history]))
    target_fc = torch.einsum("ehk,kehd->ehd", pair_coefficients, torch.stack([x[1] for x in history]))
    target_proj = torch.einsum("ehk,kehd->ehd", pair_coefficients, torch.stack([x[2] for x in history]))
    target = (target_router, target_fc, target_proj)
    reconstructed, recoveries, metadata = reconstruct_family(
        left, target, history + [target], 4, "coupled_four", "moving", 0, 1e-10, "cpu"
    )
    actual = _state_chord(reconstructed, left)
    assert metadata["basis_transition_indices"] == [0, 1, 2, 3]
    assert recoveries["paired_parameter_recovery"] > 0.999999
    assert recoveries["router_parameter_recovery"] > 0.999999
    assert energy_recovery(actual[1], target_fc) > 0.999999
    assert energy_recovery(actual[2], target_proj) > 0.999999


def test_separate_three_reconstructs_distinct_incoming_and_outgoing_coordinates() -> None:
    torch.manual_seed(31)
    left = _zero_state()
    history = []
    for _ in range(3):
        history.append(
            (
                torch.randn_like(left.router),
                torch.randn_like(left.c_fc),
                torch.randn_like(left.c_proj).transpose(1, 2),
            )
        )
    fc_coefficients = torch.randn(2, 6, 3)
    proj_coefficients = torch.randn(2, 6, 3)
    router_coefficients = torch.randn(2, 3)
    target = (
        torch.einsum("ek,ker->er", router_coefficients, torch.stack([x[0] for x in history])),
        torch.einsum("ehk,kehd->ehd", fc_coefficients, torch.stack([x[1] for x in history])),
        torch.einsum("ehk,kehd->ehd", proj_coefficients, torch.stack([x[2] for x in history])),
    )
    reconstructed, recoveries, metadata = reconstruct_family(
        left, target, history + [target], 3, "separate_three_plus_three", "moving", 0, 1e-10, "cpu"
    )
    assert metadata["basis_transition_indices"] == [0, 1, 2]
    assert recoveries["paired_parameter_recovery"] > 0.999999
    assert recoveries["router_parameter_recovery"] > 0.999999
    assert energy_recovery(_state_chord(reconstructed, left)[1], target[1]) > 0.999999


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA regression test")
@pytest.mark.parametrize(("width", "rank"), [(1536, 4), (768, 3)])
def test_fixed_local_basis_uses_native_cuda_minimum(width: int, rank: int) -> None:
    basis = fixed_local_basis(width, rank, seed=20260901, device="cuda")
    assert basis.shape == (rank, width)
    torch.testing.assert_close(basis @ basis.T, torch.eye(rank), atol=1e-5, rtol=1e-5)
