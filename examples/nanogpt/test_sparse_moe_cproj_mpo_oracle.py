from __future__ import annotations

import torch

from examples.nanogpt.analyze_sparse_moe_cproj_mpo_oracle import (
    coordinate_count,
    materialize_mpo,
    matrix_to_physical,
    physical_to_matrix,
    truncated_mpo_svd,
)


OUTPUT_MODES = [3, 4, 4, 4, 4]
INPUT_MODES = [6, 4, 4, 4, 4]


def test_registered_rank9_budget() -> None:
    assert coordinate_count(OUTPUT_MODES, INPUT_MODES, 9) == 4194
    assert abs((768 * 1536 / 4194) - 281.27038626609444) < 1e-12


def test_physical_roundtrip() -> None:
    generator = torch.Generator().manual_seed(3)
    matrix = torch.randn(2, 12, 24, generator=generator)
    output_modes = [3, 4]
    input_modes = [6, 4]
    physical = matrix_to_physical(matrix, output_modes, input_modes)
    recovered = physical_to_matrix(physical, output_modes, input_modes)
    assert torch.equal(recovered, matrix)


def test_materialize_matches_direct_two_core_contraction() -> None:
    generator = torch.Generator().manual_seed(5)
    first = torch.randn(2, 1, 2, 3, 4, generator=generator)
    second = torch.randn(2, 4, 3, 2, 1, generator=generator)
    actual = materialize_mpo([first, second], [2, 3], [3, 2])
    direct = torch.einsum("eaoir,erpjb->eaopijb", first, second)
    direct = direct[:, 0, ..., 0].reshape(2, 6, 6)
    assert torch.allclose(actual, direct)


def test_full_rank_tt_svd_reconstructs_small_matrix() -> None:
    generator = torch.Generator().manual_seed(7)
    matrix = torch.randn(2, 12, 24, generator=generator)
    cores = truncated_mpo_svd(matrix, [3, 4], [6, 4], rank=18)
    recovered = materialize_mpo(cores, [3, 4], [6, 4])
    assert torch.allclose(recovered, matrix, atol=2e-5, rtol=2e-5)


def test_mpo_gradients_are_finite() -> None:
    generator = torch.Generator().manual_seed(11)
    cores = [
        torch.nn.Parameter(torch.randn(2, 1, 2, 2, 3, generator=generator)),
        torch.nn.Parameter(torch.randn(2, 3, 2, 2, 1, generator=generator)),
    ]
    loss = materialize_mpo(cores, [2, 2], [2, 2]).square().mean()
    loss.backward()
    assert all(core.grad is not None and torch.isfinite(core.grad).all() for core in cores)
