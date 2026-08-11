from __future__ import annotations

import torch

from examples.nanogpt.analyze_sparse_moe_shared_orthogonal_gauge_mpo import (
    apply_output_gauge,
    orthogonal_procrustes_right,
)


def test_registered_state_accounting() -> None:
    dense = 12 * 8 * 768 * 1536
    mpo = 12 * 8 * 4194
    orthogonal_dof = 768 * 767 // 2
    cached = 768 * 768
    assert mpo == 402624
    assert orthogonal_dof == 294528
    assert dense / (mpo + orthogonal_dof) == 162.44120077113743
    assert dense / (mpo + cached) == 114.10795124782356


def test_apply_output_gauge_matches_left_matrix_multiplication() -> None:
    generator = torch.Generator().manual_seed(3)
    hidden = torch.randn(11, 7, generator=generator)
    matrix = torch.randn(5, 7, generator=generator)
    raw = hidden @ matrix.T
    q, _ = torch.linalg.qr(torch.randn(5, 5, generator=generator))
    expected = hidden @ (q @ matrix).T
    assert torch.allclose(apply_output_gauge(raw, q), expected, atol=1e-6)


def test_procrustes_recovers_exact_right_transform() -> None:
    generator = torch.Generator().manual_seed(5)
    prediction = torch.randn(32, 7, generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(
        torch.randn(7, 7, generator=generator, dtype=torch.float64)
    )
    target = prediction @ q
    recovered = orthogonal_procrustes_right([prediction], [target])
    identity = torch.eye(7, dtype=recovered.dtype)
    assert torch.allclose(recovered.T @ recovered, identity, atol=1e-5, rtol=1e-5)
    assert torch.allclose(prediction.float() @ recovered, target.float(), atol=2e-5, rtol=2e-5)


def test_procrustes_uses_equal_relative_layer_weight() -> None:
    generator = torch.Generator().manual_seed(7)
    first = torch.randn(20, 4, generator=generator)
    second = 1000 * torch.randn(20, 4, generator=generator)
    q, _ = torch.linalg.qr(torch.randn(4, 4, generator=generator))
    target_first = first @ q
    target_second = second @ q
    recovered = orthogonal_procrustes_right(
        [first, second], [target_first, target_second]
    )
    first_error = (first @ recovered - target_first).square().sum()
    second_relative_error = (
        (second @ recovered - target_second).square().sum()
        / target_second.square().sum()
    )
    assert float(first_error) < 1e-8
    assert float(second_relative_error) < 1e-10
