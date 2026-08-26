from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_residual_qtt_basis import (
    canonicalize_basis_matrices,
    choose_rank_cap,
    residual_temporal_basis,
    tensorize_basis,
    tt_parameter_count,
    tt_reconstruct,
    tt_svd,
)


def test_residual_basis_uses_w0_gauge() -> None:
    direction = torch.arange(12.0).reshape(3, 4)
    positions = torch.stack(
        (torch.ones_like(direction), 1.0 + direction, 1.0 + 2.0 * direction)
    )
    residuals, values, basis = residual_temporal_basis(
        positions, maximum_rank=2
    )
    assert torch.count_nonzero(residuals[0]) == 0
    assert int((values > values[0] * 1e-10).sum()) == 1
    recovered = basis[:, 0].reshape_as(direction)
    assert abs(float(torch.cosine_similarity(recovered.flatten(), direction.flatten(), dim=0))) > 0.999


def test_tt_svd_exactly_recovers_rank_one_tensor() -> None:
    tensor = torch.einsum(
        "i,j,k->ijk",
        torch.tensor([1.0, 2.0]),
        torch.tensor([-1.0, 0.5, 2.0]),
        torch.tensor([3.0, -2.0]),
    )
    cores = tt_svd(tensor, rank_cap=1)
    recovered = tt_reconstruct(cores)
    assert torch.allclose(recovered, tensor, atol=1e-5, rtol=1e-5)


def test_budgeted_rank_cap_includes_current_coordinates() -> None:
    modes = (4, 3, 2, 2)
    cap = choose_rank_cap(
        modes, scalar_budget=40, current_coordinate_scalars=4
    )
    assert tt_parameter_count(modes, cap) + 4 <= 40
    assert tt_parameter_count(modes, cap + 1) + 4 > 40


def test_tensor_layouts_preserve_pc_axis_and_values() -> None:
    tensor = torch.arange(2 * 3072 * 768, dtype=torch.float32).reshape(
        2, 3072, 768
    )
    for layout in (
        "row_then_column_binary",
        "morton_binary",
        "morton_reverse_column_bits",
        "morton_nibble",
    ):
        transformed = tensorize_basis(tensor, layout=layout)
        assert transformed.shape[0] == 2
        assert transformed[0].numel() == 3072 * 768
        assert torch.equal(
            transformed.flatten(1).sort(dim=1).values,
            tensor.flatten(1).sort(dim=1).values,
        )


def test_cproj_uses_transposed_canonical_orientation() -> None:
    tensor = torch.randn(2, 768, 3072)
    canonical, transposed = canonicalize_basis_matrices(tensor)
    assert transposed
    assert canonical.shape == (2, 3072, 768)
    assert torch.equal(canonical, tensor.transpose(-2, -1))
