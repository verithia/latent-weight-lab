from __future__ import annotations

import torch
import torch.nn.functional as F

from examples.nanogpt.analyze_layer_private_2of4_mlp_teacher_fit import (
    gather_values,
    magnitude_indices,
)
from examples.nanogpt.analyze_shared_base_private_2of4_mlp_teacher_fit import (
    SharedBasePrivate24MLP,
    least_squares_base,
    support_mask,
)


def toy_family() -> SharedBasePrivate24MLP:
    generator = torch.Generator().manual_seed(41)
    dense_fc = torch.randn(2, 8, 4, generator=generator)
    dense_proj = torch.randn(2, 4, 8, generator=generator)
    mean_fc, mean_proj = dense_fc.mean(0), dense_proj.mean(0)
    indices_fc = torch.stack(
        [magnitude_indices(weight - mean_fc) for weight in dense_fc]
    )
    indices_proj = torch.stack(
        [magnitude_indices(weight - mean_proj) for weight in dense_proj]
    )
    base_fc = least_squares_base(dense_fc, indices_fc)
    base_proj = least_squares_base(dense_proj, indices_proj)
    return SharedBasePrivate24MLP(
        base_fc=base_fc,
        base_proj=base_proj,
        values_fc=torch.stack(
            [
                gather_values(weight - base_fc, index)
                for weight, index in zip(dense_fc, indices_fc)
            ]
        ),
        indices_fc=indices_fc,
        values_proj=torch.stack(
            [
                gather_values(weight - base_proj, index)
                for weight, index in zip(dense_proj, indices_proj)
            ]
        ),
        indices_proj=indices_proj,
    )


def test_support_mask_has_exactly_two_entries_per_group() -> None:
    indices = torch.tensor(
        [[[[0, 3], [1, 2]], [[1, 3], [0, 2]]]], dtype=torch.uint8
    )
    mask = support_mask(indices)
    assert mask.shape == (1, 2, 8)
    assert mask.reshape(1, 2, 2, 4).sum(dim=-1).eq(2).all()


def test_least_squares_base_has_zero_omitted_residual_sum() -> None:
    generator = torch.Generator().manual_seed(43)
    weights = torch.randn(5, 3, 8, generator=generator)
    indices = torch.stack(
        [magnitude_indices(weight - weights.mean(0)) for weight in weights]
    )
    base = least_squares_base(weights, indices)
    omitted = ~support_mask(indices)
    residual_sum = ((weights - base) * omitted).sum(0)
    torch.testing.assert_close(residual_sum, torch.zeros_like(residual_sum), atol=1e-6, rtol=0)


def test_family_owns_shared_base_and_private_sparse_values_only() -> None:
    module = toy_family()
    assert sum(parameter.numel() for parameter in module.parameters()) == 128
    assert module.base_fc.shape == (8, 4)
    assert module.base_proj.shape == (4, 8)
    assert module.indices_fc.dtype == torch.uint8
    assert module.indices_proj.dtype == torch.uint8


def test_forward_matches_materialized_base_plus_residual() -> None:
    module = toy_family()
    values = torch.randn(7, 4)
    c_fc, c_proj = module.weights(1)
    expected = F.linear(F.gelu(F.linear(values, c_fc)), c_proj)
    torch.testing.assert_close(module.forward_layer(1, values), expected)


def test_gradients_reach_shared_and_private_state() -> None:
    module = toy_family()
    loss = module.forward_layer(0, torch.randn(7, 4)).square().mean()
    loss.backward()
    for parameter in module.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
