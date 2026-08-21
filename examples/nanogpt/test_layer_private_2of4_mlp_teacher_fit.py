from __future__ import annotations

import torch
import torch.nn.functional as F

from examples.nanogpt.analyze_layer_private_2of4_mlp_teacher_fit import (
    LayerPrivate24MLP,
    gather_values,
    magnitude_indices,
    random_indices,
    unpack_weight,
)


def test_magnitude_indices_are_stable_on_ties() -> None:
    weight = torch.tensor([[2.0, -2.0, 1.0, 0.0, 1.0, 4.0, -3.0, 2.0]])
    indices = magnitude_indices(weight)
    assert torch.equal(indices, torch.tensor([[[0, 1], [1, 2]]], dtype=torch.uint8))


def test_random_indices_are_deterministic_and_two_of_four() -> None:
    first = random_indices((7, 12), seed=23)
    second = random_indices((7, 12), seed=23)
    other = random_indices((7, 12), seed=24)
    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    assert first.shape == (7, 3, 2)
    assert torch.all(first[..., 0] < first[..., 1])


def test_pack_unpack_preserves_selected_values_and_zeros_others() -> None:
    weight = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    indices = torch.tensor(
        [[[0, 3], [1, 2]], [[1, 3], [0, 2]], [[0, 2], [1, 3]]],
        dtype=torch.uint8,
    )
    values = gather_values(weight, indices)
    unpacked = unpack_weight(values, indices)
    grouped = unpacked.reshape(3, 2, 4)
    assert torch.count_nonzero(grouped, dim=-1).le(2).all()
    torch.testing.assert_close(
        grouped.gather(-1, indices.long()), values
    )


def family() -> LayerPrivate24MLP:
    generator = torch.Generator().manual_seed(31)
    dense_fc = torch.randn(2, 8, 4, generator=generator)
    dense_proj = torch.randn(2, 4, 8, generator=generator)
    fc_indices = torch.stack([magnitude_indices(weight) for weight in dense_fc])
    proj_indices = torch.stack(
        [magnitude_indices(weight) for weight in dense_proj]
    )
    return LayerPrivate24MLP(
        values_fc=torch.stack(
            [gather_values(weight, index) for weight, index in zip(dense_fc, fc_indices)]
        ),
        indices_fc=fc_indices,
        values_proj=torch.stack(
            [gather_values(weight, index) for weight, index in zip(dense_proj, proj_indices)]
        ),
        indices_proj=proj_indices,
    )


def test_family_stores_only_nonzero_values_as_parameters() -> None:
    module = family()
    assert sum(parameter.numel() for parameter in module.parameters()) == 64
    assert module.indices_fc.dtype == torch.uint8
    assert module.indices_proj.dtype == torch.uint8
    for layer in range(module.layers):
        c_fc, c_proj = module.weights(layer)
        assert torch.count_nonzero(c_fc, dim=-1).eq(2).all()
        assert torch.count_nonzero(c_proj.reshape(4, 2, 4), dim=-1).eq(2).all()


def test_forward_matches_materialized_sparse_weights() -> None:
    module = family()
    values = torch.randn(5, 4)
    c_fc, c_proj = module.weights(1)
    expected = F.linear(F.gelu(F.linear(values, c_fc)), c_proj)
    torch.testing.assert_close(module.forward_layer(1, values), expected)
