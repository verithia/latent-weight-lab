from __future__ import annotations

import torch

from examples.nanogpt.analyze_attention_stepzero_functional_atlas import (
    KroneckerAtlas,
    empirical_second_moment,
    kronecker_subspace_overlap,
    top_kronecker_pairs,
)


def test_top_pairs_are_ranked_by_kfac_product() -> None:
    output = torch.tensor([1.0, 4.0])
    inputs = torch.tensor([2.0, 3.0, 5.0])
    out_index, in_index, score = top_kronecker_pairs(output, inputs, 3)
    assert score.tolist() == [20.0, 12.0, 8.0]
    assert list(zip(out_index.tolist(), in_index.tolist(), strict=True)) == [
        (1, 2),
        (1, 1),
        (1, 0),
    ]


def test_atlas_apply_and_adjoint_are_exact() -> None:
    generator = torch.Generator().manual_seed(11)
    inputs = torch.randn(32, 5, generator=generator)
    errors = torch.randn(32, 4, generator=generator)
    atlas = KroneckerAtlas.from_second_moments(
        empirical_second_moment(inputs),
        empirical_second_moment(errors),
        coordinate_count=7,
    )
    coordinates = torch.randn(7, generator=generator)
    cotangent = torch.randn(4, 5, generator=generator)
    left = (atlas.apply(coordinates) * cotangent).sum()
    right = (coordinates * atlas.adjoint(cotangent)).sum()
    torch.testing.assert_close(left, right, atol=2e-5, rtol=2e-5)


def test_identical_atlas_overlap_is_one() -> None:
    generator = torch.Generator().manual_seed(17)
    inputs = torch.randn(48, 6, generator=generator)
    errors = torch.randn(48, 5, generator=generator)
    atlas = KroneckerAtlas.from_second_moments(
        empirical_second_moment(inputs),
        empirical_second_moment(errors),
        coordinate_count=11,
    )
    assert abs(kronecker_subspace_overlap(atlas, atlas, chunk_size=3) - 1.0) < 1e-5


def test_second_moment_is_uncentered() -> None:
    rows = torch.tensor([[1.0, 0.0], [3.0, 2.0]])
    expected = torch.tensor([[5.0, 3.0], [3.0, 2.0]])
    torch.testing.assert_close(empirical_second_moment(rows), expected)

