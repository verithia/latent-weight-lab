from __future__ import annotations

from dataclasses import asdict

import pytest
import torch

from examples.nanogpt.analyze_attention_stepzero_functional_atlas import (
    KroneckerAtlas,
    empirical_second_moment,
    kronecker_subspace_overlap,
    model_from_exact_stepzero_targets,
    top_kronecker_pairs,
)
from examples.nanogpt.model import GPT, GPTConfig


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


def test_exact_stepzero_reconstruction_checks_stored_targets() -> None:
    config = GPTConfig(
        block_size=8,
        vocab_size=32,
        n_layer=1,
        n_head=1,
        n_embd=8,
        bias=False,
    )
    seed = 29
    torch.manual_seed(seed)
    original = GPT(config)
    stored = {
        name: value.detach().clone()
        for name, value in original.named_parameters()
        if name.endswith("attn.c_attn.weight")
        or name.endswith("attn.c_proj.weight")
    }
    payload = {
        "step": 0,
        "model_config": asdict(config),
        "parameters": stored,
    }
    reconstructed = model_from_exact_stepzero_targets(payload, "cpu", seed)
    names = dict(reconstructed.named_parameters())
    assert all(torch.equal(names[name], value) for name, value in stored.items())

    corrupted = {**stored}
    target = next(iter(corrupted))
    corrupted[target] = corrupted[target].clone()
    corrupted[target].view(-1)[0] += 1.0
    with pytest.raises(ValueError, match="does not match stored targets"):
        model_from_exact_stepzero_targets(
            {**payload, "parameters": corrupted}, "cpu", seed
        )
