from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_activation_matched_givens_fit import (
    activation_matched_permutations,
)
from examples.nanogpt.analyze_mlp_global_givens_transport_fit import (
    fit_global_givens_transport,
)
from examples.nanogpt.model import LearnedGivensOutputMix


def test_activation_matching_is_unique_and_finds_correlated_pairs() -> None:
    generator = torch.Generator().manual_seed(7)
    latent = torch.randn(256, 4, generator=generator)
    values = torch.stack(
        (
            latent[:, 0],
            latent[:, 0] + 0.01 * torch.randn(256, generator=generator),
            latent[:, 1],
            latent[:, 1] + 0.01 * torch.randn(256, generator=generator),
            latent[:, 2],
            latent[:, 2] + 0.01 * torch.randn(256, generator=generator),
            latent[:, 3],
            latent[:, 3] + 0.01 * torch.randn(256, generator=generator),
        ),
        dim=1,
    )
    permutations, diagnostics = activation_matched_permutations(
        values,
        stages=2,
        neighbors=4,
        seed=11,
    )
    assert permutations.shape == (2, 8)
    expected = {frozenset(pair) for pair in ((0, 1), (2, 3), (4, 5), (6, 7))}
    observed = {
        frozenset(pair.tolist())
        for pair in permutations[0].reshape(-1, 2)
    }
    assert observed == expected
    edges = [
        tuple(sorted(pair.tolist()))
        for row in permutations
        for pair in row.reshape(-1, 2)
    ]
    assert len(edges) == len(set(edges))
    assert len(diagnostics) == 2


def test_custom_connectivity_recovers_exact_flow() -> None:
    torch.manual_seed(13)
    source = torch.randn(32, 8)
    flow = LearnedGivensOutputMix(8, 2, 17)
    permutations = torch.stack(
        (torch.arange(8), torch.tensor([0, 2, 1, 3, 4, 6, 5, 7]))
    )
    with torch.no_grad():
        flow.permutations.copy_(permutations)
        flow.inverse_permutations.copy_(torch.argsort(permutations, dim=1))
        flow.angles.normal_(std=0.1)
        target = flow(source)
    result = fit_global_givens_transport(
        source,
        target,
        stages=2,
        seed=17,
        steps=500,
        learning_rate=0.03,
        permutations=permutations,
    )
    assert result["endpoint_recovery"] > 0.999
