from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_residual_implicit_coordinate_basis import (
    axis_features,
    decoder_scalar_count,
    exact_subspace_capture,
    initialization_features,
    maximum_width,
)


def test_axis_features_are_deterministic_and_hash_free() -> None:
    first = axis_features(8, maximum_frequencies=3, device="cpu")
    second = axis_features(8, maximum_frequencies=3, device="cpu")
    assert torch.equal(first, second)
    assert first.shape == (8, 10)


def test_initialization_features_are_deterministic_and_compact() -> None:
    weight = torch.linspace(-0.1, 0.1, 12)
    first = initialization_features(weight, frequencies=3)
    second = initialization_features(weight, frequencies=3)
    assert torch.equal(first, second)
    assert first.shape == (12, 11)


def test_maximum_width_respects_complete_budget() -> None:
    width, stored = maximum_width(64, 16, 10_000)
    assert stored == decoder_scalar_count(64, width, 16)
    assert stored <= 10_000
    assert decoder_scalar_count(64, width + 1, 16) > 10_000


def test_best_remixing_recovers_arbitrarily_mixed_span() -> None:
    torch.manual_seed(5)
    targets, _ = torch.linalg.qr(torch.randn(100, 4))
    mixing = torch.randn(4, 4)
    generated = targets @ mixing
    weighted, minimum, maximum, _captures = exact_subspace_capture(
        generated, targets, torch.tensor([0.4, 0.3, 0.2, 0.1])
    )
    assert weighted > 0.999
    assert minimum > 0.999
    assert maximum <= 1.0
