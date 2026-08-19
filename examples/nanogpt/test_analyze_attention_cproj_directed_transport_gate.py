from __future__ import annotations

import torch

from examples.nanogpt.analyze_attention_cproj_directed_transport_gate import (
    build_candidate,
    directed_pass,
)


def test_directed_pass_shapes_and_finiteness() -> None:
    generator = torch.Generator().manual_seed(17)
    source = torch.randn(2, 12, 12, generator=generator) * 0.02
    target = torch.randn(2, 12, 12, generator=generator) * 1e-3
    for side in ("input", "output"):
        prediction, rows = directed_pass(
            source,
            target,
            side=side,
            schedule=[2, 2],
            ridge_ratio=1e-6,
            chunk_size=6,
        )
        assert prediction.shape == source.shape
        assert torch.isfinite(prediction).all()
        assert len(rows) == 2


def test_candidate_family_radius_is_exact() -> None:
    generator = torch.Generator().manual_seed(19)
    source = torch.randn(3, 12, 12, generator=generator) * 0.02
    target = torch.randn(3, 12, 12, generator=generator) * 1e-3
    passes = [
        {"side": "input", "schedule": [2, 2]},
        {"side": "output", "schedule": [2, 2]},
    ]
    prediction, diagnostics, scale = build_candidate(
        source,
        target,
        passes,
        ridge_ratio=1e-6,
        chunk_size=6,
        family_radius_ratio=1.0,
    )
    torch.testing.assert_close(prediction.norm(), target.norm(), rtol=1e-5, atol=1e-8)
    assert len(diagnostics) == 2
    assert scale > 0.0
