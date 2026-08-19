from __future__ import annotations

import torch

from examples.nanogpt.analyze_attention_cproj_lowbit_polar_gate import (
    normalize_family,
    quantize_blocks,
    theoretical_storage,
)


def test_registered_codecs_are_finite_and_full_shape() -> None:
    generator = torch.Generator().manual_seed(29)
    values = torch.randn(2, 16, 16, generator=generator)
    for codec in ("binary", "ternary", "int4"):
        reconstructed, stats = quantize_blocks(
            values,
            codec=codec,
            block_size=64,
            ternary_threshold_rms=0.6,
        )
        assert reconstructed.shape == values.shape
        assert torch.isfinite(reconstructed).all()
        assert 0.0 <= stats["zero_fraction"] <= 1.0


def test_binary_scale_is_blockwise_least_squares() -> None:
    values = torch.tensor([[[1.0, -2.0, 3.0, -4.0]]])
    reconstructed, _ = quantize_blocks(
        values,
        codec="binary",
        block_size=4,
        ternary_threshold_rms=0.6,
    )
    torch.testing.assert_close(
        reconstructed,
        torch.tensor([[[2.5, -2.5, 2.5, -2.5]]]),
    )


def test_family_radius_and_storage_accounting() -> None:
    target = torch.arange(1, 17, dtype=torch.float32).reshape(1, 4, 4)
    raw, _ = quantize_blocks(
        target,
        codec="binary",
        block_size=8,
        ternary_threshold_rms=0.6,
    )
    prediction, scale = normalize_family(target, raw, 1.0)
    torch.testing.assert_close(prediction.norm(), target.norm())
    assert scale > 0.0
    storage = theoretical_storage(elements=4096, bits=1, block_size=4096)
    assert storage["code_bytes"] == 512
    assert storage["fp16_scale_bytes"] == 2
    assert storage["storage_ratio"] < 0.032
