from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_residual_channelconv_basis import (
    BRANCHES,
    CHANNELS,
    KERNEL_SIZE,
    circular_channel_projection_energy,
    materialize_permuted_channel_convolution,
)


def test_exact_state_is_inside_one_percent() -> None:
    stored = BRANCHES * CHANNELS * CHANNELS * KERNEL_SIZE
    dense = 3072 * 768
    assert stored == 20_480
    assert abs(stored / dense - 0.008680555555555556) < 1e-15


def test_synthetic_family_projection_is_exact() -> None:
    generator = torch.Generator().manual_seed(17)
    kernels = torch.randn(
        BRANCHES, CHANNELS, CHANNELS, KERNEL_SIZE, generator=generator
    )
    matrix = materialize_permuted_channel_convolution(kernels).unsqueeze(0)
    projected = circular_channel_projection_energy(
        matrix, target="mlp.c_fc"
    )
    total = matrix.double().square().sum(dim=(-2, -1))
    torch.testing.assert_close(projected, total, rtol=1e-10, atol=1e-8)


def test_cproj_uses_same_canonical_hidden_gauge() -> None:
    generator = torch.Generator().manual_seed(23)
    kernels = torch.randn(
        BRANCHES, CHANNELS, CHANNELS, KERNEL_SIZE, generator=generator
    )
    matrix = materialize_permuted_channel_convolution(kernels).unsqueeze(0)
    cfc = circular_channel_projection_energy(matrix, target="mlp.c_fc")
    cproj = circular_channel_projection_energy(matrix, target="mlp.c_proj")
    torch.testing.assert_close(cfc, cproj)
