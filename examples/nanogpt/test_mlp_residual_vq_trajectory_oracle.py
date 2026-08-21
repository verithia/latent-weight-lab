from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_residual_vq_trajectory_oracle import (
    encode_decode,
    expert_function_and_jvp,
    fit_codebooks,
    recovery,
)


def test_rvq_exactly_recovers_two_atom_blocks() -> None:
    atoms = torch.eye(8)
    values = torch.stack((2.0 * atoms[1] - 0.5 * atoms[6], -atoms[3] + atoms[7]))
    decoded, _, _ = encode_decode(values, [atoms, atoms], chunk_size=2)
    assert torch.allclose(values, decoded)
    assert recovery(values, decoded) == 1.0


def test_fitted_codebook_is_finite_and_improves_over_zero() -> None:
    generator = torch.Generator().manual_seed(7)
    source = torch.randn(64, 16, generator=generator)
    codebooks = fit_codebooks(
        source,
        stages=2,
        atoms=8,
        iterations=3,
        seed=11,
        chunk_size=32,
    )
    decoded, _, _ = encode_decode(source, codebooks, chunk_size=32)
    assert torch.isfinite(decoded).all()
    assert recovery(source, decoded) > 0.0


def test_expert_function_jvp_matches_finite_difference() -> None:
    generator = torch.Generator().manual_seed(13)
    c_fc = torch.randn(2, 12, 6, generator=generator) * 0.1
    c_proj = torch.randn(2, 6, 12, generator=generator) * 0.1
    inputs = torch.randn(2, 4, 6, generator=generator)
    directions = torch.randn(2, 4, 6, generator=generator)
    output, jvp = expert_function_and_jvp(c_fc, c_proj, inputs, directions)
    epsilon = 1e-3
    plus, _ = expert_function_and_jvp(c_fc, c_proj, inputs + epsilon * directions, directions)
    minus, _ = expert_function_and_jvp(c_fc, c_proj, inputs - epsilon * directions, directions)
    finite_difference = (plus - minus) / (2.0 * epsilon)
    assert torch.allclose(output, expert_function_and_jvp(c_fc, c_proj, inputs, directions)[0])
    assert torch.allclose(jvp, finite_difference, atol=2e-4, rtol=2e-3)
