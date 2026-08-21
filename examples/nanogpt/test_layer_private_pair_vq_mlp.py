from __future__ import annotations

import torch

from examples.nanogpt.analyze_layer_private_pair_vq_mlp import (
    euclidean_pair_quantize,
)


def test_pair_vq_recovers_finite_pair_dictionary() -> None:
    generator = torch.Generator().manual_seed(71)
    dictionary = torch.randn(32, 2, generator=generator)
    codes = torch.randint(0, 32, (4096,), generator=generator)
    vectors = dictionary.index_select(0, codes)
    codebook, compact_codes, decoded, metrics = euclidean_pair_quantize(
        vectors.reshape(128, 64),
        vector_length=2,
        codebook_size=256,
        sample_vectors=4096,
        iterations=8,
        assignment_chunk=512,
        seed=73,
    )
    assert codebook.dtype == torch.bfloat16
    assert compact_codes.dtype == torch.uint8
    assert metrics["weight_energy_recovery"] > 0.999
    torch.testing.assert_close(decoded, vectors.reshape(128, 64), rtol=0.02, atol=0.02)


def test_pair_vq_state_byte_accounting() -> None:
    weight = torch.randn(32, 16)
    codebook, codes, _, _ = euclidean_pair_quantize(
        weight,
        vector_length=2,
        codebook_size=256,
        sample_vectors=256,
        iterations=2,
        assignment_chunk=128,
        seed=79,
    )
    measured = sum(
        tensor.numel() * tensor.element_size() for tensor in (codebook, codes)
    )
    assert measured == 256 * 2 * 2 + 256


def test_registered_full_model_pair_vq_accounting() -> None:
    weights = 56_623_104
    pairs = weights // 2
    codebook_values = 24 * 256 * 2
    persistent_bytes = pairs + codebook_values * 2
    assert pairs == 28_311_552
    assert persistent_bytes == 28_336_128
    assert 2 * weights / persistent_bytes > 3.99
    assert weights / codebook_values == 4608
