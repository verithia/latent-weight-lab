from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from examples.nanogpt.analyze_layer_private_product_quantized_mlp import (
    QuantizedDenseMLPFamily,
    signed_spherical_product_quantize,
    stack_optional_dense_gains,
)


def test_signed_product_quantizer_recovers_small_direction_dictionary() -> None:
    directions = F.normalize(
        torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [4.0, -1.0, 2.0, 0.5],
                [-2.0, 3.0, 0.5, 1.0],
                [0.5, 1.0, -4.0, 2.0],
            ]
        ),
        dim=1,
    )
    scales = torch.linspace(-3.0, 3.0, 64)
    vectors = torch.stack(
        [scales[index] * directions[index % 4] for index in range(64)]
    )
    codebook, codes, amplitudes, decoded, metrics = (
        signed_spherical_product_quantize(
            vectors.reshape(16, 16),
            block_length=4,
            codebook_size=4,
            sample_vectors=64,
            iterations=6,
            assignment_chunk=32,
            seed=31,
        )
    )
    assert codebook.dtype == torch.bfloat16
    assert codes.dtype == torch.uint8
    assert amplitudes.dtype == torch.bfloat16
    assert metrics["weight_energy_recovery"] > 0.999
    torch.testing.assert_close(decoded, vectors.reshape(16, 16), rtol=0.01, atol=0.01)


def test_product_quantizer_state_byte_accounting() -> None:
    weight = torch.randn(32, 16)
    codebook, codes, amplitudes, _, _ = signed_spherical_product_quantize(
        weight,
        block_length=8,
        codebook_size=8,
        sample_vectors=64,
        iterations=2,
        assignment_chunk=32,
        seed=37,
    )
    measured = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (codebook, codes, amplitudes)
    )
    assert measured == 8 * 8 * 2 + 64 + 64 * 2


def test_quantized_family_forward_matches_explicit_dense_mlp() -> None:
    generator = torch.Generator().manual_seed(41)
    c_fc = torch.randn(2, 6, 4, generator=generator)
    c_proj = torch.randn(2, 4, 6, generator=generator)
    pre_gain = torch.randn(2, 6, generator=generator)
    output_log_gain = torch.randn(2, 4, generator=generator)
    family = QuantizedDenseMLPFamily(
        c_fc=c_fc,
        c_proj=c_proj,
        pre_gain=pre_gain,
        output_log_gain=output_log_gain,
    )
    values = torch.randn(7, 4, generator=generator)
    hidden = F.gelu(F.linear(values, c_fc[1]) * pre_gain[1])
    expected = F.linear(hidden, c_proj[1]) * output_log_gain[1].exp()
    torch.testing.assert_close(family.forward_layer(1, values), expected)


def test_absent_dense_gains_materialize_identity_function() -> None:
    c_fc = nn.Linear(4, 6, bias=False)
    c_proj = nn.Linear(6, 4, bias=False)
    blocks = [
        SimpleNamespace(
            mlp=SimpleNamespace(
                c_fc=c_fc,
                c_proj=c_proj,
                pregelu_gain=None,
                residual_output_log_gain=None,
                residual_output_gain_scale=1.0,
            )
        )
    ]
    pre_gain, output_log_gain = stack_optional_dense_gains(blocks)
    torch.testing.assert_close(pre_gain, torch.ones(1, 6))
    torch.testing.assert_close(output_log_gain, torch.zeros(1, 4))


def test_registered_full_model_byte_and_coordinate_accounting() -> None:
    weights = 56_623_104
    blocks = weights // 16
    codebook_values = 24 * 256 * 16
    persistent_bytes = blocks + blocks * 2 + codebook_values * 2
    assert blocks == 3_538_944
    assert persistent_bytes == 10_813_440
    assert 2 * weights / persistent_bytes > 10.47
    assert weights / (blocks + codebook_values) > 15.56
