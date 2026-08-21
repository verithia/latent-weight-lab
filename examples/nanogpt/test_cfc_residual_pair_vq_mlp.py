from __future__ import annotations

import torch

from examples.nanogpt.analyze_cfc_residual_pair_vq_mlp import _matrix_metric
from examples.nanogpt.analyze_layer_private_pair_vq_mlp import (
    euclidean_pair_quantize,
)


def test_residual_pair_stage_improves_first_stage_recovery() -> None:
    generator = torch.Generator().manual_seed(89)
    target = torch.randn(256, 16, generator=generator)
    first = target + 0.1 * torch.randn(256, 16, generator=generator)
    residual = target - first
    _, _, decoded_residual, _ = euclidean_pair_quantize(
        residual,
        vector_length=2,
        codebook_size=256,
        sample_vectors=2048,
        iterations=6,
        assignment_chunk=1024,
        seed=97,
    )
    before = _matrix_metric(target, first)["weight_energy_recovery"]
    after = _matrix_metric(target, first + decoded_residual)[
        "weight_energy_recovery"
    ]
    assert after > before
    assert after > 0.999


def test_registered_asymmetric_residual_state_accounting() -> None:
    weights = 56_623_104
    pair_codes_per_side = weights // 4
    code_bytes = 3 * pair_codes_per_side
    codebook_bytes = 3 * 12 * 256 * 2 * 2
    persistent_bytes = code_bytes + codebook_bytes
    assert code_bytes == 42_467_328
    assert codebook_bytes == 36_864
    assert persistent_bytes == 42_504_192
    assert 2 * weights / persistent_bytes > 2.66
    assert weights / (3 * 12 * 256 * 2) == 3072
