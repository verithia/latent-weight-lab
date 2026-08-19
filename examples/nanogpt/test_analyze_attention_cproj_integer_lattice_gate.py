from __future__ import annotations

import torch

from examples.nanogpt.analyze_attention_cproj_integer_lattice_gate import (
    block_absmax,
    fp16_scales,
    quantize_on_lattice,
)


def test_fixed_lattice_decode_is_deterministic() -> None:
    values = torch.tensor([[[0.25, -0.5, 0.75, -1.0]]])
    scales = torch.tensor([[[0.25]]])
    codes, decoded = quantize_on_lattice(values, scales, qmax=7)
    codes2, decoded2 = quantize_on_lattice(values, scales, qmax=7)
    assert torch.equal(codes, codes2)
    assert torch.equal(decoded, decoded2)
    torch.testing.assert_close(decoded, values)


def test_running_max_scale_is_monotone_and_fp16_representable() -> None:
    first = torch.tensor([[[1.0, -2.0, 3.0, -4.0]]])
    second = torch.tensor([[[2.0, -1.0, 6.0, -3.0]]])
    running = block_absmax(first, 4)
    scale1 = fp16_scales(running, 7)
    running.copy_(torch.maximum(running, block_absmax(second, 4)))
    scale2 = fp16_scales(running, 7)
    assert bool((scale2 >= scale1).all())
    assert torch.equal(scale2, scale2.half().float())


def test_zero_scale_decodes_to_zero() -> None:
    values = torch.zeros(2, 4, 4)
    scales = torch.zeros(2, 2, 1)
    codes, decoded = quantize_on_lattice(values, scales, qmax=7)
    assert not bool(codes.any())
    assert not bool(decoded.any())
