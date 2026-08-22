from __future__ import annotations

import numpy as np
import pytest
import torch

from examples.nanogpt.pair_vq_fp16_reserved_escape_cuda import (
    decode_reserved_escape,
    encode_reserved_escape,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA reserved-escape test"
)


def _fp16_from_words(words: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(words.astype(np.uint16, copy=False)).view(torch.float16).cuda()


@pytest.mark.parametrize("scope", ["c_fc", "c_proj"])
@pytest.mark.parametrize("granularity", ["scope", "block"])
def test_reserved_escape_cuda_random_words_are_bit_exact(
    scope: str, granularity: str
) -> None:
    rng = np.random.default_rng(20260822)
    source = _fp16_from_words(rng.integers(0, 65536, size=8192, dtype=np.uint16))
    state = encode_reserved_escape(
        source, scope=scope, granularity=granularity, block_words=4096
    )
    recovered = decode_reserved_escape(state)
    assert torch.equal(recovered.view(torch.uint8), source.view(torch.uint8))
    assert state.persistent_tensor_bytes <= source.numel() * 2 + 8192


@pytest.mark.parametrize("scope", ["c_fc", "c_proj"])
def test_reserved_escape_cuda_preserves_special_payloads(scope: str) -> None:
    specials = np.array(
        [
            0x0000,
            0x8000,
            0x0001,
            0x03FF,
            0x0400,
            0x7BFF,
            0x7C00,
            0xFC00,
            0x7E00,
            0x7D01,
            0xFE00,
            0xFFFF,
        ],
        dtype=np.uint16,
    )
    words = np.resize(specials, 4096)
    source = _fp16_from_words(words)
    state = encode_reserved_escape(
        source, scope=scope, granularity="block", block_words=4096
    )
    recovered = decode_reserved_escape(state)
    assert torch.equal(recovered.view(torch.uint8), source.view(torch.uint8))


def test_reserved_escape_cuda_rejects_nonregistered_shapes() -> None:
    source = torch.zeros(4095, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="complete decode blocks"):
        encode_reserved_escape(
            source, scope="c_fc", granularity="scope", block_words=4096
        )

