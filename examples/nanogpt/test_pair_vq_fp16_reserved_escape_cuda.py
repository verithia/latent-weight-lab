from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from examples.nanogpt.pair_vq_fp16_reserved_escape_cuda import (
    decode_reserved_escape,
    encode_reserved_escape,
)
from examples.nanogpt.muon_pair_vq import MuonPairVQ, MuonPairVQLinear


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


def _make_optimizer_pair(
    granularity: str,
) -> tuple[list[MuonPairVQLinear], MuonPairVQ]:
    modules = [
        MuonPairVQLinear(
            64,
            64,
            bias=False,
            stages=1,
            base_seed=2201 + index,
            weight_std=0.02,
            layer_id=index,
            fast_residual=True,
            fp16_ambient_momentum=True,
            fp16_reserved_escape_granularity=granularity,
            reserved_escape_scope=scope,
        ).cuda()
        for index, scope in enumerate(("c_fc", "c_proj"))
    ]
    optimizer = MuonPairVQ(
        modules, lr=0.01, momentum=0.95, weight_decay=0.1, ns_steps=1
    )
    return modules, optimizer


@pytest.mark.parametrize("granularity", ["scope", "block"])
def test_reserved_escape_optimizer_transition_and_resume_are_bit_exact(
    granularity: str,
) -> None:
    torch.manual_seed(2203)
    raw_modules, raw_optimizer = _make_optimizer_pair("")
    compact_modules, compact_optimizer = _make_optimizer_pair(granularity)
    for raw, compact in zip(raw_modules, compact_modules, strict=True):
        compact.load_state_dict(raw.state_dict(), strict=True)

    for _ in range(3):
        for raw, compact in zip(raw_modules, compact_modules, strict=True):
            gradient = torch.randn_like(raw.weight)
            raw.weight.grad = gradient.clone()
            compact.weight.grad = gradient.clone()
        raw_optimizer.step()
        compact_optimizer.step()
        decoded = compact_optimizer._decode_reserved_escape_momentum()
        for raw, compact in zip(raw_modules, compact_modules, strict=True):
            torch.testing.assert_close(compact.weight, raw.weight, rtol=0.0, atol=0.0)
            scope, start, stop = compact_optimizer._reserved_escape_slices[
                id(compact.weight)
            ]
            raw_momentum = raw_optimizer.state[raw.weight]["ambient_momentum"]
            assert torch.equal(
                decoded[scope][start:stop].view_as(compact.weight).view(torch.uint8),
                raw_momentum.view(torch.uint8),
            )
        assert not any(
            "ambient_momentum" in state
            for state in compact_optimizer.state.values()
        )

    model_state = [copy.deepcopy(module.state_dict()) for module in compact_modules]
    optimizer_state = copy.deepcopy(compact_optimizer.state_dict())
    restored_modules, restored_optimizer = _make_optimizer_pair(granularity)
    for restored, state in zip(restored_modules, model_state, strict=True):
        restored.load_state_dict(state, strict=True)
    restored_optimizer.load_state_dict(optimizer_state)
    for compact, restored in zip(compact_modules, restored_modules, strict=True):
        gradient = torch.randn_like(compact.weight)
        compact.weight.grad = gradient.clone()
        restored.weight.grad = gradient.clone()
    compact_optimizer.step()
    restored_optimizer.step()
    for compact, restored in zip(compact_modules, restored_modules, strict=True):
        torch.testing.assert_close(
            restored.weight, compact.weight, rtol=0.0, atol=0.0
        )
    compact_payload = compact_optimizer.state[
        compact_optimizer._reserved_escape_owner
    ]["reserved_escape_momentum"]
    restored_payload = restored_optimizer.state[
        restored_optimizer._reserved_escape_owner
    ]["reserved_escape_momentum"]
    assert compact_payload.keys() == restored_payload.keys()
    for scope in compact_payload:
        assert compact_payload[scope].keys() == restored_payload[scope].keys()
        for key, value in compact_payload[scope].items():
            restored_value = restored_payload[scope][key]
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, restored_value)
            else:
                assert value == restored_value
