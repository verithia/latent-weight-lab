"""Bit-exact GPU reserved-escape storage for Pair-VQ FP16 momentum.

This module is deliberately a standalone codec first.  It does not change the
optimizer until checkpoint, transition, and performance gates have passed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only on CPU-only installs
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _byte_histogram_kernel(
        raw_ptr,
        counts_ptr,
        n_elements,
        block_words: tl.constexpr,
        TILE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * TILE + tl.arange(0, TILE)
        valid = offsets < n_elements
        high = tl.load(raw_ptr + 2 * offsets + 1, mask=valid, other=0).to(tl.int32)
        regions = offsets // block_words
        tl.atomic_add(counts_ptr + regions * 256 + high, 1, mask=valid)

    @triton.jit
    def _pack_codes_kernel(
        raw_ptr,
        inverse_ptr,
        packed_ptr,
        escape_mask_ptr,
        n_elements,
        block_words: tl.constexpr,
        code_bits: tl.constexpr,
        dictionary_size: tl.constexpr,
        block_local: tl.constexpr,
        GROUP_TILE: tl.constexpr,
    ):
        groups = tl.program_id(0) * GROUP_TILE + tl.arange(0, GROUP_TILE)
        packed = tl.zeros((GROUP_TILE,), dtype=tl.uint64)
        escape_code: tl.constexpr = dictionary_size - 1
        for lane in range(8):
            words = groups * 8 + lane
            valid = words < n_elements
            high = tl.load(raw_ptr + 2 * words + 1, mask=valid, other=0).to(tl.int32)
            regions = words // block_words if block_local else 0
            codes = tl.load(
                inverse_ptr + regions * 256 + high,
                mask=valid,
                other=escape_code,
            ).to(tl.uint64)
            packed |= codes << (lane * code_bits)
            tl.store(
                escape_mask_ptr + words,
                (codes == escape_code).to(tl.uint8),
                mask=valid,
            )
        packed_bytes = (n_elements * code_bits + 7) // 8
        for byte_index in range(code_bits):
            output_offsets = groups * code_bits + byte_index
            tl.store(
                packed_ptr + output_offsets,
                ((packed >> (8 * byte_index)) & 255).to(tl.uint8),
                mask=output_offsets < packed_bytes,
            )

    @triton.jit
    def _unpack_codes_kernel(
        packed_ptr,
        tables_ptr,
        high_ptr,
        escape_mask_ptr,
        n_elements,
        block_words: tl.constexpr,
        code_bits: tl.constexpr,
        dictionary_size: tl.constexpr,
        block_local: tl.constexpr,
        GROUP_TILE: tl.constexpr,
    ):
        groups = tl.program_id(0) * GROUP_TILE + tl.arange(0, GROUP_TILE)
        packed_bytes = (n_elements * code_bits + 7) // 8
        packed = tl.zeros((GROUP_TILE,), dtype=tl.uint64)
        for byte_index in range(code_bits):
            input_offsets = groups * code_bits + byte_index
            byte = tl.load(
                packed_ptr + input_offsets,
                mask=input_offsets < packed_bytes,
                other=0,
            ).to(tl.uint64)
            packed |= byte << (8 * byte_index)
        escape_code: tl.constexpr = dictionary_size - 1
        code_mask: tl.constexpr = dictionary_size - 1
        for lane in range(8):
            words = groups * 8 + lane
            valid = words < n_elements
            codes = ((packed >> (lane * code_bits)) & code_mask).to(tl.int32)
            escaped = codes == escape_code
            regions = words // block_words if block_local else 0
            safe_codes = tl.where(escaped, 0, codes)
            high = tl.load(
                tables_ptr + regions * (dictionary_size - 1) + safe_codes,
                mask=valid,
                other=0,
            )
            tl.store(high_ptr + words, high, mask=valid)
            tl.store(
                escape_mask_ptr + words,
                escaped.to(tl.uint8),
                mask=valid,
            )


@dataclass
class ReservedEscapeState:
    """Persistent tensors for one c_fc or c_proj scope."""

    low_bytes: torch.Tensor
    packed_codes: torch.Tensor
    exception_high_bytes: torch.Tensor
    tables: torch.Tensor
    block_offsets: torch.Tensor
    n_elements: int
    block_words: int
    dictionary_size: int
    block_local: bool

    @property
    def code_bits(self) -> int:
        return self.dictionary_size.bit_length() - 1

    @property
    def persistent_tensor_bytes(self) -> int:
        tensors = (
            self.low_bytes,
            self.packed_codes,
            self.exception_high_bytes,
            self.tables,
            self.block_offsets,
        )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def to_payload(self) -> dict[str, Any]:
        return {
            "low_bytes": self.low_bytes,
            "packed_codes": self.packed_codes,
            "exception_high_bytes": self.exception_high_bytes,
            "tables": self.tables,
            "block_offsets": self.block_offsets,
            "n_elements": self.n_elements,
            "block_words": self.block_words,
            "dictionary_size": self.dictionary_size,
            "block_local": self.block_local,
        }

    @classmethod
    def from_payload(
        cls, payload: dict[str, Any], *, device: torch.device
    ) -> "ReservedEscapeState":
        tensor_fields = {
            "low_bytes": torch.uint8,
            "packed_codes": torch.uint8,
            "exception_high_bytes": torch.uint8,
            "tables": torch.uint8,
            "block_offsets": torch.int32,
        }
        converted = {}
        for name, dtype in tensor_fields.items():
            value = payload.get(name)
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"reserved-escape payload is missing {name}")
            converted[name] = value.to(device=device, dtype=dtype)
        state = cls(
            **converted,
            n_elements=int(payload["n_elements"]),
            block_words=int(payload["block_words"]),
            dictionary_size=int(payload["dictionary_size"]),
            block_local=bool(payload["block_local"]),
        )
        if state.n_elements <= 0 or state.n_elements % state.block_words:
            raise ValueError("reserved-escape payload has an invalid element count")
        if state.dictionary_size not in {8, 16, 32, 64}:
            raise ValueError("reserved-escape payload has an invalid dictionary size")
        if state.low_bytes.numel() != state.n_elements:
            raise ValueError("reserved-escape payload low-byte length mismatch")
        expected_codes = (state.n_elements * state.code_bits + 7) // 8
        if state.packed_codes.numel() != expected_codes:
            raise ValueError("reserved-escape payload code length mismatch")
        expected_blocks = state.n_elements // state.block_words
        if state.block_offsets.numel() != expected_blocks + 1:
            raise ValueError("reserved-escape payload offset length mismatch")
        expected_tables = expected_blocks if state.block_local else 1
        if tuple(state.tables.shape) != (
            expected_tables,
            state.dictionary_size - 1,
        ):
            raise ValueError("reserved-escape payload table shape mismatch")
        return state


def _require_cuda() -> None:
    if triton is None:
        raise RuntimeError("Triton is required for the GPU reserved-escape codec")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the GPU reserved-escape codec")


def _validate_input(values: torch.Tensor, *, block_words: int) -> torch.Tensor:
    _require_cuda()
    if values.device.type != "cuda" or values.dtype != torch.float16:
        raise ValueError("values must be a CUDA float16 tensor")
    flat = values.detach().contiguous().reshape(-1)
    if flat.numel() == 0 or flat.numel() % block_words:
        raise ValueError("the registered codec requires complete decode blocks")
    if block_words % 256:
        raise ValueError("block_words must be divisible by 256")
    return flat


def _dictionary_sizes(scope: str, *, granularity: str) -> tuple[int, ...]:
    if scope == "c_fc":
        return (8, 16, 32) if granularity == "adaptive_block" else (16,)
    if scope == "c_proj":
        return (16, 32, 64) if granularity == "adaptive_block" else (32,)
    raise ValueError(f"unknown MLP scope {scope!r}")


def _select_dictionary_size(
    counts: torch.Tensor,
    *,
    n_elements: int,
    candidates: tuple[int, ...],
) -> int:
    """Choose the candidate with minimum exact variable bytes.

    Low literal bytes and block offsets are common to all candidates, so the
    comparison includes only packed indices, dictionaries, and escaped high
    bytes. Candidate order is ascending and torch.argmin therefore breaks
    exact ties toward the smaller dictionary.
    """

    if len(candidates) == 1:
        return candidates[0]
    regions = counts.shape[0]
    costs = []
    for dictionary_size in candidates:
        retained = torch.topk(
            counts,
            k=dictionary_size - 1,
            dim=1,
            largest=True,
            sorted=False,
        ).values.sum(dtype=torch.int64)
        exceptions = n_elements - retained
        code_bits = dictionary_size.bit_length() - 1
        fixed = (
            (n_elements * code_bits + 7) // 8
            + regions * (dictionary_size - 1)
        )
        costs.append(exceptions + fixed)
    return candidates[int(torch.argmin(torch.stack(costs)).item())]


@torch.no_grad()
def encode_reserved_escape(
    values: torch.Tensor,
    *,
    scope: str,
    granularity: str,
    block_words: int = 4096,
) -> ReservedEscapeState:
    """Encode FP16 words without changing any bit, entirely on the GPU."""

    flat = _validate_input(values, block_words=block_words)
    if granularity not in {"scope", "block", "adaptive_block"}:
        raise ValueError(
            "granularity must be 'scope', 'block', or 'adaptive_block'"
        )
    n_elements = flat.numel()
    n_blocks = n_elements // block_words
    raw = flat.view(torch.uint8)

    block_counts = torch.zeros(
        (n_blocks, 256), device=flat.device, dtype=torch.int32
    )
    histogram_grid = (triton.cdiv(n_elements, 256),)
    _byte_histogram_kernel[histogram_grid](
        raw,
        block_counts,
        n_elements,
        block_words=block_words,
        TILE=256,
    )
    block_local = granularity in {"block", "adaptive_block"}
    counts = (
        block_counts
        if block_local
        else block_counts.sum(dim=0, dtype=torch.int32, keepdim=True)
    )
    dictionary_size = _select_dictionary_size(
        counts,
        n_elements=n_elements,
        candidates=_dictionary_sizes(scope, granularity=granularity),
    )
    code_bits = dictionary_size.bit_length() - 1
    symbols = torch.arange(256, device=flat.device, dtype=torch.int64)
    scores = counts.to(torch.int64) * 512 + (255 - symbols)
    top = torch.topk(
        scores,
        k=dictionary_size - 1,
        dim=1,
        largest=True,
        sorted=True,
    ).indices
    tables = top.to(torch.uint8).contiguous()
    inverse = torch.full(
        (tables.shape[0], 256),
        dictionary_size - 1,
        device=flat.device,
        dtype=torch.uint8,
    )
    code_values = torch.arange(
        dictionary_size - 1, device=flat.device, dtype=torch.uint8
    ).expand_as(tables)
    inverse.scatter_(1, top, code_values)

    low_bytes = raw[0::2].contiguous()
    packed_codes = torch.empty(
        (n_elements * code_bits + 7) // 8,
        device=flat.device,
        dtype=torch.uint8,
    )
    escape_mask = torch.empty(n_elements, device=flat.device, dtype=torch.uint8)
    group_count = triton.cdiv(n_elements, 8)
    pack_grid = (triton.cdiv(group_count, 256),)
    _pack_codes_kernel[pack_grid](
        raw,
        inverse,
        packed_codes,
        escape_mask,
        n_elements,
        block_words=block_words,
        code_bits=code_bits,
        dictionary_size=dictionary_size,
        block_local=block_local,
        GROUP_TILE=256,
    )
    high = raw[1::2]
    escaped = escape_mask.bool()
    exception_high_bytes = high[escaped].contiguous()
    exception_counts = escape_mask.reshape(n_blocks, block_words).sum(
        dim=1, dtype=torch.int32
    )
    block_offsets = torch.empty(
        n_blocks + 1, device=flat.device, dtype=torch.int32
    )
    block_offsets[0] = 0
    torch.cumsum(exception_counts, dim=0, dtype=torch.int32, out=block_offsets[1:])
    return ReservedEscapeState(
        low_bytes=low_bytes,
        packed_codes=packed_codes,
        exception_high_bytes=exception_high_bytes,
        tables=tables,
        block_offsets=block_offsets,
        n_elements=n_elements,
        block_words=block_words,
        dictionary_size=dictionary_size,
        block_local=block_local,
    )


@torch.no_grad()
def decode_reserved_escape(state: ReservedEscapeState) -> torch.Tensor:
    """Decode one scope into a flat FP16 tensor with identical word bits."""

    _require_cuda()
    n_elements = state.n_elements
    high = torch.empty(n_elements, device=state.low_bytes.device, dtype=torch.uint8)
    escape_mask = torch.empty_like(high)
    group_count = triton.cdiv(n_elements, 8)
    unpack_grid = (triton.cdiv(group_count, 256),)
    _unpack_codes_kernel[unpack_grid](
        state.packed_codes,
        state.tables,
        high,
        escape_mask,
        n_elements,
        block_words=state.block_words,
        code_bits=state.code_bits,
        dictionary_size=state.dictionary_size,
        block_local=state.block_local,
        GROUP_TILE=256,
    )
    escaped = escape_mask.bool()
    high[escaped] = state.exception_high_bytes
    raw = torch.empty(n_elements * 2, device=high.device, dtype=torch.uint8)
    raw[0::2].copy_(state.low_bytes)
    raw[1::2].copy_(high)
    return raw.view(torch.float16)
