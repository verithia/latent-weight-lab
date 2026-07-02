from __future__ import annotations

import math
import os
from pathlib import Path

import torch
from torch import nn


def next_power_of_two(value: int) -> int:
    if value <= 0:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()


def normalized_fht_last_dim(values: torch.Tensor) -> torch.Tensor:
    size = values.shape[-1]
    if size <= 0 or size & (size - 1):
        raise ValueError("FHT size must be a positive power of two")
    output = values.clone()
    step = 1
    while step < size:
        shape = output.shape
        output = output.reshape(*shape[:-1], -1, 2, step)
        first = output[..., 0, :].clone()
        second = output[..., 1, :].clone()
        output[..., 0, :] = first + second
        output[..., 1, :] = first - second
        output = output.reshape(shape)
        step *= 2
    return output / math.sqrt(size)


def lowbias32(x: int) -> int:
    x &= 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= x >> 16
    return x & 0xFFFFFFFF


def sign_word_for(seed: int, block: int, layer: int, word: int) -> int:
    seed32 = (int(seed) ^ (int(seed) >> 32)) & 0xFFFFFFFF
    x = seed32
    x ^= (0x9E3779B9 * (int(block) + 1)) & 0xFFFFFFFF
    x ^= (0x85EBCA6B * (int(layer) + 1)) & 0xFFFFFFFF
    x ^= (0xC2B2AE35 * (int(word) + 1)) & 0xFFFFFFFF
    return lowbias32(x)


def signs_for(
    reference: torch.Tensor, block: int, layer: int, seed: int, block_size: int
) -> torch.Tensor:
    values = []
    for pos in range(block_size):
        bits = sign_word_for(seed, block, layer, pos >> 5)
        values.append(1.0 if ((bits >> (pos & 31)) & 1) else -1.0)
    return torch.tensor(values, device=reference.device, dtype=reference.dtype)


def block_fht_slice_torch(
    latent: torch.Tensor,
    size: int,
    layers: int,
    seed: int,
    start: int,
    stop: int,
) -> torch.Tensor:
    if not 0 <= start <= stop <= size:
        raise ValueError(f"invalid slice [{start}, {stop}) for size {size}")
    block_size = next_power_of_two(latent.numel())
    pieces = []
    first_block = start // block_size
    last_block = (stop + block_size - 1) // block_size
    for block in range(first_block, last_block):
        block_start = block * block_size
        local_start = max(start - block_start, 0)
        local_stop = min(stop - block_start, block_size)
        values = latent.new_zeros(block_size)
        values[: latent.numel()] = latent
        values = values * signs_for(latent, block, 0, seed, block_size)
        for layer in range(layers):
            values = normalized_fht_last_dim(values)
            values = values * signs_for(latent, block, layer + 1, seed, block_size)
        pieces.append(values[local_start:local_stop])
    return torch.cat(pieces) if pieces else latent.new_empty(0)


_BLOCK_FHT_EXT = None
_BLOCK_FHT_EXT_ERROR: Exception | None = None


def _load_block_fht_ext():
    global _BLOCK_FHT_EXT, _BLOCK_FHT_EXT_ERROR
    if _BLOCK_FHT_EXT is not None or _BLOCK_FHT_EXT_ERROR is not None:
        return _BLOCK_FHT_EXT
    try:
        from torch.utils.cpp_extension import load

        root = Path(__file__).resolve().parents[1]
        cuda_home = os.environ.get("CUDA_HOME")
        if cuda_home:
            os.environ["PATH"] = f"{cuda_home}/bin:" + os.environ.get("PATH", "")
        try:
            import ninja

            os.environ["PATH"] = f"{ninja.BIN_DIR}:" + os.environ.get("PATH", "")
        except Exception:
            pass
        _BLOCK_FHT_EXT = load(
            name="latent_weight_lab_block_fht_ext_scaled_v2",
            sources=[
                str(root / "csrc" / "block_fht_ext.cpp"),
                str(root / "csrc" / "block_fht_ext_cuda.cu"),
            ],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            extra_cflags=["-O3"],
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001
        _BLOCK_FHT_EXT_ERROR = exc
        _BLOCK_FHT_EXT = None
    return _BLOCK_FHT_EXT


class _BlockFHTSliceFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        latent: torch.Tensor,
        size: int,
        layers: int,
        seed: int,
        start: int,
        stop: int,
    ) -> torch.Tensor:
        latent = latent.contiguous()
        ext = _load_block_fht_ext() if latent.is_cuda and latent.dtype == torch.float32 else None
        ctx.latent_size = latent.numel()
        ctx.size = int(size)
        ctx.layers = int(layers)
        ctx.seed = int(seed)
        ctx.start = int(start)
        ctx.stop = int(stop)
        ctx.used_ext = ext is not None
        if ext is not None:
            return ext.forward(latent, ctx.size, ctx.layers, ctx.seed, ctx.start, ctx.stop)[0]
        with torch.enable_grad():
            detached = latent.detach().requires_grad_(True)
            out = block_fht_slice_torch(
                detached, ctx.size, ctx.layers, ctx.seed, ctx.start, ctx.stop
            )
        ctx.save_for_backward(detached, out)
        return out.detach()

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        grad_out = grad_out.contiguous()
        if ctx.used_ext:
            ext = _load_block_fht_ext()
            grad_latent = ext.backward(
                grad_out,
                ctx.latent_size,
                ctx.size,
                ctx.layers,
                ctx.seed,
                ctx.start,
                ctx.stop,
            )
            return grad_latent, None, None, None, None, None
        detached, out = ctx.saved_tensors
        grad_latent = torch.autograd.grad(out, detached, grad_out, retain_graph=False)[0]
        return grad_latent, None, None, None, None, None


def block_fht_slice(
    latent: torch.Tensor,
    size: int,
    layers: int,
    seed: int,
    start: int,
    stop: int,
) -> torch.Tensor:
    return _BlockFHTSliceFn.apply(latent, int(size), int(layers), int(seed), int(start), int(stop))


class BlockFHT(nn.Module):
    def __init__(
        self,
        latent: int | torch.Tensor | nn.Parameter,
        size: int,
        layers: int = 3,
        seed: int = 0,
        latent_init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if size <= 0:
            raise ValueError("size must be positive")
        if layers not in {1, 2, 3}:
            raise ValueError("layers must be 1, 2, or 3")
        if isinstance(latent, int):
            if latent <= 0:
                raise ValueError("latent size must be positive")
            self.latent_size = int(latent)
            self.latent = nn.Parameter(torch.empty(self.latent_size))
            nn.init.normal_(self.latent, std=float(latent_init_std))
        else:
            if latent.dim() != 1:
                raise ValueError("latent tensor must be 1D")
            self.latent_size = int(latent.numel())
            self.latent = latent if isinstance(latent, nn.Parameter) else nn.Parameter(latent)
        self.size = int(size)
        self.layers = int(layers)
        self.seed = int(seed)
        self.block_size = next_power_of_two(self.latent_size)

    def extra_repr(self) -> str:
        return (
            f"latent_size={self.latent_size}, size={self.size}, layers={self.layers}, "
            f"seed={self.seed}, block_size={self.block_size}"
        )

    def slice(self, start: int, stop: int) -> torch.Tensor:
        return block_fht_slice(self.latent, self.size, self.layers, self.seed, start, stop)

    def forward(self) -> torch.Tensor:
        return self.slice(0, self.size)

    @property
    def cuda_extension_error(self) -> Exception | None:
        return _BLOCK_FHT_EXT_ERROR
