from __future__ import annotations

import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
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


def block_fht_grad_latent(
    latent: torch.Tensor,
    grad_out: torch.Tensor,
    size: int,
    layers: int,
    seed: int,
    start: int = 0,
    stop: int | None = None,
) -> torch.Tensor:
    stop = int(size) if stop is None else int(stop)
    grad_out = grad_out.contiguous().to(dtype=latent.dtype)
    ext = _load_block_fht_ext() if latent.is_cuda and latent.dtype == torch.float32 else None
    if ext is not None:
        return ext.backward(
            grad_out,
            latent.numel(),
            int(size),
            int(layers),
            int(seed),
            int(start),
            int(stop),
        )
    with torch.enable_grad():
        latent_for_grad = latent.detach().requires_grad_(True)
        weight_flat = block_fht_slice(
            latent_for_grad,
            int(size),
            int(layers),
            int(seed),
            int(start),
            int(stop),
        )
        return torch.autograd.grad(
            weight_flat,
            latent_for_grad,
            grad_out,
            retain_graph=False,
            allow_unused=False,
        )[0]


def block_fht_linear_forward(
    input: torch.Tensor,
    latent: torch.Tensor,
    out_features: int,
    layers: int,
    seed: int,
) -> torch.Tensor:
    if input.dim() < 2:
        raise ValueError("input must have at least 2 dimensions")
    in_features = input.shape[-1]
    flat_input = input.reshape(-1, in_features).contiguous()
    ext = (
        _load_block_fht_ext()
        if flat_input.is_cuda and latent.is_cuda and flat_input.dtype == torch.float32 and latent.dtype == torch.float32
        else None
    )
    size = int(in_features) * int(out_features)
    if ext is not None and next_power_of_two(latent.numel()) <= 16384:
        out = ext.linear_forward(flat_input, latent.contiguous(), int(out_features), int(layers), int(seed))
    else:
        weight = block_fht_slice(latent, size, int(layers), int(seed), 0, size).view(
            int(out_features), int(in_features)
        )
        out = F.linear(flat_input, weight)
    return out.reshape(*input.shape[:-1], int(out_features))


class _BlockFHTLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        latent: torch.Tensor,
        bias: torch.Tensor | None,
        in_features: int,
        out_features: int,
        size: int,
        layers: int,
        seed: int,
    ) -> torch.Tensor:
        weight_flat = block_fht_slice(latent, int(size), int(layers), int(seed), 0, int(size))
        weight = weight_flat.view(int(out_features), int(in_features)).to(dtype=input.dtype)
        output = F.linear(input, weight, bias)
        ctx.save_for_backward(input, latent)
        ctx.has_bias = bias is not None
        ctx.in_features = int(in_features)
        ctx.out_features = int(out_features)
        ctx.size = int(size)
        ctx.layers = int(layers)
        ctx.seed = int(seed)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, latent = ctx.saved_tensors
        grad_input = grad_latent = grad_bias = None
        grad_2d = grad_output.reshape(-1, ctx.out_features)
        input_2d = input.reshape(-1, ctx.in_features)

        if ctx.needs_input_grad[0]:
            with torch.no_grad():
                weight_flat = block_fht_slice(latent, ctx.size, ctx.layers, ctx.seed, 0, ctx.size)
                weight = weight_flat.view(ctx.out_features, ctx.in_features).to(dtype=grad_output.dtype)
            grad_input = grad_2d.matmul(weight).reshape_as(input)

        if ctx.needs_input_grad[1]:
            grad_weight = grad_2d.transpose(0, 1).to(dtype=latent.dtype).matmul(
                input_2d.to(dtype=latent.dtype)
            )
            grad_latent = block_fht_grad_latent(
                latent,
                grad_weight.reshape(-1),
                ctx.size,
                ctx.layers,
                ctx.seed,
            )

        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = grad_2d.sum(dim=0).to(dtype=grad_output.dtype)

        return grad_input, grad_latent, grad_bias, None, None, None, None, None


class BlockFHTLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        latent_dim: int | None = None,
        latent_ratio: float = 0.01,
        layers: int = 3,
        seed: int = 0,
        latent_init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        size = self.in_features * self.out_features
        if latent_dim is None:
            latent_dim = max(1, round(size * float(latent_ratio)))
        self.generator = BlockFHT(
            int(latent_dim),
            size=size,
            layers=layers,
            seed=seed,
            latent_init_std=latent_init_std,
        )
        self.bias = nn.Parameter(torch.zeros(self.out_features)) if bias else None
        self._cached_weight: torch.Tensor | None = None

    @property
    def weight(self) -> torch.Tensor:
        return self.generator().view(self.out_features, self.in_features)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self._cached_weight is not None:
            return F.linear(input, self._cached_weight, self.bias)
        return _BlockFHTLinearFn.apply(
            input,
            self.generator.latent,
            self.bias,
            self.in_features,
            self.out_features,
            self.generator.size,
            self.generator.layers,
            self.generator.seed,
        )

    def forward_fused(self, input: torch.Tensor) -> torch.Tensor:
        out = block_fht_linear_forward(
            input,
            self.generator.latent,
            self.out_features,
            self.generator.layers,
            self.generator.seed,
        )
        if self.bias is not None:
            out = out + self.bias
        return out

    def materialize_weight_cache(self, dtype: torch.dtype | None = None) -> None:
        if self._cached_weight is not None:
            return
        with torch.no_grad():
            weight = self.weight
            if dtype is not None:
                weight = weight.to(dtype=dtype)
        self._cached_weight = weight.detach().requires_grad_(True)

    def flush_weight_cache_to_latent_grad(self) -> None:
        if self._cached_weight is None:
            return
        if self._cached_weight.grad is not None:
            grad_weight = self._cached_weight.grad.reshape(-1).to(dtype=self.generator.latent.dtype)
            grad_latent = block_fht_grad_latent(
                self.generator.latent,
                grad_weight,
                self.generator.size,
                self.generator.layers,
                self.generator.seed,
            )
            if self.generator.latent.grad is None:
                self.generator.latent.grad = grad_latent
            else:
                self.generator.latent.grad.add_(grad_latent)
        self._cached_weight = None


def prepare_block_fht_weight_cache(model: nn.Module, dtype: torch.dtype | None = None) -> None:
    for module in model.modules():
        if isinstance(module, BlockFHTLinear):
            module.materialize_weight_cache(dtype=dtype)


def flush_block_fht_weight_cache(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, BlockFHTLinear):
            module.flush_weight_cache_to_latent_grad()


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
