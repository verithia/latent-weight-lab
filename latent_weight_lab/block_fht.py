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
    """Generate the exact ``sign_word_for`` bit pattern without a Python loop.

    The fallback FHT can request tens of thousands of signs per block.  Building
    them one Python scalar at a time dominated CUDA inference on hosts without
    the optional extension, even though the signs are already destined for the
    accelerator.  The uint32 hash is evaluated in vectorized int64 arithmetic
    with an explicit low-32-bit mask after every mixing operation, preserving
    ``sign_word_for`` bit-for-bit.
    """
    mask = (1 << 32) - 1
    words = torch.arange((int(block_size) + 31) // 32, device=reference.device, dtype=torch.int64)
    seed32 = (int(seed) ^ (int(seed) >> 32)) & mask
    mixed = torch.full_like(words, seed32)
    mixed = (mixed ^ ((0x9E3779B9 * (int(block) + 1)) & mask)) & mask
    mixed = (mixed ^ ((0x85EBCA6B * (int(layer) + 1)) & mask)) & mask
    mixed = (mixed ^ ((0xC2B2AE35 * (words + 1)) & mask)) & mask
    mixed = (mixed ^ (mixed >> 16)) & mask
    mixed = (mixed * 0x7FEB352D) & mask
    mixed = (mixed ^ (mixed >> 15)) & mask
    mixed = (mixed * 0x846CA68B) & mask
    mixed = (mixed ^ (mixed >> 16)) & mask
    positions = torch.arange(32, device=reference.device, dtype=torch.int64)
    bits = (mixed.unsqueeze(1) >> positions) & 1
    return (2.0 * bits.reshape(-1)[:block_size] - 1.0).to(dtype=reference.dtype)


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
    latent_flat = latent.reshape(-1)
    block_size = next_power_of_two(latent_flat.numel())
    pieces = []
    first_block = start // block_size
    last_block = (stop + block_size - 1) // block_size
    for block in range(first_block, last_block):
        block_start = block * block_size
        local_start = max(start - block_start, 0)
        local_stop = min(stop - block_start, block_size)
        values = latent_flat.new_zeros(block_size)
        values[: latent_flat.numel()] = latent_flat
        values = values * signs_for(latent_flat, block, 0, seed, block_size)
        for layer in range(layers):
            values = normalized_fht_last_dim(values)
            values = values * signs_for(latent_flat, block, layer + 1, seed, block_size)
        pieces.append(values[local_start:local_stop])
    return torch.cat(pieces) if pieces else latent_flat.new_empty(0)


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
            name="latent_weight_lab_block_fht_ext_scaled_v7",
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
    return _BlockFHTSliceFn.apply(
        latent.reshape(-1),
        int(size),
        int(layers),
        int(seed),
        int(start),
        int(stop),
    )


def _fixed_basis_transform_torch(
    values: torch.Tensor,
    permutations: torch.Tensor,
    signs: torch.Tensor,
    block_size: int,
    *,
    inverse: bool,
    shared_input: bool,
) -> torch.Tensor:
    """Reference fixed signed/permuted block-Hadamard transform."""

    if permutations.dim() == 1:
        permutations = permutations.unsqueeze(0)
    if signs.dim() == 1:
        signs = signs.unsqueeze(0)
    if permutations.shape != signs.shape:
        raise ValueError("permutations and signs must have identical shapes")
    bases, width = permutations.shape
    if values.shape[-1] != width:
        raise ValueError(
            f"basis width mismatch: expected {width}, got {values.shape[-1]}"
        )
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("block_size must be a positive power of two")
    if width % block_size:
        raise ValueError("block_size must divide the basis width")
    if not shared_input and values.shape[0] != bases:
        raise ValueError(
            "per-basis values must have the basis count as their first axis"
        )
    outputs = []
    for basis in range(bases):
        selected = values if shared_input else values[basis]
        permutation = permutations[basis].to(device=values.device)
        sign = signs[basis].to(device=values.device, dtype=values.dtype)
        if inverse:
            transformed = selected * sign
            grouped = transformed.reshape(
                *transformed.shape[:-1], width // block_size, block_size
            )
            transformed = normalized_fht_last_dim(grouped).reshape_as(
                selected
            )
            inverse_permutation = torch.argsort(permutation)
            outputs.append(
                transformed.index_select(-1, inverse_permutation)
            )
        else:
            transformed = selected.index_select(-1, permutation)
            grouped = transformed.reshape(
                *transformed.shape[:-1], width // block_size, block_size
            )
            transformed = normalized_fht_last_dim(grouped).reshape_as(
                selected
            )
            outputs.append(transformed * sign)
    return torch.stack(outputs, dim=0)


class _FixedBasisTransformFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        values: torch.Tensor,
        permutations: torch.Tensor,
        signs: torch.Tensor,
        block_size: int,
        inverse: bool,
        shared_input: bool,
    ) -> torch.Tensor:
        ext = _load_block_fht_ext()
        if ext is None:
            raise RuntimeError(
                "fixed basis CUDA transform extension failed to load"
            )
        ctx.block_size = int(block_size)
        ctx.inverse = bool(inverse)
        ctx.shared_input = bool(shared_input)
        ctx.save_for_backward(permutations, signs)
        return ext.fixed_basis_transform(
            values.contiguous(),
            permutations.contiguous(),
            signs.contiguous(),
            ctx.block_size,
            ctx.inverse,
            ctx.shared_input,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        permutations, signs = ctx.saved_tensors
        ext = _load_block_fht_ext()
        if ext is None:
            raise RuntimeError(
                "fixed basis CUDA transform extension failed to load"
            )
        width = permutations.shape[1]
        flat_grad = grad_output.reshape(-1, width).contiguous()
        grad_values = ext.fixed_basis_transform(
            flat_grad,
            permutations.contiguous(),
            signs.contiguous(),
            ctx.block_size,
            not ctx.inverse,
            False,
        )
        if ctx.shared_input:
            grad_values = grad_values.sum(dim=0)
        else:
            grad_values = grad_values.reshape(-1, width)
        return grad_values, None, None, None, None, None


def fixed_basis_transform(
    values: torch.Tensor,
    permutations: torch.Tensor,
    signs: torch.Tensor,
    block_size: int,
    *,
    inverse: bool = False,
    shared_input: bool = False,
) -> torch.Tensor:
    """Apply one or more fixed signed/permuted block-Hadamard bases.

    A shared input has shape ``[..., width]`` and produces
    ``[bases, ..., width]``. A per-basis input has shape
    ``[bases, ..., width]`` and preserves that shape. CUDA uses one batched
    extension launch; CPU uses the exact PyTorch reference.
    """

    if permutations.dim() == 1:
        permutations = permutations.unsqueeze(0)
    if signs.dim() == 1:
        signs = signs.unsqueeze(0)
    bases, width = permutations.shape
    if values.shape[-1] != width:
        raise ValueError(
            f"basis width mismatch: expected {width}, got {values.shape[-1]}"
        )
    if not shared_input and values.shape[0] != bases:
        raise ValueError(
            "per-basis values must have the basis count as their first axis"
        )
    if values.is_cuda:
        leading = values.shape[:-1]
        if shared_input:
            flattened = values.reshape(-1, width)
            transformed = _FixedBasisTransformFn.apply(
                flattened,
                permutations,
                signs,
                int(block_size),
                bool(inverse),
                True,
            )
            return transformed.reshape(bases, *leading, width)
        flattened = values.reshape(bases * math.prod(leading[1:]), width)
        transformed = _FixedBasisTransformFn.apply(
            flattened,
            permutations,
            signs,
            int(block_size),
            bool(inverse),
            False,
        )
        return transformed.reshape(*leading, width)
    return _fixed_basis_transform_torch(
        values,
        permutations,
        signs,
        int(block_size),
        inverse=bool(inverse),
        shared_input=bool(shared_input),
    )


class _PostGeluMultiheadMixFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        activated: torch.Tensor,
        condition: torch.Tensor,
        slope: torch.Tensor,
        condition_permutations: torch.Tensor,
        condition_signs: torch.Tensor,
        update_permutations: torch.Tensor,
        update_signs: torch.Tensor,
        output_permutations: torch.Tensor,
        output_signs: torch.Tensor,
        block_size: int,
        scale: float,
    ) -> torch.Tensor:
        ext = _load_block_fht_ext()
        if ext is None:
            raise RuntimeError("post-GELU CUDA mixer extension failed to load")
        condition_spectral = ext.fixed_basis_transform(
            condition.reshape(-1, condition.shape[-1]).contiguous(),
            condition_permutations.contiguous(),
            condition_signs.contiguous(),
            int(block_size),
            False,
            True,
        )
        update_spectral = ext.fixed_basis_transform(
            activated.reshape(-1, activated.shape[-1]).contiguous(),
            update_permutations.contiguous(),
            update_signs.contiguous(),
            int(block_size),
            False,
            True,
        )
        correction = ext.postgelu_mix_forward(
            update_spectral,
            condition_spectral,
            slope.contiguous(),
            float(scale),
        )
        transformed = ext.fixed_basis_transform(
            correction.reshape(-1, correction.shape[-1]).contiguous(),
            output_permutations.contiguous(),
            output_signs.contiguous(),
            int(block_size),
            True,
            False,
        )
        output = ext.postgelu_sum_heads(
            transformed,
            activated.reshape(-1, activated.shape[-1]).contiguous(),
            True,
        )
        ctx.block_size = int(block_size)
        ctx.scale = float(scale)
        ctx.activated_shape = tuple(activated.shape)
        ctx.save_for_backward(
            update_spectral,
            condition_spectral,
            slope,
            condition_permutations,
            condition_signs,
            update_permutations,
            update_signs,
            output_permutations,
            output_signs,
        )
        return output.reshape_as(activated)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            update_spectral,
            condition_spectral,
            slope,
            condition_permutations,
            condition_signs,
            update_permutations,
            update_signs,
            output_permutations,
            output_signs,
        ) = ctx.saved_tensors
        ext = _load_block_fht_ext()
        if ext is None:
            raise RuntimeError("post-GELU CUDA mixer extension failed to load")
        width = grad_output.shape[-1]
        flat_grad = grad_output.reshape(-1, width).contiguous()
        grad_correction = ext.fixed_basis_transform(
            flat_grad,
            output_permutations.contiguous(),
            output_signs.contiguous(),
            ctx.block_size,
            False,
            True,
        )
        (
            grad_update_spectral,
            grad_condition_spectral,
            grad_slope,
        ) = ext.postgelu_mix_backward(
            grad_correction,
            update_spectral,
            condition_spectral,
            slope.contiguous(),
            ctx.scale,
        )
        grad_activated_heads = ext.fixed_basis_transform(
            grad_update_spectral.reshape(-1, width).contiguous(),
            update_permutations.contiguous(),
            update_signs.contiguous(),
            ctx.block_size,
            True,
            False,
        )
        grad_condition_heads = ext.fixed_basis_transform(
            grad_condition_spectral.reshape(-1, width).contiguous(),
            condition_permutations.contiguous(),
            condition_signs.contiguous(),
            ctx.block_size,
            True,
            False,
        )
        grad_activated = ext.postgelu_sum_heads(
            grad_activated_heads,
            flat_grad,
            True,
        ).reshape(ctx.activated_shape)
        grad_condition = ext.postgelu_sum_heads(
            grad_condition_heads,
            flat_grad,
            False,
        ).reshape(ctx.activated_shape)
        return (
            grad_activated,
            grad_condition,
            grad_slope,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def postgelu_multihead_mix(
    activated: torch.Tensor,
    condition: torch.Tensor,
    slope: torch.Tensor,
    condition_permutations: torch.Tensor,
    condition_signs: torch.Tensor,
    update_permutations: torch.Tensor,
    update_signs: torch.Tensor,
    output_permutations: torch.Tensor,
    output_signs: torch.Tensor,
    block_size: int,
    scale: float,
) -> torch.Tensor:
    """Apply the exact fixed-basis multihead mixer with a fused CUDA VJP."""

    return _PostGeluMultiheadMixFn.apply(
        activated,
        condition,
        slope,
        condition_permutations,
        condition_signs,
        update_permutations,
        update_signs,
        output_permutations,
        output_signs,
        int(block_size),
        float(scale),
    )


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
    latent_flat = latent.reshape(-1).contiguous()
    ext = _load_block_fht_ext() if latent_flat.is_cuda and latent_flat.dtype == torch.float32 else None
    if ext is not None:
        return ext.backward(
            grad_out,
            latent_flat.numel(),
            int(size),
            int(layers),
            int(seed),
            int(start),
            int(stop),
        ).reshape_as(latent)
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
    weight_scale: float = 1.0,
) -> torch.Tensor:
    if input.dim() < 2:
        raise ValueError("input must have at least 2 dimensions")
    in_features = input.shape[-1]
    flat_input = input.reshape(-1, in_features).contiguous()
    latent_flat = latent.reshape(-1).contiguous()
    ext = (
        _load_block_fht_ext()
        if flat_input.is_cuda
        and latent_flat.is_cuda
        and flat_input.dtype == latent_flat.dtype
        and flat_input.dtype in {torch.bfloat16, torch.float16, torch.float32}
        else None
    )
    size = int(in_features) * int(out_features)
    block_size = next_power_of_two(latent_flat.numel())
    if ext is not None and block_size <= 16384 and block_size % int(in_features) == 0:
        out = ext.linear_forward(
            flat_input,
            latent_flat,
            int(out_features),
            int(layers),
            int(seed),
            float(weight_scale),
        )
    else:
        weight = block_fht_slice(latent, size, int(layers), int(seed), 0, size).view(
            int(out_features), int(in_features)
        )
        weight = weight * float(weight_scale)
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
        weight_scale: float,
        modulation_alpha: float,
    ) -> torch.Tensor:
        weight_flat = block_fht_slice(latent, int(size), int(layers), int(seed), 0, int(size))
        weight_flat = weight_flat * float(weight_scale)
        if float(modulation_alpha) != 0.0:
            weight_flat = weight_flat + float(modulation_alpha) * latent.square().sum()
        weight = weight_flat.view(int(out_features), int(in_features)).to(dtype=input.dtype)
        output = F.linear(input, weight, bias)
        ctx.save_for_backward(input, latent)
        ctx.has_bias = bias is not None
        ctx.in_features = int(in_features)
        ctx.out_features = int(out_features)
        ctx.size = int(size)
        ctx.layers = int(layers)
        ctx.seed = int(seed)
        ctx.weight_scale = float(weight_scale)
        ctx.modulation_alpha = float(modulation_alpha)
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
                weight_flat = weight_flat * ctx.weight_scale
                if ctx.modulation_alpha != 0.0:
                    weight_flat = weight_flat + ctx.modulation_alpha * latent.square().sum()
                weight = weight_flat.view(ctx.out_features, ctx.in_features).to(dtype=grad_output.dtype)
            grad_input = grad_2d.matmul(weight).reshape_as(input)

        if ctx.needs_input_grad[1]:
            grad_weight = grad_2d.transpose(0, 1).to(dtype=latent.dtype).matmul(
                input_2d.to(dtype=latent.dtype)
            )
            grad_latent = ctx.weight_scale * block_fht_grad_latent(
                latent,
                grad_weight.reshape(-1),
                ctx.size,
                ctx.layers,
                ctx.seed,
            )
            if ctx.modulation_alpha != 0.0:
                grad_latent = grad_latent + 2.0 * ctx.modulation_alpha * latent * grad_weight.sum()

        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = grad_2d.sum(dim=0).to(dtype=grad_output.dtype)

        return grad_input, grad_latent, grad_bias, None, None, None, None, None, None, None


class BlockFHTLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        latent_dim: int | None = None,
        latent_shape: tuple[int, int] | None = None,
        latent_ratio: float = 0.01,
        layers: int = 3,
        seed: int = 0,
        latent_init_std: float = 0.02,
        weight_scale: float = 1.0,
        modulation_alpha: float = 0.0,
        modulation_centered: bool = False,
        quadratic_scale: float = 0.0,
        quadratic_seed_offset: int = 104729,
        residual_base_scale: float = 0.0,
        residual_base_std: float = 0.02,
        output_gain: bool = False,
        input_gain: bool = False,
        spectral_rank: int = 0,
        spectral_out_groups: int = 1,
        spectral_in_groups: int = 1,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        size = self.in_features * self.out_features
        if latent_dim is None:
            latent_dim = max(1, round(size * float(latent_ratio)))
        if latent_shape is not None:
            latent_shape = (int(latent_shape[0]), int(latent_shape[1]))
            if min(latent_shape) <= 0:
                raise ValueError("latent_shape dimensions must be positive")
            if math.prod(latent_shape) != int(latent_dim):
                raise ValueError("latent_shape product must equal latent_dim")
            latent: int | tuple[int, int] = latent_shape
        else:
            latent = int(latent_dim)
        self.generator = BlockFHT(
            latent,
            size=size,
            layers=layers,
            seed=seed,
            latent_init_std=latent_init_std,
        )
        self.bias = nn.Parameter(torch.zeros(self.out_features)) if bias else None
        self._cached_weight: torch.Tensor | None = None
        self.weight_scale = float(weight_scale)
        self.modulation_alpha = float(modulation_alpha)
        self.modulation_centered = bool(modulation_centered)
        self.latent_init_std = float(latent_init_std)
        self.quadratic_scale = float(quadratic_scale)
        self.quadratic_seed_offset = int(quadratic_seed_offset)
        if not math.isfinite(self.quadratic_scale):
            raise ValueError("quadratic_scale must be finite")
        if self.quadratic_scale != 0.0 and self.latent_init_std <= 0.0:
            raise ValueError("latent_init_std must be positive for a quadratic FHT chart")
        if self.quadratic_seed_offset <= 0:
            raise ValueError("quadratic_seed_offset must be positive")
        self.residual_base_scale = float(residual_base_scale)
        self.output_gain = nn.Parameter(torch.ones(self.out_features)) if output_gain else None
        self.input_gain = nn.Parameter(torch.ones(self.in_features)) if input_gain else None
        self.spectral_rank = int(spectral_rank)
        self.spectral_out_groups = int(spectral_out_groups)
        self.spectral_in_groups = int(spectral_in_groups)
        if self.spectral_rank:
            if self.spectral_rank > min(self.in_features, self.out_features):
                raise ValueError("spectral_rank exceeds matrix dimensions")
            if self.out_features % self.spectral_out_groups or self.in_features % self.spectral_in_groups:
                raise ValueError("spectral gain groups must divide feature dimensions")
            self.register_buffer("spectral_u", self._hadamard_columns(self.out_features, self.spectral_rank, seed + 17), persistent=True)
            self.register_buffer("spectral_v", self._hadamard_columns(self.in_features, self.spectral_rank, seed + 31), persistent=True)
            self.spectral_core = nn.Parameter(torch.zeros(self.spectral_rank, self.spectral_rank))
            self.spectral_log_out_gain = nn.Parameter(torch.zeros(self.spectral_out_groups))
            self.spectral_log_in_gain = nn.Parameter(torch.zeros(self.spectral_in_groups))
        else:
            self.spectral_u = self.spectral_v = None
            self.spectral_core = self.spectral_log_out_gain = self.spectral_log_in_gain = None
        if self.residual_base_scale != 0.0:
            base_weight = torch.empty(self.out_features, self.in_features)
            nn.init.normal_(base_weight, mean=0.0, std=float(residual_base_std))
            self.register_buffer("residual_base_weight", base_weight, persistent=True)
        else:
            self.residual_base_weight = None

    @staticmethod
    def _hadamard_columns(length: int, rank: int, seed: int) -> torch.Tensor:
        rows = torch.arange(length, dtype=torch.int64).view(-1, 1)
        cols = (torch.arange(rank, dtype=torch.int64) + int(seed)).view(1, -1)
        bits = rows.bitwise_and(cols)
        parity = torch.zeros_like(bits)
        while bits.any():
            parity = parity.bitwise_xor(bits.bitwise_and(1))
            bits = bits.bitwise_right_shift(1)
        return (1.0 - 2.0 * parity.float()) / math.sqrt(length)

    def _modulation_offset(self) -> torch.Tensor:
        offset = self.generator.latent.square().sum()
        if self.modulation_centered:
            expected = self.generator.latent.numel() * self.latent_init_std * self.latent_init_std
            offset = offset - offset.new_tensor(expected)
        return offset

    def _quadratic_chart(self) -> torch.Tensor:
        """Generate a centered, variance-normalized nonlinear FHT chart.

        The ordinary BlockFHT generator is a single fixed affine subspace
        ``A z``.  This chart adds a fixed quadratic feature map
        ``B z * C z`` using independent seeded FHT maps.  It therefore changes
        the tangent with ``z`` without introducing a learned basis or any
        additional trainable parameter.
        """
        linear = self.generator()
        if self.quadratic_scale == 0.0:
            return linear
        first = block_fht_slice(
            self.generator.latent,
            self.generator.size,
            self.generator.layers,
            self.generator.seed + self.quadratic_seed_offset,
            0,
            self.generator.size,
        )
        second = block_fht_slice(
            self.generator.latent,
            self.generator.size,
            self.generator.layers,
            self.generator.seed + 2 * self.quadratic_seed_offset,
            0,
            self.generator.size,
        )
        product = first * second
        product = product - product.mean()
        scale = self.quadratic_scale
        variance_ratio = self.generator.latent_size / self.generator.block_size
        normalization = 1.0 / math.sqrt(
            1.0 + scale * scale * variance_ratio
        )
        return normalization * (
            linear + scale * product / self.latent_init_std
        )

    @property
    def weight(self) -> torch.Tensor:
        weight = self._quadratic_chart() * self.weight_scale
        if self.modulation_alpha != 0.0:
            weight = weight + self.modulation_alpha * self._modulation_offset()
        weight = weight.view(self.out_features, self.in_features)
        if self.residual_base_weight is not None:
            weight = self.residual_base_weight.to(device=weight.device, dtype=weight.dtype) + self.residual_base_scale * weight
        if self.spectral_core is not None:
            correction = self.spectral_u.to(dtype=weight.dtype).matmul(self.spectral_core.to(dtype=weight.dtype)).matmul(self.spectral_v.to(dtype=weight.dtype).transpose(0, 1))
            weight = weight + correction
            out_gain = self.spectral_log_out_gain.exp().repeat_interleave(self.out_features // self.spectral_out_groups).to(dtype=weight.dtype)
            in_gain = self.spectral_log_in_gain.exp().repeat_interleave(self.in_features // self.spectral_in_groups).to(dtype=weight.dtype)
            weight = weight * out_gain.view(-1, 1) * in_gain.view(1, -1)
        if self.output_gain is not None:
            weight = weight * self.output_gain.to(device=weight.device, dtype=weight.dtype).view(-1, 1)
        if self.input_gain is not None:
            weight = weight * self.input_gain.to(device=weight.device, dtype=weight.dtype).view(1, -1)
        return weight

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self._cached_weight is not None:
            return F.linear(input, self._cached_weight, self.bias)
        if self.residual_base_weight is not None or self.modulation_centered or self.quadratic_scale != 0.0 or self.output_gain is not None or self.input_gain is not None or self.spectral_core is not None:
            return F.linear(input, self.weight.to(dtype=input.dtype), self.bias)
        return _BlockFHTLinearFn.apply(
            input,
            self.generator.latent,
            self.bias,
            self.in_features,
            self.out_features,
            self.generator.size,
            self.generator.layers,
            self.generator.seed,
            self.weight_scale,
            self.modulation_alpha,
        )

    def forward_fused(self, input: torch.Tensor, weight_scale: float = 1.0) -> torch.Tensor:
        if self.quadratic_scale != 0.0 or self.output_gain is not None or self.input_gain is not None or self.spectral_core is not None:
            return F.linear(input, self.weight.to(dtype=input.dtype), self.bias)
        out = block_fht_linear_forward(
            input,
            self.generator.latent,
            self.out_features,
            self.generator.layers,
            self.generator.seed,
            weight_scale=weight_scale * self.weight_scale,
        )
        if self.modulation_alpha != 0.0:
            modulation = self.modulation_alpha * self._modulation_offset()
            out = out + modulation.to(dtype=out.dtype) * input.sum(dim=-1, keepdim=True)
        if self.residual_base_weight is not None:
            base_out = F.linear(input, self.residual_base_weight.to(device=input.device, dtype=input.dtype))
            out = base_out + self.residual_base_scale * out
        if self.bias is not None:
            out = out + self.bias
        return out

    def materialize_weight_cache(self, dtype: torch.dtype | None = None) -> None:
        if self.residual_base_weight is not None or self.spectral_core is not None:
            return
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
            grad_weight = self._cached_weight.grad.to(dtype=self.generator.latent.dtype)
            cached_weight = self._cached_weight.detach().to(dtype=self.generator.latent.dtype)
            generated_grad_weight = grad_weight
            if self.output_gain is not None:
                output_gain = self.output_gain.detach().to(dtype=self.generator.latent.dtype)
                if torch.any(output_gain == 0):
                    raise RuntimeError("cannot flush a cached BlockFHT weight with a zero output gain")
                output_gain_grad = (grad_weight * (cached_weight / output_gain.view(-1, 1))).sum(dim=1)
                if self.output_gain.grad is None:
                    self.output_gain.grad = output_gain_grad.to(dtype=self.output_gain.dtype)
                else:
                    self.output_gain.grad.add_(output_gain_grad.to(dtype=self.output_gain.dtype))
                generated_grad_weight = generated_grad_weight * output_gain.view(-1, 1)
            if self.input_gain is not None:
                input_gain = self.input_gain.detach().to(dtype=self.generator.latent.dtype)
                if torch.any(input_gain == 0):
                    raise RuntimeError("cannot flush a cached BlockFHT weight with a zero input gain")
                input_gain_grad = (grad_weight * (cached_weight / input_gain.view(1, -1))).sum(dim=0)
                if self.input_gain.grad is None:
                    self.input_gain.grad = input_gain_grad.to(dtype=self.input_gain.dtype)
                else:
                    self.input_gain.grad.add_(input_gain_grad.to(dtype=self.input_gain.dtype))
                generated_grad_weight = generated_grad_weight * input_gain.view(1, -1)
            generated_grad_weight = generated_grad_weight.reshape(-1)
            if self.residual_base_weight is not None:
                generated_grad_weight = generated_grad_weight * self.residual_base_scale
            if self.quadratic_scale == 0.0:
                grad_latent = self.weight_scale * block_fht_grad_latent(
                    self.generator.latent,
                    generated_grad_weight,
                    self.generator.size,
                    self.generator.layers,
                    self.generator.seed,
                )
            else:
                with torch.no_grad():
                    first = block_fht_slice(
                        self.generator.latent,
                        self.generator.size,
                        self.generator.layers,
                        self.generator.seed + self.quadratic_seed_offset,
                        0,
                        self.generator.size,
                    )
                    second = block_fht_slice(
                        self.generator.latent,
                        self.generator.size,
                        self.generator.layers,
                        self.generator.seed + 2 * self.quadratic_seed_offset,
                        0,
                        self.generator.size,
                    )
                product_grad = generated_grad_weight - generated_grad_weight.mean()
                scale = self.quadratic_scale
                variance_ratio = (
                    self.generator.latent_size / self.generator.block_size
                )
                normalization = 1.0 / math.sqrt(
                    1.0 + scale * scale * variance_ratio
                )
                grad_latent = block_fht_grad_latent(
                    self.generator.latent,
                    generated_grad_weight,
                    self.generator.size,
                    self.generator.layers,
                    self.generator.seed,
                )
                grad_latent = grad_latent + scale / self.latent_init_std * (
                    block_fht_grad_latent(
                        self.generator.latent,
                        product_grad * second,
                        self.generator.size,
                        self.generator.layers,
                        self.generator.seed + self.quadratic_seed_offset,
                    )
                    + block_fht_grad_latent(
                        self.generator.latent,
                        product_grad * first,
                        self.generator.size,
                        self.generator.layers,
                        self.generator.seed + 2 * self.quadratic_seed_offset,
                    )
                )
                grad_latent = self.weight_scale * normalization * grad_latent
            if self.modulation_alpha != 0.0:
                grad_latent = grad_latent + 2.0 * self.modulation_alpha * self.generator.latent * generated_grad_weight.sum()
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


def suspend_block_fht_weight_cache(model: nn.Module) -> list[tuple[BlockFHTLinear, torch.Tensor]]:
    """Temporarily bypass materialized weights for a live-latent forward.

    Mapping-stability uses a perturbed latent point.  Its forward must not see
    the unperturbed cached weights, but CE microbatches in the same optimizer
    step still benefit from that cache.  The returned handles retain the exact
    leaf tensors (and their accumulated CE gradients) for later restoration
    and normal cache-to-latent gradient projection.
    """
    suspended: list[tuple[BlockFHTLinear, torch.Tensor]] = []
    for module in model.modules():
        if isinstance(module, BlockFHTLinear) and module._cached_weight is not None:
            suspended.append((module, module._cached_weight))
            module._cached_weight = None
    return suspended


def restore_block_fht_weight_cache(suspended: list[tuple[BlockFHTLinear, torch.Tensor]]) -> None:
    """Restore a cache suspended by :func:`suspend_block_fht_weight_cache`."""
    for module, weight in suspended:
        if module._cached_weight is not None:
            raise RuntimeError("cannot restore BlockFHT cache over a live cached weight")
        module._cached_weight = weight


class BlockFHT(nn.Module):
    def __init__(
        self,
        latent: int | tuple[int, int] | torch.Tensor | nn.Parameter,
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
        elif isinstance(latent, tuple):
            if len(latent) != 2 or min(int(dim) for dim in latent) <= 0:
                raise ValueError("latent shape must contain two positive dimensions")
            latent_shape = (int(latent[0]), int(latent[1]))
            self.latent_size = math.prod(latent_shape)
            self.latent = nn.Parameter(torch.empty(latent_shape))
            nn.init.normal_(self.latent, std=float(latent_init_std))
        else:
            if latent.dim() not in {1, 2}:
                raise ValueError("latent tensor must be 1D or 2D")
            self.latent_size = int(latent.numel())
            self.latent = latent if isinstance(latent, nn.Parameter) else nn.Parameter(latent)
        self.size = int(size)
        self.layers = int(layers)
        self.seed = int(seed)
        self.block_size = next_power_of_two(self.latent_size)

    def extra_repr(self) -> str:
        return (
            f"latent_size={self.latent_size}, latent_shape={tuple(self.latent.shape)}, "
            f"size={self.size}, layers={self.layers}, "
            f"seed={self.seed}, block_size={self.block_size}"
        )

    def slice(self, start: int, stop: int) -> torch.Tensor:
        return block_fht_slice(self.latent, self.size, self.layers, self.seed, start, stop)

    def forward(self) -> torch.Tensor:
        return self.slice(0, self.size)

    @property
    def cuda_extension_error(self) -> Exception | None:
        return _BLOCK_FHT_EXT_ERROR
