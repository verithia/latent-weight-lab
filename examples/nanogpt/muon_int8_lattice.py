"""Muon training over a persistent blockwise int8 weight lattice.

The learned state is a signed int8 displacement from a reproducible frozen
initial weight plus one FP16 scale per block.  A dense FP32 weight is
materialized transiently for the forward/backward pass; it is deliberately
excluded from ``state_dict``.  The optimizer computes the ordinary dense Muon
request and projects the requested next weight onto a monotone running-scale
integer lattice.  An optional FP16 optimizer-side compression residual carries
sub-quantum requests forward without changing the model or inference codec.

This is a weight-state codec, not a low-dimensional Mapping-Network latent.
The dense gradient and Muon momentum remain ambient unless a separate codec
is explicitly enabled and validated.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as F

from examples.nanogpt.muon import muon_update


class MuonInt8LatticeLinear(nn.Module):
    """Linear layer with a reproducible base and persistent int8 displacement."""

    qmax = 127

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        block_size: int = 4096,
        base_seed: int = 271828,
        weight_std: float = 0.02,
        layer_id: int = -1,
        error_feedback: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.block_size = int(block_size)
        self.layer_id = int(layer_id)
        self.error_feedback = bool(error_feedback)
        if self.in_features <= 0 or self.out_features <= 0:
            raise ValueError("int8 lattice dimensions must be positive")
        if self.block_size <= 0:
            raise ValueError("int8 lattice block size must be positive")
        if not math.isfinite(weight_std) or weight_std <= 0.0:
            raise ValueError("int8 lattice weight_std must be positive and finite")

        self.register_buffer(
            "base_seed",
            torch.tensor(int(base_seed), dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "base_weight_std",
            torch.tensor(float(weight_std), dtype=torch.float64),
            persistent=True,
        )
        element_count = self.in_features * self.out_features
        block_count = (element_count + self.block_size - 1) // self.block_size
        self.register_buffer(
            "codes",
            torch.zeros(element_count, dtype=torch.int8),
            persistent=True,
        )
        self.register_buffer(
            "scales",
            torch.zeros(block_count, dtype=torch.float16),
            persistent=True,
        )
        self.register_buffer(
            "optimizer_step",
            torch.zeros((), dtype=torch.int64),
            persistent=True,
        )

        base = self._make_base(device=torch.device("cpu"))
        self.register_buffer("base_weight", base, persistent=False)
        self.register_buffer("weight", base.clone(), persistent=False)
        self.weight.requires_grad_(True)
        self.bias = (
            nn.Parameter(torch.zeros(self.out_features)) if bias else None
        )

    @property
    def element_count(self) -> int:
        return self.in_features * self.out_features

    @property
    def persistent_codec_bytes(self) -> int:
        return (
            self.codes.numel() * self.codes.element_size()
            + self.scales.numel() * self.scales.element_size()
        )

    @property
    def fp32_weight_bytes(self) -> int:
        return self.element_count * torch.tensor([], dtype=torch.float32).element_size()

    @property
    def codec_storage_ratio(self) -> float:
        return self.persistent_codec_bytes / self.fp32_weight_bytes

    def storage_accounting(self) -> dict[str, int | float | str]:
        return {
            "elements": self.element_count,
            "block_size": self.block_size,
            "blocks": self.scales.numel(),
            "persistent_codec_bytes": self.persistent_codec_bytes,
            "fp32_weight_bytes": self.fp32_weight_bytes,
            "codec_storage_ratio": self.codec_storage_ratio,
            "codec_reduction": 1.0 / self.codec_storage_ratio,
            "transient_materialized_weight_bytes": self.weight.numel()
            * self.weight.element_size(),
            "transient_reproducible_base_bytes": self.base_weight.numel()
            * self.base_weight.element_size(),
            "optimizer_momentum": "dense_fp32_not_in_codec_count",
            "optimizer_error_feedback": (
                "dense_fp16_not_in_codec_count"
                if self.error_feedback
                else "disabled"
            ),
        }

    def _make_base(self, *, device: torch.device) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.base_seed.item()))
        base = torch.randn(
            self.out_features,
            self.in_features,
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        )
        base.mul_(float(self.base_weight_std.item()))
        return base.to(device=device)

    def _apply(self, fn, recurse: bool = True):
        """Keep the optimizer-owned transient weight a leaf after device moves.

        ``nn.Module.to`` applies a differentiable copy operation to buffers
        that require gradients.  Optimizers reject the resulting non-leaf
        tensor, so detach only the derived materialization after the ordinary
        buffer migration.  No optimizer may exist when a model is moved in
        the registered train/resume lifecycle.
        """

        result = super()._apply(fn, recurse=recurse)
        self._buffers["weight"] = self.weight.detach().requires_grad_(True)
        return result

    @torch.no_grad()
    def rematerialize_weight_(self) -> None:
        flat_scale = self.scales.float().repeat_interleave(self.block_size)
        flat_scale = flat_scale[: self.element_count]
        decoded = self.codes.float() * flat_scale
        self.weight.copy_(self.base_weight + decoded.view_as(self.weight))

    @torch.no_grad()
    def project_weight_(self, requested_weight: torch.Tensor) -> None:
        if tuple(requested_weight.shape) != tuple(self.weight.shape):
            raise ValueError("requested int8 lattice weight has the wrong shape")
        displacement = (
            requested_weight.float() - self.base_weight.float()
        ).reshape(-1)
        pad = self.scales.numel() * self.block_size - self.element_count
        padded = F.pad(displacement, (0, pad)) if pad else displacement
        block_absmax = padded.view(-1, self.block_size).abs().amax(dim=1)
        required_scales = (block_absmax / float(self.qmax)).to(torch.float16)
        self.scales.copy_(torch.maximum(self.scales, required_scales))
        flat_scale = self.scales.float().repeat_interleave(self.block_size)
        flat_scale = flat_scale[: self.element_count]
        safe = flat_scale.clamp_min(torch.finfo(torch.float32).tiny)
        encoded = torch.round(displacement / safe).clamp(
            -self.qmax, self.qmax
        ).to(torch.int8)
        encoded = torch.where(
            flat_scale > 0,
            encoded,
            torch.zeros_like(encoded),
        )
        self.codes.copy_(encoded)
        self.rematerialize_weight_()
        self.optimizer_step.add_(1)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        with torch.no_grad():
            regenerated = self._make_base(device=self.base_weight.device)
            self.base_weight.copy_(regenerated)
            self.rematerialize_weight_()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, self.weight, self.bias)


class MuonInt8Lattice(torch.optim.Optimizer):
    """Ordinary Muon request followed by causal int8-lattice projection."""

    def __init__(
        self,
        modules: list[MuonInt8LatticeLinear],
        *,
        lr: float,
        momentum: float,
        weight_decay: float,
        ns_steps: int,
    ) -> None:
        if not modules:
            raise ValueError("MuonInt8Lattice requires at least one module")
        for module in modules:
            module.weight.requires_grad_(True)
        self.modules_by_id = {id(module.weight): module for module in modules}
        defaults = {
            "lr": float(lr),
            "momentum": float(momentum),
            "weight_decay": float(weight_decay),
            "ns_steps": int(ns_steps),
        }
        super().__init__(
            [{"params": [module.weight for module in modules]}], defaults
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = float(group["lr"])
            momentum = float(group["momentum"])
            weight_decay = float(group["weight_decay"])
            ns_steps = int(group["ns_steps"])
            for weight in group["params"]:
                gradient = weight.grad
                if gradient is None:
                    continue
                state: dict[str, Any] = self.state[weight]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(weight)
                buffer = state["momentum_buffer"]
                buffer.mul_(momentum).add_(gradient)
                update = muon_update(
                    gradient.add(buffer, alpha=momentum), steps=ns_steps
                )
                requested = weight.float()
                if weight_decay != 0.0:
                    requested = requested * (1.0 - lr * weight_decay)
                requested = requested.add(update.float(), alpha=-lr)
                module = self.modules_by_id[id(weight)]
                if module.error_feedback:
                    if "compression_residual" not in state:
                        state["compression_residual"] = torch.zeros_like(
                            weight, dtype=torch.float16
                        )
                    residual = state["compression_residual"]
                    projection_target = requested.add(residual.float())
                    module.project_weight_(projection_target)
                    residual.copy_(
                        (projection_target - module.weight.float()).to(
                            dtype=residual.dtype
                        )
                    )
                else:
                    module.project_weight_(requested)
        return loss
