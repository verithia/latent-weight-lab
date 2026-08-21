"""Compact projected-Muon training over layer-private pair codebooks.

The persistent model state is one or two uint8 codes per two weights plus
small FP32 256x2 codebooks.  A dense FP32 weight and gradient are materialized
only for the current forward/backward.  The optimizer carries momentum only
as one two-value vector per codeword and immediately projects each Muon
request back into the compact code state.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as F

from examples.nanogpt.muon import muon_update


def _normal_cartesian_codebook(std: float, *, device: torch.device) -> torch.Tensor:
    probabilities = (torch.arange(16, dtype=torch.float32) + 0.5) / 16.0
    levels = math.sqrt(2.0) * torch.erfinv(2.0 * probabilities - 1.0) * float(std)
    first, second = torch.meshgrid(levels, levels, indexing="ij")
    return torch.stack((first.reshape(-1), second.reshape(-1)), dim=1).to(device)


@torch.no_grad()
def _nearest_codes_exact(vectors: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    parts = []
    for start in range(0, vectors.shape[0], 32768):
        stop = min(start + 32768, vectors.shape[0])
        values = vectors[start:stop]
        distances = (
            values.square().sum(dim=1, keepdim=True)
            + codebook.square().sum(dim=1)[None, :]
            - 2.0 * values @ codebook.T
        )
        parts.append(distances.argmin(dim=1).to(torch.uint8))
    return torch.cat(parts)


@torch.no_grad()
def _nearest_cartesian_codes(
    vectors: torch.Tensor, codebook: torch.Tensor
) -> torch.Tensor:
    """Exact nearest codes for the 16x16 Cartesian initialization grid."""
    first_levels = codebook[::16, 0]
    second_levels = codebook[:16, 1]
    first_midpoints = (first_levels[:-1] + first_levels[1:]) * 0.5
    second_midpoints = (second_levels[:-1] + second_levels[1:]) * 0.5
    first = torch.bucketize(vectors[:, 0].contiguous(), first_midpoints)
    second = torch.bucketize(vectors[:, 1].contiguous(), second_midpoints)
    return (first * 16 + second).to(torch.uint8)


class MuonPairVQLinear(nn.Module):
    """Linear layer whose only persistent matrix state is pair VQ."""

    vector_length = 2
    codebook_size = 256

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        stages: int,
        base_seed: int,
        weight_std: float,
        layer_id: int,
        fast_residual: bool = False,
        neighbor_candidates: int = 16,
        code_refresh_interval: int = 8,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.stages = int(stages)
        self.layer_id = int(layer_id)
        self.fast_residual = bool(fast_residual)
        self.neighbor_candidates = int(neighbor_candidates)
        self.code_refresh_interval = int(code_refresh_interval)
        if self.in_features <= 0 or self.out_features <= 0:
            raise ValueError("pair-VQ dimensions must be positive")
        if self.in_features * self.out_features % self.vector_length:
            raise ValueError("pair-VQ element count must be divisible by two")
        if self.stages not in (1, 2):
            raise ValueError("pair-VQ stages must be one or two")
        if not 1 <= self.neighbor_candidates <= self.codebook_size:
            raise ValueError("pair-VQ neighbor count is invalid")
        if self.code_refresh_interval <= 0:
            raise ValueError("pair-VQ refresh interval must be positive")
        if not math.isfinite(weight_std) or weight_std <= 0.0:
            raise ValueError("pair-VQ weight_std must be positive and finite")

        generator = torch.Generator(device="cpu").manual_seed(int(base_seed))
        target = torch.randn(
            self.out_features,
            self.in_features,
            generator=generator,
            dtype=torch.float32,
        ).mul_(float(weight_std))
        target_pairs = target.reshape(-1, self.vector_length)
        codebooks, codes = [], []
        residual = target_pairs
        for _stage in range(self.stages):
            stage_std = max(float(residual.std()), torch.finfo(torch.float32).tiny)
            codebook = _normal_cartesian_codebook(stage_std, device=torch.device("cpu"))
            stage_codes = _nearest_cartesian_codes(residual, codebook)
            decoded = codebook.index_select(0, stage_codes.long())
            codebooks.append(codebook)
            codes.append(stage_codes)
            residual = residual - decoded
        self.register_buffer("codebooks", torch.stack(codebooks), persistent=True)
        self.register_buffer("codes", torch.stack(codes), persistent=True)
        pair_count = target_pairs.shape[0]
        self.register_buffer(
            "fast_levels",
            (
                torch.zeros(2, 16, dtype=torch.float32)
                if self.fast_residual
                else torch.empty(0, dtype=torch.float32)
            ),
            persistent=True,
        )
        self.register_buffer(
            "fast_codes",
            (
                torch.zeros(pair_count, dtype=torch.uint8)
                if self.fast_residual
                else torch.empty(0, dtype=torch.uint8)
            ),
            persistent=True,
        )
        self.register_buffer(
            "optimizer_step", torch.zeros((), dtype=torch.int64), persistent=True
        )
        decoded_weight = self.decode_weight().detach()
        self.register_buffer("weight", decoded_weight, persistent=False)
        self.weight.requires_grad_(True)
        self.bias = nn.Parameter(torch.zeros(self.out_features)) if bias else None
        self._last_projection_diagnostics: dict[str, float | int] | None = None

    @property
    def element_count(self) -> int:
        return self.in_features * self.out_features

    @property
    def persistent_codec_bytes(self) -> int:
        return (
            self.codebooks.numel() * self.codebooks.element_size()
            + self.codes.numel() * self.codes.element_size()
            + self.fast_levels.numel() * self.fast_levels.element_size()
            + self.fast_codes.numel() * self.fast_codes.element_size()
            + self.optimizer_step.numel() * self.optimizer_step.element_size()
        )

    @property
    def compact_momentum_bytes(self) -> int:
        return self.codebooks.numel() * torch.tensor([], dtype=torch.float32).element_size()

    @property
    def transient_weight_bytes(self) -> int:
        return self.weight.numel() * self.weight.element_size()

    def storage_accounting(self) -> dict[str, int | float | str]:
        dense_bf16 = self.element_count * 2
        dense_fp32_weight_and_momentum = self.element_count * 8
        persistent_training = self.persistent_codec_bytes + self.compact_momentum_bytes
        return {
            "elements": self.element_count,
            "stages": self.stages,
            "fast_residual": self.fast_residual,
            "persistent_codec_bytes": self.persistent_codec_bytes,
            "compact_momentum_bytes": self.compact_momentum_bytes,
            "persistent_training_bytes": persistent_training,
            "model_compression_vs_dense_bf16": dense_bf16 / self.persistent_codec_bytes,
            "training_compression_vs_dense_fp32_weight_plus_momentum": (
                dense_fp32_weight_and_momentum / persistent_training
            ),
            "transient_materialized_weight_bytes": self.transient_weight_bytes,
            "transient_gradient_bytes": self.transient_weight_bytes,
            "dense_master_weight": "disabled",
            "dense_optimizer_momentum": "disabled",
            "ambient_error_buffer": "disabled",
        }

    def _apply(self, fn, recurse: bool = True):
        result = super()._apply(fn, recurse=recurse)
        self._buffers["weight"] = self.weight.detach().requires_grad_(True)
        return result

    def decode_pairs(self, stage: int) -> torch.Tensor:
        return self.codebooks[int(stage)].index_select(
            0, self.codes[int(stage)].long()
        )

    def decode_weight(self) -> torch.Tensor:
        pairs = sum(self.decode_pairs(stage) for stage in range(self.stages))
        if self.fast_residual:
            fast_codes = self.fast_codes.long()
            pairs = pairs + torch.stack(
                (
                    self.fast_levels[0].index_select(0, fast_codes // 16),
                    self.fast_levels[1].index_select(0, fast_codes % 16),
                ),
                dim=1,
            )
        return pairs.reshape(self.out_features, self.in_features).float()

    def decode_slow_pairs(self) -> torch.Tensor:
        return sum(self.decode_pairs(stage) for stage in range(self.stages))

    def decode_fast_pairs(self) -> torch.Tensor:
        if not self.fast_residual:
            return torch.zeros_like(self.decode_pairs(0))
        fast_codes = self.fast_codes.long()
        return torch.stack(
            (
                self.fast_levels[0].index_select(0, fast_codes // 16),
                self.fast_levels[1].index_select(0, fast_codes % 16),
            ),
            dim=1,
        )

    @torch.no_grad()
    def rematerialize_weight_(self) -> None:
        self.weight.copy_(self.decode_weight())

    @torch.no_grad()
    def _local_reassign_(
        self,
        *,
        stage: int,
        requested_pairs: torch.Tensor,
    ) -> int:
        codebook = self.codebooks[stage]
        pairwise = torch.cdist(codebook, codebook).square()
        neighbors = pairwise.topk(
            self.neighbor_candidates, largest=False, dim=1
        ).indices
        old_codes = self.codes[stage].long()
        new_parts = []
        for start in range(0, requested_pairs.shape[0], 32768):
            stop = min(start + 32768, requested_pairs.shape[0])
            candidate_ids = neighbors.index_select(0, old_codes[start:stop])
            candidates = codebook[candidate_ids]
            distances = (
                requested_pairs[start:stop, None, :] - candidates
            ).square().sum(dim=2)
            choice = distances.argmin(dim=1)
            selected = candidate_ids.gather(1, choice[:, None]).squeeze(1)
            new_parts.append(selected.to(torch.uint8))
        new_codes = torch.cat(new_parts)
        changes = int((new_codes != self.codes[stage]).sum())
        self.codes[stage].copy_(new_codes)
        return changes

    @torch.no_grad()
    def _centroid_projection_(
        self,
        *,
        stage: int,
        requested_pairs: torch.Tensor,
    ) -> None:
        codes = self.codes[stage].long()
        accum = torch.zeros_like(self.codebooks[stage])
        accum.index_add_(0, codes, requested_pairs)
        counts = torch.bincount(codes, minlength=self.codebook_size)
        live = counts > 0
        self.codebooks[stage, live] = accum[live] / counts[live, None]

    @torch.no_grad()
    def _fit_fast_residual_(self, residual: torch.Tensor) -> int:
        if not self.fast_residual:
            return 0
        old_codes = self.fast_codes.clone()
        assignments = []
        for coordinate in range(2):
            values = residual[:, coordinate]
            mean = values.mean()
            std = values.std(unbiased=False)
            old_levels = self.fast_levels[coordinate]
            old_std = old_levels.std(unbiased=False)
            if float(std) <= torch.finfo(torch.float32).tiny:
                levels = torch.full_like(old_levels, mean)
            elif float(old_std) > torch.finfo(torch.float32).tiny:
                levels = (old_levels - old_levels.mean()) / old_std
                levels = (levels * std + mean).sort().values
            else:
                probabilities = (
                    torch.arange(16, device=values.device, dtype=torch.float32)
                    + 0.5
                ) / 16.0
                levels = (
                    math.sqrt(2.0)
                    * torch.erfinv(2.0 * probabilities - 1.0)
                    * std
                    + mean
                )
            for _iteration in range(2):
                midpoints = (levels[:-1] + levels[1:]) * 0.5
                indices = torch.bucketize(values.contiguous(), midpoints)
                sums = torch.zeros_like(levels)
                sums.index_add_(0, indices, values)
                counts = torch.bincount(indices, minlength=16)
                live = counts > 0
                levels[live] = sums[live] / counts[live]
                levels = levels.sort().values
            indices = torch.bucketize(
                values.contiguous(), (levels[:-1] + levels[1:]) * 0.5
            )
            self.fast_levels[coordinate].copy_(levels)
            assignments.append(indices)
        new_codes = (assignments[0] * 16 + assignments[1]).to(torch.uint8)
        self.fast_codes.copy_(new_codes)
        return int((new_codes != old_codes).sum())

    @torch.no_grad()
    def project_requested_weight_(
        self,
        requested_weight: torch.Tensor,
        *,
        refresh_codes: bool,
    ) -> dict[str, float | int]:
        if tuple(requested_weight.shape) != tuple(self.weight.shape):
            raise ValueError("requested pair-VQ weight has the wrong shape")
        old = self.weight.detach().float().clone()
        target_pairs = requested_weight.detach().float().reshape(-1, 2)
        code_changes = 0
        fast_pairs = self.decode_fast_pairs()
        for stage in range(self.stages):
            other = sum(
                self.decode_pairs(other_stage)
                for other_stage in range(self.stages)
                if other_stage != stage
            )
            other = other + fast_pairs
            residual_target = target_pairs - other
            if refresh_codes:
                code_changes += self._local_reassign_(
                    stage=stage, requested_pairs=residual_target
                )
            self._centroid_projection_(
                stage=stage, requested_pairs=residual_target
            )
        fast_code_changes = 0
        if self.fast_residual:
            fast_code_changes = self._fit_fast_residual_(
                target_pairs - self.decode_slow_pairs()
            )
            code_changes += fast_code_changes
        self.rematerialize_weight_()
        requested_delta = requested_weight.float() - old
        achieved_delta = self.weight.float() - old
        request_energy = float(requested_delta.square().sum())
        residual_energy = float((self.weight.float() - requested_weight.float()).square().sum())
        achieved_energy = float(achieved_delta.square().sum())
        inner = float((requested_delta * achieved_delta).sum())
        cosine = inner / max(
            math.sqrt(max(request_energy, 0.0) * max(achieved_energy, 0.0)),
            1e-30,
        )
        diagnostics: dict[str, float | int] = {
            "layer": self.layer_id,
            "stages": self.stages,
            "fast_residual": int(self.fast_residual),
            "in_features": self.in_features,
            "out_features": self.out_features,
            "optimizer_step": int(self.optimizer_step),
            "request_energy": request_energy,
            "projection_residual_energy": residual_energy,
            "requested_step_energy_recovery": 1.0
            - residual_energy / max(request_energy, 1e-30),
            "requested_update_cosine": cosine,
            "code_changes": code_changes,
            "fast_code_changes": fast_code_changes,
            "refresh_codes": int(refresh_codes),
        }
        self._last_projection_diagnostics = diagnostics
        self.optimizer_step.add_(1)
        return diagnostics

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
            self.rematerialize_weight_()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, self.weight, self.bias)


class MuonPairVQ(torch.optim.Optimizer):
    """Muon request with compact code-conditioned momentum and projection."""

    def __init__(
        self,
        modules: list[MuonPairVQLinear],
        *,
        lr: float,
        momentum: float,
        weight_decay: float,
        ns_steps: int,
    ) -> None:
        if not modules:
            raise ValueError("MuonPairVQ requires at least one module")
        for module in modules:
            module.weight.requires_grad_(True)
        self.modules_by_id = {id(module.weight): module for module in modules}
        self._diagnostics: list[dict[str, float | int]] = []
        defaults = {
            "lr": float(lr),
            "momentum": float(momentum),
            "weight_decay": float(weight_decay),
            "ns_steps": int(ns_steps),
        }
        super().__init__([{"params": [module.weight for module in modules]}], defaults)

    def load_state_dict(self, state_dict):
        result = super().load_state_dict(state_dict)
        for weight, state in self.state.items():
            momentum = state.get("compact_momentum")
            if momentum is not None:
                module = self.modules_by_id[id(weight)]
                state["compact_momentum"] = momentum.to(
                    device=weight.device,
                    dtype=module.codebooks.dtype,
                )
        return result

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._diagnostics = []
        for group in self.param_groups:
            lr = float(group["lr"])
            momentum_coefficient = float(group["momentum"])
            weight_decay = float(group["weight_decay"])
            ns_steps = int(group["ns_steps"])
            for weight in group["params"]:
                gradient = weight.grad
                if gradient is None:
                    continue
                module = self.modules_by_id[id(weight)]
                state: dict[str, Any] = self.state[weight]
                if "compact_momentum" not in state:
                    state["compact_momentum"] = torch.zeros_like(module.codebooks)
                compact_momentum = state["compact_momentum"]
                gradient_pairs = gradient.float().reshape(-1, 2)
                expanded = torch.zeros_like(gradient_pairs)
                for stage in range(module.stages):
                    codes = module.codes[stage].long()
                    accum = torch.zeros_like(module.codebooks[stage])
                    accum.index_add_(0, codes, gradient_pairs)
                    counts = torch.bincount(codes, minlength=module.codebook_size)
                    live = counts > 0
                    means = torch.zeros_like(module.codebooks[stage])
                    means[live] = accum[live] / counts[live, None]
                    compact_momentum[stage].mul_(momentum_coefficient).add_(means)
                    expanded.add_(compact_momentum[stage].index_select(0, codes))
                expanded.div_(module.stages)
                requested_gradient = gradient.float() + momentum_coefficient * expanded.reshape_as(gradient)
                update = muon_update(requested_gradient, steps=ns_steps)
                requested = weight.float()
                if weight_decay != 0.0:
                    requested = requested * (1.0 - lr * weight_decay)
                requested = requested.add(update.float(), alpha=-lr)
                refresh = (
                    int(module.optimizer_step) % module.code_refresh_interval == 0
                )
                self._diagnostics.append(
                    module.project_requested_weight_(
                        requested, refresh_codes=refresh
                    )
                )
        return loss

    def consume_diagnostics(self) -> list[dict[str, float | int]]:
        diagnostics = self._diagnostics
        self._diagnostics = []
        return diagnostics
