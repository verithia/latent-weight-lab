#!/usr/bin/env python3
"""Fit non-Hadamard compact tangents to dense MLP residual PCs.

This optimistic, noncausal oracle compares basis-free families under the
same approximately-one-percent state contract:

* three independently modulated rectangular Toeplitz operators; and
* a ten-stage learned sparse-expander chain of general 2x2 blocks;
* an open-boundary live matrix-product-operator tangent; and
* a cyclic live tensor-ring tangent.
* a learned full-rank sinusoidal row/column coordinate field.

Residual PCs are fitting/evaluation targets only.  Candidate artifacts retain
live coordinates and integer seeds, never an ambient target vector or PCA
basis.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import nn

from examples.nanogpt.analyze_mlp_highcadence_basis import file_sha256
from examples.nanogpt.analyze_mlp_residual_product_fht_basis import (
    deterministic_weighted_mixture,
)
from examples.nanogpt.analyze_mlp_residual_qtt_basis import residual_temporal_basis
from examples.nanogpt.analyze_parameter_trajectory import (
    PARAMETER_PATTERN,
    load_snapshots,
    write_csv,
)


TARGET_SEED_OFFSETS = {"mlp.c_fc": 0, "mlp.c_proj": 1}
MATRIX_ROW_MODES = (3, 4, 4, 4, 4, 4)
MATRIX_COLUMN_MODES = (3, 4, 4, 4, 4, 1)


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class CompactTangent(Protocol):
    coordinate_tensors: tuple[torch.Tensor, ...]
    trainable_scalar_count: int

    def weight(self) -> torch.Tensor: ...

    def jvp(
        self, vector: torch.Tensor, *, differentiable_anchor: bool = False
    ) -> torch.Tensor: ...

    def coordinate_metric(self) -> torch.Tensor: ...

    def clamp_coordinates(self, bound: float) -> None: ...

    def ideal_forward_scalar_ops(self) -> int: ...


def seeded_signs(length: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (
        torch.randint(0, 2, (length,), generator=generator, dtype=torch.float32)
        .mul_(2.0)
        .sub_(1.0)
    )


class DiagonalToeplitzDiagonal(nn.Module):
    """Sum of three live diagonal--Toeplitz--diagonal branches."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        branches: int,
        seed: int,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.branches = int(branches)
        self.seed = int(seed)
        kernel_length = self.in_features + self.out_features - 1
        self.left_log_gain = nn.Parameter(
            torch.zeros(self.branches, self.out_features)
        )
        self.right_log_gain = nn.Parameter(
            torch.zeros(self.branches, self.in_features)
        )
        self.kernel_delta = nn.Parameter(torch.zeros(self.branches, kernel_length))
        offsets = (
            torch.arange(self.out_features).view(-1, 1)
            - torch.arange(self.in_features).view(1, -1)
            + self.in_features
            - 1
        )
        self.register_buffer("offsets", offsets.long(), persistent=False)
        left_signs = []
        right_signs = []
        base_kernels = []
        for branch in range(self.branches):
            branch_seed = self.seed + 104729 * branch
            left_signs.append(
                seeded_signs(self.out_features, seed=branch_seed + 1)
            )
            right_signs.append(
                seeded_signs(self.in_features, seed=branch_seed + 2)
            )
            generator = torch.Generator(device="cpu").manual_seed(branch_seed + 3)
            base_kernels.append(
                torch.randn(kernel_length, generator=generator)
                / math.sqrt(self.in_features)
            )
        self.register_buffer(
            "left_signs", torch.stack(left_signs), persistent=False
        )
        self.register_buffer(
            "right_signs", torch.stack(right_signs), persistent=False
        )
        self.register_buffer(
            "base_kernels", torch.stack(base_kernels), persistent=False
        )

    @property
    def coordinate_tensors(self) -> tuple[torch.Tensor, ...]:
        return self.left_log_gain, self.right_log_gain, self.kernel_delta

    @property
    def trainable_scalar_count(self) -> int:
        return sum(tensor.numel() for tensor in self.coordinate_tensors)

    def _coordinates(
        self, *, differentiable_anchor: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coordinates = self.coordinate_tensors
        if not differentiable_anchor:
            coordinates = tuple(tensor.detach() for tensor in coordinates)
        left_log, right_log, kernel_delta = coordinates
        left = self.left_signs.to(left_log) * torch.exp(left_log.clamp(-6.0, 6.0))
        right = self.right_signs.to(right_log) * torch.exp(
            right_log.clamp(-6.0, 6.0)
        )
        kernel = self.base_kernels.to(kernel_delta) + kernel_delta
        return left, right, kernel

    def _toeplitz(self, kernels: torch.Tensor) -> torch.Tensor:
        flat = kernels[:, self.offsets.reshape(-1)]
        return flat.reshape(
            self.branches, self.out_features, self.in_features
        )

    def weight(self) -> torch.Tensor:
        left, right, kernels = self._coordinates(differentiable_anchor=True)
        toeplitz = self._toeplitz(kernels)
        return (
            left.unsqueeze(-1) * toeplitz * right.unsqueeze(-2)
        ).sum(dim=0) / math.sqrt(self.branches)

    def split_coordinates(
        self, vector: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        left_count = self.left_log_gain.numel()
        right_count = self.right_log_gain.numel()
        return (
            vector[:left_count].reshape_as(self.left_log_gain),
            vector[left_count : left_count + right_count].reshape_as(
                self.right_log_gain
            ),
            vector[left_count + right_count :].reshape_as(self.kernel_delta),
        )

    def jvp(
        self, vector: torch.Tensor, *, differentiable_anchor: bool = False
    ) -> torch.Tensor:
        left_direction, right_direction, kernel_direction = self.split_coordinates(
            vector
        )
        left, right, kernels = self._coordinates(
            differentiable_anchor=differentiable_anchor
        )
        toeplitz = self._toeplitz(kernels)
        toeplitz_direction = self._toeplitz(kernel_direction)
        left_tangent = left * left_direction
        right_tangent = right * right_direction
        tangent = (
            left_tangent.unsqueeze(-1) * toeplitz * right.unsqueeze(-2)
            + left.unsqueeze(-1) * toeplitz_direction * right.unsqueeze(-2)
            + left.unsqueeze(-1) * toeplitz * right_tangent.unsqueeze(-2)
        )
        return tangent.sum(dim=0) / math.sqrt(self.branches)

    def coordinate_metric(self) -> torch.Tensor:
        with torch.no_grad():
            left, right, kernels = self._coordinates(differentiable_anchor=False)
            toeplitz = self._toeplitz(kernels)
            scale_square = 1.0 / self.branches
            branch_weight_square = (
                left.unsqueeze(-1) * toeplitz * right.unsqueeze(-2)
            ).square() * scale_square
            left_metric = branch_weight_square.sum(dim=-1)
            right_metric = branch_weight_square.sum(dim=-2)
            kernel_metric = torch.zeros_like(kernels)
            kernel_entries = (
                left.unsqueeze(-1) * right.unsqueeze(-2)
            ).square() * scale_square
            offsets = self.offsets.reshape(-1)
            for branch in range(self.branches):
                kernel_metric[branch].scatter_add_(
                    0, offsets, kernel_entries[branch].reshape(-1)
                )
        return torch.cat(
            (
                left_metric.reshape(-1),
                right_metric.reshape(-1),
                kernel_metric.reshape(-1),
            )
        ).clamp_min(1e-12)

    def clamp_coordinates(self, bound: float) -> None:
        with torch.no_grad():
            self.left_log_gain.clamp_(-bound, bound)
            self.right_log_gain.clamp_(-bound, bound)
            self.kernel_delta.clamp_(-bound, bound)

    def ideal_forward_scalar_ops(self) -> int:
        fft_length = 1 << (self.in_features + self.out_features - 2).bit_length()
        # Per branch: one input FFT, one inverse FFT, complex pointwise product,
        # and the two real diagonal scales.  This deliberately excludes the
        # refresh-time FFT of a changed kernel.
        return self.branches * (
            2 * fft_length * int(math.log2(fft_length))
            + 3 * fft_length
            + self.in_features
            + self.out_features
        )


class LearnedSparseExpander(nn.Module):
    """General learned 2x2 blocks on procedural random perfect matchings."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        padded_features: int,
        depth: int,
        group_size: int,
        seed: int,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.padded_features = int(padded_features)
        self.depth = int(depth)
        self.group_size = int(group_size)
        self.seed = int(seed)
        if self.padded_features % 2 != 0:
            raise ValueError("padded_features must be even")
        pair_count = self.padded_features // 2
        if pair_count % self.group_size != 0:
            raise ValueError("group size must divide the pair count")
        self.unique_blocks = pair_count // self.group_size
        self.block_delta = nn.Parameter(
            torch.zeros(self.depth, self.unique_blocks, 2, 2)
        )
        self.output_log_gain = nn.Parameter(torch.zeros(self.out_features))
        permutations = []
        inverse_permutations = []
        group_indices = []
        base_blocks = []
        for stage in range(self.depth):
            generator = torch.Generator(device="cpu").manual_seed(
                self.seed + 104729 * stage
            )
            permutation = torch.randperm(
                self.padded_features, generator=generator
            )
            inverse = torch.empty_like(permutation)
            inverse[permutation] = torch.arange(self.padded_features)
            group = torch.arange(pair_count) % self.unique_blocks
            group = group[torch.randperm(pair_count, generator=generator)]
            angle = 2.0 * math.pi * torch.rand(
                self.unique_blocks, generator=generator
            )
            cosine = angle.cos()
            sine = angle.sin()
            block = torch.stack(
                (
                    torch.stack((cosine, -sine), dim=-1),
                    torch.stack((sine, cosine), dim=-1),
                ),
                dim=-2,
            )
            permutations.append(permutation)
            inverse_permutations.append(inverse)
            group_indices.append(group)
            base_blocks.append(block)
        self.register_buffer(
            "permutations", torch.stack(permutations), persistent=False
        )
        self.register_buffer(
            "inverse_permutations",
            torch.stack(inverse_permutations),
            persistent=False,
        )
        self.register_buffer(
            "group_indices", torch.stack(group_indices), persistent=False
        )
        self.register_buffer(
            "base_blocks", torch.stack(base_blocks), persistent=False
        )

    @property
    def coordinate_tensors(self) -> tuple[torch.Tensor, ...]:
        return self.block_delta, self.output_log_gain

    @property
    def trainable_scalar_count(self) -> int:
        return sum(tensor.numel() for tensor in self.coordinate_tensors)

    def _apply_stage(
        self, values: torch.Tensor, blocks: torch.Tensor, stage: int
    ) -> torch.Tensor:
        permutation = self.permutations[stage].to(values.device)
        inverse = self.inverse_permutations[stage].to(values.device)
        groups = self.group_indices[stage].to(values.device)
        paired = values.index_select(-1, permutation).reshape(
            *values.shape[:-1], self.padded_features // 2, 2
        )
        expanded_blocks = blocks.index_select(0, groups)
        mixed = torch.einsum("...pi,pji->...pj", paired, expanded_blocks)
        return mixed.reshape(*values.shape[:-1], self.padded_features).index_select(
            -1, inverse
        )

    def _blocks(self, *, differentiable_anchor: bool) -> torch.Tensor:
        delta = self.block_delta
        if not differentiable_anchor:
            delta = delta.detach()
        return self.base_blocks.to(delta) + delta

    def weight(self) -> torch.Tensor:
        matrix = torch.eye(
            self.out_features,
            self.padded_features,
            device=self.block_delta.device,
            dtype=self.block_delta.dtype,
        )
        blocks = self._blocks(differentiable_anchor=True)
        for stage in range(self.depth):
            matrix = self._apply_stage(matrix, blocks[stage], stage)
        gain = torch.exp(self.output_log_gain.clamp(-6.0, 6.0))
        return gain.view(-1, 1) * matrix[:, : self.in_features]

    def split_coordinates(
        self, vector: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        block_count = self.block_delta.numel()
        return (
            vector[:block_count].reshape_as(self.block_delta),
            vector[block_count:].reshape_as(self.output_log_gain),
        )

    def jvp(
        self, vector: torch.Tensor, *, differentiable_anchor: bool = False
    ) -> torch.Tensor:
        block_direction, output_direction = self.split_coordinates(vector)
        matrix = torch.eye(
            self.out_features,
            self.padded_features,
            device=self.block_delta.device,
            dtype=self.block_delta.dtype,
        )
        tangent = torch.zeros_like(matrix)
        blocks = self._blocks(differentiable_anchor=differentiable_anchor)
        for stage in range(self.depth):
            tangent = self._apply_stage(tangent, blocks[stage], stage) + self._apply_stage(
                matrix, block_direction[stage], stage
            )
            matrix = self._apply_stage(matrix, blocks[stage], stage)
        output_anchor = (
            self.output_log_gain
            if differentiable_anchor
            else self.output_log_gain.detach()
        )
        gain = torch.exp(output_anchor.clamp(-6.0, 6.0))
        active = ((output_anchor > -6.0) & (output_anchor < 6.0)).to(gain.dtype)
        gain_tangent = gain * active * output_direction
        return (
            gain.view(-1, 1) * tangent[:, : self.in_features]
            + gain_tangent.view(-1, 1) * matrix[:, : self.in_features]
        )

    def coordinate_metric(self) -> torch.Tensor:
        # Orthogonal procedural anchors keep every stage well scaled.  The
        # constant block preconditioner is an order-correct diagonal proxy;
        # exactness comes from CG's J^T J products, not this preconditioner.
        block_metric = torch.full_like(
            self.block_delta,
            max(self.out_features / self.padded_features, 1e-6),
        )
        with torch.no_grad():
            row_metric = self.weight().square().sum(dim=1).clamp_min(1e-12)
        return torch.cat((block_metric.reshape(-1), row_metric.reshape(-1)))

    def clamp_coordinates(self, bound: float) -> None:
        with torch.no_grad():
            self.block_delta.clamp_(-bound, bound)
            self.output_log_gain.clamp_(-bound, bound)

    def ideal_forward_scalar_ops(self) -> int:
        block_multiplications = self.depth * (self.padded_features // 2) * 4
        return block_multiplications + self.out_features


def _tensor_to_matrix(
    tensor: torch.Tensor,
    *,
    row_modes: tuple[int, ...],
    column_modes: tuple[int, ...],
) -> torch.Tensor:
    """Undo a procedural Morton pairing of row and column tensor modes."""
    if len(row_modes) != len(column_modes):
        raise ValueError("row and column mode lists must have equal length")
    physical_modes = tuple(
        row * column for row, column in zip(row_modes, column_modes, strict=True)
    )
    if tuple(tensor.shape) != physical_modes:
        raise ValueError(
            f"tensor shape {tuple(tensor.shape)} != physical modes {physical_modes}"
        )
    interleaved: list[int] = []
    for row, column in zip(row_modes, column_modes, strict=True):
        interleaved.extend((row, column))
    expanded = tensor.reshape(*interleaved)
    row_axes = tuple(range(0, 2 * len(row_modes), 2))
    column_axes = tuple(range(1, 2 * len(row_modes), 2))
    return expanded.permute(*(row_axes + column_axes)).contiguous().reshape(
        math.prod(row_modes), math.prod(column_modes)
    )


def _contract_open_cores(cores: tuple[torch.Tensor, ...]) -> torch.Tensor:
    state = cores[0]
    for core in cores[1:]:
        state = torch.einsum("a...b,bdc->a...dc", state, core)
    return state.squeeze(0).squeeze(-1)


def _contract_ring_cores(cores: tuple[torch.Tensor, ...]) -> torch.Tensor:
    state = cores[0]
    for core in cores[1:]:
        state = torch.einsum("a...b,bdc->a...dc", state, core)
    return state.diagonal(dim1=0, dim2=-1).sum(dim=-1)


class LiveTensorNetwork(nn.Module):
    """Live TT/MPO or tensor-ring cores with procedural random anchors."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        row_modes: tuple[int, ...],
        column_modes: tuple[int, ...],
        topology: str,
        bond: int,
        seed: int,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.row_modes = tuple(int(value) for value in row_modes)
        self.column_modes = tuple(int(value) for value in column_modes)
        self.physical_modes = tuple(
            row * column
            for row, column in zip(
                self.row_modes, self.column_modes, strict=True
            )
        )
        self.topology = str(topology)
        self.bond = int(bond)
        self.seed = int(seed)
        canonical_shape = (
            math.prod(self.row_modes),
            math.prod(self.column_modes),
        )
        requested_shape = (self.out_features, self.in_features)
        if requested_shape == canonical_shape:
            self.transpose_output = False
        elif requested_shape == tuple(reversed(canonical_shape)):
            self.transpose_output = True
        else:
            raise ValueError(
                f"requested shape {requested_shape} is not {canonical_shape} "
                "or its transpose"
            )
        if self.topology == "open":
            total = math.prod(self.physical_modes)
            ranks = [1]
            prefix = 1
            for mode in self.physical_modes[:-1]:
                prefix *= mode
                ranks.append(min(self.bond, prefix, total // prefix))
            ranks.append(1)
        elif self.topology == "ring":
            ranks = [self.bond] * (len(self.physical_modes) + 1)
        else:
            raise ValueError(f"unsupported tensor-network topology {topology}")
        self.ranks = tuple(ranks)
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        deltas = []
        for index, mode in enumerate(self.physical_modes):
            left_rank = self.ranks[index]
            right_rank = self.ranks[index + 1]
            shape = (left_rank, mode, right_rank)
            base = torch.randn(shape, generator=generator) / math.sqrt(left_rank)
            self.register_buffer(f"base_core_{index}", base, persistent=False)
            deltas.append(nn.Parameter(torch.zeros(shape)))
        self.core_delta = nn.ParameterList(deltas)

    @property
    def coordinate_tensors(self) -> tuple[torch.Tensor, ...]:
        return tuple(self.core_delta)

    @property
    def trainable_scalar_count(self) -> int:
        return sum(tensor.numel() for tensor in self.coordinate_tensors)

    def _cores(self, *, differentiable_anchor: bool) -> tuple[torch.Tensor, ...]:
        result = []
        for index, delta in enumerate(self.core_delta):
            if not differentiable_anchor:
                delta = delta.detach()
            result.append(getattr(self, f"base_core_{index}").to(delta) + delta)
        return tuple(result)

    def _tensor(self, cores: tuple[torch.Tensor, ...]) -> torch.Tensor:
        if self.topology == "open":
            return _contract_open_cores(cores)
        return _contract_ring_cores(cores)

    def _matrix(self, tensor: torch.Tensor) -> torch.Tensor:
        matrix = _tensor_to_matrix(
            tensor,
            row_modes=self.row_modes,
            column_modes=self.column_modes,
        )
        return matrix.T.contiguous() if self.transpose_output else matrix

    def weight(self) -> torch.Tensor:
        return self._matrix(self._tensor(self._cores(differentiable_anchor=True)))

    def split_coordinates(self, vector: torch.Tensor) -> tuple[torch.Tensor, ...]:
        result = []
        offset = 0
        for delta in self.core_delta:
            count = delta.numel()
            result.append(vector[offset : offset + count].reshape_as(delta))
            offset += count
        if offset != vector.numel():
            raise ValueError("coordinate vector has the wrong size")
        return tuple(result)

    def jvp(
        self, vector: torch.Tensor, *, differentiable_anchor: bool = False
    ) -> torch.Tensor:
        cores = self._cores(differentiable_anchor=differentiable_anchor)
        directions = self.split_coordinates(vector)
        state = cores[0]
        tangent = directions[0]
        for core, direction in zip(cores[1:], directions[1:], strict=True):
            tangent = torch.einsum(
                "a...b,bdc->a...dc", tangent, core
            ) + torch.einsum("a...b,bdc->a...dc", state, direction)
            state = torch.einsum("a...b,bdc->a...dc", state, core)
        if self.topology == "open":
            tensor = tangent.squeeze(0).squeeze(-1)
        else:
            tensor = tangent.diagonal(dim1=0, dim2=-1).sum(dim=-1)
        return self._matrix(tensor)

    def coordinate_metric(self) -> torch.Tensor:
        # The procedural anchor uses variance-preserving core scales.  A unit
        # diagonal is deliberately conservative; exactness remains in CG's
        # matrix-free J^T J action, and a synthetic own-tangent gate detects
        # any material conditioning failure.
        return torch.ones(
            self.trainable_scalar_count,
            device=self.core_delta[0].device,
            dtype=self.core_delta[0].dtype,
        )

    def clamp_coordinates(self, bound: float) -> None:
        with torch.no_grad():
            for delta in self.core_delta:
                delta.clamp_(-bound, bound)

    def ideal_forward_scalar_ops(self) -> int:
        # Scalar products needed to materialize the tensor by left-to-right
        # contraction.  A fused MPO matvec is a separate performance gate.
        prefix = self.physical_modes[0]
        operations = 0
        for index, mode in enumerate(self.physical_modes[1:], start=1):
            operations += (
                prefix
                * self.ranks[0]
                * self.ranks[index]
                * mode
                * self.ranks[index + 1]
            )
            prefix *= mode
        if self.topology == "ring":
            operations += math.prod(self.physical_modes) * self.bond
        return operations


class SinusoidalCoordinateField(nn.Module):
    """Full-rank implicit matrix with compact learned row/column codes."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int,
        seed: int,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("sinusoidal coordinate rank must be positive")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.seed = int(seed)
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        self.row_codes = nn.Parameter(
            torch.randn(
                self.out_features, self.rank, generator=generator
            )
        )
        self.column_codes = nn.Parameter(
            torch.randn(
                self.in_features, self.rank, generator=generator
            )
        )
        self.inverse_sqrt_rank = 1.0 / math.sqrt(self.rank)

    @property
    def coordinate_tensors(self) -> tuple[torch.Tensor, ...]:
        return self.row_codes, self.column_codes

    @property
    def trainable_scalar_count(self) -> int:
        return sum(tensor.numel() for tensor in self.coordinate_tensors)

    def _coordinates(
        self, *, differentiable_anchor: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if differentiable_anchor:
            return self.row_codes, self.column_codes
        return self.row_codes.detach(), self.column_codes.detach()

    def _phase(
        self, *, differentiable_anchor: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row, column = self._coordinates(
            differentiable_anchor=differentiable_anchor
        )
        phase = (row @ column.transpose(0, 1)) * self.inverse_sqrt_rank
        return row, column, phase

    def weight(self) -> torch.Tensor:
        _row, _column, phase = self._phase(differentiable_anchor=True)
        return torch.sin(phase)

    def split_coordinates(
        self, vector: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row_count = self.row_codes.numel()
        return (
            vector[:row_count].reshape_as(self.row_codes),
            vector[row_count:].reshape_as(self.column_codes),
        )

    def jvp(
        self, vector: torch.Tensor, *, differentiable_anchor: bool = False
    ) -> torch.Tensor:
        row_direction, column_direction = self.split_coordinates(vector)
        row, column, phase = self._phase(
            differentiable_anchor=differentiable_anchor
        )
        phase_direction = (
            row_direction @ column.transpose(0, 1)
            + row @ column_direction.transpose(0, 1)
        ) * self.inverse_sqrt_rank
        return torch.cos(phase) * phase_direction

    def coordinate_metric(self) -> torch.Tensor:
        with torch.no_grad():
            row, column, phase = self._phase(differentiable_anchor=False)
            cosine_square = torch.cos(phase).square()
            row_metric = (
                cosine_square @ column.square()
            ) * self.inverse_sqrt_rank**2
            column_metric = (
                cosine_square.transpose(0, 1) @ row.square()
            ) * self.inverse_sqrt_rank**2
        return torch.cat(
            (row_metric.reshape(-1), column_metric.reshape(-1))
        ).clamp_min(1e-12)

    def clamp_coordinates(self, bound: float) -> None:
        with torch.no_grad():
            self.row_codes.clamp_(-bound, bound)
            self.column_codes.clamp_(-bound, bound)

    def ideal_forward_scalar_ops(self) -> int:
        dense = self.out_features * self.in_features
        # Rank-r phase GEMM (r multiply-adds per entry), then one sine.
        return dense * (self.rank + 1)


def flatten_tensors(tensors: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


def coordinate_vjp(module: CompactTangent, target: torch.Tensor) -> torch.Tensor:
    gradients = torch.autograd.grad(
        module.weight(),
        module.coordinate_tensors,
        grad_outputs=target.to(module.coordinate_tensors[0]),
        create_graph=False,
        retain_graph=False,
    )
    return flatten_tensors(tuple(gradient.detach() for gradient in gradients))


def natural_action(
    module: CompactTangent,
    target: torch.Tensor,
    *,
    differentiable_anchor: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    direction = (
        coordinate_vjp(module, target) / module.coordinate_metric().detach()
    ).detach()
    action = module.jvp(direction, differentiable_anchor=differentiable_anchor)
    cosine = torch.sum(action.float() * target.float()) / (
        action.float().norm() * target.float().norm()
    ).clamp_min(1e-30)
    return action, cosine, cosine.square()


def cg_project(
    module: CompactTangent,
    target: torch.Tensor,
    *,
    maximum_iterations: int,
    relative_tolerance: float,
    damping_ratio: float,
) -> dict[str, Any]:
    target = target.float()
    right = coordinate_vjp(module, target)
    metric = module.coordinate_metric().detach()
    damping = damping_ratio * float(metric.mean())

    estimate = torch.zeros_like(right)
    projected = torch.zeros_like(target)
    best_projected = projected.clone()
    best_error_energy = target.double().square().sum()
    best_iteration = 0
    stale_iterations = 0
    residual = right.clone()
    preconditioned = residual / (metric + damping).clamp_min(1e-12)
    direction = preconditioned.clone()
    rz = torch.dot(residual.double(), preconditioned.double())
    initial_norm = residual.double().norm().clamp_min(1e-30)
    iterations = 0
    for iteration in range(maximum_iterations):
        ambient_action = module.jvp(direction)
        action = coordinate_vjp(module, ambient_action) + damping * direction
        denominator = torch.dot(direction.double(), action.double()).clamp_min(1e-30)
        step = rz / denominator
        estimate.add_(direction, alpha=float(step))
        projected.add_(ambient_action, alpha=float(step))
        residual.add_(action, alpha=-float(step))
        iterations = iteration + 1
        error_energy = (target - projected).double().square().sum()
        if error_energy < best_error_energy:
            best_error_energy = error_energy
            best_projected.copy_(projected)
            best_iteration = iterations
            stale_iterations = 0
        else:
            stale_iterations += 1
        if float(residual.double().norm() / initial_norm) <= relative_tolerance:
            break
        if stale_iterations >= 32:
            break
        next_preconditioned = residual / (metric + damping).clamp_min(1e-12)
        next_rz = torch.dot(residual.double(), next_preconditioned.double())
        direction.mul_(float(next_rz / rz)).add_(next_preconditioned)
        rz = next_rz
    target_energy = target.double().square().sum().clamp_min(1e-30)
    return {
        "cg_projection_capture": float(1.0 - best_error_energy / target_energy),
        "cg_iterations": iterations,
        "cg_best_iteration": best_iteration,
        "cg_final_normal_relative_residual": float(
            residual.double().norm() / initial_norm
        ),
    }


def fit_anchor(
    module: CompactTangent,
    basis: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    updates: int,
    learning_rate: float,
    mixture_width: int,
    bound: float,
    seed: int,
) -> list[dict[str, Any]]:
    optimizer = torch.optim.Adam(module.coordinate_tensors, lr=learning_rate)
    history: list[dict[str, Any]] = []
    for update in range(updates):
        target = deterministic_weighted_mixture(
            basis,
            probabilities,
            update=update,
            width=mixture_width,
            seed=seed,
        )
        optimizer.zero_grad(set_to_none=True)
        _action, cosine, score = natural_action(
            module, target, differentiable_anchor=True
        )
        regularizer = 1e-5 * torch.stack(
            [coordinate.square().mean() for coordinate in module.coordinate_tensors]
        ).mean()
        (-score + regularizer).backward()
        optimizer.step()
        module.clamp_coordinates(bound)
        if update == 0 or (update + 1) % 16 == 0 or update + 1 == updates:
            flat = flatten_tensors(
                tuple(tensor.detach() for tensor in module.coordinate_tensors)
            )
            history.append(
                {
                    "fit_update": update + 1,
                    "mixture_action_capture": float(score.detach()),
                    "mixture_action_cosine": float(cosine.detach()),
                    "anchor_rms": float(flat.square().mean().sqrt()),
                    "anchor_max_abs": float(flat.abs().max()),
                }
            )
    return history


def evaluate_basis(
    module: CompactTangent,
    basis: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    family: str,
    cg_iterations: int,
    cg_tolerance: float,
    cg_damping_ratio: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, target in enumerate(basis):
        _action, cosine, score = natural_action(
            module, target, differentiable_anchor=False
        )
        projection = cg_project(
            module,
            target,
            maximum_iterations=cg_iterations,
            relative_tolerance=cg_tolerance,
            damping_ratio=cg_damping_ratio,
        )
        rows.append(
            {
                "family": family,
                "pc": index + 1,
                "variance_weight": float(probabilities[index]),
                "natural_action_cosine": float(cosine),
                "natural_action_capture": float(score),
                **projection,
            }
        )
    weights = probabilities.double()
    captures = torch.tensor(
        [row["cg_projection_capture"] for row in rows],
        dtype=torch.float64,
        device=weights.device,
    )
    return rows, {
        "family": family,
        "weighted_cg_projection_capture": float((weights * captures).sum()),
        "minimum_cg_projection_capture": float(captures.min()),
        "maximum_cg_projection_capture": float(captures.max()),
        "maximum_cg_normal_relative_residual": max(
            row["cg_final_normal_relative_residual"] for row in rows
        ),
    }


def build_family(
    family: str,
    *,
    in_features: int,
    out_features: int,
    seed: int,
) -> CompactTangent:
    if family == "dtd3":
        return DiagonalToeplitzDiagonal(
            in_features, out_features, branches=3, seed=seed
        )
    if family == "expander10":
        return LearnedSparseExpander(
            in_features,
            out_features,
            padded_features=4096,
            depth=10,
            group_size=4,
            seed=seed,
        )
    if family == "tt24":
        return LiveTensorNetwork(
            in_features,
            out_features,
            row_modes=MATRIX_ROW_MODES,
            column_modes=MATRIX_COLUMN_MODES,
            topology="open",
            bond=24,
            seed=seed,
        )
    if family == "tr17":
        return LiveTensorNetwork(
            in_features,
            out_features,
            row_modes=MATRIX_ROW_MODES,
            column_modes=MATRIX_COLUMN_MODES,
            topology="ring",
            bond=17,
            seed=seed,
        )
    if family == "sinc6":
        return SinusoidalCoordinateField(
            in_features,
            out_features,
            rank=6,
            seed=seed,
        )
    raise ValueError(f"unsupported family {family}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--targets", default="mlp.c_fc,mlp.c_proj")
    parser.add_argument("--families", default="dtd3,expander10")
    parser.add_argument("--basis-rank", type=int, default=16)
    parser.add_argument("--base-seed", type=int, default=20260827)
    parser.add_argument("--fit-updates", type=int, default=128)
    parser.add_argument("--fit-lr", type=float, default=0.02)
    parser.add_argument("--mixture-width", type=int, default=4)
    parser.add_argument("--anchor-bound", type=float, default=0.5)
    parser.add_argument("--fit-seed", type=int, default=20260827)
    parser.add_argument("--cg-iterations", type=int, default=32)
    parser.add_argument("--self-check-cg-iterations", type=int, default=128)
    parser.add_argument("--cg-tolerance", type=float, default=1e-5)
    parser.add_argument("--cg-damping-ratio", type=float, default=1e-6)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    targets = {item for item in args.targets.split(",") if item}
    families = [item for item in args.families.split(",") if item]
    paths = sorted(args.snapshot_dir.glob("step_*.pt"))
    steps, values, snapshot_metadata = load_snapshots(
        paths, layers={args.layer}, targets=targets
    )
    all_rows: list[dict[str, Any]] = []
    all_summary: list[dict[str, Any]] = []
    all_history: list[dict[str, Any]] = []
    anchors: dict[str, Any] = {}
    accounting: dict[str, Any] = {}
    retained_fractions: dict[str, float] = {}
    for parameter, tensors in sorted(values.items()):
        match = PARAMETER_PATTERN.match(parameter)
        if match is None:
            raise ValueError(f"unsupported parameter {parameter}")
        target_name = match.group("target")
        positions = torch.stack(tensors).to(args.device, dtype=torch.float32)
        _residuals, eigenvalues, basis = residual_temporal_basis(
            positions, maximum_rank=args.basis_rank
        )
        retained = eigenvalues[: basis.shape[1]]
        probabilities = retained / retained.sum().clamp_min(1e-30)
        basis_matrices = basis.T.reshape(basis.shape[1], *positions.shape[1:])
        retained_fractions[parameter] = float(
            retained.sum() / eigenvalues.sum().clamp_min(1e-30)
        )
        out_features, in_features = positions.shape[1:]
        dense_scalars = out_features * in_features
        target_anchors: dict[str, Any] = {}
        for family_index, family in enumerate(families):
            module = build_family(
                family,
                in_features=in_features,
                out_features=out_features,
                seed=args.base_seed
                + 1009 * TARGET_SEED_OFFSETS[target_name]
                + 1000003 * family_index,
            ).to(args.device)
            if module.trainable_scalar_count > 0.01 * dense_scalars:
                raise ValueError(
                    f"{family} exceeds one-percent state: "
                    f"{module.trainable_scalar_count}/{dense_scalars}"
                )
            self_check_generator = torch.Generator(device=args.device).manual_seed(
                args.base_seed
                + 1009 * TARGET_SEED_OFFSETS[target_name]
                + 1000003 * family_index
                + 7000003
            )
            self_check_direction = torch.randn(
                module.trainable_scalar_count,
                generator=self_check_generator,
                device=args.device,
            )
            self_check = cg_project(
                module,
                module.jvp(self_check_direction),
                maximum_iterations=args.self_check_cg_iterations,
                relative_tolerance=min(args.cg_tolerance, 1e-7),
                damping_ratio=min(args.cg_damping_ratio, 1e-9),
            )
            if self_check["cg_projection_capture"] < 0.999:
                raise RuntimeError(
                    f"{family} failed own-tangent recovery: {self_check}"
                )
            history = fit_anchor(
                module,
                basis_matrices,
                probabilities,
                updates=args.fit_updates,
                learning_rate=args.fit_lr,
                mixture_width=args.mixture_width,
                bound=args.anchor_bound,
                seed=args.fit_seed
                + 1009 * TARGET_SEED_OFFSETS[target_name]
                + 1000003 * family_index,
            )
            rows, summary = evaluate_basis(
                module,
                basis_matrices,
                probabilities,
                family=family,
                cg_iterations=args.cg_iterations,
                cg_tolerance=args.cg_tolerance,
                cg_damping_ratio=args.cg_damping_ratio,
            )
            for row in rows:
                row.update({"parameter": parameter, "target": target_name})
            summary.update(
                {
                    "parameter": parameter,
                    "target": target_name,
                    "stored_scalars": module.trainable_scalar_count,
                    "stored_scalar_fraction": module.trainable_scalar_count
                    / dense_scalars,
                    "ideal_forward_scalar_ops": module.ideal_forward_scalar_ops(),
                    "ideal_forward_ops_to_dense_madds": module.ideal_forward_scalar_ops()
                    / dense_scalars,
                    "self_check_projection_capture": self_check[
                        "cg_projection_capture"
                    ],
                    "self_check_cg_iterations": self_check["cg_iterations"],
                    "self_check_cg_final_normal_relative_residual": self_check[
                        "cg_final_normal_relative_residual"
                    ],
                }
            )
            for row in history:
                row.update(
                    {
                        "parameter": parameter,
                        "target": target_name,
                        "family": family,
                    }
                )
            all_rows.extend(rows)
            all_summary.append(summary)
            all_history.extend(history)
            target_anchors[family] = {
                "coordinate_state": {
                    name: value.detach().cpu()
                    for name, value in module.named_parameters()
                },
                "seed": module.seed,
                "stored_scalars": module.trainable_scalar_count,
            }
            accounting[f"{parameter}:{family}"] = {
                "dense_scalars": dense_scalars,
                "stored_scalars": module.trainable_scalar_count,
                "stored_scalar_fraction": module.trainable_scalar_count
                / dense_scalars,
                "ideal_forward_scalar_ops": module.ideal_forward_scalar_ops(),
                "ideal_forward_ops_to_dense_madds": module.ideal_forward_scalar_ops()
                / dense_scalars,
            }
            del module
            torch.cuda.empty_cache()
        anchors[target_name] = target_anchors
        del positions, basis_matrices
        torch.cuda.empty_cache()

    args.output.mkdir(parents=True, exist_ok=True)
    rows_path = args.output / "pc_projection.csv"
    summary_path = args.output / "summary.csv"
    history_path = args.output / "fit_history.csv"
    anchors_path = args.output / "compact_anchors.pt"
    write_csv(rows_path, all_rows)
    write_csv(summary_path, all_summary)
    write_csv(history_path, all_history)
    torch.save(anchors, anchors_path)
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_mlp_residual_nonfht_basis_v1",
        "steps": steps,
        "snapshot_metadata": snapshot_metadata,
        "layer": args.layer,
        "targets": sorted(targets),
        "families": families,
        "basis_rank": args.basis_rank,
        "retained_residual_energy_fraction": retained_fractions,
        "fit": {
            "updates": args.fit_updates,
            "learning_rate": args.fit_lr,
            "mixture_width": args.mixture_width,
            "anchor_bound": args.anchor_bound,
        },
        "cg": {
            "iterations": args.cg_iterations,
            "self_check_iterations": args.self_check_cg_iterations,
            "relative_tolerance": args.cg_tolerance,
            "damping_ratio": args.cg_damping_ratio,
        },
        "accounting": accounting,
        "candidate_contract": {
            "stored": "live coordinates only",
            "procedural": "integer-seeded signs, kernels, matchings, grouping, base 2x2 rotations, and tensor-network anchor cores",
            "forbidden": "ambient PCA vector, dense atom, shadow weight, learned index table, or stored connectivity",
        },
        "analysis_execution": {
            "git_commit": git_commit(script.parents[2]),
            "entrypoint": str(script),
            "entrypoint_sha256": file_sha256(script),
            "command": sys.argv,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "device": args.device,
        },
        "outputs": {
            rows_path.name: file_sha256(rows_path),
            summary_path.name: file_sha256(summary_path),
            history_path.name: file_sha256(history_path),
            anchors_path.name: file_sha256(anchors_path),
        },
        "limitations": [
            "The same 239-state horizon is used for optimistic noncausal anchor fitting and evaluation.",
            "Jacobian recovery is necessary but not sufficient for online optimization or CE closure.",
            "Ideal forward arithmetic assumes fused matrix-free implementations that have not been performance-gated.",
        ],
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "summary": all_summary,
                "metadata": str(metadata_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
