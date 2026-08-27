#!/usr/bin/env python3
"""Fit non-Hadamard compact tangents to dense MLP residual PCs.

This optimistic, noncausal oracle compares two basis-free families under the
same approximately-one-percent state contract:

* three independently modulated rectangular Toeplitz operators; and
* a ten-stage learned sparse-expander chain of general 2x2 blocks.

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

    def normal(vector: torch.Tensor) -> torch.Tensor:
        return coordinate_vjp(module, module.jvp(vector)) + damping * vector

    estimate = torch.zeros_like(right)
    residual = right.clone()
    preconditioned = residual / (metric + damping).clamp_min(1e-12)
    direction = preconditioned.clone()
    rz = torch.dot(residual.double(), preconditioned.double())
    initial_norm = residual.double().norm().clamp_min(1e-30)
    iterations = 0
    for iteration in range(maximum_iterations):
        action = normal(direction)
        denominator = torch.dot(direction.double(), action.double()).clamp_min(1e-30)
        step = rz / denominator
        estimate.add_(direction, alpha=float(step))
        residual.add_(action, alpha=-float(step))
        iterations = iteration + 1
        if float(residual.double().norm() / initial_norm) <= relative_tolerance:
            break
        next_preconditioned = residual / (metric + damping).clamp_min(1e-12)
        next_rz = torch.dot(residual.double(), next_preconditioned.double())
        direction.mul_(float(next_rz / rz)).add_(next_preconditioned)
        rz = next_rz
    projected = module.jvp(estimate)
    target_energy = target.double().square().sum().clamp_min(1e-30)
    error_energy = (target - projected).double().square().sum()
    return {
        "cg_projection_capture": float(1.0 - error_energy / target_energy),
        "cg_iterations": iterations,
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
            "relative_tolerance": args.cg_tolerance,
            "damping_ratio": args.cg_damping_ratio,
        },
        "accounting": accounting,
        "candidate_contract": {
            "stored": "live coordinates only",
            "procedural": "integer-seeded signs, kernels, matchings, grouping, and base 2x2 rotations",
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
