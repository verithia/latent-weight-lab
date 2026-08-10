#!/usr/bin/env python3
"""Step-zero task-oriented Kronecker atlas for attention V and c_proj.

This module contains the representation-critical part of the preregistered
zero-update oracle.  It intentionally does not launch training.  A layerwise
atlas is selected from uncentered empirical second moments of the exact
step-zero model's linear inputs and backpropagated CE errors.  Later dense
states and directions may be projected into the frozen atlas, but they never
participate in selecting its basis.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import torch


TARGETS = ("v", "cproj")


def empirical_second_moment(rows: torch.Tensor) -> torch.Tensor:
    """Return the uncentered KFAC/Fisher factor ``E[row row^T]``."""

    if rows.ndim != 2 or rows.shape[0] == 0:
        raise ValueError("second-moment rows must be nonempty and rank two")
    values = rows.float()
    return values.T @ values / values.shape[0]


def top_kronecker_pairs(
    output_eigenvalues: torch.Tensor,
    input_eigenvalues: torch.Tensor,
    coordinate_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select tensor-product eigenvectors by KFAC eigenvalue product."""

    if output_eigenvalues.ndim != 1 or input_eigenvalues.ndim != 1:
        raise ValueError("Kronecker eigenvalues must be vectors")
    total = output_eigenvalues.numel() * input_eigenvalues.numel()
    if not 0 < int(coordinate_count) <= total:
        raise ValueError("coordinate_count is outside the Kronecker basis")
    products = (
        output_eigenvalues.clamp_min(0).reshape(-1, 1)
        * input_eigenvalues.clamp_min(0).reshape(1, -1)
    )
    values, indices = torch.topk(
        products.reshape(-1), int(coordinate_count), sorted=True
    )
    input_width = input_eigenvalues.numel()
    return indices // input_width, indices % input_width, values


@dataclass(frozen=True)
class KroneckerAtlas:
    """An orthonormal subset of a matrix Kronecker eigenbasis."""

    output_basis: torch.Tensor
    input_basis: torch.Tensor
    output_indices: torch.Tensor
    input_indices: torch.Tensor
    scores: torch.Tensor

    @classmethod
    def from_second_moments(
        cls,
        input_second_moment: torch.Tensor,
        output_second_moment: torch.Tensor,
        coordinate_count: int,
    ) -> "KroneckerAtlas":
        if (
            input_second_moment.ndim != 2
            or input_second_moment.shape[0] != input_second_moment.shape[1]
            or output_second_moment.ndim != 2
            or output_second_moment.shape[0] != output_second_moment.shape[1]
        ):
            raise ValueError("KFAC factors must be square matrices")
        input_values, input_vectors = torch.linalg.eigh(
            input_second_moment.float()
        )
        output_values, output_vectors = torch.linalg.eigh(
            output_second_moment.float()
        )
        output_indices, input_indices, scores = top_kronecker_pairs(
            output_values,
            input_values,
            int(coordinate_count),
        )
        return cls(
            output_basis=output_vectors.contiguous(),
            input_basis=input_vectors.contiguous(),
            output_indices=output_indices.contiguous(),
            input_indices=input_indices.contiguous(),
            scores=scores.contiguous(),
        )

    @property
    def coordinate_count(self) -> int:
        return int(self.output_indices.numel())

    @property
    def shape(self) -> tuple[int, int]:
        return self.output_basis.shape[0], self.input_basis.shape[0]

    def apply(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Map compact coordinates to a full matrix without a learned basis."""

        if coordinates.ndim != 1 or coordinates.numel() != self.coordinate_count:
            raise ValueError("coordinate vector does not match the atlas")
        core = coordinates.new_zeros(
            self.output_basis.shape[1], self.input_basis.shape[1]
        )
        core.index_put_(
            (
                self.output_indices.to(device=coordinates.device),
                self.input_indices.to(device=coordinates.device),
            ),
            coordinates,
            accumulate=True,
        )
        output = self.output_basis.to(
            device=coordinates.device, dtype=coordinates.dtype
        )
        inputs = self.input_basis.to(
            device=coordinates.device, dtype=coordinates.dtype
        )
        return output @ core @ inputs.T

    def adjoint(self, weight: torch.Tensor) -> torch.Tensor:
        """Apply the exact Frobenius adjoint of :meth:`apply`."""

        if weight.ndim != 2 or tuple(weight.shape) != self.shape:
            raise ValueError("weight matrix does not match the atlas")
        output = self.output_basis.to(device=weight.device, dtype=weight.dtype)
        inputs = self.input_basis.to(device=weight.device, dtype=weight.dtype)
        core = output.T @ weight @ inputs
        return core[
            self.output_indices.to(device=weight.device),
            self.input_indices.to(device=weight.device),
        ]

    def fixed_storage_bytes(self) -> int:
        tensors = (
            self.output_basis,
            self.input_basis,
            self.output_indices,
            self.input_indices,
            self.scores,
        )
        return sum(value.numel() * value.element_size() for value in tensors)


def kronecker_subspace_overlap(
    left: KroneckerAtlas,
    right: KroneckerAtlas,
    *,
    chunk_size: int = 256,
) -> float:
    """Return normalized ``||Q_left^T Q_right||_F^2`` without dense Q."""

    if left.shape != right.shape or left.coordinate_count != right.coordinate_count:
        raise ValueError("atlas overlap requires equal shapes and coordinate counts")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    device = left.output_basis.device
    output_cross = left.output_basis.T @ right.output_basis.to(device)
    input_cross = left.input_basis.T @ right.input_basis.to(device)
    right_output = right.output_indices.to(device)
    right_input = right.input_indices.to(device)
    total = output_cross.new_zeros(())
    for start in range(0, left.coordinate_count, int(chunk_size)):
        stop = min(start + int(chunk_size), left.coordinate_count)
        left_output = left.output_indices[start:stop].to(device)
        left_input = left.input_indices[start:stop].to(device)
        output_inner = output_cross[left_output[:, None], right_output[None, :]]
        input_inner = input_cross[left_input[:, None], right_input[None, :]]
        total = total + (output_inner * input_inner).square().sum()
    return float(total / left.coordinate_count)


class AttentionKFACCollector:
    """Capture step-zero inputs and output errors for V and attention c_proj."""

    def __init__(
        self,
        model: torch.nn.Module,
        layers: Iterable[int],
        sample_cap: int,
    ) -> None:
        self.layers = set(int(layer) for layer in layers)
        self.sample_cap = int(sample_cap)
        if self.sample_cap <= 0:
            raise ValueError("sample_cap must be positive")
        self.inputs: dict[tuple[int, str], list[torch.Tensor]] = defaultdict(list)
        self.errors: dict[tuple[int, str], list[torch.Tensor]] = defaultdict(list)
        self.counts: dict[tuple[int, str], int] = defaultdict(int)
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        for layer, block in enumerate(model.transformer.h):
            if layer not in self.layers:
                continue
            self.handles.append(
                block.attn.c_attn.register_forward_hook(
                    self._hook(layer, "v", final_value_slice=True)
                )
            )
            self.handles.append(
                block.attn.c_proj.register_forward_hook(
                    self._hook(layer, "cproj", final_value_slice=False)
                )
            )

    def _hook(self, layer: int, target: str, *, final_value_slice: bool):
        def hook(module, inputs, output):
            if not torch.is_tensor(output) or not output.requires_grad:
                raise RuntimeError("KFAC acquisition requires a CE backward pass")
            key = (layer, target)
            source = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
            take = min(self.sample_cap - self.counts[key], source.shape[0])
            if take <= 0:
                return
            self.inputs[key].append(source[:take].cpu())
            self.counts[key] += int(take)

            def save_error(gradient: torch.Tensor) -> None:
                error = gradient
                if final_value_slice:
                    n_embd = int(module.out_features) // 3
                    error = error[..., 2 * n_embd :]
                error = error.detach().float().reshape(-1, error.shape[-1])
                self.errors[key].append(error[:take].cpu())

            output.register_hook(save_error)

        return hook

    def complete(self) -> bool:
        return all(
            self.counts[(layer, target)] >= self.sample_cap
            and sum(value.shape[0] for value in self.errors[(layer, target)])
            >= self.sample_cap
            for layer in self.layers
            for target in TARGETS
        )

    def rows(self, layer: int, target: str) -> tuple[torch.Tensor, torch.Tensor]:
        key = (int(layer), target)
        if target not in TARGETS or not self.inputs[key] or not self.errors[key]:
            raise ValueError(f"missing KFAC rows for {key}")
        return (
            torch.cat(self.inputs[key], dim=0)[: self.sample_cap],
            torch.cat(self.errors[key], dim=0)[: self.sample_cap],
        )

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def collect_stepzero_second_moments(
    model: torch.nn.Module,
    batches: Iterable[torch.Tensor],
    layers: Iterable[int],
    sample_cap: int,
    device: str,
) -> dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]]:
    """Run exact step-zero CE backward passes and return KFAC factors."""

    selected_layers = [int(layer) for layer in layers]
    collector = AttentionKFACCollector(model, selected_layers, sample_cap)
    model.eval()
    try:
        for batch in batches:
            if batch.ndim != 2 or batch.shape[1] < 2:
                raise ValueError("KFAC batches must contain at least two tokens")
            tokens = batch[:, :-1].to(device)
            targets = batch[:, 1:].to(device)
            model.zero_grad(set_to_none=True)
            _logits, loss = model(tokens, targets)
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("step-zero calibration loss is not finite")
            loss.backward()
            if collector.complete():
                break
        if not collector.complete():
            raise RuntimeError("step-zero KFAC sample cap was not reached")
        result = {}
        for layer in selected_layers:
            for target in TARGETS:
                inputs, errors = collector.rows(layer, target)
                result[(layer, target)] = (
                    empirical_second_moment(inputs),
                    empirical_second_moment(errors),
                )
        return result
    finally:
        model.zero_grad(set_to_none=True)
        collector.close()

