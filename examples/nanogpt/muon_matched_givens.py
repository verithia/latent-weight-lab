"""Task-selected sparse Givens updates for materialized MLP projections."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as F

from examples.nanogpt.muon import zeropower_via_newtonschulz5
from examples.nanogpt.fast_task_matching import (
    fast_muon_matched_permutations,
)


def _complete_unique_matchings(
    edge_scores: dict[tuple[int, int], float],
    *,
    width: int,
    stages: int,
    seed: int,
) -> tuple[torch.Tensor, list[dict[str, float | int]]]:
    """Greedily edge-color scores into unique perfect matchings."""
    ordered_edges = sorted(
        (
            (score, left, right)
            for (left, right), score in edge_scores.items()
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    used_edges: set[tuple[int, int]] = set()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    permutations: list[torch.Tensor] = []
    diagnostics: list[dict[str, float | int]] = []
    for stage in range(stages):
        occupied = [False] * width
        pairs: list[tuple[int, int, float, bool]] = []
        for score, left, right in ordered_edges:
            edge = (left, right)
            if edge in used_edges or occupied[left] or occupied[right]:
                continue
            occupied[left] = occupied[right] = True
            used_edges.add(edge)
            pairs.append((left, right, score, True))
            if len(pairs) == width // 2:
                break

        remaining = [
            index
            for index, is_occupied in enumerate(occupied)
            if not is_occupied
        ]
        if remaining:
            order = torch.randperm(
                len(remaining), generator=generator
            ).tolist()
            remaining = [remaining[index] for index in order]
        while remaining:
            left = remaining.pop()
            partner_index = next(
                (
                    index
                    for index, right in enumerate(remaining)
                    if (min(left, right), max(left, right))
                    not in used_edges
                ),
                None,
            )
            if partner_index is None:
                repaired = False
                for right_index, right in enumerate(remaining):
                    for pair_index, (
                        prior_left,
                        prior_right,
                        _prior_score,
                        _prior_candidate,
                    ) in enumerate(pairs):
                        prior_edge = (
                            min(prior_left, prior_right),
                            max(prior_left, prior_right),
                        )
                        for first, second in (
                            (prior_left, prior_right),
                            (prior_right, prior_left),
                        ):
                            first_edge = (
                                min(left, first), max(left, first)
                            )
                            second_edge = (
                                min(right, second), max(right, second)
                            )
                            if (
                                first_edge in used_edges
                                or second_edge in used_edges
                                or first_edge == second_edge
                            ):
                                continue
                            used_edges.remove(prior_edge)
                            used_edges.add(first_edge)
                            used_edges.add(second_edge)
                            pairs[pair_index] = (
                                left,
                                first,
                                edge_scores.get(first_edge, 0.0),
                                first_edge in edge_scores,
                            )
                            pairs.append(
                                (
                                    right,
                                    second,
                                    edge_scores.get(second_edge, 0.0),
                                    second_edge in edge_scores,
                                )
                            )
                            remaining.pop(right_index)
                            repaired = True
                            break
                        if repaired:
                            break
                    if repaired:
                        break
                if not repaired:
                    raise RuntimeError(
                        "could not complete a unique task matching"
                    )
                continue
            right = remaining.pop(partner_index)
            edge = (min(left, right), max(left, right))
            used_edges.add(edge)
            pairs.append(
                (
                    left,
                    right,
                    edge_scores.get(edge, 0.0),
                    edge in edge_scores,
                )
            )

        if len(pairs) != width // 2:
            raise RuntimeError("task matching is incomplete")
        permutation = torch.tensor(
            [
                index
                for left, right, _score, _candidate in pairs
                for index in (left, right)
            ],
            dtype=torch.long,
        )
        if not torch.equal(
            torch.sort(permutation).values, torch.arange(width)
        ):
            raise RuntimeError("task matching is not a permutation")
        permutations.append(permutation)
        diagnostics.append(
            {
                "stage": stage,
                "pairs": len(pairs),
                "candidate_edge_fraction": (
                    sum(candidate for *_rest, candidate in pairs)
                    / len(pairs)
                ),
                "mean_abs_coordinate_gradient": (
                    sum(
                        score
                        for _left, _right, score, _candidate in pairs
                    )
                    / len(pairs)
                ),
            }
        )
    return torch.stack(permutations), diagnostics


def muon_matched_permutations(
    weight: torch.Tensor,
    direction: torch.Tensor,
    *,
    stages: int,
    neighbors: int,
    seed: int,
) -> tuple[torch.Tensor, list[dict[str, float | int]]]:
    """Select hidden-channel pairs from the identity-angle gradient."""
    if (
        weight.ndim != 2
        or weight.shape != direction.shape
        or weight.shape[1] <= 0
        or weight.shape[1] % 2
    ):
        raise ValueError(
            "weight and direction must be same-shaped matrices with even width"
        )
    width = int(weight.shape[1])
    if stages <= 0 or neighbors < stages or neighbors >= width:
        raise ValueError("require 0 < stages <= neighbors < width")
    weight = weight.float()
    direction = direction.float()
    cross = weight.T @ direction
    scores = (cross - cross.T).abs()
    scores.fill_diagonal_(-1.0)
    top_scores, top_indices = torch.topk(scores, k=neighbors, dim=1)
    top_scores = top_scores.cpu()
    top_indices = top_indices.cpu()
    del cross, scores

    edge_scores: dict[tuple[int, int], float] = {}
    for left in range(width):
        for raw_score, raw_right in zip(
            top_scores[left].tolist(),
            top_indices[left].tolist(),
            strict=True,
        ):
            right = int(raw_right)
            edge = (left, right) if left < right else (right, left)
            edge_scores[edge] = max(
                float(raw_score), edge_scores.get(edge, -1.0)
            )
    return _complete_unique_matchings(
        edge_scores, width=width, stages=stages, seed=seed
    )


def random_unique_matchings(
    *,
    width: int,
    stages: int,
    seed: int,
) -> torch.Tensor:
    """Return deterministic task-independent edge-disjoint matchings.

    A randomized relabeling followed by the circle-method one-factorization
    gives ``width - 1`` perfect matchings in which every unordered pair
    appears exactly once.  This is the equal-coordinate random-connectivity
    control for task-selected Givens charts.
    """
    if width <= 0 or width % 2 or stages <= 0 or stages >= width:
        raise ValueError("require even width and 0 < stages < width")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    vertices = torch.randperm(width, generator=generator).tolist()
    result: list[torch.Tensor] = []
    for _stage in range(stages):
        pairs = [
            (vertices[index], vertices[-index - 1])
            for index in range(width // 2)
        ]
        result.append(
            torch.tensor(
                [item for pair in pairs for item in pair],
                dtype=torch.long,
            )
        )
        vertices = [vertices[0], vertices[-1], *vertices[1:-1]]
    return torch.stack(result)


def diagonal_metric_angles(
    weight: torch.Tensor,
    requested_update: torch.Tensor,
    permutations: torch.Tensor,
) -> torch.Tensor:
    """Closed-form diagonal tangent-metric coordinates at the identity."""
    if weight.ndim != 2 or weight.shape != requested_update.shape:
        raise ValueError(
            "weight and requested_update must be same-shaped matrices"
        )
    if (
        permutations.ndim != 2
        or permutations.shape[1] != weight.shape[1]
    ):
        raise ValueError("permutations have an incompatible shape")
    stages = int(permutations.shape[0])
    angles = torch.empty(
        stages,
        weight.shape[1] // 2,
        device=weight.device,
        dtype=torch.float32,
    )
    source = weight.float()
    update = requested_update.float()
    for stage in range(stages):
        pairs = permutations[stage].view(-1, 2)
        left = pairs[:, 0]
        right = pairs[:, 1]
        weight_left = source.index_select(-1, left)
        weight_right = source.index_select(-1, right)
        update_left = update.index_select(-1, left)
        update_right = update.index_select(-1, right)
        coordinate_inner = (
            (weight_left * update_right).sum(dim=0)
            - (weight_right * update_left).sum(dim=0)
        )
        coordinate_norm = (
            weight_left.square().sum(dim=0)
            + weight_right.square().sum(dim=0)
        ).clamp_min(1e-30)
        angles[stage] = coordinate_inner / coordinate_norm
    return angles


def apply_givens_flow(
    values: torch.Tensor,
    angles: torch.Tensor,
    permutations: torch.Tensor,
    inverse_permutations: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the selected pair rotations along the last dimension."""
    if (
        angles.ndim != 2
        or permutations.ndim != 2
        or angles.shape[0] != permutations.shape[0]
        or permutations.shape[1] != values.shape[-1]
        or angles.shape[1] * 2 != values.shape[-1]
    ):
        raise ValueError("Givens flow shapes are incompatible")
    if inverse_permutations is not None and (
        inverse_permutations.shape != permutations.shape
    ):
        raise ValueError("inverse_permutations have an incompatible shape")
    result = values
    for stage in range(int(angles.shape[0])):
        permutation = permutations[stage]
        inverse = (
            torch.argsort(permutation)
            if inverse_permutations is None
            else inverse_permutations[stage]
        )
        permuted = result.index_select(-1, permutation)
        pairs = permuted.reshape(
            *permuted.shape[:-1], angles.shape[1], 2
        )
        angle = angles[stage].to(
            device=values.device, dtype=values.dtype
        )
        cosine = angle.cos()
        sine = angle.sin()
        first = cosine * pairs[..., 0] - sine * pairs[..., 1]
        second = sine * pairs[..., 0] + cosine * pairs[..., 1]
        rotated = torch.stack((first, second), dim=-1).reshape_as(
            permuted
        )
        result = rotated.index_select(-1, inverse)
    return result


class MuonMatchedGivensLinear(nn.Module):
    """Dense folded base updated through sparse task-selected rotations.

    The materialized weight is a persistent, gradient-bearing buffer rather
    than an optimizer-visible model parameter.  The custom optimizer owns it,
    so gradient scaling, clipping, exact Muon momentum, and resume state stay
    in the normal optimizer/checkpoint path without reporting a dense
    trainable parameter tensor.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        stages: int,
        neighbors: int,
        refresh_interval: int,
        matching_seed: int,
        weight_std: float,
        layer_id: int = -1,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.stages = int(stages)
        self.neighbors = int(neighbors)
        self.refresh_interval = int(refresh_interval)
        self.matching_seed = int(matching_seed)
        self.layer_id = int(layer_id)
        if (
            self.in_features <= 0
            or self.in_features % 2
            or self.out_features <= 0
        ):
            raise ValueError(
                "MuonMatchedGivensLinear requires positive dimensions "
                "and an even input width"
            )
        if (
            self.stages <= 0
            or self.neighbors < self.stages
            or self.neighbors >= self.in_features
        ):
            raise ValueError(
                "require 0 < stages <= neighbors < in_features"
            )
        if self.refresh_interval <= 0:
            raise ValueError("refresh_interval must be positive")
        if not math.isfinite(weight_std) or weight_std <= 0.0:
            raise ValueError("weight_std must be positive and finite")

        weight = torch.empty(self.out_features, self.in_features)
        nn.init.normal_(weight, mean=0.0, std=float(weight_std))
        self.register_buffer("weight", weight, persistent=True)
        self.bias = (
            nn.Parameter(torch.zeros(self.out_features))
            if bias
            else None
        )
        initial = torch.arange(self.in_features).repeat(
            self.stages, 1
        )
        self.register_buffer(
            "selected_permutations", initial, persistent=True
        )
        self.register_buffer(
            "selected_inverse_permutations",
            torch.argsort(initial, dim=1),
            persistent=True,
        )
        self.register_buffer(
            "last_angles",
            torch.zeros(self.stages, self.in_features // 2),
            persistent=True,
        )
        self.register_buffer(
            "optimizer_step",
            torch.zeros((), dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "last_refresh_step",
            torch.full((), -1, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "refresh_count",
            torch.zeros((), dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "matching_valid",
            torch.zeros((), dtype=torch.bool),
            persistent=True,
        )

    @property
    def coordinate_count(self) -> int:
        return self.stages * (self.in_features // 2)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.linear(values, self.weight, self.bias)


class MuonMatchedGivens(torch.optim.Optimizer):
    """Muon state with a folded sparse-Givens materialized-weight step."""

    def __init__(
        self,
        modules: list[MuonMatchedGivensLinear],
        *,
        lr: float,
        momentum: float,
        weight_decay: float,
        ns_steps: int,
    ) -> None:
        if not modules:
            raise ValueError("MuonMatchedGivens requires at least one module")
        for module in modules:
            module.weight.requires_grad_(True)
        self.modules_by_id = {id(module.weight): module for module in modules}
        self.last_step_diagnostics: list[dict[str, Any]] = []
        defaults = {
            "lr": float(lr),
            "momentum": float(momentum),
            "weight_decay": float(weight_decay),
            "ns_steps": int(ns_steps),
        }
        super().__init__(
            [{"params": [module.weight for module in modules]}],
            defaults,
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        diagnostics: list[dict[str, Any]] = []
        for group in self.param_groups:
            lr = float(group["lr"])
            momentum = float(group["momentum"])
            weight_decay = float(group["weight_decay"])
            ns_steps = int(group["ns_steps"])
            for weight in group["params"]:
                gradient = weight.grad
                if gradient is None:
                    continue
                module = self.modules_by_id[id(weight)]
                state = self.state[weight]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(weight)
                buffer = state["momentum_buffer"]
                buffer.mul_(momentum).add_(gradient)
                combined = gradient.add(buffer, alpha=momentum)
                polar = zeropower_via_newtonschulz5(
                    combined, steps=ns_steps
                ).float()
                scale = max(
                    1.0,
                    polar.shape[0]
                    / max(1, polar.numel() / polar.shape[0]),
                ) ** 0.5
                direction = -scale * polar
                step_index = int(module.optimizer_step)
                refresh = (
                    not bool(module.matching_valid)
                    or step_index % module.refresh_interval == 0
                )
                matching_summary = None
                if refresh:
                    permutations, matching_diagnostics = (
                        muon_matched_permutations(
                            weight,
                            direction,
                            stages=module.stages,
                            neighbors=module.neighbors,
                            seed=module.matching_seed,
                        )
                    )
                    module.selected_permutations.copy_(
                        permutations.to(
                            device=weight.device,
                            dtype=torch.long,
                        )
                    )
                    module.selected_inverse_permutations.copy_(
                        torch.argsort(
                            module.selected_permutations,
                            dim=1,
                        )
                    )
                    module.matching_valid.fill_(True)
                    module.last_refresh_step.fill_(step_index)
                    module.refresh_count.add_(1)
                    matching_summary = {
                        "mean_candidate_edge_fraction": sum(
                            float(
                                row["candidate_edge_fraction"]
                            )
                            for row in matching_diagnostics
                        )
                        / len(matching_diagnostics),
                        "mean_abs_coordinate_gradient": sum(
                            float(
                                row[
                                    "mean_abs_coordinate_gradient"
                                ]
                            )
                            for row in matching_diagnostics
                        )
                        / len(matching_diagnostics),
                    }
                requested_update = lr * (
                    direction - weight_decay * weight.float()
                )
                angles = diagonal_metric_angles(
                    weight,
                    requested_update,
                    module.selected_permutations,
                )
                rotated = apply_givens_flow(
                    weight,
                    angles,
                    module.selected_permutations,
                    module.selected_inverse_permutations,
                )
                if weight_decay != 0.0:
                    rotated.mul_(1.0 - lr * weight_decay)
                update = rotated - weight
                requested_energy = requested_update.float().square().sum()
                residual_energy = (
                    requested_update.float() - update.float()
                ).square().sum()
                weight.copy_(rotated)
                module.last_angles.copy_(
                    angles.to(dtype=module.last_angles.dtype)
                )
                module.optimizer_step.add_(1)
                diagnostics.append(
                    {
                        "step": step_index,
                        "layer": module.layer_id,
                        "refresh": refresh,
                        "refresh_count": int(module.refresh_count),
                        "coordinates": module.coordinate_count,
                        "angle_rms": float(
                            angles.square().mean().sqrt()
                        ),
                        "angle_max_abs": float(angles.abs().max()),
                        "requested_update_recovery": float(
                            1.0
                            - residual_energy
                            / requested_energy.clamp_min(1e-30)
                        ),
                        "matching": matching_summary,
                    }
                )
        self.last_step_diagnostics = diagnostics
        return loss

    def consume_diagnostics(self) -> list[dict[str, Any]]:
        diagnostics = self.last_step_diagnostics
        self.last_step_diagnostics = []
        return diagnostics
