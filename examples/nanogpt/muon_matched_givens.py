"""Task-selected sparse Givens updates for materialized MLP projections."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as F

from examples.nanogpt.muon import zeropower_via_newtonschulz5
from examples.nanogpt.fast_task_matching import (
    color_sorted_edges,
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


def _gelu_derivative(values: torch.Tensor) -> torch.Tensor:
    """Derivative of the exact erf GELU used by ``torch.nn.GELU``."""
    values = values.float()
    return 0.5 * (1.0 + torch.erf(values / math.sqrt(2.0))) + (
        values
        * torch.exp(-0.5 * values.square())
        / math.sqrt(2.0 * math.pi)
    )


@torch.no_grad()
def _weight_shear_permutations(
    source: torch.Tensor,
    residual: torch.Tensor,
    *,
    stages: int,
    neighbors: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Select sparse symmetric-shear pairs in weight Frobenius geometry."""
    source = source.float()
    residual = residual.float()
    width = int(source.shape[1])
    cross = source.T @ residual
    norm = source.square().sum(dim=0)
    scores = (cross + cross.T).square() / (
        norm[:, None] + norm[None, :]
    ).clamp_min(1e-30)
    scores.fill_diagonal_(-1.0)
    top_scores, top_indices = torch.topk(scores, k=neighbors, dim=1)
    order = torch.argsort(top_scores.reshape(-1), descending=True)
    left = (
        torch.arange(width, device=source.device)
        .repeat_interleave(neighbors)
        .index_select(0, order)
    )
    right = top_indices.reshape(-1).index_select(0, order)
    edges = torch.stack(
        (torch.minimum(left, right), torch.maximum(left, right)), dim=1
    ).to(device="cpu", dtype=torch.int32)
    permutations, diagnostics = color_sorted_edges(
        edges,
        width=width,
        stages=stages,
        seed=seed,
    )
    diagnostics.update(
        {
            "candidate_edges": int(edges.shape[0]),
            "preselection_neighbors": int(neighbors),
            "score_family": "weight_frobenius_symmetric_shear",
        }
    )
    return permutations, diagnostics


def _shear_coordinates(
    source: torch.Tensor,
    residual: torch.Tensor,
    pairs: torch.Tensor,
) -> torch.Tensor:
    """Exact one-coordinate tangent fit for disjoint symmetric shears."""
    left, right = pairs.unbind(dim=1)
    source_left = source[:, left].double()
    source_right = source[:, right].double()
    residual_left = residual[:, left].double()
    residual_right = residual[:, right].double()
    numerator = (
        (source_right * residual_left).sum(dim=0)
        + (source_left * residual_right).sum(dim=0)
    )
    denominator = (
        source_left.square().sum(dim=0)
        + source_right.square().sum(dim=0)
    ).clamp_min(1e-30)
    return numerator / denominator


def _apply_symmetric_shear_stage(
    source: torch.Tensor,
    pairs: torch.Tensor,
    coordinates: torch.Tensor,
) -> torch.Tensor:
    """Apply exact determinant-one ``exp([[0,s],[s,0]])`` pair maps."""
    left, right = pairs.unbind(dim=1)
    coordinate = coordinates.to(device=source.device, dtype=source.dtype)
    cosine = torch.cosh(coordinate)
    sine = torch.sinh(coordinate)
    source_left = source[:, left]
    source_right = source[:, right]
    result = source.clone()
    result[:, left] = cosine * source_left + sine * source_right
    result[:, right] = sine * source_left + cosine * source_right
    return result


@torch.no_grad()
def _fit_weight_shear_recipe(
    source: torch.Tensor,
    requested_update: torch.Tensor,
    permutations: torch.Tensor,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    current = source.float().clone()
    target = source.float() + requested_update.float()
    recipe: list[tuple[torch.Tensor, torch.Tensor]] = []
    for permutation in permutations:
        pairs = permutation.to(source.device).reshape(-1, 2)
        coordinates = _shear_coordinates(
            current, target - current, pairs
        )
        current = _apply_symmetric_shear_stage(
            current, pairs, coordinates
        )
        recipe.append((pairs, coordinates))
    return recipe


@torch.no_grad()
def _fit_functional_shear_recipe(
    source: torch.Tensor,
    requested_update: torch.Tensor,
    inputs: torch.Tensor,
    pre_gelu: torch.Tensor,
    cproj_weight: torch.Tensor,
    permutations: torch.Tensor,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Fit the registered exact linearized post-GELU/``c_proj`` metric."""
    current = source.float().clone()
    inputs = inputs.to(device=source.device, dtype=torch.float32)
    pre_gelu = pre_gelu.to(device=source.device, dtype=torch.float32)
    cproj = cproj_weight.to(device=source.device, dtype=torch.float32)
    slopes = _gelu_derivative(pre_gelu)
    projected = inputs @ current
    target_output = (
        slopes * (inputs @ requested_update.float())
    ) @ cproj.T
    residual_output = target_output.clone()
    cproj_gram = cproj.T @ cproj
    cproj_norm = cproj_gram.diagonal()
    recipe: list[tuple[torch.Tensor, torch.Tensor]] = []
    for permutation in permutations:
        pairs = permutation.to(source.device).reshape(-1, 2)
        left, right = pairs.unbind(dim=1)
        projected_target = residual_output @ cproj
        left_direction = slopes[:, left] * projected[:, right]
        right_direction = slopes[:, right] * projected[:, left]
        numerator = (
            left_direction * projected_target[:, left]
            + right_direction * projected_target[:, right]
        ).sum(dim=0)
        denominator = (
            cproj_norm[left] * left_direction.square().sum(dim=0)
            + cproj_norm[right] * right_direction.square().sum(dim=0)
            + 2.0
            * cproj_gram[left, right]
            * (left_direction * right_direction).sum(dim=0)
        ).clamp_min(1e-30)
        coordinates = numerator.double() / denominator.double()
        projected_updated = _apply_symmetric_shear_stage(
            projected, pairs, coordinates
        )
        residual_output.sub_(
            (slopes * (projected_updated - projected)) @ cproj.T
        )
        current = _apply_symmetric_shear_stage(
            current, pairs, coordinates
        )
        projected = projected_updated
        recipe.append((pairs, coordinates))
    return recipe


@torch.no_grad()
def functional_coordinate_mix_update(
    weight: torch.Tensor,
    requested_update: torch.Tensor,
    selection_direction: torch.Tensor,
    inputs: torch.Tensor,
    pre_gelu: torch.Tensor,
    cproj_weight: torch.Tensor,
    *,
    parent_stages: int,
    shear_stages: int,
    neighbors: int,
    seed: int,
    beta: float,
    learning_rate: float,
    weight_decay: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Reproduce the promoted fixed-topology mixed-coordinate ``c_fc`` step."""
    source = weight.float().T.contiguous()
    target = requested_update.float().T.contiguous()
    selection = selection_direction.float().T.contiguous()
    parent_permutations, parent_selection = fast_muon_matched_permutations(
        source,
        selection,
        stages=parent_stages,
        neighbors=neighbors,
        seed=seed,
    )
    parent_angles = diagonal_metric_angles(
        source, target, parent_permutations.to(source.device)
    )
    after_parent = apply_givens_flow(
        source,
        parent_angles,
        parent_permutations.to(source.device),
    )
    parent_update = after_parent - source
    residual = target - parent_update
    shear_permutations, shear_selection = _weight_shear_permutations(
        after_parent,
        residual,
        stages=shear_stages,
        neighbors=neighbors,
        seed=seed + 2,
    )
    shear_permutations = shear_permutations.to(source.device)
    weight_recipe = _fit_weight_shear_recipe(
        after_parent, residual, shear_permutations
    )
    functional_recipe = _fit_functional_shear_recipe(
        after_parent,
        residual,
        inputs,
        pre_gelu,
        cproj_weight,
        shear_permutations,
    )
    current = after_parent
    mixed_coordinates: list[torch.Tensor] = []
    for (weight_pairs, weight_coordinates), (
        functional_pairs,
        functional_coordinates,
    ) in zip(weight_recipe, functional_recipe, strict=True):
        if not torch.equal(weight_pairs, functional_pairs):
            raise RuntimeError("functional and weight shear topology differs")
        coordinates = (
            (1.0 - float(beta)) * weight_coordinates
            + float(beta) * functional_coordinates
        )
        current = _apply_symmetric_shear_stage(
            current, weight_pairs, coordinates
        )
        mixed_coordinates.append(coordinates)
    current.mul_(1.0 - float(learning_rate) * float(weight_decay))
    update = current.T.contiguous() - weight.float()
    requested_energy = requested_update.float().square().sum().clamp_min(1e-30)
    residual_energy = (
        requested_update.float() - update
    ).square().sum()
    all_shears = torch.cat(mixed_coordinates)
    return update, {
        "coordinates": int(
            (parent_stages + shear_stages) * source.shape[1] // 2
        ),
        "parent_stages": int(parent_stages),
        "shear_stages": int(shear_stages),
        "functional_coordinate_mix_beta": float(beta),
        "functional_samples": int(inputs.shape[0]),
        "angle_rms": float(parent_angles.square().mean().sqrt()),
        "shear_rms": float(all_shears.square().mean().sqrt()),
        "requested_update_recovery": float(
            1.0 - residual_energy / requested_energy
        ),
        "parent_matching": parent_selection,
        "shear_matching": shear_selection,
    }


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
        residual_stages: int,
        neighbors: int,
        refresh_interval: int,
        fast_fresh_matching: bool,
        matching_seed: int,
        weight_std: float,
        layer_id: int = -1,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.stages = int(stages)
        self.residual_stages = int(residual_stages)
        self.neighbors = int(neighbors)
        self.refresh_interval = int(refresh_interval)
        self.fast_fresh_matching = bool(fast_fresh_matching)
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
            or self.residual_stages < 0
            or self.residual_stages > 64
            or self.neighbors < max(
                self.stages, self.residual_stages
            )
            or self.neighbors >= self.in_features
        ):
            raise ValueError(
                "require 0 < stages, 0 <= residual_stages <= 64, "
                "and max(stages, residual_stages) <= neighbors "
                "< in_features"
            )
        if self.refresh_interval <= 0:
            raise ValueError("refresh_interval must be positive")
        if self.fast_fresh_matching and self.refresh_interval != 1:
            raise ValueError(
                "fast fresh matching requires refresh_interval=1"
            )
        if self.residual_stages and not self.fast_fresh_matching:
            raise ValueError(
                "residual matching requires fast fresh matching"
            )
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
        if self.residual_stages:
            residual_initial = torch.arange(self.in_features).repeat(
                self.residual_stages, 1
            )
            self.register_buffer(
                "residual_selected_permutations",
                residual_initial,
                persistent=True,
            )
            self.register_buffer(
                "residual_selected_inverse_permutations",
                torch.argsort(residual_initial, dim=1),
                persistent=True,
            )
            self.register_buffer(
                "residual_last_angles",
                torch.zeros(
                    self.residual_stages, self.in_features // 2
                ),
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
        return (
            self.stages + self.residual_stages
        ) * (self.in_features // 2)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.linear(values, self.weight, self.bias)


class MuonFunctionalShearLinear(nn.Module):
    """Materialized ``c_fc`` base updated by a bounded functional chart.

    The dense tensor is a persistent buffer owned by the custom optimizer,
    exactly as for :class:`MuonMatchedGivensLinear`.  Forward activations are
    detached into a bounded per-step sample used only to define the update
    metric; they are not learned parameters and are never checkpointed.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        parent_stages: int,
        shear_stages: int,
        neighbors: int,
        coordinate_mix_beta: float,
        functional_sample_cap: int,
        matching_seed: int,
        weight_std: float,
        layer_id: int,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.parent_stages = int(parent_stages)
        self.shear_stages = int(shear_stages)
        self.neighbors = int(neighbors)
        self.coordinate_mix_beta = float(coordinate_mix_beta)
        self.functional_sample_cap = int(functional_sample_cap)
        self.matching_seed = int(matching_seed)
        self.layer_id = int(layer_id)
        if (
            self.in_features <= 0
            or self.out_features <= 0
            or self.out_features % 2
            or self.parent_stages <= 0
            or self.shear_stages <= 0
            or self.parent_stages > 64
            or self.shear_stages > 64
            or self.neighbors < max(self.parent_stages, self.shear_stages)
            or self.neighbors >= self.out_features
            or not 0.0 <= self.coordinate_mix_beta <= 1.0
            or self.functional_sample_cap <= 0
            or not math.isfinite(weight_std)
            or weight_std <= 0.0
        ):
            raise ValueError("invalid functional-shear c_fc configuration")
        self.weight_std = float(weight_std)
        weight = torch.empty(self.out_features, self.in_features)
        # Match ``nn.Linear`` constructor RNG consumption. GPT's shared
        # initializer replaces this with the configured normal distribution,
        # preserving paired-seed initialization against the dense control.
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5.0))
        self.register_buffer("weight", weight, persistent=True)
        if bias:
            bound = 1.0 / math.sqrt(self.in_features)
            self.bias = nn.Parameter(
                torch.empty(self.out_features).uniform_(-bound, bound)
            )
        else:
            self.bias = None
        self.register_buffer(
            "optimizer_step",
            torch.zeros((), dtype=torch.int64),
            persistent=True,
        )
        self._functional_inputs: torch.Tensor | None = None
        self._functional_pre_gelu: torch.Tensor | None = None

    @property
    def coordinate_count(self) -> int:
        return (
            (self.parent_stages + self.shear_stages)
            * (self.out_features // 2)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.linear(values, self.weight, self.bias)

    @torch.no_grad()
    def record_functional_context(
        self, inputs: torch.Tensor, pre_gelu: torch.Tensor
    ) -> None:
        if not self.training or self._functional_inputs is not None:
            return
        if inputs.shape[:-1] != pre_gelu.shape[:-1]:
            raise ValueError("functional c_fc context is not aligned")
        flat_inputs = inputs.detach().reshape(-1, self.in_features)
        flat_pre = pre_gelu.detach().reshape(-1, self.out_features)
        count = min(self.functional_sample_cap, flat_inputs.shape[0])
        self._functional_inputs = flat_inputs[:count].contiguous()
        self._functional_pre_gelu = flat_pre[:count].contiguous()

    def consume_functional_context(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self._functional_inputs
        pre_gelu = self._functional_pre_gelu
        self._functional_inputs = None
        self._functional_pre_gelu = None
        if inputs is None or pre_gelu is None:
            raise RuntimeError(
                f"missing functional c_fc context for layer {self.layer_id}"
            )
        return inputs, pre_gelu

    def clear_functional_context(self) -> None:
        self._functional_inputs = None
        self._functional_pre_gelu = None


class MuonFunctionalShear(torch.optim.Optimizer):
    """Muon state with the promoted mixed functional/weight ``c_fc`` step."""

    def __init__(
        self,
        module_pairs: list[tuple[MuonFunctionalShearLinear, nn.Module]],
        *,
        lr: float,
        momentum: float,
        weight_decay: float,
        ns_steps: int,
    ) -> None:
        if not module_pairs:
            raise ValueError("MuonFunctionalShear requires at least one module")
        modules = [module for module, _cproj in module_pairs]
        for module in modules:
            module.weight.requires_grad_(True)
        self.modules_by_id = {id(module.weight): module for module in modules}
        self.cproj_by_id = {
            id(module.weight): cproj for module, cproj in module_pairs
        }
        self.last_step_diagnostics: list[dict[str, Any]] = []
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
                cproj = self.cproj_by_id[id(weight)]
                cproj_weight = getattr(cproj, "weight", None)
                if cproj_weight is None:
                    raise RuntimeError("functional c_fc requires c_proj.weight")
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
                requested_update = lr * (
                    direction - weight_decay * weight.float()
                )
                inputs, pre_gelu = module.consume_functional_context()
                update, row = functional_coordinate_mix_update(
                    weight,
                    requested_update,
                    direction,
                    inputs,
                    pre_gelu,
                    cproj_weight,
                    parent_stages=module.parent_stages,
                    shear_stages=module.shear_stages,
                    neighbors=module.neighbors,
                    seed=module.matching_seed + int(module.optimizer_step),
                    beta=module.coordinate_mix_beta,
                    learning_rate=lr,
                    weight_decay=weight_decay,
                )
                weight.add_(update.to(dtype=weight.dtype))
                row.update(
                    {
                        "step": int(module.optimizer_step),
                        "layer": module.layer_id,
                        "report_refresh": int(module.optimizer_step) == 0,
                        "optimizer": "muon_functional_shear",
                    }
                )
                diagnostics.append(row)
                module.optimizer_step.add_(1)
        self.last_step_diagnostics = diagnostics
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        super().zero_grad(set_to_none=set_to_none)
        for module in self.modules_by_id.values():
            module.clear_functional_context()

    def consume_diagnostics(self) -> list[dict[str, Any]]:
        diagnostics = self.last_step_diagnostics
        self.last_step_diagnostics = []
        return diagnostics


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
                    module.fast_fresh_matching
                    or not bool(module.matching_valid)
                    or step_index % module.refresh_interval == 0
                )
                matching_summary = None
                residual_matching_summary = None
                if refresh:
                    if module.fast_fresh_matching:
                        permutations, matching_diagnostics = (
                            fast_muon_matched_permutations(
                                weight,
                                direction,
                                stages=module.stages,
                                neighbors=module.neighbors,
                                seed=module.matching_seed + step_index,
                            )
                        )
                        matching_summary = {
                            "selector": "fast_fresh_single_pass",
                            "candidate_edge_fraction": float(
                                matching_diagnostics[
                                    "candidate_edge_fraction"
                                ]
                            ),
                            "minimum_stage_candidate_edge_fraction": (
                                float(
                                    matching_diagnostics[
                                        "minimum_stage_candidate_edge_fraction"
                                    ]
                                )
                            ),
                            "prepared_seconds": float(
                                matching_diagnostics[
                                    "prepared_seconds"
                                ]
                            ),
                            "native_seconds": float(
                                matching_diagnostics["native_seconds"]
                            ),
                            "total_seconds": float(
                                matching_diagnostics["total_seconds"]
                            ),
                            "native_output_validated": bool(
                                matching_diagnostics[
                                    "native_output_validated"
                                ]
                            ),
                            "native_library_sha256": str(
                                matching_diagnostics[
                                    "native_library_sha256"
                                ]
                            ),
                            "native_source_sha256": str(
                                matching_diagnostics["source_sha256"]
                            ),
                        }
                    else:
                        permutations, matching_diagnostics = (
                            muon_matched_permutations(
                                weight,
                                direction,
                                stages=module.stages,
                                neighbors=module.neighbors,
                                seed=module.matching_seed,
                            )
                        )
                        matching_summary = {
                            "selector": "legacy_greedy",
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
                requested_update = lr * (
                    direction - weight_decay * weight.float()
                )
                parent_angles = diagonal_metric_angles(
                    weight,
                    requested_update,
                    module.selected_permutations,
                )
                after_parent = apply_givens_flow(
                    weight,
                    parent_angles,
                    module.selected_permutations,
                    module.selected_inverse_permutations,
                )
                parent_update = after_parent.float() - weight.float()
                residual_update = (
                    requested_update.float() - parent_update
                )
                residual_angles = None
                rotated = after_parent
                if module.residual_stages:
                    residual_permutations, residual_diagnostics = (
                        fast_muon_matched_permutations(
                            after_parent,
                            residual_update,
                            stages=module.residual_stages,
                            neighbors=module.neighbors,
                            seed=(
                                module.matching_seed
                                + step_index
                                + 1
                            ),
                        )
                    )
                    module.residual_selected_permutations.copy_(
                        residual_permutations.to(
                            device=weight.device,
                            dtype=torch.long,
                        )
                    )
                    module.residual_selected_inverse_permutations.copy_(
                        torch.argsort(
                            module.residual_selected_permutations,
                            dim=1,
                        )
                    )
                    residual_angles = diagonal_metric_angles(
                        after_parent,
                        residual_update,
                        module.residual_selected_permutations,
                    )
                    rotated = apply_givens_flow(
                        after_parent,
                        residual_angles,
                        module.residual_selected_permutations,
                        module.residual_selected_inverse_permutations,
                    )
                    residual_matching_summary = {
                        "selector": "fast_fresh_residual_pass",
                        "candidate_edge_fraction": float(
                            residual_diagnostics[
                                "candidate_edge_fraction"
                            ]
                        ),
                        "minimum_stage_candidate_edge_fraction": (
                            float(
                                residual_diagnostics[
                                    "minimum_stage_candidate_edge_fraction"
                                ]
                            )
                        ),
                        "prepared_seconds": float(
                            residual_diagnostics[
                                "prepared_seconds"
                            ]
                        ),
                        "native_seconds": float(
                            residual_diagnostics["native_seconds"]
                        ),
                        "total_seconds": float(
                            residual_diagnostics["total_seconds"]
                        ),
                        "native_output_validated": bool(
                            residual_diagnostics[
                                "native_output_validated"
                            ]
                        ),
                        "native_library_sha256": str(
                            residual_diagnostics[
                                "native_library_sha256"
                            ]
                        ),
                        "native_source_sha256": str(
                            residual_diagnostics["source_sha256"]
                        ),
                    }
                if weight_decay != 0.0:
                    rotated.mul_(1.0 - lr * weight_decay)
                update = rotated - weight
                requested_energy = requested_update.float().square().sum()
                residual_energy = (
                    requested_update.float() - update.float()
                ).square().sum()
                weight.copy_(rotated)
                module.last_angles.copy_(
                    parent_angles.to(dtype=module.last_angles.dtype)
                )
                if residual_angles is not None:
                    module.residual_last_angles.copy_(
                        residual_angles.to(
                            dtype=module.residual_last_angles.dtype
                        )
                    )
                module.optimizer_step.add_(1)
                all_angles = (
                    torch.cat((parent_angles, residual_angles), dim=0)
                    if residual_angles is not None
                    else parent_angles
                )
                diagnostics.append(
                    {
                        "step": step_index,
                        "layer": module.layer_id,
                        "refresh": refresh,
                        "report_refresh": (
                            refresh
                            and (
                                not module.fast_fresh_matching
                                or step_index == 0
                            )
                        ),
                        "fast_fresh_matching": (
                            module.fast_fresh_matching
                        ),
                        "refresh_count": int(module.refresh_count),
                        "coordinates": module.coordinate_count,
                        "angle_rms": float(
                            all_angles.square().mean().sqrt()
                        ),
                        "angle_max_abs": float(
                            all_angles.abs().max()
                        ),
                        "parent_stages": module.stages,
                        "residual_stages": module.residual_stages,
                        "requested_update_recovery": float(
                            1.0
                            - residual_energy
                            / requested_energy.clamp_min(1e-30)
                        ),
                        "matching": matching_summary,
                        "residual_matching": (
                            residual_matching_summary
                        ),
                    }
                )
        self.last_step_diagnostics = diagnostics
        return loss

    def consume_diagnostics(self) -> list[dict[str, Any]]:
        diagnostics = self.last_step_diagnostics
        self.last_step_diagnostics = []
        return diagnostics
