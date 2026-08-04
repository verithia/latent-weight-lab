"""Task-selected sparse Givens updates for materialized MLP projections."""

from __future__ import annotations

import math
import os
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


@torch.no_grad()
def _score_selected_permutations(
    scores: torch.Tensor,
    *,
    stages: int,
    neighbors: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compile deterministic matchings from a dense symmetric edge score."""
    if (
        scores.ndim != 2
        or scores.shape[0] != scores.shape[1]
        or scores.shape[0] % 2
    ):
        raise ValueError("scores must be an even square matrix")
    width = int(scores.shape[0])
    if not 0 < stages <= neighbors < width:
        raise ValueError("require 0 < stages <= neighbors < width")
    local = scores.float().clone()
    local.fill_diagonal_(-torch.inf)
    top_scores, top_indices = torch.topk(local, k=neighbors, dim=1)
    order = torch.argsort(top_scores.reshape(-1), descending=True)
    left = (
        torch.arange(width, device=scores.device)
        .repeat_interleave(neighbors)
        .index_select(0, order)
    )
    right = top_indices.reshape(-1).index_select(0, order)
    edges = torch.stack(
        (torch.minimum(left, right), torch.maximum(left, right)), dim=1
    ).to(device="cpu", dtype=torch.int32)
    permutations, diagnostics = color_sorted_edges(
        edges, width=width, stages=stages, seed=seed
    )
    diagnostics["candidate_edges"] = int(edges.shape[0])
    return permutations, diagnostics


@torch.no_grad()
def task_gradient_output_pass(
    source: torch.Tensor,
    residual: torch.Tensor,
    task_gradient: torch.Tensor,
    *,
    stages: int,
    neighbors: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Apply the preregistered residual-plus-task selected output rotation."""
    if (
        source.ndim != 2
        or residual.shape != source.shape
        or task_gradient.shape != source.shape
    ):
        raise ValueError("source, residual, and task gradient must agree")
    source_f = source.float()
    residual_f = residual.float()
    gradient_f = task_gradient.float()
    residual_cross = source_f.T @ residual_f
    residual_inner = residual_cross - residual_cross.T
    column_energy = source_f.square().sum(dim=0)
    coordinate_norm = (
        column_energy[:, None] + column_energy[None, :]
    ).clamp_min(1e-30)
    angle = residual_inner / coordinate_norm
    residual_score = residual_inner.square() / coordinate_norm
    gradient_cross = source_f.T @ gradient_f
    task_inner = gradient_cross - gradient_cross.T
    task_score = -angle * task_inner
    mask = torch.triu(
        torch.ones_like(residual_score, dtype=torch.bool), diagonal=1
    )
    residual_rms = residual_score[mask].square().mean().sqrt().clamp_min(1e-30)
    task_rms = task_score[mask].square().mean().sqrt().clamp_min(1e-30)
    scores = residual_score / residual_rms + task_score / task_rms
    permutations, matching = _score_selected_permutations(
        scores, stages=stages, neighbors=neighbors, seed=seed
    )
    permutations = permutations.to(source.device)
    angles = diagonal_metric_angles(source_f, residual_f, permutations)
    updated = apply_givens_flow(
        source_f,
        angles,
        permutations,
        torch.argsort(permutations, dim=1),
    )
    return updated, permutations, angles, {
        **matching,
        "coordinates": int(stages * source.shape[1] // 2),
        "residual_score_rms": float(residual_rms),
        "task_score_rms": float(task_rms),
        "positive_task_edge_fraction": float(
            (task_score[mask] > 0.0).float().mean()
        ),
        "maximum_abs_angle": float(angles.abs().max()),
        "mean_abs_angle": float(angles.abs().mean()),
    }


@torch.no_grad()
def minimax_directed_output_pass(
    source: torch.Tensor,
    residual: torch.Tensor,
    activations: torch.Tensor,
    first_gradient: torch.Tensor,
    second_gradient: torch.Tensor,
    *,
    incoming: int,
    ridge_ratio: float,
    trust_output_energy: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Fit the preregistered global directed map from online-only state."""
    if source.ndim != 2 or residual.shape != source.shape:
        raise ValueError("source and residual must be same-shaped matrices")
    if activations.ndim != 2 or activations.shape[1] != source.shape[0]:
        raise ValueError("activation/source shapes disagree")
    if first_gradient.shape != source.T.shape or second_gradient.shape != source.T.shape:
        raise ValueError("task-gradient/source shapes disagree")
    outputs = int(source.shape[1])
    if not 0 < incoming <= outputs:
        raise ValueError("incoming support size is invalid")
    if not math.isfinite(ridge_ratio) or not 0.0 < ridge_ratio < 1.0:
        raise ValueError("ridge ratio must be finite and in (0, 1)")
    h = activations.to(source.device, dtype=torch.float32)
    source_f = source.float()
    residual_f = residual.float()
    design = h @ source_f
    target = h @ residual_f
    energy = design.square().sum(dim=0)
    ridge = float(ridge_ratio) * float(energy.mean().clamp_min(1e-30))
    cross = design.T @ target
    beta = cross / (energy[:, None] + ridge)
    residual_score = 2.0 * beta * cross - beta.square() * energy[:, None]
    first = first_gradient.float()
    second = second_gradient.float()
    first = first / first.norm().clamp_min(1e-30)
    second = second / second.norm().clamp_min(1e-30)
    first_task = -beta * (first @ source_f).T
    second_task = -beta * (second @ source_f).T
    residual_rms = residual_score.square().mean().sqrt().clamp_min(1e-30)
    first_rms = first_task.square().mean().sqrt().clamp_min(1e-30)
    second_rms = second_task.square().mean().sqrt().clamp_min(1e-30)
    agreement = torch.minimum(first_task / first_rms, second_task / second_rms)
    score = residual_score / residual_rms + agreement
    supports = torch.topk(
        score, k=incoming, dim=0, largest=True, sorted=True
    ).indices
    selected_design = design[:, supports.T].permute(1, 0, 2).contiguous()
    target_by_output = target.T.unsqueeze(-1)
    gram = selected_design.transpose(1, 2).double() @ selected_design.double()
    rhs = selected_design.transpose(1, 2).double() @ target_by_output.double()
    joint_ridge = (
        float(ridge_ratio)
        * torch.diagonal(gram, dim1=-2, dim2=-1)
        .mean(dim=1)
        .clamp_min(1e-30)
    )
    eye = torch.eye(
        incoming, device=gram.device, dtype=gram.dtype
    ).unsqueeze(0)
    coefficients = torch.linalg.solve(
        gram + joint_ridge[:, None, None] * eye, rhs
    ).squeeze(-1).float()
    mapping = torch.zeros(
        outputs, outputs, device=source.device, dtype=torch.float32
    )
    targets = torch.arange(outputs, device=source.device).unsqueeze(1).expand(
        -1, incoming
    )
    mapping[supports.T, targets] = coefficients
    raw_delta = source_f @ mapping
    raw_energy = float(raw_delta.double().square().sum())
    trust_scale = min(
        1.0,
        math.sqrt(float(trust_output_energy) / max(raw_energy, 1e-30)),
    )
    delta = raw_delta * trust_scale
    bounded_energy = float(delta.double().square().sum())
    both_positive = (first_task > 0.0) & (second_task > 0.0)
    return source_f + delta, {
        "coordinates": int(outputs * incoming),
        "incoming_per_target": int(incoming),
        "single_edge_ridge": ridge,
        "minimum_joint_ridge": float(joint_ridge.min()),
        "maximum_joint_ridge": float(joint_ridge.max()),
        "positive_task_agreement_fraction": float(
            both_positive.float().mean()
        ),
        "raw_output_delta_energy": raw_energy,
        "bounded_output_delta_energy": bounded_energy,
        "trust_output_energy": float(trust_output_energy),
        "trust_scale": trust_scale,
        "trust_energy_obeyed": (
            bounded_energy
            <= trust_output_energy + max(1e-12, 1e-5 * trust_output_energy)
        ),
    }


@torch.no_grad()
def hybrid_task_directed_output_update(
    source: torch.Tensor,
    residual: torch.Tensor,
    activations: torch.Tensor,
    current_gradient: torch.Tensor,
    momentum_combined_gradient: torch.Tensor,
    *,
    task_stages: int,
    directed_incoming: int,
    control_stages: int,
    neighbors: int,
    ridge_ratio: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Apply task-Givens then directed-minimax under an output32 trust cap."""
    control_permutations, control_matching = fast_muon_matched_permutations(
        source,
        residual,
        stages=control_stages,
        neighbors=neighbors,
        seed=seed,
    )
    control_permutations = control_permutations.to(source.device)
    control_angles = diagonal_metric_angles(
        source, residual, control_permutations
    )
    control = apply_givens_flow(
        source,
        control_angles,
        control_permutations,
        torch.argsort(control_permutations, dim=1),
    )
    trust_energy = float((control - source).double().square().sum())
    after_task, task_permutations, task_angles, task_diagnostics = (
        task_gradient_output_pass(
            source,
            residual,
            current_gradient.T.contiguous(),
            stages=task_stages,
            neighbors=neighbors,
            seed=seed,
        )
    )
    remaining = residual.float() - (after_task.float() - source.float())
    after_directed, directed_diagnostics = minimax_directed_output_pass(
        after_task,
        remaining,
        activations,
        current_gradient,
        momentum_combined_gradient,
        incoming=directed_incoming,
        ridge_ratio=ridge_ratio,
        trust_output_energy=trust_energy,
    )
    raw_delta = after_directed.float() - source.float()
    raw_energy = float(raw_delta.double().square().sum())
    combined_scale = min(
        1.0, math.sqrt(trust_energy / max(raw_energy, 1e-30))
    )
    bounded_delta = raw_delta * combined_scale
    bounded_energy = float(bounded_delta.double().square().sum())
    updated = source.float() + bounded_delta
    return updated, task_permutations, task_angles, {
        "coordinates": int(task_diagnostics["coordinates"])
        + int(directed_diagnostics["coordinates"]),
        "control_stages": int(control_stages),
        "control_matching": control_matching,
        "control_maximum_abs_angle": float(control_angles.abs().max()),
        "task": task_diagnostics,
        "directed": directed_diagnostics,
        "raw_combined_output_delta_energy": raw_energy,
        "bounded_combined_output_delta_energy": bounded_energy,
        "combined_trust_output_energy": trust_energy,
        "combined_trust_scale": combined_scale,
        "combined_trust_energy_obeyed": (
            bounded_energy <= trust_energy + max(1e-12, 1e-5 * trust_energy)
        ),
    }


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
    *,
    max_condition_number: float | None = None,
    fit_diagnostics: dict[str, float | bool] | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Fit the linearized post-GELU metric inside a bounded map chart.

    The provisional functional recipe is recursive: every fitted shear is
    applied before the next coordinate is estimated.  Bounding only the
    eventual weight update is therefore too late to prevent this internal
    recurrence from overflowing.  When a condition limit is configured, use
    the same composed log-condition budget while constructing the recipe.
    """
    current = source.float().clone()
    inputs = inputs.to(device=source.device, dtype=torch.float32)
    pre_gelu = pre_gelu.to(device=source.device, dtype=torch.float32)
    cproj = cproj_weight.to(device=source.device, dtype=torch.float32)
    maximum_log_condition = math.inf
    if max_condition_number is not None:
        if (
            not math.isfinite(float(max_condition_number))
            or float(max_condition_number) <= 1.0
        ):
            raise ValueError("max condition number must be finite and > 1")
        maximum_log_condition = math.log(float(max_condition_number))
    context_finite = bool(
        torch.isfinite(current).all()
        and torch.isfinite(requested_update).all()
        and torch.isfinite(inputs).all()
        and torch.isfinite(pre_gelu).all()
        and torch.isfinite(cproj).all()
    )
    if not context_finite:
        recipe = [
            (
                permutation.to(source.device).reshape(-1, 2),
                torch.full(
                    (permutation.numel() // 2,),
                    float("nan"),
                    dtype=torch.float64,
                    device=source.device,
                ),
            )
            for permutation in permutations
        ]
        if fit_diagnostics is not None:
            fit_diagnostics.update(
                {
                    "functional_fit_context_finite": False,
                    "functional_fit_coordinate_finite_fraction": 0.0,
                    "functional_fit_condition_projection_active": False,
                    "functional_fit_condition_projection_min_scale": 1.0,
                    "functional_fit_log_condition_bound": 0.0,
                }
            )
        return recipe
    slopes = _gelu_derivative(pre_gelu)
    projected = inputs @ current
    target_output = (
        slopes * (inputs @ requested_update.float())
    ) @ cproj.T
    residual_output = target_output.clone()
    cproj_gram = cproj.T @ cproj
    cproj_norm = cproj_gram.diagonal()
    recipe: list[tuple[torch.Tensor, torch.Tensor]] = []
    used_log_condition = 0.0
    finite_coordinates = 0
    total_coordinates = 0
    condition_projection_active = False
    minimum_condition_scale = 1.0
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
        coordinate_is_finite = torch.isfinite(coordinates)
        finite_coordinates += int(coordinate_is_finite.sum())
        total_coordinates += int(coordinates.numel())
        if not bool(coordinate_is_finite.all()):
            recipe.append((pairs, coordinates))
            continue
        stage_log_condition = 2.0 * float(coordinates.abs().max())
        remaining_log_condition = max(
            maximum_log_condition - used_log_condition,
            0.0,
        )
        condition_scale = 1.0
        if stage_log_condition > remaining_log_condition:
            condition_scale = (
                remaining_log_condition / stage_log_condition
                if stage_log_condition > 0.0
                else 1.0
            )
            coordinates = coordinates * condition_scale
            stage_log_condition *= condition_scale
            condition_projection_active = True
            minimum_condition_scale = min(
                minimum_condition_scale,
                condition_scale,
            )
        used_log_condition += stage_log_condition
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
    if fit_diagnostics is not None:
        fit_diagnostics.update(
            {
                "functional_fit_context_finite": context_finite,
                "functional_fit_coordinate_finite_fraction": (
                    finite_coordinates / max(total_coordinates, 1)
                ),
                "functional_fit_condition_projection_active": (
                    condition_projection_active
                ),
                "functional_fit_condition_projection_min_scale": (
                    minimum_condition_scale
                ),
                "functional_fit_log_condition_bound": used_log_condition,
            }
        )
    return recipe


def mix_shear_recipes(
    weight_recipe: list[tuple[torch.Tensor, torch.Tensor]],
    functional_recipe: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    beta: float,
    project_to_weight_norm: bool,
    max_condition_number: float | None = None,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], dict[str, float | bool]]:
    """Mix recipe directions, optionally retaining weight-fit L2 magnitude.

    The projection is global across every registered stage/pair coordinate.
    It therefore preserves the selected mixed direction exactly and changes
    only magnitude.  The weight-fit recipe supplies a parameter-free,
    optimizer-step-local radius.
    """
    if not 0.0 <= float(beta) <= 1.0:
        raise ValueError("beta must be in [0, 1]")
    if len(weight_recipe) != len(functional_recipe):
        raise ValueError("recipe lengths differ")
    mixed: list[tuple[torch.Tensor, torch.Tensor]] = []
    weight_energy = 0.0
    mixed_energy = 0.0
    weight_finite = 0
    functional_finite = 0
    total_coordinates = 0
    functional_fallback = False
    for (weight_pairs, weight_coordinates), (
        functional_pairs,
        functional_coordinates,
    ) in zip(weight_recipe, functional_recipe, strict=True):
        if not torch.equal(weight_pairs, functional_pairs):
            raise RuntimeError("functional and weight shear topology differs")
        weight_finite += int(torch.isfinite(weight_coordinates).sum())
        functional_finite += int(
            torch.isfinite(functional_coordinates).sum()
        )
        total_coordinates += int(weight_coordinates.numel())
    if weight_finite != total_coordinates:
        raise FloatingPointError("weight shear recipe is nonfinite")
    functional_fallback = functional_finite != total_coordinates
    for (weight_pairs, weight_coordinates), (
        functional_pairs,
        functional_coordinates,
    ) in zip(weight_recipe, functional_recipe, strict=True):
        if functional_fallback:
            coordinates = weight_coordinates
        else:
            coordinates = (
                (1.0 - float(beta)) * weight_coordinates
                + float(beta) * functional_coordinates
            )
        mixed.append((weight_pairs, coordinates))
        weight_energy += float(weight_coordinates.double().square().sum())
        mixed_energy += float(coordinates.double().square().sum())
    weight_norm = math.sqrt(weight_energy)
    mixed_norm_before = math.sqrt(mixed_energy)
    mixed_log_condition_before = sum(
        2.0 * float(coordinates.abs().max())
        for _pairs, coordinates in mixed
    )
    coordinate_scale = 1.0
    if (
        project_to_weight_norm
        and mixed_norm_before > weight_norm
        and mixed_norm_before > 0.0
    ):
        coordinate_scale = weight_norm / mixed_norm_before
    condition_scale = 1.0
    if max_condition_number is not None:
        if (
            not math.isfinite(float(max_condition_number))
            or float(max_condition_number) <= 1.0
        ):
            raise ValueError("max condition number must be finite and > 1")
        maximum_log_condition = math.log(float(max_condition_number))
        if mixed_log_condition_before > maximum_log_condition:
            condition_scale = (
                maximum_log_condition / mixed_log_condition_before
            )
            coordinate_scale = min(coordinate_scale, condition_scale)
    if coordinate_scale < 1.0:
        mixed = [
            (pairs, coordinates * coordinate_scale)
            for pairs, coordinates in mixed
        ]
    return mixed, {
        "weight_coordinate_l2": weight_norm,
        "mixed_coordinate_l2_before_projection": mixed_norm_before,
        "mixed_coordinate_l2_after_projection": (
            mixed_norm_before * coordinate_scale
        ),
        "coordinate_norm_projection_scale": coordinate_scale,
        "coordinate_norm_projection_active": bool(
            project_to_weight_norm
            and mixed_norm_before > weight_norm
        ),
        "mixed_log_condition_bound_before_projection": (
            mixed_log_condition_before
        ),
        "mixed_log_condition_bound_after_projection": (
            mixed_log_condition_before * coordinate_scale
        ),
        "maximum_condition_number": (
            0.0
            if max_condition_number is None
            else float(max_condition_number)
        ),
        "condition_projection_scale": condition_scale,
        "condition_projection_active": bool(condition_scale < 1.0),
        "weight_recipe_finite_fraction": (
            weight_finite / max(total_coordinates, 1)
        ),
        "functional_recipe_finite_fraction": (
            functional_finite / max(total_coordinates, 1)
        ),
        "functional_fallback_to_weight_recipe": functional_fallback,
    }


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
    project_to_weight_norm: bool,
    max_condition_number: float | None,
    learning_rate: float,
    weight_decay: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Reproduce the promoted fixed-topology mixed-coordinate ``c_fc`` step."""
    weight_before = weight.float()
    source = weight_before.T.contiguous()
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
    functional_fit_diagnostics: dict[str, float | bool] = {}
    functional_recipe = _fit_functional_shear_recipe(
        after_parent,
        residual,
        inputs,
        pre_gelu,
        cproj_weight,
        shear_permutations,
        max_condition_number=max_condition_number,
        fit_diagnostics=functional_fit_diagnostics,
    )
    mixed_recipe, projection_diagnostics = mix_shear_recipes(
        weight_recipe,
        functional_recipe,
        beta=beta,
        project_to_weight_norm=project_to_weight_norm,
        max_condition_number=max_condition_number,
    )
    current = after_parent
    mixed_coordinates: list[torch.Tensor] = []
    for weight_pairs, coordinates in mixed_recipe:
        current = _apply_symmetric_shear_stage(
            current, weight_pairs, coordinates
        )
        mixed_coordinates.append(coordinates)
    current.mul_(1.0 - float(learning_rate) * float(weight_decay))
    weight_after = current.T.contiguous()
    update = weight_after - weight_before
    requested_energy = requested_update.float().square().sum().clamp_min(1e-30)
    residual_energy = (
        requested_update.float() - update
    ).square().sum()
    all_shears = torch.cat(mixed_coordinates)
    finite_coordinates = torch.isfinite(all_shears)
    finite_update = torch.isfinite(update)
    source_rms = weight_before.square().mean().sqrt()
    result_rms = weight_after.square().mean().sqrt()
    # Each exp([[0,s],[s,0]]) has singular values exp(+/-s).  The sum of
    # per-stage maxima is a conservative log-condition-growth bound for the
    # composed shear map, even when pairings change between stages.
    shear_log_condition_bound = sum(
        float(coordinates.abs().max()) * 2.0
        for coordinates in mixed_coordinates
    )
    return update, {
        "coordinates": int(
            (parent_stages + shear_stages) * source.shape[1] // 2
        ),
        "parent_stages": int(parent_stages),
        "shear_stages": int(shear_stages),
        "functional_coordinate_mix_beta": float(beta),
        **projection_diagnostics,
        **functional_fit_diagnostics,
        "functional_samples": int(inputs.shape[0]),
        "angle_rms": float(parent_angles.square().mean().sqrt()),
        "shear_rms": float(all_shears.square().mean().sqrt()),
        "shear_max_abs": float(all_shears.abs().max()),
        "shear_log_condition_bound": shear_log_condition_bound,
        "coordinate_finite_fraction": float(finite_coordinates.float().mean()),
        "update_finite_fraction": float(finite_update.float().mean()),
        "weight_rms_before": float(source_rms),
        "weight_rms_after": float(result_rms),
        "weight_rms_ratio": float(result_rms / source_rms.clamp_min(1e-30)),
        "weight_max_abs_before": float(weight_before.abs().max()),
        "weight_max_abs_after": float(weight_after.abs().max()),
        "update_rms": float(update.square().mean().sqrt()),
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
        output_stages: int = 0,
        neighbors: int,
        refresh_interval: int,
        fast_fresh_matching: bool,
        matching_seed: int,
        weight_std: float,
        layer_id: int = -1,
        hybrid_output: bool = False,
        hybrid_directed_incoming: int = 0,
        hybrid_control_output_stages: int = 32,
        hybrid_ridge_ratio: float = 1e-6,
        hybrid_functional_sample_cap: int = 2048,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.stages = int(stages)
        self.residual_stages = int(residual_stages)
        self.output_stages = int(output_stages)
        self.neighbors = int(neighbors)
        self.refresh_interval = int(refresh_interval)
        self.fast_fresh_matching = bool(fast_fresh_matching)
        self.matching_seed = int(matching_seed)
        self.layer_id = int(layer_id)
        self.hybrid_output = bool(hybrid_output)
        self.hybrid_directed_incoming = int(hybrid_directed_incoming)
        self.hybrid_control_output_stages = int(
            hybrid_control_output_stages
        )
        self.hybrid_ridge_ratio = float(hybrid_ridge_ratio)
        self.hybrid_functional_sample_cap = int(
            hybrid_functional_sample_cap
        )
        if (
            self.in_features <= 0
            or self.in_features % 2
            or self.out_features <= 0
            or (self.output_stages and self.out_features % 2)
        ):
            raise ValueError(
                "MuonMatchedGivensLinear requires positive dimensions "
                "and even widths for every enabled rotation side"
            )
        if (
            self.stages <= 0
            or self.residual_stages < 0
            or self.output_stages < 0
            or self.neighbors < max(
                self.stages, self.residual_stages, self.output_stages
            )
            or self.neighbors >= self.in_features
            or (
                self.output_stages
                and self.neighbors >= self.out_features
            )
        ):
            raise ValueError(
                "require 0 < stages, 0 <= residual/output stages, "
                "and max(stages, residual_stages, output_stages) <= "
                "neighbors < in_features (and < out_features when "
                "output stages are enabled)"
            )
        if self.refresh_interval <= 0:
            raise ValueError("refresh_interval must be positive")
        if self.fast_fresh_matching and self.refresh_interval != 1:
            raise ValueError(
                "fast fresh matching requires refresh_interval=1"
            )
        if (
            self.residual_stages or self.output_stages
        ) and not self.fast_fresh_matching:
            raise ValueError(
                "residual/output matching requires fast fresh matching"
            )
        if not math.isfinite(weight_std) or weight_std <= 0.0:
            raise ValueError("weight_std must be positive and finite")
        if self.hybrid_output:
            if (
                not self.fast_fresh_matching
                or self.output_stages <= 0
                or self.hybrid_directed_incoming <= 0
                or self.hybrid_directed_incoming > self.out_features
                or self.hybrid_control_output_stages <= 0
                or self.hybrid_control_output_stages > self.neighbors
                or not math.isfinite(self.hybrid_ridge_ratio)
                or not 0.0 < self.hybrid_ridge_ratio < 1.0
                or self.hybrid_functional_sample_cap <= 0
            ):
                raise ValueError("invalid hybrid-output c_proj configuration")
        elif self.hybrid_directed_incoming != 0:
            raise ValueError(
                "hybrid directed coordinates require hybrid output"
            )

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
        if self.output_stages:
            output_initial = torch.arange(self.out_features).repeat(
                self.output_stages, 1
            )
            self.register_buffer(
                "output_selected_permutations",
                output_initial,
                persistent=True,
            )
            self.register_buffer(
                "output_selected_inverse_permutations",
                torch.argsort(output_initial, dim=1),
                persistent=True,
            )
            self.register_buffer(
                "output_last_angles",
                torch.zeros(
                    self.output_stages, self.out_features // 2
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
        self._hybrid_output_inputs: torch.Tensor | None = None

    @property
    def coordinate_count(self) -> int:
        return (
            (self.stages + self.residual_stages)
            * (self.in_features // 2)
            + self.output_stages * (self.out_features // 2)
            + (
                self.hybrid_directed_incoming * self.out_features
                if self.hybrid_output
                else 0
            )
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.linear(values, self.weight, self.bias)

    @torch.no_grad()
    def record_hybrid_output_context(self, values: torch.Tensor) -> None:
        if (
            not self.hybrid_output
            or not self.training
            or self._hybrid_output_inputs is not None
        ):
            return
        if values.shape[-1] != self.in_features:
            raise ValueError("hybrid c_proj activation width mismatch")
        flat = values.detach().reshape(-1, self.in_features)
        count = min(self.hybrid_functional_sample_cap, flat.shape[0])
        self._hybrid_output_inputs = flat[:count].contiguous()

    def consume_hybrid_output_context(self) -> torch.Tensor:
        values = self._hybrid_output_inputs
        self._hybrid_output_inputs = None
        if values is None:
            raise RuntimeError(
                f"missing hybrid c_proj context for layer {self.layer_id}"
            )
        return values

    def clear_hybrid_output_context(self) -> None:
        self._hybrid_output_inputs = None


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
        project_to_weight_norm: bool,
        max_condition_number: float | None,
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
        self.project_to_weight_norm = bool(project_to_weight_norm)
        self.max_condition_number = (
            None
            if max_condition_number is None
            else float(max_condition_number)
        )
        self.functional_sample_cap = int(functional_sample_cap)
        self.matching_seed = int(matching_seed)
        self.layer_id = int(layer_id)
        if (
            self.in_features <= 0
            or self.out_features <= 0
            or self.out_features % 2
            or self.parent_stages <= 0
            or self.shear_stages <= 0
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
                    project_to_weight_norm=(
                        module.project_to_weight_norm
                    ),
                    max_condition_number=(
                        module.max_condition_number
                    ),
                    learning_rate=lr,
                    weight_decay=weight_decay,
                )
                weight.add_(update.to(dtype=weight.dtype))
                row.update(
                    {
                        "step": int(module.optimizer_step),
                        "layer": module.layer_id,
                        "report_refresh": (
                            int(module.optimizer_step) == 0
                            or int(module.optimizer_step)
                            < int(
                                os.environ.get(
                                    "MUON_FUNCTIONAL_SHEAR_DIAGNOSTIC_STEPS",
                                    "0",
                                )
                            )
                        ),
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


@torch.no_grad()
def batched_multistage_directed_sparse_update(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    incoming_schedule: tuple[int, ...] | list[int],
    ridge_ratio: float,
    chunk_size: int,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Fit a batched product of directed sparse output mixers.

    ``source`` and ``target`` are ``[layers, rows, output_channels]``.
    Each stage sparsifies the exact minimum-norm output action, jointly
    refits the selected incoming columns per target channel, and fits the
    next stage against the residual from the already transformed source.
    This is the production-batched form of the preregistered 30+29+29
    midpoint discriminator; no learned or persistent dense basis is used.
    """
    if source.ndim != 3 or source.shape != target.shape:
        raise ValueError("source and target must be same-shaped rank-3 tensors")
    batch, rows, width = source.shape
    schedule = tuple(int(value) for value in incoming_schedule)
    if (
        batch <= 0
        or rows <= 0
        or width <= 0
        or len(schedule) < 2
        or any(value <= 0 or value > min(rows, width) for value in schedule)
    ):
        raise ValueError("invalid directed-product dimensions or schedule")
    if not 0.0 < float(ridge_ratio) < 1.0 or int(chunk_size) <= 0:
        raise ValueError("invalid directed-product solver settings")

    source_f = source.float()
    target_f = target.float()
    transformed = source_f.clone()
    prediction = torch.zeros_like(target_f)
    stage_rows: list[dict[str, Any]] = []
    for stage_index, incoming in enumerate(schedule):
        remaining = target_f - prediction
        row_gram = transformed @ transformed.transpose(1, 2)
        row_scale = row_gram.diagonal(dim1=1, dim2=2).mean(dim=1)
        row_gram.diagonal(dim1=1, dim2=2).add_(
            float(ridge_ratio) * row_scale[:, None]
        )
        minimum_norm_action = transformed.transpose(1, 2) @ torch.linalg.solve(
            row_gram, remaining
        )
        indices = torch.topk(
            minimum_norm_action.abs(), k=incoming, dim=1
        ).indices
        del row_gram, minimum_norm_action

        stage_update = torch.empty_like(remaining)
        eye = torch.eye(
            incoming, device=source.device, dtype=torch.float32
        )[None, None]
        for start in range(0, width, int(chunk_size)):
            stop = min(start + int(chunk_size), width)
            columns = stop - start
            selected = indices[:, :, start:stop]
            dictionary = torch.gather(
                transformed.unsqueeze(3).expand(-1, -1, -1, columns),
                2,
                selected[:, None].expand(-1, rows, -1, -1),
            ).permute(0, 3, 1, 2).contiguous()
            targets = (
                remaining[:, :, start:stop]
                .permute(0, 2, 1)
                .contiguous()
                .unsqueeze(-1)
            )
            gram = dictionary.transpose(-1, -2) @ dictionary
            rhs = dictionary.transpose(-1, -2) @ targets
            diagonal_mean = gram.diagonal(
                dim1=-2, dim2=-1
            ).mean(dim=-1)
            gram.add_(
                eye
                * (float(ridge_ratio) * diagonal_mean)[..., None, None]
            )
            coefficients = torch.linalg.solve(gram, rhs)
            stage_update[:, :, start:stop] = (
                (dictionary @ coefficients)
                .squeeze(-1)
                .permute(0, 2, 1)
            )

        transformed.add_(stage_update)
        prediction.add_(stage_update)
        remaining_after = target_f - prediction
        target_energy = target_f.square().sum(dim=(1, 2)).clamp_min(1e-30)
        prediction_norm = prediction.square().sum(dim=(1, 2)).sqrt()
        target_norm = target_energy.sqrt()
        stage_rows.append(
            {
                "stage_index": int(stage_index),
                "incoming_per_target": int(incoming),
                "coordinates_per_member": int(incoming * width),
                "member_target_recovery": (
                    1.0
                    - remaining_after.square().sum(dim=(1, 2))
                    / target_energy
                ).cpu().tolist(),
                "member_target_cosine": (
                    (prediction * target_f).sum(dim=(1, 2))
                    / (prediction_norm * target_norm).clamp_min(1e-30)
                ).cpu().tolist(),
            }
        )
    return prediction, stage_rows


class MuonDirectedProductLinear(nn.Module):
    """Materialized c_fc updated through a task-selected sparse product."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        incoming_schedule: tuple[int, ...] | list[int],
        ridge_ratio: float,
        chunk_size: int,
        family_radius_ratio: float,
        error_feedback: bool,
        error_feedback_decay: float,
        weight_std: float,
        layer_id: int,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.incoming_schedule = tuple(
            int(value) for value in incoming_schedule
        )
        self.ridge_ratio = float(ridge_ratio)
        self.chunk_size = int(chunk_size)
        self.family_radius_ratio = float(family_radius_ratio)
        self.error_feedback = bool(error_feedback)
        self.error_feedback_decay = float(error_feedback_decay)
        self.weight_std = float(weight_std)
        self.layer_id = int(layer_id)
        if (
            self.in_features <= 0
            or self.out_features <= 0
            or len(self.incoming_schedule) < 2
            or any(
                value <= 0
                or value > min(self.in_features, self.out_features)
                for value in self.incoming_schedule
            )
            or not 0.0 < self.ridge_ratio < 1.0
            or self.chunk_size <= 0
            or not math.isfinite(self.family_radius_ratio)
            or self.family_radius_ratio <= 0.0
            or not math.isfinite(self.error_feedback_decay)
            or not 0.0 <= self.error_feedback_decay <= 1.0
            or not math.isfinite(self.weight_std)
            or self.weight_std <= 0.0
        ):
            raise ValueError("invalid directed-product c_fc configuration")
        weight = torch.empty(self.out_features, self.in_features)
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

    @property
    def coordinate_count(self) -> int:
        return sum(self.incoming_schedule) * self.out_features

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.linear(values, self.weight, self.bias)


class MuonDirectedProduct(torch.optim.Optimizer):
    """Exact Muon momentum followed by the selected batched sparse product."""

    def __init__(
        self,
        modules: list[MuonDirectedProductLinear],
        *,
        lr: float,
        momentum: float,
        weight_decay: float,
        ns_steps: int,
    ) -> None:
        if not modules:
            raise ValueError("MuonDirectedProduct requires at least one module")
        reference = modules[0]
        if any(
            module.in_features != reference.in_features
            or module.out_features != reference.out_features
            or module.incoming_schedule != reference.incoming_schedule
            or module.ridge_ratio != reference.ridge_ratio
            or module.chunk_size != reference.chunk_size
            or module.family_radius_ratio != reference.family_radius_ratio
            or module.error_feedback != reference.error_feedback
            or module.error_feedback_decay != reference.error_feedback_decay
            for module in modules
        ):
            raise ValueError("directed-product modules must share one geometry")
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
            active_weights = [
                weight
                for weight in group["params"]
                if weight.grad is not None
            ]
            if not active_weights:
                continue
            requested: list[torch.Tensor] = []
            modules: list[MuonDirectedProductLinear] = []
            for weight in active_weights:
                module = self.modules_by_id[id(weight)]
                modules.append(module)
                state = self.state[weight]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(weight)
                buffer = state["momentum_buffer"]
                gradient = weight.grad
                assert gradient is not None
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
                requested.append(
                    lr * (direction - weight_decay * weight.float())
                )

            source = torch.stack(
                [weight.float().T for weight in active_weights], dim=0
            ).contiguous()
            requested_target = torch.stack(
                [update.T for update in requested], dim=0
            ).contiguous()
            reference = modules[0]
            if reference.error_feedback:
                feedback = []
                for weight in active_weights:
                    state = self.state[weight]
                    if "compression_residual" not in state:
                        state["compression_residual"] = torch.zeros_like(
                            weight, dtype=torch.float32
                        )
                    feedback.append(
                        state["compression_residual"].float().T
                    )
                feedback_target = torch.stack(feedback, dim=0).contiguous()
                target = requested_target + (
                    reference.error_feedback_decay * feedback_target
                )
            else:
                feedback_target = torch.zeros_like(requested_target)
                target = requested_target
            raw_prediction, stage_rows = (
                batched_multistage_directed_sparse_update(
                    source,
                    target,
                    incoming_schedule=reference.incoming_schedule,
                    ridge_ratio=reference.ridge_ratio,
                    chunk_size=reference.chunk_size,
                )
            )
            raw_family_norm = raw_prediction.square().sum().sqrt()
            dense_family_norm = requested_target.square().sum().sqrt()
            corrected_family_norm = target.square().sum().sqrt()
            target_family_norm = (
                reference.family_radius_ratio * corrected_family_norm
            )
            family_scale = target_family_norm / raw_family_norm.clamp_min(
                1e-30
            )
            prediction = raw_prediction * family_scale
            compression_residual = target - prediction
            requested_residual = requested_target - prediction
            for index, (weight, module) in enumerate(
                zip(active_weights, modules, strict=True)
            ):
                update = prediction[index].T.contiguous()
                weight.add_(update.to(dtype=weight.dtype))
                if module.error_feedback:
                    self.state[weight]["compression_residual"] = (
                        compression_residual[index].T.contiguous()
                    )
                target_energy = (
                    requested_target[index].square().sum().clamp_min(1e-30)
                )
                corrected_energy = target[index].square().sum().clamp_min(1e-30)
                diagnostics.append(
                    {
                        "optimizer": "muon_directed_product",
                        "step": int(module.optimizer_step),
                        "layer": module.layer_id,
                        "incoming_schedule": list(
                            module.incoming_schedule
                        ),
                        "coordinates": module.coordinate_count,
                        "ridge_ratio": module.ridge_ratio,
                        "solver_chunk_size": module.chunk_size,
                        "family_radius_ratio": module.family_radius_ratio,
                        "error_feedback": module.error_feedback,
                        "error_feedback_decay": module.error_feedback_decay,
                        "family_scale": float(family_scale),
                        "dense_family_fro": float(dense_family_norm),
                        "corrected_family_fro": float(
                            corrected_family_norm
                        ),
                        "raw_prediction_family_fro": float(
                            raw_family_norm
                        ),
                        "target_family_fro": float(target_family_norm),
                        "update_fro": float(update.float().norm()),
                        "feedback_input_fro": float(
                            feedback_target[index].norm()
                        ),
                        "feedback_output_fro": float(
                            compression_residual[index].norm()
                        ),
                        "requested_update_recovery": float(
                            1.0 - requested_residual[index].square().sum()
                            / target_energy
                        ),
                        "requested_update_cosine": float(
                            (
                                prediction[index]
                                * requested_target[index]
                            ).sum()
                            / (
                                prediction[index].norm()
                                * requested_target[index].norm()
                            ).clamp_min(1e-30)
                        ),
                        "corrected_target_recovery": float(
                            1.0
                            - compression_residual[index].square().sum()
                            / corrected_energy
                        ),
                        "stage_rows": [
                            {
                                **{
                                    key: value
                                    for key, value in row.items()
                                    if not key.startswith("member_")
                                },
                                "target_recovery": row[
                                    "member_target_recovery"
                                ][index],
                                "target_cosine": row[
                                    "member_target_cosine"
                                ][index],
                            }
                            for row in stage_rows
                        ],
                    }
                )
                module.optimizer_step.add_(1)
        self.last_step_diagnostics = diagnostics
        return loss

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
        error_feedback: bool = False,
        error_feedback_decay: float = 1.0,
    ) -> None:
        if not modules:
            raise ValueError("MuonMatchedGivens requires at least one module")
        if (
            not math.isfinite(error_feedback_decay)
            or not 0.0 <= error_feedback_decay <= 1.0
        ):
            raise ValueError(
                "MuonMatchedGivens error-feedback decay must be in [0, 1]"
            )
        for module in modules:
            module.weight.requires_grad_(True)
        self.modules_by_id = {id(module.weight): module for module in modules}
        self.last_step_diagnostics: list[dict[str, Any]] = []
        defaults = {
            "lr": float(lr),
            "momentum": float(momentum),
            "weight_decay": float(weight_decay),
            "ns_steps": int(ns_steps),
            "error_feedback": bool(error_feedback),
            "error_feedback_decay": float(error_feedback_decay),
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
            error_feedback = bool(group.get("error_feedback", False))
            error_feedback_decay = float(
                group.get("error_feedback_decay", 1.0)
            )
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
                requested_update = lr * (
                    direction - weight_decay * weight.float()
                )
                if error_feedback:
                    if "compression_residual" not in state:
                        state["compression_residual"] = torch.zeros_like(
                            weight, dtype=torch.float32
                        )
                    feedback_input = state["compression_residual"].float()
                    corrected_update = requested_update + (
                        error_feedback_decay * feedback_input
                    )
                    matching_direction = direction
                    if lr != 0.0:
                        matching_direction = matching_direction + (
                            error_feedback_decay * feedback_input / lr
                        )
                else:
                    feedback_input = torch.zeros_like(
                        requested_update, dtype=torch.float32
                    )
                    corrected_update = requested_update
                    matching_direction = direction
                step_index = int(module.optimizer_step)
                refresh = (
                    module.fast_fresh_matching
                    or not bool(module.matching_valid)
                    or step_index % module.refresh_interval == 0
                )
                matching_summary = None
                residual_matching_summary = None
                output_matching_summary = None
                if refresh:
                    if module.fast_fresh_matching:
                        permutations, matching_diagnostics = (
                            fast_muon_matched_permutations(
                                weight,
                                matching_direction,
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
                                matching_direction,
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
                parent_angles = diagonal_metric_angles(
                    weight,
                    corrected_update,
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
                    corrected_update.float() - parent_update
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
                output_angles = None
                if module.hybrid_output:
                    output_residual_update = (
                        corrected_update.float()
                        - (rotated.float() - weight.float())
                    )
                    output_source = rotated.transpose(0, 1).contiguous()
                    output_direction = (
                        output_residual_update.transpose(0, 1).contiguous()
                    )
                    activations = module.consume_hybrid_output_context()
                    hybrid_output, output_permutations, output_angles, (
                        hybrid_diagnostics
                    ) = hybrid_task_directed_output_update(
                        output_source,
                        output_direction,
                        activations,
                        gradient.float(),
                        combined.float(),
                        task_stages=module.output_stages,
                        directed_incoming=(
                            module.hybrid_directed_incoming
                        ),
                        control_stages=(
                            module.hybrid_control_output_stages
                        ),
                        neighbors=module.neighbors,
                        ridge_ratio=module.hybrid_ridge_ratio,
                        seed=(
                            module.matching_seed + step_index + 2
                        ),
                    )
                    module.output_selected_permutations.copy_(
                        output_permutations.to(
                            device=weight.device, dtype=torch.long
                        )
                    )
                    module.output_selected_inverse_permutations.copy_(
                        torch.argsort(
                            module.output_selected_permutations, dim=1
                        )
                    )
                    rotated = hybrid_output.transpose(
                        0, 1
                    ).contiguous()
                    output_matching_summary = {
                        "selector": "causal_task16_then_minimax8",
                        **hybrid_diagnostics,
                    }
                if module.output_stages and not module.hybrid_output:
                    output_residual_update = (
                        corrected_update.float()
                        - (rotated.float() - weight.float())
                    )
                    output_source = rotated.transpose(0, 1).contiguous()
                    output_direction = (
                        output_residual_update.transpose(0, 1).contiguous()
                    )
                    output_permutations, output_diagnostics = (
                        fast_muon_matched_permutations(
                            output_source,
                            output_direction,
                            stages=module.output_stages,
                            neighbors=module.neighbors,
                            seed=(
                                module.matching_seed
                                + step_index
                                + 2
                            ),
                        )
                    )
                    module.output_selected_permutations.copy_(
                        output_permutations.to(
                            device=weight.device,
                            dtype=torch.long,
                        )
                    )
                    module.output_selected_inverse_permutations.copy_(
                        torch.argsort(
                            module.output_selected_permutations,
                            dim=1,
                        )
                    )
                    output_angles = diagonal_metric_angles(
                        output_source,
                        output_direction,
                        module.output_selected_permutations,
                    )
                    rotated = apply_givens_flow(
                        output_source,
                        output_angles,
                        module.output_selected_permutations,
                        module.output_selected_inverse_permutations,
                    ).transpose(0, 1).contiguous()
                    output_matching_summary = {
                        "selector": "fast_fresh_output_pass",
                        "candidate_edge_fraction": float(
                            output_diagnostics[
                                "candidate_edge_fraction"
                            ]
                        ),
                        "minimum_stage_candidate_edge_fraction": float(
                            output_diagnostics[
                                "minimum_stage_candidate_edge_fraction"
                            ]
                        ),
                        "prepared_seconds": float(
                            output_diagnostics["prepared_seconds"]
                        ),
                        "native_seconds": float(
                            output_diagnostics["native_seconds"]
                        ),
                        "total_seconds": float(
                            output_diagnostics["total_seconds"]
                        ),
                        "native_output_validated": bool(
                            output_diagnostics[
                                "native_output_validated"
                            ]
                        ),
                        "native_library_sha256": str(
                            output_diagnostics["native_library_sha256"]
                        ),
                        "native_source_sha256": str(
                            output_diagnostics["source_sha256"]
                        ),
                    }
                if weight_decay != 0.0:
                    rotated.mul_(1.0 - lr * weight_decay)
                update = rotated - weight
                requested_energy = requested_update.float().square().sum()
                corrected_energy = corrected_update.float().square().sum()
                requested_residual = requested_update.float() - update.float()
                compression_residual = (
                    corrected_update.float() - update.float()
                )
                requested_residual_energy = requested_residual.square().sum()
                corrected_residual_energy = compression_residual.square().sum()
                weight.copy_(rotated)
                if error_feedback:
                    state["compression_residual"] = (
                        compression_residual.contiguous()
                    )
                module.last_angles.copy_(
                    parent_angles.to(dtype=module.last_angles.dtype)
                )
                if residual_angles is not None:
                    module.residual_last_angles.copy_(
                        residual_angles.to(
                            dtype=module.residual_last_angles.dtype
                        )
                    )
                if output_angles is not None:
                    module.output_last_angles.copy_(
                        output_angles.to(
                            dtype=module.output_last_angles.dtype
                        )
                    )
                module.optimizer_step.add_(1)
                angle_parts = [parent_angles.reshape(-1)]
                if residual_angles is not None:
                    angle_parts.append(residual_angles.reshape(-1))
                if output_angles is not None:
                    angle_parts.append(output_angles.reshape(-1))
                all_angles = torch.cat(angle_parts)
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
                        "error_feedback": error_feedback,
                        "error_feedback_decay": error_feedback_decay,
                        "angle_rms": float(
                            all_angles.square().mean().sqrt()
                        ),
                        "angle_max_abs": float(
                            all_angles.abs().max()
                        ),
                        "parent_stages": module.stages,
                        "residual_stages": module.residual_stages,
                        "output_stages": module.output_stages,
                        "hybrid_output": module.hybrid_output,
                        "hybrid_directed_incoming": (
                            module.hybrid_directed_incoming
                        ),
                        "requested_update_recovery": float(
                            1.0
                            - requested_residual_energy
                            / requested_energy.clamp_min(1e-30)
                        ),
                        "corrected_target_recovery": float(
                            1.0
                            - corrected_residual_energy
                            / corrected_energy.clamp_min(1e-30)
                        ),
                        "feedback_input_fro": float(
                            feedback_input.norm()
                        ),
                        "feedback_output_fro": float(
                            compression_residual.norm()
                        ),
                        "matching": matching_summary,
                        "residual_matching": (
                            residual_matching_summary
                        ),
                        "output_matching": output_matching_summary,
                    }
                )
        self.last_step_diagnostics = diagnostics
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        super().zero_grad(set_to_none=set_to_none)
        for module in self.modules_by_id.values():
            module.clear_hybrid_output_context()

    def consume_diagnostics(self) -> list[dict[str, Any]]:
        diagnostics = self.last_step_diagnostics
        self.last_step_diagnostics = []
        return diagnostics
