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

from latent_weight_lab.block_fht import normalized_fht_last_dim
from examples.nanogpt.muon import muon_update
from examples.nanogpt.pair_vq_fp16_reserved_escape_cuda import (
    ReservedEscapeState,
    decode_reserved_escape,
    encode_reserved_escape,
)
from examples.nanogpt.pair_vq_hierarchical_lloyd_cuda import (
    hierarchical_lloyd_stats,
)


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


def _decode_cartesian_pair_codec(
    levels: torch.Tensor, codes: torch.Tensor
) -> torch.Tensor:
    if tuple(levels.shape) != (2, 16):
        raise ValueError("Cartesian pair levels must have shape (2, 16)")
    selected = codes.long()
    return torch.stack(
        (
            levels[0].index_select(0, selected // 16),
            levels[1].index_select(0, selected % 16),
        ),
        dim=1,
    )


@torch.no_grad()
def _fit_cartesian_pair_codec_(
    vectors: torch.Tensor,
    levels: torch.Tensor,
    codes: torch.Tensor,
) -> int:
    """Fit a 16x16 Cartesian pair dictionary without dense persistent state."""
    if vectors.ndim != 2 or vectors.shape[1] != 2:
        raise ValueError("Cartesian pair values must have shape (pairs, 2)")
    if tuple(levels.shape) != (2, 16):
        raise ValueError("Cartesian pair levels must have shape (2, 16)")
    if codes.ndim != 1 or codes.numel() != vectors.shape[0]:
        raise ValueError("Cartesian pair codes have the wrong shape")
    old_codes = codes.clone()
    assignments = []
    for coordinate in range(2):
        values = vectors[:, coordinate]
        mean = values.mean()
        std = values.std(unbiased=False)
        old_levels = levels[coordinate]
        old_std = old_levels.std(unbiased=False)
        if float(std) <= torch.finfo(torch.float32).tiny:
            fitted = torch.full_like(old_levels, mean)
        elif float(old_std) > torch.finfo(torch.float32).tiny:
            fitted = (old_levels - old_levels.mean()) / old_std
            fitted = (fitted * std + mean).sort().values
        else:
            probabilities = (
                torch.arange(16, device=values.device, dtype=torch.float32)
                + 0.5
            ) / 16.0
            fitted = (
                math.sqrt(2.0)
                * torch.erfinv(2.0 * probabilities - 1.0)
                * std
                + mean
            )
        for _iteration in range(2):
            midpoints = (fitted[:-1] + fitted[1:]) * 0.5
            indices = torch.bucketize(values.contiguous(), midpoints)
            sums = torch.zeros_like(fitted)
            sums.index_add_(0, indices, values)
            counts = torch.bincount(indices, minlength=16)
            live = counts > 0
            candidate = sums / counts.clamp_min(1).to(dtype=sums.dtype)
            fitted = torch.where(live, candidate, fitted)
            fitted = fitted.sort().values
        indices = torch.bucketize(
            values.contiguous(), (fitted[:-1] + fitted[1:]) * 0.5
        )
        levels[coordinate].copy_(fitted)
        assignments.append(indices)
    new_codes = (assignments[0] * 16 + assignments[1]).to(torch.uint8)
    codes.copy_(new_codes)
    return int((new_codes != old_codes).sum())


@torch.no_grad()
def _fit_stochastic_cartesian_pair_codec_(
    vectors: torch.Tensor,
    levels: torch.Tensor,
    codes: torch.Tensor,
    *,
    seed: int,
    uniform_levels: bool = False,
) -> tuple[int, dict[str, float]]:
    """Fit Cartesian levels, then round each coordinate without local bias."""
    if vectors.ndim != 2 or vectors.shape[1] != 2:
        raise ValueError("Cartesian pair values must have shape (pairs, 2)")
    old_codes = codes.clone()
    if not uniform_levels:
        _fit_cartesian_pair_codec_(vectors, levels, codes)
    generator = torch.Generator(device=vectors.device)
    generator.manual_seed(int(seed) % (2**63 - 1))
    assignments = []
    expected_parts = []
    variance_parts = []
    boundary_clipped_values = 0
    for coordinate in range(2):
        values = vectors[:, coordinate].float()
        if uniform_levels:
            support_min = values.amin()
            support_max = values.amax()
            fractions = torch.arange(
                16, device=values.device, dtype=torch.float32
            ).div_(15.0)
            ordered = support_min + fractions * (support_max - support_min)
        else:
            ordered = levels[coordinate].sort().values
        # Lloyd centroids sit strictly inside the observed range.  Leaving
        # those centroids as the outer stochastic levels silently clips every
        # tail request and defeats the promised unbiased local retraction.
        # Reuse the two existing boundary entries as exact support endpoints;
        # this changes no byte accounting and leaves the 14 interior Lloyd
        # levels untouched.
        ordered[0] = values.amin()
        ordered[-1] = values.amax()
        levels[coordinate].copy_(ordered)
        boundary_clipped_values += int(
            ((values < ordered[0]) | (values > ordered[-1])).sum()
        )
        upper = torch.searchsorted(ordered, values.contiguous()).clamp(0, 15)
        lower = (upper - 1).clamp(0, 15)
        below = values <= ordered[0]
        above = values >= ordered[-1]
        lower = torch.where(below, torch.zeros_like(lower), lower)
        upper = torch.where(below, torch.zeros_like(upper), upper)
        lower = torch.where(above, torch.full_like(lower, 15), lower)
        upper = torch.where(above, torch.full_like(upper, 15), upper)
        low_value = ordered.index_select(0, lower)
        high_value = ordered.index_select(0, upper)
        width = high_value - low_value
        probability = torch.where(
            width > 0,
            ((values - low_value) / width).clamp(0.0, 1.0),
            torch.zeros_like(values),
        )
        draw = torch.rand(
            values.shape,
            generator=generator,
            device=values.device,
            dtype=torch.float32,
        )
        assignments.append(torch.where(draw < probability, upper, lower))
        expected_parts.append(low_value + probability * width)
        variance_parts.append(probability * (1.0 - probability) * width.square())
    new_codes = (assignments[0] * 16 + assignments[1]).to(torch.uint8)
    codes.copy_(new_codes)
    expected = torch.stack(expected_parts, dim=1)
    expected_bias_energy = float((expected - vectors.float()).square().sum())
    target_energy = float(vectors.float().square().sum())
    sampling_variance = float(torch.stack(variance_parts, dim=1).sum())
    return int((new_codes != old_codes).sum()), {
        "stochastic_fast_expected_bias_energy": expected_bias_energy,
        "stochastic_fast_target_energy": target_energy,
        "stochastic_fast_expected_bias_recovery": (
            1.0 - expected_bias_energy / max(target_energy, 1e-30)
        ),
        "stochastic_fast_sampling_variance": sampling_variance,
        "stochastic_fast_sampling_variance_ratio": (
            sampling_variance / max(target_energy, 1e-30)
        ),
        "stochastic_fast_boundary_clipped_values": boundary_clipped_values,
        "stochastic_fast_uniform_levels": int(uniform_levels),
    }


@torch.no_grad()
def _fit_block_local_uniform_stochastic_cartesian_pair_codec_(
    vectors: torch.Tensor,
    bounds: torch.Tensor,
    codes: torch.Tensor,
    *,
    pairs_per_group: int,
    seed: int,
) -> tuple[int, dict[str, float]]:
    """Unbiased 16x16 coding with exact support bounds per FHT block."""
    if vectors.ndim != 2 or vectors.shape[1] != 2:
        raise ValueError("block-local Cartesian values must have shape (pairs, 2)")
    if pairs_per_group <= 0 or vectors.shape[0] % pairs_per_group:
        raise ValueError("pairs_per_group must divide the pair count")
    group_count = vectors.shape[0] // pairs_per_group
    if tuple(bounds.shape) != (group_count, 2, 2):
        raise ValueError("block-local bounds must have shape (groups, 2, 2)")
    if codes.ndim != 1 or codes.numel() != vectors.shape[0]:
        raise ValueError("block-local Cartesian codes have the wrong shape")

    grouped = vectors.float().reshape(group_count, pairs_per_group, 2)
    support_min = grouped.amin(dim=1)
    support_max = grouped.amax(dim=1)
    bounds[:, :, 0].copy_(support_min)
    bounds[:, :, 1].copy_(support_max)
    width = support_max - support_min
    step = width / 15.0
    normalized = torch.where(
        width[:, None, :] > 0,
        (grouped - support_min[:, None, :])
        / width[:, None, :]
        * 15.0,
        torch.zeros_like(grouped),
    ).clamp_(0.0, 15.0)
    lower = normalized.floor().to(torch.long).clamp_(0, 15)
    upper = (lower + 1).clamp_(0, 15)
    probability = (normalized - lower.float()).clamp_(0.0, 1.0)

    generator = torch.Generator(device=vectors.device)
    generator.manual_seed(int(seed) % (2**63 - 1))
    draw = torch.rand(
        probability.shape,
        generator=generator,
        device=vectors.device,
        dtype=torch.float32,
    )
    assignment = torch.where(draw < probability, upper, lower)
    new_codes = (
        assignment[:, :, 0] * 16 + assignment[:, :, 1]
    ).reshape(-1).to(torch.uint8)
    old_codes = codes.clone()
    codes.copy_(new_codes)

    low_value = support_min[:, None, :] + lower.float() * step[:, None, :]
    expected = low_value + probability * step[:, None, :]
    variance = probability * (1.0 - probability) * step[:, None, :].square()
    expected_bias_energy = float((expected - grouped).square().sum())
    target_energy = float(grouped.square().sum())
    sampling_variance = float(variance.sum())
    return int((new_codes != old_codes).sum()), {
        "stochastic_fast_expected_bias_energy": expected_bias_energy,
        "stochastic_fast_target_energy": target_energy,
        "stochastic_fast_expected_bias_recovery": (
            1.0 - expected_bias_energy / max(target_energy, 1e-30)
        ),
        "stochastic_fast_sampling_variance": sampling_variance,
        "stochastic_fast_sampling_variance_ratio": (
            sampling_variance / max(target_energy, 1e-30)
        ),
        "stochastic_fast_boundary_clipped_values": 0,
        "stochastic_fast_uniform_levels": 1,
        "stochastic_fast_block_local_groups": int(group_count),
    }


def _decode_grouped_cartesian_pair_codec(
    levels: torch.Tensor,
    codes: torch.Tensor,
    *,
    pairs_per_group: int,
) -> torch.Tensor:
    if levels.ndim != 3 or tuple(levels.shape[1:]) != (2, 16):
        raise ValueError("grouped Cartesian levels must have shape (groups, 2, 16)")
    group_count = levels.shape[0]
    if codes.ndim != 1 or codes.numel() != group_count * pairs_per_group:
        raise ValueError("grouped Cartesian codes have the wrong shape")
    selected = codes.reshape(group_count, pairs_per_group).long()
    first = levels[:, 0, :].gather(1, selected // 16)
    second = levels[:, 1, :].gather(1, selected % 16)
    return torch.stack((first, second), dim=2).reshape(-1, 2)


@torch.no_grad()
def _fit_grouped_cartesian_pair_codec_(
    vectors: torch.Tensor,
    levels: torch.Tensor,
    codes: torch.Tensor,
    *,
    pairs_per_group: int,
) -> int:
    """Vectorized Lloyd fitting for output-group-local 4-bit scalar levels."""
    if vectors.ndim != 2 or vectors.shape[1] != 2:
        raise ValueError("Cartesian pair values must have shape (pairs, 2)")
    if levels.ndim != 3 or tuple(levels.shape[1:]) != (2, 16):
        raise ValueError("grouped Cartesian levels must have shape (groups, 2, 16)")
    group_count = levels.shape[0]
    if vectors.shape[0] != group_count * pairs_per_group:
        raise ValueError("grouped Cartesian vectors have the wrong shape")
    if codes.ndim != 1 or codes.numel() != vectors.shape[0]:
        raise ValueError("grouped Cartesian codes have the wrong shape")

    old_codes = codes.clone()
    values = vectors.reshape(group_count, pairs_per_group, 2).permute(0, 2, 1)
    mean = values.mean(dim=2, keepdim=True)
    std = values.std(dim=2, unbiased=False, keepdim=True)
    old_mean = levels.mean(dim=2, keepdim=True)
    old_std = levels.std(dim=2, unbiased=False, keepdim=True)
    probabilities = (
        torch.arange(16, device=values.device, dtype=torch.float32) + 0.5
    ) / 16.0
    normal_levels = math.sqrt(2.0) * torch.erfinv(2.0 * probabilities - 1.0)
    normalized_old = (levels - old_mean) / old_std.clamp_min(
        torch.finfo(torch.float32).tiny
    )
    initialized = normal_levels.reshape(1, 1, 16).expand_as(levels)
    fitted = torch.where(
        old_std > torch.finfo(torch.float32).tiny,
        normalized_old,
        initialized,
    )
    fitted = (fitted * std + mean).sort(dim=2).values

    flat_bin_count = group_count * 2 * 16
    group_offsets = (
        torch.arange(group_count, device=values.device, dtype=torch.long)
        .reshape(group_count, 1, 1)
        .mul_(32)
    )
    coordinate_offsets = torch.tensor(
        [0, 16], device=values.device, dtype=torch.long
    ).reshape(1, 2, 1)
    for _iteration in range(2):
        midpoints = ((fitted[:, :, :-1] + fitted[:, :, 1:]) * 0.5).contiguous()
        assignments = torch.searchsorted(midpoints, values.contiguous())
        flat_indices = (
            assignments + group_offsets + coordinate_offsets
        ).reshape(-1)
        sums = torch.zeros(flat_bin_count, device=values.device, dtype=torch.float32)
        sums.scatter_add_(0, flat_indices, values.reshape(-1))
        counts = torch.zeros(flat_bin_count, device=values.device, dtype=torch.float32)
        counts.scatter_add_(0, flat_indices, torch.ones_like(values).reshape(-1))
        sums = sums.reshape(group_count, 2, 16)
        counts = counts.reshape(group_count, 2, 16)
        live = counts > 0
        fitted[live] = sums[live] / counts[live]
        fitted = fitted.sort(dim=2).values

    midpoints = ((fitted[:, :, :-1] + fitted[:, :, 1:]) * 0.5).contiguous()
    assignments = torch.searchsorted(midpoints, values.contiguous())
    new_codes = (
        assignments[:, 0, :] * 16 + assignments[:, 1, :]
    ).reshape(-1).to(torch.uint8)
    levels.copy_(fitted)
    codes.copy_(new_codes)
    return int((new_codes != old_codes).sum())


def _polar_directions(
    *, device: torch.device, angle_count: int = 32
) -> torch.Tensor:
    if angle_count <= 0:
        raise ValueError("polar angle count must be positive")
    angles = (
        torch.arange(angle_count, device=device, dtype=torch.float32)
        * (2.0 * math.pi / float(angle_count))
    )
    return torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)


def _decode_polar_pair_codec(
    radial_levels: torch.Tensor,
    center: torch.Tensor,
    codes: torch.Tensor,
) -> torch.Tensor:
    if tuple(radial_levels.shape) != (8,):
        raise ValueError("polar radial levels must have shape (8,)")
    if tuple(center.shape) != (2,):
        raise ValueError("polar center must have shape (2,)")
    selected = codes.long()
    radii = radial_levels.index_select(0, selected // 32)
    directions = _polar_directions(device=codes.device).index_select(
        0, selected % 32
    )
    return center + radii[:, None] * directions


@torch.no_grad()
def _fit_polar_pair_codec_(
    vectors: torch.Tensor,
    radial_levels: torch.Tensor,
    center: torch.Tensor,
    codes: torch.Tensor,
) -> int:
    """Fit a direction-aware 32-angle x 8-radius pair dictionary."""
    if vectors.ndim != 2 or vectors.shape[1] != 2:
        raise ValueError("polar pair values must have shape (pairs, 2)")
    if tuple(radial_levels.shape) != (8,):
        raise ValueError("polar radial levels must have shape (8,)")
    if tuple(center.shape) != (2,):
        raise ValueError("polar center must have shape (2,)")
    if codes.ndim != 1 or codes.numel() != vectors.shape[0]:
        raise ValueError("polar pair codes have the wrong shape")

    old_codes = codes.clone()
    fitted_center = vectors.mean(dim=0)
    centered = vectors - fitted_center
    angles = torch.atan2(centered[:, 1], centered[:, 0])
    angle_indices = torch.remainder(
        torch.round(angles * (32.0 / (2.0 * math.pi))).long(), 32
    )
    directions = _polar_directions(device=vectors.device).index_select(
        0, angle_indices
    )
    projected_radii = (centered * directions).sum(dim=1).clamp_min_(0.0)
    rms = projected_radii.square().mean().sqrt()
    probabilities = (
        torch.arange(8, device=vectors.device, dtype=torch.float32) + 0.5
    ) / 8.0
    rayleigh_levels = torch.sqrt(-torch.log1p(-probabilities))
    fitted = rayleigh_levels * rms
    for _iteration in range(3):
        midpoints = (fitted[:-1] + fitted[1:]) * 0.5
        radius_indices = torch.bucketize(projected_radii.contiguous(), midpoints)
        sums = torch.zeros_like(fitted)
        sums.index_add_(0, radius_indices, projected_radii)
        counts = torch.bincount(radius_indices, minlength=8)
        live = counts > 0
        fitted[live] = sums[live] / counts[live]
        fitted = fitted.sort().values
    radius_indices = torch.bucketize(
        projected_radii.contiguous(), (fitted[:-1] + fitted[1:]) * 0.5
    )
    new_codes = (radius_indices * 32 + angle_indices).to(torch.uint8)
    center.copy_(fitted_center)
    radial_levels.copy_(fitted)
    codes.copy_(new_codes)
    return int((new_codes != old_codes).sum())


def _decode_conditional_polar_pair_codec(
    radial_levels: torch.Tensor,
    center: torch.Tensor,
    codes: torch.Tensor,
) -> torch.Tensor:
    if radial_levels.ndim != 2 or radial_levels.numel() != 256:
        raise ValueError(
            "conditional polar radial levels must be a two-dimensional "
            "256-entry table"
        )
    if tuple(center.shape) != (2,):
        raise ValueError("conditional polar center must have shape (2,)")
    selected = codes.long()
    angle_count, _radius_count = radial_levels.shape
    angle_indices = selected % angle_count
    radius_indices = selected // angle_count
    radii = radial_levels[angle_indices, radius_indices]
    directions = _polar_directions(
        device=codes.device, angle_count=angle_count
    ).index_select(0, angle_indices)
    return center + radii[:, None] * directions


@torch.no_grad()
def _fit_conditional_polar_pair_codec_(
    vectors: torch.Tensor,
    radial_levels: torch.Tensor,
    center: torch.Tensor,
    codes: torch.Tensor,
    *,
    lloyd_iterations: int = 3,
) -> int:
    """Fit a 256-entry angle-by-conditional-radius product dictionary."""
    if vectors.ndim != 2 or vectors.shape[1] != 2:
        raise ValueError("conditional polar pair values must have shape (pairs, 2)")
    if radial_levels.ndim != 2 or radial_levels.numel() != 256:
        raise ValueError(
            "conditional polar radial levels must be a two-dimensional "
            "256-entry table"
        )
    if tuple(center.shape) != (2,):
        raise ValueError("conditional polar center must have shape (2,)")
    if codes.ndim != 1 or codes.numel() != vectors.shape[0]:
        raise ValueError("conditional polar pair codes have the wrong shape")
    if lloyd_iterations <= 0:
        raise ValueError("conditional polar Lloyd iterations must be positive")

    old_codes = codes.clone()
    angle_count, radius_count = radial_levels.shape
    fitted_center = vectors.mean(dim=0)
    centered = vectors - fitted_center
    angles = torch.atan2(centered[:, 1], centered[:, 0])
    angle_indices = torch.remainder(
        torch.round(
            angles * (float(angle_count) / (2.0 * math.pi))
        ).long(),
        angle_count,
    )
    directions = _polar_directions(
        device=vectors.device, angle_count=angle_count
    ).index_select(0, angle_indices)
    projected_radii = (centered * directions).sum(dim=1).clamp_min_(0.0)
    counts_by_angle = torch.bincount(angle_indices, minlength=angle_count)
    sum_squares = torch.zeros(
        angle_count, device=vectors.device, dtype=torch.float32
    )
    sum_squares.index_add_(0, angle_indices, projected_radii.square())
    rms_by_angle = (
        sum_squares / counts_by_angle.clamp_min(1).to(torch.float32)
    ).sqrt()
    probabilities = (
        torch.arange(radius_count, device=vectors.device, dtype=torch.float32)
        + 0.5
    ) / float(radius_count)
    if radius_count >= 16:
        # The measured optimizer carry is strongly heavy-tailed: a few percent
        # of pairs can contain roughly forty percent of radial energy.  Seed
        # the larger radial codebook across the observed log-radius range so
        # Lloyd fitting does not need many expensive full-matrix passes merely
        # to discover the tail.  Eight-bin historical codecs deliberately keep
        # their old Rayleigh initialization for a controlled comparison.
        max_by_angle = torch.zeros_like(rms_by_angle)
        max_by_angle.scatter_reduce_(
            0,
            angle_indices,
            projected_radii,
            reduce="amax",
            include_self=True,
        )
        lower = (0.2 * rms_by_angle).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        upper = torch.maximum(max_by_angle, lower)
        fitted = torch.exp(
            lower.log()[:, None]
            + probabilities[None, :]
            * (upper.log() - lower.log())[:, None]
        )
    else:
        rayleigh_levels = torch.sqrt(-torch.log1p(-probabilities))
        fitted = rms_by_angle[:, None] * rayleigh_levels[None, :]

    for _iteration in range(lloyd_iterations):
        midpoints = (fitted[:, :-1] + fitted[:, 1:]) * 0.5
        thresholds = midpoints.index_select(0, angle_indices)
        radius_indices = (
            projected_radii[:, None] > thresholds
        ).sum(dim=1)
        flat_indices = angle_indices * radius_count + radius_indices
        sums = torch.zeros(256, device=vectors.device, dtype=torch.float32)
        sums.index_add_(0, flat_indices, projected_radii)
        counts = torch.bincount(flat_indices, minlength=256)
        sums = sums.reshape(angle_count, radius_count)
        counts = counts.reshape(angle_count, radius_count)
        live = counts > 0
        fitted[live] = sums[live] / counts[live].to(torch.float32)
        fitted = fitted.sort(dim=1).values

    midpoints = (fitted[:, :-1] + fitted[:, 1:]) * 0.5
    thresholds = midpoints.index_select(0, angle_indices)
    radius_indices = (projected_radii[:, None] > thresholds).sum(dim=1)
    new_codes = (radius_indices * angle_count + angle_indices).to(torch.uint8)
    center.copy_(fitted_center)
    radial_levels.copy_(fitted)
    codes.copy_(new_codes)
    return int((new_codes != old_codes).sum())


def _decode_residual_conditional_polar_pair_codec(
    levels: torch.Tensor,
    centers: torch.Tensor,
    codes: torch.Tensor,
) -> torch.Tensor:
    """Decode two additive 16x16 conditional-polar bytes per pair."""
    if tuple(levels.shape) != (2, 16, 16):
        raise ValueError(
            "residual conditional-polar levels must have shape (2, 16, 16)"
        )
    if tuple(centers.shape) != (2, 2):
        raise ValueError(
            "residual conditional-polar centers must have shape (2, 2)"
        )
    if codes.ndim != 2 or codes.shape[0] != 2:
        raise ValueError(
            "residual conditional-polar codes must have shape (2, pairs)"
        )
    coarse = _decode_conditional_polar_pair_codec(
        levels[0], centers[0], codes[0]
    )
    residual = _decode_conditional_polar_pair_codec(
        levels[1], centers[1], codes[1]
    )
    return coarse + residual


@torch.no_grad()
def _fit_residual_conditional_polar_pair_codec_(
    vectors: torch.Tensor,
    levels: torch.Tensor,
    centers: torch.Tensor,
    codes: torch.Tensor,
) -> int:
    """Fit one 16x16 conditional-polar code and another on its residual."""
    if vectors.ndim != 2 or vectors.shape[1] != 2:
        raise ValueError(
            "residual conditional-polar values must have shape (pairs, 2)"
        )
    if tuple(levels.shape) != (2, 16, 16):
        raise ValueError(
            "residual conditional-polar levels must have shape (2, 16, 16)"
        )
    if tuple(centers.shape) != (2, 2):
        raise ValueError(
            "residual conditional-polar centers must have shape (2, 2)"
        )
    if codes.ndim != 2 or tuple(codes.shape) != (2, vectors.shape[0]):
        raise ValueError(
            "residual conditional-polar codes must have shape (2, pairs)"
        )
    coarse_changes = _fit_conditional_polar_pair_codec_(
        vectors, levels[0], centers[0], codes[0]
    )
    coarse = _decode_conditional_polar_pair_codec(
        levels[0], centers[0], codes[0]
    )
    residual_changes = _fit_conditional_polar_pair_codec_(
        vectors - coarse, levels[1], centers[1], codes[1]
    )
    return coarse_changes + residual_changes


@torch.no_grad()
def _residual_conditional_polar_diagnostics(
    vectors: torch.Tensor,
    levels: torch.Tensor,
    centers: torch.Tensor,
    codes: torch.Tensor,
    *,
    include_decomposition: bool = False,
    probe_lloyd_iterations: tuple[int, ...] = (),
) -> dict[str, float | int]:
    coarse = _decode_conditional_polar_pair_codec(
        levels[0], centers[0], codes[0]
    )
    residual_target = vectors - coarse
    residual = _decode_conditional_polar_pair_codec(
        levels[1], centers[1], codes[1]
    )
    final_error = residual_target - residual
    target_energy = vectors.square().sum().clamp_min(1e-30)
    residual_target_energy = residual_target.square().sum().clamp_min(1e-30)
    final_error_energy = final_error.square().sum()
    result: dict[str, float | int] = {
        "feedback_stage1_codec_energy_recovery": float(
            1.0 - residual_target_energy / target_energy
        ),
        "feedback_residual_codec_energy_recovery": float(
            1.0 - final_error_energy / residual_target_energy
        ),
        "feedback_stage1_active_codes": int(codes[0].unique().numel()),
        "feedback_residual_active_codes": int(codes[1].unique().numel()),
        "feedback_residual_target_energy": float(residual_target_energy),
        "feedback_residual_quantization_energy": float(final_error_energy),
    }
    if include_decomposition:
        details = _conditional_polar_pair_diagnostics(
            residual_target,
            levels[1],
            centers[1],
            codes[1],
        )
        for key, value in details.items():
            result[key.replace("feedback_", "feedback_residual_", 1)] = value
        result["feedback_residual_center_energy_ratio"] = float(
            centers[1].square().sum()
            * float(vectors.shape[0])
            / residual_target_energy
        )
        for iterations in probe_lloyd_iterations:
            if iterations <= 0:
                raise ValueError("residual probe Lloyd iterations must be positive")
            probe_levels = levels[1].clone()
            probe_center = centers[1].clone()
            probe_codes = codes[1].clone()
            _fit_conditional_polar_pair_codec_(
                residual_target,
                probe_levels,
                probe_center,
                probe_codes,
                lloyd_iterations=int(iterations),
            )
            probe_decoded = _decode_conditional_polar_pair_codec(
                probe_levels,
                probe_center,
                probe_codes,
            )
            probe_error = (residual_target - probe_decoded).square().sum()
            result[
                f"feedback_residual_lloyd{int(iterations)}_codec_energy_recovery"
            ] = float(1.0 - probe_error / residual_target_energy)
    return result


def _decode_free_pair_vq_rvq2(
    codebooks: torch.Tensor,
    codes: torch.Tensor,
) -> torch.Tensor:
    """Decode two additive free 256-vector Euclidean pair codebooks."""
    if tuple(codebooks.shape) != (2, 256, 2):
        raise ValueError("free pair-RVQ codebooks must have shape (2, 256, 2)")
    if codes.ndim != 2 or codes.shape[0] != 2:
        raise ValueError("free pair-RVQ codes must have shape (2, pairs)")
    return codebooks[0].index_select(0, codes[0].long()) + codebooks[
        1
    ].index_select(0, codes[1].long())


@torch.no_grad()
def _initialize_free_pair_codebook_(
    vectors: torch.Tensor,
    codebook: torch.Tensor,
) -> None:
    """Initialize a covariance-matched 256-point Gaussian spiral."""
    if vectors.ndim != 2 or vectors.shape[1] != 2:
        raise ValueError("free pair-VQ values must have shape (pairs, 2)")
    if tuple(codebook.shape) != (256, 2):
        raise ValueError("free pair-VQ codebook must have shape (256, 2)")
    center = vectors.mean(dim=0)
    centered = vectors - center
    covariance = centered.T @ centered / max(vectors.shape[0], 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    covariance_sqrt = (
        eigenvectors
        @ torch.diag(eigenvalues.clamp_min(1e-30).sqrt())
        @ eigenvectors.T
    )
    indices = torch.arange(256, device=vectors.device, dtype=torch.float32)
    probabilities = (indices + 0.5) / 256.0
    # For a two-dimensional Gaussian, the high-rate optimum point density is
    # proportional to sqrt(p), whose radial CDF is 1-exp(-r^2/4).
    radii = torch.sqrt(-4.0 * torch.log1p(-probabilities))
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    angles = indices * golden_angle
    standard = torch.stack((radii * angles.cos(), radii * angles.sin()), dim=1)
    codebook.copy_(center + standard @ covariance_sqrt.T)


@torch.no_grad()
def _transport_free_pair_codebook_(
    target: torch.Tensor,
    codebook: torch.Tensor,
    codes: torch.Tensor,
) -> None:
    """Transport a warm codebook between target first and second moments."""
    decoded = codebook.index_select(0, codes.long())
    old_mean = decoded.mean(dim=0)
    new_mean = target.mean(dim=0)
    old_centered = decoded - old_mean
    new_centered = target - new_mean
    old_covariance = old_centered.T @ old_centered / max(decoded.shape[0], 1)
    new_covariance = new_centered.T @ new_centered / max(target.shape[0], 1)
    old_values, old_vectors = torch.linalg.eigh(old_covariance)
    new_values, new_vectors = torch.linalg.eigh(new_covariance)
    old_inverse_sqrt = (
        old_vectors
        @ torch.diag(old_values.clamp_min(1e-30).rsqrt())
        @ old_vectors.T
    )
    new_sqrt = (
        new_vectors
        @ torch.diag(new_values.clamp_min(1e-30).sqrt())
        @ new_vectors.T
    )
    codebook.copy_((codebook - old_mean) @ old_inverse_sqrt @ new_sqrt + new_mean)


@torch.no_grad()
def _free_pair_local_reassign(
    vectors: torch.Tensor,
    codebook: torch.Tensor,
    codes: torch.Tensor,
    *,
    neighbor_candidates: int,
) -> torch.Tensor:
    if not 1 <= neighbor_candidates <= 256:
        raise ValueError("free pair-VQ neighbor count must be in [1, 256]")
    pairwise = torch.cdist(codebook, codebook).square()
    neighbors = pairwise.topk(
        neighbor_candidates, largest=False, dim=1
    ).indices
    parts = []
    for start in range(0, vectors.shape[0], 65536):
        stop = min(start + 65536, vectors.shape[0])
        candidate_ids = neighbors.index_select(0, codes[start:stop].long())
        candidates = codebook[candidate_ids]
        distances = (vectors[start:stop, None, :] - candidates).square().sum(dim=2)
        choices = distances.argmin(dim=1)
        parts.append(candidate_ids.gather(1, choices[:, None]).squeeze(1))
    return torch.cat(parts).to(torch.uint8)


@torch.no_grad()
def _free_pair_centroid_update_(
    vectors: torch.Tensor,
    codebook: torch.Tensor,
    codes: torch.Tensor,
) -> None:
    assignments = codes.long()
    sums = torch.zeros_like(codebook)
    sums.index_add_(0, assignments, vectors)
    counts = torch.bincount(assignments, minlength=256)
    live = counts > 0
    codebook[live] = sums[live] / counts[live, None]


@torch.no_grad()
def _fit_free_pair_vq_rvq2_(
    vectors: torch.Tensor,
    codebooks: torch.Tensor,
    codes: torch.Tensor,
    *,
    neighbor_candidates: int,
) -> int:
    """Fit two free pair-code stages with exact initialization and local updates."""
    if vectors.ndim != 2 or vectors.shape[1] != 2:
        raise ValueError("free pair-RVQ values must have shape (pairs, 2)")
    if tuple(codebooks.shape) != (2, 256, 2):
        raise ValueError("free pair-RVQ codebooks must have shape (2, 256, 2)")
    if codes.ndim != 2 or tuple(codes.shape) != (2, vectors.shape[0]):
        raise ValueError("free pair-RVQ codes must have shape (2, pairs)")

    old_codes = codes.clone()
    residual = vectors
    for stage in range(2):
        codebook = codebooks[stage]
        stage_codes = codes[stage]
        initialized = bool(codebook.abs().sum() > 0)
        if not initialized:
            _initialize_free_pair_codebook_(residual, codebook)
            for _iteration in range(3):
                stage_codes.copy_(_nearest_codes_exact(residual, codebook))
                _free_pair_centroid_update_(residual, codebook, stage_codes)
        else:
            _transport_free_pair_codebook_(residual, codebook, stage_codes)
            for _iteration in range(2):
                stage_codes.copy_(
                    _free_pair_local_reassign(
                        residual,
                        codebook,
                        stage_codes,
                        neighbor_candidates=neighbor_candidates,
                    )
                )
                _free_pair_centroid_update_(residual, codebook, stage_codes)
        residual = residual - codebook.index_select(0, stage_codes.long())
    return int((codes != old_codes).sum())


@torch.no_grad()
def _free_pair_vq_rvq2_diagnostics(
    vectors: torch.Tensor,
    codebooks: torch.Tensor,
    codes: torch.Tensor,
    *,
    include_exact_assignment: bool = False,
) -> dict[str, float | int]:
    coarse = codebooks[0].index_select(0, codes[0].long())
    residual_target = vectors - coarse
    residual = codebooks[1].index_select(0, codes[1].long())
    error = residual_target - residual
    target_energy = vectors.square().sum().clamp_min(1e-30)
    residual_target_energy = residual_target.square().sum().clamp_min(1e-30)
    error_energy = error.square().sum()

    def entropy_bits(assignments: torch.Tensor) -> float:
        counts = torch.bincount(assignments.long(), minlength=256).to(torch.float32)
        probabilities = counts[counts > 0] / counts.sum().clamp_min(1.0)
        return float(-(probabilities * probabilities.log2()).sum())

    result: dict[str, float | int] = {
        "feedback_stage1_codec_energy_recovery": float(
            1.0 - residual_target_energy / target_energy
        ),
        "feedback_residual_codec_energy_recovery": float(
            1.0 - error_energy / residual_target_energy
        ),
        "feedback_codec_energy_recovery": float(1.0 - error_energy / target_energy),
        "feedback_stage1_active_codes": int(codes[0].unique().numel()),
        "feedback_residual_active_codes": int(codes[1].unique().numel()),
        "feedback_stage1_code_entropy_bits": entropy_bits(codes[0]),
        "feedback_residual_code_entropy_bits": entropy_bits(codes[1]),
        "feedback_residual_target_energy": float(residual_target_energy),
        "feedback_residual_quantization_energy": float(error_energy),
    }
    if include_exact_assignment:
        exact_stage2_codes = _nearest_codes_exact(
            residual_target, codebooks[1]
        )
        exact_residual = codebooks[1].index_select(
            0, exact_stage2_codes.long()
        )
        exact_error_energy = (residual_target - exact_residual).square().sum()
        exact_full_recovery = float(1.0 - exact_error_energy / target_energy)
        result.update(
            {
                "feedback_exact_same_codebook_energy_recovery": exact_full_recovery,
                "feedback_exact_same_codebook_residual_recovery": float(
                    1.0 - exact_error_energy / residual_target_energy
                ),
                "feedback_local_assignment_recovery_gap": float(
                    exact_full_recovery - result["feedback_codec_energy_recovery"]
                ),
            }
        )
    return result


@torch.no_grad()
def _signed_block_fht(
    vectors: torch.Tensor,
    *,
    block_size: int,
    seed: int,
) -> torch.Tensor:
    """Return signed normalized-FHT blocks without persistent transform state."""
    if block_size < 2 or block_size & (block_size - 1):
        raise ValueError("block-FHT pair probe size must be a power of two")
    flat = vectors.reshape(-1)
    if flat.numel() % block_size:
        raise ValueError("block-FHT pair probe size must divide the element count")
    if block_size == 2:
        return vectors.reshape(-1, block_size)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    signs = (
        torch.randint(
            0,
            2,
            (block_size,),
            generator=generator,
            dtype=torch.int8,
        )
        .to(device=vectors.device, dtype=torch.float32)
        .mul_(2.0)
        .sub_(1.0)
    )
    return normalized_fht_last_dim(flat.reshape(-1, block_size) * signs)


@torch.no_grad()
def _block_fht_free_pair_vq_counterfactual(
    vectors: torch.Tensor,
    *,
    block_size: int,
    seed: int,
) -> dict[str, float | int]:
    """Probe exact free-pair coding after invertible signed block mixing."""
    if vectors.ndim != 2 or vectors.shape[1] != 2:
        raise ValueError("block-FHT pair probe values must have shape (pairs, 2)")
    transformed = _signed_block_fht(
        vectors,
        block_size=block_size,
        seed=seed,
    ).reshape(-1, 2)
    codebooks = torch.zeros(
        2, 256, 2, device=vectors.device, dtype=torch.float32
    )
    codes = torch.zeros(
        2, transformed.shape[0], device=vectors.device, dtype=torch.uint8
    )
    _fit_free_pair_vq_rvq2_(
        transformed,
        codebooks,
        codes,
        neighbor_candidates=16,
    )
    diagnostics = _free_pair_vq_rvq2_diagnostics(
        transformed,
        codebooks,
        codes,
    )
    source_energy = vectors.square().sum().clamp_min(1e-30)
    transformed_energy = transformed.square().sum()
    return {
        "full_recovery": diagnostics["feedback_codec_energy_recovery"],
        "residual_recovery": diagnostics[
            "feedback_residual_codec_energy_recovery"
        ],
        "stage1_recovery": diagnostics[
            "feedback_stage1_codec_energy_recovery"
        ],
        "stage1_active_codes": diagnostics["feedback_stage1_active_codes"],
        "stage2_active_codes": diagnostics["feedback_residual_active_codes"],
        "stage1_entropy_bits": diagnostics[
            "feedback_stage1_code_entropy_bits"
        ],
        "stage2_entropy_bits": diagnostics[
            "feedback_residual_code_entropy_bits"
        ],
        "parseval_relative_error": float(
            (transformed_energy - source_energy).abs() / source_energy
        ),
    }


@torch.no_grad()
def _fit_scalar_codebook(
    values: torch.Tensor,
    *,
    level_count: int,
    iterations: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("scalar-codebook values must be a nonempty vector")
    if level_count < 2 or level_count > 256:
        raise ValueError("scalar-codebook level count must be in [2, 256]")
    mean = values.mean()
    std = values.std(unbiased=False).clamp_min(torch.finfo(torch.float32).tiny)
    probabilities = (
        torch.arange(
            level_count,
            device=values.device,
            dtype=torch.float32,
        )
        + 0.5
    ) / level_count
    levels = mean + std * math.sqrt(2.0) * torch.erfinv(
        2.0 * probabilities - 1.0
    )
    for _iteration in range(iterations):
        codes = torch.bucketize(
            values.contiguous(), (levels[:-1] + levels[1:]) * 0.5
        )
        sums = torch.zeros_like(levels)
        sums.index_add_(0, codes, values)
        counts = torch.bincount(codes, minlength=level_count)
        live = counts > 0
        candidate = sums / counts.clamp_min(1).to(dtype=sums.dtype)
        levels = torch.where(live, candidate, levels)
        levels = levels.sort().values
    codes = torch.bucketize(
        values.contiguous(), (levels[:-1] + levels[1:]) * 0.5
    )
    return levels, codes


@torch.no_grad()
def _fit_scalar_codebooks_batched(
    values: torch.Tensor,
    *,
    level_count: int,
    iterations: int,
    hierarchical: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit independent scalar codebooks with one reduction per phase.

    This is the production counterpart of the sealed B64/L16 fitter oracle.
    The hierarchical path uses block-private CUDA histograms; the reference
    path preserves PyTorch's flattened, offset-index reduction.
    """
    if values.ndim != 2 or values.dtype != torch.float32:
        raise ValueError("batched scalar-codebook values must be an FP32 matrix")
    if values.shape[0] <= 0 or values.shape[1] <= 0:
        raise ValueError("batched scalar-codebook values must be nonempty")
    if level_count < 2 or level_count > 256:
        raise ValueError("scalar-codebook level count must be in [2, 256]")
    batch = values.shape[0]
    mean = torch.stack([row.mean() for row in values])
    std = torch.stack([row.std(unbiased=False) for row in values]).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    probabilities = (
        torch.arange(level_count, device=values.device, dtype=torch.float32)
        + 0.5
    ) / level_count
    levels = mean[:, None] + std[:, None] * math.sqrt(2.0) * torch.erfinv(
        2.0 * probabilities[None, :] - 1.0
    )
    offsets = (
        torch.arange(batch, device=values.device, dtype=torch.int64)
        * level_count
    )
    flat_values = values.reshape(-1)
    for _iteration in range(iterations):
        midpoints = ((levels[:, :-1] + levels[:, 1:]) * 0.5).contiguous()
        if hierarchical:
            sums, counts = hierarchical_lloyd_stats(
                values.contiguous(), midpoints, level_count=level_count
            )
        else:
            codes = torch.searchsorted(midpoints, values.contiguous())
            codes.add_(offsets[:, None])
            flat_codes = codes.reshape(-1)
            sums = torch.zeros(
                batch * level_count,
                device=values.device,
                dtype=torch.float32,
            )
            sums.index_add_(0, flat_codes, flat_values)
            counts = torch.bincount(
                flat_codes, minlength=batch * level_count
            )
            sums = sums.reshape(batch, level_count)
            counts = counts.reshape(batch, level_count)
        live = counts > 0
        candidate = sums / counts.clamp_min(1).to(dtype=sums.dtype)
        levels = torch.where(live, candidate, levels).sort(dim=1).values
    midpoints = ((levels[:, :-1] + levels[:, 1:]) * 0.5).contiguous()
    codes = torch.searchsorted(midpoints, values.contiguous())
    return levels, codes


@torch.no_grad()
def _fit_fractional_residual_lattice_feedback_batch_(
    entries: list[dict[str, Any]],
) -> list[int]:
    """Fit the exact 24-module B64/L16 feedback state in frozen phases.

    Persistent tensors and the byte-exact codec are unchanged.  Only the
    execution schedule differs from the serial fitter: equal production roles
    are stacked, fitted, and then packed back into their existing tensors.
    """
    if len(entries) != 24:
        raise ValueError("hierarchical B64/L16 fitting requires 24 modules")
    element_counts = {int(entry["vectors"].numel()) for entry in entries}
    devices = {entry["vectors"].device for entry in entries}
    if element_counts != {2_359_296} or len(devices) != 1:
        raise ValueError(
            "hierarchical B64/L16 fitting requires 24 equal production tensors"
        )
    cfc_indices = [
        index for index, entry in enumerate(entries)
        if int(entry["coordinate_bits"]) == 5
        and int(entry["block_size"]) == 64
        and int(entry["lloyd_iterations"]) == 16
    ]
    cproj_indices = [
        index for index, entry in enumerate(entries)
        if int(entry["coordinate_bits"]) == 4
        and int(entry["block_size"]) == 32
        and int(entry["lloyd_iterations"]) == 4
    ]
    if len(cfc_indices) != 12 or len(cproj_indices) != 12:
        raise ValueError("hierarchical B64/L16 fitting requires 12 c_fc and 12 c_proj")
    old_packed = [entry["packed"].clone() for entry in entries]

    base_transformed_parts = [
        _signed_block_fht(
            entry["vectors"], block_size=32, seed=int(entry["seed"])
        )
        for entry in entries
    ]
    base_transformed = torch.stack(base_transformed_parts)
    base_gains = torch.stack(
        [
            transformed.square().mean(dim=1).sqrt().clamp_min(1e-30)
            for transformed in base_transformed_parts
        ]
    )
    gain_outputs = [
        _fit_scalar_codebook(row, level_count=256, iterations=4)
        for row in base_gains.log()
    ]
    gain_levels = torch.stack([output[0] for output in gain_outputs])
    gain_codes = torch.stack([output[1] for output in gain_outputs])
    decoded_gains = torch.gather(gain_levels, 1, gain_codes).exp()
    normalized = torch.stack(
        [
            (transformed / decoded_gains[index, :, None]).reshape(-1)
            for index, transformed in enumerate(base_transformed_parts)
        ]
    )
    base_levels, base_codes = _fit_scalar_codebooks_batched(
        normalized, level_count=128, iterations=4, hierarchical=False
    )
    refined_outputs = [
        _fit_scalar_codebook(
            row,
            level_count=256,
            iterations=4,
        )
        for row in normalized
    ]
    refined_levels = torch.stack([output[0] for output in refined_outputs])
    refined_codes = torch.stack([output[1] for output in refined_outputs])

    for index, entry in enumerate(entries):
        vectors = entry["vectors"]
        levels = entry["levels"]
        packed = entry["packed"]
        element_count = vectors.numel()
        layout = _fractional_lattice_feedback_layout(element_count)
        transformed = base_transformed[index]
        layer_decoded_gains = decoded_gains[index]
        layer_base_codes = base_codes[index]
        layer_refined_codes = refined_codes[index]
        base_error = (
            transformed
            - base_levels[index].index_select(0, layer_base_codes).reshape_as(
                transformed
            ) * layer_decoded_gains[:, None]
        ).square().sum(dim=1)
        refined_error = (
            transformed
            - refined_levels[index].index_select(
                0, layer_refined_codes
            ).reshape_as(transformed) * layer_decoded_gains[:, None]
        ).square().sum(dim=1)
        selected = torch.topk(
            base_error - refined_error,
            layout["selected_blocks"],
            largest=True,
            sorted=False,
        ).indices
        selector = torch.zeros(
            layout["block_count"], device=vectors.device, dtype=torch.uint8
        )
        selector[selected] = 1
        selected_mask = selector.bool()
        stored_coordinates = layer_base_codes.reshape(
            layout["block_count"], 32
        ).clone()
        selected_refined = layer_refined_codes.reshape(
            layout["block_count"], 32
        )[selected_mask]
        stored_coordinates[selected_mask] = selected_refined & 127
        levels[:256].copy_(gain_levels[index])
        levels[256:384].copy_(base_levels[index])
        levels[384:640].copy_(refined_levels[index])
        base_stop = _fractional_residual_lattice_feedback_layout(
            element_count,
            coordinate_bits=int(entry["coordinate_bits"]),
            block_size=int(entry["block_size"]),
        )["base_bytes"]
        gain_stream, base_stream, selector_stream, refinement_stream = (
            _fractional_lattice_feedback_segments(
                packed[:base_stop], element_count=element_count
            )
        )
        gain_stream.copy_(gain_codes[index].to(torch.uint8))
        base_stream.copy_(_pack_fixed_width_codes(stored_coordinates, bits=7))
        selector_stream.copy_(_pack_fixed_width_codes(selector, bits=1))
        refinement_stream.copy_(
            _pack_fixed_width_codes(selected_refined >> 7, bits=1)
        )

    residual_transformed: dict[int, torch.Tensor] = {}
    for index, entry in enumerate(entries):
        vectors = entry["vectors"]
        element_count = vectors.numel()
        layout = _fractional_residual_lattice_feedback_layout(
            element_count,
            coordinate_bits=int(entry["coordinate_bits"]),
            block_size=int(entry["block_size"]),
        )
        base = _decode_fractional_lattice_feedback(
            entry["levels"][:640],
            entry["packed"][: layout["base_bytes"]],
            element_count=element_count,
            seed=int(entry["seed"]),
        )
        residual_transformed[index] = _signed_block_fht(
            vectors - base,
            block_size=int(entry["block_size"]),
            seed=int(entry["seed"]),
        )

    for role_indices in (cfc_indices, cproj_indices):
        first = entries[role_indices[0]]
        coordinate_bits = int(first["coordinate_bits"])
        iterations = int(first["lloyd_iterations"])
        transformed_parts = [
            residual_transformed[index] for index in role_indices
        ]
        transformed = torch.stack(transformed_parts)
        gains = torch.stack(
            [
                layer.square().mean(dim=1).sqrt().clamp_min(1e-30)
                for layer in transformed_parts
            ]
        )
        residual_gain_outputs = [
            _fit_scalar_codebook(
                row,
                level_count=256,
                iterations=iterations,
            )
            for row in gains.log()
        ]
        residual_gain_levels = torch.stack(
            [output[0] for output in residual_gain_outputs]
        )
        residual_gain_codes = torch.stack(
            [output[1] for output in residual_gain_outputs]
        )
        residual_decoded_gains = torch.gather(
            residual_gain_levels, 1, residual_gain_codes
        ).exp()
        residual_normalized = torch.stack(
            [
                (
                    layer / residual_decoded_gains[row, :, None]
                ).reshape(-1)
                for row, layer in enumerate(transformed_parts)
            ]
        )
        residual_coordinate_levels, residual_coordinate_codes = (
            _fit_scalar_codebooks_batched(
                residual_normalized,
                level_count=1 << coordinate_bits,
                iterations=iterations,
                hierarchical=True,
            )
        )
        for row, index in enumerate(role_indices):
            entry = entries[index]
            levels = entry["levels"]
            packed = entry["packed"]
            layout = _fractional_residual_lattice_feedback_layout(
                entry["vectors"].numel(),
                coordinate_bits=coordinate_bits,
                block_size=int(entry["block_size"]),
            )
            base_stop = layout["base_bytes"]
            gain_stop = base_stop + layout["residual_gain_bytes"]
            levels[640:896].copy_(residual_gain_levels[row])
            levels[896 : 896 + (1 << coordinate_bits)].copy_(
                residual_coordinate_levels[row]
            )
            packed[base_stop:gain_stop].copy_(
                residual_gain_codes[row].to(torch.uint8)
            )
            packed[gain_stop:].copy_(
                _pack_fixed_width_codes(
                    residual_coordinate_codes[row], bits=coordinate_bits
                )
            )
    return [
        int((entry["packed"] != before).sum())
        for entry, before in zip(entries, old_packed, strict=True)
    ]


def _pack_fixed_width_codes(codes: torch.Tensor, *, bits: int) -> torch.Tensor:
    """Pack eight unsigned fixed-width codes into exactly ``bits`` bytes."""
    if not 1 <= bits <= 7:
        raise ValueError("packed code width must be in [1, 7]")
    flat = codes.reshape(-1).long()
    if bool((flat < 0).any()) or bool((flat >= (1 << bits)).any()):
        raise ValueError("packed code exceeds the requested bit width")
    padded_count = ((flat.numel() + 7) // 8) * 8
    if padded_count != flat.numel():
        flat = F.pad(flat, (0, padded_count - flat.numel()))
    groups = flat.reshape(-1, 8)
    code_shifts = bits * torch.arange(8, device=flat.device, dtype=torch.int64)
    words = (groups << code_shifts).sum(dim=1)
    byte_shifts = 8 * torch.arange(bits, device=flat.device, dtype=torch.int64)
    return ((words[:, None] >> byte_shifts) & 255).to(torch.uint8).reshape(-1)


def _unpack_fixed_width_codes(
    packed: torch.Tensor,
    *,
    bits: int,
    count: int,
) -> torch.Tensor:
    """Inverse of :func:`_pack_fixed_width_codes` for a known code count."""
    if not 1 <= bits <= 7:
        raise ValueError("packed code width must be in [1, 7]")
    group_count = (int(count) + 7) // 8
    if packed.numel() != group_count * bits:
        raise ValueError("packed byte stream has the wrong length")
    byte_groups = packed.reshape(group_count, bits).long()
    byte_shifts = 8 * torch.arange(
        bits, device=packed.device, dtype=torch.int64
    )
    words = (byte_groups << byte_shifts).sum(dim=1)
    code_shifts = bits * torch.arange(
        8, device=packed.device, dtype=torch.int64
    )
    codes = ((words[:, None] >> code_shifts) & ((1 << bits) - 1)).to(
        torch.uint8
    )
    return codes.reshape(-1)[:count]


def _signed_block_fht_signs(
    block_size: int,
    *,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return (
        torch.randint(
            0,
            2,
            (block_size,),
            generator=generator,
            dtype=torch.int8,
        )
        .to(device=device, dtype=torch.float32)
        .mul_(2.0)
        .sub_(1.0)
    )


def _inverse_signed_block_fht(
    transformed: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    if transformed.ndim != 2:
        raise ValueError("signed block-FHT inverse expects a block matrix")
    block_size = transformed.shape[1]
    if block_size < 2 or block_size & (block_size - 1):
        raise ValueError("signed block-FHT inverse size must be a power of two")
    if block_size == 2:
        return transformed
    signs = _signed_block_fht_signs(
        block_size,
        seed=seed,
        device=transformed.device,
    )
    return normalized_fht_last_dim(transformed) * signs


def _fractional_lattice_feedback_layout(element_count: int) -> dict[str, int]:
    """Byte-exact B32/q7/q8/p=.25 production feedback layout."""
    if element_count % 256:
        raise ValueError(
            "fractional lattice feedback requires an element count divisible by 256"
        )
    block_count = element_count // 32
    selected_blocks = block_count // 4
    gain_bytes = block_count
    base_bytes = element_count * 7 // 8
    selector_bytes = block_count // 8
    refinement_bytes = selected_blocks * 4
    return {
        "block_count": block_count,
        "selected_blocks": selected_blocks,
        "gain_bytes": gain_bytes,
        "base_bytes": base_bytes,
        "selector_bytes": selector_bytes,
        "refinement_bytes": refinement_bytes,
        "total_bytes": (
            gain_bytes + base_bytes + selector_bytes + refinement_bytes
        ),
    }


def _fractional_lattice_feedback_segments(
    packed: torch.Tensor,
    *,
    element_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    layout = _fractional_lattice_feedback_layout(element_count)
    if packed.ndim != 1 or packed.numel() != layout["total_bytes"]:
        raise ValueError("fractional lattice feedback stream has the wrong shape")
    gain_stop = layout["gain_bytes"]
    base_stop = gain_stop + layout["base_bytes"]
    selector_stop = base_stop + layout["selector_bytes"]
    return (
        packed[:gain_stop],
        packed[gain_stop:base_stop],
        packed[base_stop:selector_stop],
        packed[selector_stop:],
    )


def _decode_fractional_lattice_feedback(
    levels: torch.Tensor,
    packed: torch.Tensor,
    *,
    element_count: int,
    seed: int,
) -> torch.Tensor:
    """Decode the packed B32 q7/q8 temporal feedback into ambient pairs."""
    if tuple(levels.shape) != (640,):
        raise ValueError("fractional lattice levels must have shape (640,)")
    layout = _fractional_lattice_feedback_layout(element_count)
    gain_stream, base_stream, selector_stream, refinement_stream = (
        _fractional_lattice_feedback_segments(
            packed,
            element_count=element_count,
        )
    )
    gain_levels = levels[:256]
    base_levels = levels[256:384]
    refined_levels = levels[384:640]
    decoded_gains = gain_levels.index_select(0, gain_stream.long()).exp()
    coordinate_codes = _unpack_fixed_width_codes(
        base_stream,
        bits=7,
        count=element_count,
    ).reshape(layout["block_count"], 32)
    selector = _unpack_fixed_width_codes(
        selector_stream,
        bits=1,
        count=layout["block_count"],
    ).bool()
    selected_count = int(selector.sum())
    if selected_count not in (0, layout["selected_blocks"]):
        raise ValueError("fractional lattice selector has an invalid population")
    decoded = base_levels.index_select(
        0, coordinate_codes.long().reshape(-1)
    ).reshape_as(coordinate_codes)
    if selected_count:
        high_bits = _unpack_fixed_width_codes(
            refinement_stream,
            bits=1,
            count=layout["selected_blocks"] * 32,
        ).reshape(layout["selected_blocks"], 32)
        refined_codes = coordinate_codes[selector].long() + 128 * high_bits.long()
        decoded[selector] = refined_levels.index_select(
            0, refined_codes.reshape(-1)
        ).reshape(layout["selected_blocks"], 32)
    decoded.mul_(decoded_gains[:, None])
    ambient = _inverse_signed_block_fht(decoded, seed=seed)
    return ambient.reshape(-1, 2)


@torch.no_grad()
def _fit_fractional_lattice_feedback_(
    vectors: torch.Tensor,
    levels: torch.Tensor,
    packed: torch.Tensor,
    *,
    seed: int,
) -> int:
    """Fit and pack the selected B32/q7/q8/p=.25 temporal feedback."""
    if vectors.ndim != 2 or vectors.shape[1] != 2:
        raise ValueError("fractional lattice feedback expects ambient pairs")
    element_count = vectors.numel()
    layout = _fractional_lattice_feedback_layout(element_count)
    old_packed = packed.clone()
    transformed = _signed_block_fht(vectors, block_size=32, seed=seed)
    gains = transformed.square().mean(dim=1).sqrt().clamp_min(1e-30)
    gain_levels, gain_codes = _fit_scalar_codebook(
        gains.log(), level_count=256
    )
    decoded_gains = gain_levels.index_select(0, gain_codes).exp()
    normalized = (transformed / decoded_gains[:, None]).reshape(-1)
    base_levels, base_codes = _fit_scalar_codebook(normalized, level_count=128)
    refined_levels, refined_codes = _fit_scalar_codebook(
        normalized, level_count=256
    )
    base_error = (
        transformed
        - base_levels.index_select(0, base_codes).reshape_as(transformed)
        * decoded_gains[:, None]
    ).square().sum(dim=1)
    refined_error = (
        transformed
        - refined_levels.index_select(0, refined_codes).reshape_as(transformed)
        * decoded_gains[:, None]
    ).square().sum(dim=1)
    selected = torch.topk(
        base_error - refined_error,
        layout["selected_blocks"],
        largest=True,
        sorted=False,
    ).indices
    selector = torch.zeros(
        layout["block_count"], device=vectors.device, dtype=torch.uint8
    )
    selector[selected] = 1
    selected_mask = selector.bool()
    stored_coordinates = base_codes.reshape(layout["block_count"], 32).clone()
    selected_refined = refined_codes.reshape(layout["block_count"], 32)[
        selected_mask
    ]
    stored_coordinates[selected_mask] = selected_refined & 127
    levels[:256].copy_(gain_levels)
    levels[256:384].copy_(base_levels)
    levels[384:640].copy_(refined_levels)
    gain_stream, base_stream, selector_stream, refinement_stream = (
        _fractional_lattice_feedback_segments(
            packed,
            element_count=element_count,
        )
    )
    gain_stream.copy_(gain_codes.to(torch.uint8))
    base_stream.copy_(
        _pack_fixed_width_codes(stored_coordinates, bits=7)
    )
    selector_stream.copy_(_pack_fixed_width_codes(selector, bits=1))
    refinement_stream.copy_(
        _pack_fixed_width_codes((selected_refined >> 7), bits=1)
    )
    return int((packed != old_packed).sum())


def _fractional_residual_lattice_feedback_layout(
    element_count: int,
    *,
    coordinate_bits: int = 4,
    block_size: int = 32,
) -> dict[str, int]:
    """Byte-exact q7/q8 base plus a block-scaled residual layout."""
    if coordinate_bits < 2 or coordinate_bits > 7:
        raise ValueError("residual lattice coordinate bits must be in [2, 7]")
    if block_size < 2 or block_size & (block_size - 1):
        raise ValueError("residual lattice block size must be a power of two")
    if element_count % block_size:
        raise ValueError("residual lattice block size must divide the source")
    base = _fractional_lattice_feedback_layout(element_count)
    residual_block_count = element_count // block_size
    residual_gain_bytes = residual_block_count
    residual_code_bytes = ((element_count + 7) // 8) * coordinate_bits
    return {
        "base_bytes": base["total_bytes"],
        "block_count": residual_block_count,
        "block_size": block_size,
        "coordinate_bits": coordinate_bits,
        "residual_gain_bytes": residual_gain_bytes,
        "residual_code_bytes": residual_code_bytes,
        "total_bytes": (
            base["total_bytes"] + residual_gain_bytes + residual_code_bytes
        ),
    }


def _decode_fractional_residual_lattice_feedback(
    levels: torch.Tensor,
    packed: torch.Tensor,
    *,
    element_count: int,
    seed: int,
    coordinate_bits: int = 4,
    block_size: int = 32,
) -> torch.Tensor:
    """Decode a fractional base and additive innovation lattice."""
    level_count = 1 << coordinate_bits
    expected_level_count = 896 + level_count
    if tuple(levels.shape) != (expected_level_count,):
        raise ValueError(
            "fractional residual lattice levels must have shape "
            f"({expected_level_count},)"
        )
    layout = _fractional_residual_lattice_feedback_layout(
        element_count,
        coordinate_bits=coordinate_bits,
        block_size=block_size,
    )
    if packed.ndim != 1 or packed.numel() != layout["total_bytes"]:
        raise ValueError("fractional residual lattice stream has the wrong shape")
    base_stop = layout["base_bytes"]
    gain_stop = base_stop + layout["residual_gain_bytes"]
    base = _decode_fractional_lattice_feedback(
        levels[:640],
        packed[:base_stop],
        element_count=element_count,
        seed=seed,
    )
    gain_codes = packed[base_stop:gain_stop].long()
    coordinate_codes = _unpack_fixed_width_codes(
        packed[gain_stop:],
        bits=coordinate_bits,
        count=element_count,
    ).reshape(layout["block_count"], block_size)
    decoded_gains = levels[640:896].index_select(0, gain_codes).exp()
    transformed = levels[896 : 896 + level_count].index_select(
        0, coordinate_codes.long().reshape(-1)
    ).reshape_as(coordinate_codes)
    transformed.mul_(decoded_gains[:, None])
    innovation = _inverse_signed_block_fht(transformed, seed=seed)
    return base + innovation.reshape(-1, 2)


@torch.no_grad()
def _fit_fractional_residual_lattice_feedback_(
    vectors: torch.Tensor,
    levels: torch.Tensor,
    packed: torch.Tensor,
    *,
    seed: int,
    coordinate_bits: int = 4,
    block_size: int = 32,
    lloyd_iterations: int = 4,
) -> int:
    """Fit the q7/q8 base and a block-scaled quantization residual."""
    if vectors.ndim != 2 or vectors.shape[1] != 2:
        raise ValueError("fractional residual lattice feedback expects ambient pairs")
    element_count = vectors.numel()
    level_count = 1 << coordinate_bits
    expected_level_count = 896 + level_count
    layout = _fractional_residual_lattice_feedback_layout(
        element_count,
        coordinate_bits=coordinate_bits,
        block_size=block_size,
    )
    if tuple(levels.shape) != (expected_level_count,):
        raise ValueError(
            "fractional residual lattice levels must have shape "
            f"({expected_level_count},)"
        )
    if packed.ndim != 1 or packed.numel() != layout["total_bytes"]:
        raise ValueError("fractional residual lattice stream has the wrong shape")
    base_stop = layout["base_bytes"]
    gain_stop = base_stop + layout["residual_gain_bytes"]
    base_changes = _fit_fractional_lattice_feedback_(
        vectors,
        levels[:640],
        packed[:base_stop],
        seed=seed,
    )
    base = _decode_fractional_lattice_feedback(
        levels[:640],
        packed[:base_stop],
        element_count=element_count,
        seed=seed,
    )
    innovation = vectors - base
    transformed = _signed_block_fht(
        innovation, block_size=block_size, seed=seed
    )
    gains = transformed.square().mean(dim=1).sqrt().clamp_min(1e-30)
    gain_levels, gain_codes = _fit_scalar_codebook(
        gains.log(), level_count=256, iterations=lloyd_iterations
    )
    decoded_gains = gain_levels.index_select(0, gain_codes).exp()
    normalized = (transformed / decoded_gains[:, None]).reshape(-1)
    coordinate_levels, coordinate_codes = _fit_scalar_codebook(
        normalized,
        level_count=level_count,
        iterations=lloyd_iterations,
    )
    old_residual = packed[base_stop:].clone()
    levels[640:896].copy_(gain_levels)
    levels[896 : 896 + level_count].copy_(coordinate_levels)
    packed[base_stop:gain_stop].copy_(gain_codes.to(torch.uint8))
    packed[gain_stop:].copy_(
        _pack_fixed_width_codes(
            coordinate_codes,
            bits=coordinate_bits,
        )
    )
    return base_changes + int((packed[base_stop:] != old_residual).sum())


@torch.no_grad()
def _fractional_residual_lattice_source_decomposition(
    vectors: torch.Tensor,
    *,
    seed: int,
    block_sizes: tuple[int, ...],
    coordinate_bits: tuple[int, ...],
    lloyd_iterations: tuple[int, ...],
    axis_block_size: int = 0,
    axis_coordinate_bits: int = 5,
) -> dict[str, float | int]:
    """Decompose residual-lattice error on one fixed q7/q8 innovation.

    This is an acquisition oracle only. It never mutates the live codec or
    registers dense state.
    """
    if vectors.ndim != 2 or vectors.shape[1] != 2:
        raise ValueError("fractional source probe expects ambient pairs")
    if not block_sizes or not coordinate_bits or not lloyd_iterations:
        raise ValueError("fractional source probe requires blocks, bits, and iterations")
    element_count = vectors.numel()
    base_layout = _fractional_lattice_feedback_layout(element_count)
    base_levels = torch.zeros(640, device=vectors.device, dtype=torch.float32)
    base_packed = torch.zeros(
        base_layout["total_bytes"], device=vectors.device, dtype=torch.uint8
    )
    _fit_fractional_lattice_feedback_(
        vectors, base_levels, base_packed, seed=seed
    )
    base = _decode_fractional_lattice_feedback(
        base_levels,
        base_packed,
        element_count=element_count,
        seed=seed,
    )
    innovation = vectors - base
    source_energy = vectors.square().sum().clamp_min(1e-30)
    innovation_energy = innovation.square().sum().clamp_min(1e-30)

    def entropy(codes: torch.Tensor, level_count: int) -> float:
        counts = torch.bincount(codes.reshape(-1), minlength=level_count).float()
        probabilities = counts[counts > 0] / counts.sum().clamp_min(1.0)
        return float(-(probabilities * probabilities.log2()).sum())

    def score(
        transformed: torch.Tensor,
        *,
        bits: int,
        iterations: int,
        exact_gains: bool,
        axis_private: bool,
    ) -> dict[str, float | int]:
        level_count = 1 << bits
        actual_gains = transformed.square().mean(dim=1).sqrt().clamp_min(1e-30)
        gain_active = transformed.shape[0]
        gain_entropy = float("nan")
        if exact_gains:
            decoded_gains = actual_gains
        else:
            gain_levels, gain_codes = _fit_scalar_codebook(
                actual_gains.log(), level_count=256, iterations=iterations
            )
            decoded_gains = gain_levels.index_select(0, gain_codes).exp()
            gain_active = int(gain_codes.unique().numel())
            gain_entropy = entropy(gain_codes, 256)
        normalized = transformed / decoded_gains[:, None]
        if axis_private:
            decoded_columns: list[torch.Tensor] = []
            active_counts: list[int] = []
            entropies: list[float] = []
            for axis in range(transformed.shape[1]):
                levels, codes = _fit_scalar_codebook(
                    normalized[:, axis],
                    level_count=level_count,
                    iterations=iterations,
                )
                decoded_columns.append(levels.index_select(0, codes))
                active_counts.append(int(codes.unique().numel()))
                entropies.append(entropy(codes, level_count))
            decoded_normalized = torch.stack(decoded_columns, dim=1)
            coordinate_active = min(active_counts)
            coordinate_entropy = min(entropies)
        else:
            levels, codes = _fit_scalar_codebook(
                normalized.reshape(-1),
                level_count=level_count,
                iterations=iterations,
            )
            decoded_normalized = levels.index_select(0, codes).reshape_as(
                normalized
            )
            coordinate_active = int(codes.unique().numel())
            coordinate_entropy = entropy(codes, level_count)
        decoded = decoded_normalized * decoded_gains[:, None]
        error_energy = (transformed - decoded).square().sum()
        return {
            "full_recovery": float(1.0 - error_energy / source_energy),
            "innovation_recovery": float(1.0 - error_energy / innovation_energy),
            "error_energy": float(error_energy),
            "coordinate_active_codes_min": coordinate_active,
            "coordinate_entropy_bits_min": coordinate_entropy,
            "gain_active_codes": gain_active,
            "gain_entropy_bits": gain_entropy,
        }

    output: dict[str, float | int] = {
        "base_full_recovery": float(1.0 - innovation_energy / source_energy),
        "innovation_energy_ratio": float(innovation_energy / source_energy),
    }
    transformed_by_block: dict[int, torch.Tensor] = {}
    for block_size in block_sizes:
        transformed = _signed_block_fht(
            innovation, block_size=block_size, seed=seed
        )
        transformed_by_block[block_size] = transformed
        output[f"b{block_size}_parseval_relative_error"] = float(
            (transformed.square().sum() - innovation_energy).abs()
            / innovation_energy
        )
        for bits in coordinate_bits:
            for iterations in lloyd_iterations:
                metrics = score(
                    transformed,
                    bits=bits,
                    iterations=iterations,
                    exact_gains=False,
                    axis_private=False,
                )
                prefix = f"b{block_size}_q{bits}_lloyd{iterations}_quantgain_"
                for key, value in metrics.items():
                    output[prefix + key] = value

    reference_block_size = 32 if 32 in transformed_by_block else block_sizes[0]
    reference = transformed_by_block[reference_block_size]
    for bits in coordinate_bits:
        for iterations in lloyd_iterations:
            metrics = score(
                reference,
                bits=bits,
                iterations=iterations,
                exact_gains=True,
                axis_private=False,
            )
            prefix = (
                f"b{reference_block_size}_q{bits}_lloyd{iterations}_exactgain_"
            )
            for key, value in metrics.items():
                output[prefix + key] = value

    if axis_block_size:
        transformed = transformed_by_block.get(axis_block_size)
        if transformed is None:
            transformed = _signed_block_fht(
                innovation, block_size=axis_block_size, seed=seed
            )
        iterations = max(lloyd_iterations)
        metrics = score(
            transformed,
            bits=axis_coordinate_bits,
            iterations=iterations,
            exact_gains=False,
            axis_private=True,
        )
        prefix = (
            f"b{axis_block_size}_q{axis_coordinate_bits}_lloyd{iterations}_"
            "quantgain_axis_"
        )
        for key, value in metrics.items():
            output[prefix + key] = value
    return output


@torch.no_grad()
def _block_fht_gain_lattice_counterfactual(
    vectors: torch.Tensor,
    *,
    block_size: int,
    coordinate_bits: int,
    seed: int,
) -> dict[str, float | int]:
    """Measure a block-gain plus bit-packed coordinate lattice oracle."""
    if vectors.ndim != 2 or vectors.shape[1] != 2:
        raise ValueError("block-lattice probe values must have shape (pairs, 2)")
    if coordinate_bits < 2 or coordinate_bits > 8:
        raise ValueError("block-lattice coordinate bits must be in [2, 8]")
    transformed = _signed_block_fht(
        vectors,
        block_size=block_size,
        seed=seed,
    )
    source_energy = vectors.square().sum().clamp_min(1e-30)
    transformed_energy = transformed.square().sum()
    gains = transformed.square().mean(dim=1).sqrt().clamp_min(1e-30)
    gain_levels, gain_codes = _fit_scalar_codebook(
        gains.log(),
        level_count=256,
    )
    decoded_gains = gain_levels.index_select(0, gain_codes).exp()
    normalized = (transformed / decoded_gains[:, None]).reshape(-1)
    coordinate_levels, coordinate_codes = _fit_scalar_codebook(
        normalized,
        level_count=1 << coordinate_bits,
    )
    decoded = coordinate_levels.index_select(
        0, coordinate_codes
    ).reshape_as(transformed) * decoded_gains[:, None]
    error_energy = (transformed - decoded).square().sum()

    def entropy_bits(codes: torch.Tensor, level_count: int) -> float:
        counts = torch.bincount(codes, minlength=level_count).to(torch.float32)
        probabilities = counts[counts > 0] / counts.sum().clamp_min(1.0)
        return float(-(probabilities * probabilities.log2()).sum())

    return {
        "full_recovery": float(1.0 - error_energy / source_energy),
        "coordinate_active_codes": int(coordinate_codes.unique().numel()),
        "coordinate_entropy_bits": entropy_bits(
            coordinate_codes, 1 << coordinate_bits
        ),
        "gain_active_codes": int(gain_codes.unique().numel()),
        "gain_entropy_bits": entropy_bits(gain_codes, 256),
        "parseval_relative_error": float(
            (transformed_energy - source_energy).abs() / source_energy
        ),
        "physical_bits_per_weight": float(
            coordinate_bits + 8.0 / block_size
        ),
    }


@torch.no_grad()
def _block_gain_axis_adaptation_counterfactual(
    vectors: torch.Tensor,
    *,
    block_size: int,
    coordinate_bits: int,
    seed: int,
) -> dict[str, float | int]:
    """Separate axis-codebook waste from transform-gauge mismatch.

    This is deliberately an acquisition oracle.  The dense KLT is never
    registered as a production codec; it supplies a ceiling that can justify
    (or reject) a later fast learned-butterfly approximation.
    """
    if vectors.ndim != 2 or vectors.shape[1] != 2:
        raise ValueError("axis-adaptation probe values must have shape (pairs, 2)")
    if block_size < 2 or block_size & (block_size - 1):
        raise ValueError("axis-adaptation block size must be a power of two")
    if vectors.numel() % block_size:
        raise ValueError("axis-adaptation block size must divide the source")
    if coordinate_bits < 2 or coordinate_bits > 8:
        raise ValueError("axis-adaptation coordinate bits must be in [2, 8]")

    source = vectors.reshape(-1, block_size).to(torch.float32)
    source_energy = source.square().sum().clamp_min(1e-30)
    level_count = 1 << coordinate_bits

    def entropy_bits(codes: torch.Tensor, levels: int) -> float:
        counts = torch.bincount(codes.reshape(-1), minlength=levels).to(
            torch.float32
        )
        probabilities = counts[counts > 0] / counts.sum().clamp_min(1.0)
        return float(-(probabilities * probabilities.log2()).sum())

    def quantize(
        transformed: torch.Tensor,
        *,
        axis_private: bool,
    ) -> tuple[torch.Tensor, dict[str, float | int]]:
        gains = transformed.square().mean(dim=1).sqrt().clamp_min(1e-30)
        gain_levels, gain_codes = _fit_scalar_codebook(
            gains.log(), level_count=256
        )
        decoded_gains = gain_levels.index_select(0, gain_codes).exp()
        normalized = transformed / decoded_gains[:, None]
        if axis_private:
            decoded_axes: list[torch.Tensor] = []
            active: list[int] = []
            entropies: list[float] = []
            for axis in range(block_size):
                levels, codes = _fit_scalar_codebook(
                    normalized[:, axis], level_count=level_count
                )
                decoded_axes.append(levels.index_select(0, codes))
                active.append(int(codes.unique().numel()))
                entropies.append(entropy_bits(codes, level_count))
            decoded_normalized = torch.stack(decoded_axes, dim=1)
            coordinate_active_min = min(active)
            coordinate_entropy_min = min(entropies)
        else:
            coordinate_levels, coordinate_codes = _fit_scalar_codebook(
                normalized.reshape(-1), level_count=level_count
            )
            decoded_normalized = coordinate_levels.index_select(
                0, coordinate_codes
            ).reshape_as(normalized)
            coordinate_active_min = int(coordinate_codes.unique().numel())
            coordinate_entropy_min = entropy_bits(
                coordinate_codes, level_count
            )
        decoded = decoded_normalized * decoded_gains[:, None]
        return decoded, {
            "coordinate_active_codes_min": coordinate_active_min,
            "coordinate_entropy_bits_min": coordinate_entropy_min,
            "gain_active_codes": int(gain_codes.unique().numel()),
            "gain_entropy_bits": entropy_bits(gain_codes, 256),
        }

    fht = _signed_block_fht(
        vectors, block_size=block_size, seed=seed
    )
    center = source.mean(dim=0, keepdim=True)
    centered = source - center
    # This is a geometry oracle, so TF32 roundoff must not masquerade as KLT
    # distortion.  Re-orthogonalize the eigensystem and score decoded KLT
    # vectors after mapping them back to the ambient source coordinates.
    previous_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        covariance = centered.T @ centered / max(centered.shape[0], 1)
        _eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        eigenvectors, _upper = torch.linalg.qr(eigenvectors)
        klt = centered @ eigenvectors
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_allow_tf32

    output: dict[str, float | int] = {
        "physical_bits_per_weight": float(
            coordinate_bits + 8.0 / block_size
        ),
        "fht_parseval_relative_error": float(
            (fht.square().sum() - source_energy).abs() / source_energy
        ),
        "klt_parseval_relative_error": float(
            (klt.square().sum() - centered.square().sum()).abs()
            / centered.square().sum().clamp_min(1e-30)
        ),
        "source_mean_energy_ratio": float(
            center.square().sum() * source.shape[0] / source_energy
        ),
    }
    for name, transformed, axis_private in (
        ("fht_global", fht, False),
        ("fht_axis", fht, True),
        ("klt_global", klt, False),
        ("klt_axis", klt, True),
    ):
        decoded, diagnostics = quantize(
            transformed, axis_private=axis_private
        )
        if name.startswith("klt"):
            previous_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            try:
                reconstructed = decoded @ eigenvectors.T + center
            finally:
                torch.backends.cuda.matmul.allow_tf32 = previous_allow_tf32
            error_energy = (source - reconstructed).square().sum()
        else:
            error_energy = (transformed - decoded).square().sum()
        output[f"{name}_full_recovery"] = float(
            1.0 - error_energy / source_energy
        )
        for key, value in diagnostics.items():
            output[f"{name}_{key}"] = value

    fht_variance = fht.var(dim=0, unbiased=False)
    klt_variance = klt.var(dim=0, unbiased=False)
    output["fht_axis_variance_cv"] = float(
        fht_variance.std(unbiased=False)
        / fht_variance.mean().clamp_min(1e-30)
    )
    output["klt_axis_variance_cv"] = float(
        klt_variance.std(unbiased=False)
        / klt_variance.mean().clamp_min(1e-30)
    )
    return output


@torch.no_grad()
def _block_fht_fractional_lattice_counterfactual(
    vectors: torch.Tensor,
    *,
    block_size: int,
    base_coordinate_bits: int,
    refinement_fractions: tuple[float, ...],
    seed: int,
) -> dict[str, float | int]:
    """Measure block-selected q_b/q_(b+1) variable-rate refinement."""
    if vectors.ndim != 2 or vectors.shape[1] != 2:
        raise ValueError("fractional-lattice values must have shape (pairs, 2)")
    if block_size < 2 or block_size & (block_size - 1):
        raise ValueError("fractional-lattice block size must be a power of two")
    if vectors.numel() % block_size:
        raise ValueError("fractional-lattice block size must divide the source")
    if base_coordinate_bits < 2 or base_coordinate_bits >= 8:
        raise ValueError("fractional-lattice base bits must be in [2, 7]")
    if not refinement_fractions or any(
        fraction <= 0.0 or fraction >= 1.0
        for fraction in refinement_fractions
    ):
        raise ValueError("fractional-lattice fractions must be in (0, 1)")

    transformed = _signed_block_fht(
        vectors, block_size=block_size, seed=seed
    )
    source_energy = vectors.square().sum().clamp_min(1e-30)
    gains = transformed.square().mean(dim=1).sqrt().clamp_min(1e-30)
    gain_levels, gain_codes = _fit_scalar_codebook(
        gains.log(), level_count=256
    )
    decoded_gains = gain_levels.index_select(0, gain_codes).exp()
    normalized = (transformed / decoded_gains[:, None]).reshape(-1)

    base_levels, base_codes = _fit_scalar_codebook(
        normalized, level_count=1 << base_coordinate_bits
    )
    refined_levels, refined_codes = _fit_scalar_codebook(
        normalized, level_count=1 << (base_coordinate_bits + 1)
    )
    base_decoded = base_levels.index_select(0, base_codes).reshape_as(
        transformed
    ) * decoded_gains[:, None]
    refined_decoded = refined_levels.index_select(
        0, refined_codes
    ).reshape_as(transformed) * decoded_gains[:, None]
    base_block_error = (transformed - base_decoded).square().sum(dim=1)
    refined_block_error = (transformed - refined_decoded).square().sum(dim=1)
    improvement = base_block_error - refined_block_error

    def entropy_bits(codes: torch.Tensor, levels: int) -> float:
        counts = torch.bincount(codes.reshape(-1), minlength=levels).to(
            torch.float32
        )
        probabilities = counts[counts > 0] / counts.sum().clamp_min(1.0)
        return float(-(probabilities * probabilities.log2()).sum())

    output: dict[str, float | int] = {
        "base_full_recovery": float(
            1.0 - base_block_error.sum() / source_energy
        ),
        "uniform_refined_full_recovery": float(
            1.0 - refined_block_error.sum() / source_energy
        ),
        "base_active_codes": int(base_codes.unique().numel()),
        "refined_active_codes": int(refined_codes.unique().numel()),
        "base_entropy_bits": entropy_bits(
            base_codes, 1 << base_coordinate_bits
        ),
        "refined_entropy_bits": entropy_bits(
            refined_codes, 1 << (base_coordinate_bits + 1)
        ),
        "gain_active_codes": int(gain_codes.unique().numel()),
        "gain_entropy_bits": entropy_bits(gain_codes, 256),
        "parseval_relative_error": float(
            (transformed.square().sum() - source_energy).abs()
            / source_energy
        ),
    }
    for fraction in refinement_fractions:
        selected_count = int(round(float(fraction) * transformed.shape[0]))
        selected_count = min(max(selected_count, 1), transformed.shape[0] - 1)
        selected = torch.topk(
            improvement, selected_count, largest=True, sorted=False
        ).indices
        mask = torch.zeros(
            transformed.shape[0], device=transformed.device, dtype=torch.bool
        )
        mask[selected] = True
        decoded = torch.where(
            mask[:, None], refined_decoded, base_decoded
        )
        actual_fraction = float(mask.to(torch.float32).mean())
        token = str(float(fraction)).replace(".", "p")
        output[f"p{token}_full_recovery"] = float(
            1.0 - (transformed - decoded).square().sum() / source_energy
        )
        output[f"p{token}_selected_fraction"] = actual_fraction
        output[f"p{token}_physical_bits_per_weight"] = float(
            base_coordinate_bits
            + actual_fraction
            + (1.0 + 8.0) / block_size
        )
        output[f"p{token}_selected_improvement_fraction"] = float(
            improvement.index_select(0, selected).sum()
            / improvement.clamp_min(0.0).sum().clamp_min(1e-30)
        )
    return output


@torch.no_grad()
def _conditional_polar_pair_diagnostics(
    vectors: torch.Tensor,
    radial_levels: torch.Tensor,
    center: torch.Tensor,
    codes: torch.Tensor,
) -> dict[str, float | int]:
    """Decompose conditional-polar error and report empirical state usage."""
    if vectors.ndim != 2 or vectors.shape[1] != 2:
        raise ValueError("conditional polar diagnostic values must have shape (pairs, 2)")
    if radial_levels.ndim != 2 or radial_levels.numel() != 256:
        raise ValueError(
            "conditional polar radial levels must be a two-dimensional "
            "256-entry table"
        )
    angle_count, radius_count = radial_levels.shape
    selected = codes.long()
    angle_indices = selected % angle_count
    radius_indices = selected // angle_count
    centered = vectors - center
    directions = _polar_directions(
        device=vectors.device, angle_count=angle_count
    ).index_select(0, angle_indices)
    projected_radii = (centered * directions).sum(dim=1).clamp_min_(0.0)
    angular_residual = centered - projected_radii[:, None] * directions
    fitted_radii = radial_levels[angle_indices, radius_indices]
    radial_residual = projected_radii - fitted_radii
    angular_error = angular_residual.square().sum()
    radial_error = radial_residual.square().sum()
    quantization_error = (vectors - (center + fitted_radii[:, None] * directions)).square().sum()
    target_energy = vectors.square().sum().clamp_min(1e-30)
    projected_energy = projected_radii.square().sum().clamp_min(1e-30)
    quantization_energy = quantization_error.clamp_min(1e-30)

    def entropy_bits(indices: torch.Tensor, bins: int) -> float:
        counts = torch.bincount(indices, minlength=bins).to(torch.float32)
        probabilities = counts[counts > 0] / counts.sum().clamp_min(1.0)
        return float(-(probabilities * probabilities.log2()).sum())

    code_counts = torch.bincount(selected, minlength=256)
    angle_counts = torch.bincount(angle_indices, minlength=angle_count)
    radius_counts = torch.bincount(radius_indices, minlength=radius_count)
    angle_squares = torch.zeros(
        angle_count, device=vectors.device, dtype=torch.float32
    )
    angle_squares.index_add_(0, angle_indices, projected_radii.square())
    rms_by_angle = (
        angle_squares / angle_counts.clamp_min(1).to(torch.float32)
    ).sqrt()
    live_angle_rms = rms_by_angle[angle_counts > 0]
    top_radius = radius_indices == radius_count - 1
    decomposition_error = (quantization_error - angular_error - radial_error).abs()
    joint_entropy = entropy_bits(selected, 256)
    angle_entropy = entropy_bits(angle_indices, angle_count)
    return {
        "feedback_polar_angular_error_fraction": float(
            angular_error / quantization_energy
        ),
        "feedback_polar_radial_error_fraction": float(
            radial_error / quantization_energy
        ),
        "feedback_polar_angular_distortion": float(angular_error / target_energy),
        "feedback_polar_radial_distortion": float(radial_error / target_energy),
        "feedback_polar_decomposition_relative_error": float(
            decomposition_error / quantization_energy
        ),
        "feedback_polar_radial_recovery_given_angle": float(
            1.0 - radial_error / projected_energy
        ),
        "feedback_code_entropy_bits": joint_entropy,
        "feedback_angle_entropy_bits": angle_entropy,
        "feedback_conditional_radius_entropy_bits": joint_entropy - angle_entropy,
        "feedback_radius_entropy_bits": entropy_bits(
            radius_indices, radius_count
        ),
        "feedback_active_codes": int((code_counts > 0).sum()),
        "feedback_active_angles": int((angle_counts > 0).sum()),
        "feedback_active_radii": int((radius_counts > 0).sum()),
        "feedback_top_radius_fraction": float(top_radius.to(torch.float32).mean()),
        "feedback_top_radius_energy_fraction": float(
            projected_radii[top_radius].square().sum() / projected_energy
        ),
        "feedback_radial_rms_by_angle_cv": float(
            live_angle_rms.std(unbiased=False)
            / live_angle_rms.mean().clamp_min(1e-30)
        ),
        "feedback_projected_radius_max_to_rms": float(
            projected_radii.max() / projected_radii.square().mean().sqrt().clamp_min(1e-30)
        ),
    }


@torch.no_grad()
def _nearest_small_vector_codes(
    vectors: torch.Tensor, codebook: torch.Tensor
) -> torch.Tensor:
    """Exact nearest assignment to a small learned two-vector codebook."""
    if vectors.ndim != 2 or vectors.shape[1] != 2:
        raise ValueError("RVQ pair values must have shape (pairs, 2)")
    if codebook.ndim != 2 or codebook.shape[1] != 2:
        raise ValueError("RVQ codebook must have shape (entries, 2)")
    parts = []
    code_energy = codebook.square().sum(dim=1)
    for start in range(0, vectors.shape[0], 65536):
        stop = min(start + 65536, vectors.shape[0])
        values = vectors[start:stop]
        distances = (
            values.square().sum(dim=1, keepdim=True)
            + code_energy[None, :]
            - 2.0 * values @ codebook.T
        )
        parts.append(distances.argmin(dim=1))
    return torch.cat(parts)


@torch.no_grad()
def _initialize_rvq_codebook_(
    vectors: torch.Tensor, codebook: torch.Tensor
) -> None:
    """Initialize 16 joint atoms from a covariance-matched 4x4 PCA grid."""
    if tuple(codebook.shape) != (16, 2):
        raise ValueError("RVQ stage codebook must have shape (16, 2)")
    covariance = vectors.T @ vectors / max(vectors.shape[0], 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    scales = eigenvalues.clamp_min(0.0).sqrt()
    quantiles = torch.tensor(
        [-1.15034938, -0.31863936, 0.31863936, 1.15034938],
        device=vectors.device,
        dtype=torch.float32,
    )
    first, second = torch.meshgrid(quantiles, quantiles, indexing="ij")
    principal = torch.stack((first.reshape(-1), second.reshape(-1)), dim=1)
    codebook.copy_((principal * scales) @ eigenvectors.T)


@torch.no_grad()
def _lloyd_vector_stage_(
    vectors: torch.Tensor,
    codebook: torch.Tensor,
    *,
    iterations: int,
) -> torch.Tensor:
    assignments = torch.zeros(
        vectors.shape[0], device=vectors.device, dtype=torch.long
    )
    for _iteration in range(iterations):
        assignments = _nearest_small_vector_codes(vectors, codebook)
        sums = torch.zeros_like(codebook)
        sums.index_add_(0, assignments, vectors)
        counts = torch.bincount(assignments, minlength=codebook.shape[0])
        live = counts > 0
        codebook[live] = sums[live] / counts[live, None]
    return assignments


def _decode_rvq_pair_codec(
    codebooks: torch.Tensor,
    center: torch.Tensor,
    codes: torch.Tensor,
) -> torch.Tensor:
    if tuple(codebooks.shape) != (2, 16, 2):
        raise ValueError("RVQ codebooks must have shape (2, 16, 2)")
    if tuple(center.shape) != (2,):
        raise ValueError("RVQ center must have shape (2,)")
    selected = codes.long()
    return (
        center
        + codebooks[0].index_select(0, selected // 16)
        + codebooks[1].index_select(0, selected % 16)
    )


@torch.no_grad()
def _fit_rvq_pair_codec_(
    vectors: torch.Tensor,
    codebooks: torch.Tensor,
    center: torch.Tensor,
    codes: torch.Tensor,
) -> int:
    """Fit two learned four-bit 2D residual-vector stages."""
    if vectors.ndim != 2 or vectors.shape[1] != 2:
        raise ValueError("RVQ pair values must have shape (pairs, 2)")
    if tuple(codebooks.shape) != (2, 16, 2):
        raise ValueError("RVQ codebooks must have shape (2, 16, 2)")
    if tuple(center.shape) != (2,):
        raise ValueError("RVQ center must have shape (2,)")
    if codes.ndim != 1 or codes.numel() != vectors.shape[0]:
        raise ValueError("RVQ pair codes have the wrong shape")

    old_codes = codes.clone()
    fitted_center = vectors.mean(dim=0)
    centered = vectors - fitted_center
    initialized = bool(codebooks.abs().sum() > 0)
    if not initialized:
        _initialize_rvq_codebook_(centered, codebooks[0])
        first = _lloyd_vector_stage_(centered, codebooks[0], iterations=2)
        residual = centered - codebooks[0].index_select(0, first)
        _initialize_rvq_codebook_(residual, codebooks[1])
        second = _lloyd_vector_stage_(residual, codebooks[1], iterations=2)
    else:
        # Preserve the previously decoded vectors when moving the explicit
        # center to the current target mean.  Without this gauge transport,
        # every fit sees an artificial global translation.
        codebooks[0].add_(center - fitted_center)
        first = codes.long() // 16
        second = codes.long() % 16

    first_target = centered - codebooks[1].index_select(0, second)
    first = _lloyd_vector_stage_(first_target, codebooks[0], iterations=1)
    second_target = centered - codebooks[0].index_select(0, first)
    second = _lloyd_vector_stage_(second_target, codebooks[1], iterations=1)
    new_codes = (first * 16 + second).to(torch.uint8)
    center.copy_(fitted_center)
    codes.copy_(new_codes)
    return int((new_codes != old_codes).sum())


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
        stochastic_fast_retraction: bool = False,
        stochastic_fast_fht_block_size: int = 0,
        stochastic_fast_uniform_levels: bool = False,
        stochastic_fast_block_local_levels: bool = False,
        error_feedback: bool = False,
        forward_visible_feedback: bool = False,
        fp16_ambient_momentum: bool = False,
        fp16_reserved_escape_granularity: str = "",
        reserved_escape_scope: str = "",
        fp16_ambient_reference_probe_steps: tuple[int, ...] = (),
        feedback_codec: str = "cartesian4x4",
        feedback_output_group_size: int = 0,
        feedback_residual_probe_steps: tuple[int, ...] = (),
        feedback_residual_probe_layers: tuple[int, ...] = (),
        feedback_residual_probe_lloyd_iterations: tuple[int, ...] = (),
        feedback_transform_probe_block_sizes: tuple[int, ...] = (),
        feedback_lattice_probe_block_sizes: tuple[int, ...] = (),
        feedback_lattice_probe_coordinate_bits: tuple[int, ...] = (),
        feedback_axis_adaptation_probe_block_size: int = 0,
        feedback_axis_adaptation_probe_coordinate_bits: int = 7,
        feedback_fractional_probe_block_size: int = 0,
        feedback_fractional_probe_base_coordinate_bits: int = 7,
        feedback_fractional_probe_refinement_fractions: tuple[float, ...] = (),
        neighbor_candidates: int = 16,
        code_refresh_interval: int = 8,
        lazy_retraction_interval: int = 1,
        lazy_retraction_forced_steps: tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.stages = int(stages)
        self.layer_id = int(layer_id)
        self.fast_residual = bool(fast_residual)
        self.stochastic_fast_retraction = bool(stochastic_fast_retraction)
        self.stochastic_fast_fht_block_size = int(
            stochastic_fast_fht_block_size
        )
        self.stochastic_fast_uniform_levels = bool(
            stochastic_fast_uniform_levels
        )
        self.stochastic_fast_block_local_levels = bool(
            stochastic_fast_block_local_levels
        )
        self.base_seed = int(base_seed)
        self.error_feedback = bool(error_feedback)
        self.forward_visible_feedback = bool(forward_visible_feedback)
        self.fp16_ambient_momentum = bool(fp16_ambient_momentum)
        self.fp16_reserved_escape_granularity = str(
            fp16_reserved_escape_granularity
        )
        self.reserved_escape_scope = str(reserved_escape_scope)
        self.fp16_ambient_reference_probe_steps = tuple(
            int(step) for step in fp16_ambient_reference_probe_steps
        )
        self.feedback_codec = str(feedback_codec)
        self.feedback_output_group_size = int(feedback_output_group_size)
        self.feedback_residual_probe_steps = tuple(
            int(step) for step in feedback_residual_probe_steps
        )
        self.feedback_residual_probe_layers = tuple(
            int(layer) for layer in feedback_residual_probe_layers
        )
        self.feedback_residual_probe_lloyd_iterations = tuple(
            int(iterations)
            for iterations in feedback_residual_probe_lloyd_iterations
        )
        self.feedback_transform_probe_block_sizes = tuple(
            int(block_size) for block_size in feedback_transform_probe_block_sizes
        )
        self.feedback_lattice_probe_block_sizes = tuple(
            int(block_size) for block_size in feedback_lattice_probe_block_sizes
        )
        self.feedback_lattice_probe_coordinate_bits = tuple(
            int(bits) for bits in feedback_lattice_probe_coordinate_bits
        )
        self.feedback_axis_adaptation_probe_block_size = int(
            feedback_axis_adaptation_probe_block_size
        )
        self.feedback_axis_adaptation_probe_coordinate_bits = int(
            feedback_axis_adaptation_probe_coordinate_bits
        )
        self.feedback_fractional_probe_block_size = int(
            feedback_fractional_probe_block_size
        )
        self.feedback_fractional_probe_base_coordinate_bits = int(
            feedback_fractional_probe_base_coordinate_bits
        )
        self.feedback_fractional_probe_refinement_fractions = tuple(
            float(fraction)
            for fraction in feedback_fractional_probe_refinement_fractions
        )
        self.neighbor_candidates = int(neighbor_candidates)
        self.code_refresh_interval = int(code_refresh_interval)
        self.lazy_retraction_interval = int(lazy_retraction_interval)
        self.lazy_retraction_forced_steps = tuple(
            int(step) for step in lazy_retraction_forced_steps
        )
        if self.in_features <= 0 or self.out_features <= 0:
            raise ValueError("pair-VQ dimensions must be positive")
        if self.in_features * self.out_features % self.vector_length:
            raise ValueError("pair-VQ element count must be divisible by two")
        if self.stages not in (1, 2):
            raise ValueError("pair-VQ stages must be one or two")
        if self.stochastic_fast_retraction and not self.fast_residual:
            raise ValueError("stochastic fast retraction requires a fast residual")
        if self.forward_visible_feedback and not self.error_feedback:
            raise ValueError("forward-visible feedback requires error feedback")
        if self.fp16_ambient_reference_probe_steps and not self.fp16_ambient_momentum:
            raise ValueError(
                "FP16 ambient reference probes require FP16 ambient momentum"
            )
        if self.fp16_reserved_escape_granularity not in {
            "",
            "scope",
            "block",
            "adaptive_block",
        }:
            raise ValueError(
                "FP16 reserved-escape granularity must be '', 'scope', "
                "'block', or 'adaptive_block'"
            )
        if self.fp16_reserved_escape_granularity and (
            not self.fp16_ambient_momentum
            or self.reserved_escape_scope not in {"c_fc", "c_proj"}
            or self.element_count % 4096
        ):
            raise ValueError(
                "FP16 reserved-escape momentum requires FP16 ambient momentum, "
                "an explicit c_fc/c_proj scope, and a 4096-divisible matrix"
            )
        if any(step < 0 for step in self.fp16_ambient_reference_probe_steps):
            raise ValueError(
                "FP16 ambient reference probe steps must be nonnegative"
            )
        if self.stochastic_fast_fht_block_size < 0:
            raise ValueError("stochastic fast FHT block size must be nonnegative")
        if self.stochastic_fast_fht_block_size and (
            not self.stochastic_fast_retraction
            or self.stochastic_fast_fht_block_size < 2
            or self.stochastic_fast_fht_block_size
            & (self.stochastic_fast_fht_block_size - 1)
            or self.in_features * self.out_features
            % self.stochastic_fast_fht_block_size
        ):
            raise ValueError(
                "stochastic fast FHT block size requires stochastic retraction "
                "and must be a power-of-two divisor of the element count"
            )
        if self.stochastic_fast_uniform_levels and (
            not self.stochastic_fast_retraction
            or not self.stochastic_fast_fht_block_size
        ):
            raise ValueError(
                "uniform stochastic levels require FHT-preconditioned "
                "stochastic fast retraction"
            )
        if self.stochastic_fast_block_local_levels and (
            not self.stochastic_fast_uniform_levels
            or not self.stochastic_fast_fht_block_size
        ):
            raise ValueError(
                "block-local stochastic levels require FHT-preconditioned "
                "uniform stochastic retraction"
            )
        if self.feedback_output_group_size < 0:
            raise ValueError("feedback output group size must be nonnegative")
        if any(step < 0 for step in self.feedback_residual_probe_steps):
            raise ValueError("feedback residual probe steps must be nonnegative")
        if any(layer < 0 for layer in self.feedback_residual_probe_layers):
            raise ValueError("feedback residual probe layers must be nonnegative")
        if self.feedback_residual_probe_layers and not self.feedback_residual_probe_steps:
            raise ValueError("feedback residual probe layers require probe steps")
        if any(
            iterations <= 0
            for iterations in self.feedback_residual_probe_lloyd_iterations
        ):
            raise ValueError(
                "feedback residual probe Lloyd iterations must be positive"
            )
        fractional_residual_codecs = (
            "fractional_lattice_q7q8_b32_p25_rq4",
            "fractional_lattice_q7q8_b32_p25_rq4_cfcq5",
            "fractional_lattice_q7q8_b32_p25_rq4_cfcq5b64l16",
        )
        if self.feedback_residual_probe_steps and self.feedback_codec not in (
            "conditional_polar16x16_rvq2",
            "free_vq256_rvq2",
            *fractional_residual_codecs,
        ):
            raise ValueError(
                "feedback residual probes require a supported two-stage codec"
            )
        if (
            self.feedback_residual_probe_lloyd_iterations
            and self.feedback_codec
            not in ("conditional_polar16x16_rvq2", *fractional_residual_codecs)
        ):
            raise ValueError(
                "feedback residual Lloyd probes require a supported residual codec"
            )
        if any(
            block_size < 2
            or block_size & (block_size - 1)
            or self.element_count % block_size
            for block_size in self.feedback_transform_probe_block_sizes
        ):
            raise ValueError(
                "feedback transform probe sizes must be power-of-two divisors"
            )
        if self.feedback_transform_probe_block_sizes and (
            not self.feedback_residual_probe_steps
            or self.feedback_codec != "free_vq256_rvq2"
        ):
            raise ValueError(
                "feedback transform probes require free-VQ residual probe steps"
            )
        if any(
            block_size < 2
            or block_size & (block_size - 1)
            or self.element_count % block_size
            for block_size in self.feedback_lattice_probe_block_sizes
        ):
            raise ValueError(
                "feedback lattice probe sizes must be power-of-two divisors"
            )
        if any(
            bits < 2 or bits > 8
            for bits in self.feedback_lattice_probe_coordinate_bits
        ):
            raise ValueError("feedback lattice probe bits must be in [2, 8]")
        if bool(self.feedback_lattice_probe_block_sizes) != bool(
            self.feedback_lattice_probe_coordinate_bits
        ):
            raise ValueError(
                "feedback lattice probes require both block sizes and bit widths"
            )
        if self.feedback_lattice_probe_block_sizes and (
            not self.feedback_residual_probe_steps
            or self.feedback_codec
            not in ("free_vq256_rvq2", *fractional_residual_codecs)
        ):
            raise ValueError(
                "feedback lattice probes require free-VQ residual probe steps"
            )
        if self.feedback_axis_adaptation_probe_block_size < 0:
            raise ValueError("feedback axis-adaptation block size must be nonnegative")
        if self.feedback_axis_adaptation_probe_block_size and (
            self.feedback_axis_adaptation_probe_block_size < 2
            or self.feedback_axis_adaptation_probe_block_size
            & (self.feedback_axis_adaptation_probe_block_size - 1)
            or self.element_count
            % self.feedback_axis_adaptation_probe_block_size
        ):
            raise ValueError(
                "feedback axis-adaptation block size must be a power-of-two divisor"
            )
        if not 2 <= self.feedback_axis_adaptation_probe_coordinate_bits <= 8:
            raise ValueError(
                "feedback axis-adaptation coordinate bits must be in [2, 8]"
            )
        if self.feedback_axis_adaptation_probe_block_size and (
            not self.feedback_residual_probe_steps
            or self.feedback_codec
            not in ("free_vq256_rvq2", *fractional_residual_codecs)
        ):
            raise ValueError(
                "feedback axis-adaptation probes require free-VQ residual probe steps"
            )
        if self.feedback_fractional_probe_block_size < 0:
            raise ValueError("feedback fractional-probe block size must be nonnegative")
        if self.feedback_fractional_probe_block_size and (
            self.feedback_fractional_probe_block_size < 2
            or self.feedback_fractional_probe_block_size
            & (self.feedback_fractional_probe_block_size - 1)
            or self.element_count % self.feedback_fractional_probe_block_size
        ):
            raise ValueError(
                "feedback fractional-probe block size must be a power-of-two divisor"
            )
        if not 2 <= self.feedback_fractional_probe_base_coordinate_bits < 8:
            raise ValueError("feedback fractional-probe base bits must be in [2, 7]")
        if any(
            fraction <= 0.0 or fraction >= 1.0
            for fraction in self.feedback_fractional_probe_refinement_fractions
        ):
            raise ValueError("feedback fractional-probe fractions must be in (0, 1)")
        if bool(self.feedback_fractional_probe_block_size) != bool(
            self.feedback_fractional_probe_refinement_fractions
        ):
            raise ValueError(
                "feedback fractional probes require a block size and fractions"
            )
        if self.feedback_fractional_probe_block_size and (
            not self.feedback_residual_probe_steps
            or self.feedback_codec != "free_vq256_rvq2"
        ):
            raise ValueError(
                "feedback fractional probes require free-VQ residual probe steps"
            )
        if self.feedback_codec not in (
            "cartesian4x4",
            "polar32x8",
            "conditional_polar32x8",
            "conditional_polar16x16",
            "conditional_polar16x16_rvq2",
            "free_vq256_rvq2",
            "rvq4x4",
            "fractional_lattice_q7q8_b32_p25",
            "fractional_lattice_q7q8_b32_p25_rq4",
            "fractional_lattice_q7q8_b32_p25_rq4_cfcq5",
            "fractional_lattice_q7q8_b32_p25_rq4_cfcq5b64l16",
        ):
            raise ValueError("unknown pair-VQ feedback codec")
        if (
            self.feedback_codec in (
                "polar32x8",
                "conditional_polar32x8",
                "conditional_polar16x16",
                "conditional_polar16x16_rvq2",
                "free_vq256_rvq2",
                "rvq4x4",
                "fractional_lattice_q7q8_b32_p25",
                "fractional_lattice_q7q8_b32_p25_rq4",
                "fractional_lattice_q7q8_b32_p25_rq4_cfcq5",
                "fractional_lattice_q7q8_b32_p25_rq4_cfcq5b64l16",
            )
            and self.feedback_output_group_size != 0
        ):
            raise ValueError("joint pair feedback currently requires matrix-global levels")
        if self.feedback_output_group_size and (
            self.out_features % self.feedback_output_group_size
            or self.in_features % self.vector_length
        ):
            raise ValueError(
                "grouped feedback requires an even input width and an output "
                "width divisible by the group size"
            )
        if not 1 <= self.neighbor_candidates <= self.codebook_size:
            raise ValueError("pair-VQ neighbor count is invalid")
        if self.code_refresh_interval <= 0:
            raise ValueError("pair-VQ refresh interval must be positive")
        if self.lazy_retraction_interval <= 0:
            raise ValueError("pair-VQ lazy retraction interval must be positive")
        if any(step <= 0 for step in self.lazy_retraction_forced_steps):
            raise ValueError("pair-VQ forced retraction steps must be positive")
        if tuple(sorted(set(self.lazy_retraction_forced_steps))) != (
            self.lazy_retraction_forced_steps
        ):
            raise ValueError(
                "pair-VQ forced retraction steps must be sorted and unique"
            )
        if self.lazy_retraction_interval > 1 and (
            not self.forward_visible_feedback
            or not self.fp16_ambient_momentum
            or self.code_refresh_interval != self.lazy_retraction_interval
        ):
            raise ValueError(
                "lazy Pair-VQ retraction requires forward-visible feedback, "
                "FP16 ambient momentum, and a matching code-refresh interval"
            )
        if self.lazy_retraction_forced_steps and self.lazy_retraction_interval == 1:
            raise ValueError(
                "forced Pair-VQ retraction steps require a lazy interval"
            )
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
            "fast_group_bounds",
            (
                torch.zeros(
                    self.element_count // self.stochastic_fast_fht_block_size,
                    2,
                    2,
                    dtype=torch.float32,
                )
                if self.fast_residual
                and self.stochastic_fast_block_local_levels
                else torch.empty(0, dtype=torch.float32)
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
        self._last_stochastic_fast_diagnostics: dict[str, float] = {}

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
            + self.fast_group_bounds.numel()
            * self.fast_group_bounds.element_size()
            + self.optimizer_step.numel() * self.optimizer_step.element_size()
        )

    @property
    def compact_momentum_bytes(self) -> int:
        return self.codebooks.numel() * torch.tensor([], dtype=torch.float32).element_size()

    @property
    def ambient_momentum_bytes(self) -> int:
        if not self.fp16_ambient_momentum:
            return 0
        if self.fp16_reserved_escape_granularity:
            return math.ceil(self.element_count * 14 / 8)
        return (
            self.element_count
            * torch.tensor([], dtype=torch.float16).element_size()
        )

    @property
    def optimizer_momentum_bytes(self) -> int:
        if self.fp16_ambient_momentum:
            return self.ambient_momentum_bytes
        return self.compact_momentum_bytes

    def is_compact_boundary(self) -> bool:
        """Whether persistent codes/feedback reproduce the materialized weight."""
        step = int(self.optimizer_step)
        return (
            self.lazy_retraction_interval == 1
            or step == 0
            or step % self.lazy_retraction_interval == 0
            or step in self.lazy_retraction_forced_steps
        )

    @property
    def feedback_residual_lattice_coordinate_bits(self) -> int:
        if (
            self.feedback_codec
            in (
                "fractional_lattice_q7q8_b32_p25_rq4_cfcq5",
                "fractional_lattice_q7q8_b32_p25_rq4_cfcq5b64l16",
            )
            and self.out_features > self.in_features
        ):
            return 5
        return 4

    @property
    def feedback_residual_lattice_block_size(self) -> int:
        if (
            self.feedback_codec
            == "fractional_lattice_q7q8_b32_p25_rq4_cfcq5b64l16"
            and self.out_features > self.in_features
        ):
            return 64
        return 32

    @property
    def feedback_residual_lattice_lloyd_iterations(self) -> int:
        if (
            self.feedback_codec
            == "fractional_lattice_q7q8_b32_p25_rq4_cfcq5b64l16"
            and self.out_features > self.in_features
        ):
            return 16
        return 4

    @property
    def compact_feedback_bytes(self) -> int:
        if not self.error_feedback:
            return 0
        pair_codes = self.element_count // self.vector_length
        if self.feedback_codec == "polar32x8":
            metadata_values = 8 + 2
        elif self.feedback_codec in (
            "conditional_polar32x8",
            "conditional_polar16x16",
        ):
            metadata_values = 256 + 2
        elif self.feedback_codec == "conditional_polar16x16_rvq2":
            metadata_values = 2 * (256 + 2)
        elif self.feedback_codec == "free_vq256_rvq2":
            metadata_values = 2 * 256 * 2
        elif self.feedback_codec == "rvq4x4":
            metadata_values = 2 * 16 * 2 + 2
        elif self.feedback_codec == "fractional_lattice_q7q8_b32_p25":
            return (
                _fractional_lattice_feedback_layout(self.element_count)[
                    "total_bytes"
                ]
                + 640 * torch.tensor([], dtype=torch.float32).element_size()
            )
        elif self.feedback_codec in (
            "fractional_lattice_q7q8_b32_p25_rq4",
            "fractional_lattice_q7q8_b32_p25_rq4_cfcq5",
            "fractional_lattice_q7q8_b32_p25_rq4_cfcq5b64l16",
        ):
            coordinate_bits = self.feedback_residual_lattice_coordinate_bits
            return (
                _fractional_residual_lattice_feedback_layout(
                    self.element_count,
                    coordinate_bits=coordinate_bits,
                    block_size=self.feedback_residual_lattice_block_size,
                )[
                    "total_bytes"
                ]
                + (896 + (1 << coordinate_bits))
                * torch.tensor([], dtype=torch.float32).element_size()
            )
        else:
            metadata_values = self.feedback_group_count * 2 * 16
        level_bytes = metadata_values * torch.tensor(
            [], dtype=torch.float32
        ).element_size()
        code_stages = (
            2
            if self.feedback_codec
            in ("conditional_polar16x16_rvq2", "free_vq256_rvq2")
            else 1
        )
        return (
            code_stages
            * pair_codes
            * torch.tensor([], dtype=torch.uint8).element_size()
            + level_bytes
        )

    @property
    def feedback_group_count(self) -> int:
        if self.feedback_output_group_size == 0:
            return 1
        return self.out_features // self.feedback_output_group_size

    @property
    def feedback_pairs_per_group(self) -> int:
        return (
            self.element_count
            // self.vector_length
            // self.feedback_group_count
        )

    @property
    def feedback_level_shape(self) -> tuple[int, ...]:
        if self.feedback_codec == "polar32x8":
            return (8,)
        if self.feedback_codec == "conditional_polar32x8":
            return (32, 8)
        if self.feedback_codec == "conditional_polar16x16":
            return (16, 16)
        if self.feedback_codec == "conditional_polar16x16_rvq2":
            return (2, 16, 16)
        if self.feedback_codec == "free_vq256_rvq2":
            return (2, 256, 2)
        if self.feedback_codec == "rvq4x4":
            return (2, 16, 2)
        if self.feedback_codec == "fractional_lattice_q7q8_b32_p25":
            return (640,)
        if self.feedback_codec in (
            "fractional_lattice_q7q8_b32_p25_rq4",
            "fractional_lattice_q7q8_b32_p25_rq4_cfcq5",
            "fractional_lattice_q7q8_b32_p25_rq4_cfcq5b64l16",
        ):
            return (896 + (1 << self.feedback_residual_lattice_coordinate_bits),)
        if self.feedback_output_group_size == 0:
            return (2, 16)
        return (self.feedback_group_count, 2, 16)

    @property
    def feedback_center_shape(self) -> tuple[int, ...] | None:
        if self.feedback_codec == "conditional_polar16x16_rvq2":
            return (2, 2)
        if self.feedback_codec in (
            "polar32x8",
            "conditional_polar32x8",
            "conditional_polar16x16",
            "rvq4x4",
        ):
            return (2,)
        return None

    @property
    def feedback_code_shape(self) -> tuple[int, ...]:
        if self.feedback_codec == "fractional_lattice_q7q8_b32_p25":
            return (
                _fractional_lattice_feedback_layout(self.element_count)[
                    "total_bytes"
                ],
            )
        if self.feedback_codec in (
            "fractional_lattice_q7q8_b32_p25_rq4",
            "fractional_lattice_q7q8_b32_p25_rq4_cfcq5",
            "fractional_lattice_q7q8_b32_p25_rq4_cfcq5b64l16",
        ):
            return (
                _fractional_residual_lattice_feedback_layout(
                    self.element_count,
                    coordinate_bits=self.feedback_residual_lattice_coordinate_bits,
                    block_size=self.feedback_residual_lattice_block_size,
                )[
                    "total_bytes"
                ],
            )
        pair_count = self.element_count // self.vector_length
        if self.feedback_codec in (
            "conditional_polar16x16_rvq2",
            "free_vq256_rvq2",
        ):
            return (2, pair_count)
        return (pair_count,)

    @property
    def feedback_fractional_lattice_seed(self) -> int:
        return (
            20261025
            + 8192 * self.layer_id
            + 131 * self.in_features
            + 17 * self.out_features
        )

    def decode_feedback(
        self,
        levels: torch.Tensor,
        codes: torch.Tensor,
        center: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.feedback_codec == "polar32x8":
            if center is None:
                raise ValueError("polar pair feedback requires a center")
            return _decode_polar_pair_codec(levels, center, codes)
        if self.feedback_codec in (
            "conditional_polar32x8",
            "conditional_polar16x16",
        ):
            if center is None:
                raise ValueError("conditional polar pair feedback requires a center")
            return _decode_conditional_polar_pair_codec(levels, center, codes)
        if self.feedback_codec == "conditional_polar16x16_rvq2":
            if center is None:
                raise ValueError(
                    "residual conditional-polar feedback requires centers"
                )
            return _decode_residual_conditional_polar_pair_codec(
                levels, center, codes
            )
        if self.feedback_codec == "free_vq256_rvq2":
            return _decode_free_pair_vq_rvq2(levels, codes)
        if self.feedback_codec == "rvq4x4":
            if center is None:
                raise ValueError("RVQ pair feedback requires a center")
            return _decode_rvq_pair_codec(levels, center, codes)
        if self.feedback_codec == "fractional_lattice_q7q8_b32_p25":
            return _decode_fractional_lattice_feedback(
                levels,
                codes,
                element_count=self.element_count,
                seed=self.feedback_fractional_lattice_seed,
            )
        if self.feedback_codec in (
            "fractional_lattice_q7q8_b32_p25_rq4",
            "fractional_lattice_q7q8_b32_p25_rq4_cfcq5",
            "fractional_lattice_q7q8_b32_p25_rq4_cfcq5b64l16",
        ):
            return _decode_fractional_residual_lattice_feedback(
                levels,
                codes,
                element_count=self.element_count,
                seed=self.feedback_fractional_lattice_seed,
                coordinate_bits=self.feedback_residual_lattice_coordinate_bits,
                block_size=self.feedback_residual_lattice_block_size,
            )
        if self.feedback_output_group_size == 0:
            return _decode_cartesian_pair_codec(levels, codes)
        return _decode_grouped_cartesian_pair_codec(
            levels,
            codes,
            pairs_per_group=self.feedback_pairs_per_group,
        )

    @torch.no_grad()
    def fit_feedback_(
        self,
        vectors: torch.Tensor,
        levels: torch.Tensor,
        codes: torch.Tensor,
        center: torch.Tensor | None = None,
    ) -> int:
        if self.feedback_codec == "polar32x8":
            if center is None:
                raise ValueError("polar pair feedback requires a center")
            return _fit_polar_pair_codec_(vectors, levels, center, codes)
        if self.feedback_codec in (
            "conditional_polar32x8",
            "conditional_polar16x16",
        ):
            if center is None:
                raise ValueError("conditional polar pair feedback requires a center")
            return _fit_conditional_polar_pair_codec_(
                vectors, levels, center, codes
            )
        if self.feedback_codec == "conditional_polar16x16_rvq2":
            if center is None:
                raise ValueError(
                    "residual conditional-polar feedback requires centers"
                )
            return _fit_residual_conditional_polar_pair_codec_(
                vectors, levels, center, codes
            )
        if self.feedback_codec == "free_vq256_rvq2":
            return _fit_free_pair_vq_rvq2_(
                vectors,
                levels,
                codes,
                neighbor_candidates=self.neighbor_candidates,
            )
        if self.feedback_codec == "rvq4x4":
            if center is None:
                raise ValueError("RVQ pair feedback requires a center")
            return _fit_rvq_pair_codec_(vectors, levels, center, codes)
        if self.feedback_codec == "fractional_lattice_q7q8_b32_p25":
            return _fit_fractional_lattice_feedback_(
                vectors,
                levels,
                codes,
                seed=self.feedback_fractional_lattice_seed,
            )
        if self.feedback_codec in (
            "fractional_lattice_q7q8_b32_p25_rq4",
            "fractional_lattice_q7q8_b32_p25_rq4_cfcq5",
            "fractional_lattice_q7q8_b32_p25_rq4_cfcq5b64l16",
        ):
            return _fit_fractional_residual_lattice_feedback_(
                vectors,
                levels,
                codes,
                seed=self.feedback_fractional_lattice_seed,
                coordinate_bits=self.feedback_residual_lattice_coordinate_bits,
                block_size=self.feedback_residual_lattice_block_size,
                lloyd_iterations=self.feedback_residual_lattice_lloyd_iterations,
            )
        if self.feedback_output_group_size == 0:
            return _fit_cartesian_pair_codec_(vectors, levels, codes)
        return _fit_grouped_cartesian_pair_codec_(
            vectors,
            levels,
            codes,
            pairs_per_group=self.feedback_pairs_per_group,
        )

    @property
    def transient_weight_bytes(self) -> int:
        return self.weight.numel() * self.weight.element_size()

    def storage_accounting(self) -> dict[str, int | float | str]:
        dense_bf16 = self.element_count * 2
        dense_fp32_weight_and_momentum = self.element_count * 8
        persistent_training = (
            self.persistent_codec_bytes
            + self.optimizer_momentum_bytes
            + self.compact_feedback_bytes
        )
        return {
            "elements": self.element_count,
            "stages": self.stages,
            "fast_residual": self.fast_residual,
            "persistent_codec_bytes": self.persistent_codec_bytes,
            "compact_momentum_bytes": (
                0 if self.fp16_ambient_momentum else self.compact_momentum_bytes
            ),
            "ambient_momentum_bytes": self.ambient_momentum_bytes,
            "optimizer_momentum_bytes": self.optimizer_momentum_bytes,
            "compact_feedback_bytes": self.compact_feedback_bytes,
            "persistent_training_bytes": persistent_training,
            "model_compression_vs_dense_bf16": dense_bf16 / self.persistent_codec_bytes,
            "training_compression_vs_dense_fp32_weight_plus_momentum": (
                dense_fp32_weight_and_momentum / persistent_training
            ),
            "transient_materialized_weight_bytes": self.transient_weight_bytes,
            "transient_gradient_bytes": self.transient_weight_bytes,
            "dense_master_weight": "disabled",
            "dense_optimizer_momentum": (
                (
                    "fp16_reserved_escape_capacity_ceiling"
                    if self.fp16_reserved_escape_granularity
                    else "fp16_ambient"
                )
                if self.fp16_ambient_momentum
                else "disabled"
            ),
            "dense_ambient_error_buffer": "disabled",
            "forward_visible_feedback": self.forward_visible_feedback,
            "compact_temporal_carry": (
                (
                    (
                        "uint8_polar32x8_code_per_weight_pair"
                        if self.feedback_codec == "polar32x8"
                        else (
                            (
                                "uint8_conditional_polar32x8_code_per_weight_pair"
                                if self.feedback_codec == "conditional_polar32x8"
                                else "uint8_conditional_polar16x16_code_per_weight_pair"
                            )
                            if self.feedback_codec
                            in (
                                "conditional_polar32x8",
                                "conditional_polar16x16",
                            )
                            else (
                                (
                                    "two_uint8_residual_conditional_polar16x16_"
                                    "codes_per_weight_pair"
                                    if self.feedback_codec
                                    == "conditional_polar16x16_rvq2"
                                    else "two_uint8_free_vq256_codes_per_weight_pair"
                                )
                                if self.feedback_codec
                                in (
                                    "conditional_polar16x16_rvq2",
                                    "free_vq256_rvq2",
                                )
                                else (
                                    (
                                        (
                                            "packed_b32_q7_q8_p25_plus_"
                                            + (
                                                "q5_b64_lloyd16_residual"
                                                if self.feedback_residual_lattice_block_size
                                                == 64
                                                else (
                                                    "q5_residual"
                                                    if self.feedback_residual_lattice_coordinate_bits
                                                    == 5
                                                    else "q4_residual"
                                                )
                                            )
                                        )
                                        if self.feedback_codec
                                        in (
                                            "fractional_lattice_q7q8_b32_p25_rq4",
                                            "fractional_lattice_q7q8_b32_p25_rq4_cfcq5",
                                            "fractional_lattice_q7q8_b32_p25_rq4_cfcq5b64l16",
                                        )
                                        else "packed_b32_q7_q8_p25_plus_q8_gain"
                                    )
                                    if self.feedback_codec.startswith(
                                        "fractional_lattice_q7q8_b32_p25"
                                    )
                                    else "uint8_rvq4x4_code_per_weight_pair"
                                )
                            )
                        )
                    )
                    if self.feedback_codec
                    in (
                        "polar32x8",
                        "conditional_polar32x8",
                        "conditional_polar16x16",
                        "conditional_polar16x16_rvq2",
                        "free_vq256_rvq2",
                        "rvq4x4",
                        "fractional_lattice_q7q8_b32_p25",
                        "fractional_lattice_q7q8_b32_p25_rq4",
                        "fractional_lattice_q7q8_b32_p25_rq4_cfcq5",
                        "fractional_lattice_q7q8_b32_p25_rq4_cfcq5b64l16",
                    )
                    else (
                        "uint8_cartesian_code_per_weight_pair_global_levels"
                        if self.feedback_output_group_size == 0
                        else "uint8_cartesian_code_per_weight_pair_output_group_levels"
                    )
                )
                if self.error_feedback
                else "disabled"
            ),
            "feedback_codec": self.feedback_codec,
            "feedback_output_group_size": self.feedback_output_group_size,
            "feedback_group_count": self.feedback_group_count,
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
            pairs = pairs + self.decode_fast_pairs()
        return pairs.reshape(self.out_features, self.in_features).float()

    def decode_slow_pairs(self) -> torch.Tensor:
        return sum(self.decode_pairs(stage) for stage in range(self.stages))

    def decode_fast_pairs(self) -> torch.Tensor:
        if not self.fast_residual:
            return torch.zeros_like(self.decode_pairs(0))
        fast_codes = self.fast_codes.long()
        if self.stochastic_fast_block_local_levels:
            pairs_per_group = self.stochastic_fast_fht_block_size // 2
            grouped_codes = fast_codes.reshape(-1, pairs_per_group)
            low = self.fast_group_bounds[:, :, 0]
            step = (
                self.fast_group_bounds[:, :, 1] - low
            ) / 15.0
            encoded = torch.stack(
                (
                    low[:, 0, None]
                    + (grouped_codes // 16).float() * step[:, 0, None],
                    low[:, 1, None]
                    + (grouped_codes % 16).float() * step[:, 1, None],
                ),
                dim=2,
            ).reshape(-1, 2)
        else:
            encoded = torch.stack(
                (
                    self.fast_levels[0].index_select(0, fast_codes // 16),
                    self.fast_levels[1].index_select(0, fast_codes % 16),
                ),
                dim=1,
            )
        if not self.stochastic_fast_fht_block_size:
            return encoded
        return _inverse_signed_block_fht(
            encoded.reshape(-1, self.stochastic_fast_fht_block_size),
            seed=self.base_seed + 32452843,
        ).reshape(-1, self.vector_length)

    @torch.no_grad()
    def rematerialize_weight_(self) -> None:
        self.weight.copy_(self.decode_weight())

    @torch.no_grad()
    def rematerialize_forward_visible_weight_(
        self,
        levels: torch.Tensor,
        codes: torch.Tensor,
        center: torch.Tensor | None = None,
    ) -> None:
        """Rebuild the transient virtual center from compact persistent state."""
        self.rematerialize_weight_()
        if self.forward_visible_feedback:
            feedback = self.decode_feedback(levels, codes, center).reshape_as(
                self.weight
            )
            self.weight.add_(feedback.to(dtype=self.weight.dtype))

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
        candidate = accum / counts.clamp_min(1).to(dtype=accum.dtype)[:, None]
        self.codebooks[stage].copy_(
            torch.where(live[:, None], candidate, self.codebooks[stage])
        )

    @torch.no_grad()
    def _fit_fast_residual_(self, residual: torch.Tensor) -> int:
        if not self.fast_residual:
            return 0
        if self.stochastic_fast_retraction:
            codec_target = residual
            if self.stochastic_fast_fht_block_size:
                codec_target = _signed_block_fht(
                    residual,
                    block_size=self.stochastic_fast_fht_block_size,
                    seed=self.base_seed + 32452843,
                ).reshape(-1, self.vector_length)
            stochastic_seed = (
                self.base_seed
                + 104729 * int(self.optimizer_step)
                + 1000003
            )
            if self.stochastic_fast_block_local_levels:
                changes, diagnostics = (
                    _fit_block_local_uniform_stochastic_cartesian_pair_codec_(
                        codec_target,
                        self.fast_group_bounds,
                        self.fast_codes,
                        pairs_per_group=(
                            self.stochastic_fast_fht_block_size // 2
                        ),
                        seed=stochastic_seed,
                    )
                )
            else:
                changes, diagnostics = _fit_stochastic_cartesian_pair_codec_(
                    codec_target,
                    self.fast_levels,
                    self.fast_codes,
                    seed=stochastic_seed,
                    uniform_levels=self.stochastic_fast_uniform_levels,
                )
            diagnostics["stochastic_fast_fht_block_size"] = (
                self.stochastic_fast_fht_block_size
            )
            self._last_stochastic_fast_diagnostics = diagnostics
            return changes
        self._last_stochastic_fast_diagnostics = {}
        return _fit_cartesian_pair_codec_(
            residual, self.fast_levels, self.fast_codes
        )

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
        diagnostics.update(self._last_stochastic_fast_diagnostics)
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
    """Muon request with Pair-VQ projection and selected momentum coordinates."""

    def __init__(
        self,
        modules: list[MuonPairVQLinear],
        *,
        lr: float,
        momentum: float,
        weight_decay: float,
        ns_steps: int,
        polar_ridge: float = 0.0,
        hierarchical_feedback_fit: bool = False,
    ) -> None:
        if not modules:
            raise ValueError("MuonPairVQ requires at least one module")
        if not math.isfinite(float(polar_ridge)) or float(polar_ridge) < 0.0:
            raise ValueError("Pair-VQ polar ridge must be finite and non-negative")
        for module in modules:
            module.weight.requires_grad_(True)
        momentum_modes = {module.fp16_ambient_momentum for module in modules}
        if len(momentum_modes) != 1:
            raise ValueError("MuonPairVQ modules must use one momentum mode")
        self.fp16_ambient_momentum = momentum_modes.pop()
        reserved_escape_modes = {
            module.fp16_reserved_escape_granularity for module in modules
        }
        if len(reserved_escape_modes) != 1:
            raise ValueError(
                "MuonPairVQ modules must use one reserved-escape momentum mode"
            )
        self.fp16_reserved_escape_granularity = reserved_escape_modes.pop()
        self.hierarchical_feedback_fit = bool(hierarchical_feedback_fit)
        if self.fp16_reserved_escape_granularity and not self.fp16_ambient_momentum:
            raise ValueError("reserved-escape momentum requires FP16 ambient momentum")
        if self.hierarchical_feedback_fit:
            exact_codec = "fractional_lattice_q7q8_b32_p25_rq4_cfcq5b64l16"
            c_fc_count = sum(
                module.out_features > module.in_features for module in modules
            )
            c_proj_count = sum(
                module.out_features < module.in_features for module in modules
            )
            incompatible = [
                module
                for module in modules
                if (
                    not module.error_feedback
                    or module.feedback_codec != exact_codec
                    or module.forward_visible_feedback
                    or module.fp16_ambient_momentum
                    or module.lazy_retraction_interval != 1
                    or module.feedback_residual_probe_steps
                    or module.feedback_transform_probe_block_sizes
                    or module.feedback_lattice_probe_block_sizes
                    or module.feedback_axis_adaptation_probe_block_size
                    or module.feedback_fractional_probe_block_size
                )
            ]
            if (
                len(modules) != 24
                or c_fc_count != 12
                or c_proj_count != 12
                or incompatible
            ):
                raise ValueError(
                    "hierarchical feedback fitting is authorized only for the "
                    "24-module, probe-free B64/L16 compact MLP endpoint"
                )
        self.modules_by_id = {id(module.weight): module for module in modules}
        self._reserved_escape_slices: dict[int, tuple[str, int, int]] = {}
        self._reserved_escape_scope_elements: dict[str, int] = {}
        for module in modules:
            if not self.fp16_reserved_escape_granularity:
                continue
            scope = module.reserved_escape_scope
            start = self._reserved_escape_scope_elements.get(scope, 0)
            stop = start + module.element_count
            self._reserved_escape_slices[id(module.weight)] = (scope, start, stop)
            self._reserved_escape_scope_elements[scope] = stop
        self._diagnostics: list[dict[str, float | int]] = []
        self._fp32_ambient_references: dict[int, torch.Tensor] = {}
        defaults = {
            "lr": float(lr),
            "momentum": float(momentum),
            "weight_decay": float(weight_decay),
            "ns_steps": int(ns_steps),
            "polar_ridge": float(polar_ridge),
        }
        super().__init__([{"params": [module.weight for module in modules]}], defaults)
        self._reserved_escape_owner = self.param_groups[0]["params"][0]

    @property
    def reserved_escape_momentum_bytes(self) -> int:
        if not self.fp16_reserved_escape_granularity:
            return 0
        payload = self.state[self._reserved_escape_owner].get(
            "reserved_escape_momentum"
        )
        if payload is None:
            return 0
        total = 64 + len(self.modules_by_id) * 16
        for scope_payload in payload.values():
            state = ReservedEscapeState.from_payload(
                scope_payload, device=self._reserved_escape_owner.device
            )
            total += state.persistent_tensor_bytes
        return total

    @property
    def compact_boundary_ready(self) -> bool:
        """True only when every nonpersistent weight has a compact owner."""
        return all(
            module.is_compact_boundary()
            for module in self.modules_by_id.values()
        )

    def _decode_reserved_escape_momentum(self) -> dict[str, torch.Tensor]:
        payload = self.state[self._reserved_escape_owner].get(
            "reserved_escape_momentum"
        )
        if payload is None:
            return {
                scope: torch.zeros(
                    elements,
                    device=self._reserved_escape_owner.device,
                    dtype=torch.float16,
                )
                for scope, elements in self._reserved_escape_scope_elements.items()
            }
        decoded = {}
        for scope, elements in self._reserved_escape_scope_elements.items():
            state = ReservedEscapeState.from_payload(
                payload[scope], device=self._reserved_escape_owner.device
            )
            if state.n_elements != elements:
                raise ValueError(
                    f"reserved-escape {scope} element count does not match model"
                )
            if state.block_local != (
                self.fp16_reserved_escape_granularity
                in {"block", "adaptive_block"}
            ):
                raise ValueError(
                    f"reserved-escape {scope} granularity does not match model"
                )
            decoded[scope] = decode_reserved_escape(state)
        return decoded

    def _store_reserved_escape_momentum(
        self, decoded: dict[str, torch.Tensor]
    ) -> None:
        encoded = {
            scope: encode_reserved_escape(
                values,
                scope=scope,
                granularity=self.fp16_reserved_escape_granularity,
                block_words=4096,
            )
            for scope, values in decoded.items()
        }
        payload = {scope: state.to_payload() for scope, state in encoded.items()}
        self.state[self._reserved_escape_owner][
            "reserved_escape_momentum"
        ] = payload
        self._reserved_escape_summary = {
            scope: {
                "dictionary_size": state.dictionary_size,
                "bytes": state.persistent_tensor_bytes,
                "exceptions": state.exception_high_bytes.numel(),
            }
            for scope, state in encoded.items()
        }

    def load_state_dict(self, state_dict):
        result = super().load_state_dict(state_dict)
        for weight, state in self.state.items():
            module = self.modules_by_id[id(weight)]
            if module.fp16_ambient_momentum:
                if "compact_momentum" in state:
                    raise ValueError(
                        "FP16 ambient Pair-VQ resume contains compact momentum"
                    )
                if self.fp16_reserved_escape_granularity:
                    if "ambient_momentum" in state:
                        raise ValueError(
                            "reserved-escape resume contains raw ambient momentum"
                        )
                else:
                    momentum = state.get("ambient_momentum")
                    if momentum is not None:
                        state["ambient_momentum"] = momentum.to(
                            device=weight.device,
                            dtype=torch.float16,
                        )
            else:
                if "ambient_momentum" in state:
                    raise ValueError(
                        "compact Pair-VQ resume contains ambient momentum"
                    )
                momentum = state.get("compact_momentum")
                if momentum is not None:
                    state["compact_momentum"] = momentum.to(
                        device=weight.device,
                        dtype=module.codebooks.dtype,
                    )
            if module.error_feedback:
                levels = state.get("feedback_levels")
                codes = state.get("feedback_codes")
                center = state.get("feedback_center")
                if levels is not None:
                    state["feedback_levels"] = levels.to(
                        device=weight.device, dtype=torch.float32
                    )
                if codes is not None:
                    state["feedback_codes"] = codes.to(
                        device=weight.device, dtype=torch.uint8
                    )
                if center is not None:
                    state["feedback_center"] = center.to(
                        device=weight.device, dtype=torch.float32
                    )
                if levels is not None and codes is not None:
                    module.rematerialize_forward_visible_weight_(
                        state["feedback_levels"],
                        state["feedback_codes"],
                        state.get("feedback_center"),
                    )
        if self.fp16_reserved_escape_granularity:
            payload = self.state[self._reserved_escape_owner].get(
                "reserved_escape_momentum"
            )
            if payload is not None:
                if set(payload) != set(self._reserved_escape_scope_elements):
                    raise ValueError("reserved-escape resume scope set mismatch")
                for scope in self._reserved_escape_scope_elements:
                    ReservedEscapeState.from_payload(
                        payload[scope], device=self._reserved_escape_owner.device
                    )
        return result

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._diagnostics = []
        deferred_feedback: list[dict[str, Any]] = []
        decoded_reserved_momentum = (
            self._decode_reserved_escape_momentum()
            if self.fp16_reserved_escape_granularity
            else None
        )
        for group in self.param_groups:
            lr = float(group["lr"])
            momentum_coefficient = float(group["momentum"])
            weight_decay = float(group["weight_decay"])
            ns_steps = int(group["ns_steps"])
            polar_ridge = float(group["polar_ridge"])
            for weight in group["params"]:
                gradient = weight.grad
                if gradient is None:
                    continue
                module = self.modules_by_id[id(weight)]
                state: dict[str, Any] = self.state[weight]
                reference_requested_gradient = None
                if module.fp16_ambient_momentum:
                    gradient_fp32 = gradient.float()
                    if module.fp16_ambient_reference_probe_steps:
                        reference_momentum = self._fp32_ambient_references.get(
                            id(weight)
                        )
                        if reference_momentum is None:
                            reference_momentum = torch.zeros_like(
                                weight, dtype=torch.float32
                            )
                            self._fp32_ambient_references[id(weight)] = (
                                reference_momentum
                            )
                        reference_momentum.mul_(momentum_coefficient).add_(
                            gradient_fp32
                        )
                        if int(module.optimizer_step) in (
                            module.fp16_ambient_reference_probe_steps
                        ):
                            reference_requested_gradient = gradient_fp32.add(
                                reference_momentum,
                                alpha=momentum_coefficient,
                            )
                    if decoded_reserved_momentum is not None:
                        scope, start, stop = self._reserved_escape_slices[id(weight)]
                        ambient_momentum = decoded_reserved_momentum[scope][
                            start:stop
                        ].view_as(weight)
                    else:
                        if "ambient_momentum" not in state:
                            state["ambient_momentum"] = torch.zeros_like(
                                weight, dtype=torch.float16
                            )
                        ambient_momentum = state["ambient_momentum"]
                    next_momentum = ambient_momentum.float()
                    next_momentum.mul_(momentum_coefficient).add_(gradient_fp32)
                    ambient_momentum.copy_(next_momentum)
                    persisted_momentum = ambient_momentum.float()
                    requested_gradient = gradient_fp32.add(
                        persisted_momentum, alpha=momentum_coefficient
                    )
                else:
                    if "compact_momentum" not in state:
                        state["compact_momentum"] = torch.zeros_like(
                            module.codebooks
                        )
                    compact_momentum = state["compact_momentum"]
                    gradient_pairs = gradient.float().reshape(-1, 2)
                    expanded = torch.zeros_like(gradient_pairs)
                    for stage in range(module.stages):
                        codes = module.codes[stage].long()
                        accum = torch.zeros_like(module.codebooks[stage])
                        accum.index_add_(0, codes, gradient_pairs)
                        counts = torch.bincount(
                            codes, minlength=module.codebook_size
                        )
                        means = accum / counts.clamp_min(1).to(
                            dtype=accum.dtype
                        )[:, None]
                        compact_momentum[stage].mul_(
                            momentum_coefficient
                        ).add_(means)
                        expanded.add_(
                            compact_momentum[stage].index_select(0, codes)
                        )
                    expanded.div_(module.stages)
                    requested_gradient = gradient.float() + (
                        momentum_coefficient * expanded.reshape_as(gradient)
                    )
                if polar_ridge == 0.0:
                    update = muon_update(requested_gradient, steps=ns_steps)
                else:
                    update = muon_update(
                        requested_gradient,
                        steps=ns_steps,
                        polar_ridge=polar_ridge,
                    )
                reference_metrics = None
                if reference_requested_gradient is not None:
                    if polar_ridge == 0.0:
                        reference_update = muon_update(
                            reference_requested_gradient,
                            steps=ns_steps,
                        )
                    else:
                        reference_update = muon_update(
                            reference_requested_gradient,
                            steps=ns_steps,
                            polar_ridge=polar_ridge,
                        )
                    reference_flat = reference_update.double().reshape(-1)
                    candidate_flat = update.double().reshape(-1)
                    target_energy = float(reference_flat.square().sum())
                    candidate_energy = float(candidate_flat.square().sum())
                    dot = float((reference_flat * candidate_flat).sum())
                    squared_error = float(
                        (reference_flat - candidate_flat).square().sum()
                    )
                    denominator = max(
                        math.sqrt(
                            max(target_energy, 0.0)
                            * max(candidate_energy, 0.0)
                        ),
                        1e-30,
                    )
                    prepolar_flat = requested_gradient.double().reshape(-1)
                    reference_prepolar_flat = (
                        reference_requested_gradient.double().reshape(-1)
                    )
                    prepolar_target_energy = float(
                        reference_prepolar_flat.square().sum()
                    )
                    prepolar_candidate_energy = float(
                        prepolar_flat.square().sum()
                    )
                    prepolar_dot = float(
                        (reference_prepolar_flat * prepolar_flat).sum()
                    )
                    reference_metrics = {
                        "ambient_reference_polar_target_energy": target_energy,
                        "ambient_reference_polar_candidate_energy": (
                            candidate_energy
                        ),
                        "ambient_reference_polar_dot": dot,
                        "ambient_reference_polar_squared_error": squared_error,
                        "ambient_reference_polar_cosine": dot / denominator,
                        "ambient_reference_positive_line_recovery": (
                            max(dot, 0.0) ** 2
                            / max(target_energy * candidate_energy, 1e-30)
                        ),
                        "ambient_reference_prepolar_cosine": (
                            prepolar_dot
                            / max(
                                math.sqrt(
                                    prepolar_target_energy
                                    * prepolar_candidate_energy
                                ),
                                1e-30,
                            )
                        ),
                    }
                requested = weight.float()
                if weight_decay != 0.0:
                    requested = requested * (1.0 - lr * weight_decay)
                requested = requested.add(update.float(), alpha=-lr)
                current_request = requested - weight.float()
                next_optimizer_step = int(module.optimizer_step) + 1
                retraction_due = (
                    module.lazy_retraction_interval == 1
                    or next_optimizer_step % module.lazy_retraction_interval == 0
                    or next_optimizer_step
                    in module.lazy_retraction_forced_steps
                )
                if not retraction_due:
                    request_energy = float(current_request.square().sum())
                    weight.copy_(requested.to(dtype=weight.dtype))
                    module.optimizer_step.add_(1)
                    diagnostics = {
                        "layer": module.layer_id,
                        "stages": module.stages,
                        "fast_residual": int(module.fast_residual),
                        "in_features": module.in_features,
                        "out_features": module.out_features,
                        "optimizer_step": next_optimizer_step - 1,
                        "request_energy": request_energy,
                        "projection_residual_energy": 0.0,
                        "requested_step_energy_recovery": 1.0,
                        "requested_update_cosine": 1.0,
                        "code_changes": 0,
                        "fast_code_changes": 0,
                        "refresh_codes": 0,
                        "error_feedback": int(module.error_feedback),
                        "forward_visible_feedback": int(
                            module.forward_visible_feedback
                        ),
                        "projection_target_includes_prior_feedback": 0,
                        "lazy_retraction": 1,
                        "retracted": 0,
                        "compact_boundary": 0,
                        "lazy_retraction_interval": (
                            module.lazy_retraction_interval
                        ),
                    }
                    if reference_metrics is not None:
                        diagnostics.update(reference_metrics)
                    self._diagnostics.append(diagnostics)
                    continue
                if module.error_feedback:
                    if "feedback_levels" not in state:
                        state["feedback_levels"] = torch.zeros(
                            module.feedback_level_shape,
                            device=weight.device,
                            dtype=torch.float32,
                        )
                    if "feedback_codes" not in state:
                        state["feedback_codes"] = torch.zeros(
                            module.feedback_code_shape,
                            device=weight.device,
                            dtype=torch.uint8,
                        )
                    if (
                        module.feedback_center_shape is not None
                        and "feedback_center" not in state
                    ):
                        state["feedback_center"] = torch.zeros(
                            module.feedback_center_shape,
                            device=weight.device,
                            dtype=torch.float32,
                        )
                    feedback_before = module.decode_feedback(
                        state["feedback_levels"],
                        state["feedback_codes"],
                        state.get("feedback_center"),
                    ).reshape_as(weight)
                    projection_target = (
                        requested
                        if module.forward_visible_feedback
                        else requested + feedback_before
                    )
                else:
                    feedback_before = None
                    projection_target = requested
                refresh = (
                    module.lazy_retraction_interval > 1
                    or int(module.optimizer_step)
                    % module.code_refresh_interval
                    == 0
                )
                diagnostics = module.project_requested_weight_(
                    projection_target, refresh_codes=refresh
                )
                if reference_metrics is not None:
                    diagnostics.update(reference_metrics)
                diagnostics["error_feedback"] = int(module.error_feedback)
                diagnostics["forward_visible_feedback"] = int(
                    module.forward_visible_feedback
                )
                diagnostics["projection_target_includes_prior_feedback"] = int(
                    module.error_feedback and not module.forward_visible_feedback
                )
                diagnostics["lazy_retraction"] = int(
                    module.lazy_retraction_interval > 1
                )
                diagnostics["retracted"] = 1
                diagnostics["compact_boundary"] = 1
                diagnostics["lazy_retraction_interval"] = (
                    module.lazy_retraction_interval
                )
                if module.error_feedback:
                    raw_feedback = projection_target - weight.float()
                    if self.hierarchical_feedback_fit:
                        deferred_feedback.append(
                            {
                                "module": module,
                                "weight": weight,
                                "state": state,
                                "raw_feedback": raw_feedback,
                                "current_request_energy": float(
                                    current_request.square().sum()
                                ),
                                "diagnostics": diagnostics,
                                "vectors": raw_feedback.reshape(
                                    -1, module.vector_length
                                ),
                                "levels": state["feedback_levels"],
                                "packed": state["feedback_codes"],
                                "seed": module.feedback_fractional_lattice_seed,
                                "coordinate_bits": (
                                    module.feedback_residual_lattice_coordinate_bits
                                ),
                                "block_size": (
                                    module.feedback_residual_lattice_block_size
                                ),
                                "lloyd_iterations": (
                                    module.feedback_residual_lattice_lloyd_iterations
                                ),
                            }
                        )
                        continue
                    feedback_code_changes = module.fit_feedback_(
                        raw_feedback.reshape(-1, module.vector_length),
                        state["feedback_levels"],
                        state["feedback_codes"],
                        state.get("feedback_center"),
                    )
                    feedback_after = module.decode_feedback(
                        state["feedback_levels"],
                        state["feedback_codes"],
                        state.get("feedback_center"),
                    ).reshape_as(weight)
                    conservation_error = raw_feedback - feedback_after
                    current_request_energy = float(current_request.square().sum())
                    feedback_target_energy = float(raw_feedback.square().sum())
                    conservation_error_energy = float(
                        conservation_error.square().sum()
                    )
                    diagnostics.update(
                        {
                            "current_request_energy": current_request_energy,
                            "feedback_target_energy": feedback_target_energy,
                            "feedback_energy": float(feedback_after.square().sum()),
                            "feedback_to_weight_energy_ratio": float(
                                feedback_after.square().sum()
                                / weight.float().square().sum().clamp_min(1e-30)
                            ),
                            "feedback_quantization_residual_energy": (
                                conservation_error_energy
                            ),
                            "feedback_codec_energy_recovery": 1.0
                            - conservation_error_energy
                            / max(feedback_target_energy, 1e-30),
                            "conserved_requested_step_energy_recovery": 1.0
                            - conservation_error_energy
                            / max(current_request_energy, 1e-30),
                            "feedback_code_changes": feedback_code_changes,
                        }
                    )
                    if module.forward_visible_feedback:
                        base_projection_residual_energy = float(
                            diagnostics["projection_residual_energy"]
                        )
                        base_requested_step_energy_recovery = float(
                            diagnostics["requested_step_energy_recovery"]
                        )
                        module.rematerialize_forward_visible_weight_(
                            state["feedback_levels"],
                            state["feedback_codes"],
                            state.get("feedback_center"),
                        )
                        old_virtual = requested - current_request
                        achieved_virtual_delta = weight.float() - old_virtual
                        achieved_virtual_energy = float(
                            achieved_virtual_delta.square().sum()
                        )
                        requested_virtual_energy = float(
                            current_request.square().sum()
                        )
                        requested_virtual_inner = float(
                            (current_request * achieved_virtual_delta).sum()
                        )
                        diagnostics.update(
                            {
                                "base_projection_residual_energy": (
                                    base_projection_residual_energy
                                ),
                                "base_requested_step_energy_recovery": (
                                    base_requested_step_energy_recovery
                                ),
                                "projection_residual_energy": (
                                    conservation_error_energy
                                ),
                                "requested_step_energy_recovery": 1.0
                                - conservation_error_energy
                                / max(requested_virtual_energy, 1e-30),
                                "requested_update_cosine": (
                                    requested_virtual_inner
                                    / max(
                                        math.sqrt(
                                            max(requested_virtual_energy, 0.0)
                                            * max(achieved_virtual_energy, 0.0)
                                        ),
                                        1e-30,
                                    )
                                ),
                                "virtual_weight_matches_compact_state": 1,
                            }
                        )
                    if (
                        refresh
                        and module.feedback_codec
                        in (
                            "conditional_polar32x8",
                            "conditional_polar16x16",
                        )
                    ):
                        diagnostics.update(
                            _conditional_polar_pair_diagnostics(
                                raw_feedback.reshape(-1, module.vector_length),
                                state["feedback_levels"],
                                state["feedback_center"],
                                state["feedback_codes"],
                            )
                        )
                    elif (
                        refresh
                        and module.feedback_codec
                        == "conditional_polar16x16_rvq2"
                    ):
                        probe = (
                            int(diagnostics["optimizer_step"])
                            in module.feedback_residual_probe_steps
                        )
                        diagnostics.update(
                            _residual_conditional_polar_diagnostics(
                                raw_feedback.reshape(-1, module.vector_length),
                                state["feedback_levels"],
                                state["feedback_center"],
                                state["feedback_codes"],
                                include_decomposition=probe,
                                probe_lloyd_iterations=(
                                    module.feedback_residual_probe_lloyd_iterations
                                    if probe
                                    else ()
                                ),
                            )
                        )
                    elif (
                        refresh
                        and module.feedback_codec == "free_vq256_rvq2"
                    ):
                        probe = (
                            int(diagnostics["optimizer_step"])
                            in module.feedback_residual_probe_steps
                        )
                        diagnostics.update(
                            _free_pair_vq_rvq2_diagnostics(
                                raw_feedback.reshape(-1, module.vector_length),
                                state["feedback_levels"],
                                state["feedback_codes"],
                                include_exact_assignment=probe,
                            )
                        )
                        if probe:
                            for block_size in (
                                module.feedback_transform_probe_block_sizes
                            ):
                                counterfactual = (
                                    _block_fht_free_pair_vq_counterfactual(
                                        raw_feedback.reshape(
                                            -1, module.vector_length
                                        ),
                                        block_size=block_size,
                                        seed=(
                                            20261022
                                            + 8192 * module.layer_id
                                            + 131 * module.in_features
                                            + 17 * module.out_features
                                            + block_size
                                        ),
                                    )
                                )
                                for key, value in counterfactual.items():
                                    diagnostics[
                                        f"feedback_transform_b{block_size}_{key}"
                                    ] = value
                            for block_size in (
                                module.feedback_lattice_probe_block_sizes
                            ):
                                for coordinate_bits in (
                                    module.feedback_lattice_probe_coordinate_bits
                                ):
                                    counterfactual = (
                                        _block_fht_gain_lattice_counterfactual(
                                            raw_feedback.reshape(
                                                -1, module.vector_length
                                            ),
                                            block_size=block_size,
                                            coordinate_bits=coordinate_bits,
                                            seed=(
                                                20261023
                                                + 8192 * module.layer_id
                                                + 131 * module.in_features
                                                + 17 * module.out_features
                                                + block_size
                                            ),
                                        )
                                    )
                                    prefix = (
                                        f"feedback_lattice_b{block_size}_"
                                        f"q{coordinate_bits}_"
                                    )
                                    for key, value in counterfactual.items():
                                        diagnostics[prefix + key] = value
                            if module.feedback_axis_adaptation_probe_block_size:
                                counterfactual = (
                                    _block_gain_axis_adaptation_counterfactual(
                                        raw_feedback.reshape(
                                            -1, module.vector_length
                                        ),
                                        block_size=(
                                            module.feedback_axis_adaptation_probe_block_size
                                        ),
                                        coordinate_bits=(
                                            module.feedback_axis_adaptation_probe_coordinate_bits
                                        ),
                                        seed=(
                                            20261024
                                            + 8192 * module.layer_id
                                            + 131 * module.in_features
                                            + 17 * module.out_features
                                        ),
                                    )
                                )
                                for key, value in counterfactual.items():
                                    diagnostics[
                                        "feedback_axis_adapt_" + key
                                    ] = value
                            if module.feedback_fractional_probe_block_size:
                                counterfactual = (
                                    _block_fht_fractional_lattice_counterfactual(
                                        raw_feedback.reshape(
                                            -1, module.vector_length
                                        ),
                                        block_size=(
                                            module.feedback_fractional_probe_block_size
                                        ),
                                        base_coordinate_bits=(
                                            module.feedback_fractional_probe_base_coordinate_bits
                                        ),
                                        refinement_fractions=(
                                            module.feedback_fractional_probe_refinement_fractions
                                        ),
                                        seed=(
                                            20261025
                                            + 8192 * module.layer_id
                                            + 131 * module.in_features
                                            + 17 * module.out_features
                                        ),
                                    )
                                )
                                for key, value in counterfactual.items():
                                    diagnostics[
                                        "feedback_fractional_" + key
                                    ] = value
                    elif (
                        refresh
                        and module.feedback_codec
                        in (
                            "fractional_lattice_q7q8_b32_p25_rq4",
                            "fractional_lattice_q7q8_b32_p25_rq4_cfcq5",
                            "fractional_lattice_q7q8_b32_p25_rq4_cfcq5b64l16",
                        )
                        and int(diagnostics["optimizer_step"])
                        in module.feedback_residual_probe_steps
                        and (
                            not module.feedback_residual_probe_layers
                            or module.layer_id in module.feedback_residual_probe_layers
                        )
                        and module.out_features > module.in_features
                    ):
                        counterfactual = (
                            _fractional_residual_lattice_source_decomposition(
                                raw_feedback.reshape(-1, module.vector_length),
                                seed=module.feedback_fractional_lattice_seed,
                                block_sizes=module.feedback_lattice_probe_block_sizes,
                                coordinate_bits=(
                                    module.feedback_lattice_probe_coordinate_bits
                                ),
                                lloyd_iterations=(
                                    module.feedback_residual_probe_lloyd_iterations
                                ),
                                axis_block_size=(
                                    module.feedback_axis_adaptation_probe_block_size
                                ),
                                axis_coordinate_bits=(
                                    module.feedback_axis_adaptation_probe_coordinate_bits
                                ),
                            )
                        )
                        for key, value in counterfactual.items():
                            diagnostics["feedback_source_" + key] = value
                self._diagnostics.append(diagnostics)
        if deferred_feedback:
            feedback_changes = _fit_fractional_residual_lattice_feedback_batch_(
                deferred_feedback
            )
            for context, feedback_code_changes in zip(
                deferred_feedback, feedback_changes, strict=True
            ):
                module = context["module"]
                weight = context["weight"]
                state = context["state"]
                raw_feedback = context["raw_feedback"]
                diagnostics = context["diagnostics"]
                feedback_after = module.decode_feedback(
                    state["feedback_levels"],
                    state["feedback_codes"],
                    state.get("feedback_center"),
                ).reshape_as(weight)
                conservation_error = raw_feedback - feedback_after
                feedback_target_energy = float(raw_feedback.square().sum())
                conservation_error_energy = float(
                    conservation_error.square().sum()
                )
                current_request_energy = float(context["current_request_energy"])
                diagnostics.update(
                    {
                        "current_request_energy": current_request_energy,
                        "feedback_target_energy": feedback_target_energy,
                        "feedback_energy": float(feedback_after.square().sum()),
                        "feedback_to_weight_energy_ratio": float(
                            feedback_after.square().sum()
                            / weight.float().square().sum().clamp_min(1e-30)
                        ),
                        "feedback_quantization_residual_energy": (
                            conservation_error_energy
                        ),
                        "feedback_codec_energy_recovery": 1.0
                        - conservation_error_energy
                        / max(feedback_target_energy, 1e-30),
                        "conserved_requested_step_energy_recovery": 1.0
                        - conservation_error_energy
                        / max(current_request_energy, 1e-30),
                        "feedback_code_changes": feedback_code_changes,
                        "hierarchical_feedback_fit": 1,
                    }
                )
                self._diagnostics.append(diagnostics)
        if decoded_reserved_momentum is not None:
            self._store_reserved_escape_momentum(decoded_reserved_momentum)
            momentum_bytes = self.reserved_escape_momentum_bytes
            summary = self._reserved_escape_summary
            empty_scope = {
                "dictionary_size": 0,
                "bytes": 0,
                "exceptions": 0,
            }
            c_fc_summary = summary.get("c_fc", empty_scope)
            c_proj_summary = summary.get("c_proj", empty_scope)
            for diagnostics in self._diagnostics:
                diagnostics.update(
                    {
                        "reserved_escape_momentum": 1,
                        "reserved_escape_granularity": (
                            self.fp16_reserved_escape_granularity
                        ),
                        "reserved_escape_momentum_bytes": momentum_bytes,
                        "raw_fp16_momentum_bytes": sum(
                            self._reserved_escape_scope_elements.values()
                        )
                        * 2,
                        "persistent_raw_ambient_momentum_tensors": 0,
                        "reserved_escape_c_fc_dictionary_size": c_fc_summary[
                            "dictionary_size"
                        ],
                        "reserved_escape_c_fc_bytes": c_fc_summary["bytes"],
                        "reserved_escape_c_fc_exceptions": c_fc_summary[
                            "exceptions"
                        ],
                        "reserved_escape_c_proj_dictionary_size": c_proj_summary[
                            "dictionary_size"
                        ],
                        "reserved_escape_c_proj_bytes": c_proj_summary["bytes"],
                        "reserved_escape_c_proj_exceptions": c_proj_summary[
                            "exceptions"
                        ],
                    }
                )
        return loss

    def consume_diagnostics(self) -> list[dict[str, float | int]]:
        diagnostics = self._diagnostics
        self._diagnostics = []
        return diagnostics
