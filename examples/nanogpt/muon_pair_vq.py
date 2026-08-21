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
            fitted[live] = sums[live] / counts[live]
            fitted = fitted.sort().values
        indices = torch.bucketize(
            values.contiguous(), (fitted[:-1] + fitted[1:]) * 0.5
        )
        levels[coordinate].copy_(fitted)
        assignments.append(indices)
    new_codes = (assignments[0] * 16 + assignments[1]).to(torch.uint8)
    codes.copy_(new_codes)
    return int((new_codes != old_codes).sum())


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
        error_feedback: bool = False,
        feedback_codec: str = "cartesian4x4",
        feedback_output_group_size: int = 0,
        feedback_residual_probe_steps: tuple[int, ...] = (),
        feedback_residual_probe_lloyd_iterations: tuple[int, ...] = (),
        neighbor_candidates: int = 16,
        code_refresh_interval: int = 8,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.stages = int(stages)
        self.layer_id = int(layer_id)
        self.fast_residual = bool(fast_residual)
        self.error_feedback = bool(error_feedback)
        self.feedback_codec = str(feedback_codec)
        self.feedback_output_group_size = int(feedback_output_group_size)
        self.feedback_residual_probe_steps = tuple(
            int(step) for step in feedback_residual_probe_steps
        )
        self.feedback_residual_probe_lloyd_iterations = tuple(
            int(iterations)
            for iterations in feedback_residual_probe_lloyd_iterations
        )
        self.neighbor_candidates = int(neighbor_candidates)
        self.code_refresh_interval = int(code_refresh_interval)
        if self.in_features <= 0 or self.out_features <= 0:
            raise ValueError("pair-VQ dimensions must be positive")
        if self.in_features * self.out_features % self.vector_length:
            raise ValueError("pair-VQ element count must be divisible by two")
        if self.stages not in (1, 2):
            raise ValueError("pair-VQ stages must be one or two")
        if self.feedback_output_group_size < 0:
            raise ValueError("feedback output group size must be nonnegative")
        if any(step < 0 for step in self.feedback_residual_probe_steps):
            raise ValueError("feedback residual probe steps must be nonnegative")
        if any(
            iterations <= 0
            for iterations in self.feedback_residual_probe_lloyd_iterations
        ):
            raise ValueError(
                "feedback residual probe Lloyd iterations must be positive"
            )
        if (
            self.feedback_residual_probe_steps
            or self.feedback_residual_probe_lloyd_iterations
        ) and self.feedback_codec != "conditional_polar16x16_rvq2":
            raise ValueError(
                "feedback residual probes require conditional_polar16x16_rvq2"
            )
        if self.feedback_codec not in (
            "cartesian4x4",
            "polar32x8",
            "conditional_polar32x8",
            "conditional_polar16x16",
            "conditional_polar16x16_rvq2",
            "rvq4x4",
        ):
            raise ValueError("unknown pair-VQ feedback codec")
        if (
            self.feedback_codec in (
                "polar32x8",
                "conditional_polar32x8",
                "conditional_polar16x16",
                "conditional_polar16x16_rvq2",
                "rvq4x4",
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
        elif self.feedback_codec == "rvq4x4":
            metadata_values = 2 * 16 * 2 + 2
        else:
            metadata_values = self.feedback_group_count * 2 * 16
        level_bytes = metadata_values * torch.tensor(
            [], dtype=torch.float32
        ).element_size()
        code_stages = (
            2 if self.feedback_codec == "conditional_polar16x16_rvq2" else 1
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
        if self.feedback_codec == "rvq4x4":
            return (2, 16, 2)
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
        if self.feedback_codec == "rvq4x4":
            if center is None:
                raise ValueError("RVQ pair feedback requires a center")
            return _decode_rvq_pair_codec(levels, center, codes)
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
        if self.feedback_codec == "rvq4x4":
            if center is None:
                raise ValueError("RVQ pair feedback requires a center")
            return _fit_rvq_pair_codec_(vectors, levels, center, codes)
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
            + self.compact_momentum_bytes
            + self.compact_feedback_bytes
        )
        return {
            "elements": self.element_count,
            "stages": self.stages,
            "fast_residual": self.fast_residual,
            "persistent_codec_bytes": self.persistent_codec_bytes,
            "compact_momentum_bytes": self.compact_momentum_bytes,
            "compact_feedback_bytes": self.compact_feedback_bytes,
            "persistent_training_bytes": persistent_training,
            "model_compression_vs_dense_bf16": dense_bf16 / self.persistent_codec_bytes,
            "training_compression_vs_dense_fp32_weight_plus_momentum": (
                dense_fp32_weight_and_momentum / persistent_training
            ),
            "transient_materialized_weight_bytes": self.transient_weight_bytes,
            "transient_gradient_bytes": self.transient_weight_bytes,
            "dense_master_weight": "disabled",
            "dense_optimizer_momentum": "disabled",
            "dense_ambient_error_buffer": "disabled",
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
                                "two_uint8_residual_conditional_polar16x16_"
                                "codes_per_weight_pair"
                                if self.feedback_codec
                                == "conditional_polar16x16_rvq2"
                                else "uint8_rvq4x4_code_per_weight_pair"
                            )
                        )
                    )
                    if self.feedback_codec
                    in (
                        "polar32x8",
                        "conditional_polar32x8",
                        "conditional_polar16x16",
                        "conditional_polar16x16_rvq2",
                        "rvq4x4",
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
            module = self.modules_by_id[id(weight)]
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
                current_request = requested - weight.float()
                if module.error_feedback:
                    pair_count = module.element_count // module.vector_length
                    if "feedback_levels" not in state:
                        state["feedback_levels"] = torch.zeros(
                            module.feedback_level_shape,
                            device=weight.device,
                            dtype=torch.float32,
                        )
                    if "feedback_codes" not in state:
                        feedback_code_shape = (
                            (2, pair_count)
                            if module.feedback_codec
                            == "conditional_polar16x16_rvq2"
                            else (pair_count,)
                        )
                        state["feedback_codes"] = torch.zeros(
                            feedback_code_shape,
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
                    projection_target = requested + feedback_before
                else:
                    feedback_before = None
                    projection_target = requested
                refresh = (
                    int(module.optimizer_step) % module.code_refresh_interval == 0
                )
                diagnostics = module.project_requested_weight_(
                    projection_target, refresh_codes=refresh
                )
                diagnostics["error_feedback"] = int(module.error_feedback)
                if module.error_feedback:
                    raw_feedback = projection_target - weight.float()
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
                self._diagnostics.append(diagnostics)
        return loss

    def consume_diagnostics(self) -> list[dict[str, float | int]]:
        diagnostics = self._diagnostics
        self._diagnostics = []
        return diagnostics
