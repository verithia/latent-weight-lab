#!/usr/bin/env python3
"""No-training oracle for hierarchical coordinate-codebook statistics."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import torch

from examples.nanogpt.benchmark_pair_vq_batched_lloyd import (
    ROLES,
    batched_fit_scalar_codebooks,
    fixed_level_count_identity,
    serial_fit_scalar_codebooks,
)
from examples.nanogpt.pair_vq_hierarchical_lloyd_cuda import (
    hierarchical_lloyd_stats,
)


COORDINATE_ROLES = {
    "cfc_residual_coordinates",
    "cproj_residual_coordinates",
    "base_coordinates",
}
SELECTED_GROUPS = {
    "cfc_residual_coordinates": 12,
    "cproj_residual_coordinates": 12,
    "base_coordinates": 24,
    "refined_coordinates": 12,
    "base_gains": 24,
    "cfc_residual_gains": 12,
    "cproj_residual_gains": 12,
}


def hierarchical_fit_scalar_codebooks(
    values: torch.Tensor,
    *,
    level_count: int,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = values.mean(dim=1)
    std = values.std(dim=1, unbiased=False).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    probabilities = (
        torch.arange(level_count, device=values.device, dtype=torch.float32)
        + 0.5
    ) / level_count
    levels = mean[:, None] + std[:, None] * math.sqrt(2.0) * torch.erfinv(
        2.0 * probabilities[None, :] - 1.0
    )
    for _iteration in range(iterations):
        midpoints = ((levels[:, :-1] + levels[:, 1:]) * 0.5).contiguous()
        sums, counts = hierarchical_lloyd_stats(
            values, midpoints, level_count=level_count
        )
        live = counts > 0
        candidate = sums / counts.clamp_min(1).to(dtype=sums.dtype)
        levels = torch.where(live, candidate, levels).sort(dim=1).values
    midpoints = ((levels[:, :-1] + levels[:, 1:]) * 0.5).contiguous()
    codes = torch.searchsorted(midpoints, values.contiguous())
    return levels, codes


def grouped_fit(
    values: torch.Tensor,
    *,
    role: str,
    level_count: int,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    group_size = SELECTED_GROUPS[role]
    fit = (
        hierarchical_fit_scalar_codebooks
        if role in COORDINATE_ROLES
        else batched_fit_scalar_codebooks
    )
    outputs = [
        fit(
            values[start : start + group_size].contiguous(),
            level_count=level_count,
            iterations=iterations,
        )
        for start in range(0, values.shape[0], group_size)
    ]
    return (
        torch.cat([output[0] for output in outputs], dim=0),
        torch.cat([output[1] for output in outputs], dim=0),
    )


def synchronized_seconds(
    function: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    *,
    repetitions: int,
) -> tuple[float, list[float]]:
    function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter()
        function()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - started)
    return sum(samples) / len(samples), samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("hierarchical Lloyd oracle requires CUDA")
    root = Path(__file__).resolve().parents[2]
    source_paths = (
        Path(__file__).resolve(),
        root / "examples/nanogpt/pair_vq_hierarchical_lloyd_cuda.py",
        root / "csrc/pair_vq_hierarchical_lloyd_ext.cpp",
        root / "csrc/pair_vq_hierarchical_lloyd_ext_cuda.cu",
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    role_results: list[dict[str, Any]] = []
    total_serial = 0.0
    total_candidate = 0.0
    torch.cuda.reset_peak_memory_stats()
    for role_index, (role, batch, count, levels, iterations) in enumerate(ROLES):
        generator = torch.Generator(device="cuda").manual_seed(
            20261029 + role_index
        )
        base = torch.randn(count, device="cuda", generator=generator)
        row_offsets = torch.linspace(
            -0.02, 0.02, batch, device="cuda", dtype=torch.float32
        )
        values = (base[None, :] + row_offsets[:, None]).contiguous()
        serial_levels, serial_codes = serial_fit_scalar_codebooks(
            values, level_count=levels, iterations=iterations
        )
        candidate_levels, candidate_codes = grouped_fit(
            values,
            role=role,
            level_count=levels,
            iterations=iterations,
        )
        fixed_counts_exact = fixed_level_count_identity(
            values, level_count=levels
        )
        if role in COORDINATE_ROLES:
            fixed_levels = torch.linspace(
                -3.0, 3.0, levels, device="cuda", dtype=torch.float32
            ).expand(batch, -1)
            fixed_midpoints = (
                (fixed_levels[:, :-1] + fixed_levels[:, 1:]) * 0.5
            ).contiguous()
            _sums, hierarchy_counts = hierarchical_lloyd_stats(
                values, fixed_midpoints, level_count=levels
            )
            fixed_codes = torch.searchsorted(fixed_midpoints, values)
            reference_counts = torch.stack(
                [torch.bincount(row, minlength=levels) for row in fixed_codes]
            )
            fixed_counts_exact = bool(
                torch.equal(hierarchy_counts, reference_counts)
            )
        centroid_max_abs = float(
            (serial_levels - candidate_levels).abs().max()
        )
        code_mismatches = int((serial_codes != candidate_codes).sum())
        mismatch_fraction = code_mismatches / serial_codes.numel()
        serial_seconds, serial_samples = synchronized_seconds(
            lambda: serial_fit_scalar_codebooks(
                values, level_count=levels, iterations=iterations
            ),
            repetitions=args.repetitions,
        )
        candidate_seconds, candidate_samples = synchronized_seconds(
            lambda: grouped_fit(
                values,
                role=role,
                level_count=levels,
                iterations=iterations,
            ),
            repetitions=args.repetitions,
        )
        total_serial += serial_seconds
        total_candidate += candidate_seconds
        role_results.append(
            {
                "role": role,
                "implementation": (
                    "hierarchical_histogram"
                    if role in COORDINATE_ROLES
                    else "validated_reference_batch"
                ),
                "batch": batch,
                "group_size": SELECTED_GROUPS[role],
                "values_per_layer": count,
                "level_count": levels,
                "iterations": iterations,
                "fixed_level_counts_exact": fixed_counts_exact,
                "centroid_max_abs": centroid_max_abs,
                "code_mismatches": code_mismatches,
                "code_mismatch_fraction": mismatch_fraction,
                "outputs_finite": bool(torch.isfinite(candidate_levels).all()),
                "serial_seconds": serial_seconds,
                "serial_samples": serial_samples,
                "candidate_seconds": candidate_seconds,
                "candidate_samples": candidate_samples,
                "speedup": serial_seconds / candidate_seconds,
            }
        )
        del base, row_offsets, values
        torch.cuda.empty_cache()
    projected_optimizer_ms = 679.60375 - 1000.0 * (
        total_serial - total_candidate
    )
    correctness_passed = all(
        row["fixed_level_counts_exact"]
        and row["outputs_finite"]
        and row["centroid_max_abs"] <= 0.00015
        and row["code_mismatch_fraction"] <= 0.0001
        for row in role_results
    )
    peak_mib = torch.cuda.max_memory_allocated() / 2**20
    result = {
        "schema_version": "mai_124m_pair_vq_hierarchical_hybrid_result_v1",
        "recorded_at": "2026-08-26",
        "source_commit": commit,
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_paths
        },
        "device": torch.cuda.get_device_name(),
        "tile_values": 65536,
        "hierarchical_roles": sorted(COORDINATE_ROLES),
        "repetitions": args.repetitions,
        "roles": role_results,
        "total_serial_seconds": total_serial,
        "total_candidate_seconds": total_candidate,
        "aggregate_speedup": total_serial / total_candidate,
        "projected_optimizer_ms": projected_optimizer_ms,
        "correctness_passed": correctness_passed,
        "peak_allocated_mib": peak_mib,
        "scratch_passed": peak_mib <= 4096.0,
        "systems_passed": projected_optimizer_ms <= 495.0 and peak_mib <= 4096.0,
        "automatic_training": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
