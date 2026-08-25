#!/usr/bin/env python3
"""No-training oracle for fused layer-batched scalar Lloyd statistics."""
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
    serial_fit_scalar_codebooks,
)
from examples.nanogpt.pair_vq_batched_lloyd_cuda import batched_lloyd_stats


def fused_batched_fit(
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
        sums, counts = batched_lloyd_stats(
            values, midpoints, level_count=level_count
        )
        live = counts > 0
        candidate = sums / counts.clamp_min(1).to(dtype=sums.dtype)
        levels = torch.where(live, candidate, levels).sort(dim=1).values
    codes = torch.searchsorted(
        ((levels[:, :-1] + levels[:, 1:]) * 0.5).contiguous(),
        values,
    )
    return levels, codes


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
        raise RuntimeError("fused batched Lloyd oracle requires CUDA")
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    source_paths = (
        Path(__file__).resolve(),
        root / "examples/nanogpt/pair_vq_batched_lloyd_cuda.py",
        root / "csrc/pair_vq_batched_lloyd_ext.cpp",
        root / "csrc/pair_vq_batched_lloyd_ext_cuda.cu",
    )
    roles: list[dict[str, Any]] = []
    total_serial = 0.0
    total_fused = 0.0
    torch.cuda.reset_peak_memory_stats()
    for role_index, (role, batch, count, levels, iterations) in enumerate(ROLES):
        generator = torch.Generator(device="cuda").manual_seed(
            20261029 + role_index
        )
        base = torch.randn(count, device="cuda", generator=generator)
        offsets = torch.linspace(
            -0.02, 0.02, batch, device="cuda", dtype=torch.float32
        )
        values = (base[None, :] + offsets[:, None]).contiguous()
        serial_levels, serial_codes = serial_fit_scalar_codebooks(
            values, level_count=levels, iterations=iterations
        )
        fused_levels, fused_codes = fused_batched_fit(
            values, level_count=levels, iterations=iterations
        )
        fixed_levels = torch.linspace(
            -3.0, 3.0, levels, device="cuda", dtype=torch.float32
        ).expand(batch, -1)
        fixed_midpoints = (
            (fixed_levels[:, :-1] + fixed_levels[:, 1:]) * 0.5
        ).contiguous()
        _fixed_sums, fixed_counts = batched_lloyd_stats(
            values, fixed_midpoints, level_count=levels
        )
        reference_fixed_codes = torch.searchsorted(
            fixed_midpoints, values
        )
        reference_fixed_counts = torch.stack(
            [
                torch.bincount(row, minlength=levels)
                for row in reference_fixed_codes
            ]
        )
        serial_seconds, serial_samples = synchronized_seconds(
            lambda: serial_fit_scalar_codebooks(
                values, level_count=levels, iterations=iterations
            ),
            repetitions=args.repetitions,
        )
        fused_seconds, fused_samples = synchronized_seconds(
            lambda: fused_batched_fit(
                values, level_count=levels, iterations=iterations
            ),
            repetitions=args.repetitions,
        )
        total_serial += serial_seconds
        total_fused += fused_seconds
        mismatch_count = int((serial_codes != fused_codes).sum())
        roles.append(
            {
                "role": role,
                "batch": batch,
                "values_per_layer": count,
                "level_count": levels,
                "iterations": iterations,
                "fixed_level_counts_exact": bool(
                    torch.equal(fixed_counts, reference_fixed_counts)
                ),
                "centroid_max_abs": float(
                    (serial_levels - fused_levels).abs().max()
                ),
                "code_mismatches": mismatch_count,
                "code_mismatch_fraction": (
                    mismatch_count / serial_codes.numel()
                ),
                "serial_seconds": serial_seconds,
                "serial_samples": serial_samples,
                "fused_seconds": fused_seconds,
                "fused_samples": fused_samples,
                "speedup": serial_seconds / fused_seconds,
            }
        )
        del base, offsets, values
        torch.cuda.empty_cache()
    projected_optimizer_ms = 679.60375 - 1000.0 * (total_serial - total_fused)
    correctness_passed = all(
        row["fixed_level_counts_exact"]
        and row["centroid_max_abs"] <= 0.00015
        and row["code_mismatch_fraction"] <= 0.0001
        for row in roles
    )
    result = {
        "schema_version": "mai_124m_pair_vq_fused_batched_lloyd_oracle_result_v1",
        "recorded_at": "2026-08-26",
        "source_commit": commit,
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_paths
        },
        "device": torch.cuda.get_device_name(),
        "repetitions": args.repetitions,
        "roles": roles,
        "total_serial_seconds": total_serial,
        "total_fused_seconds": total_fused,
        "aggregate_speedup": total_serial / total_fused,
        "projected_optimizer_ms": projected_optimizer_ms,
        "correctness_passed": correctness_passed,
        "systems_passed": projected_optimizer_ms <= 495.0,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
