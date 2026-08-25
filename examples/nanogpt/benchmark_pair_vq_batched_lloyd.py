#!/usr/bin/env python3
"""No-training production-shape oracle for layer-batched scalar Lloyd fits."""
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

from examples.nanogpt.muon_pair_vq import _fit_scalar_codebook


ROLES = (
    ("cfc_residual_coordinates", 12, 2_359_296, 32, 16),
    ("cproj_residual_coordinates", 12, 2_359_296, 16, 4),
    ("base_coordinates", 24, 2_359_296, 128, 4),
    ("refined_coordinates", 24, 2_359_296, 256, 4),
    ("base_gains", 24, 73_728, 256, 4),
    ("cfc_residual_gains", 12, 36_864, 256, 16),
    ("cproj_residual_gains", 12, 73_728, 256, 4),
)


def batched_fit_scalar_codebooks(
    values: torch.Tensor,
    *,
    level_count: int,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit independent row-wise codebooks with one reduction per phase."""
    if values.ndim != 2 or values.dtype != torch.float32:
        raise ValueError("batched Lloyd values must be an FP32 matrix")
    batch = values.shape[0]
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
    offsets = (
        torch.arange(batch, device=values.device, dtype=torch.int64)
        * level_count
    )
    flat_values = values.reshape(-1)
    for _iteration in range(iterations):
        midpoints = ((levels[:, :-1] + levels[:, 1:]) * 0.5).contiguous()
        codes = torch.searchsorted(midpoints, values.contiguous())
        codes.add_(offsets[:, None])
        flat_codes = codes.reshape(-1)
        sums = torch.zeros(
            batch * level_count, device=values.device, dtype=torch.float32
        )
        sums.index_add_(0, flat_codes, flat_values)
        counts = torch.bincount(flat_codes, minlength=batch * level_count)
        sums = sums.reshape(batch, level_count)
        counts = counts.reshape(batch, level_count)
        live = counts > 0
        candidate = sums / counts.clamp_min(1).to(dtype=sums.dtype)
        levels = torch.where(live, candidate, levels).sort(dim=1).values
    midpoints = ((levels[:, :-1] + levels[:, 1:]) * 0.5).contiguous()
    codes = torch.searchsorted(midpoints, values.contiguous())
    return levels, codes


def serial_fit_scalar_codebooks(
    values: torch.Tensor,
    *,
    level_count: int,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = [
        _fit_scalar_codebook(
            row.contiguous(),
            level_count=level_count,
            iterations=iterations,
        )
        for row in values
    ]
    return (
        torch.stack([output[0] for output in outputs]),
        torch.stack([output[1] for output in outputs]),
    )


def fixed_level_count_identity(
    values: torch.Tensor,
    *,
    level_count: int,
) -> bool:
    batch = values.shape[0]
    levels = torch.linspace(
        -3.0, 3.0, level_count, device=values.device, dtype=torch.float32
    ).expand(batch, -1)
    midpoints = ((levels[:, :-1] + levels[:, 1:]) * 0.5).contiguous()
    batched_codes = torch.searchsorted(midpoints, values.contiguous())
    offsets = (
        torch.arange(batch, device=values.device, dtype=torch.int64)
        * level_count
    )
    flattened = (batched_codes + offsets[:, None]).reshape(-1)
    batched_counts = torch.bincount(
        flattened, minlength=batch * level_count
    ).reshape(batch, level_count)
    serial_counts = torch.stack(
        [
            torch.bincount(
                torch.bucketize(row.contiguous(), midpoints[index]),
                minlength=level_count,
            )
            for index, row in enumerate(values)
        ]
    )
    return bool(torch.equal(batched_counts, serial_counts))


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


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("batched Lloyd oracle requires CUDA")
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    source_path = Path(__file__).resolve()
    results: list[dict[str, Any]] = []
    total_serial_seconds = 0.0
    total_batched_seconds = 0.0
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
        counts_exact = fixed_level_count_identity(values, level_count=levels)
        serial_levels, serial_codes = serial_fit_scalar_codebooks(
            values, level_count=levels, iterations=iterations
        )
        batched_levels, batched_codes = batched_fit_scalar_codebooks(
            values, level_count=levels, iterations=iterations
        )
        centroid_max_abs = float(
            (serial_levels - batched_levels).abs().max()
        )
        code_mismatches = int((serial_codes != batched_codes).sum())
        code_mismatch_fraction = code_mismatches / serial_codes.numel()
        serial_seconds, serial_samples = synchronized_seconds(
            lambda: serial_fit_scalar_codebooks(
                values, level_count=levels, iterations=iterations
            ),
            repetitions=args.repetitions,
        )
        batched_seconds, batched_samples = synchronized_seconds(
            lambda: batched_fit_scalar_codebooks(
                values, level_count=levels, iterations=iterations
            ),
            repetitions=args.repetitions,
        )
        total_serial_seconds += serial_seconds
        total_batched_seconds += batched_seconds
        results.append(
            {
                "role": role,
                "batch": batch,
                "values_per_layer": count,
                "level_count": levels,
                "iterations": iterations,
                "fixed_level_counts_exact": counts_exact,
                "centroid_max_abs": centroid_max_abs,
                "code_mismatches": code_mismatches,
                "code_mismatch_fraction": code_mismatch_fraction,
                "serial_seconds": serial_seconds,
                "serial_samples": serial_samples,
                "batched_seconds": batched_seconds,
                "batched_samples": batched_samples,
                "speedup": serial_seconds / batched_seconds,
                "serial_levels_sha256": tensor_sha256(serial_levels),
                "batched_levels_sha256": tensor_sha256(batched_levels),
            }
        )
        del base, row_offsets, values
        torch.cuda.empty_cache()
    projected_optimizer_ms = 679.60375 - 1000.0 * (
        total_serial_seconds - total_batched_seconds
    )
    correctness_passed = all(
        row["fixed_level_counts_exact"]
        and row["centroid_max_abs"] <= 0.00015
        and row["code_mismatch_fraction"] <= 0.0001
        for row in results
    )
    result = {
        "schema_version": "mai_124m_pair_vq_batched_lloyd_oracle_result_v1",
        "recorded_at": "2026-08-26",
        "source_commit": commit,
        "source": {
            "path": str(source_path.relative_to(root)),
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        },
        "device": torch.cuda.get_device_name(),
        "repetitions": args.repetitions,
        "roles": results,
        "total_serial_seconds": total_serial_seconds,
        "total_batched_seconds": total_batched_seconds,
        "aggregate_speedup": total_serial_seconds / total_batched_seconds,
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
