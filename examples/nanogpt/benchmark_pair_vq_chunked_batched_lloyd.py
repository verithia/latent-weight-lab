#!/usr/bin/env python3
"""No-training group-size bracket for reference layer-batched Lloyd fits."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import torch

from examples.nanogpt.benchmark_pair_vq_batched_lloyd import (
    ROLES,
    batched_fit_scalar_codebooks,
    serial_fit_scalar_codebooks,
)


def chunked_fit(
    values: torch.Tensor,
    *,
    group_size: int,
    level_count: int,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = [
        batched_fit_scalar_codebooks(
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
        raise RuntimeError("chunked batched Lloyd oracle requires CUDA")
    root = Path(__file__).resolve().parents[2]
    source_path = Path(__file__).resolve()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    roles: list[dict[str, Any]] = []
    total_serial = 0.0
    total_selected = 0.0
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
        serial_seconds, serial_samples = synchronized_seconds(
            lambda: serial_fit_scalar_codebooks(
                values, level_count=levels, iterations=iterations
            ),
            repetitions=args.repetitions,
        )
        total_serial += serial_seconds
        group_sizes = (12, 6, 4) if batch == 12 else (24, 12, 8, 6)
        candidates = []
        for group_size in group_sizes:
            candidate_levels, candidate_codes = chunked_fit(
                values,
                group_size=group_size,
                level_count=levels,
                iterations=iterations,
            )
            centroid_max_abs = float(
                (serial_levels - candidate_levels).abs().max()
            )
            mismatches = int((serial_codes != candidate_codes).sum())
            mismatch_fraction = mismatches / serial_codes.numel()
            seconds, samples = synchronized_seconds(
                lambda group_size=group_size: chunked_fit(
                    values,
                    group_size=group_size,
                    level_count=levels,
                    iterations=iterations,
                ),
                repetitions=args.repetitions,
            )
            candidates.append(
                {
                    "group_size": group_size,
                    "centroid_max_abs": centroid_max_abs,
                    "code_mismatches": mismatches,
                    "code_mismatch_fraction": mismatch_fraction,
                    "seconds": seconds,
                    "samples": samples,
                    "speedup_vs_serial": serial_seconds / seconds,
                    "correctness_passed": (
                        centroid_max_abs <= 0.00015
                        and mismatch_fraction <= 0.0001
                    ),
                }
            )
        eligible = [row for row in candidates if row["correctness_passed"]]
        selected = min(eligible, key=lambda row: row["seconds"])
        total_selected += float(selected["seconds"])
        roles.append(
            {
                "role": role,
                "batch": batch,
                "values_per_layer": count,
                "level_count": levels,
                "iterations": iterations,
                "serial_seconds": serial_seconds,
                "serial_samples": serial_samples,
                "candidates": candidates,
                "selected_group_size": selected["group_size"],
                "selected_seconds": selected["seconds"],
            }
        )
        del base, offsets, values
        torch.cuda.empty_cache()
    projected_optimizer_ms = 679.60375 - 1000.0 * (
        total_serial - total_selected
    )
    result = {
        "schema_version": "mai_124m_pair_vq_chunked_batched_lloyd_result_v1",
        "recorded_at": "2026-08-26",
        "source_commit": commit,
        "source": {
            "path": str(source_path.relative_to(root)),
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        },
        "device": torch.cuda.get_device_name(),
        "repetitions": args.repetitions,
        "roles": roles,
        "total_serial_seconds": total_serial,
        "total_selected_seconds": total_selected,
        "aggregate_speedup": total_serial / total_selected,
        "projected_optimizer_ms": projected_optimizer_ms,
        "correctness_passed": all(
            any(candidate["correctness_passed"] for candidate in role["candidates"])
            for role in roles
        ),
        "systems_passed": projected_optimizer_ms <= 495.0,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
