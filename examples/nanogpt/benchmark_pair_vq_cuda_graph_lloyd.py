#!/usr/bin/env python3
"""No-training CUDA-graph oracle for validated grouped Pair-VQ Lloyd fits."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from examples.nanogpt.benchmark_pair_vq_batched_lloyd import (
    ROLES,
    batched_fit_scalar_codebooks,
    fixed_level_count_identity,
    serial_fit_scalar_codebooks,
)


SELECTED_GROUPS = {
    "cfc_residual_coordinates": 12,
    "cproj_residual_coordinates": 12,
    "base_coordinates": 24,
    "refined_coordinates": 12,
    "base_gains": 24,
    "cfc_residual_gains": 12,
    "cproj_residual_gains": 12,
}


def graphsafe_batched_fit_scalar_codebooks(
    values: torch.Tensor,
    *,
    level_count: int,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference batched Lloyd fit with graph-safe exact integer counts."""
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
        counts = torch.zeros(
            batch * level_count, device=values.device, dtype=torch.int64
        )
        counts.index_add_(0, flat_codes, torch.ones_like(flat_codes))
        sums = sums.reshape(batch, level_count)
        counts = counts.reshape(batch, level_count)
        live = counts > 0
        candidate = sums / counts.clamp_min(1).to(dtype=sums.dtype)
        levels = torch.where(live, candidate, levels).sort(dim=1).values
    midpoints = ((levels[:, :-1] + levels[:, 1:]) * 0.5).contiguous()
    codes = torch.searchsorted(midpoints, values.contiguous())
    return levels, codes


def graphsafe_count_identity(values: torch.Tensor, *, level_count: int) -> bool:
    """Prove bincount and int64 index_add agree for fixed assignments."""
    batch = values.shape[0]
    levels = torch.linspace(
        -3.0, 3.0, level_count, device=values.device, dtype=torch.float32
    ).expand(batch, -1)
    midpoints = ((levels[:, :-1] + levels[:, 1:]) * 0.5).contiguous()
    codes = torch.searchsorted(midpoints, values.contiguous())
    offsets = (
        torch.arange(batch, device=values.device, dtype=torch.int64)
        * level_count
    )
    flat_codes = (codes + offsets[:, None]).reshape(-1)
    reference = torch.bincount(
        flat_codes, minlength=batch * level_count
    )
    candidate = torch.zeros_like(reference)
    candidate.index_add_(0, flat_codes, torch.ones_like(flat_codes))
    return bool(torch.equal(reference, candidate))


@dataclass
class GraphRunner:
    static_input: torch.Tensor
    graph: torch.cuda.CUDAGraph
    static_levels: torch.Tensor
    static_codes: torch.Tensor


def capture_runner(
    *,
    group_size: int,
    count: int,
    level_count: int,
    iterations: int,
    pool: tuple[int, int],
    fit_function: Callable[..., tuple[torch.Tensor, torch.Tensor]],
) -> GraphRunner:
    static_input = torch.zeros(
        group_size, count, device="cuda", dtype=torch.float32
    )
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        fit_function(
            static_input,
            level_count=level_count,
            iterations=iterations,
        )
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=pool):
        static_levels, static_codes = fit_function(
            static_input,
            level_count=level_count,
            iterations=iterations,
        )
    return GraphRunner(static_input, graph, static_levels, static_codes)


def graph_fit(
    runner: GraphRunner,
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    group_size = runner.static_input.shape[0]
    if values.shape[0] % group_size:
        raise ValueError("production batch must be divisible by captured group")
    levels = torch.empty(
        values.shape[0],
        runner.static_levels.shape[1],
        device=values.device,
        dtype=runner.static_levels.dtype,
    )
    codes = torch.empty(
        values.shape,
        device=values.device,
        dtype=runner.static_codes.dtype,
    )
    for start in range(0, values.shape[0], group_size):
        stop = start + group_size
        runner.static_input.copy_(values[start:stop])
        runner.graph.replay()
        levels[start:stop].copy_(runner.static_levels)
        codes[start:stop].copy_(runner.static_codes)
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
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--graphsafe-counts", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-graph Lloyd oracle requires CUDA")

    root = Path(__file__).resolve().parents[2]
    source_path = Path(__file__).resolve()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    fit_function = (
        graphsafe_batched_fit_scalar_codebooks
        if args.graphsafe_counts
        else batched_fit_scalar_codebooks
    )
    base_result: dict[str, Any] = {
        "schema_version": (
            "mai_124m_pair_vq_graphsafe_count_oracle_result_v1"
            if args.graphsafe_counts
            else "mai_124m_pair_vq_cuda_graph_lloyd_result_v1"
        ),
        "recorded_at": "2026-08-26",
        "source_commit": commit,
        "source": {
            "path": str(source_path.relative_to(root)),
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        },
        "device": torch.cuda.get_device_name(),
        "repetitions": args.repetitions,
        "selected_groups": SELECTED_GROUPS,
        "count_mode": "int64_index_add" if args.graphsafe_counts else "bincount",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    runners: dict[str, GraphRunner] = {}
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        allocated_before = torch.cuda.memory_allocated()
        reserved_before = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        shared_pool = torch.cuda.graph_pool_handle()
        for role, _batch, count, levels, iterations in ROLES:
            runners[role] = capture_runner(
                group_size=SELECTED_GROUPS[role],
                count=count,
                level_count=levels,
                iterations=iterations,
                pool=shared_pool,
                fit_function=fit_function,
            )
        torch.cuda.synchronize()
        retained_allocated = torch.cuda.memory_allocated() - allocated_before
        retained_reserved = torch.cuda.memory_reserved() - reserved_before
    except Exception as error:
        result = {
            **base_result,
            "status": "capture_rejected",
            "capture_passed": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "automatic_training": False,
        }
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    role_results: list[dict[str, Any]] = []
    total_serial_seconds = 0.0
    total_graph_seconds = 0.0
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
        count_primitive_exact = (
            graphsafe_count_identity(values, level_count=levels)
            if args.graphsafe_counts
            else None
        )
        serial_levels, serial_codes = serial_fit_scalar_codebooks(
            values, level_count=levels, iterations=iterations
        )
        graph_levels, graph_codes = graph_fit(runners[role], values)
        centroid_max_abs = float((serial_levels - graph_levels).abs().max())
        code_mismatches = int((serial_codes != graph_codes).sum())
        mismatch_fraction = code_mismatches / serial_codes.numel()
        outputs_finite = bool(
            torch.isfinite(graph_levels).all()
            and torch.isfinite(graph_codes.to(torch.float32)).all()
        )
        serial_seconds, serial_samples = synchronized_seconds(
            lambda: serial_fit_scalar_codebooks(
                values, level_count=levels, iterations=iterations
            ),
            repetitions=args.repetitions,
        )
        graph_seconds, graph_samples = synchronized_seconds(
            lambda: graph_fit(runners[role], values),
            repetitions=args.repetitions,
        )
        total_serial_seconds += serial_seconds
        total_graph_seconds += graph_seconds
        role_results.append(
            {
                "role": role,
                "batch": batch,
                "group_size": SELECTED_GROUPS[role],
                "values_per_layer": count,
                "level_count": levels,
                "iterations": iterations,
                "fixed_level_counts_exact": counts_exact,
                "count_primitive_exact": count_primitive_exact,
                "outputs_finite": outputs_finite,
                "centroid_max_abs": centroid_max_abs,
                "code_mismatches": code_mismatches,
                "code_mismatch_fraction": mismatch_fraction,
                "serial_seconds": serial_seconds,
                "serial_samples": serial_samples,
                "graph_seconds": graph_seconds,
                "graph_samples": graph_samples,
                "speedup": serial_seconds / graph_seconds,
            }
        )
        del base, row_offsets, values, serial_levels, serial_codes
        del graph_levels, graph_codes

    projected_optimizer_ms = 679.60375 - 1000.0 * (
        total_serial_seconds - total_graph_seconds
    )
    correctness_passed = all(
        row["fixed_level_counts_exact"]
        and (row["count_primitive_exact"] is not False)
        and row["outputs_finite"]
        and row["centroid_max_abs"] <= 0.00015
        and row["code_mismatch_fraction"] <= 0.0001
        for row in role_results
    )
    retained_scratch_mib = max(retained_allocated, retained_reserved) / 2**20
    scratch_passed = retained_scratch_mib <= 8192.0
    systems_passed = projected_optimizer_ms <= 495.0 and scratch_passed
    result = {
        **base_result,
        "status": (
            "all_oracle_gates_passed"
            if correctness_passed and systems_passed
            else "correctness_or_systems_rejected"
        ),
        "capture_passed": True,
        "roles": role_results,
        "total_serial_seconds": total_serial_seconds,
        "total_graph_seconds": total_graph_seconds,
        "aggregate_speedup": total_serial_seconds / total_graph_seconds,
        "projected_optimizer_ms": projected_optimizer_ms,
        "correctness_passed": correctness_passed,
        "retained_allocated_mib": retained_allocated / 2**20,
        "retained_reserved_mib": retained_reserved / 2**20,
        "retained_scratch_mib": retained_scratch_mib,
        "scratch_passed": scratch_passed,
        "systems_passed": systems_passed,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "automatic_training": False,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
