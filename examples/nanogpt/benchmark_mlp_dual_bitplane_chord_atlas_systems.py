#!/usr/bin/env python3
"""Frozen H50 dual-bitplane chord-atlas systems discriminator."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F


SCHEMA_VERSION = "nanogpt_mlp_dual_bitplane_chord_atlas_systems_v1"
PLAN_SCHEMA_VERSION = "nanogpt_mlp_dual_bitplane_chord_atlas_systems_plan_v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def deployment_accounting(
    *, layers: int = 12, width: int = 768, candidate_hidden: int = 2304, planes: int = 2
) -> dict[str, int | float]:
    roles = 2
    binary_values = roles * planes * candidate_hidden * width
    binary_bytes = binary_values // 8
    scale_values = roles * planes * candidate_hidden
    scale_bytes = 2 * scale_values
    coordinate_values = layers * roles * planes * candidate_hidden
    coordinate_bytes = 2 * coordinate_values
    total = binary_bytes + scale_bytes + coordinate_bytes
    dense_values = layers * 2 * 4 * width * width
    dense_bytes = 2 * dense_values
    cache_bytes = layers * roles * candidate_hidden * width * 2
    return {
        "binary_endpoint_values": binary_values,
        "binary_endpoint_bytes": binary_bytes,
        "fp16_endpoint_scale_values": scale_values,
        "fp16_endpoint_scale_bytes": scale_bytes,
        "fp16_chord_coordinate_values": coordinate_values,
        "fp16_chord_coordinate_bytes": coordinate_bytes,
        "total_checkpoint_bytes": total,
        "dense_replaced_mlp_fp16_bytes": dense_bytes,
        "checkpoint_byte_fraction": total / dense_bytes,
        "continuous_coordinate_values": coordinate_values,
        "continuous_coordinate_fraction": coordinate_values / dense_values,
        "all_layer_materialized_fp16_cache_bytes": cache_bytes,
        "materialized_cache_fraction_of_dense_fp16_mlp": cache_bytes / dense_bytes,
    }


def materialize_endpoint(
    sign_planes: torch.Tensor,
    scales: torch.Tensor,
    coordinates: torch.Tensor,
) -> torch.Tensor:
    if sign_planes.ndim != 3 or sign_planes.shape[0] != 2:
        raise ValueError("H50 requires exactly two sign planes")
    if scales.shape != sign_planes.shape[:2] or coordinates.shape != scales.shape:
        raise ValueError("scale/coordinate shape mismatch")
    weights = coordinates * scales
    return torch.sum(weights[:, :, None] * sign_planes, dim=0)


def measure_cuda(
    function: Callable[[], torch.Tensor], *, warmups: int, trials: int
) -> float:
    for _ in range(warmups):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(trials):
        function()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / trials)


def benchmark_token_count(
    tokens: int,
    *,
    width: int,
    dense_hidden: int,
    candidate_hidden: int,
    warmups: int,
    trials: int,
    device: str,
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(202608500 + tokens)
    inputs = torch.randn(tokens, width, generator=generator).to(
        device=device, dtype=torch.float16
    )
    dense_u = torch.randn(dense_hidden, width, generator=generator).to(
        device=device, dtype=torch.float16
    )
    dense_v = torch.randn(dense_hidden, width, generator=generator).to(
        device=device, dtype=torch.float16
    )
    sign_u = (
        torch.randint(0, 2, (2, candidate_hidden, width), generator=generator)
        .mul_(2)
        .sub_(1)
        .to(device=device, dtype=torch.float16)
    )
    sign_v = (
        torch.randint(0, 2, (2, candidate_hidden, width), generator=generator)
        .mul_(2)
        .sub_(1)
        .to(device=device, dtype=torch.float16)
    )
    scale_u = torch.rand(2, candidate_hidden, generator=generator).to(
        device=device, dtype=torch.float16
    )
    scale_v = torch.rand(2, candidate_hidden, generator=generator).to(
        device=device, dtype=torch.float16
    )
    coordinate_u = torch.randn(2, candidate_hidden, generator=generator).to(
        device=device, dtype=torch.float16
    )
    coordinate_v = torch.randn(2, candidate_hidden, generator=generator).to(
        device=device, dtype=torch.float16
    )

    with torch.no_grad():
        candidate_u = materialize_endpoint(sign_u, scale_u, coordinate_u)
        candidate_v = materialize_endpoint(sign_v, scale_v, coordinate_v)

        def dense() -> torch.Tensor:
            return F.gelu(inputs @ dense_u.T) @ dense_v

        def candidate() -> torch.Tensor:
            return F.gelu(inputs @ candidate_u.T) @ candidate_v

        def materialize_both() -> torch.Tensor:
            return materialize_endpoint(
                sign_u, scale_u, coordinate_u
            ) + materialize_endpoint(sign_v, scale_v, coordinate_v)

        dense_ms = measure_cuda(dense, warmups=warmups, trials=trials)
        candidate_ms = measure_cuda(candidate, warmups=warmups, trials=trials)
        materialization_ms = measure_cuda(
            materialize_both, warmups=warmups, trials=trials
        )

    return {
        "tokens": tokens,
        "dense_ms": dense_ms,
        "cached_candidate_ms": candidate_ms,
        "cached_candidate_over_dense": candidate_ms / dense_ms,
        "materialize_both_endpoints_ms": materialization_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    plan = json.loads(args.plan.read_text())
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unexpected H50 plan schema")
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("the binding H50 systems discriminator requires CUDA")

    frozen = plan["frozen_systems_discriminator"]
    architecture = plan["architecture"]
    expected = plan["exact_checkpoint_accounting"]
    accounting = deployment_accounting(
        layers=architecture["layers"],
        width=architecture["model_width"],
        candidate_hidden=architecture["candidate_hidden_width"],
        planes=architecture["shared_binary_planes_per_role"],
    )
    for key in (
        "dense_replaced_mlp_fp16_bytes",
        "binary_endpoint_values",
        "binary_endpoint_bytes",
        "fp16_endpoint_scale_values",
        "fp16_endpoint_scale_bytes",
        "fp16_chord_coordinate_values",
        "fp16_chord_coordinate_bytes",
        "total_checkpoint_bytes",
        "continuous_coordinate_values",
        "all_layer_materialized_fp16_cache_bytes",
    ):
        if accounting[key] != expected[key]:
            raise RuntimeError(f"accounting mismatch for {key}: {accounting[key]} != {expected[key]}")

    torch.cuda.set_device(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)
    measurements = [
        benchmark_token_count(
            tokens,
            width=architecture["model_width"],
            dense_hidden=architecture["dense_hidden_width"],
            candidate_hidden=architecture["candidate_hidden_width"],
            warmups=frozen["warmup_trials"],
            trials=frozen["timed_trials"],
            device=args.device,
        )
        for tokens in frozen["token_counts"]
    ]
    peak = int(torch.cuda.max_memory_allocated(args.device))
    latency_pass = all(
        row["cached_candidate_over_dense"]
        <= frozen["maximum_cached_candidate_over_dense_latency_each_token_count"]
        for row in measurements
    )
    memory_pass = peak <= frozen["maximum_peak_cuda_bytes"]
    decision = (
        "PASS_H50_SYSTEMS_AUTHORIZE_REPRESENTATION_AUDIT"
        if latency_pass and memory_pass
        else "REJECT_H50_SYSTEMS"
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "finished",
        "decision": decision,
        "source_commit": git_commit(REPO_ROOT),
        "source_sha256": file_sha256(Path(__file__)),
        "plan_path": str(args.plan.resolve()),
        "plan_sha256": file_sha256(args.plan),
        "accounting": accounting,
        "measurements": measurements,
        "peak_cuda_bytes": peak,
        "latency_gate_pass": latency_pass,
        "memory_gate_pass": memory_pass,
        "cuda_device": torch.cuda.get_device_name(args.device),
        "torch_version": torch.__version__,
        "elapsed_seconds": time.time() - started,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    result_path = args.output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    size = sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file())
    if size > frozen["maximum_result_bytes"]:
        raise RuntimeError(f"result directory exceeded gate: {size}")
    print(json.dumps({**result, "result_bytes": size}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
