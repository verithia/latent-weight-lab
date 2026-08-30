#!/usr/bin/env python3
"""Frozen H49 top-1 three-expert MLP systems discriminator."""

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


SCHEMA_VERSION = "nanogpt_mlp_top1_three_expert_systems_v1"
PLAN_SCHEMA_VERSION = "nanogpt_mlp_top1_three_expert_binary_systems_plan_v1"


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
    *, layers: int = 12, experts: int = 3, expert_width: int = 1024, width: int = 768, rank: int = 2
) -> dict[str, int | float]:
    neurons = experts * expert_width
    binary_values = 2 * neurons * width
    binary_bytes = binary_values // 8
    scale_values = 2 * neurons
    scale_bytes = 2 * scale_values
    factor_values = layers * 2 * rank * (neurons + width)
    factor_bytes = 2 * factor_values
    gain_values = layers * neurons
    gain_bytes = 2 * gain_values
    router_values = layers * (width * experts + experts)
    router_bytes = 2 * router_values
    total = binary_bytes + scale_bytes + factor_bytes + gain_bytes + router_bytes
    dense_values = layers * 2 * 4 * width * width
    dense_bytes = 2 * dense_values
    continuous = factor_values + gain_values + router_values
    return {
        "binary_endpoint_values": binary_values,
        "binary_endpoint_bytes": binary_bytes,
        "fp16_endpoint_scale_values": scale_values,
        "fp16_endpoint_scale_bytes": scale_bytes,
        "fp16_private_factor_values": factor_values,
        "fp16_private_factor_bytes": factor_bytes,
        "fp16_layer_gain_values": gain_values,
        "fp16_layer_gain_bytes": gain_bytes,
        "fp16_router_values": router_values,
        "fp16_router_bytes": router_bytes,
        "total_checkpoint_bytes": total,
        "dense_replaced_mlp_fp16_bytes": dense_bytes,
        "checkpoint_byte_fraction": total / dense_bytes,
        "continuous_coordinate_values": continuous,
        "continuous_coordinate_fraction": continuous / dense_values,
    }


def dispatch_with_indices(
    inputs: torch.Tensor,
    indices: list[torch.Tensor],
    expert_u: torch.Tensor,
    expert_v: torch.Tensor,
) -> torch.Tensor:
    output = torch.empty_like(inputs)
    for expert, index in enumerate(indices):
        selected = torch.index_select(inputs, 0, index)
        result = F.gelu(selected @ expert_u[expert].T) @ expert_v[expert]
        output.index_copy_(0, index, result)
    return output


def dispatch_with_routes(
    inputs: torch.Tensor,
    routes: torch.Tensor,
    expert_u: torch.Tensor,
    expert_v: torch.Tensor,
) -> torch.Tensor:
    indices = [
        torch.nonzero(routes == expert, as_tuple=False).flatten()
        for expert in range(expert_u.shape[0])
    ]
    return dispatch_with_indices(inputs, indices, expert_u, expert_v)


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
    hidden: int,
    experts: int,
    expert_width: int,
    warmups: int,
    trials: int,
    device: str,
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(202608491 + tokens)
    inputs = torch.randn(tokens, width, generator=generator).to(
        device=device, dtype=torch.float16
    )
    dense_u = torch.randn(hidden, width, generator=generator).to(
        device=device, dtype=torch.float16
    )
    dense_v = torch.randn(hidden, width, generator=generator).to(
        device=device, dtype=torch.float16
    )
    expert_u = torch.randn(
        experts, expert_width, width, generator=generator
    ).to(device=device, dtype=torch.float16)
    expert_v = torch.randn(
        experts, expert_width, width, generator=generator
    ).to(device=device, dtype=torch.float16)
    balanced_router = torch.randn(width, experts, generator=generator).to(
        device=device, dtype=torch.float16
    )
    balanced_bias = torch.zeros(experts, device=device, dtype=torch.float16)
    skewed_router = balanced_router
    skewed_bias = torch.tensor(
        [32752.0, 0.0, 0.0], device=device, dtype=torch.float16
    )

    with torch.no_grad():
        balanced_routes = (inputs @ balanced_router + balanced_bias).argmax(dim=-1)
        skewed_routes = (inputs @ skewed_router + skewed_bias).argmax(dim=-1)
        balanced_indices = [
            torch.nonzero(balanced_routes == expert, as_tuple=False).flatten()
            for expert in range(experts)
        ]
        balanced_counts = [int(index.numel()) for index in balanced_indices]
        skewed_counts = [
            int((skewed_routes == expert).sum()) for expert in range(experts)
        ]
        if min(balanced_counts) < int(0.20 * tokens):
            raise RuntimeError(f"balanced router occupancy gate failed: {balanced_counts}")

        def dense() -> torch.Tensor:
            return F.gelu(inputs @ dense_u.T) @ dense_v

        def skewed() -> torch.Tensor:
            routes = (inputs @ skewed_router + skewed_bias).argmax(dim=-1)
            return dispatch_with_routes(inputs, routes, expert_u, expert_v)

        def balanced() -> torch.Tensor:
            routes = (inputs @ balanced_router + balanced_bias).argmax(dim=-1)
            return dispatch_with_routes(inputs, routes, expert_u, expert_v)

        def prepartitioned() -> torch.Tensor:
            # Charge the router matmul but use cached grouping to expose the
            # optimistic lower bound of a fused/grouped dispatch kernel.
            _logits = inputs @ balanced_router + balanced_bias
            return dispatch_with_indices(
                inputs, balanced_indices, expert_u, expert_v
            ) + _logits[:, :1] * 0

        dense_ms = measure_cuda(dense, warmups=warmups, trials=trials)
        skewed_ms = measure_cuda(skewed, warmups=warmups, trials=trials)
        balanced_ms = measure_cuda(balanced, warmups=warmups, trials=trials)
        lower_bound_ms = measure_cuda(
            prepartitioned, warmups=warmups, trials=trials
        )
    return {
        "tokens": tokens,
        "balanced_counts": balanced_counts,
        "skewed_counts": skewed_counts,
        "dense_ms": dense_ms,
        "skewed_dispatch_ms": skewed_ms,
        "skewed_over_dense": skewed_ms / dense_ms,
        "balanced_dispatch_ms": balanced_ms,
        "balanced_over_dense": balanced_ms / dense_ms,
        "prepartitioned_lower_bound_ms": lower_bound_ms,
        "prepartitioned_lower_bound_over_dense": lower_bound_ms / dense_ms,
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
        raise ValueError("unexpected H49 plan schema")
    architecture = plan["architecture"]
    systems = plan["frozen_systems_discriminator"]
    accounting = deployment_accounting(
        layers=12,
        experts=int(architecture["experts"]),
        expert_width=int(architecture["expert_hidden_width"]),
        width=int(architecture["model_width"]),
        rank=2,
    )
    expected = plan["exact_checkpoint_accounting"]
    for key in (
        "dense_replaced_mlp_fp16_bytes",
        "binary_endpoint_values",
        "binary_endpoint_bytes",
        "fp16_endpoint_scale_values",
        "fp16_endpoint_scale_bytes",
        "fp16_private_factor_values",
        "fp16_private_factor_bytes",
        "fp16_layer_gain_values",
        "fp16_layer_gain_bytes",
        "fp16_router_values",
        "fp16_router_bytes",
        "total_checkpoint_bytes",
        "continuous_coordinate_values",
    ):
        if accounting[key] != expected[key]:
            raise AssertionError((key, accounting[key], expected[key]))
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("H49 systems gate requires CUDA")
    torch.cuda.reset_peak_memory_stats()
    rows = [
        benchmark_token_count(
            int(tokens),
            width=int(architecture["model_width"]),
            hidden=int(architecture["dense_hidden_width"]),
            experts=int(architecture["experts"]),
            expert_width=int(architecture["expert_hidden_width"]),
            warmups=int(systems["warmup_trials"]),
            trials=int(systems["timed_trials"]),
            device=args.device,
        )
        for tokens in systems["token_counts"]
    ]
    threshold = float(
        systems["maximum_candidate_over_dense_latency_each_regime_each_token_count"]
    )
    row_gates = [
        {
            "tokens": row["tokens"],
            "skewed_pass": row["skewed_over_dense"] <= threshold,
            "balanced_pass": row["balanced_over_dense"] <= threshold,
            "lower_bound_pass": row[
                "prepartitioned_lower_bound_over_dense"
            ]
            <= threshold,
        }
        for row in rows
    ]
    peak = int(torch.cuda.max_memory_allocated())
    gate = {
        "threshold": threshold,
        "row_gates": row_gates,
        "peak_cuda_bytes": peak,
        "peak_cuda_pass": peak <= int(systems["maximum_peak_cuda_bytes"]),
    }
    gate["all_gates_pass"] = bool(
        gate["peak_cuda_pass"]
        and all(
            row["skewed_pass"] and row["balanced_pass"]
            for row in row_gates
        )
    )
    gate["decision"] = (
        "PROMOTE_H49_TO_FUNCTION_JVP_AUDIT"
        if gate["all_gates_pass"]
        else "REJECT_H49_REFERENCE_TOP1_DISPATCH"
    )
    source = Path(__file__).resolve()
    result = {
        "schema_version": SCHEMA_VERSION,
        "source_commit": git_commit(source.parents[2]),
        "source_sha256": file_sha256(source),
        "plan": str(args.plan),
        "plan_sha256": file_sha256(args.plan),
        "device": torch.cuda.get_device_name(),
        "accounting": accounting,
        "rows": rows,
        "gate": gate,
        "runtime_seconds": time.time() - started,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    result_path = args.output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if result_path.stat().st_size > int(systems["maximum_result_bytes"]):
        raise RuntimeError("H49 result exceeds frozen size gate")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
