#!/usr/bin/env python3
"""Fit compact shared Kronecker dictionaries to temporal MLP state bases.

For a matrix A with row index (r_o,r_i) and column index (c_o,c_i), the
Van-Loan rearrangement R(A) has shape (r_o*c_o, r_i*c_i).  A Kronecker atom
P kron Q becomes one rank-one matrix in that rearranged domain.  Fitting all
temporal PCs jointly as

    R(A_k) ~= L C_k R^T

shares learned left/right atom dictionaries across the complete state basis,
while each PC keeps only a small core.  Unlike matrix SVD truncation, every
Kronecker atom may itself be full matrix-rank.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_mlp_highcadence_basis import file_sha256
from examples.nanogpt.analyze_mlp_tangent_drift import temporal_basis
from examples.nanogpt.analyze_parameter_trajectory import (
    PARAMETER_PATTERN,
    load_snapshots,
    parse_int_list,
    write_csv,
)


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def kron_rearrange(
    matrices: torch.Tensor,
    *,
    row_outer: int,
    row_inner: int,
    column_outer: int,
    column_inner: int,
) -> torch.Tensor:
    """Map [..., ro*ri, co*ci] to [..., ro*co, ri*ci]."""
    if matrices.shape[-2:] != (
        row_outer * row_inner,
        column_outer * column_inner,
    ):
        raise ValueError("matrix shape does not match Kronecker factorization")
    prefix = matrices.shape[:-2]
    return (
        matrices.reshape(
            *prefix, row_outer, row_inner, column_outer, column_inner
        )
        .permute(
            *range(len(prefix)),
            len(prefix),
            len(prefix) + 2,
            len(prefix) + 1,
            len(prefix) + 3,
        )
        .contiguous()
        .reshape(*prefix, row_outer * column_outer, row_inner * column_inner)
    )


def kron_unrearrange(
    rearranged: torch.Tensor,
    *,
    row_outer: int,
    row_inner: int,
    column_outer: int,
    column_inner: int,
) -> torch.Tensor:
    prefix = rearranged.shape[:-2]
    return (
        rearranged.reshape(
            *prefix, row_outer, column_outer, row_inner, column_inner
        )
        .permute(
            *range(len(prefix)),
            len(prefix),
            len(prefix) + 2,
            len(prefix) + 1,
            len(prefix) + 3,
        )
        .contiguous()
        .reshape(
            *prefix,
            row_outer * row_inner,
            column_outer * column_inner,
        )
    )


def initial_mode_bases(
    tensor: torch.Tensor, maximum_rank: int, *, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    modes, left_size, right_size = tensor.shape
    generator_state = torch.random.get_rng_state()
    cuda_state = (
        torch.cuda.get_rng_state(tensor.device)
        if tensor.is_cuda
        else None
    )
    torch.manual_seed(seed)
    left_unfold = tensor.permute(1, 0, 2).reshape(left_size, modes * right_size)
    right_unfold = tensor.permute(2, 0, 1).reshape(right_size, modes * left_size)
    left_q = min(maximum_rank + 4, min(left_unfold.shape))
    right_q = min(maximum_rank + 4, min(right_unfold.shape))
    left, _, _ = torch.pca_lowrank(
        left_unfold, q=left_q, center=False, niter=6
    )
    right, _, _ = torch.pca_lowrank(
        right_unfold, q=right_q, center=False, niter=6
    )
    torch.random.set_rng_state(generator_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, tensor.device)
    return left[:, :maximum_rank], right[:, :maximum_rank]


def hooi_fit(
    tensor: torch.Tensor,
    *,
    left_rank: int,
    right_rank: int,
    left_initial: torch.Tensor,
    right_initial: torch.Tensor,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Fit a Tucker-(uncompressed temporal, p, q) basis by HOOI."""
    modes, left_size, right_size = tensor.shape
    left = left_initial[:, :left_rank].contiguous()
    right = right_initial[:, :right_rank].contiguous()
    for _ in range(iterations):
        projected_left = torch.einsum("kmn,nq->kmq", tensor, right)
        projected_left = projected_left.permute(1, 0, 2).reshape(
            left_size, modes * right_rank
        )
        left = torch.linalg.svd(
            projected_left, full_matrices=False
        ).U[:, :left_rank]
        projected_right = torch.einsum("kmn,mp->knp", tensor, left)
        projected_right = projected_right.permute(1, 0, 2).reshape(
            right_size, modes * left_rank
        )
        right = torch.linalg.svd(
            projected_right, full_matrices=False
        ).U[:, :right_rank]
    core = torch.einsum("mp,kmn,nq->kpq", left, tensor, right)
    capture = float(
        core.double().square().sum()
        / tensor.double().square().sum().clamp_min(1e-30)
    )
    return left, right, core, capture


def materialize_kron_operator(
    left_atoms: torch.Tensor,
    right_atoms: torch.Tensor,
    core: torch.Tensor,
) -> torch.Tensor:
    # Rearranged operator is L C R^T; undoing the rearrangement gives the
    # corresponding sum of Kronecker products.
    rearranged = torch.einsum("mp,pq,nq->mn", left_atoms, core, right_atoms)
    return rearranged


def direct_kron_apply(
    inputs: torch.Tensor,
    left_atoms: torch.Tensor,
    right_atoms: torch.Tensor,
    core: torch.Tensor,
) -> torch.Tensor:
    """Apply sum_ab core[a,b] (A_a kron B_b) without materializing W."""
    # inputs: [tokens, c_o, c_i]
    # left_atoms: [p, r_o, c_o], right_atoms: [q, r_i, c_i]
    partial = torch.einsum("prc,tcj->tprj", left_atoms, inputs)
    return torch.einsum("tprj,pq,qij->tri", partial, core, right_atoms)


def cuda_benchmark(callable_, *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        callable_()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        callable_()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iterations)


def target_factorization(target: str) -> tuple[int, int, int, int]:
    if target == "mlp.c_fc":
        return 64, 48, 24, 32
    if target == "mlp.c_proj":
        return 24, 32, 64, 48
    raise ValueError(f"unsupported target {target}")


def benchmark_fit(
    *,
    left: torch.Tensor,
    right: torch.Tensor,
    core: torch.Tensor,
    factorization: tuple[int, int, int, int],
    tokens: int,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    row_outer, row_inner, column_outer, column_inner = factorization
    dtype = torch.bfloat16
    left_matrices = (
        left.T.reshape(left.shape[1], row_outer, column_outer).to(dtype)
    )
    right_matrices = (
        right.T.reshape(right.shape[1], row_inner, column_inner).to(dtype)
    )
    dynamic_core = core[0].to(dtype)
    inputs = torch.randn(
        (tokens, column_outer, column_inner),
        device=left.device,
        dtype=dtype,
    )
    rearranged = materialize_kron_operator(
        left.flatten(0, 0), right.flatten(0, 0), core[0]
    )
    dense_weight = kron_unrearrange(
        rearranged,
        row_outer=row_outer,
        row_inner=row_inner,
        column_outer=column_outer,
        column_inner=column_inner,
    ).to(dtype)

    def direct_call() -> torch.Tensor:
        return direct_kron_apply(
            inputs, left_matrices, right_matrices, dynamic_core
        )

    flat_inputs = inputs.flatten(1)

    def dense_call() -> torch.Tensor:
        return flat_inputs @ dense_weight.T

    direct_output = direct_call().flatten(1)
    dense_output = dense_call()
    denominator = dense_output.float().abs().max().clamp_min(1e-12)
    relative_max_error = float(
        (direct_output.float() - dense_output.float()).abs().max() / denominator
    )
    direct_ms = cuda_benchmark(direct_call, warmup=warmup, iterations=iterations)
    dense_ms = cuda_benchmark(dense_call, warmup=warmup, iterations=iterations)
    return {
        "benchmark_tokens": tokens,
        "direct_apply_ms": direct_ms,
        "dense_matmul_ms": dense_ms,
        "measured_dense_over_direct_throughput": dense_ms / direct_ms,
        "relative_max_output_error_bf16": relative_max_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--targets", default="mlp.c_fc,mlp.c_proj")
    parser.add_argument("--basis-rank", type=int, default=16)
    parser.add_argument("--dictionary-ranks", default="1,2,3,4,5,6,7")
    parser.add_argument("--hooi-iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=260826)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--benchmark-tokens", type=int, default=2048)
    parser.add_argument("--benchmark-warmup", type=int, default=5)
    parser.add_argument("--benchmark-iterations", type=int, default=20)
    args = parser.parse_args()
    started = time.time()
    targets = {item for item in args.targets.split(",") if item}
    ranks = parse_int_list(args.dictionary_ranks)
    paths = sorted(args.snapshot_dir.glob("step_*.pt"))
    steps, values, snapshot_metadata = load_snapshots(
        paths, layers={args.layer}, targets=targets
    )
    fit_rows: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []
    factors: dict[str, Any] = {}
    for parameter, tensors in sorted(values.items()):
        match = PARAMETER_PATTERN.match(parameter)
        if match is None:
            raise ValueError(f"unsupported parameter {parameter}")
        target = match.group("target")
        positions = torch.stack(tensors).to(args.device, dtype=torch.float32)
        _, all_values, basis = temporal_basis(
            positions.flatten(1), maximum_rank=args.basis_rank
        )
        retained_rank = basis.shape[1]
        retained_values = all_values[:retained_rank]
        normalized_values = retained_values / retained_values.sum().clamp_min(1e-30)
        basis_matrices = basis[:, :retained_rank].T.reshape(
            retained_rank, *positions.shape[1:]
        )
        factorization = target_factorization(target)
        tensor = kron_rearrange(
            basis_matrices,
            row_outer=factorization[0],
            row_inner=factorization[1],
            column_outer=factorization[2],
            column_inner=factorization[3],
        ) * normalized_values.sqrt().to(basis_matrices.dtype).view(-1, 1, 1)
        maximum_rank = max(ranks)
        left_initial, right_initial = initial_mode_bases(
            tensor, maximum_rank, seed=args.seed
        )
        parameter_size = positions[0].numel()
        row_outer, row_inner, column_outer, column_inner = factorization
        per_atom_flop_ratio = 1.0 / row_inner + 1.0 / column_outer
        target_factors: dict[int, Any] = {}
        for rank in ranks:
            left, right, core, capture = hooi_fit(
                tensor,
                left_rank=rank,
                right_rank=rank,
                left_initial=left_initial,
                right_initial=right_initial,
                iterations=args.hooi_iterations,
            )
            stored = left.numel() + right.numel() + core.numel()
            atom_count = rank * rank
            fit_rows.append(
                {
                    "parameter": parameter,
                    "layer": args.layer,
                    "target": target,
                    "temporal_basis_rank": retained_rank,
                    "temporal_basis_energy_fraction": float(
                        retained_values.sum() / all_values.sum().clamp_min(1e-30)
                    ),
                    "left_dictionary_rank": rank,
                    "right_dictionary_rank": rank,
                    "kronecker_atom_count": atom_count,
                    "stored_scalars": stored,
                    "stored_scalar_fraction": stored / parameter_size,
                    "weighted_state_basis_energy_capture": capture,
                    "ideal_direct_apply_flop_ratio_vs_dense": atom_count
                    * per_atom_flop_ratio,
                    "materialization_madd_per_generated_weight": atom_count,
                    "row_outer": row_outer,
                    "row_inner": row_inner,
                    "column_outer": column_outer,
                    "column_inner": column_inner,
                }
            )
            target_factors[rank] = {
                "left": left.detach().cpu(),
                "right": right.detach().cpu(),
                "core": core.detach().cpu(),
                "capture": capture,
            }
            if str(args.device).startswith("cuda") and args.benchmark_tokens > 0:
                measured = benchmark_fit(
                    left=left,
                    right=right,
                    core=core,
                    factorization=factorization,
                    tokens=args.benchmark_tokens,
                    warmup=args.benchmark_warmup,
                    iterations=args.benchmark_iterations,
                )
                benchmark_rows.append(
                    {
                        "parameter": parameter,
                        "dictionary_rank": rank,
                        "kronecker_atom_count": atom_count,
                        "ideal_direct_apply_flop_ratio_vs_dense": atom_count
                        * per_atom_flop_ratio,
                        **measured,
                    }
                )
        factors[target] = {
            "parameter": parameter,
            "factorization": factorization,
            "temporal_eigenvalues": retained_values.detach().cpu(),
            "ranks": target_factors,
        }
        del positions, basis_matrices, tensor
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

    args.output.mkdir(parents=True, exist_ok=True)
    fit_path = args.output / "kron_tucker_fit.csv"
    benchmark_path = args.output / "direct_apply_benchmark.csv"
    factors_path = args.output / "dictionary_factors.pt"
    write_csv(fit_path, fit_rows)
    write_csv(benchmark_path, benchmark_rows)
    torch.save(factors, factors_path)
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_mlp_kron_tucker_basis_v1",
        "steps": steps,
        "snapshot_metadata": snapshot_metadata,
        "layer": args.layer,
        "targets": sorted(targets),
        "basis_rank": args.basis_rank,
        "dictionary_ranks": ranks,
        "method": {
            "fit": "full-horizon variance-weighted Tucker HOOI in balanced Van-Loan rearrangement",
            "factorization_c_fc": [64, 48, 24, 32],
            "factorization_c_proj": [24, 32, 64, 48],
            "benchmark": "BF16 forward-only direct Kronecker contraction against the exactly materialized same operator",
        },
        "analysis_execution": {
            "git_commit": git_commit(script.parents[2]),
            "entrypoint": str(script),
            "entrypoint_sha256": file_sha256(script),
            "command": sys.argv,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "device": args.device,
        },
        "outputs": {
            fit_path.name: file_sha256(fit_path),
            benchmark_path.name: file_sha256(benchmark_path),
            factors_path.name: file_sha256(factors_path),
        },
        "limitations": [
            "Full-horizon state-basis fitting is a noncausal representation ceiling.",
            "Forward microbenchmarks do not include backward or optimizer cost.",
            "HOOI is locally optimized and may underestimate the best Tucker fit.",
            "Euclidean state-basis capture is not fixed-evaluation CE.",
        ],
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"fit_rows": len(fit_rows), "benchmark_rows": len(benchmark_rows), "metadata": str(metadata_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
