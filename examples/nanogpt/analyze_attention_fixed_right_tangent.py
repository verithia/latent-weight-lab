#!/usr/bin/env python3
"""Measure the best dense-direction fit of a truly fixed-right Cayley chart.

The trained attention repair updates both low-rank Cayley factors.  This
oracle freezes the exact seeded right frames and solves the linearized
least-squares problem for only the left coordinates.  It therefore measures
the representational ceiling of the fixed Mapping-Network-style chart,
independently of AdamW or Muon update heuristics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_attention_cayley_factor_optimizer import (
    TARGETS,
    direction_metrics,
    make_charts,
    write_csv,
)
from examples.nanogpt.analyze_parameter_trajectory import parse_int_list
from examples.nanogpt.parameter_trajectory import OPTIMIZER_PROBE_SCHEMA_VERSION


Coordinates = tuple[torch.Tensor | None, torch.Tensor | None]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def coordinate_dot(first: Coordinates, second: Coordinates) -> torch.Tensor:
    terms = [
        (left * right).sum()
        for left, right in zip(first, second, strict=True)
        if left is not None and right is not None
    ]
    if not terms:
        raise ValueError("empty Cayley coordinate tuple")
    return sum(terms[1:], terms[0])


def coordinate_add(
    first: Coordinates, second: Coordinates, *, alpha: float | torch.Tensor
) -> Coordinates:
    return tuple(
        None if left is None else left + alpha * right
        for left, right in zip(first, second, strict=True)
    )  # type: ignore[return-value]


def fixed_right_tangent(
    *,
    weight: torch.Tensor,
    input_right: torch.Tensor | None,
    output_right: torch.Tensor | None,
    coordinates: Coordinates,
) -> torch.Tensor:
    input_left, output_left = coordinates
    result = torch.zeros_like(weight)
    if input_right is not None:
        assert input_left is not None
        skew = input_left @ input_right.T - input_right @ input_left.T
        result = result - 2.0 * (weight @ skew)
    if output_right is not None:
        assert output_left is not None
        skew = output_left @ output_right.T - output_right @ output_left.T
        result = result - 2.0 * (skew @ weight)
    return result


def fixed_right_adjoint(
    *,
    weight: torch.Tensor,
    input_right: torch.Tensor | None,
    output_right: torch.Tensor | None,
    direction: torch.Tensor,
) -> Coordinates:
    input_gradient = (
        2.0 * (direction.T @ weight - weight.T @ direction) @ input_right
        if input_right is not None
        else None
    )
    output_gradient = (
        2.0 * (weight @ direction.T - direction @ weight.T) @ output_right
        if output_right is not None
        else None
    )
    return input_gradient, output_gradient


def project_fixed_right_tangent(
    *,
    weight: torch.Tensor,
    input_right: torch.Tensor | None,
    output_right: torch.Tensor | None,
    target: torch.Tensor,
    maximum_iterations: int,
    tolerance: float,
    ridge: float,
) -> tuple[torch.Tensor, dict[str, float | int | bool]]:
    if maximum_iterations <= 0 or tolerance <= 0.0 or ridge < 0.0:
        raise ValueError("invalid conjugate-gradient controls")
    right_frames = (input_right, output_right)
    zero: Coordinates = tuple(
        None if right is None else torch.zeros_like(right)
        for right in right_frames
    )  # type: ignore[assignment]
    solution = zero
    residual = fixed_right_adjoint(
        weight=weight,
        input_right=input_right,
        output_right=output_right,
        direction=target,
    )
    search = tuple(
        None if value is None else value.clone() for value in residual
    )  # type: ignore[assignment]
    residual_squared = coordinate_dot(residual, residual)
    initial_squared = residual_squared.clone()
    converged = float(initial_squared) == 0.0
    iterations = 0
    for iteration in range(maximum_iterations):
        if converged:
            break
        mapped = fixed_right_tangent(
            weight=weight,
            input_right=input_right,
            output_right=output_right,
            coordinates=search,
        )
        normal = fixed_right_adjoint(
            weight=weight,
            input_right=input_right,
            output_right=output_right,
            direction=mapped,
        )
        if ridge:
            normal = coordinate_add(normal, search, alpha=ridge)
        denominator = coordinate_dot(search, normal)
        if float(denominator) <= 0.0:
            break
        step = residual_squared / denominator
        solution = coordinate_add(solution, search, alpha=step)
        updated = coordinate_add(residual, normal, alpha=-step)
        updated_squared = coordinate_dot(updated, updated)
        iterations = iteration + 1
        relative = torch.sqrt(
            updated_squared / initial_squared.clamp_min(torch.finfo(torch.float64).tiny)
        )
        if float(relative) <= tolerance:
            residual = updated
            residual_squared = updated_squared
            converged = True
            break
        beta = updated_squared / residual_squared.clamp_min(
            torch.finfo(torch.float64).tiny
        )
        search = coordinate_add(updated, search, alpha=beta)
        residual = updated
        residual_squared = updated_squared

    projected = fixed_right_tangent(
        weight=weight,
        input_right=input_right,
        output_right=output_right,
        coordinates=solution,
    )
    residual_direction = target - projected
    return projected, {
        "cg_iterations": iterations,
        "cg_converged": converged,
        "cg_relative_normal_residual": float(
            torch.sqrt(
                residual_squared
                / initial_squared.clamp_min(torch.finfo(torch.float64).tiny)
            )
        ),
        "projection_residual_dot_fraction": float(
            (residual_direction * projected).sum().abs()
            / (target.norm() * projected.norm()).clamp_min(1e-30)
        ),
    }


def fixed_frames(
    *,
    weight: torch.Tensor,
    rank: int,
    base_seed: int,
    layer: int,
    target: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    input_chart, output_chart = make_charts(
        weight=weight,
        rank=rank,
        base_seed=base_seed,
        layer=layer,
        target=target,
    )
    frames = []
    for chart in (input_chart, output_chart):
        frames.append(
            None
            if chart is None
            else torch.nn.functional.normalize(
                chart.right.detach().reshape(chart.features, chart.rank), dim=0
            ).to(device=weight.device, dtype=weight.dtype)
        )
    return frames[0], frames[1]


def weighted_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    weights = torch.tensor(
        [float(row["target_fro"]) ** 2 for row in rows], dtype=torch.float64
    )

    def weighted(key: str) -> float:
        values = torch.tensor(
            [float(row[key]) for row in rows], dtype=torch.float64
        )
        return float((weights * values).sum() / weights.sum())

    return {
        "cells": len(rows),
        "cosine": weighted("cosine"),
        "positive_line_recovery": weighted("positive_line_recovery"),
        "projected_energy_fraction": weighted("projected_energy_fraction"),
        "projection_residual_dot_fraction": weighted(
            "projection_residual_dot_fraction"
        ),
        "maximum_cg_relative_normal_residual": max(
            float(row["cg_relative_normal_residual"]) for row in rows
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--production-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", default="0,3,6,9,11")
    parser.add_argument("--steps", default="2372")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cg-iterations", type=int, default=80)
    parser.add_argument("--cg-tolerance", type=float, default=1e-7)
    parser.add_argument("--ridge", type=float, default=1e-12)
    args = parser.parse_args()
    started = time.time()
    layers = parse_int_list(args.layers)
    steps = parse_int_list(args.steps)
    config = json.loads(args.production_config.read_text())
    ranks = {
        str(key): int(value)
        for key, value in config["block_fht_attn_cayley_ranks"].items()
    }
    base_seed = int(config["block_fht_attn_cayley_seed"])
    probe_paths = [args.probe_dir / f"step_{step:06d}.pt" for step in steps]
    if any(not path.is_file() for path in probe_paths):
        raise ValueError("required optimizer probe is absent")

    rows: list[dict[str, Any]] = []
    run_identity_sha256: str | None = None
    for path in probe_paths:
        probe = torch.load(path, map_location="cpu", weights_only=False)
        if probe.get("schema_version") != OPTIMIZER_PROBE_SCHEMA_VERSION:
            raise ValueError("unexpected optimizer probe schema")
        if run_identity_sha256 is None:
            run_identity_sha256 = probe["run_identity_sha256"]
        elif probe["run_identity_sha256"] != run_identity_sha256:
            raise ValueError("optimizer probes do not share one identity")
        n_embd = int(probe["model_config"]["n_embd"])
        step = int(probe["step"])
        for layer in layers:
            entries = (
                (
                    "qk_shared",
                    f"transformer.h.{layer}.attn.c_attn.weight",
                    slice(0, 2 * n_embd),
                ),
                (
                    "v",
                    f"transformer.h.{layer}.attn.c_attn.weight",
                    slice(2 * n_embd, None),
                ),
                (
                    "cproj",
                    f"transformer.h.{layer}.attn.c_proj.weight",
                    slice(None),
                ),
            )
            for target_name, parameter_name, selection in entries:
                record = probe["parameters"][parameter_name]
                weight = record["weight_before_step"][selection].to(
                    args.device, dtype=torch.float64
                )
                dense_direction = record["applied_direction_per_lr"][selection].to(
                    args.device, dtype=torch.float64
                )
                rank_key = str(TARGETS[target_name]["rank_key"])
                rank = ranks[rank_key]
                input_right, output_right = fixed_frames(
                    weight=weight,
                    rank=rank,
                    base_seed=base_seed,
                    layer=layer,
                    target=target_name,
                )
                projected, diagnostics = project_fixed_right_tangent(
                    weight=weight,
                    input_right=input_right,
                    output_right=output_right,
                    target=dense_direction,
                    maximum_iterations=args.cg_iterations,
                    tolerance=args.cg_tolerance,
                    ridge=args.ridge,
                )
                metrics = direction_metrics(dense_direction, projected)
                row = {
                    "step": step,
                    "layer": layer,
                    "target": target_name,
                    "rank": rank,
                    **metrics,
                    "projected_energy_fraction": float(
                        projected.square().sum()
                        / dense_direction.square().sum().clamp_min(1e-30)
                    ),
                    **diagnostics,
                }
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
        del probe
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    args.output.mkdir(parents=True, exist_ok=True)
    cells_path = args.output / "attention_fixed_right_tangent_cells.csv"
    write_csv(cells_path, rows)
    by_target = {
        target: weighted_summary([row for row in rows if row["target"] == target])
        for target in TARGETS
    }
    result = {
        "schema_version": "mai_124m_attention_fixed_right_tangent_v1",
        "scientific_question": (
            "How much exact dense Muon direction can the seeded fixed-right "
            "QK32/V16/c-proj8 Cayley tangent represent?"
        ),
        "source_commit": git_commit(Path(__file__).resolve().parents[2]),
        "source_sha256": file_sha256(Path(__file__)),
        "production_config": {
            "path": str(args.production_config),
            "sha256": file_sha256(args.production_config),
        },
        "optimizer_probe_run_identity_sha256": run_identity_sha256,
        "optimizer_probe_paths": [
            {"path": str(path), "sha256": file_sha256(path)}
            for path in probe_paths
        ],
        "layers": layers,
        "steps": steps,
        "ranks": ranks,
        "solver": {
            "maximum_iterations": args.cg_iterations,
            "tolerance": args.cg_tolerance,
            "ridge": args.ridge,
        },
        "aggregate": weighted_summary(rows),
        "by_target": by_target,
        "outputs": {"cells_sha256": file_sha256(cells_path)},
        "elapsed_seconds": time.time() - started,
        "limitations": [
            "This is a linearized identity-chart representational ceiling, not a training result.",
            "The right frames are seeded and fixed; no dense-trajectory direction is used to choose a basis.",
            "Only the registered layers and steps are sampled."
        ],
    }
    result_path = args.output / "attention_fixed_right_tangent_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["aggregate"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
