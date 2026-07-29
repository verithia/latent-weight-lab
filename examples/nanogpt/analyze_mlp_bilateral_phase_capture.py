#!/usr/bin/env python3
"""Project dense c_proj phase chords onto the production bilateral tangent.

This no-update diagnostic asks a representation question, not an optimizer
question: at the exact identity point, how much of each dense-Muon phase
chord can the selected fixed-FHT Cayley chart express?  The projection is
computed matrix-free with exact JVP/VJP products and a damped conjugate-
gradient solve of ``(J^T J + damping I) x = J^T target``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.func import functional_call, jvp, vjp

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.nanogpt.analyze_parameter_trajectory import (
    load_snapshots,
    parse_int_list,
    write_csv,
)
from examples.nanogpt.model import LearnedFHTBlockOrthogonalOutputMix


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


class BilateralWeightChart(torch.nn.Module):
    """Exact production c_proj chart, isolated from the language model."""

    def __init__(
        self,
        hidden_features: int,
        output_features: int,
        *,
        hidden_stages: int,
        output_stages: int,
        rotation_block_size: int,
        basis_block_size: int,
        hidden_seed: int,
        output_seed: int,
        coordinate_scale: float,
        gain_scale: float,
    ) -> None:
        super().__init__()
        self.hidden_rotation = LearnedFHTBlockOrthogonalOutputMix(
            features=hidden_features,
            stages=hidden_stages,
            rotation_block_size=rotation_block_size,
            basis_block_size=basis_block_size,
            seed=hidden_seed,
            coordinate_scale=coordinate_scale,
        )
        self.hidden_log_gain = torch.nn.Parameter(
            torch.zeros(hidden_features)
        )
        self.output_rotation = LearnedFHTBlockOrthogonalOutputMix(
            features=output_features,
            stages=output_stages,
            rotation_block_size=rotation_block_size,
            basis_block_size=basis_block_size,
            seed=output_seed,
            coordinate_scale=coordinate_scale,
        )
        self.output_log_gain = torch.nn.Parameter(
            torch.zeros(output_features)
        )
        self.gain_scale = float(gain_scale)

    def forward(self, base_weight: torch.Tensor) -> torch.Tensor:
        hidden_rotation = self.hidden_rotation.matrix(base_weight)
        charted = base_weight @ hidden_rotation.transpose(0, 1)
        hidden_gain = (
            self.gain_scale * self.hidden_log_gain
        ).exp().to(device=base_weight.device, dtype=base_weight.dtype)
        charted = charted * hidden_gain
        transposed = charted.transpose(0, 1)
        output_gain = (
            self.gain_scale * self.output_log_gain
        ).exp().to(device=base_weight.device, dtype=base_weight.dtype)
        transposed = transposed * output_gain
        output_rotation = self.output_rotation.matrix(transposed)
        transposed = transposed @ output_rotation
        return transposed.transpose(0, 1).contiguous()


class FixedBaseChartView(torch.nn.Module):
    def __init__(
        self,
        chart: BilateralWeightChart,
        base_weight: torch.Tensor,
    ) -> None:
        super().__init__()
        self.chart = chart
        self.register_buffer("base_weight", base_weight)

    def forward(self) -> torch.Tensor:
        return self.chart(self.base_weight)


def _tuple_dot(
    left: tuple[torch.Tensor, ...],
    right: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    return sum(
        (left_value * right_value).sum()
        for left_value, right_value in zip(left, right, strict=True)
    )


def singular_frame_components(
    base: torch.Tensor,
    delta: torch.Tensor,
) -> dict[str, torch.Tensor]:
    u, _, vh = torch.linalg.svd(base, full_matrices=False)
    core = u.transpose(0, 1) @ delta @ vh.transpose(0, 1)
    diagonal_values = torch.diagonal(core)
    singular_value = (u * diagonal_values.unsqueeze(0)) @ vh
    occupied = u @ core @ vh
    mixing = occupied - singular_value
    rotation = delta - occupied
    return {
        "total": delta,
        "singular_value": singular_value,
        "in_frame_mixing": mixing,
        "subspace_rotation": rotation,
    }


def project_target(
    chart: BilateralWeightChart,
    base_weight: torch.Tensor,
    target: torch.Tensor,
    *,
    damping_ratio: float,
    cg_steps: int,
    trace_seed: int,
) -> dict[str, float]:
    """Project one matrix target onto the chart identity Jacobian."""
    if not math.isfinite(damping_ratio) or damping_ratio <= 0.0:
        raise ValueError("damping_ratio must be positive and finite")
    if cg_steps <= 0:
        raise ValueError("cg_steps must be positive")
    if base_weight.shape != target.shape or target.ndim != 2:
        raise ValueError("base_weight and target must be same-shaped matrices")
    target_energy = target.square().sum()
    if float(target_energy) <= 0.0:
        raise ValueError("projection target must be nonzero")

    view = FixedBaseChartView(chart, base_weight)
    named = dict(view.named_parameters())
    names = sorted(named)
    primals = tuple(named[name].detach() for name in names)

    def materialize(*coordinates: torch.Tensor) -> torch.Tensor:
        replacements = {
            name: value
            for name, value in zip(names, coordinates, strict=True)
        }
        return functional_call(view, replacements, (), strict=False)

    identity_weight, pullback = vjp(materialize, *primals)
    identity_error = (identity_weight - base_weight).norm()
    if float(identity_error) > 1e-10 * float(base_weight.norm()):
        raise RuntimeError(
            f"bilateral chart is not identity: relative error "
            f"{float(identity_error / base_weight.norm().clamp_min(1e-30))}"
        )
    rhs = tuple(value.detach() for value in pullback(target))
    coordinate_count = sum(value.numel() for value in primals)
    generator = torch.Generator(device=primals[0].device)
    generator.manual_seed(int(trace_seed))
    probe = tuple(
        (
            torch.randint(
                0,
                2,
                value.shape,
                device=value.device,
                generator=generator,
            ).to(dtype=value.dtype)
            * 2.0
            - 1.0
        )
        for value in primals
    )
    _, probe_image = jvp(materialize, primals, probe)
    mean_eigenvalue = (
        probe_image.square().sum() / float(coordinate_count)
    )
    damping = (
        float(damping_ratio) * mean_eigenvalue.clamp_min(1e-30)
    )

    def system(
        direction: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        _, image = jvp(materialize, primals, direction)
        pulled = pullback(image)
        return tuple(
            value.detach() + damping * direction_value
            for value, direction_value in zip(
                pulled, direction, strict=True
            )
        )

    solution = tuple(torch.zeros_like(value) for value in rhs)
    residual = tuple(value.clone() for value in rhs)
    conjugate = tuple(value.clone() for value in residual)
    initial_norm_squared = _tuple_dot(residual, residual).clamp_min(1e-30)
    norm_squared = initial_norm_squared
    completed_steps = 0
    for step in range(cg_steps):
        image = system(conjugate)
        denominator = _tuple_dot(conjugate, image)
        if not torch.isfinite(denominator) or denominator <= 0:
            break
        alpha = norm_squared / denominator
        solution = tuple(
            value + alpha * direction
            for value, direction in zip(
                solution, conjugate, strict=True
            )
        )
        residual = tuple(
            value - alpha * image_value
            for value, image_value in zip(
                residual, image, strict=True
            )
        )
        next_norm_squared = _tuple_dot(residual, residual)
        completed_steps = step + 1
        if next_norm_squared <= initial_norm_squared * 1e-16:
            norm_squared = next_norm_squared
            break
        beta = next_norm_squared / norm_squared.clamp_min(1e-30)
        conjugate = tuple(
            residual_value + beta * direction
            for residual_value, direction in zip(
                residual, conjugate, strict=True
            )
        )
        norm_squared = next_norm_squared

    _, projected = jvp(materialize, primals, solution)
    error = target - projected
    projected_energy = projected.square().sum()
    dot = (target * projected).sum()
    return {
        "coordinate_count": float(coordinate_count),
        "mean_eigenvalue": float(mean_eigenvalue),
        "damping": float(damping),
        "cg_steps": float(completed_steps),
        "cg_relative_normal_residual": float(
            norm_squared.clamp_min(0).sqrt()
            / initial_norm_squared.sqrt()
        ),
        "target_fro": float(target_energy.sqrt()),
        "projected_fro": float(projected_energy.sqrt()),
        "recovered_energy_fraction": float(
            1.0 - error.square().sum() / target_energy
        ),
        "projected_energy_fraction": float(
            projected_energy / target_energy
        ),
        "target_projected_cosine": float(
            dot
            / (
                target_energy.sqrt()
                * projected_energy.sqrt().clamp_min(1e-30)
            )
        ),
        "residual_projected_cosine": float(
            (error * projected).sum()
            / (
                error.norm().clamp_min(1e-30)
                * projected_energy.sqrt().clamp_min(1e-30)
            )
        ),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        phase = f"{row['phase_start']}->{row['phase_end']}"
        groups.setdefault((str(row["component"]), phase), []).append(row)
        groups.setdefault((str(row["component"]), "all_phases"), []).append(row)
    result: list[dict[str, Any]] = []
    for (component, phase), selected in sorted(groups.items()):
        energy = torch.tensor(
            [float(row["target_fro"]) ** 2 for row in selected],
            dtype=torch.float64,
        )
        recovery = torch.tensor(
            [float(row["recovered_energy_fraction"]) for row in selected],
            dtype=torch.float64,
        )
        normal_residual = torch.tensor(
            [float(row["cg_relative_normal_residual"]) for row in selected],
            dtype=torch.float64,
        )
        result.append(
            {
                "component": component,
                "phase": phase,
                "cells": len(selected),
                "energy_weighted_recovery": float(
                    (energy * recovery).sum() / energy.sum().clamp_min(1e-30)
                ),
                "mean_recovery": float(recovery.mean()),
                "median_recovery": float(recovery.median()),
                "minimum_recovery": float(recovery.min()),
                "maximum_recovery": float(recovery.max()),
                "mean_cg_relative_normal_residual": float(
                    normal_residual.mean()
                ),
                "maximum_cg_relative_normal_residual": float(
                    normal_residual.max()
                ),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", default="0,3,6,9,11")
    parser.add_argument("--phase-boundaries", default="0,60,120,180,238")
    parser.add_argument("--components", default="total,singular_value,in_frame_mixing,subspace_rotation")
    parser.add_argument("--hidden-stages", type=int, default=2)
    parser.add_argument("--output-stages", type=int, default=4)
    parser.add_argument("--rotation-block-size", type=int, default=32)
    parser.add_argument("--basis-block-size", type=int, default=256)
    parser.add_argument("--hidden-seed", type=int, default=314159)
    parser.add_argument("--output-seed", type=int, default=271828)
    parser.add_argument("--coordinate-scale", type=float, default=4.0)
    parser.add_argument("--gain-scale", type=float, default=4.0)
    parser.add_argument("--damping-ratio", type=float, default=1e-6)
    parser.add_argument("--cg-steps", type=int, default=24)
    parser.add_argument("--trace-seed", type=int, default=20260729)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    layers = parse_int_list(args.layers)
    boundaries = parse_int_list(args.phase_boundaries)
    components = [item for item in args.components.split(",") if item]
    allowed_components = {
        "total",
        "singular_value",
        "in_frame_mixing",
        "subspace_rotation",
    }
    if (
        not layers
        or len(boundaries) < 2
        or boundaries != sorted(set(boundaries))
        or not components
        or not set(components) <= allowed_components
    ):
        raise ValueError("invalid layers, phase boundaries, or components")
    paths = [args.snapshot_dir / f"step_{step:06d}.pt" for step in boundaries]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"phase-boundary snapshots are absent: {missing}")
    steps, values, snapshot_metadata = load_snapshots(
        paths,
        layers=set(layers),
        targets={"mlp.c_proj"},
    )
    if steps != boundaries:
        raise ValueError("loaded snapshot steps do not match phase boundaries")

    rows: list[dict[str, Any]] = []
    for name, tensors in sorted(values.items()):
        layer = int(name.split(".")[2])
        for phase_index, (start, end) in enumerate(
            zip(steps[:-1], steps[1:], strict=True)
        ):
            base = tensors[phase_index].to(
                device=args.device,
                dtype=torch.float64,
            )
            terminal = tensors[phase_index + 1].to(
                device=args.device,
                dtype=torch.float64,
            )
            component_targets = singular_frame_components(
                base,
                terminal - base,
            )
            for component in components:
                chart = BilateralWeightChart(
                    hidden_features=base.shape[1],
                    output_features=base.shape[0],
                    hidden_stages=args.hidden_stages,
                    output_stages=args.output_stages,
                    rotation_block_size=args.rotation_block_size,
                    basis_block_size=args.basis_block_size,
                    hidden_seed=args.hidden_seed + layer * 64,
                    output_seed=args.output_seed + layer * 64,
                    coordinate_scale=args.coordinate_scale,
                    gain_scale=args.gain_scale,
                ).to(device=args.device, dtype=torch.float64)
                metrics = project_target(
                    chart,
                    base,
                    component_targets[component],
                    damping_ratio=args.damping_ratio,
                    cg_steps=args.cg_steps,
                    trace_seed=(
                        args.trace_seed
                        + layer * 1009
                        + phase_index * 101
                        + components.index(component)
                    ),
                )
                row = {
                    "parameter": name,
                    "layer": layer,
                    "phase_start": start,
                    "phase_end": end,
                    "component": component,
                    **metrics,
                }
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
                del chart
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            del base, terminal, component_targets
    aggregates = aggregate_rows(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    detail_path = args.output / "bilateral_phase_capture.csv"
    aggregate_path = args.output / "bilateral_phase_capture_aggregate.csv"
    write_csv(detail_path, rows)
    write_csv(aggregate_path, aggregates)

    script = Path(__file__).resolve()
    repo = script.parents[2]
    metadata = {
        "schema_version": "nanogpt_mlp_bilateral_phase_capture_v1",
        "snapshot_metadata": snapshot_metadata,
        "layers": layers,
        "phase_boundaries": boundaries,
        "components": components,
        "chart": {
            "formula": "R_out^T D_out W_base R_hidden^T D_hidden",
            "hidden_stages": args.hidden_stages,
            "output_stages": args.output_stages,
            "rotation_block_size": args.rotation_block_size,
            "basis_block_size": args.basis_block_size,
            "hidden_seed": args.hidden_seed,
            "output_seed": args.output_seed,
            "layer_seed_stride": 64,
            "coordinate_scale": args.coordinate_scale,
            "gain_scale": args.gain_scale,
            "initialization": "exact identity; all rotation and log-gain coordinates zero",
            "learned_dense_basis": False,
            "lora_adapter": False,
        },
        "solver": {
            "system": "(J^T J + damping I) x = J^T target",
            "damping_ratio_to_hutchinson_mean_eigenvalue": args.damping_ratio,
            "cg_steps": args.cg_steps,
            "trace_seed": args.trace_seed,
            "analysis_dtype": "float64",
        },
        "analysis_execution": {
            "git_commit": git_commit(repo),
            "entrypoint": str(script),
            "entrypoint_sha256": file_sha256(script),
            "command": sys.argv,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "device": args.device,
        },
        "snapshot_files": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in paths
        ],
        "output": {
            "detail": {
                "path": str(detail_path),
                "sha256": file_sha256(detail_path),
            },
            "aggregate": {
                "path": str(aggregate_path),
                "sha256": file_sha256(aggregate_path),
            },
        },
        "limitations": [
            "This is local identity-Jacobian coverage, not nonlinear finite-chart reachability.",
            "Damped finite-step CG gives a conservative approximation to the least-squares projection.",
            "Only preregistered representative layers 0,3,6,9,11 are analyzed.",
            "Weight-space capture does not prove task-loss usefulness or optimizer direction selection.",
        ],
    }
    metadata_path = args.output / "bilateral_phase_capture_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "rows": len(rows),
                "aggregates": aggregates,
                "metadata": str(metadata_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
