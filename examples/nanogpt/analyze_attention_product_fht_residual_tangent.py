#!/usr/bin/env python3
"""Project dense attention motion into a zero-preserving product-FHT residual.

The candidate chart is an additive residual ``W(D) - W(0)``.  ``W`` is a
product of fixed seeded global FHT/sign mixers and learned diagonal factors.
At initialization the residual is exactly zero, while its tangent is nonlinear
in future states and contains no learned dense or low-rank basis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_parameter_trajectory import (
    load_snapshots,
    parse_int_list,
)
from examples.nanogpt.parameter_trajectory import OPTIMIZER_PROBE_SCHEMA_VERSION
from latent_weight_lab import ProductFHTLinear
from latent_weight_lab.block_fht import normalized_fht_last_dim, next_power_of_two


Coordinates = tuple[torch.Tensor, torch.Tensor]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def coordinate_dot(left: Coordinates, right: Coordinates) -> torch.Tensor:
    return (
        (left[0] * right[0]).flatten(1).sum(dim=1)
        + (left[1] * right[1]).flatten(1).sum(dim=1)
    )


def coordinate_add(
    left: Coordinates,
    right: Coordinates,
    alpha: torch.Tensor | float,
) -> Coordinates:
    if isinstance(alpha, torch.Tensor):
        diagonal_alpha = alpha[:, None, None]
        output_alpha = alpha[:, None]
    else:
        diagonal_alpha = output_alpha = alpha
    return (
        left[0] + diagonal_alpha * right[0],
        left[1] + output_alpha * right[1],
    )


class ProductFHTResidualTangent:
    """Exact batched JVP/adjoint at the zero-residual initialization."""

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        factors: int,
        seed: int,
        device: str,
    ) -> None:
        padded = next_power_of_two(max(in_features, out_features))
        self.module = ProductFHTLinear(
            in_features,
            out_features,
            factors=factors,
            seed=seed,
            weight_std=1.0 / math.sqrt(padded),
            bias=False,
            diagonal_scale=1.0,
            weight_space_muon=False,
            natural_gradient=False,
        ).to(device=device, dtype=torch.float32)
        self.in_features = in_features
        self.out_features = out_features
        self.factors = factors
        self.padded_features = padded
        self.scale = self.module.weight_std * math.sqrt(padded)
        signs = self.module.product_factor_signs.float()
        matrix = torch.eye(
            out_features, padded, device=device, dtype=torch.float32
        )
        stages: list[torch.Tensor] = []
        for factor in range(factors):
            matrix = normalized_fht_last_dim(matrix * signs[factor])
            stages.append(matrix)
        self.signs = signs
        self.stages = stages
        self.final_matrix = matrix

    @property
    def coordinate_count(self) -> int:
        return self.factors * self.padded_features + self.out_features

    @property
    def ambient_count(self) -> int:
        return self.out_features * self.in_features

    def zeros(self, batch: int) -> Coordinates:
        options = {
            "device": self.final_matrix.device,
            "dtype": self.final_matrix.dtype,
        }
        return (
            torch.zeros(batch, self.factors, self.padded_features, **options),
            torch.zeros(batch, self.out_features, **options),
        )

    def jvp(self, direction: Coordinates) -> torch.Tensor:
        diagonal_direction, output_direction = direction
        batch = diagonal_direction.shape[0]
        tangent = torch.zeros(
            batch,
            self.out_features,
            self.padded_features,
            device=self.final_matrix.device,
            dtype=self.final_matrix.dtype,
        )
        for factor in range(self.factors):
            tangent = normalized_fht_last_dim(
                tangent * self.signs[factor][None, None, :]
            )
            tangent = tangent + (
                self.stages[factor][None, :, :]
                * diagonal_direction[:, factor, None, :]
            )
        tangent = tangent + (
            output_direction[:, :, None] * self.final_matrix[None, :, :]
        )
        return self.scale * tangent[:, :, : self.in_features]

    def adjoint(self, target: torch.Tensor) -> Coordinates:
        batch = target.shape[0]
        cotangent = torch.zeros(
            batch,
            self.out_features,
            self.padded_features,
            device=target.device,
            dtype=target.dtype,
        )
        cotangent[:, :, : self.in_features] = self.scale * target
        output_gradient = (
            cotangent * self.final_matrix[None, :, :]
        ).sum(dim=2)
        diagonal_gradient = torch.zeros(
            batch,
            self.factors,
            self.padded_features,
            device=target.device,
            dtype=target.dtype,
        )
        for factor in reversed(range(self.factors)):
            diagonal_gradient[:, factor, :] = (
                cotangent * self.stages[factor][None, :, :]
            ).sum(dim=1)
            cotangent = normalized_fht_last_dim(cotangent)
            cotangent = cotangent * self.signs[factor][None, None, :]
        return diagonal_gradient, output_gradient

    def project(
        self,
        target: torch.Tensor,
        *,
        maximum_iterations: int,
        tolerance: float,
        ridge: float,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        target = target.to(
            device=self.final_matrix.device, dtype=torch.float32
        )
        solution = self.zeros(target.shape[0])
        residual = self.adjoint(target)
        search = (residual[0].clone(), residual[1].clone())
        residual_squared = coordinate_dot(residual, residual)
        initial_squared = residual_squared.clone().clamp_min(1e-30)
        iterations = 0
        for iteration in range(maximum_iterations):
            mapped = self.jvp(search)
            normal = self.adjoint(mapped)
            if ridge:
                normal = coordinate_add(normal, search, ridge)
            denominator = coordinate_dot(search, normal).clamp_min(1e-30)
            step = residual_squared / denominator
            solution = coordinate_add(solution, search, step)
            updated = coordinate_add(residual, normal, -step)
            updated_squared = coordinate_dot(updated, updated)
            iterations = iteration + 1
            if float(
                torch.sqrt(updated_squared / initial_squared).max()
            ) <= tolerance:
                residual = updated
                residual_squared = updated_squared
                break
            beta = updated_squared / residual_squared.clamp_min(1e-30)
            search = coordinate_add(updated, search, beta)
            residual = updated
            residual_squared = updated_squared
        projected = self.jvp(solution)
        remainder = target - projected
        target_norm = target.flatten(1).norm(dim=1).clamp_min(1e-30)
        projected_norm = projected.flatten(1).norm(dim=1).clamp_min(1e-30)
        dot = (target * projected).flatten(1).sum(dim=1)
        return projected, {
            "iterations": iterations,
            "relative_normal_residual": torch.sqrt(
                residual_squared / initial_squared
            ).detach().cpu(),
            "projection_orthogonality_error": (
                (remainder * projected).flatten(1).sum(dim=1).abs()
                / (target_norm * projected_norm)
            ).detach().cpu(),
            "target_norm": target_norm.detach().cpu(),
            "projected_norm": projected_norm.detach().cpu(),
            "cosine": (
                dot / (target_norm * projected_norm)
            ).detach().cpu(),
            "energy_recovery": (
                projected_norm.square() / target_norm.square()
            ).detach().cpu(),
        }


def weighted_summary(rows: list[dict[str, Any]], kind: str) -> dict[str, float]:
    selected = [row for row in rows if row["kind"] == kind]
    target_energy = sum(float(row["target_fro"]) ** 2 for row in selected)
    projected_energy = sum(
        float(row["projected_fro"]) ** 2 for row in selected
    )
    coordinate_fraction = sum(
        float(row["coordinate_fraction"])
        * float(row["target_fro"]) ** 2
        for row in selected
    ) / target_energy
    recovery = projected_energy / target_energy
    return {
        "cells": len(selected),
        "energy_recovery": recovery,
        "coordinate_fraction": coordinate_fraction,
        "normalized_enrichment": recovery / coordinate_fraction,
        "minimum_cell_recovery": min(
            float(row["energy_recovery"]) for row in selected
        ),
        "maximum_orthogonality_error": max(
            float(row["projection_orthogonality_error"])
            for row in selected
        ),
        "maximum_relative_normal_residual": max(
            float(row["relative_normal_residual"]) for row in selected
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--probe-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", default="0,3,6,9,11")
    parser.add_argument("--phase-boundaries", default="0,60,120,180,238")
    parser.add_argument("--probe-steps", default="0,60,120,180")
    parser.add_argument("--factors", default="6,12")
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--cg-iterations", type=int, default=100)
    parser.add_argument("--cg-tolerance", type=float, default=1e-6)
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    layers = parse_int_list(args.layers)
    boundaries = parse_int_list(args.phase_boundaries)
    probe_steps = parse_int_list(args.probe_steps)
    factors = parse_int_list(args.factors)
    phases = list(zip(boundaries[:-1], boundaries[1:], strict=True))
    if len(probe_steps) != len(phases):
        raise ValueError("one optimizer probe is required per phase")
    snapshot_paths = [
        args.snapshot_dir / f"step_{step:06d}.pt" for step in boundaries
    ]
    probe_paths = [
        args.probe_dir / f"step_{step:06d}.pt" for step in probe_steps
    ]
    missing = [
        str(path)
        for path in (*snapshot_paths, *probe_paths)
        if not path.is_file()
    ]
    if missing:
        raise ValueError("missing inputs: " + ", ".join(missing))
    steps, values, snapshot_metadata = load_snapshots(
        snapshot_paths,
        layers=set(layers),
        targets={"attn.c_attn", "attn.c_proj"},
    )
    if steps != boundaries:
        raise ValueError("snapshot boundaries do not match")
    probes = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in probe_paths
    ]
    if any(
        probe.get("schema_version") != OPTIMIZER_PROBE_SCHEMA_VERSION
        for probe in probes
    ):
        raise ValueError("unexpected optimizer probe schema")
    identities = {probe["run_identity_sha256"] for probe in probes}
    if identities != {snapshot_metadata["run_identity_sha256"]}:
        raise ValueError("snapshot and optimizer probe identities differ")
    step_index = {step: index for index, step in enumerate(steps)}
    rows: list[dict[str, Any]] = []
    for factor_count in factors:
        for layer in layers:
            for target_index, target_name in enumerate(
                ("attn.c_attn", "attn.c_proj")
            ):
                name = f"transformer.h.{layer}.{target_name}.weight"
                dense_directions = torch.stack(
                    [
                        probe["parameters"][name]["applied_direction_per_lr"]
                        for probe in probes
                    ]
                )
                chords = torch.stack(
                    [
                        values[name][step_index[end]]
                        - values[name][step_index[start]]
                        for start, end in phases
                    ]
                )
                target_batch = torch.cat((dense_directions, chords), dim=0)
                chart = ProductFHTResidualTangent(
                    in_features=target_batch.shape[2],
                    out_features=target_batch.shape[1],
                    factors=factor_count,
                    seed=args.base_seed + layer * 8 + target_index,
                    device=args.device,
                )
                _, diagnostics = chart.project(
                    target_batch,
                    maximum_iterations=args.cg_iterations,
                    tolerance=args.cg_tolerance,
                    ridge=args.ridge,
                )
                coordinate_fraction = (
                    chart.coordinate_count / chart.ambient_count
                )
                for index in range(target_batch.shape[0]):
                    phase_index = index % len(phases)
                    kind = "dense_muon_direction" if index < len(phases) else "phase_chord"
                    rows.append(
                        {
                            "factors": factor_count,
                            "layer": layer,
                            "target": target_name,
                            "kind": kind,
                            "phase_start": phases[phase_index][0],
                            "phase_end": phases[phase_index][1],
                            "probe_step": probe_steps[phase_index],
                            "seed": args.base_seed + layer * 8 + target_index,
                            "coordinate_count": chart.coordinate_count,
                            "ambient_count": chart.ambient_count,
                            "coordinate_fraction": coordinate_fraction,
                            "target_fro": float(
                                diagnostics["target_norm"][index]
                            ),
                            "projected_fro": float(
                                diagnostics["projected_norm"][index]
                            ),
                            "cosine": float(diagnostics["cosine"][index]),
                            "energy_recovery": float(
                                diagnostics["energy_recovery"][index]
                            ),
                            "normalized_enrichment": float(
                                diagnostics["energy_recovery"][index]
                                / coordinate_fraction
                            ),
                            "projection_orthogonality_error": float(
                                diagnostics["projection_orthogonality_error"][index]
                            ),
                            "relative_normal_residual": float(
                                diagnostics["relative_normal_residual"][index]
                            ),
                            "cg_iterations": diagnostics["iterations"],
                        }
                    )
                del chart, target_batch, dense_directions, chords
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
    summaries: dict[str, Any] = {}
    promoted: int | None = None
    for factor_count in factors:
        factor_rows = [row for row in rows if row["factors"] == factor_count]
        aggregate = weighted_summary(factor_rows, "dense_muon_direction")
        by_target = {
            target: weighted_summary(
                [row for row in factor_rows if row["target"] == target],
                "dense_muon_direction",
            )
            for target in ("attn.c_attn", "attn.c_proj")
        }
        chord = weighted_summary(factor_rows, "phase_chord")
        passed = (
            aggregate["energy_recovery"] >= 0.10
            and aggregate["normalized_enrichment"] >= 2.0
            and all(
                summary["energy_recovery"] >= 0.02
                for summary in by_target.values()
            )
            and aggregate["maximum_orthogonality_error"] <= 1e-4
        )
        summaries[str(factor_count)] = {
            "dense_muon_direction": aggregate,
            "dense_muon_direction_by_target": by_target,
            "phase_chord": chord,
            "registered_gate_passed": passed,
        }
        if passed and promoted is None:
            promoted = factor_count
    args.output.mkdir(parents=True, exist_ok=True)
    cells_path = args.output / "attention_product_fht_residual_cells.csv"
    write_csv(cells_path, rows)
    repo_root = Path(__file__).resolve().parents[2]
    result = {
        "schema_version": "mai_124m_attention_product_fht_residual_tangent_v1",
        "source_commit": git_commit(repo_root),
        "source_sha256": file_sha256(Path(__file__)),
        "snapshot_run_identity_sha256": snapshot_metadata[
            "run_identity_sha256"
        ],
        "inputs": {
            "snapshots": [
                {"path": str(path), "sha256": file_sha256(path)}
                for path in snapshot_paths
            ],
            "optimizer_probes": [
                {"path": str(path), "sha256": file_sha256(path)}
                for path in probe_paths
            ],
        },
        "geometry": {
            "family": "zero_preserving_product_fht_residual",
            "factors": factors,
            "base_seed": args.base_seed,
            "layer_seed_stride": 8,
            "target_seed_offsets": {"attn.c_attn": 0, "attn.c_proj": 1},
            "learned_dense_basis": False,
        },
        "solver": {
            "maximum_iterations": args.cg_iterations,
            "tolerance": args.cg_tolerance,
            "ridge": args.ridge,
        },
        "summaries": summaries,
        "decision": {
            "classification": (
                "PROMOTE_PRODUCT_FHT_RESIDUAL"
                if promoted is not None
                else "REJECT_PRODUCT_FHT_RESIDUAL_TANGENT"
            ),
            "promoted_factors": promoted,
            "registered_thresholds": {
                "aggregate_recovery_minimum": 0.10,
                "normalized_enrichment_minimum": 2.0,
                "per_target_recovery_minimum": 0.02,
                "maximum_projection_orthogonality_error": 1e-4,
            },
        },
        "cells_csv": {
            "path": str(cells_path),
            "sha256": file_sha256(cells_path),
        },
        "limitations": [
            "One optimizer trajectory is not the global solution manifold.",
            "Euclidean weight recovery is not the full attention functional metric.",
            "Passing this oracle would admit only one short causal training test.",
        ],
        "elapsed_seconds": time.time() - started,
    }
    result_path = args.output / "attention_product_fht_residual_result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["decision"], sort_keys=True))
    for factor_count, summary in summaries.items():
        print(factor_count, json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
