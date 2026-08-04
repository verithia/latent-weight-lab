#!/usr/bin/env python3
"""Gate a fixed-connectivity FHT block-skew attention orbit.

The chart conjugates learned 32-wide skew blocks by fixed signed/permuted
block-FHT bases.  It is an orthogonal weight orbit with no learned dense
basis and no additive low-rank path.  This script projects exact dense-Muon
directions and matched dense trajectory chords into its identity tangent.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_parameter_trajectory import (
    load_snapshots,
    parse_int_list,
)
from examples.nanogpt.model import LearnedFHTBlockOrthogonalOutputMix
from examples.nanogpt.parameter_trajectory import OPTIMIZER_PROBE_SCHEMA_VERSION


Coordinates = tuple[torch.Tensor, ...]


TARGETS = {
    "qk": {"selection": "qk", "sides": ("input", "output"), "seed": 0},
    "v": {"selection": "v", "sides": ("input", "output"), "seed": 16},
    "cproj": {"selection": "all", "sides": ("output",), "seed": 32},
}


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


class FixedFHTBlockSkewSide:
    """Exact JVP and adjoint for one input- or output-side orbit."""

    def __init__(
        self,
        *,
        weight: torch.Tensor,
        side: str,
        stages: int,
        rotation_block_size: int,
        basis_block_size: int,
        seed: int,
    ) -> None:
        if side not in {"input", "output"}:
            raise ValueError(f"unsupported side {side}")
        self.weight = weight
        self.side = side
        self.values = weight if side == "input" else weight.T
        self.mixer = LearnedFHTBlockOrthogonalOutputMix(
            features=int(self.values.shape[1]),
            stages=int(stages),
            rotation_block_size=int(rotation_block_size),
            basis_block_size=int(basis_block_size),
            seed=int(seed),
        ).to(device=weight.device, dtype=weight.dtype)
        self.basis_values = tuple(
            self.mixer._basis(self.values, stage, inverse=False)
            for stage in range(self.mixer.stages)
        )

    @property
    def coordinate_count(self) -> int:
        return int(self.mixer.coordinates.numel())

    def zeros(self) -> torch.Tensor:
        return torch.zeros(
            self.mixer.stages,
            self.mixer.rotation_blocks,
            self.mixer.coordinates_per_block,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )

    def _skew(self, coordinates: torch.Tensor) -> torch.Tensor:
        skew = coordinates.new_zeros(
            self.mixer.rotation_blocks,
            self.mixer.rotation_block_size,
            self.mixer.rotation_block_size,
        )
        rows = self.mixer.upper_rows.to(coordinates.device)
        columns = self.mixer.upper_columns.to(coordinates.device)
        skew[:, rows, columns] = coordinates
        skew[:, columns, rows] = -coordinates
        return skew

    def jvp(self, coordinates: torch.Tensor) -> torch.Tensor:
        if coordinates.shape != self.zeros().shape:
            raise ValueError("coordinate shape mismatch")
        delta_values = torch.zeros_like(self.values)
        for stage in range(self.mixer.stages):
            blocks = self.basis_values[stage].reshape(
                self.values.shape[0],
                self.mixer.rotation_blocks,
                self.mixer.rotation_block_size,
            )
            skew = self._skew(coordinates[stage])
            delta_blocks = 2.0 * torch.einsum(
                "mgi,gij->mgj", blocks, skew
            )
            delta_values = delta_values + self.mixer._basis(
                delta_blocks.reshape_as(self.values), stage, inverse=True
            )
        if self.side == "input":
            return -delta_values
        return delta_values.T

    def adjoint(self, direction: torch.Tensor) -> torch.Tensor:
        direction_values = -direction if self.side == "input" else direction.T
        gradients: list[torch.Tensor] = []
        rows = self.mixer.upper_rows.to(direction.device)
        columns = self.mixer.upper_columns.to(direction.device)
        for stage in range(self.mixer.stages):
            source = self.basis_values[stage].reshape(
                self.values.shape[0],
                self.mixer.rotation_blocks,
                self.mixer.rotation_block_size,
            )
            cotangent = self.mixer._basis(
                direction_values, stage, inverse=False
            ).reshape_as(source)
            cross = torch.einsum("mgi,mgj->gij", source, cotangent)
            skew_gradient = 2.0 * (cross - cross.transpose(1, 2))
            gradients.append(skew_gradient[:, rows, columns])
        return torch.stack(gradients)


class TargetedBilateralTangent:
    def __init__(
        self,
        *,
        weight: torch.Tensor,
        sides: tuple[str, ...],
        stages: int,
        rotation_block_size: int,
        basis_block_size: int,
        seed: int,
    ) -> None:
        self.weight = weight
        self.charts = tuple(
            FixedFHTBlockSkewSide(
                weight=weight,
                side=side,
                stages=stages,
                rotation_block_size=rotation_block_size,
                basis_block_size=basis_block_size,
                seed=seed + index * 1_000_003,
            )
            for index, side in enumerate(sides)
        )

    @property
    def coordinate_count(self) -> int:
        return sum(chart.coordinate_count for chart in self.charts)

    def zeros(self) -> Coordinates:
        return tuple(chart.zeros() for chart in self.charts)

    def jvp(self, coordinates: Coordinates) -> torch.Tensor:
        return sum(
            (
                chart.jvp(value)
                for chart, value in zip(self.charts, coordinates, strict=True)
            ),
            torch.zeros_like(self.weight),
        )

    def adjoint(self, direction: torch.Tensor) -> Coordinates:
        return tuple(chart.adjoint(direction) for chart in self.charts)


def coordinate_dot(first: Coordinates, second: Coordinates) -> torch.Tensor:
    return sum(
        ((left * right).sum() for left, right in zip(first, second, strict=True)),
        torch.zeros((), device=first[0].device, dtype=first[0].dtype),
    )


def coordinate_add(
    first: Coordinates, second: Coordinates, alpha: torch.Tensor | float
) -> Coordinates:
    return tuple(
        left + alpha * right
        for left, right in zip(first, second, strict=True)
    )


def project(
    chart: TargetedBilateralTangent,
    target: torch.Tensor,
    *,
    maximum_iterations: int,
    tolerance: float,
    ridge: float,
) -> tuple[torch.Tensor, dict[str, float | int | bool]]:
    solution = chart.zeros()
    residual = chart.adjoint(target)
    search = tuple(value.clone() for value in residual)
    residual_squared = coordinate_dot(residual, residual)
    initial_squared = residual_squared.clone().clamp_min(1e-30)
    converged = float(residual_squared) == 0.0
    iterations = 0
    for iteration in range(maximum_iterations):
        if converged:
            break
        mapped = chart.jvp(search)
        normal = chart.adjoint(mapped)
        if ridge:
            normal = coordinate_add(normal, search, ridge)
        denominator = coordinate_dot(search, normal)
        if float(denominator) <= 0.0:
            break
        step = residual_squared / denominator
        solution = coordinate_add(solution, search, step)
        updated = coordinate_add(residual, normal, -step)
        updated_squared = coordinate_dot(updated, updated)
        iterations = iteration + 1
        relative = torch.sqrt(updated_squared / initial_squared)
        if float(relative) <= tolerance:
            residual = updated
            residual_squared = updated_squared
            converged = True
            break
        beta = updated_squared / residual_squared.clamp_min(1e-30)
        search = coordinate_add(updated, search, beta)
        residual = updated
        residual_squared = updated_squared
    projected = chart.jvp(solution)
    remainder = target - projected
    target_norm = target.norm().clamp_min(1e-30)
    projected_norm = projected.norm().clamp_min(1e-30)
    return projected, {
        "iterations": iterations,
        "converged": converged,
        "relative_normal_residual": float(
            torch.sqrt(residual_squared / initial_squared)
        ),
        "projection_orthogonality_error": float(
            (remainder * projected).sum().abs()
            / (target_norm * projected_norm)
        ),
        "target_fro": float(target_norm),
        "projected_fro": float(projected_norm),
        "energy_recovery": float(projected_norm.square() / target_norm.square()),
    }


def weighted_summary(rows: list[dict[str, Any]], kind: str) -> dict[str, float]:
    selected = [row for row in rows if row["kind"] == kind]
    target_energy = sum(float(row["target_fro"]) ** 2 for row in selected)
    projected_energy = sum(float(row["projected_fro"]) ** 2 for row in selected)
    coordinate_fraction = sum(
        float(row["coordinate_fraction"]) * float(row["target_fro"]) ** 2
        for row in selected
    ) / target_energy
    recovery = projected_energy / target_energy
    return {
        "cells": float(len(selected)),
        "energy_recovery": recovery,
        "coordinate_fraction": coordinate_fraction,
        "normalized_enrichment": recovery / coordinate_fraction,
        "minimum_cell_recovery": min(float(row["energy_recovery"]) for row in selected),
        "maximum_orthogonality_error": max(
            float(row["projection_orthogonality_error"]) for row in selected
        ),
        "maximum_relative_normal_residual": max(
            float(row["relative_normal_residual"]) for row in selected
        ),
    }


def select_target(
    tensor: torch.Tensor, target: str, n_embd: int
) -> torch.Tensor:
    selection = TARGETS[target]["selection"]
    if selection == "qk":
        return tensor[: 2 * n_embd]
    if selection == "v":
        return tensor[2 * n_embd :]
    return tensor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--probe-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    plan = json.loads(args.plan.read_text())
    if plan.get("schema_version") != "mai_124m_attention_fht_block_skew_gate_plan_v1":
        raise ValueError("unexpected plan schema")
    oracle = plan["oracle"]
    layers = [int(value) for value in oracle["layers"]]
    boundaries = [int(value) for value in oracle["phase_boundaries"]]
    probe_steps = [int(value) for value in oracle["probe_steps"]]
    stage_counts = [int(value) for value in oracle["stage_counts"]]
    phases = list(zip(boundaries[:-1], boundaries[1:], strict=True))
    snapshot_paths = [
        args.snapshot_dir / f"step_{step:06d}.pt" for step in boundaries
    ]
    probe_paths = [args.probe_dir / f"step_{step:06d}.pt" for step in probe_steps]
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
    if steps != boundaries or len(phases) != len(probes):
        raise ValueError("phase inventory mismatch")
    step_index = {step: index for index, step in enumerate(steps)}
    rows: list[dict[str, Any]] = []
    for stages in stage_counts:
        for phase_index, ((phase_start, phase_end), probe) in enumerate(
            zip(phases, probes, strict=True)
        ):
            n_embd = int(probe["model_config"]["n_embd"])
            for layer in layers:
                for target_index, target in enumerate(TARGETS):
                    suffix = "attn.c_proj" if target == "cproj" else "attn.c_attn"
                    name = f"transformer.h.{layer}.{suffix}.weight"
                    record = probe["parameters"][name]
                    weight = select_target(
                        record["weight_before_step"], target, n_embd
                    ).to(args.device, dtype=torch.float32)
                    dense_direction = select_target(
                        record["applied_direction_per_lr"], target, n_embd
                    ).to(args.device, dtype=torch.float32)
                    chord = select_target(
                        values[name][step_index[phase_end]]
                        - values[name][step_index[phase_start]],
                        target,
                        n_embd,
                    ).to(args.device, dtype=torch.float32)
                    chart = TargetedBilateralTangent(
                        weight=weight,
                        sides=TARGETS[target]["sides"],
                        stages=stages,
                        rotation_block_size=int(oracle["rotation_block_size"]),
                        basis_block_size=int(oracle["basis_block_size"]),
                        seed=int(oracle["base_seed"])
                        + layer * 64
                        + int(TARGETS[target]["seed"]),
                    )
                    for kind, requested in (
                        ("dense_muon_direction", dense_direction),
                        ("phase_chord", chord),
                    ):
                        _, diagnostics = project(
                            chart,
                            requested,
                            maximum_iterations=int(oracle["cg_iterations"]),
                            tolerance=float(oracle["cg_tolerance"]),
                            ridge=float(oracle["ridge"]),
                        )
                        coordinate_fraction = chart.coordinate_count / weight.numel()
                        row = {
                            "stages": stages,
                            "phase_start": phase_start,
                            "phase_end": phase_end,
                            "probe_step": probe_steps[phase_index],
                            "layer": layer,
                            "target": target,
                            "kind": kind,
                            "sides": "+".join(TARGETS[target]["sides"]),
                            "coordinate_count": chart.coordinate_count,
                            "ambient_count": weight.numel(),
                            "coordinate_fraction": coordinate_fraction,
                            "normalized_enrichment": diagnostics["energy_recovery"]
                            / coordinate_fraction,
                            **diagnostics,
                        }
                        rows.append(row)
                        print(json.dumps(row, sort_keys=True), flush=True)
                    del chart, weight, dense_direction, chord
                    if args.device.startswith("cuda"):
                        torch.cuda.empty_cache()
    thresholds = plan["decision_rule"]["thresholds"]
    summaries: dict[str, Any] = {}
    promoted: list[int] = []
    for stages in stage_counts:
        selected = [row for row in rows if int(row["stages"]) == stages]
        dense = weighted_summary(selected, "dense_muon_direction")
        chord = weighted_summary(selected, "phase_chord")
        by_target = {
            target: weighted_summary(
                [row for row in selected if row["target"] == target],
                "dense_muon_direction",
            )
            for target in TARGETS
        }
        passed = (
            dense["energy_recovery"] >= float(thresholds["aggregate_recovery_minimum"])
            and dense["normalized_enrichment"]
            >= float(thresholds["normalized_enrichment_minimum"])
            and all(
                summary["energy_recovery"]
                >= float(thresholds["per_target_recovery_minimum"])
                for summary in by_target.values()
            )
            and chord["normalized_enrichment"]
            >= float(thresholds["chord_enrichment_minimum"])
            and max(
                dense["maximum_orthogonality_error"],
                chord["maximum_orthogonality_error"],
            )
            <= float(thresholds["maximum_projection_error"])
            and max(
                dense["maximum_relative_normal_residual"],
                chord["maximum_relative_normal_residual"],
            )
            <= float(thresholds["maximum_normal_residual"])
        )
        summaries[str(stages)] = {
            "dense_muon_direction": dense,
            "dense_muon_direction_by_target": by_target,
            "phase_chord": chord,
            "registered_gate_passed": passed,
        }
        if passed:
            promoted.append(stages)
    selected_stages = (
        max(
            promoted,
            key=lambda value: (
                summaries[str(value)]["dense_muon_direction"]["energy_recovery"]
                - summaries[str(value)]["dense_muon_direction"]["coordinate_fraction"],
                -value,
            ),
        )
        if promoted
        else None
    )
    args.output.mkdir(parents=True, exist_ok=True)
    cells_path = args.output / "attention_fht_block_skew_cells.csv"
    write_csv(cells_path, rows)
    repo_root = Path(__file__).resolve().parents[2]
    result = {
        "schema_version": "mai_124m_attention_fht_block_skew_tangent_v1",
        "source_commit": git_commit(repo_root),
        "source_sha256": file_sha256(Path(__file__)),
        "plan": {"path": str(args.plan), "sha256": file_sha256(args.plan)},
        "run_identity_sha256": snapshot_metadata["run_identity_sha256"],
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
        "summaries": summaries,
        "decision": {
            "classification": (
                "PROMOTE_FHT_BLOCK_SKEW_ORBIT"
                if selected_stages is not None
                else "REJECT_FHT_BLOCK_SKEW_ORBIT"
            ),
            "selected_stages": selected_stages,
            "selection_rule": (
                "largest recovery-minus-coordinate-fraction excess; "
                "ties prefer fewer stages"
            ),
            "thresholds": thresholds,
        },
        "cells_csv": {"path": str(cells_path), "sha256": file_sha256(cells_path)},
        "elapsed_seconds": time.time() - started,
    }
    result_path = args.output / "attention_fht_block_skew_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
