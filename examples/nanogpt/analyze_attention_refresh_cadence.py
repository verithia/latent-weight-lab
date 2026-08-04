#!/usr/bin/env python3
"""Measure causal sparse-attention connectivity staleness by horizon."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_attention_fht_block_skew_tangent import (
    TARGETS,
    file_sha256,
    project,
    select_target,
    weighted_summary,
    write_csv,
)
from examples.nanogpt.analyze_attention_persistent_givens_tangent import (
    PersistentGivensTangent,
    parameter_name,
    select_connectivity,
)
from examples.nanogpt.analyze_parameter_trajectory import load_snapshots
from examples.nanogpt.parameter_trajectory import OPTIMIZER_PROBE_SCHEMA_VERSION


def git_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def horizon_passes(
    *,
    current: dict[str, float],
    chord: dict[str, float],
    chord_over_random: float,
    by_target: dict[str, dict[str, float]],
    thresholds: dict[str, float],
) -> bool:
    return (
        current["energy_recovery"]
        >= thresholds["current_dense_recovery_minimum"]
        and current["normalized_enrichment"]
        >= thresholds["current_dense_enrichment_minimum"]
        and chord["energy_recovery"]
        >= thresholds["future_chord_recovery_minimum"]
        and chord_over_random
        >= thresholds["future_chord_over_random_minimum"]
        and all(
            summary["energy_recovery"]
            >= thresholds["per_target_chord_recovery_minimum"]
            for summary in by_target.values()
        )
        and max(
            current["maximum_orthogonality_error"],
            chord["maximum_orthogonality_error"],
        )
        <= thresholds["maximum_projection_error"]
        and max(
            current["maximum_relative_normal_residual"],
            chord["maximum_relative_normal_residual"],
        )
        <= thresholds["maximum_normal_residual"]
    )


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
    if (
        plan.get("schema_version")
        != "mai_124m_attention_refresh_cadence_gate_plan_v1"
    ):
        raise ValueError("unexpected plan schema")
    replay = plan["replay"]
    oracle = plan["oracle"]
    layers = [int(value) for value in replay["layers"]]
    phase_starts = [int(value) for value in oracle["phase_starts"]]
    horizons = [int(value) for value in oracle["horizons"]]
    stage_count = int(oracle["stage_count"])
    matching_seed = int(oracle["matching_seed"])
    random_seed = int(oracle["random_seed"])
    phase_seed_stride = int(oracle["phase_seed_stride"])
    cg_iterations = int(oracle["cg_iterations"])
    cg_tolerance = float(oracle["cg_tolerance"])
    ridge = float(oracle["ridge"])
    required_snapshot_steps = sorted(
        {
            step
            for start in phase_starts
            for step in [start]
            + [start + horizon for horizon in horizons if start + horizon <= 238]
        }
    )
    snapshot_paths = [
        args.snapshot_dir / f"step_{step:06d}.pt"
        for step in required_snapshot_steps
    ]
    probe_paths = [
        args.probe_dir / f"step_{step:06d}.pt" for step in phase_starts
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
    probes = {
        step: torch.load(path, map_location="cpu", weights_only=False)
        for step, path in zip(phase_starts, probe_paths, strict=True)
    }
    if any(
        probe.get("schema_version") != OPTIMIZER_PROBE_SCHEMA_VERSION
        for probe in probes.values()
    ):
        raise ValueError("unexpected optimizer probe schema")
    identities = {probe["run_identity_sha256"] for probe in probes.values()}
    if identities != {snapshot_metadata["run_identity_sha256"]}:
        raise ValueError("snapshot and optimizer probe identities differ")
    step_index = {step: index for index, step in enumerate(steps)}
    rows: list[dict[str, Any]] = []
    connectivity_rows: list[dict[str, Any]] = []
    for phase_start in phase_starts:
        connectivity, selected_rows = select_connectivity(
            probe=probes[phase_start],
            layers=layers,
            stages=stage_count,
            neighbors=int(oracle["neighbors"]),
            matching_seed=matching_seed + phase_start * phase_seed_stride,
            random_seed=random_seed + phase_start * phase_seed_stride,
        )
        connectivity_rows.extend(
            {"phase_start": phase_start, **row} for row in selected_rows
        )
        probe = probes[phase_start]
        n_embd = int(probe["model_config"]["n_embd"])
        for connectivity_name in ("task_selected", "random"):
            selected = connectivity[connectivity_name]
            for layer in layers:
                for target, metadata in TARGETS.items():
                    name = parameter_name(layer, target)
                    record = probe["parameters"][name]
                    weight = select_target(
                        record["weight_before_step"], target, n_embd
                    ).to(args.device, dtype=torch.float32)
                    dense = select_target(
                        record["applied_direction_per_lr"], target, n_embd
                    ).to(args.device, dtype=torch.float32)
                    permutations = {
                        side: selected[(layer, target, side)]
                        for side in metadata["sides"]
                    }
                    chart = PersistentGivensTangent(
                        weight=weight,
                        sides=metadata["sides"],
                        permutations=permutations,
                        stages=stage_count,
                    )
                    _, current_diagnostics = project(
                        chart,
                        dense,
                        maximum_iterations=cg_iterations,
                        tolerance=cg_tolerance,
                        ridge=ridge,
                    )
                    coordinate_fraction = chart.coordinate_count / weight.numel()
                    rows.append(
                        {
                            "connectivity": connectivity_name,
                            "phase_start": phase_start,
                            "horizon": 0,
                            "layer": layer,
                            "target": target,
                            "kind": "dense_muon_direction",
                            "coordinate_fraction": coordinate_fraction,
                            "normalized_enrichment": (
                                current_diagnostics["energy_recovery"]
                                / coordinate_fraction
                            ),
                            **current_diagnostics,
                        }
                    )
                    for horizon in horizons:
                        endpoint = phase_start + horizon
                        if endpoint not in step_index:
                            continue
                        chord = select_target(
                            values[name][step_index[endpoint]]
                            - values[name][step_index[phase_start]],
                            target,
                            n_embd,
                        ).to(args.device, dtype=torch.float32)
                        _, diagnostics = project(
                            chart,
                            chord,
                            maximum_iterations=cg_iterations,
                            tolerance=cg_tolerance,
                            ridge=ridge,
                        )
                        row = {
                            "connectivity": connectivity_name,
                            "phase_start": phase_start,
                            "phase_end": endpoint,
                            "horizon": horizon,
                            "layer": layer,
                            "target": target,
                            "kind": "future_phase_chord",
                            "coordinate_fraction": coordinate_fraction,
                            "normalized_enrichment": (
                                diagnostics["energy_recovery"]
                                / coordinate_fraction
                            ),
                            **diagnostics,
                        }
                        rows.append(row)
                        print(json.dumps(row, sort_keys=True), flush=True)
                    del chart, weight, dense
                    if args.device.startswith("cuda"):
                        torch.cuda.empty_cache()
    thresholds = {
        key: float(value)
        for key, value in plan["decision_rule"]["thresholds"].items()
    }
    summaries: dict[str, Any] = {}
    passing_horizons: list[int] = []
    for horizon in horizons:
        horizon_summary: dict[str, Any] = {}
        for connectivity_name in ("task_selected", "random"):
            selected = [
                row
                for row in rows
                if row["connectivity"] == connectivity_name
                and (
                    row["kind"] == "dense_muon_direction"
                    or int(row["horizon"]) == horizon
                )
            ]
            horizon_summary[connectivity_name] = {
                "dense_muon_direction": weighted_summary(
                    selected, "dense_muon_direction"
                ),
                "future_phase_chord": weighted_summary(
                    selected, "future_phase_chord"
                ),
                "future_phase_chord_by_target": {
                    target: weighted_summary(
                        [row for row in selected if row["target"] == target],
                        "future_phase_chord",
                    )
                    for target in TARGETS
                },
            }
        task = horizon_summary["task_selected"]
        random = horizon_summary["random"]
        chord_over_random = task["future_phase_chord"][
            "energy_recovery"
        ] / max(random["future_phase_chord"]["energy_recovery"], 1e-30)
        passed = horizon_passes(
            current=task["dense_muon_direction"],
            chord=task["future_phase_chord"],
            chord_over_random=chord_over_random,
            by_target=task["future_phase_chord_by_target"],
            thresholds=thresholds,
        )
        summaries[str(horizon)] = {
            **horizon_summary,
            "task_future_chord_over_random": chord_over_random,
            "registered_gate_passed": passed,
        }
        if passed:
            passing_horizons.append(horizon)
    selected_horizon = max(passing_horizons) if passing_horizons else None
    args.output.mkdir(parents=True, exist_ok=True)
    cells_path = args.output / "attention_refresh_cadence_cells.csv"
    connectivity_path = args.output / "attention_refresh_connectivity.csv"
    write_csv(cells_path, rows)
    write_csv(connectivity_path, connectivity_rows)
    repo_root = Path(__file__).resolve().parents[2]
    result = {
        "schema_version": "mai_124m_attention_refresh_cadence_v1",
        "source_commit": git_commit(repo_root),
        "source_sha256": file_sha256(Path(__file__)),
        "plan": {"path": str(args.plan), "sha256": file_sha256(args.plan)},
        "run_identity_sha256": snapshot_metadata["run_identity_sha256"],
        "summaries": summaries,
        "decision": {
            "classification": (
                "PROMOTE_REFRESH_CADENCE"
                if selected_horizon is not None
                else "REJECT_CAUSAL_GIVENS_REFRESH"
            ),
            "selected_horizon_updates": selected_horizon,
            "thresholds": thresholds,
        },
        "artifacts": {
            "cells_sha256": file_sha256(cells_path),
            "connectivity_sha256": file_sha256(connectivity_path),
        },
        "elapsed_seconds": time.time() - started,
    }
    result_path = args.output / "attention_refresh_cadence_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
