#!/usr/bin/env python3
"""Gate sparse attention connectivity selected only from available gradients.

Unlike the earlier optimistic oracle, connectivity here cannot inspect the
dense Muon target it will be scored against.  It is selected from the clipped
task-gradient descent direction available to a real BlockFHT-generated layer,
then evaluated on the dense Muon direction and the unseen 15-update chord.
"""

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
from examples.nanogpt.parameter_trajectory import (
    OPTIMIZER_PROBE_SCHEMA_VERSION,
)


def git_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def replace_selection_direction(
    probe: dict[str, Any],
    *,
    field: str = "gradient_after_clip",
    descent: bool = True,
) -> dict[str, Any]:
    """Return a shallow probe view usable by the existing selector.

    The existing exact selector reads ``applied_direction_per_lr``.  Replacing
    only that view lets this audit reuse its proven matching implementation
    without mutating the source probe or allowing the selector to see Muon.
    """
    parameters: dict[str, dict[str, Any]] = {}
    for name, record in probe["parameters"].items():
        if field not in record:
            raise ValueError(f"probe parameter {name} has no field {field}")
        selected = record[field]
        if descent:
            selected = -selected
        parameters[name] = {
            **record,
            "applied_direction_per_lr": selected,
        }
    return {**probe, "parameters": parameters}


def gate_passes(
    *,
    task_gradient: dict[str, float],
    dense_muon: dict[str, float],
    dense_muon_over_random: float,
    future_chord: dict[str, float],
    future_chord_over_random: float,
    future_by_target: dict[str, dict[str, float]],
    thresholds: dict[str, float],
) -> tuple[bool, list[str]]:
    checks = {
        "task_gradient_recovery": (
            task_gradient["energy_recovery"]
            >= thresholds["task_gradient_recovery_minimum"]
        ),
        "task_gradient_enrichment": (
            task_gradient["normalized_enrichment"]
            >= thresholds["task_gradient_enrichment_minimum"]
        ),
        "dense_muon_recovery": (
            dense_muon["energy_recovery"]
            >= thresholds["dense_muon_recovery_minimum"]
        ),
        "dense_muon_over_random": (
            dense_muon_over_random
            >= thresholds["dense_muon_over_random_minimum"]
        ),
        "future_chord_recovery": (
            future_chord["energy_recovery"]
            >= thresholds["future_chord_recovery_minimum"]
        ),
        "future_chord_over_random": (
            future_chord_over_random
            >= thresholds["future_chord_over_random_minimum"]
        ),
        "per_target_future_chord": all(
            summary["energy_recovery"]
            >= thresholds["per_target_chord_recovery_minimum"]
            for summary in future_by_target.values()
        ),
        "projection_error": max(
            task_gradient["maximum_orthogonality_error"],
            dense_muon["maximum_orthogonality_error"],
            future_chord["maximum_orthogonality_error"],
        )
        <= thresholds["maximum_projection_error"],
        "normal_residual": max(
            task_gradient["maximum_relative_normal_residual"],
            dense_muon["maximum_relative_normal_residual"],
            future_chord["maximum_relative_normal_residual"],
        )
        <= thresholds["maximum_normal_residual"],
    }
    failures = [name for name, passed in checks.items() if not passed]
    return not failures, failures


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
        != "mai_124m_attention_gradient_selected_givens_gate_plan_v1"
    ):
        raise ValueError("unexpected plan schema")
    oracle = plan["oracle"]
    layers = [int(value) for value in oracle["layers"]]
    phase_starts = [int(value) for value in oracle["phase_starts"]]
    horizon = int(oracle["horizon"])
    stage_count = int(oracle["stage_count"])
    snapshot_steps = sorted(
        {step for start in phase_starts for step in (start, start + horizon)}
    )
    snapshot_paths = [
        args.snapshot_dir / f"step_{step:06d}.pt" for step in snapshot_steps
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
    connectivity_by_phase: dict[int, dict[str, Any]] = {}
    for phase_start in phase_starts:
        probe = probes[phase_start]
        selection_probe = replace_selection_direction(probe)
        connectivity, selected_rows = select_connectivity(
            probe=selection_probe,
            layers=layers,
            stages=stage_count,
            neighbors=int(oracle["neighbors"]),
            matching_seed=(
                int(oracle["matching_seed"])
                + phase_start * int(oracle["phase_seed_stride"])
            ),
            random_seed=(
                int(oracle["random_seed"])
                + phase_start * int(oracle["phase_seed_stride"])
            ),
        )
        connectivity_by_phase[phase_start] = connectivity
        connectivity_rows.extend(
            {"phase_start": phase_start, **row} for row in selected_rows
        )
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
                    task_descent = -select_target(
                        record["gradient_after_clip"], target, n_embd
                    ).to(args.device, dtype=torch.float32)
                    dense_muon = select_target(
                        record["applied_direction_per_lr"], target, n_embd
                    ).to(args.device, dtype=torch.float32)
                    chord = select_target(
                        values[name][step_index[phase_start + horizon]]
                        - values[name][step_index[phase_start]],
                        target,
                        n_embd,
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
                    coordinate_fraction = chart.coordinate_count / weight.numel()
                    for kind, requested in (
                        ("task_gradient_descent", task_descent),
                        ("dense_muon_direction", dense_muon),
                        ("future_phase_chord", chord),
                    ):
                        _, diagnostics = project(
                            chart,
                            requested,
                            maximum_iterations=int(oracle["cg_iterations"]),
                            tolerance=float(oracle["cg_tolerance"]),
                            ridge=float(oracle["ridge"]),
                        )
                        rows.append(
                            {
                                "connectivity": connectivity_name,
                                "phase_start": phase_start,
                                "phase_end": phase_start + horizon,
                                "horizon": horizon,
                                "layer": layer,
                                "target": target,
                                "kind": kind,
                                "coordinate_fraction": coordinate_fraction,
                                "normalized_enrichment": (
                                    diagnostics["energy_recovery"]
                                    / coordinate_fraction
                                ),
                                **diagnostics,
                            }
                        )
                    del chart, weight, task_descent, dense_muon, chord
                    if args.device.startswith("cuda"):
                        torch.cuda.empty_cache()

    summaries: dict[str, Any] = {}
    for connectivity_name in ("task_selected", "random"):
        selected = [
            row for row in rows if row["connectivity"] == connectivity_name
        ]
        summaries[connectivity_name] = {
            kind: weighted_summary(selected, kind)
            for kind in (
                "task_gradient_descent",
                "dense_muon_direction",
                "future_phase_chord",
            )
        }
        summaries[connectivity_name]["future_phase_chord_by_target"] = {
            target: weighted_summary(
                [row for row in selected if row["target"] == target],
                "future_phase_chord",
            )
            for target in TARGETS
        }

    task = summaries["task_selected"]
    random = summaries["random"]
    dense_muon_over_random = task["dense_muon_direction"][
        "energy_recovery"
    ] / max(random["dense_muon_direction"]["energy_recovery"], 1e-30)
    future_chord_over_random = task["future_phase_chord"][
        "energy_recovery"
    ] / max(random["future_phase_chord"]["energy_recovery"], 1e-30)
    thresholds = {
        key: float(value)
        for key, value in plan["decision_rule"]["thresholds"].items()
    }
    passed, failures = gate_passes(
        task_gradient=task["task_gradient_descent"],
        dense_muon=task["dense_muon_direction"],
        dense_muon_over_random=dense_muon_over_random,
        future_chord=task["future_phase_chord"],
        future_chord_over_random=future_chord_over_random,
        future_by_target=task["future_phase_chord_by_target"],
        thresholds=thresholds,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    cells_path = args.output / "gradient_selected_givens_cells.csv"
    connectivity_path = args.output / "gradient_selected_givens_connectivity.pt"
    write_csv(cells_path, rows)
    torch.save(
        {
            "connectivity_by_phase": connectivity_by_phase,
            "summary": connectivity_rows,
        },
        connectivity_path,
    )
    repo_root = Path(__file__).resolve().parents[2]
    result = {
        "schema_version": "mai_124m_attention_gradient_selected_givens_v1",
        "source_commit": git_commit(repo_root),
        "source_sha256": file_sha256(Path(__file__)),
        "plan": {"path": str(args.plan), "sha256": file_sha256(args.plan)},
        "run_identity_sha256": snapshot_metadata["run_identity_sha256"],
        "selection": {
            "field": "negative gradient_after_clip",
            "uses_dense_muon_target": False,
            "phase_starts": phase_starts,
            "stage_count": stage_count,
            "horizon": horizon,
            "connectivity_artifact": {
                "path": str(connectivity_path),
                "sha256": file_sha256(connectivity_path),
            },
        },
        "summaries": summaries,
        "comparisons": {
            "dense_muon_over_random": dense_muon_over_random,
            "future_chord_over_random": future_chord_over_random,
        },
        "decision": {
            "classification": (
                "AUTHORIZE_DIFFERENTIABLE_BLOCKFHT_ORBIT_IMPLEMENTATION"
                if passed
                else "REJECT_GRADIENT_SELECTED_GIVENS"
            ),
            "passed": passed,
            "failures": failures,
            "thresholds": thresholds,
            "automatic_training_authorized": False,
        },
        "cells_csv": {
            "path": str(cells_path),
            "sha256": file_sha256(cells_path),
        },
        "elapsed_seconds": time.time() - started,
    }
    result_path = args.output / "gradient_selected_givens_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
