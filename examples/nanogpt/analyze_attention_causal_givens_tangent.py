#!/usr/bin/env python3
"""Gate causally refreshed sparse Givens attention connectivity."""

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
        != "mai_124m_attention_causal_givens_gate_plan_v1"
    ):
        raise ValueError("unexpected plan schema")
    oracle = plan["oracle"]
    layers = [int(value) for value in oracle["layers"]]
    boundaries = [int(value) for value in oracle["phase_boundaries"]]
    phase_starts = boundaries[:-1]
    stage_counts = [int(value) for value in oracle["stage_counts"]]
    maximum_stages = max(stage_counts)
    snapshot_paths = [
        args.snapshot_dir / f"step_{step:06d}.pt" for step in boundaries
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
    connectivity_by_phase: dict[
        int, dict[str, dict[tuple[int, str, str], torch.Tensor]]
    ] = {}
    connectivity_rows: list[dict[str, Any]] = []
    for phase_start in phase_starts:
        connectivity, rows = select_connectivity(
            probe=probes[phase_start],
            layers=layers,
            stages=maximum_stages,
            neighbors=int(oracle["neighbors"]),
            matching_seed=int(oracle["matching_seed"]) + phase_start * 8192,
            random_seed=int(oracle["random_seed"]) + phase_start * 8192,
        )
        connectivity_by_phase[phase_start] = connectivity
        connectivity_rows.extend(
            {"phase_start": phase_start, **row} for row in rows
        )
    step_index = {step: index for index, step in enumerate(steps)}
    end_by_start = dict(zip(boundaries[:-1], boundaries[1:], strict=True))
    rows: list[dict[str, Any]] = []
    for stages_count in stage_counts:
        for phase_start in phase_starts:
            phase_end = end_by_start[phase_start]
            probe = probes[phase_start]
            n_embd = int(probe["model_config"]["n_embd"])
            for connectivity_name in ("task_selected", "random"):
                selected_connectivity = connectivity_by_phase[phase_start][
                    connectivity_name
                ]
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
                        chord = select_target(
                            values[name][step_index[phase_end]]
                            - values[name][step_index[phase_start]],
                            target,
                            n_embd,
                        ).to(args.device, dtype=torch.float32)
                        permutations = {
                            side: selected_connectivity[(layer, target, side)]
                            for side in metadata["sides"]
                        }
                        chart = PersistentGivensTangent(
                            weight=weight,
                            sides=metadata["sides"],
                            permutations=permutations,
                            stages=stages_count,
                        )
                        for kind, requested in (
                            ("dense_muon_direction", dense),
                            ("future_phase_chord", chord),
                        ):
                            _, diagnostics = project(
                                chart,
                                requested,
                                maximum_iterations=int(oracle["cg_iterations"]),
                                tolerance=float(oracle["cg_tolerance"]),
                                ridge=float(oracle["ridge"]),
                            )
                            coordinate_fraction = (
                                chart.coordinate_count / weight.numel()
                            )
                            row = {
                                "stages": stages_count,
                                "connectivity": connectivity_name,
                                "phase_start": phase_start,
                                "phase_end": phase_end,
                                "layer": layer,
                                "target": target,
                                "kind": kind,
                                "coordinate_count": chart.coordinate_count,
                                "ambient_count": weight.numel(),
                                "coordinate_fraction": coordinate_fraction,
                                "normalized_enrichment": (
                                    diagnostics["energy_recovery"]
                                    / coordinate_fraction
                                ),
                                **diagnostics,
                            }
                            rows.append(row)
                            print(json.dumps(row, sort_keys=True), flush=True)
                        del chart, weight, dense, chord
                        if args.device.startswith("cuda"):
                            torch.cuda.empty_cache()
    thresholds = plan["decision_rule"]["thresholds"]
    summaries: dict[str, Any] = {}
    promoted: list[int] = []
    for stages_count in stage_counts:
        stage_summary: dict[str, Any] = {}
        for connectivity_name in ("task_selected", "random"):
            selected = [
                row
                for row in rows
                if int(row["stages"]) == stages_count
                and row["connectivity"] == connectivity_name
            ]
            stage_summary[connectivity_name] = {
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
        task = stage_summary["task_selected"]
        random = stage_summary["random"]
        dense = task["dense_muon_direction"]
        chord = task["future_phase_chord"]
        dense_over_random = dense["energy_recovery"] / max(
            random["dense_muon_direction"]["energy_recovery"], 1e-30
        )
        chord_over_random = chord["energy_recovery"] / max(
            random["future_phase_chord"]["energy_recovery"], 1e-30
        )
        passed = (
            dense["energy_recovery"]
            >= float(thresholds["dense_recovery_minimum"])
            and dense["normalized_enrichment"]
            >= float(thresholds["dense_enrichment_minimum"])
            and dense_over_random
            >= float(thresholds["dense_over_random_minimum"])
            and chord["energy_recovery"]
            >= float(thresholds["future_chord_recovery_minimum"])
            and chord_over_random
            >= float(thresholds["future_chord_over_random_minimum"])
            and all(
                summary["energy_recovery"]
                >= float(thresholds["per_target_chord_recovery_minimum"])
                for summary in task["future_phase_chord_by_target"].values()
            )
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
        summaries[str(stages_count)] = {
            **stage_summary,
            "task_dense_over_random": dense_over_random,
            "task_future_chord_over_random": chord_over_random,
            "registered_gate_passed": passed,
        }
        if passed:
            promoted.append(stages_count)
    selected_stages = (
        max(
            promoted,
            key=lambda value: (
                summaries[str(value)]["task_selected"]
                ["future_phase_chord"]["energy_recovery"]
                - summaries[str(value)]["task_selected"]
                ["future_phase_chord"]["coordinate_fraction"],
                -value,
            ),
        )
        if promoted
        else None
    )
    args.output.mkdir(parents=True, exist_ok=True)
    cells_path = args.output / "attention_causal_givens_cells.csv"
    connectivity_path = args.output / "attention_causal_givens_connectivity.pt"
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
        "schema_version": "mai_124m_attention_causal_givens_tangent_v1",
        "source_commit": git_commit(repo_root),
        "source_sha256": file_sha256(Path(__file__)),
        "plan": {"path": str(args.plan), "sha256": file_sha256(args.plan)},
        "run_identity_sha256": snapshot_metadata["run_identity_sha256"],
        "selection": {
            "causal_phase_starts": phase_starts,
            "rows": connectivity_rows,
            "artifact": {
                "path": str(connectivity_path),
                "sha256": file_sha256(connectivity_path),
            },
        },
        "summaries": summaries,
        "decision": {
            "classification": (
                "PROMOTE_CAUSAL_GIVENS"
                if selected_stages is not None
                else "REJECT_CAUSAL_GIVENS"
            ),
            "selected_stages": selected_stages,
            "thresholds": thresholds,
        },
        "cells_csv": {
            "path": str(cells_path),
            "sha256": file_sha256(cells_path),
        },
        "elapsed_seconds": time.time() - started,
    }
    result_path = args.output / "attention_causal_givens_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
