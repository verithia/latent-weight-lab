#!/usr/bin/env python3
"""Causal moving-tangent upper bound in the coupled attention V/O quotient.

At each target time, the atlas contains only exact quotient chords and Muon
tangents strictly earlier than that target.  A single global chord-memory
window is selected on an intermediate phase and then frozen for the terminal
phase.  This is a dense-teacher upper bound for trackability, not a deployable
or trainable decoder.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_attention_affine_delta_path_oracle import (
    batch_digest,
    minimum_layer_recovery,
    solve_span_coefficients,
    trajectory_inventory,
    weighted,
    write_rows,
)
from examples.nanogpt.analyze_attention_paper_activation_oracle import (
    all_finite,
    file_sha256,
    terminal_attention_metrics,
)
from examples.nanogpt.analyze_attention_stepzero_functional_atlas import (
    git_commit,
    load_target_snapshot,
)
from examples.nanogpt.analyze_attention_vo_quotient_manifold import (
    fit_and_transport,
    quotient_output,
    quotient_tangent,
    split_v_o,
)
from examples.nanogpt.analyze_residual_compatibility import fixed_validation_batches


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_SCHEMA = "mai_124m_attention_vo_moving_tangent_plan_v1"
RESULT_SCHEMA = "mai_124m_attention_vo_moving_tangent_result_v1"


def validate_plan(plan: dict[str, Any], args: argparse.Namespace) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unexpected V/O moving-tangent plan schema")
    protocol = plan["protocol"]
    frozen = {
        "parameter_updates": 0,
        "window_candidates": [2, 4, 8, 16],
        "selection_chord_start_minimum": 1200,
        "selection_chord_end_maximum": 1740,
        "test_chord_start_minimum": 1800,
        "selection_muon_step": 1782,
        "test_muon_step": 2372,
        "available_probe_steps": [0, 594, 1188, 1782, 2372],
        "fit_metric_seed": 20260809,
        "eval_metric_seed": 20260810,
        "metric_batch_size": 2,
        "metric_block_size": 256,
        "metric_batches": 2,
        "span_relative_cutoff": 1e-8,
        "coordinate_fraction": 0.01,
    }
    for field, expected in frozen.items():
        if protocol.get(field) != expected:
            raise ValueError(f"frozen moving-tangent protocol changed: {field}")
    if plan["decision_rule"]["thresholds"] != {
        "aggregate_recovery_minimum": 0.8,
        "minimum_every_layer_recovery": 0.6,
        "minimum_late_layer_8_to_11_recovery": 0.6,
    }:
        raise ValueError("moving-tangent thresholds changed")
    if any(bool(value) for value in plan["authorization"].values()):
        raise ValueError("moving-tangent oracle must not pre-authorize a successor")
    identity = plan["identity"]
    paths = {
        Path(__file__): identity["entrypoint_sha256"],
        REPO_ROOT / identity["design"]: identity["design_sha256"],
        REPO_ROOT / identity["dense_config"]: identity["dense_config_sha256"],
        args.terminal_checkpoint: identity["terminal_checkpoint_sha256"],
        args.data_dir / "manifest.json": identity["dataset_manifest_sha256"],
    }
    for path, expected in paths.items():
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"pinned moving-tangent identity mismatch: {path}")
    inventory, digest = trajectory_inventory(args.trajectory_dir)
    if (
        len(inventory) != int(identity["trajectory_file_count"])
        or sum(int(item["size"]) for item in inventory)
        != int(identity["trajectory_total_bytes"])
        or digest != identity["trajectory_inventory_sha256"]
    ):
        raise ValueError("moving-tangent trajectory inventory mismatch")
    for name, expected in identity["optimizer_probe_sha256"].items():
        path = args.probe_dir / name
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"moving-tangent optimizer probe mismatch: {path}")
    if Path(identity["trajectory_directory"]) != args.trajectory_dir:
        raise ValueError("trajectory directory differs from plan")
    if Path(identity["optimizer_probe_directory"]) != args.probe_dir:
        raise ValueError("probe directory differs from plan")
    if Path(identity["output_directory_must_be_absent"]) != args.output_dir:
        raise ValueError("output directory differs from plan")


def phase_summary(rows: list[dict[str, Any]], window: int, phase: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for kind in ("chord", "muon_direction"):
        selected = [
            row
            for row in rows
            if int(row["window"]) == int(window)
            and row["phase"] == phase
            and row["kind"] == kind
        ]
        late = [row for row in selected if int(row["layer"]) >= 8]
        output[kind] = {
            "aggregate_eval_recovery": weighted(
                selected, "eval_recovery", "eval_energy"
            ),
            "minimum_layer_eval_recovery": minimum_layer_recovery(
                selected, "eval_recovery", "eval_energy"
            ),
            "minimum_late_layer_eval_recovery": minimum_layer_recovery(
                late, "eval_recovery", "eval_energy"
            ),
            "maximum_atlas_atoms": max(int(row["atlas_atoms"]) for row in selected),
            "maximum_atlas_rank": max(int(row["atlas_rank"]) for row in selected),
        }
    return output


def selection_score(summary: dict[str, Any]) -> float:
    return min(
        float(summary[kind][field])
        for kind in ("chord", "muon_direction")
        for field in (
            "aggregate_eval_recovery",
            "minimum_layer_eval_recovery",
            "minimum_late_layer_eval_recovery",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--trajectory-dir", required=True, type=Path)
    parser.add_argument("--probe-dir", required=True, type=Path)
    parser.add_argument("--terminal-checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    validate_plan(plan, args)
    if args.output_dir.exists():
        raise FileExistsError(f"output already exists: {args.output_dir}")
    started = time.time()
    protocol = plan["protocol"]
    layers = [int(value) for value in protocol["layers"]]
    steps = [int(value) for value in protocol["trajectory_steps"]]
    probe_steps = [int(value) for value in protocol["available_probe_steps"]]
    windows = [int(value) for value in protocol["window_candidates"]]
    _inventory, inventory_sha = trajectory_inventory(args.trajectory_dir)
    config = json.loads((REPO_ROOT / plan["identity"]["dense_config"]).read_text())
    n_embd = int(config["n_embd"])

    snapshots: dict[int, dict[str, torch.Tensor]] = {}
    run_identity = None
    for step in steps:
        payload = load_target_snapshot(args.trajectory_dir / f"step_{step:06d}.pt")
        if run_identity is None:
            run_identity = payload["run_identity_sha256"]
        elif payload["run_identity_sha256"] != run_identity:
            raise ValueError("trajectory snapshots do not share one run identity")
        snapshots[step] = payload["parameters"]
    if run_identity != plan["identity"]["trajectory_run_identity_sha256"]:
        raise ValueError("trajectory run identity mismatch")
    probes: dict[int, dict[str, Any]] = {}
    for step in probe_steps:
        payload = torch.load(
            args.probe_dir / f"step_{step:06d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        if payload["run_identity_sha256"] != run_identity:
            raise ValueError("optimizer probe run identity mismatch")
        probes[step] = payload

    fit_batches = fixed_validation_batches(
        args.data_dir,
        int(protocol["metric_batch_size"]),
        int(protocol["metric_block_size"]),
        int(protocol["metric_batches"]),
        int(protocol["fit_metric_seed"]),
    )
    eval_batches = fixed_validation_batches(
        args.data_dir,
        int(protocol["metric_batch_size"]),
        int(protocol["metric_block_size"]),
        int(protocol["metric_batches"]),
        int(protocol["eval_metric_seed"]),
    )
    fit_batch_sha = batch_digest(fit_batches)
    eval_batch_sha = batch_digest(eval_batches)
    if fit_batch_sha == eval_batch_sha:
        raise ValueError("fit and evaluation metric batches are identical")
    fit_metrics = terminal_attention_metrics(
        args.terminal_checkpoint, fit_batches, layers, args.device
    )
    eval_metrics = terminal_attention_metrics(
        args.terminal_checkpoint, eval_batches, layers, args.device
    )

    rows: list[dict[str, Any]] = []
    for layer in layers:
        print(f"analyzing moving quotient layer {layer}", flush=True)
        fit_sources = fit_metrics[layer]["value_sources"]
        eval_sources = eval_metrics[layer]["value_sources"]
        pairs = {
            step: tuple(value.to(args.device) for value in split_v_o(
                snapshots[step], layer, n_embd
            ))
            for step in steps
        }
        fit_outputs = {
            step: quotient_output(fit_sources, *pairs[step]).reshape(-1)
            for step in steps
        }
        eval_outputs = {
            step: quotient_output(eval_sources, *pairs[step]).reshape(-1)
            for step in steps
        }
        fit_probe_directions: dict[int, torch.Tensor] = {}
        eval_probe_directions: dict[int, torch.Tensor] = {}
        for step in probe_steps:
            parameter = probes[step]["parameters"]
            current_v, current_o = split_v_o(
                {name: value["weight_before_step"] for name, value in parameter.items()},
                layer,
                n_embd,
            )
            direction_v, direction_o = split_v_o(
                {name: value["applied_direction_per_lr"] for name, value in parameter.items()},
                layer,
                n_embd,
            )
            values = [
                current_v.to(args.device),
                current_o.to(args.device),
                direction_v.to(args.device),
                direction_o.to(args.device),
            ]
            fit_probe_directions[step] = quotient_tangent(
                fit_sources, *values
            ).reshape(-1)
            eval_probe_directions[step] = quotient_tangent(
                eval_sources, *values
            ).reshape(-1)

        all_chords = list(zip(steps[:-1], steps[1:], strict=True))
        target_specs: list[tuple[str, str, int, int]] = []
        for start, end in all_chords:
            if (
                start >= int(protocol["selection_chord_start_minimum"])
                and end <= int(protocol["selection_chord_end_maximum"])
            ):
                target_specs.append(("selection", "chord", start, end))
            elif start >= int(protocol["test_chord_start_minimum"]):
                target_specs.append(("test", "chord", start, end))
        target_specs.extend(
            [
                (
                    "selection",
                    "muon_direction",
                    int(protocol["selection_muon_step"]),
                    int(protocol["selection_muon_step"]),
                ),
                (
                    "test",
                    "muon_direction",
                    int(protocol["test_muon_step"]),
                    int(protocol["test_muon_step"]),
                ),
            ]
        )
        allowed_coordinates = round(
            2 * n_embd * n_embd * float(protocol["coordinate_fraction"])
        )
        for window in windows:
            for phase, kind, start, end in target_specs:
                target_time = start
                past_chords = [pair for pair in all_chords if pair[1] <= target_time]
                past_chords = past_chords[-window:]
                past_probes = [step for step in probe_steps if step < target_time]
                fit_atoms = [
                    fit_outputs[chord_end] - fit_outputs[chord_start]
                    for chord_start, chord_end in past_chords
                ] + [fit_probe_directions[step] for step in past_probes]
                eval_atoms = [
                    eval_outputs[chord_end] - eval_outputs[chord_start]
                    for chord_start, chord_end in past_chords
                ] + [eval_probe_directions[step] for step in past_probes]
                if not fit_atoms or len(fit_atoms) > allowed_coordinates:
                    raise RuntimeError("invalid causal moving-atlas budget")
                fit_basis = torch.stack(fit_atoms, dim=1)
                eval_basis = torch.stack(eval_atoms, dim=1)
                if kind == "chord":
                    fit_target = fit_outputs[end] - fit_outputs[start]
                    eval_target = eval_outputs[end] - eval_outputs[start]
                else:
                    fit_target = fit_probe_directions[start]
                    eval_target = eval_probe_directions[start]
                recovery, energy, rank = fit_and_transport(
                    fit_basis,
                    eval_basis,
                    fit_target,
                    eval_target,
                    float(protocol["span_relative_cutoff"]),
                )
                rows.append(
                    {
                        "window": window,
                        "phase": phase,
                        "kind": kind,
                        "layer": layer,
                        "step_start": start,
                        "step_end": end,
                        "atlas_atoms": len(fit_atoms),
                        "atlas_rank": rank,
                        "past_chord_count": len(past_chords),
                        "past_probe_steps": "|".join(str(value) for value in past_probes),
                        "allowed_coordinates": allowed_coordinates,
                        "eval_recovery": recovery,
                        "eval_energy": energy,
                    }
                )
        del pairs, fit_outputs, eval_outputs, fit_probe_directions, eval_probe_directions
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    selection_summaries = {
        str(window): phase_summary(rows, window, "selection") for window in windows
    }
    selected_window = min(
        windows,
        key=lambda window: (-selection_score(selection_summaries[str(window)]), window),
    )
    test_summary = phase_summary(rows, selected_window, "test")
    thresholds = plan["decision_rule"]["thresholds"]
    checks: dict[str, bool] = {}
    for kind in ("chord", "muon_direction"):
        metric = test_summary[kind]
        checks[f"{kind}_aggregate"] = float(metric["aggregate_eval_recovery"]) >= float(
            thresholds["aggregate_recovery_minimum"]
        )
        checks[f"{kind}_every_layer"] = float(
            metric["minimum_layer_eval_recovery"]
        ) >= float(thresholds["minimum_every_layer_recovery"])
        checks[f"{kind}_late_layers"] = float(
            metric["minimum_late_layer_eval_recovery"]
        ) >= float(thresholds["minimum_late_layer_8_to_11_recovery"])
    passed = all(checks.values())

    args.output_dir.mkdir(parents=True)
    cells_path = args.output_dir / "attention_vo_moving_tangent_cells.csv"
    write_rows(cells_path, rows)
    result = {
        "schema_version": RESULT_SCHEMA,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "classification": (
            "ATTENTION_VO_MOVING_TANGENT_PASS"
            if passed
            else "ATTENTION_VO_MOVING_TANGENT_REJECT"
        ),
        "execution": {
            "host": "PRO6",
            "device": args.device,
            "git_commit": git_commit(),
            "parameter_updates": 0,
            "elapsed_seconds": time.time() - started,
        },
        "identity": {
            "plan_sha256": file_sha256(args.plan),
            "trajectory_inventory_sha256": inventory_sha,
            "trajectory_run_identity_sha256": run_identity,
            "terminal_checkpoint_sha256": file_sha256(args.terminal_checkpoint),
            "dataset_manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
            "fit_metric_batch_sha256": fit_batch_sha,
            "eval_metric_batch_sha256": eval_batch_sha,
        },
        "protocol": protocol,
        "selection_summaries": selection_summaries,
        "selection_scores": {
            str(window): selection_score(selection_summaries[str(window)])
            for window in windows
        },
        "selected_window": selected_window,
        "test_summary": test_summary,
        "checks": checks,
        "decision": {
            "causal_moving_tangent_design_gate_authorized": passed,
            "model_implementation_authorized": False,
            "mfu_preflight_authorized": False,
            "language_model_training_authorized": False,
            "larger_rung_authorized": False,
        },
        "artifacts": {
            "cells": {"path": str(cells_path), "sha256": file_sha256(cells_path)}
        },
        "all_reported_values_finite": all_finite(
            {"selection": selection_summaries, "test": test_summary}
        ),
    }
    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
