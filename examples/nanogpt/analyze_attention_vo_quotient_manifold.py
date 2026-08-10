#!/usr/bin/env python3
"""Discovery/held-out manifold oracle in the coupled attention V/O quotient.

For head ``h``, the frozen-QK attention residual depends on
``M_h = O[:, h_slice] @ V_h``.  Hidden-head rotations of V and O are gauge,
so this oracle never scores the two weights independently.  It builds a tiny
discovery-only functional atlas from exact quotient states, local chords, and
optimizer tangents, then transports fitted coefficients to disjoint batches
and held-out training time.  No parameter is updated.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

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
from examples.nanogpt.analyze_mlp_cproj_paper_activation_oracle import (
    explained_energy,
)
from examples.nanogpt.analyze_residual_compatibility import fixed_validation_batches


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_SCHEMA = "mai_124m_attention_vo_quotient_manifold_plan_v1"
RESULT_SCHEMA = "mai_124m_attention_vo_quotient_manifold_result_v1"


def split_v_o(
    parameters: dict[str, torch.Tensor], layer: int, n_embd: int
) -> tuple[torch.Tensor, torch.Tensor]:
    prefix = f"transformer.h.{int(layer)}.attn"
    packed = parameters[f"{prefix}.c_attn.weight"].float()
    output = parameters[f"{prefix}.c_proj.weight"].float()
    return packed[2 * int(n_embd) :], output


def quotient_output(
    value_sources: torch.Tensor,
    value_weight: torch.Tensor,
    output_weight: torch.Tensor,
) -> torch.Tensor:
    heads = int(value_sources.shape[1])
    if value_weight.shape[0] % heads or output_weight.shape[1] % heads:
        raise ValueError("V/O weights do not divide evenly into attention heads")
    head_dim = value_weight.shape[0] // heads
    states = [
        F.linear(
            value_sources[:, head],
            value_weight[head * head_dim : (head + 1) * head_dim],
        )
        for head in range(heads)
    ]
    return F.linear(torch.cat(states, dim=-1), output_weight)


def quotient_tangent(
    value_sources: torch.Tensor,
    value_weight: torch.Tensor,
    output_weight: torch.Tensor,
    value_direction: torch.Tensor,
    output_direction: torch.Tensor,
) -> torch.Tensor:
    heads = int(value_sources.shape[1])
    head_dim = value_weight.shape[0] // heads
    states = []
    direction_states = []
    for head in range(heads):
        section = slice(head * head_dim, (head + 1) * head_dim)
        states.append(F.linear(value_sources[:, head], value_weight[section]))
        direction_states.append(
            F.linear(value_sources[:, head], value_direction[section])
        )
    state = torch.cat(states, dim=-1)
    direction_state = torch.cat(direction_states, dim=-1)
    return F.linear(direction_state, output_weight) + F.linear(
        state, output_direction
    )


def validate_plan(plan: dict[str, Any], args: argparse.Namespace) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unexpected V/O quotient plan schema")
    protocol = plan["protocol"]
    frozen = {
        "parameter_updates": 0,
        "coordinate_fraction": 0.01,
        "trajectory_discovery_max_step": 1140,
        "trajectory_heldout_min_step": 1200,
        "discovery_probe_steps": [0, 594],
        "heldout_probe_steps": [1782, 2372],
        "fit_metric_seed": 20260809,
        "eval_metric_seed": 20260810,
        "metric_batch_size": 2,
        "metric_block_size": 256,
        "metric_batches": 2,
        "span_relative_cutoff": 1e-8,
    }
    for field, expected in frozen.items():
        if protocol.get(field) != expected:
            raise ValueError(f"frozen V/O quotient protocol changed: {field}")
    if plan["decision_rule"]["thresholds"] != {
        "aggregate_recovery_minimum": 0.9,
        "minimum_every_layer_recovery": 0.75,
        "minimum_late_layer_8_to_11_recovery": 0.75,
    }:
        raise ValueError("V/O quotient thresholds changed")
    if any(bool(value) for value in plan["authorization"].values()):
        raise ValueError("V/O quotient oracle must not pre-authorize a successor")
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
            raise ValueError(f"pinned V/O quotient identity mismatch: {path}")
    inventory, digest = trajectory_inventory(args.trajectory_dir)
    if (
        len(inventory) != int(identity["trajectory_file_count"])
        or sum(int(item["size"]) for item in inventory)
        != int(identity["trajectory_total_bytes"])
        or digest != identity["trajectory_inventory_sha256"]
    ):
        raise ValueError("V/O trajectory inventory mismatch")
    for name, expected in identity["optimizer_probe_sha256"].items():
        path = args.probe_dir / name
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"V/O optimizer probe mismatch: {path}")
    if Path(identity["trajectory_directory"]) != args.trajectory_dir:
        raise ValueError("trajectory directory differs from plan")
    if Path(identity["optimizer_probe_directory"]) != args.probe_dir:
        raise ValueError("probe directory differs from plan")
    if Path(identity["output_directory_must_be_absent"]) != args.output_dir:
        raise ValueError("output directory differs from plan")


def fit_and_transport(
    fit_basis: torch.Tensor,
    eval_basis: torch.Tensor,
    fit_target: torch.Tensor,
    eval_target: torch.Tensor,
    relative_cutoff: float,
) -> tuple[float, float, int]:
    coefficients, rank = solve_span_coefficients(
        fit_basis, fit_target, float(relative_cutoff)
    )
    recovery, energy = explained_energy(eval_target, eval_basis @ coefficients)
    return recovery, energy, rank


def summarize(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for kind in ("state", "chord", "muon_direction"):
        selected = [row for row in rows if row["arm"] == arm and row["kind"] == kind]
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
            "maximum_atlas_rank": max(int(row["atlas_rank"]) for row in selected),
            "maximum_atlas_atoms": max(int(row["atlas_atoms"]) for row in selected),
        }
    return output


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
    discovery_max = int(protocol["trajectory_discovery_max_step"])
    heldout_min = int(protocol["trajectory_heldout_min_step"])
    discovery_probes = [int(value) for value in protocol["discovery_probe_steps"]]
    heldout_probes = [int(value) for value in protocol["heldout_probe_steps"]]
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
    for step in discovery_probes + heldout_probes:
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
    atlas_rows: list[dict[str, Any]] = []
    for layer in layers:
        print(f"analyzing quotient layer {layer}", flush=True)
        fit_sources = fit_metrics[layer]["value_sources"]
        eval_sources = eval_metrics[layer]["value_sources"]
        pairs = {
            step: tuple(value.to(args.device) for value in split_v_o(
                snapshots[step], layer, n_embd
            ))
            for step in steps
        }
        initial_v, initial_o = pairs[steps[0]]
        fit_outputs = {
            step: quotient_output(fit_sources, *pairs[step]).reshape(-1)
            for step in steps
        }
        eval_outputs = {
            step: quotient_output(eval_sources, *pairs[step]).reshape(-1)
            for step in steps
        }
        fit_initial = fit_outputs[steps[0]]
        eval_initial = eval_outputs[steps[0]]

        joint_fit_atoms: list[torch.Tensor] = []
        joint_eval_atoms: list[torch.Tensor] = []
        tangent_fit_atoms: list[torch.Tensor] = []
        tangent_eval_atoms: list[torch.Tensor] = []
        atom_labels: list[str] = []
        discovery_steps = [step for step in steps if 0 < step <= discovery_max]
        for step in discovery_steps:
            current_v, current_o = pairs[step]
            delta_v = current_v - initial_v
            delta_o = current_o - initial_o
            joint_fit_atoms.append(fit_outputs[step] - fit_initial)
            joint_eval_atoms.append(eval_outputs[step] - eval_initial)
            tangent_fit_atoms.append(
                quotient_tangent(
                    fit_sources, initial_v, initial_o, delta_v, delta_o
                ).reshape(-1)
            )
            tangent_eval_atoms.append(
                quotient_tangent(
                    eval_sources, initial_v, initial_o, delta_v, delta_o
                ).reshape(-1)
            )
            atom_labels.append(f"state_{step}")
        for start, end in zip(steps[:-1], steps[1:], strict=True):
            if end > discovery_max:
                break
            start_v, start_o = pairs[start]
            end_v, end_o = pairs[end]
            delta_v = end_v - start_v
            delta_o = end_o - start_o
            joint_fit_atoms.append(fit_outputs[end] - fit_outputs[start])
            joint_eval_atoms.append(eval_outputs[end] - eval_outputs[start])
            tangent_fit_atoms.append(
                quotient_tangent(
                    fit_sources, initial_v, initial_o, delta_v, delta_o
                ).reshape(-1)
            )
            tangent_eval_atoms.append(
                quotient_tangent(
                    eval_sources, initial_v, initial_o, delta_v, delta_o
                ).reshape(-1)
            )
            atom_labels.append(f"chord_{start}_{end}")
        for step in discovery_probes:
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
            current_v = current_v.to(args.device)
            current_o = current_o.to(args.device)
            direction_v = direction_v.to(args.device)
            direction_o = direction_o.to(args.device)
            joint_fit_atoms.append(
                quotient_tangent(
                    fit_sources, current_v, current_o, direction_v, direction_o
                ).reshape(-1)
            )
            joint_eval_atoms.append(
                quotient_tangent(
                    eval_sources, current_v, current_o, direction_v, direction_o
                ).reshape(-1)
            )
            tangent_fit_atoms.append(
                quotient_tangent(
                    fit_sources, initial_v, initial_o, direction_v, direction_o
                ).reshape(-1)
            )
            tangent_eval_atoms.append(
                quotient_tangent(
                    eval_sources, initial_v, initial_o, direction_v, direction_o
                ).reshape(-1)
            )
            atom_labels.append(f"muon_{step}")

        bases = {
            "joint_quotient_discovery": (
                torch.stack(joint_fit_atoms, dim=1),
                torch.stack(joint_eval_atoms, dim=1),
            ),
            "initial_tangent_discovery": (
                torch.stack(tangent_fit_atoms, dim=1),
                torch.stack(tangent_eval_atoms, dim=1),
            ),
        }
        allowed_coordinates = round(
            2 * n_embd * n_embd * float(protocol["coordinate_fraction"])
        )
        if len(atom_labels) > allowed_coordinates:
            raise RuntimeError("discovery atlas exceeds frozen coordinate budget")
        atlas_rows.append(
            {
                "layer": layer,
                "atlas_atoms": len(atom_labels),
                "allowed_coordinates": allowed_coordinates,
                "coordinate_fraction_used": len(atom_labels) / (2 * n_embd * n_embd),
                "atom_labels": "|".join(atom_labels),
            }
        )

        heldout_steps = [step for step in steps if step >= heldout_min]
        heldout_chords = [
            (start, end)
            for start, end in zip(steps[:-1], steps[1:], strict=True)
            if start >= heldout_min
        ]
        targets: list[tuple[str, int, int, torch.Tensor, torch.Tensor]] = []
        targets.extend(
            (
                "state",
                0,
                step,
                fit_outputs[step] - fit_initial,
                eval_outputs[step] - eval_initial,
            )
            for step in heldout_steps
        )
        targets.extend(
            (
                "chord",
                start,
                end,
                fit_outputs[end] - fit_outputs[start],
                eval_outputs[end] - eval_outputs[start],
            )
            for start, end in heldout_chords
        )
        for step in heldout_probes:
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
            current_v = current_v.to(args.device)
            current_o = current_o.to(args.device)
            direction_v = direction_v.to(args.device)
            direction_o = direction_o.to(args.device)
            targets.append(
                (
                    "muon_direction",
                    step,
                    step,
                    quotient_tangent(
                        fit_sources,
                        current_v,
                        current_o,
                        direction_v,
                        direction_o,
                    ).reshape(-1),
                    quotient_tangent(
                        eval_sources,
                        current_v,
                        current_o,
                        direction_v,
                        direction_o,
                    ).reshape(-1),
                )
            )
        for arm, (fit_basis, eval_basis) in bases.items():
            for kind, start, end, fit_target, eval_target in targets:
                recovery, energy, rank = fit_and_transport(
                    fit_basis,
                    eval_basis,
                    fit_target,
                    eval_target,
                    float(protocol["span_relative_cutoff"]),
                )
                rows.append(
                    {
                        "arm": arm,
                        "kind": kind,
                        "layer": layer,
                        "step_start": start,
                        "step_end": end,
                        "atlas_atoms": fit_basis.shape[1],
                        "atlas_rank": rank,
                        "allowed_coordinates": allowed_coordinates,
                        "eval_recovery": recovery,
                        "eval_energy": energy,
                    }
                )
        del pairs, fit_outputs, eval_outputs
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    summaries = {
        arm: summarize(rows, arm)
        for arm in ("joint_quotient_discovery", "initial_tangent_discovery")
    }
    thresholds = plan["decision_rule"]["thresholds"]
    primary = summaries["joint_quotient_discovery"]
    checks: dict[str, bool] = {}
    for kind in ("state", "chord", "muon_direction"):
        metric = primary[kind]
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
    cells_path = args.output_dir / "attention_vo_quotient_manifold_cells.csv"
    atlas_path = args.output_dir / "attention_vo_quotient_manifold_atlas.csv"
    write_rows(cells_path, rows)
    write_rows(atlas_path, atlas_rows)
    result = {
        "schema_version": RESULT_SCHEMA,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "classification": (
            "ATTENTION_VO_QUOTIENT_MANIFOLD_PASS"
            if passed
            else "ATTENTION_VO_QUOTIENT_MANIFOLD_REJECT"
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
        "summaries": summaries,
        "checks": checks,
        "decision": {
            "paired_two_sided_decoder_design_gate_authorized": passed,
            "model_implementation_authorized": False,
            "mfu_preflight_authorized": False,
            "language_model_training_authorized": False,
            "larger_rung_authorized": False,
        },
        "artifacts": {
            "cells": {"path": str(cells_path), "sha256": file_sha256(cells_path)},
            "atlas": {"path": str(atlas_path), "sha256": file_sha256(atlas_path)},
        },
        "all_reported_values_finite": all_finite(summaries),
    }
    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
