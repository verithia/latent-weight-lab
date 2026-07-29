#!/usr/bin/env python3
"""Test coherent phase-start Muon state against future dense c_proj motion."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.nanogpt.analyze_mlp_bilateral_phase_capture import (
    file_sha256,
    git_commit,
)
from examples.nanogpt.analyze_mlp_task_gradient_direction import (
    direction_metrics,
    right_orthogonal_tangent,
    span_recovery,
)
from examples.nanogpt.analyze_parameter_trajectory import (
    load_snapshots,
    parse_int_list,
    write_csv,
)
from examples.nanogpt.parameter_trajectory import (
    OPTIMIZER_PROBE_SCHEMA_VERSION,
)


DIRECTION_ORDER = (
    "exact_applied_direction",
    "exact_applied_right_tangent",
    "muon_polar_descent",
    "muon_polar_right_tangent",
    "combined_momentum_descent",
    "raw_gradient_descent",
)
DESCRIPTIVE_DIRECTION = "momentum_buffer_descent"
ALL_DIRECTION_NAMES = (*DIRECTION_ORDER, DESCRIPTIVE_DIRECTION)


def reconstruct_directions(
    state: dict[str, torch.Tensor],
    hyperparameters: dict[str, float | int],
) -> dict[str, torch.Tensor]:
    """Reconstruct the signed directions used by the exact Muon step."""
    required = {
        "weight_before_step",
        "gradient_after_clip",
        "momentum_buffer_before_step",
        "combined_momentum_update",
        "polar_update",
        "applied_direction_per_lr",
    }
    if set(state) != required:
        raise ValueError("optimizer probe tensor inventory is incomplete")
    weight = state["weight_before_step"].float()
    gradient = state["gradient_after_clip"].float()
    buffer = state["momentum_buffer_before_step"].float()
    combined = state["combined_momentum_update"].float()
    polar = state["polar_update"].float()
    applied = state["applied_direction_per_lr"].float()
    if any(value.shape != weight.shape for value in state.values()):
        raise ValueError("optimizer probe tensors must have one matrix shape")

    momentum = float(hyperparameters["momentum"])
    weight_decay = float(hyperparameters["weight_decay"])
    polar_scale = float(hyperparameters["polar_scale"])
    expected_combined = gradient + momentum * (
        momentum * buffer + gradient
    )
    expected_applied = -weight_decay * weight - polar_scale * polar
    torch.testing.assert_close(combined, expected_combined, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(applied, expected_applied, rtol=2e-6, atol=2e-6)

    polar_descent = -polar
    directions = {
        "raw_gradient_descent": -gradient,
        "momentum_buffer_descent": -buffer,
        "combined_momentum_descent": -combined,
        "muon_polar_descent": polar_descent,
        "exact_applied_direction": applied,
        "muon_polar_right_tangent": right_orthogonal_tangent(
            weight, polar_descent
        ),
        "exact_applied_right_tangent": right_orthogonal_tangent(
            weight, applied
        ),
    }
    return directions


def aggregate_rows(
    rows: list[dict[str, Any]],
    expected_cells: int,
) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for name in ALL_DIRECTION_NAMES:
        selected = [row for row in rows if row["direction"] == name]
        if not selected:
            continue
        energy = torch.tensor(
            [float(row["target_chord_fro"]) ** 2 for row in selected],
            dtype=torch.float64,
        )

        def weighted(key: str) -> float:
            values = torch.tensor(
                [float(row[key]) for row in selected],
                dtype=torch.float64,
            )
            return float((energy * values).sum() / energy.sum())

        aggregate[name] = {
            "cells": len(selected),
            "energy_weighted_cosine": weighted("cosine"),
            "energy_weighted_positive_step_line_recovery": weighted(
                "positive_step_line_recovery"
            ),
            "minimum_cell_cosine": min(
                float(row["cosine"]) for row in selected
            ),
            "maximum_cell_cosine": max(
                float(row["cosine"]) for row in selected
            ),
            "positive_cells": sum(
                float(row["cosine"]) > 0.0 for row in selected
            ),
        }

    promoted: str | None = None
    for name in DIRECTION_ORDER:
        metrics = aggregate[name]
        if (
            int(metrics["cells"]) == expected_cells
            and float(
                metrics["energy_weighted_positive_step_line_recovery"]
            )
            >= 0.10
            and int(metrics["positive_cells"]) == expected_cells
        ):
            promoted = name
            break
    aggregate["promoted_direction"] = promoted
    aggregate["decision"] = (
        f"PROMOTE_COHERENT_MUON_DIRECTION_{promoted.upper()}"
        if promoted is not None
        else "REJECT_SINGLE_PHASE_START_MUON_STATE_DIRECTIONS"
    )
    return aggregate


def grouped_recovery(
    rows: list[dict[str, Any]],
    direction: str,
    key: str,
) -> dict[str, float]:
    selected = [row for row in rows if row["direction"] == direction]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[str(row[key])].append(row)
    output: dict[str, float] = {}
    for value, group in sorted(grouped.items()):
        energy = torch.tensor(
            [float(row["target_chord_fro"]) ** 2 for row in group],
            dtype=torch.float64,
        )
        recovery = torch.tensor(
            [
                float(row["positive_step_line_recovery"])
                for row in group
            ],
            dtype=torch.float64,
        )
        output[value] = float((energy * recovery).sum() / energy.sum())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--probe-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", default="0,3,6,9,11")
    parser.add_argument("--phase-boundaries", default="0,60,120,180,238")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    layers = parse_int_list(args.layers)
    boundaries = parse_int_list(args.phase_boundaries)
    if (
        not layers
        or len(boundaries) < 2
        or boundaries != sorted(set(boundaries))
    ):
        raise ValueError("invalid layers or phase boundaries")
    phase_pairs = list(zip(boundaries[:-1], boundaries[1:], strict=True))
    snapshot_paths = [
        args.snapshot_dir / f"step_{step:06d}.pt" for step in boundaries
    ]
    probe_paths = [
        args.probe_dir / f"step_{step:06d}.pt"
        for step, _end in phase_pairs
    ]
    missing = [
        str(path)
        for path in (*snapshot_paths, *probe_paths)
        if not path.is_file()
    ]
    if missing:
        raise ValueError(f"required optimizer-state inputs are absent: {missing}")

    steps, values, snapshot_metadata = load_snapshots(
        snapshot_paths,
        layers=set(layers),
        targets={"mlp.c_proj"},
    )
    if steps != boundaries:
        raise ValueError("loaded snapshot steps do not match phase boundaries")
    step_index = {step: index for index, step in enumerate(steps)}

    rows: list[dict[str, Any]] = []
    span_rows: list[dict[str, Any]] = []
    run_identity_sha256: str | None = None
    execution_provenance: dict[str, Any] | None = None
    for phase_start, phase_end in phase_pairs:
        path = args.probe_dir / f"step_{phase_start:06d}.pt"
        probe = torch.load(path, map_location="cpu", weights_only=False)
        if probe.get("schema_version") != OPTIMIZER_PROBE_SCHEMA_VERSION:
            raise ValueError(f"unexpected optimizer probe schema: {path}")
        if int(probe.get("step", -1)) != phase_start:
            raise ValueError(f"optimizer probe step mismatch: {path}")
        observed_identity = probe.get("run_identity_sha256")
        if run_identity_sha256 is None:
            run_identity_sha256 = observed_identity
            execution_provenance = probe.get("execution_provenance")
        elif observed_identity != run_identity_sha256:
            raise ValueError("optimizer probes do not share one run identity")

        for layer in layers:
            parameter = f"transformer.h.{layer}.mlp.c_proj.weight"
            source = values[parameter][step_index[phase_start]].float()
            target = values[parameter][step_index[phase_end]].float()
            state = probe["parameters"][parameter]
            torch.testing.assert_close(
                state["weight_before_step"],
                source,
                rtol=0.0,
                atol=0.0,
            )
            chord = (target - source).to(args.device)
            state_on_device = {
                name: tensor.to(args.device)
                for name, tensor in state.items()
            }
            directions = reconstruct_directions(
                state_on_device,
                probe["hyperparameters"][parameter],
            )
            active_base_directions = [
                directions[name]
                for name in (
                    "raw_gradient_descent",
                    "momentum_buffer_descent",
                    "combined_momentum_descent",
                    "muon_polar_descent",
                    "exact_applied_direction",
                )
                if float(directions[name].norm()) > 0.0
            ]
            span_rows.append(
                {
                    "parameter": parameter,
                    "layer": layer,
                    "phase_start": phase_start,
                    "phase_end": phase_end,
                    "causal_direction_span_recovery": span_recovery(
                        chord, active_base_directions
                    ),
                    "target_chord_fro": float(chord.double().norm()),
                }
            )
            for name in ALL_DIRECTION_NAMES:
                direction = directions[name]
                if float(direction.norm()) == 0.0:
                    continue
                row = {
                    "parameter": parameter,
                    "layer": layer,
                    "phase_start": phase_start,
                    "phase_end": phase_end,
                    "direction": name,
                    **direction_metrics(chord, direction),
                }
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
            del chord, state_on_device, directions
        del probe
        if "cuda" in args.device:
            torch.cuda.empty_cache()

    expected_cells = len(layers) * len(phase_pairs)
    aggregate = aggregate_rows(rows, expected_cells)
    span_energy = torch.tensor(
        [float(row["target_chord_fro"]) ** 2 for row in span_rows],
        dtype=torch.float64,
    )
    span_values = torch.tensor(
        [float(row["causal_direction_span_recovery"]) for row in span_rows],
        dtype=torch.float64,
    )
    aggregate["causal_direction_span"] = {
        "cells": len(span_rows),
        "energy_weighted_recovery": float(
            (span_energy * span_values).sum() / span_energy.sum()
        ),
    }
    aggregate["exact_applied_recovery_by_phase_start"] = grouped_recovery(
        rows, "exact_applied_direction", "phase_start"
    )
    aggregate["exact_applied_recovery_by_layer"] = grouped_recovery(
        rows, "exact_applied_direction", "layer"
    )

    args.output.mkdir(parents=True, exist_ok=True)
    detail_path = args.output / "optimizer_state_direction.csv"
    span_path = args.output / "optimizer_state_direction_span.csv"
    aggregate_path = args.output / "optimizer_state_direction_aggregate.json"
    write_csv(detail_path, rows)
    write_csv(span_path, span_rows)
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_mlp_optimizer_state_direction_v1",
        "decision": aggregate["decision"],
        "promoted_direction": aggregate["promoted_direction"],
        "decision_rule": (
            "promote the first registered direction with >=10% "
            "energy-weighted positive-step line recovery and positive cosine "
            "in all 20 layer/phase cells"
        ),
        "direction_order": list(DIRECTION_ORDER),
        "descriptive_direction": DESCRIPTIVE_DIRECTION,
        "layers": layers,
        "phase_boundaries": boundaries,
        "target": "mlp.c_proj",
        "run_identity_sha256": run_identity_sha256,
        "execution_provenance": execution_provenance,
        "snapshot_metadata": snapshot_metadata,
        "analysis_execution": {
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": str(script),
            "entrypoint_sha256": file_sha256(script),
            "command": sys.argv,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "device": args.device,
        },
        "input_files": {
            "snapshots": [
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
                for path in snapshot_paths
            ],
            "optimizer_probes": [
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
                for path in probe_paths
            ],
        },
        "outputs": {
            "detail_sha256": file_sha256(detail_path),
            "span_sha256": file_sha256(span_path),
            "aggregate_sha256": file_sha256(aggregate_path),
        },
        "limitations": [
            "A phase-start direction is compared with a 58-60 update future chord.",
            "The probe captures exact training state but only five representative layers.",
            "Line recovery tests one frozen direction and does not fit a trainable language model.",
        ],
    }
    metadata_path = args.output / "optimizer_state_direction_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "rows": len(rows),
                "span_rows": len(span_rows),
                "aggregate": aggregate,
                "metadata": str(metadata_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
