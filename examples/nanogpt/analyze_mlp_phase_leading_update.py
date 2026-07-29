#!/usr/bin/env python3
"""Measure whether early dense Muon motion predicts each full phase chord.

The saved trajectory does not contain boundary gradients or optimizer states.
It does contain exact MLP weights every six updates.  This diagnostic uses
the realized first 6/12/18-update displacement of each registered phase as a
conservative proxy for a phase-start task-conditioned direction, then
measures its one-dimensional capture of the complete phase chord.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
from examples.nanogpt.analyze_parameter_trajectory import (
    load_snapshots,
    parse_int_list,
    write_csv,
)


def leading_direction_metrics(
    lead: torch.Tensor,
    chord: torch.Tensor,
) -> dict[str, float]:
    if lead.shape != chord.shape or lead.ndim != 2:
        raise ValueError("lead and chord must be same-shaped matrices")
    lead_energy = lead.square().sum()
    chord_energy = chord.square().sum()
    if float(lead_energy) <= 0.0 or float(chord_energy) <= 0.0:
        raise ValueError("lead and chord must be nonzero")
    dot = (lead * chord).sum()
    cosine = dot / (lead_energy.sqrt() * chord_energy.sqrt())
    scale = dot / lead_energy
    residual = chord - scale * lead
    return {
        "cosine": float(cosine),
        "signed_capture": float(
            cosine.square() * torch.sign(cosine)
        ),
        "one_direction_energy_capture": float(cosine.square()),
        "optimal_lead_scale": float(scale),
        "lead_to_chord_fro_ratio": float(
            lead_energy.sqrt() / chord_energy.sqrt()
        ),
        "optimal_residual_energy_fraction": float(
            residual.square().sum() / chord_energy
        ),
        "lead_fro": float(lead_energy.sqrt()),
        "chord_fro": float(chord_energy.sqrt()),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for lookahead in sorted({int(row["lookahead"]) for row in rows}):
        selected = [
            row for row in rows if int(row["lookahead"]) == lookahead
        ]
        energy = torch.tensor(
            [float(row["chord_fro"]) ** 2 for row in selected],
            dtype=torch.float64,
        )
        capture = torch.tensor(
            [
                float(row["one_direction_energy_capture"])
                for row in selected
            ],
            dtype=torch.float64,
        )
        cosine = torch.tensor(
            [float(row["cosine"]) for row in selected],
            dtype=torch.float64,
        )
        result.append(
            {
                "lookahead": lookahead,
                "cells": len(selected),
                "energy_weighted_capture": float(
                    (energy * capture).sum()
                    / energy.sum().clamp_min(1e-30)
                ),
                "energy_weighted_cosine": float(
                    (energy * cosine).sum()
                    / energy.sum().clamp_min(1e-30)
                ),
                "mean_capture": float(capture.mean()),
                "minimum_capture": float(capture.min()),
                "maximum_capture": float(capture.max()),
                "minimum_cosine": float(cosine.min()),
                "positive_cosine_fraction": float((cosine > 0).double().mean()),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", default="0,3,6,9,11")
    parser.add_argument(
        "--phase-boundaries", default="0,60,120,180,238"
    )
    parser.add_argument("--lookaheads", default="6,12,18")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    layers = parse_int_list(args.layers)
    boundaries = parse_int_list(args.phase_boundaries)
    lookaheads = parse_int_list(args.lookaheads)
    if (
        not layers
        or len(boundaries) < 2
        or boundaries != sorted(set(boundaries))
        or not lookaheads
        or min(lookaheads) <= 0
    ):
        raise ValueError("invalid layers, boundaries, or lookaheads")
    required_steps = set(boundaries)
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        for lookahead in lookaheads:
            step = start + lookahead
            if step >= end:
                raise ValueError(
                    f"lookahead {lookahead} reaches phase end {start}->{end}"
                )
            required_steps.add(step)
    ordered_steps = sorted(required_steps)
    paths = [
        args.snapshot_dir / f"step_{step:06d}.pt"
        for step in ordered_steps
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"required snapshots are absent: {missing}")
    steps, values, snapshot_metadata = load_snapshots(
        paths,
        layers=set(layers),
        targets={"mlp.c_proj"},
    )
    if steps != ordered_steps:
        raise ValueError("loaded steps do not match requested steps")
    step_index = {step: index for index, step in enumerate(steps)}

    rows: list[dict[str, Any]] = []
    for name, tensors in sorted(values.items()):
        layer = int(name.split(".")[2])
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            base = tensors[step_index[start]].to(
                device=args.device,
                dtype=torch.float64,
            )
            terminal = tensors[step_index[end]].to(
                device=args.device,
                dtype=torch.float64,
            )
            chord = terminal - base
            for lookahead in lookaheads:
                early = tensors[step_index[start + lookahead]].to(
                    device=args.device,
                    dtype=torch.float64,
                )
                metrics = leading_direction_metrics(
                    early - base,
                    chord,
                )
                row = {
                    "parameter": name,
                    "layer": layer,
                    "phase_start": start,
                    "phase_end": end,
                    "lookahead": lookahead,
                    **metrics,
                }
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
                del early
            del base, terminal, chord
    aggregates = aggregate_rows(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    detail_path = args.output / "phase_leading_update.csv"
    aggregate_path = args.output / "phase_leading_update_aggregate.csv"
    write_csv(detail_path, rows)
    write_csv(aggregate_path, aggregates)

    script = Path(__file__).resolve()
    repo = script.parents[2]
    metadata = {
        "schema_version": "nanogpt_mlp_phase_leading_update_v1",
        "snapshot_metadata": snapshot_metadata,
        "layers": layers,
        "phase_boundaries": boundaries,
        "lookaheads": lookaheads,
        "target": "mlp.c_proj",
        "analysis_dtype": "float64",
        "interpretation": (
            "Realized multi-update Muon displacement is a saved "
            "phase-start direction proxy, not a raw task gradient."
        ),
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
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
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
            "The earliest saved displacement integrates six optimizer updates.",
            "It includes momentum, weight decay, stochastic batches, and schedule effects.",
            "It is not the raw gradient or isolated Muon orthogonalized update.",
            "Only five preregistered representative layers are analyzed.",
        ],
    }
    metadata_path = args.output / "phase_leading_update_metadata.json"
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
