#!/usr/bin/env python3
"""Test phase-start task-gradient directions against future c_proj motion."""

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

from examples.nanogpt.analyze_mlp_activation_update_alignment import (
    file_sha256,
    git_commit,
    load_snapshot,
    model_from_snapshot,
)
from examples.nanogpt.analyze_mlp_activation_matched_givens_fit import (
    tensor_sha256,
)
from examples.nanogpt.analyze_mlp_global_givens_transport_fit import (
    parse_cells,
)
from examples.nanogpt.analyze_parameter_trajectory import (
    load_snapshots,
    parse_int_list,
    write_csv,
)
from examples.nanogpt.analyze_residual_compatibility import (
    fixed_validation_batches,
)
from examples.nanogpt.muon import zeropower_via_newtonschulz5


DIRECTION_NAMES = (
    "raw_descent",
    "muon_polar_descent",
    "raw_right_tangent",
    "muon_right_tangent",
)


def right_orthogonal_tangent(
    weight: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    """Project a direction through the canonical right-orbit skew generator.

    This is ``weight @ skew(weight.T @ direction)`` without materializing the
    square hidden-coordinate generator.
    """
    if weight.ndim != 2 or weight.shape != direction.shape:
        raise ValueError("weight and direction must be same-shaped matrices")
    weight = weight.float()
    direction = direction.float()
    return 0.5 * (
        (weight @ weight.T) @ direction
        - (weight @ direction.T) @ weight
    )


def direction_metrics(
    target: torch.Tensor,
    direction: torch.Tensor,
) -> dict[str, float]:
    """Return signed alignment and positive-step line recovery."""
    target = target.double()
    direction = direction.double()
    target_energy = target.square().sum().clamp_min(1e-30)
    direction_energy = direction.square().sum().clamp_min(1e-30)
    dot = (target * direction).sum()
    cosine = dot / (target_energy * direction_energy).sqrt()
    scale = dot / direction_energy
    positive_dot = dot.clamp_min(0.0)
    positive_recovery = (
        positive_dot.square() / (target_energy * direction_energy)
    )
    return {
        "cosine": float(cosine),
        "optimal_signed_scale": float(scale),
        "positive_step_line_recovery": float(positive_recovery),
        "direction_fro": float(direction_energy.sqrt()),
        "target_chord_fro": float(target_energy.sqrt()),
    }


def span_recovery(
    target: torch.Tensor,
    directions: list[torch.Tensor],
) -> float:
    """Return optimal unrestricted recovery in a small direction span."""
    target = target.double().reshape(-1)
    matrix = torch.stack(
        [direction.double().reshape(-1) for direction in directions],
        dim=1,
    )
    coefficients = torch.linalg.lstsq(matrix, target).solution
    residual = target - matrix @ coefficients
    return float(
        1.0
        - residual.square().sum()
        / target.square().sum().clamp_min(1e-30)
    )


def collect_cproj_gradients(
    model: torch.nn.Module,
    batches: list[torch.Tensor],
    layers: list[int],
    device: str,
) -> tuple[dict[int, torch.Tensor], float]:
    """Average next-token gradients for selected c_proj weights."""
    selected: dict[int, torch.nn.Parameter] = {}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for layer in layers:
        parameter = model.transformer.h[layer].mlp.c_proj.weight
        parameter.requires_grad_(True)
        selected[layer] = parameter
    model.zero_grad(set_to_none=True)
    losses: list[float] = []
    for tokens in batches:
        tokens = tokens.to(device)
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]
        _logits, loss = model(inputs, targets)
        if loss is None:
            raise RuntimeError("model did not return a task loss")
        losses.append(float(loss.detach()))
        (loss / len(batches)).backward()
    gradients: dict[int, torch.Tensor] = {}
    for layer, parameter in selected.items():
        if parameter.grad is None:
            raise RuntimeError(f"missing c_proj gradient for layer {layer}")
        gradients[layer] = parameter.grad.detach().float().cpu()
    return gradients, sum(losses) / len(losses)


def aggregate(
    rows: list[dict[str, Any]],
    gradient_directions: dict[tuple[str, int, int, str], torch.Tensor],
) -> tuple[dict[str, Any], str, str | None]:
    output: dict[str, Any] = {}
    for window in ("fit", "holdout"):
        selected = [row for row in rows if row["gradient_window"] == window]
        energy = torch.tensor(
            [float(row["target_chord_fro"]) ** 2 for row in selected],
            dtype=torch.float64,
        )
        window_result: dict[str, Any] = {
            "cells": len(selected),
            "energy_weighted_span_recovery": float(
                (
                    energy
                    * torch.tensor(
                        [float(row["all_direction_span_recovery"]) for row in selected],
                        dtype=torch.float64,
                    )
                ).sum()
                / energy.sum()
            ),
        }
        for name in DIRECTION_NAMES:
            values = [
                row for row in selected if row["direction"] == name
            ]
            direction_energy = torch.tensor(
                [float(row["target_chord_fro"]) ** 2 for row in values],
                dtype=torch.float64,
            )

            def weighted(key: str) -> float:
                metric = torch.tensor(
                    [float(row[key]) for row in values],
                    dtype=torch.float64,
                )
                return float(
                    (direction_energy * metric).sum()
                    / direction_energy.sum()
                )

            window_result[name] = {
                "energy_weighted_cosine": weighted("cosine"),
                "energy_weighted_positive_step_line_recovery": weighted(
                    "positive_step_line_recovery"
                ),
                "minimum_cell_cosine": min(
                    float(row["cosine"]) for row in values
                ),
                "positive_cells": sum(
                    float(row["cosine"]) > 0.0 for row in values
                ),
            }
        output[window] = window_result

    cross_window: dict[str, Any] = {}
    cells = sorted(
        {
            (int(row["layer"]), int(row["phase_start"]))
            for row in rows
        }
    )
    for name in DIRECTION_NAMES:
        cosines: list[float] = []
        for layer, phase in cells:
            fit = gradient_directions[("fit", layer, phase, name)].double()
            holdout = gradient_directions[
                ("holdout", layer, phase, name)
            ].double()
            cosine = float(
                (fit * holdout).sum()
                / (fit.norm() * holdout.norm()).clamp_min(1e-30)
            )
            cosines.append(cosine)
        cross_window[name] = {
            "mean_direction_cosine": sum(cosines) / len(cosines),
            "minimum_direction_cosine": min(cosines),
        }
    output["cross_window"] = cross_window

    promoted: str | None = None
    for name in DIRECTION_NAMES:
        if (
            min(
                float(
                    output[window][name][
                        "energy_weighted_positive_step_line_recovery"
                    ]
                )
                for window in ("fit", "holdout")
            )
            >= 0.10
            and min(
                int(output[window][name]["positive_cells"])
                for window in ("fit", "holdout")
            )
            == len(cells)
            and float(cross_window[name]["mean_direction_cosine"])
            >= 0.25
        ):
            promoted = name
            break
    decision = (
        f"PROMOTE_TASK_GRADIENT_DIRECTION_{promoted.upper()}"
        if promoted is not None
        else "REJECT_STATELESS_PHASE_START_TASK_GRADIENT_DIRECTIONS"
    )
    output["decision"] = decision
    output["promoted_direction"] = promoted
    return output, decision, promoted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cells", default="0:0,0:180,6:60,11:120")
    parser.add_argument("--phase-boundaries", default="0,60,120,180,238")
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--fit-seed", type=int, default=20260731)
    parser.add_argument("--holdout-seed", type=int, default=20260801)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    cells = parse_cells(args.cells)
    boundaries = parse_int_list(args.phase_boundaries)
    end_by_start = dict(zip(boundaries[:-1], boundaries[1:], strict=True))
    required_steps = sorted(
        {
            step
            for _layer, phase in cells
            for step in (phase, end_by_start[phase])
        }
    )
    paths = [
        args.snapshot_dir / f"step_{step:06d}.pt"
        for step in required_steps
    ]
    if any(not path.is_file() for path in paths):
        raise ValueError("required phase snapshots are absent")
    layers = {layer for layer, _phase in cells}
    steps, values, snapshot_metadata = load_snapshots(
        paths,
        layers=layers,
        targets={"mlp.c_proj"},
    )
    step_index = {step: index for index, step in enumerate(steps)}
    layers_by_phase: dict[int, list[int]] = defaultdict(list)
    for layer, phase in cells:
        layers_by_phase[phase].append(layer)

    windows = (
        ("fit", args.fit_seed),
        ("holdout", args.holdout_seed),
    )
    batches_by_window = {
        window: fixed_validation_batches(
            args.data_dir,
            args.batch_size,
            args.block_size + 1,
            args.batches,
            seed,
        )
        for window, seed in windows
    }
    gradients: dict[tuple[str, int, int], torch.Tensor] = {}
    losses: dict[str, float] = {}
    for phase in sorted(layers_by_phase):
        payload = load_snapshot(
            args.snapshot_dir / f"step_{phase:06d}.pt"
        )
        model = model_from_snapshot(payload, args.device)
        phase_layers = sorted(layers_by_phase[phase])
        try:
            for window, _seed in windows:
                collected, loss = collect_cproj_gradients(
                    model,
                    batches_by_window[window],
                    phase_layers,
                    args.device,
                )
                losses[f"{window}_phase{phase}"] = loss
                for layer, gradient in collected.items():
                    gradients[(window, layer, phase)] = gradient
        finally:
            del model, payload
            if "cuda" in args.device:
                torch.cuda.empty_cache()

    rows: list[dict[str, Any]] = []
    gradient_directions: dict[
        tuple[str, int, int, str], torch.Tensor
    ] = {}
    saved: dict[str, torch.Tensor] = {}
    for layer, phase_start in cells:
        phase_end = end_by_start[phase_start]
        parameter = f"transformer.h.{layer}.mlp.c_proj.weight"
        source = values[parameter][step_index[phase_start]].float()
        target = values[parameter][step_index[phase_end]].float()
        chord = target - source
        for window, _seed in windows:
            gradient = gradients[(window, layer, phase_start)]
            raw_descent = -gradient
            muon_descent = -zeropower_via_newtonschulz5(
                gradient, steps=args.muon_ns_steps
            )
            directions = {
                "raw_descent": raw_descent,
                "muon_polar_descent": muon_descent,
                "raw_right_tangent": right_orthogonal_tangent(
                    source, raw_descent
                ),
                "muon_right_tangent": right_orthogonal_tangent(
                    source, muon_descent
                ),
            }
            span = span_recovery(chord, list(directions.values()))
            saved[f"{window}_layer{layer}_phase{phase_start}_gradient"] = (
                gradient
            )
            for name, direction in directions.items():
                gradient_directions[
                    (window, layer, phase_start, name)
                ] = direction.cpu()
                row = {
                    "parameter": parameter,
                    "layer": layer,
                    "phase_start": phase_start,
                    "phase_end": phase_end,
                    "gradient_window": window,
                    "gradient_loss": losses[
                        f"{window}_phase{phase_start}"
                    ],
                    "gradient_sha256": tensor_sha256(gradient),
                    "direction": name,
                    "all_direction_span_recovery": span,
                    **direction_metrics(chord, direction),
                }
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)

    result, decision, promoted = aggregate(
        rows, gradient_directions
    )
    args.output.mkdir(parents=True, exist_ok=True)
    detail_path = args.output / "task_gradient_direction.csv"
    aggregate_path = args.output / "task_gradient_direction_aggregate.json"
    gradients_path = args.output / "task_gradient_tensors.pt"
    write_csv(detail_path, rows)
    aggregate_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    torch.save(
        {
            "schema_version": "nanogpt_mlp_task_gradient_tensors_v1",
            "fit_seed": args.fit_seed,
            "holdout_seed": args.holdout_seed,
            "batches": args.batches,
            "batch_size": args.batch_size,
            "block_size": args.block_size,
            "gradients": saved,
        },
        gradients_path,
    )
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_mlp_task_gradient_direction_v1",
        "decision": decision,
        "promoted_direction": promoted,
        "decision_rule": (
            "promote the first registered direction with >=10% "
            "energy-weighted positive-step line recovery on both disjoint "
            "windows, positive cosine in every cell, and >=0.25 mean "
            "cross-window direction cosine"
        ),
        "directions": {
            "raw_descent": "-task_gradient",
            "muon_polar_descent": (
                "-Newton-Schulz-5(task_gradient), without unavailable "
                "historical momentum"
            ),
            "raw_right_tangent": (
                "W @ skew(W.T @ -task_gradient)"
            ),
            "muon_right_tangent": (
                "W @ skew(W.T @ -Newton-Schulz-5(task_gradient))"
            ),
        },
        "gradient_protocol": {
            "split": "validation",
            "fit_seed": args.fit_seed,
            "holdout_seed": args.holdout_seed,
            "batches": args.batches,
            "batch_size": args.batch_size,
            "block_size": args.block_size,
            "tokens_per_window": (
                args.batches * args.batch_size * args.block_size
            ),
            "loss": "mean next-token cross entropy",
        },
        "cells": [
            {"layer": layer, "phase_start": phase}
            for layer, phase in cells
        ],
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
        "outputs": {
            "detail_sha256": file_sha256(detail_path),
            "aggregate_sha256": file_sha256(aggregate_path),
            "gradients_sha256": file_sha256(gradients_path),
        },
        "limitations": [
            "Fixed validation-window gradients are not the original stochastic training gradients.",
            "The stateless Muon control omits historical momentum, which is absent from model-only snapshots.",
            "This is a direction diagnostic, not a language-model training result.",
        ],
    }
    metadata_path = args.output / "task_gradient_direction_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "rows": len(rows),
                "decision": decision,
                "promoted_direction": promoted,
                "aggregate": result,
                "metadata": str(metadata_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
