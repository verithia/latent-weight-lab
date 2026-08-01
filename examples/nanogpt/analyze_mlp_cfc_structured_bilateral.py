#!/usr/bin/env python3
"""Screen equal-coordinate two-sided sparse-Givens charts for ``mlp.c_fc``.

The terminal hidden88 replay checkpoint supplies a fixed dense Muon update.
Every structured candidate spends exactly the same number of continuous
coordinates as the selected 88-stage expansion-only chart.  Coordinates are
reallocated between expansion/output rotations ``Q_out W`` and input
rotations ``W Q_in``; no parameter or optimizer update is performed.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.nanogpt.analyze_mlp_cfc_exact_current_matcher import (
    _optimizer_and_group_for_parameter,
    _weight_decay_after_rotation,
    build_candidates,
    diagonal_metric_causal_givens_update,
    exact_muon_update,
    file_sha256,
    fixed_batches,
    load_model_and_optimizer,
)
from examples.nanogpt.analyze_mlp_cfc_residual_structure import (
    residual_metrics,
    validate_identity,
    write_csv,
)
from examples.nanogpt.analyze_mlp_cfc_trust_radius import (
    collect_gradient_window,
    repeated_losses,
    summarize,
)
from examples.nanogpt.fast_task_matching import (
    fast_muon_matched_permutations,
)


SCHEMA_VERSION = "nanogpt_mlp_cfc_structured_bilateral_v1"
CONTROL = "output88_input0"
DENSE = "dense_exact"
ALLOCATION_CANDIDATES = (
    "output80_input32",
    "output72_input64",
    "output64_input96",
    "output56_input128",
)
CANDIDATES = (CONTROL, DENSE, *ALLOCATION_CANDIDATES)


def git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def stage_chunks(stages: int) -> list[int]:
    """Split a stage count into matcher-supported causal passes."""
    if stages < 0:
        raise ValueError("stage count cannot be negative")
    chunks = []
    remaining = int(stages)
    while remaining:
        chunk = min(64, remaining)
        chunks.append(chunk)
        remaining -= chunk
    return chunks


def _axis_view(values: torch.Tensor, axis: str) -> torch.Tensor:
    if axis == "output":
        return values.T.contiguous()
    if axis == "input":
        return values.contiguous()
    raise ValueError(f"unknown axis: {axis}")


def _from_axis_view(values: torch.Tensor, axis: str) -> torch.Tensor:
    return values.T.contiguous() if axis == "output" else values.contiguous()


def fit_axis_passes(
    current_weight: torch.Tensor,
    target_update: torch.Tensor,
    *,
    original_weight: torch.Tensor,
    axis: str,
    stages: int,
    neighbors: int,
    seed: int,
    first_selection_direction: torch.Tensor | None,
    native_cache: Path | None,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Fit causal residual Givens passes along one matrix axis."""
    current = current_weight.float()
    records: list[dict[str, Any]] = []
    for pass_index, pass_stages in enumerate(stage_chunks(stages)):
        predicted = current - original_weight.float()
        residual = target_update.float() - predicted
        selection_direction = (
            first_selection_direction.float()
            if pass_index == 0 and first_selection_direction is not None
            else residual
        )
        source_view = _axis_view(current, axis)
        residual_view = _axis_view(residual, axis)
        selection_view = _axis_view(selection_direction, axis)
        pass_seed = int(seed) + pass_index
        permutations, selection = fast_muon_matched_permutations(
            source_view,
            selection_view,
            stages=pass_stages,
            neighbors=neighbors,
            seed=pass_seed,
            cache_dir=native_cache,
        )
        rotation_view, fit = diagonal_metric_causal_givens_update(
            source_view,
            residual_view,
            stages=pass_stages,
            seed=pass_seed,
            permutations=permutations,
        )
        current = _from_axis_view(source_view + rotation_view, axis)
        records.append(
            {
                "axis": axis,
                "pass_index": pass_index,
                "stages": pass_stages,
                "coordinates": int(pass_stages * source_view.shape[1] // 2),
                "selection": selection,
                "fit": fit,
            }
        )
    return current, records


def structured_bilateral_update(
    weight: torch.Tensor,
    dense_update: torch.Tensor,
    polar_descent_per_lr: torch.Tensor,
    *,
    output_stages: int,
    input_stages: int,
    neighbors: int,
    seed: int,
    learning_rate: float,
    weight_decay: float,
    native_cache: Path | None,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Fit output then input actions and apply decoupled weight decay once."""
    if weight.shape != dense_update.shape or weight.shape != polar_descent_per_lr.shape:
        raise ValueError("weight and update tensors must have identical shapes")
    current, output_records = fit_axis_passes(
        weight,
        dense_update,
        original_weight=weight,
        axis="output",
        stages=output_stages,
        neighbors=neighbors,
        seed=seed,
        first_selection_direction=polar_descent_per_lr,
        native_cache=native_cache,
    )
    current, input_records = fit_axis_passes(
        current,
        dense_update,
        original_weight=weight,
        axis="input",
        stages=input_stages,
        neighbors=neighbors,
        seed=seed + 101,
        first_selection_direction=None,
        native_cache=native_cache,
    )
    rotation_update = current - weight.float()
    final_update = _weight_decay_after_rotation(
        weight,
        rotation_update,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    return final_update, [*output_records, *input_records]


def median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def aggregate(
    loss_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    *,
    windows: list[str],
    maximum_replicate_range: float,
    minimum_recovery: float,
    median_recovery: float,
) -> dict[str, Any]:
    summaries: dict[str, dict[str, dict[str, float]]] = {}
    for window in windows:
        summaries[window] = {}
        for candidate in ("baseline", *CANDIDATES):
            summaries[window][candidate] = summarize(
                [
                    float(row["loss"])
                    for row in loss_rows
                    if row["window"] == window
                    and row["candidate"] == candidate
                ]
            )
    stable = all(
        summary["range"] <= maximum_replicate_range
        for values in summaries.values()
        for summary in values.values()
    )
    dense_positive = all(
        values[DENSE]["maximum"] < values[CONTROL]["minimum"]
        for values in summaries.values()
    )
    results: dict[str, dict[str, Any]] = {}
    for candidate in ALLOCATION_CANDIDATES:
        recoveries = {}
        for window, values in summaries.items():
            gap = values[CONTROL]["mean"] - values[DENSE]["mean"]
            recoveries[window] = (
                values[CONTROL]["mean"] - values[candidate]["mean"]
            ) / max(gap, 1e-30)
        minimum = min(recoveries.values())
        med = median(list(recoveries.values()))
        beats_control = all(
            values[candidate]["maximum"] < values[CONTROL]["minimum"]
            for values in summaries.values()
        )
        results[candidate] = {
            "recovery_by_window": recoveries,
            "minimum_recovery": minimum,
            "median_recovery": med,
            "beats_output88_every_window": beats_control,
            "sufficient": all(
                (
                    stable,
                    dense_positive,
                    beats_control,
                    minimum >= minimum_recovery,
                    med >= median_recovery,
                )
            ),
        }
    eligible = [
        candidate
        for candidate in ALLOCATION_CANDIDATES
        if results[candidate]["sufficient"]
    ]
    selected = (
        max(
            eligible,
            key=lambda candidate: (
                results[candidate]["minimum_recovery"],
                results[candidate]["median_recovery"],
            ),
        )
        if eligible
        else None
    )
    coordinate_counts = {
        candidate: sorted(
            {
                int(row["coordinates_per_layer"])
                for row in metric_rows
                if row["candidate"] == candidate
            }
        )
        for candidate in CANDIDATES
    }
    equal_coordinate_budget = all(
        values == coordinate_counts[CONTROL]
        for candidate, values in coordinate_counts.items()
        if candidate != DENSE
    )
    if not dense_positive:
        decision = "DENSE_EXACT_NOT_POSITIVE_CONTROL"
    elif not equal_coordinate_budget:
        decision = "INVALID_UNEQUAL_COORDINATE_BUDGET"
    elif selected is None:
        decision = "REJECT_STRUCTURED_BILATERAL_STAGE_ALLOCATIONS"
    else:
        decision = f"SELECT_{selected.upper()}_FOR_PRODUCTION_INTEGRATION"
    return {
        "decision": decision,
        "selected_candidate": selected,
        "summaries": summaries,
        "candidate_results": results,
        "gates": {
            "numerically_stable": stable,
            "dense_beats_output88_every_window": dense_positive,
            "equal_coordinate_budget": equal_coordinate_budget,
        },
        "coordinate_counts": coordinate_counts,
        "thresholds": {
            "maximum_replicate_range": maximum_replicate_range,
            "minimum_recovery": minimum_recovery,
            "median_recovery": median_recovery,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--native-cache", type=Path)
    args = parser.parse_args()
    started = time.time()
    plan = validate_identity(args.checkpoint, args.config, args.data_dir, args.plan)
    protocol = plan["fixed_protocol"]
    rule = plan["decision_rule"]
    allocations = {
        str(row["candidate"]): (
            int(row["output_stages"]), int(row["input_stages"])
        )
        for row in protocol["stage_allocations"]
    }
    if tuple(allocations) != (CONTROL, *ALLOCATION_CANDIDATES):
        raise ValueError("registered stage allocations do not match the implementation")
    layers = [int(layer) for layer in protocol["layers"]]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fit_batches = fixed_batches(
        args.data_dir,
        "train",
        batch_size=int(protocol["batch_size"]),
        block_size=int(protocol["block_size"]),
        batches=int(protocol["fit_batches"]),
        seed=int(protocol["fit_train_seed"]),
    )
    windows = [
        f"validation_{index + 1}"
        for index in range(len(protocol["validation_seeds"]))
    ]
    validation_batches = {
        window: fixed_batches(
            args.data_dir,
            "val",
            batch_size=int(protocol["batch_size"]),
            block_size=int(protocol["block_size"]),
            batches=int(protocol["validation_batches_per_window"]),
            seed=int(seed),
        )
        for window, seed in zip(
            windows, protocol["validation_seeds"], strict=True
        )
    }
    model, optimizer, checkpoint = load_model_and_optimizer(
        args.checkpoint, config, args.device
    )
    fit_loss, gradients = collect_gradient_window(
        model,
        fit_batches,
        layers,
        device=args.device,
        dtype=torch.bfloat16,
    )
    updates: dict[str, dict[int, torch.Tensor]] = {
        candidate: {} for candidate in CANDIDATES
    }
    metric_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    expected_coordinates: int | None = None
    for layer in layers:
        weight = model.transformer.h[layer].mlp.c_fc.weight
        owner, group = _optimizer_and_group_for_parameter(optimizer, weight)
        buffer = owner.state[weight].get("momentum_buffer")
        if buffer is None:
            raise RuntimeError(f"missing c_fc momentum at layer {layer}")
        dense_update, descent, optimizer_diagnostics = exact_muon_update(
            weight.detach(),
            gradients[layer].to(weight.device),
            buffer,
            learning_rate=float(group["lr"]),
            momentum=float(group["momentum"]),
            weight_decay=float(group["weight_decay"]),
            ns_steps=int(group["ns_steps"]),
        )
        polar_descent = (
            descent
            + float(group["weight_decay"]) * weight.detach().float()
        )
        matched, control_selections = build_candidates(
            weight.detach(),
            dense_update,
            polar_descent,
            parent_stages=64,
            residual_stages=24,
            neighbors=int(protocol["matching_neighbors"]),
            seed=int(protocol["matching_seed"]) + layer * 1009,
            learning_rate=float(group["lr"]),
            weight_decay=float(group["weight_decay"]),
            native_cache=args.native_cache,
        )
        updates[CONTROL][layer] = matched["fresh_expansion88"].cpu()
        updates[DENSE][layer] = dense_update.float().cpu()
        selection_rows.extend(
            {"layer": layer, "candidate": CONTROL, **row}
            for row in control_selections
            if row["selection"] != "random_expansion88"
        )
        for candidate in ALLOCATION_CANDIDATES:
            output_stages, input_stages = allocations[candidate]
            update, records = structured_bilateral_update(
                weight.detach(),
                dense_update,
                polar_descent,
                output_stages=output_stages,
                input_stages=input_stages,
                neighbors=int(protocol["matching_neighbors"]),
                seed=int(protocol["matching_seed"]) + layer * 1009,
                learning_rate=float(group["lr"]),
                weight_decay=float(group["weight_decay"]),
                native_cache=args.native_cache,
            )
            updates[candidate][layer] = update.cpu()
            selection_rows.extend(
                {"layer": layer, "candidate": candidate, **row}
                for row in records
            )
        output_width = int(weight.shape[0])
        input_width = int(weight.shape[1])
        layer_coordinates = {
            candidate: (
                output_stages * (output_width // 2)
                + input_stages * (input_width // 2)
            )
            for candidate, (output_stages, input_stages) in allocations.items()
        }
        if len(set(layer_coordinates.values())) != 1:
            raise RuntimeError("stage allocations are not equal-coordinate")
        expected_coordinates = next(iter(layer_coordinates.values()))
        for candidate in CANDIDATES:
            candidate_update = updates[candidate][layer].to(weight.device)
            coordinates = (
                int(weight.numel())
                if candidate == DENSE
                else layer_coordinates[candidate]
            )
            metric_rows.append(
                {
                    "layer": layer,
                    "candidate": candidate,
                    "coordinates_per_layer": coordinates,
                    "coordinate_fraction": coordinates / weight.numel(),
                    **residual_metrics(dense_update, candidate_update),
                    **{
                        f"optimizer_{key}": value
                        for key, value in optimizer_diagnostics.items()
                    },
                }
            )

    repeats = int(protocol["evaluation_repeats"])
    loss_rows: list[dict[str, Any]] = []
    for window, batches in validation_batches.items():
        baseline = repeated_losses(
            model,
            batches,
            None,
            repeats=repeats,
            device=args.device,
            dtype=torch.float32,
        )
        for repeat, loss in enumerate(baseline):
            loss_rows.append(
                {
                    "window": window,
                    "candidate": "baseline",
                    "repeat": repeat,
                    "loss": loss,
                }
            )
        for candidate in CANDIDATES:
            values = repeated_losses(
                model,
                batches,
                updates[candidate],
                repeats=repeats,
                device=args.device,
                dtype=torch.float32,
            )
            for repeat, loss in enumerate(values):
                loss_rows.append(
                    {
                        "window": window,
                        "candidate": candidate,
                        "repeat": repeat,
                        "loss": loss,
                    }
                )
    result = aggregate(
        loss_rows,
        metric_rows,
        windows=windows,
        maximum_replicate_range=float(rule["maximum_replicate_range"]),
        minimum_recovery=float(rule["minimum_recovery"]),
        median_recovery=float(rule["median_recovery"]),
    )
    result.update(
        {
            "fit_gradient_loss_bfloat16": fit_loss,
            "parameter_updates": 0,
            "equal_coordinates_per_layer": expected_coordinates,
        }
    )
    args.output.mkdir(parents=True, exist_ok=True)
    losses_path = args.output / "cfc_structured_bilateral_losses.csv"
    metrics_path = args.output / "cfc_structured_bilateral_metrics.csv"
    selections_path = args.output / "cfc_structured_bilateral_selections.json"
    aggregate_path = args.output / "cfc_structured_bilateral_aggregate.json"
    write_csv(losses_path, loss_rows)
    write_csv(metrics_path, metric_rows)
    selections_path.write_text(
        json.dumps(selection_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    aggregate_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "decision": result["decision"],
        "parameter_updates": 0,
        "checkpoint_next_iter": int(checkpoint["next_iter"]),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "config_sha256": file_sha256(args.config),
        "dataset_manifest_sha256": file_sha256(
            args.data_dir / "manifest.json"
        ),
        "plan_sha256": file_sha256(args.plan),
        "analysis_execution": {
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": str(Path(__file__).resolve()),
            "entrypoint_sha256": file_sha256(Path(__file__).resolve()),
            "command": sys.argv,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "device": args.device,
            "direct_foreground_polling": True,
            "watchdog": False,
            "callback": False,
        },
        "protocol": protocol,
        "outputs": {
            "losses_sha256": file_sha256(losses_path),
            "metrics_sha256": file_sha256(metrics_path),
            "selections_sha256": file_sha256(selections_path),
            "aggregate_sha256": file_sha256(aggregate_path),
        },
        "limitations": plan["limitations"],
    }
    metadata_path = args.output / "cfc_structured_bilateral_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "selected_candidate": result["selected_candidate"],
                "aggregate": str(aggregate_path),
                "metadata": str(metadata_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
