#!/usr/bin/env python3
"""Test an exact-base affine BlockFHT chart on the dense attention path.

This is a zero-update upper bound.  Coordinates are fitted on one frozen
terminal-dense functional metric and evaluated without refitting on disjoint
validation batches.  Dense states, local trajectory chords, the affine span
of the discovery path, and exact Muon directions must all generalize before
an affine-delta training implementation can be considered.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_attention_paper_activation_oracle import (
    AttentionFunctionalMetric,
    all_finite,
    file_sha256,
    terminal_attention_metrics,
)
from examples.nanogpt.analyze_mlp_cproj_paper_activation_oracle import (
    cgls,
    explained_energy,
)
from examples.nanogpt.analyze_residual_compatibility import fixed_validation_batches
from examples.nanogpt.train import require_block_fht_native_extension
from latent_weight_lab.block_fht import block_fht_grad_latent, block_fht_slice


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_SCHEMA = "mai_124m_attention_affine_delta_path_oracle_plan_v1"
RESULT_SCHEMA = "mai_124m_attention_affine_delta_path_oracle_result_v1"


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def trajectory_inventory(directory: Path) -> tuple[list[dict[str, Any]], str]:
    items = []
    for path in sorted(directory.glob("step_*.pt")):
        items.append(
            {
                "name": path.name,
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return items, canonical_sha256(items)


def batch_digest(batches: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for batch in batches:
        value = batch.detach().cpu().contiguous()
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def cosine_lr(iter_num: int, config: dict[str, Any]) -> float:
    warmup = int(config["warmup_iters"])
    decay = int(config["lr_decay_iters"])
    maximum = float(config["learning_rate"])
    minimum = float(config["min_lr"])
    if iter_num < warmup:
        return maximum * (iter_num + 1) / (warmup + 1)
    if iter_num > decay:
        return minimum
    ratio = (iter_num - warmup) / (decay - warmup)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return minimum + coefficient * (maximum - minimum)


def cumulative_lr_clock(
    steps: list[int], config: dict[str, Any]
) -> dict[int, float]:
    maximum = max(steps)
    cumulative = [0.0]
    for update in range(maximum):
        cumulative.append(cumulative[-1] + cosine_lr(update, config))
    denominator = cumulative[-1]
    return {step: cumulative[step] / denominator for step in steps}


def solve_span_coefficients(
    basis_outputs: torch.Tensor,
    target: torch.Tensor,
    relative_cutoff: float,
) -> tuple[torch.Tensor, int]:
    """Solve a small least-squares problem through the basis Gram matrix."""

    gram = basis_outputs.T @ basis_outputs
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    maximum = eigenvalues.amax().clamp_min(torch.finfo(eigenvalues.dtype).tiny)
    keep = eigenvalues > maximum * relative_cutoff
    if not bool(keep.any()):
        return torch.zeros(
            basis_outputs.shape[1],
            dtype=basis_outputs.dtype,
            device=basis_outputs.device,
        ), 0
    selected_values = eigenvalues[keep]
    selected_vectors = eigenvectors[:, keep]
    right = basis_outputs.T @ target
    coefficients = selected_vectors @ (
        (selected_vectors.T @ right) / selected_values
    )
    return coefficients, int(keep.sum())


def weighted(rows: list[dict[str, Any]], field: str, energy: str) -> float:
    denominator = sum(float(row[energy]) for row in rows)
    if denominator <= 0.0:
        return 0.0
    return sum(float(row[field]) * float(row[energy]) for row in rows) / denominator


def minimum_layer_recovery(
    rows: list[dict[str, Any]], field: str, energy: str
) -> float:
    values = []
    for layer in sorted({int(row["layer"]) for row in rows}):
        selected = [row for row in rows if int(row["layer"]) == layer]
        values.append(weighted(selected, field, energy))
    return min(values)


def classify_summary(
    summary: dict[str, float], thresholds: dict[str, float]
) -> tuple[bool, dict[str, bool]]:
    aggregate = float(thresholds["aggregate_recovery_minimum"])
    layer = float(thresholds["minimum_layer_recovery_minimum"])
    checks = {
        "state_image": summary["heldout_state_eval_recovery"] >= aggregate,
        "state_image_every_layer": summary["minimum_layer_state_eval_recovery"]
        >= layer,
        "local_chord": summary["heldout_chord_eval_recovery"] >= aggregate,
        "local_chord_every_layer": summary[
            "minimum_layer_chord_eval_recovery"
        ]
        >= layer,
        "discovery_affine_span": summary[
            "heldout_discovery_span_eval_recovery"
        ]
        >= aggregate,
        "discovery_affine_span_every_layer": summary[
            "minimum_layer_discovery_span_eval_recovery"
        ]
        >= layer,
        "exact_muon": summary["heldout_muon_eval_recovery"] >= aggregate,
        "exact_muon_every_layer": summary[
            "minimum_layer_muon_eval_recovery"
        ]
        >= layer,
    }
    return all(checks.values()), checks


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_plan(plan: dict[str, Any], args: argparse.Namespace) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("affine-delta path plan schema mismatch")
    if int(plan["protocol"]["parameter_updates"]) != 0:
        raise ValueError("this oracle must make zero parameter updates")
    if any(bool(value) for value in plan["authorization"].values()):
        raise ValueError("the preregistration must not pre-authorize implementation")
    identity = plan["identity"]
    expected = {
        "trajectory_directory": args.trajectory_dir,
        "probe_directory": args.probe_dir,
        "terminal_checkpoint": args.terminal_checkpoint,
    }
    for field, observed in expected.items():
        if Path(identity[field]) != observed:
            raise ValueError(f"{field} differs from the immutable plan")
    if file_sha256(Path(__file__)) != identity["entrypoint_sha256"]:
        raise ValueError("analyzer hash differs from the immutable plan")
    if file_sha256(args.terminal_checkpoint) != identity["terminal_checkpoint_sha256"]:
        raise ValueError("terminal checkpoint hash mismatch")
    if file_sha256(args.data_dir / "manifest.json") != identity[
        "dataset_manifest_sha256"
    ]:
        raise ValueError("dataset manifest hash mismatch")
    for field in (
        "parent_activation_result",
        "mlp_affine_delta_control",
        "product_fht_residual_control",
    ):
        path = REPO_ROOT / identity[field]
        if file_sha256(path) != identity[f"{field}_sha256"]:
            raise ValueError(f"{field} hash mismatch")


def target_tensor(
    tensor: torch.Tensor, target: str, n_embd: int
) -> torch.Tensor:
    value = tensor.float()
    if target == "v":
        return value[2 * n_embd :]
    if target == "cproj":
        return value
    raise ValueError(f"unsupported target: {target}")


def main() -> None:
    parser = argparse.ArgumentParser()
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
    require_block_fht_native_extension(True)
    started = time.time()
    protocol = plan["protocol"]
    config = json.loads((REPO_ROOT / plan["identity"]["dense_config"]).read_text())
    layers = [int(value) for value in protocol["layers"]]
    steps = [int(value) for value in protocol["trajectory_steps"]]
    discovery_max = int(protocol["discovery_max_step"])
    heldout_min = int(protocol["heldout_min_step"])
    probe_steps = [int(value) for value in protocol["probe_steps"]]
    heldout_probes = {int(value) for value in protocol["heldout_probe_steps"]}

    inventory, inventory_sha = trajectory_inventory(args.trajectory_dir)
    identity = plan["identity"]
    if len(inventory) != int(identity["trajectory_file_count"]):
        raise ValueError("trajectory file count mismatch")
    if sum(int(item["size"]) for item in inventory) != int(
        identity["trajectory_total_bytes"]
    ):
        raise ValueError("trajectory byte count mismatch")
    if inventory_sha != identity["trajectory_inventory_sha256"]:
        raise ValueError("trajectory inventory hash mismatch")

    snapshots: dict[int, dict[str, torch.Tensor]] = {}
    run_identity = None
    for step in steps:
        payload = torch.load(
            args.trajectory_dir / f"step_{step:06d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        if int(payload["step"]) != step:
            raise ValueError("trajectory step mismatch")
        if run_identity is None:
            run_identity = payload["run_identity_sha256"]
        elif payload["run_identity_sha256"] != run_identity:
            raise ValueError("trajectory snapshots do not share one run identity")
        snapshots[step] = payload["parameters"]
    if run_identity != identity["probe_run_identity_sha256"]:
        raise ValueError("trajectory run identity mismatch")

    probes = {}
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
        raise ValueError("fit and evaluation functional batches are identical")
    print("collecting disjoint frozen terminal-dense metrics", flush=True)
    fit_inputs = terminal_attention_metrics(
        args.terminal_checkpoint, fit_batches, layers, args.device
    )
    eval_inputs = terminal_attention_metrics(
        args.terminal_checkpoint, eval_batches, layers, args.device
    )

    base_seed = int(protocol["block_fht_seed"])
    latent_init_std = float(config.get("block_fht_latent_init_std", 0.02))
    clock = cumulative_lr_clock(steps, config)
    rows: list[dict[str, Any]] = []
    temporal: dict[str, list[dict[str, Any]]] = {target: [] for target in protocol["targets"]}

    for layer in layers:
        print(f"analyzing layer {layer}", flush=True)
        for target, spec in protocol["targets"].items():
            parameter_name = f"transformer.h.{layer}.{spec['parameter']}"
            n_embd = int(config["n_embd"])
            initial = target_tensor(
                snapshots[steps[0]][parameter_name], target, n_embd
            ).to(args.device)
            size = initial.numel()
            latent_dim = max(1, round(size * float(protocol["latent_ratio"])))
            seed = base_seed + int(spec["seed_stride"]) * layer + int(
                spec["seed_offset"]
            )
            weight_scale = float(spec["target_std"]) / latent_init_std
            template = torch.zeros(latent_dim, device=args.device)

            def apply_a(coordinate: torch.Tensor) -> torch.Tensor:
                return (
                    block_fht_slice(
                        coordinate,
                        size,
                        int(protocol["block_fht_layers"]),
                        seed,
                        0,
                        size,
                    )
                    * weight_scale
                ).view_as(initial)

            def adjoint_a(weight: torch.Tensor) -> torch.Tensor:
                return block_fht_grad_latent(
                    template,
                    (weight.reshape(-1) * weight_scale).contiguous(),
                    size,
                    int(protocol["block_fht_layers"]),
                    seed,
                    0,
                    size,
                )

            fit_metric = AttentionFunctionalMetric(
                target=target,
                cproj_inputs=fit_inputs[layer]["cproj_inputs"],
                value_sources=fit_inputs[layer]["value_sources"],
                output_weight=fit_inputs[layer]["output_weight"],
            )
            eval_metric = AttentionFunctionalMetric(
                target=target,
                cproj_inputs=eval_inputs[layer]["cproj_inputs"],
                value_sources=eval_inputs[layer]["value_sources"],
                output_weight=eval_inputs[layer]["output_weight"],
            )

            def fit_coordinate(weight: torch.Tensor):
                target_output = fit_metric.apply(weight)

                def apply(coordinate: torch.Tensor) -> torch.Tensor:
                    return fit_metric.apply(apply_a(coordinate))

                def adjoint(output: torch.Tensor) -> torch.Tensor:
                    return adjoint_a(fit_metric.adjoint(output))

                return cgls(
                    apply,
                    adjoint,
                    target_output,
                    template,
                    int(protocol["cgls_iterations"]),
                )

            def metrics_for(
                weight: torch.Tensor, coordinate: torch.Tensor, fit_prediction: torch.Tensor
            ) -> dict[str, float]:
                prediction_weight = apply_a(coordinate)
                fit_target = fit_metric.apply(weight)
                eval_target = eval_metric.apply(weight)
                fit_recovery, fit_energy = explained_energy(
                    fit_target, fit_prediction
                )
                eval_recovery, eval_energy = explained_energy(
                    eval_target, eval_metric.apply(prediction_weight)
                )
                euclidean_recovery, euclidean_energy = explained_energy(
                    weight, prediction_weight
                )
                return {
                    "fit_recovery": fit_recovery,
                    "fit_energy": fit_energy,
                    "eval_recovery": eval_recovery,
                    "eval_energy": eval_energy,
                    "euclidean_recovery": euclidean_recovery,
                    "euclidean_energy": euclidean_energy,
                }

            coordinates = {steps[0]: template.clone()}
            state_weights = {steps[0]: torch.zeros_like(initial)}
            for step in steps[1:]:
                current = target_tensor(
                    snapshots[step][parameter_name], target, n_embd
                ).to(args.device)
                displacement = current - initial
                coordinate, fit_prediction, iterations = fit_coordinate(displacement)
                coordinates[step] = coordinate
                state_weights[step] = displacement
                values = metrics_for(displacement, coordinate, fit_prediction)
                rows.append(
                    {
                        "kind": "state",
                        "target": target,
                        "layer": layer,
                        "step_start": 0,
                        "step_end": step,
                        "split": "discovery" if step <= discovery_max else "heldout",
                        "seed": seed,
                        "latent_dim": latent_dim,
                        "iterations": iterations,
                        **values,
                    }
                )

            for start, end in zip(steps[:-1], steps[1:]):
                if start <= discovery_max < end:
                    continue
                chord = state_weights[end] - state_weights[start]
                coordinate, fit_prediction, iterations = fit_coordinate(chord)
                values = metrics_for(chord, coordinate, fit_prediction)
                rows.append(
                    {
                        "kind": "chord",
                        "target": target,
                        "layer": layer,
                        "step_start": start,
                        "step_end": end,
                        "split": "discovery" if end <= discovery_max else "heldout",
                        "seed": seed,
                        "latent_dim": latent_dim,
                        "iterations": iterations,
                        **values,
                    }
                )

            discovery_steps = [
                step for step in steps if 0 < step <= discovery_max
            ]
            coordinate_basis = torch.stack(
                [coordinates[step] for step in discovery_steps], dim=1
            )
            output_basis = torch.stack(
                [
                    fit_metric.apply(apply_a(coordinates[step])).reshape(-1)
                    for step in discovery_steps
                ],
                dim=1,
            )
            for step in [value for value in steps if value >= heldout_min]:
                fit_target = fit_metric.apply(state_weights[step]).reshape(-1)
                coefficients, rank = solve_span_coefficients(
                    output_basis,
                    fit_target,
                    float(protocol["span_relative_cutoff"]),
                )
                coordinate = coordinate_basis @ coefficients
                eval_target = eval_metric.apply(state_weights[step])
                eval_recovery, eval_energy = explained_energy(
                    eval_target, eval_metric.apply(apply_a(coordinate))
                )
                rows.append(
                    {
                        "kind": "discovery_span",
                        "target": target,
                        "layer": layer,
                        "step_start": 0,
                        "step_end": step,
                        "split": "heldout",
                        "seed": seed,
                        "latent_dim": latent_dim,
                        "span_rank": rank,
                        "eval_recovery": eval_recovery,
                        "eval_energy": eval_energy,
                    }
                )

            clock_steps = [step for step in steps if step <= discovery_max]
            design = torch.tensor(
                [[1.0, clock[step]] for step in clock_steps],
                device=args.device,
                dtype=template.dtype,
            )
            coordinate_values = torch.stack(
                [coordinates[step] for step in clock_steps], dim=0
            )
            clock_fit = torch.linalg.lstsq(design, coordinate_values).solution
            for step in [value for value in steps if value >= heldout_min]:
                coordinate = (
                    torch.tensor(
                        [1.0, clock[step]],
                        device=args.device,
                        dtype=template.dtype,
                    )
                    @ clock_fit
                )
                eval_target = eval_metric.apply(state_weights[step])
                recovery, energy = explained_energy(
                    eval_target, eval_metric.apply(apply_a(coordinate))
                )
                rows.append(
                    {
                        "kind": "lr_clock_line",
                        "target": target,
                        "layer": layer,
                        "step_start": 0,
                        "step_end": step,
                        "split": "heldout",
                        "seed": seed,
                        "latent_dim": latent_dim,
                        "eval_recovery": recovery,
                        "eval_energy": energy,
                    }
                )

            centered = coordinate_values - coordinate_values.mean(dim=0)
            singular = torch.linalg.svdvals(centered)
            energy = singular.double().square()
            fraction = energy.cumsum(0) / energy.sum().clamp_min(1e-30)
            rank_at = lambda threshold: int(
                torch.searchsorted(
                    fraction,
                    torch.tensor(
                        threshold,
                        device=fraction.device,
                        dtype=fraction.dtype,
                    ),
                ).item()
                + 1
            )
            temporal[target].append(
                {
                    "layer": layer,
                    "discovery_coordinate_rank_90": rank_at(0.90),
                    "discovery_coordinate_rank_95": rank_at(0.95),
                    "discovery_coordinate_rank_99": rank_at(0.99),
                }
            )

            for step in probe_steps:
                parameter = probes[step]["parameters"][parameter_name]
                direction = target_tensor(
                    parameter["applied_direction_per_lr"], target, n_embd
                ).to(args.device)
                coordinate, fit_prediction, iterations = fit_coordinate(direction)
                values = metrics_for(direction, coordinate, fit_prediction)
                rows.append(
                    {
                        "kind": "muon_direction",
                        "target": target,
                        "layer": layer,
                        "step_start": step,
                        "step_end": step,
                        "split": "heldout" if step in heldout_probes else "discovery",
                        "seed": seed,
                        "latent_dim": latent_dim,
                        "iterations": iterations,
                        **values,
                    }
                )

    summaries: dict[str, Any] = {}
    thresholds = plan["decision_rule"]["thresholds"]
    for target in protocol["targets"]:
        def selected(kind: str) -> list[dict[str, Any]]:
            return [
                row
                for row in rows
                if row["target"] == target
                and row["kind"] == kind
                and row["split"] == "heldout"
            ]

        state = selected("state")
        chord = selected("chord")
        span = selected("discovery_span")
        muon = selected("muon_direction")
        clock_rows = selected("lr_clock_line")
        summary = {
            "heldout_state_fit_recovery": weighted(
                state, "fit_recovery", "fit_energy"
            ),
            "heldout_state_eval_recovery": weighted(
                state, "eval_recovery", "eval_energy"
            ),
            "minimum_layer_state_eval_recovery": minimum_layer_recovery(
                state, "eval_recovery", "eval_energy"
            ),
            "heldout_state_euclidean_recovery": weighted(
                state, "euclidean_recovery", "euclidean_energy"
            ),
            "heldout_chord_eval_recovery": weighted(
                chord, "eval_recovery", "eval_energy"
            ),
            "minimum_layer_chord_eval_recovery": minimum_layer_recovery(
                chord, "eval_recovery", "eval_energy"
            ),
            "heldout_discovery_span_eval_recovery": weighted(
                span, "eval_recovery", "eval_energy"
            ),
            "minimum_layer_discovery_span_eval_recovery": minimum_layer_recovery(
                span, "eval_recovery", "eval_energy"
            ),
            "heldout_muon_eval_recovery": weighted(
                muon, "eval_recovery", "eval_energy"
            ),
            "minimum_layer_muon_eval_recovery": minimum_layer_recovery(
                muon, "eval_recovery", "eval_energy"
            ),
            "heldout_lr_clock_line_eval_recovery_diagnostic": weighted(
                clock_rows, "eval_recovery", "eval_energy"
            ),
            "mean_discovery_coordinate_rank_95": sum(
                float(row["discovery_coordinate_rank_95"])
                for row in temporal[target]
            )
            / len(temporal[target]),
        }
        passed, checks = classify_summary(summary, thresholds)
        summaries[target] = {**summary, "checks": checks, "passed": passed}

    args.output_dir.mkdir(parents=True)
    rows_path = args.output_dir / "attention_affine_delta_path_cells.csv"
    write_rows(rows_path, rows)
    passed_targets = [
        target for target, summary in summaries.items() if summary["passed"]
    ]
    all_targets_pass = len(passed_targets) == len(protocol["targets"])
    result = {
        "schema_version": RESULT_SCHEMA,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "classification": (
            "ATTENTION_AFFINE_DELTA_PATH_GATE_PASS"
            if all_targets_pass
            else "ATTENTION_AFFINE_DELTA_PATH_GATE_REJECT"
        ),
        "execution": {
            "host": "PRO6",
            "device": args.device,
            "git_commit": git_commit(),
            "entrypoint": "examples.nanogpt.analyze_attention_affine_delta_path_oracle",
            "parameter_updates": 0,
            "elapsed_seconds": time.time() - started,
        },
        "identity": {
            "plan_path": str(args.plan),
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
        "temporal_coordinate_diagnostics": temporal,
        "decision": {
            "passed_targets": passed_targets,
            "exact_config_mfu_preflight_authorized": all_targets_pass,
            "model_implementation_authorized": all_targets_pass,
            "language_model_training_authorized": False,
            "larger_rung_authorized": False,
        },
        "artifacts": {
            "cells_csv": str(rows_path),
            "cells_csv_sha256": file_sha256(rows_path),
        },
        "all_reported_values_finite": all_finite(summaries),
        "limitations": [
            "Coordinates and local chords are optimistic oracle fits to dense answers.",
            "The early affine-span coefficients see each held-out target, but the span itself is frozen from discovery states.",
            "The frozen terminal metrics isolate decoder geometry from residual-stream coevolution.",
            "A pass authorizes only implementation plus an exact-config MFU gate, not training or a larger rung.",
        ],
    }
    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
