#!/usr/bin/env python3
"""Compare a fast stateless task matcher with the legacy greedy oracle.

The cadence-15 replay showed that stored Givens connectivity is already stale
after 15 updates.  This no-training diagnostic asks whether fresh topology can
instead be recomputed from the current exact Muon direction on every update.
The candidate scans the score-sorted task edges once in compiled C++ and
edge-colors them into 64 perfect matchings.  It is compared with the original
64-pass greedy matcher and with equal-coordinate random connectivity.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.nanogpt.analyze_mlp_activation_update_alignment import (
    load_snapshot,
    model_from_snapshot,
)
from examples.nanogpt.analyze_mlp_muon_chart_staleness import (
    file_sha256,
    git_commit,
    load_registered_probe,
    requested_update,
    write_csv,
)
from examples.nanogpt.analyze_mlp_muon_matched_functional_metric import (
    evaluate_and_collect,
    evaluate_with_updates,
    output_space_metrics,
    task_descent_metrics,
)
from examples.nanogpt.analyze_mlp_muon_matched_givens import (
    diagonal_metric_causal_givens_update,
)
from examples.nanogpt.analyze_mlp_task_gradient_direction import (
    collect_cproj_gradients,
)
from examples.nanogpt.analyze_parameter_trajectory import parse_int_list
from examples.nanogpt.analyze_residual_compatibility import (
    fixed_validation_batches,
)
from examples.nanogpt.fast_task_matching import (
    build_task_edge_coloring,
    fast_muon_matched_permutations,
)
from examples.nanogpt.muon_matched_givens import (
    muon_matched_permutations,
    random_unique_matchings,
)
from examples.nanogpt.parameter_trajectory import (
    SCHEMA_VERSION as TRAJECTORY_SCHEMA_VERSION,
)


CANDIDATES = ("greedy", "single_pass", "random")
WINDOWS = ("fit", "holdout")


def synchronize(device: str) -> None:
    if "cuda" in device:
        torch.cuda.synchronize(torch.device(device))


def timed_call(
    function: Callable[[], Any],
    *,
    device: str,
) -> tuple[Any, float]:
    synchronize(device)
    started = time.perf_counter()
    result = function()
    synchronize(device)
    return result, time.perf_counter() - started


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / max(denominator, 1e-30)


def positive_sum(rows: list[dict[str, Any]], key: str) -> float:
    return sum(max(float(row[key]), 0.0) for row in rows)


def weighted(
    rows: list[dict[str, Any]],
    key: str,
    energy_key: str,
) -> float:
    weights = torch.tensor(
        [float(row[energy_key]) for row in rows],
        dtype=torch.float64,
    )
    values = torch.tensor(
        [float(row[key]) for row in rows],
        dtype=torch.float64,
    )
    return float((weights * values).sum() / weights.sum().clamp_min(1e-30))


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of no values")
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, int((len(ordered) - 1) * quantile + 0.5)),
    )
    return ordered[index]


def aggregate_comparison(
    rows: list[dict[str, Any]],
    finite_rows: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    *,
    minimum_greedy_retention: float,
    minimum_random_enrichment: float,
    maximum_median_seconds_per_layer: float,
    maximum_p95_seconds_per_layer: float,
    maximum_mean_finite_ce_regression: float,
) -> dict[str, Any]:
    candidates: dict[str, Any] = {}
    for candidate in CANDIDATES:
        selected = [row for row in rows if row["candidate"] == candidate]
        fit = [row for row in selected if row["window"] == "fit"]
        holdout = [
            row for row in selected if row["window"] == "holdout"
        ]
        candidates[candidate] = {
            "cells_times_windows": len(selected),
            "requested_update_recovery": weighted(
                fit,
                "requested_update_recovery",
                "requested_update_energy",
            ),
            "output_positive_line_recovery": weighted(
                selected,
                "output_positive_line_recovery",
                "target_output_energy",
            ),
            "output_fixed_scale_recovery": weighted(
                selected,
                "output_fixed_scale_recovery",
                "target_output_energy",
            ),
            "recorded_train_gradient_positive_descent": positive_sum(
                fit,
                "train_gradient_predicted_ce_decrease",
            ),
            "holdout_validation_gradient_positive_descent": positive_sum(
                holdout,
                "validation_gradient_predicted_ce_decrease",
            ),
        }
    greedy = candidates["greedy"]
    fast = candidates["single_pass"]
    random = candidates["random"]
    retention = {
        "requested_update": safe_ratio(
            fast["requested_update_recovery"],
            greedy["requested_update_recovery"],
        ),
        "output_positive_line": safe_ratio(
            fast["output_positive_line_recovery"],
            greedy["output_positive_line_recovery"],
        ),
        "recorded_train_positive_descent": safe_ratio(
            fast["recorded_train_gradient_positive_descent"],
            greedy["recorded_train_gradient_positive_descent"],
        ),
        "holdout_validation_positive_descent": safe_ratio(
            fast["holdout_validation_gradient_positive_descent"],
            greedy["holdout_validation_gradient_positive_descent"],
        ),
    }
    retention["minimum_registered_metric"] = min(retention.values())
    enrichment = {
        "requested_update_over_random": safe_ratio(
            fast["requested_update_recovery"],
            random["requested_update_recovery"],
        ),
        "output_positive_line_over_random": safe_ratio(
            fast["output_positive_line_recovery"],
            random["output_positive_line_recovery"],
        ),
        "recorded_train_positive_descent_over_random": safe_ratio(
            fast["recorded_train_gradient_positive_descent"],
            random["recorded_train_gradient_positive_descent"],
        ),
        "holdout_validation_positive_descent_over_random": safe_ratio(
            fast["holdout_validation_gradient_positive_descent"],
            random["holdout_validation_gradient_positive_descent"],
        ),
    }
    enrichment["minimum_registered_metric"] = min(enrichment.values())

    timing: dict[str, Any] = {}
    for candidate in ("greedy", "single_pass"):
        values = [
            float(row["selection_seconds"])
            for row in selection_rows
            if row["candidate"] == candidate
        ]
        timing[candidate] = {
            "cells": len(values),
            "median_seconds_per_layer": statistics.median(values),
            "p95_seconds_per_layer": percentile(values, 0.95),
            "maximum_seconds_per_layer": max(values),
        }
    timing["single_pass"]["estimated_12_layer_seconds_per_update"] = (
        12 * timing["single_pass"]["median_seconds_per_layer"]
    )
    native_candidate_fractions = [
        float(row["candidate_edge_fraction"])
        for row in selection_rows
        if row["candidate"] == "single_pass"
    ]
    timing["single_pass"]["mean_candidate_edge_fraction"] = (
        statistics.mean(native_candidate_fractions)
    )

    comparisons: list[dict[str, Any]] = []
    for endpoint in sorted({int(row["endpoint"]) for row in finite_rows}):
        for window in WINDOWS:
            losses = {
                str(row["candidate"]): float(row["loss"])
                for row in finite_rows
                if int(row["endpoint"]) == endpoint
                and row["window"] == window
            }
            if set(losses) != {"baseline", *CANDIDATES}:
                raise ValueError(
                    f"incomplete finite-CE candidates at {endpoint}/{window}"
                )
            comparisons.append(
                {
                    "endpoint": endpoint,
                    "window": window,
                    "single_pass_minus_greedy_loss": (
                        losses["single_pass"] - losses["greedy"]
                    ),
                    "single_pass_minus_random_loss": (
                        losses["single_pass"] - losses["random"]
                    ),
                }
            )
    finite = {
        "comparisons": len(comparisons),
        "single_pass_better_than_greedy": sum(
            row["single_pass_minus_greedy_loss"] < 0.0
            for row in comparisons
        ),
        "single_pass_better_than_random": sum(
            row["single_pass_minus_random_loss"] < 0.0
            for row in comparisons
        ),
        "mean_single_pass_minus_greedy_loss": statistics.mean(
            row["single_pass_minus_greedy_loss"] for row in comparisons
        ),
        "mean_single_pass_minus_random_loss": statistics.mean(
            row["single_pass_minus_random_loss"] for row in comparisons
        ),
        "details": comparisons,
    }
    passes = {
        "greedy_retention": (
            retention["minimum_registered_metric"]
            >= minimum_greedy_retention
        ),
        "random_enrichment": (
            enrichment["minimum_registered_metric"]
            >= minimum_random_enrichment
        ),
        "median_selection_latency": (
            timing["single_pass"]["median_seconds_per_layer"]
            <= maximum_median_seconds_per_layer
        ),
        "p95_selection_latency": (
            timing["single_pass"]["p95_seconds_per_layer"]
            <= maximum_p95_seconds_per_layer
        ),
        "finite_ce_regression": (
            finite["mean_single_pass_minus_greedy_loss"]
            <= maximum_mean_finite_ce_regression
        ),
    }
    decision = (
        "QUALIFY_STATELESS_FRESH_MATCHER_PRODUCTION_PREFLIGHT"
        if all(passes.values())
        else "REJECT_STATELESS_FRESH_MATCHER"
    )
    return {
        "candidates": candidates,
        "single_pass_over_greedy": retention,
        "single_pass_over_random": enrichment,
        "timing": timing,
        "finite_step": finite,
        "registered_thresholds": {
            "minimum_greedy_retention": minimum_greedy_retention,
            "minimum_random_enrichment": minimum_random_enrichment,
            "maximum_median_seconds_per_layer": (
                maximum_median_seconds_per_layer
            ),
            "maximum_p95_seconds_per_layer": (
                maximum_p95_seconds_per_layer
            ),
            "maximum_mean_finite_ce_regression": (
                maximum_mean_finite_ce_regression
            ),
        },
        "passes": passes,
        "decision": decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--probe-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--native-cache", required=True, type=Path)
    parser.add_argument("--layers", default="0,3,6,9,11")
    parser.add_argument("--steps", default="0,15,60,75,120,135,180,195")
    parser.add_argument("--stages", type=int, default=64)
    parser.add_argument("--neighbors", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--fit-seed", type=int, default=20260806)
    parser.add_argument("--holdout-seed", type=int, default=20260807)
    parser.add_argument("--matching-seed", type=int, default=271828)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--minimum-greedy-retention", type=float, default=0.85
    )
    parser.add_argument(
        "--minimum-random-enrichment", type=float, default=3.0
    )
    parser.add_argument(
        "--maximum-median-seconds-per-layer",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--maximum-p95-seconds-per-layer",
        type=float,
        default=0.06,
    )
    parser.add_argument(
        "--maximum-mean-finite-ce-regression",
        type=float,
        default=0.00025,
    )
    args = parser.parse_args()
    started = time.time()
    layers = parse_int_list(args.layers)
    steps = parse_int_list(args.steps)
    registered_steps = [0, 15, 60, 75, 120, 135, 180, 195]
    if (
        steps != registered_steps
        or args.stages != 64
        or args.neighbors != 64
    ):
        raise ValueError(
            "registered protocol requires steps 0,15,...,195 and 64/64"
        )
    snapshot_paths = [
        args.snapshot_dir / f"step_{step:06d}.pt" for step in steps
    ]
    probe_paths = [
        args.probe_dir / f"step_{step:06d}.pt" for step in steps
    ]
    missing = [
        str(path)
        for path in (*snapshot_paths, *probe_paths, args.plan)
        if not path.is_file()
    ]
    if missing:
        raise ValueError(f"registered inputs are absent: {missing}")

    native_library, native_path = build_task_edge_coloring(
        args.native_cache
    )
    del native_library
    batches_by_window = {
        "fit": fixed_validation_batches(
            args.data_dir,
            args.batch_size,
            args.block_size + 1,
            args.batches,
            args.fit_seed,
        ),
        "holdout": fixed_validation_batches(
            args.data_dir,
            args.batch_size,
            args.block_size + 1,
            args.batches,
            args.holdout_seed,
        ),
    }
    rows: list[dict[str, Any]] = []
    finite_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    run_identity: str | None = None
    for endpoint in steps:
        snapshot_path = (
            args.snapshot_dir / f"step_{endpoint:06d}.pt"
        )
        payload = load_snapshot(snapshot_path)
        if payload.get("schema_version") != TRAJECTORY_SCHEMA_VERSION:
            raise ValueError(f"unexpected trajectory snapshot: {snapshot_path}")
        probe = load_registered_probe(
            args.probe_dir / f"step_{endpoint:06d}.pt"
        )
        observed_identity = str(probe["run_identity_sha256"])
        if run_identity is None:
            run_identity = observed_identity
        elif observed_identity != run_identity:
            raise ValueError("optimizer probes have inconsistent identities")
        model = model_from_snapshot(payload, args.device)
        baseline_losses: dict[str, float] = {}
        activations: dict[str, dict[int, torch.Tensor]] = {}
        validation_gradients: dict[str, dict[int, torch.Tensor]] = {}
        for window in WINDOWS:
            baseline_losses[window], activations[window] = (
                evaluate_and_collect(
                    model,
                    batches_by_window[window],
                    layers,
                    args.device,
                )
            )
            validation_gradients[window], _loss = collect_cproj_gradients(
                model,
                batches_by_window[window],
                layers,
                args.device,
            )

        updates: dict[str, dict[int, torch.Tensor]] = {
            candidate: {} for candidate in CANDIDATES
        }
        requested_by_layer: dict[int, torch.Tensor] = {}
        train_gradients: dict[int, torch.Tensor] = {}
        for layer in layers:
            parameter = f"transformer.h.{layer}.mlp.c_proj.weight"
            source, gradient, requested = requested_update(
                probe, parameter, args.device
            )
            torch.testing.assert_close(
                payload["parameters"][parameter].to(args.device),
                source,
                rtol=0.0,
                atol=0.0,
            )
            matching_seed = (
                args.matching_seed + 1009 * layer + endpoint
            )
            (greedy_result, greedy_seconds) = timed_call(
                lambda: muon_matched_permutations(
                    source,
                    requested,
                    stages=args.stages,
                    neighbors=args.neighbors,
                    seed=matching_seed,
                ),
                device=args.device,
            )
            greedy_permutations, greedy_diagnostics = greedy_result
            (fast_result, fast_seconds) = timed_call(
                lambda: fast_muon_matched_permutations(
                    source,
                    requested,
                    stages=args.stages,
                    neighbors=args.neighbors,
                    seed=matching_seed,
                    cache_dir=args.native_cache,
                ),
                device=args.device,
            )
            fast_permutations, fast_diagnostics = fast_result
            random_permutations = random_unique_matchings(
                width=source.shape[1],
                stages=args.stages,
                seed=matching_seed,
            )
            selection_rows.extend(
                [
                    {
                        "endpoint": endpoint,
                        "layer": layer,
                        "candidate": "greedy",
                        "selection_seconds": greedy_seconds,
                        "candidate_edge_fraction": statistics.mean(
                            float(row["candidate_edge_fraction"])
                            for row in greedy_diagnostics
                        ),
                    },
                    {
                        "endpoint": endpoint,
                        "layer": layer,
                        "candidate": "single_pass",
                        "selection_seconds": fast_seconds,
                        "candidate_edge_fraction": fast_diagnostics[
                            "candidate_edge_fraction"
                        ],
                    },
                ]
            )
            permutations_by_candidate = {
                "greedy": greedy_permutations,
                "single_pass": fast_permutations,
                "random": random_permutations,
            }
            requested_energy = float(requested.float().square().sum())
            for candidate in CANDIDATES:
                predicted, _fit = diagonal_metric_causal_givens_update(
                    source,
                    requested,
                    stages=args.stages,
                    seed=matching_seed,
                    permutations=permutations_by_candidate[candidate],
                )
                updates[candidate][layer] = predicted
                residual = requested.float() - predicted.float()
                recovery = 1.0 - float(
                    residual.square().sum()
                    / requested.float().square().sum().clamp_min(1e-30)
                )
                for window in WINDOWS:
                    output = output_space_metrics(
                        activations[window][layer].to(args.device),
                        requested,
                        predicted,
                    )
                    train = task_descent_metrics(gradient, predicted)
                    validation = task_descent_metrics(
                        validation_gradients[window][layer],
                        predicted.cpu(),
                    )
                    rows.append(
                        {
                            "endpoint": endpoint,
                            "layer": layer,
                            "window": window,
                            "candidate": candidate,
                            "requested_update_recovery": recovery,
                            "requested_update_energy": requested_energy,
                            "output_fixed_scale_recovery": output[
                                "fixed_scale_recovery"
                            ],
                            "output_positive_line_recovery": output[
                                "positive_step_line_recovery"
                            ],
                            "target_output_energy": output[
                                "target_output_energy"
                            ],
                            "train_gradient_predicted_ce_decrease": train[
                                "predicted_ce_decrease"
                            ],
                            "validation_gradient_predicted_ce_decrease": (
                                validation["predicted_ce_decrease"]
                            ),
                        }
                    )
            requested_by_layer[layer] = requested
            train_gradients[layer] = gradient

        for window in WINDOWS:
            finite_rows.append(
                {
                    "endpoint": endpoint,
                    "window": window,
                    "candidate": "baseline",
                    "loss": baseline_losses[window],
                }
            )
            for candidate in CANDIDATES:
                finite_rows.append(
                    {
                        "endpoint": endpoint,
                        "window": window,
                        "candidate": candidate,
                        "loss": evaluate_with_updates(
                            model,
                            batches_by_window[window],
                            updates[candidate],
                            args.device,
                        ),
                    }
                )
        print(
            json.dumps(
                {
                    "endpoint": endpoint,
                    "greedy_seconds": sum(
                        float(row["selection_seconds"])
                        for row in selection_rows
                        if row["candidate"] == "greedy"
                        and int(row["endpoint"]) == endpoint
                    ),
                    "single_pass_seconds": sum(
                        float(row["selection_seconds"])
                        for row in selection_rows
                        if row["candidate"] == "single_pass"
                        and int(row["endpoint"]) == endpoint
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del model, payload, probe
        if "cuda" in args.device:
            torch.cuda.empty_cache()

    aggregate = aggregate_comparison(
        rows,
        finite_rows,
        selection_rows,
        minimum_greedy_retention=args.minimum_greedy_retention,
        minimum_random_enrichment=args.minimum_random_enrichment,
        maximum_median_seconds_per_layer=(
            args.maximum_median_seconds_per_layer
        ),
        maximum_p95_seconds_per_layer=(
            args.maximum_p95_seconds_per_layer
        ),
        maximum_mean_finite_ce_regression=(
            args.maximum_mean_finite_ce_regression
        ),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    detail_path = args.output / "fast_fresh_matching.csv"
    finite_path = args.output / "fast_fresh_matching_finite_ce.csv"
    timing_path = args.output / "fast_fresh_matching_timing.csv"
    aggregate_path = args.output / "fast_fresh_matching_aggregate.json"
    write_csv(detail_path, rows)
    write_csv(finite_path, finite_rows)
    write_csv(timing_path, selection_rows)
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_mlp_fast_fresh_matching_v1",
        "decision": aggregate["decision"],
        "parameter_updates": 0,
        "causal_protocol": (
            "all connectivity candidates use only the exact current Muon "
            "direction; no future direction selects or fits connectivity"
        ),
        "learned_dense_basis": False,
        "lora_adapter": False,
        "persistent_selected_connectivity": False,
        "input_run_identity_sha256": run_identity,
        "layers": layers,
        "steps": steps,
        "stage_count": args.stages,
        "neighbors": args.neighbors,
        "validation_windows": {
            "fit_seed": args.fit_seed,
            "holdout_seed": args.holdout_seed,
            "batches": args.batches,
            "batch_size": args.batch_size,
            "block_size": args.block_size,
        },
        "native_matcher": {
            "library": str(native_path),
            "library_sha256": file_sha256(native_path),
            "source": str(
                Path(__file__).with_name("csrc")
                / "task_edge_coloring.cpp"
            ),
            "source_sha256": file_sha256(
                Path(__file__).with_name("csrc")
                / "task_edge_coloring.cpp"
            ),
        },
        "plan": {
            "path": str(args.plan),
            "sha256": file_sha256(args.plan),
        },
        "analysis_execution": {
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": str(script),
            "entrypoint_sha256": file_sha256(script),
            "command": sys.argv,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "device": args.device,
        },
        "inputs": {
            "snapshots": [
                {
                    "path": str(path),
                    "sha256": file_sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in snapshot_paths
            ],
            "optimizer_probes": [
                {
                    "path": str(path),
                    "sha256": file_sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in probe_paths
            ],
        },
        "outputs": {
            "detail_sha256": file_sha256(detail_path),
            "finite_ce_sha256": file_sha256(finite_path),
            "timing_sha256": file_sha256(timing_path),
            "aggregate_sha256": file_sha256(aggregate_path),
        },
        "limitations": [
            "The analysis covers five representative c_proj layers and eight exact current-direction endpoints on one dense-Muon trajectory.",
            "A passing no-training matcher still requires integration and a separately polled real-training MFU gate.",
            "The folded materialized weight remains full-sized; this test concerns update geometry, not inference compression.",
        ],
    }
    metadata_path = args.output / "fast_fresh_matching_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "aggregate": aggregate,
                "decision": aggregate["decision"],
                "metadata": str(metadata_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
