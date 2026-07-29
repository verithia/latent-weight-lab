#!/usr/bin/env python3
"""Measure how quickly a Muon-matched c_proj connectivity chart becomes stale.

The input must be one coherent dense-Muon replay containing all-parameter
snapshots and exact pre-step optimizer probes at the same 15-update cadence.
For each registered 60-update phase start, the diagnostic selects a
64-stage hidden-side Givens connectivity from only the exact current Muon
direction.  At ages 0, 15, 30, 45, and 60 it refits only the diagonal Givens
angles to the exact future Muon direction and compares that aged connectivity
with a freshly selected connectivity at the same future step.

Refitting angles makes this a connectivity-capacity test rather than a
frozen-update test.  Future probes are used only to score/refit the already
selected connectivity; they never affect the phase-start connectivity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
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
    load_snapshot,
    model_from_snapshot,
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
from examples.nanogpt.analyze_mlp_optimizer_state_direction import (
    reconstruct_directions,
)
from examples.nanogpt.analyze_mlp_task_gradient_direction import (
    collect_cproj_gradients,
)
from examples.nanogpt.analyze_parameter_trajectory import parse_int_list
from examples.nanogpt.analyze_residual_compatibility import (
    fixed_validation_batches,
)
from examples.nanogpt.muon_matched_givens import (
    muon_matched_permutations,
)
from examples.nanogpt.parameter_trajectory import (
    OPTIMIZER_PROBE_SCHEMA_VERSION,
    SCHEMA_VERSION as TRAJECTORY_SCHEMA_VERSION,
)


CANDIDATES = ("fresh", "aged")
WINDOWS = ("fit", "holdout")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def weighted(
    rows: list[dict[str, Any]],
    value: str,
    energy: str,
) -> float:
    weights = torch.tensor(
        [float(row[energy]) for row in rows],
        dtype=torch.float64,
    )
    values = torch.tensor(
        [float(row[value]) for row in rows],
        dtype=torch.float64,
    )
    return float((weights * values).sum() / weights.sum().clamp_min(1e-30))


def positive_sum(rows: list[dict[str, Any]], value: str) -> float:
    return sum(max(float(row[value]), 0.0) for row in rows)


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / max(denominator, 1e-30)


def aggregate_retention(
    rows: list[dict[str, Any]],
    finite_rows: list[dict[str, Any]],
    *,
    stable_retention: float,
    stale_retention: float,
) -> dict[str, Any]:
    """Aggregate aged/fresh functional retention and choose a cadence gate."""
    ages = sorted({int(row["age_updates"]) for row in rows})
    by_age: dict[str, Any] = {}
    for age in ages:
        candidates: dict[str, Any] = {}
        for candidate in CANDIDATES:
            selected = [
                row
                for row in rows
                if int(row["age_updates"]) == age
                and row["candidate"] == candidate
            ]
            fit = [row for row in selected if row["window"] == "fit"]
            holdout = [
                row for row in selected if row["window"] == "holdout"
            ]
            candidates[candidate] = {
                "cells_times_windows": len(selected),
                "phase_anchor_count": len(
                    {int(row["phase_anchor"]) for row in selected}
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
                "recorded_train_gradient_signed_descent": sum(
                    float(row["train_gradient_predicted_ce_decrease"])
                    for row in fit
                ),
                "holdout_validation_gradient_positive_descent": (
                    positive_sum(
                        holdout,
                        "validation_gradient_predicted_ce_decrease",
                    )
                ),
                "holdout_validation_gradient_signed_descent": sum(
                    float(
                        row[
                            "validation_gradient_predicted_ce_decrease"
                        ]
                    )
                    for row in holdout
                ),
            }
        fresh = candidates["fresh"]
        aged = candidates["aged"]
        retention = {
            "output_positive_line": safe_ratio(
                aged["output_positive_line_recovery"],
                fresh["output_positive_line_recovery"],
            ),
            "recorded_train_positive_descent": safe_ratio(
                aged["recorded_train_gradient_positive_descent"],
                fresh["recorded_train_gradient_positive_descent"],
            ),
            "holdout_validation_positive_descent": safe_ratio(
                aged["holdout_validation_gradient_positive_descent"],
                fresh["holdout_validation_gradient_positive_descent"],
            ),
        }
        retention["minimum_registered_metric"] = min(retention.values())
        finite = [
            row for row in finite_rows if int(row["age_updates"]) == age
        ]
        finite_comparisons: list[dict[str, Any]] = []
        for phase_anchor in sorted(
            {int(row["phase_anchor"]) for row in finite}
        ):
            for window in WINDOWS:
                losses = {
                    str(row["candidate"]): float(row["loss"])
                    for row in finite
                    if int(row["phase_anchor"]) == phase_anchor
                    and row["window"] == window
                }
                if set(losses) != {"baseline", "fresh", "aged"}:
                    continue
                finite_comparisons.append(
                    {
                        "phase_anchor": phase_anchor,
                        "window": window,
                        "aged_minus_fresh_loss": (
                            losses["aged"] - losses["fresh"]
                        ),
                    }
                )
        by_age[str(age)] = {
            "candidates": candidates,
            "retention": retention,
            "finite_step": {
                "comparisons": len(finite_comparisons),
                "aged_better_than_fresh": sum(
                    item["aged_minus_fresh_loss"] < 0.0
                    for item in finite_comparisons
                ),
                "mean_aged_minus_fresh_loss": (
                    sum(
                        item["aged_minus_fresh_loss"]
                        for item in finite_comparisons
                    )
                    / max(len(finite_comparisons), 1)
                ),
                "details": finite_comparisons,
            },
        }

    required = {"15", "30", "45", "60"}
    if not required.issubset(by_age):
        raise ValueError("registered ages 15,30,45,60 are required")
    retention15 = by_age["15"]["retention"]["minimum_registered_metric"]
    retention30 = by_age["30"]["retention"]["minimum_registered_metric"]
    retention45 = by_age["45"]["retention"]["minimum_registered_metric"]
    retention60 = by_age["60"]["retention"]["minimum_registered_metric"]
    if retention60 >= stable_retention:
        decision = "R60_CONNECTIVITY_NOT_STALE"
    elif (
        retention30 >= stable_retention
        and min(retention45, retention60) < stale_retention
    ):
        decision = "QUALIFY_R30_PERFORMANCE_PREFLIGHT"
    elif (
        retention15 >= stable_retention
        and retention30 < stale_retention
    ):
        decision = "QUALIFY_R15_PERFORMANCE_PREFLIGHT"
    else:
        decision = "MIXED_CADENCE_SIGNAL_NO_TRAINING_PROMOTION"
    return {
        "by_age_updates": by_age,
        "decision": decision,
        "decision_rule": {
            "stable_retention": stable_retention,
            "stale_retention": stale_retention,
            "r60_not_stale": (
                "minimum of output-line, recorded-train-descent, and "
                "holdout-validation-descent retention at age 60 is at "
                "least stable_retention"
            ),
            "qualify_r30": (
                "age-30 minimum retention is at least stable_retention "
                "and age-45 or age-60 minimum retention is below "
                "stale_retention"
            ),
            "qualify_r15": (
                "age-15 minimum retention is at least stable_retention "
                "and age-30 minimum retention is below stale_retention"
            ),
            "otherwise": "mixed signal; no shorter-refresh training",
        },
    }


def load_registered_probe(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != OPTIMIZER_PROBE_SCHEMA_VERSION:
        raise ValueError(f"unexpected optimizer probe schema: {path}")
    return payload


def requested_update(
    probe: dict[str, Any],
    parameter: str,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state = {
        name: tensor.to(device)
        for name, tensor in probe["parameters"][parameter].items()
    }
    hyperparameters = probe["hyperparameters"][parameter]
    directions = reconstruct_directions(state, hyperparameters)
    update = (
        float(hyperparameters["lr"])
        * directions["exact_applied_direction"]
    )
    return state["weight_before_step"], state["gradient_after_clip"], update


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--probe-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--layers", default="0,3,6,9,11")
    parser.add_argument("--phase-anchors", default="0,60,120,180")
    parser.add_argument("--ages", default="0,15,30,45,60")
    parser.add_argument("--stages", type=int, default=64)
    parser.add_argument("--neighbors", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--fit-seed", type=int, default=20260804)
    parser.add_argument("--holdout-seed", type=int, default=20260805)
    parser.add_argument("--matching-seed", type=int, default=161803)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stable-retention", type=float, default=0.90)
    parser.add_argument("--stale-retention", type=float, default=0.80)
    args = parser.parse_args()
    started = time.time()
    layers = parse_int_list(args.layers)
    anchors = parse_int_list(args.phase_anchors)
    ages = parse_int_list(args.ages)
    if args.stages != 64 or ages != [0, 15, 30, 45, 60]:
        raise ValueError("the registered protocol is stage64 at ages 0..60")
    pairs = [
        (anchor, anchor + age, age)
        for anchor in anchors
        for age in ages
        if anchor + age <= 225
    ]
    endpoint_steps = sorted({endpoint for _anchor, endpoint, _age in pairs})
    snapshot_paths = [
        args.snapshot_dir / f"step_{step:06d}.pt"
        for step in endpoint_steps
    ]
    probe_paths = [
        args.probe_dir / f"step_{step:06d}.pt"
        for step in endpoint_steps
    ]
    missing = [
        str(path)
        for path in (*snapshot_paths, *probe_paths, args.plan)
        if not path.is_file()
    ]
    if missing:
        raise ValueError(f"registered inputs are absent: {missing}")

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
    anchor_permutations: dict[tuple[int, int], torch.Tensor] = {}
    matching_rows: list[dict[str, Any]] = []
    run_identity: str | None = None
    for anchor in anchors:
        probe = load_registered_probe(
            args.probe_dir / f"step_{anchor:06d}.pt"
        )
        observed_identity = str(probe["run_identity_sha256"])
        if run_identity is None:
            run_identity = observed_identity
        elif run_identity != observed_identity:
            raise ValueError("optimizer probes have inconsistent identities")
        for layer in layers:
            parameter = f"transformer.h.{layer}.mlp.c_proj.weight"
            source, _gradient, update = requested_update(
                probe, parameter, args.device
            )
            permutations, diagnostics = muon_matched_permutations(
                source,
                update,
                stages=args.stages,
                neighbors=args.neighbors,
                seed=args.matching_seed + 1009 * layer + anchor,
            )
            anchor_permutations[(anchor, layer)] = permutations.cpu()
            for diagnostic in diagnostics:
                matching_rows.append(
                    {
                        "kind": "anchor",
                        "step": anchor,
                        "layer": layer,
                        **diagnostic,
                    }
                )
        del probe
        if "cuda" in args.device:
            torch.cuda.empty_cache()

    rows: list[dict[str, Any]] = []
    finite_rows: list[dict[str, Any]] = []
    for endpoint in endpoint_steps:
        snapshot_path = (
            args.snapshot_dir / f"step_{endpoint:06d}.pt"
        )
        payload = load_snapshot(snapshot_path)
        if payload.get("schema_version") != TRAJECTORY_SCHEMA_VERSION:
            raise ValueError(f"unexpected trajectory snapshot: {snapshot_path}")
        probe = load_registered_probe(
            args.probe_dir / f"step_{endpoint:06d}.pt"
        )
        if str(probe["run_identity_sha256"]) != run_identity:
            raise ValueError("snapshot/probe replay identities disagree")
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

        fresh_updates: dict[int, torch.Tensor] = {}
        requested_by_layer: dict[int, torch.Tensor] = {}
        train_gradients: dict[int, torch.Tensor] = {}
        source_by_layer: dict[int, torch.Tensor] = {}
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
            permutations, diagnostics = muon_matched_permutations(
                source,
                requested,
                stages=args.stages,
                neighbors=args.neighbors,
                seed=args.matching_seed + 1009 * layer + endpoint,
            )
            fresh, _fit = diagonal_metric_causal_givens_update(
                source,
                requested,
                stages=args.stages,
                seed=args.matching_seed,
                permutations=permutations,
            )
            for diagnostic in diagnostics:
                matching_rows.append(
                    {
                        "kind": "fresh",
                        "step": endpoint,
                        "layer": layer,
                        **diagnostic,
                    }
                )
            source_by_layer[layer] = source
            requested_by_layer[layer] = requested
            train_gradients[layer] = gradient
            fresh_updates[layer] = fresh

        endpoint_pairs = [
            (anchor, age)
            for anchor, observed_endpoint, age in pairs
            if observed_endpoint == endpoint
        ]
        for anchor, age in endpoint_pairs:
            aged_updates: dict[int, torch.Tensor] = {}
            for layer in layers:
                aged, _fit = diagonal_metric_causal_givens_update(
                    source_by_layer[layer],
                    requested_by_layer[layer],
                    stages=args.stages,
                    seed=args.matching_seed,
                    permutations=anchor_permutations[
                        (anchor, layer)
                    ].to(args.device),
                )
                aged_updates[layer] = aged
            candidates = {"fresh": fresh_updates, "aged": aged_updates}
            for window in WINDOWS:
                finite_rows.append(
                    {
                        "phase_anchor": anchor,
                        "endpoint": endpoint,
                        "age_updates": age,
                        "window": window,
                        "candidate": "baseline",
                        "loss": baseline_losses[window],
                        "loss_change_from_baseline": 0.0,
                    }
                )
                for candidate in CANDIDATES:
                    loss = evaluate_with_updates(
                        model,
                        batches_by_window[window],
                        candidates[candidate],
                        args.device,
                    )
                    finite_rows.append(
                        {
                            "phase_anchor": anchor,
                            "endpoint": endpoint,
                            "age_updates": age,
                            "window": window,
                            "candidate": candidate,
                            "loss": loss,
                            "loss_change_from_baseline": (
                                loss - baseline_losses[window]
                            ),
                        }
                    )
            for window in WINDOWS:
                for layer in layers:
                    hidden = activations[window][layer].to(args.device)
                    for candidate in CANDIDATES:
                        update = candidates[candidate][layer]
                        output = output_space_metrics(
                            hidden,
                            requested_by_layer[layer],
                            update,
                        )
                        train = task_descent_metrics(
                            train_gradients[layer],
                            update,
                        )
                        validation = task_descent_metrics(
                            validation_gradients[window][layer],
                            update.cpu(),
                        )
                        row = {
                            "phase_anchor": anchor,
                            "endpoint": endpoint,
                            "age_updates": age,
                            "layer": layer,
                            "window": window,
                            "candidate": candidate,
                            "output_fixed_scale_recovery": output[
                                "fixed_scale_recovery"
                            ],
                            "output_positive_line_recovery": output[
                                "positive_step_line_recovery"
                            ],
                            "output_cosine": output["cosine"],
                            "target_output_energy": output[
                                "target_output_energy"
                            ],
                            "train_gradient_predicted_ce_decrease": train[
                                "predicted_ce_decrease"
                            ],
                            "validation_gradient_predicted_ce_decrease": (
                                validation["predicted_ce_decrease"]
                            ),
                            "update_fro": train["update_fro"],
                        }
                        rows.append(row)
                        print(json.dumps(row, sort_keys=True), flush=True)

        del model, payload, probe
        if "cuda" in args.device:
            torch.cuda.empty_cache()

    aggregate = aggregate_retention(
        rows,
        finite_rows,
        stable_retention=args.stable_retention,
        stale_retention=args.stale_retention,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    detail_path = args.output / "muon_chart_staleness.csv"
    finite_path = args.output / "muon_chart_staleness_finite_ce.csv"
    matching_path = args.output / "muon_chart_staleness_matchings.csv"
    aggregate_path = args.output / "muon_chart_staleness_aggregate.json"
    write_csv(detail_path, rows)
    write_csv(finite_path, finite_rows)
    write_csv(matching_path, matching_rows)
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_mlp_muon_chart_staleness_v1",
        "decision": aggregate["decision"],
        "causal_protocol": (
            "phase-start connectivity uses only the exact phase-start "
            "Muon direction; future directions refit angles and score the "
            "fixed connectivity but never change it"
        ),
        "parameter_updates": 0,
        "learned_dense_basis": False,
        "lora_adapter": False,
        "layers": layers,
        "phase_anchors": anchors,
        "ages": ages,
        "stage_count": args.stages,
        "neighbors": args.neighbors,
        "input_run_identity_sha256": run_identity,
        "validation_windows": {
            "fit_seed": args.fit_seed,
            "holdout_seed": args.holdout_seed,
            "batches": args.batches,
            "batch_size": args.batch_size,
            "block_size": args.block_size,
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
            "finite_ce_sha256": file_sha256(finite_path),
            "matchings_sha256": file_sha256(matching_path),
            "aggregate_sha256": file_sha256(aggregate_path),
        },
        "limitations": [
            "Future exact directions refit diagonal angles, so this is an upper-bound test of fixed-connectivity capacity rather than a replay of learned angle optimizer states.",
            "Five representative c_proj layers and one dense-Muon trajectory do not establish a global model-manifold dimension.",
            "A qualified cadence must still pass a real end-to-end MFU gate before any training candidate is launched.",
        ],
    }
    metadata_path = args.output / "muon_chart_staleness_metadata.json"
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
