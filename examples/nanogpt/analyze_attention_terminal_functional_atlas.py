#!/usr/bin/env python3
"""Terminal-state adaptive KFAC upper bound for attention V and c_proj.

This is a zero-update teacher oracle.  Unlike the rejected step-zero atlas,
the basis is selected from disjoint CE calibration splits at the terminal
dense checkpoint.  It therefore cannot be deployed causally; it asks only
whether a perfectly late-adapted one-percent separable KFAC chart can contain
the held-out dense path and exact Muon directions better than BlockFHT.
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
    target_tensor,
    trajectory_inventory,
    write_rows,
)
from examples.nanogpt.analyze_attention_paper_activation_oracle import (
    AttentionFunctionalMetric,
    all_finite,
    file_sha256,
    terminal_attention_metrics,
)
from examples.nanogpt.analyze_attention_stepzero_functional_atlas import (
    KroneckerAtlas,
    LinearChart,
    analyze_one_chart,
    collect_stepzero_second_moments,
    git_commit,
    kronecker_subspace_overlap,
    load_target_snapshot,
    summarize_arm,
)
from examples.nanogpt.analyze_residual_compatibility import (
    fixed_validation_batches,
    load_model,
)
from examples.nanogpt.train import require_block_fht_native_extension
from latent_weight_lab.block_fht import block_fht_grad_latent, block_fht_slice


TARGETS = ("v", "cproj")
REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_SCHEMA = "mai_124m_attention_terminal_functional_atlas_plan_v1"
RESULT_SCHEMA = "mai_124m_attention_terminal_functional_atlas_result_v1"


def validate_plan(plan: dict[str, Any], args: argparse.Namespace) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unexpected terminal-atlas executable plan schema")
    protocol = plan["protocol"]
    frozen = {
        "parameter_updates": 0,
        "basis_model_source": "terminal_dense_checkpoint",
        "coordinate_fraction": 0.01,
        "calibration_centering": False,
        "calibration_batch_size": 2,
        "calibration_block_size": 256,
        "calibration_batches": 4,
        "calibration_rows_per_layer": 2048,
        "calibration_metric_seeds": [20260821, 20260822],
        "fit_metric_seed": 20260809,
        "eval_metric_seed": 20260810,
        "metric_batch_size": 2,
        "metric_block_size": 256,
        "metric_batches": 2,
        "trajectory_discovery_max_step": 1140,
        "trajectory_heldout_min_step": 1200,
        "heldout_probe_steps": [1782, 2372],
        "cgls_iterations": 32,
        "span_relative_cutoff": 1e-8,
        "block_fht_layers": 2,
        "block_fht_seed": 1000,
    }
    for field, expected in frozen.items():
        if protocol.get(field) != expected:
            raise ValueError(f"frozen terminal-atlas protocol changed: {field}")
    thresholds = plan["decision_rule"]["thresholds"]
    if thresholds != {
        "aggregate_recovery_minimum": 0.8,
        "minimum_every_layer_recovery": 0.6,
        "minimum_late_layer_8_to_11_recovery": 0.6,
        "minimum_absolute_gain_over_blockfht": 0.1,
        "minimum_calibration_split_subspace_overlap": 0.75,
    }:
        raise ValueError("terminal-atlas thresholds changed")
    if any(bool(value) for value in plan["authorization"].values()):
        raise ValueError("teacher oracle must not pre-authorize a successor")
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
            raise ValueError(f"pinned terminal-atlas identity mismatch: {path}")
    inventory, digest = trajectory_inventory(args.trajectory_dir)
    if (
        len(inventory) != int(identity["trajectory_file_count"])
        or sum(int(item["size"]) for item in inventory)
        != int(identity["trajectory_total_bytes"])
        or digest != identity["trajectory_inventory_sha256"]
    ):
        raise ValueError("trajectory inventory mismatch")
    for name, expected in identity["optimizer_probe_sha256"].items():
        path = args.probe_dir / name
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"optimizer probe mismatch: {path}")
    if Path(identity["trajectory_directory"]) != args.trajectory_dir:
        raise ValueError("trajectory directory differs from plan")
    if Path(identity["optimizer_probe_directory"]) != args.probe_dir:
        raise ValueError("probe directory differs from plan")
    if Path(identity["output_directory_must_be_absent"]) != args.output_dir:
        raise ValueError("output directory differs from plan")


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
    require_block_fht_native_extension(True)
    started = time.time()
    protocol = plan["protocol"]
    layers = [int(value) for value in protocol["layers"]]
    steps = [int(value) for value in protocol["trajectory_steps"]]
    probe_steps = [int(value) for value in protocol["probe_steps"]]
    heldout_probes = {int(value) for value in protocol["heldout_probe_steps"]}
    _inventory, inventory_sha = trajectory_inventory(args.trajectory_dir)

    config = json.loads((REPO_ROOT / plan["identity"]["dense_config"]).read_text())
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

    calibration_moments = []
    calibration_batch_hashes = []
    for seed in protocol["calibration_metric_seeds"]:
        batches = fixed_validation_batches(
            args.data_dir,
            int(protocol["calibration_batch_size"]),
            int(protocol["calibration_block_size"]) + 1,
            int(protocol["calibration_batches"]),
            int(seed),
        )
        calibration_batch_hashes.append(batch_digest(batches))
        model = load_model(args.terminal_checkpoint, args.device)
        calibration_moments.append(
            collect_stepzero_second_moments(
                model,
                batches,
                layers,
                int(protocol["calibration_rows_per_layer"]),
                args.device,
            )
        )
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    if calibration_batch_hashes[0] == calibration_batch_hashes[1]:
        raise ValueError("calibration splits are identical")

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
    fit_inputs = terminal_attention_metrics(
        args.terminal_checkpoint, fit_batches, layers, args.device
    )
    eval_inputs = terminal_attention_metrics(
        args.terminal_checkpoint, eval_batches, layers, args.device
    )
    n_embd = int(config["n_embd"])
    latent_std = float(config.get("block_fht_latent_init_std", 0.02))
    rows: list[dict[str, Any]] = []
    overlaps: list[dict[str, Any]] = []
    storage: list[dict[str, Any]] = []

    for layer in layers:
        print(f"analyzing layer {layer}", flush=True)
        for target, spec in protocol["targets"].items():
            parameter_name = f"transformer.h.{layer}.{spec['parameter']}"
            initial = target_tensor(
                snapshots[steps[0]][parameter_name], target, n_embd
            ).to(args.device)
            coordinate_count = max(
                1, round(initial.numel() * float(protocol["coordinate_fraction"]))
            )
            primary = KroneckerAtlas.from_second_moments(
                calibration_moments[0][(layer, target)][0].to(args.device),
                calibration_moments[0][(layer, target)][1].to(args.device),
                coordinate_count,
            )
            confirmation = KroneckerAtlas.from_second_moments(
                calibration_moments[1][(layer, target)][0].to(args.device),
                calibration_moments[1][(layer, target)][1].to(args.device),
                coordinate_count,
            )
            overlap = kronecker_subspace_overlap(primary, confirmation)
            overlaps.append({"target": target, "layer": layer, "overlap": overlap})
            block_seed = (
                int(protocol["block_fht_seed"])
                + int(spec["seed_stride"]) * layer
                + int(spec["seed_offset"])
            )
            block_template = initial.new_zeros(coordinate_count)
            weight_scale = float(spec["target_std"]) / latent_std

            def apply_block(coordinate: torch.Tensor) -> torch.Tensor:
                return (
                    block_fht_slice(
                        coordinate,
                        initial.numel(),
                        int(protocol["block_fht_layers"]),
                        block_seed,
                        0,
                        initial.numel(),
                    )
                    * weight_scale
                ).view_as(initial)

            def adjoint_block(weight: torch.Tensor) -> torch.Tensor:
                return block_fht_grad_latent(
                    block_template,
                    (weight.reshape(-1) * weight_scale).contiguous(),
                    initial.numel(),
                    int(protocol["block_fht_layers"]),
                    block_seed,
                    0,
                    initial.numel(),
                )

            charts = (
                LinearChart(
                    "terminal_kfac",
                    coordinate_count,
                    primary.apply,
                    primary.adjoint,
                    primary.fixed_storage_bytes(),
                ),
                LinearChart(
                    "blockfht",
                    coordinate_count,
                    apply_block,
                    adjoint_block,
                    0,
                ),
            )
            fit_metric = AttentionFunctionalMetric(target=target, **fit_inputs[layer])
            eval_metric = AttentionFunctionalMetric(target=target, **eval_inputs[layer])
            for chart in charts:
                storage.append(
                    {
                        "arm": chart.name,
                        "target": target,
                        "layer": layer,
                        "coordinate_count": coordinate_count,
                        "fixed_storage_bytes": chart.fixed_storage_bytes,
                    }
                )
                rows.extend(
                    analyze_one_chart(
                        chart=chart,
                        target=target,
                        layer=layer,
                        steps=steps,
                        snapshots=snapshots,
                        probes=probes,
                        parameter_name=parameter_name,
                        n_embd=n_embd,
                        discovery_max=int(protocol["trajectory_discovery_max_step"]),
                        heldout_min=int(protocol["trajectory_heldout_min_step"]),
                        heldout_probes=heldout_probes,
                        fit_metric=fit_metric,
                        eval_metric=eval_metric,
                        cgls_iterations=int(protocol["cgls_iterations"]),
                        span_relative_cutoff=float(protocol["span_relative_cutoff"]),
                        device=args.device,
                    )
                )

    summaries: dict[str, Any] = {}
    thresholds = plan["decision_rule"]["thresholds"]
    for target in protocol["targets"]:
        target_summaries = {
            arm: summarize_arm(rows, arm, target)
            for arm in ("terminal_kfac", "blockfht")
        }
        overlap_values = [
            float(row["overlap"]) for row in overlaps if row["target"] == target
        ]
        target_summaries["calibration_overlap"] = {
            "mean": sum(overlap_values) / len(overlap_values),
            "minimum": min(overlap_values),
        }
        checks: dict[str, bool] = {
            "calibration_overlap": min(overlap_values)
            >= float(thresholds["minimum_calibration_split_subspace_overlap"])
        }
        for metric in ("state", "local_chord", "discovery_span", "exact_muon"):
            primary_metric = target_summaries["terminal_kfac"][metric]
            block_metric = target_summaries["blockfht"][metric]
            checks[f"{metric}_aggregate"] = float(
                primary_metric["aggregate_eval_recovery"]
            ) >= float(thresholds["aggregate_recovery_minimum"])
            checks[f"{metric}_every_layer"] = float(
                primary_metric["minimum_layer_eval_recovery"]
            ) >= float(thresholds["minimum_every_layer_recovery"])
            checks[f"{metric}_late_layers"] = float(
                primary_metric["minimum_late_layer_eval_recovery"]
            ) >= float(thresholds["minimum_late_layer_8_to_11_recovery"])
            gain = float(primary_metric["aggregate_eval_recovery"]) - float(
                block_metric["aggregate_eval_recovery"]
            )
            primary_metric["absolute_gain_over_blockfht"] = gain
            checks[f"{metric}_gain_over_blockfht"] = gain >= float(
                thresholds["minimum_absolute_gain_over_blockfht"]
            )
        target_summaries["checks"] = checks
        target_summaries["passed"] = all(checks.values())
        summaries[target] = target_summaries

    args.output_dir.mkdir(parents=True)
    cells_path = args.output_dir / "attention_terminal_functional_atlas_cells.csv"
    overlap_path = args.output_dir / "attention_terminal_functional_atlas_overlap.csv"
    storage_path = args.output_dir / "attention_terminal_functional_atlas_storage.csv"
    write_rows(cells_path, rows)
    write_rows(overlap_path, overlaps)
    write_rows(storage_path, storage)
    passed = [target for target, summary in summaries.items() if summary["passed"]]
    result = {
        "schema_version": RESULT_SCHEMA,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "classification": (
            "ATTENTION_TERMINAL_FUNCTIONAL_ATLAS_PASS_ALL"
            if len(passed) == len(protocol["targets"])
            else "ATTENTION_TERMINAL_FUNCTIONAL_ATLAS_REJECT"
        ),
        "execution": {
            "host": "PRO6",
            "device": args.device,
            "git_commit": git_commit(),
            "parameter_updates": 0,
            "basis_uses_terminal_teacher_state": True,
            "elapsed_seconds": time.time() - started,
        },
        "identity": {
            "plan_sha256": file_sha256(args.plan),
            "trajectory_inventory_sha256": inventory_sha,
            "trajectory_run_identity_sha256": run_identity,
            "terminal_checkpoint_sha256": file_sha256(args.terminal_checkpoint),
            "dataset_manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
            "calibration_batch_sha256": calibration_batch_hashes,
            "fit_metric_batch_sha256": fit_batch_sha,
            "eval_metric_batch_sha256": eval_batch_sha,
        },
        "protocol": protocol,
        "summaries": summaries,
        "decision": {
            "passed_targets": passed,
            "online_adaptive_atlas_implementation_gate_authorized": len(passed)
            == len(protocol["targets"]),
            "model_implementation_authorized": False,
            "mfu_preflight_authorized": False,
            "language_model_training_authorized": False,
            "larger_rung_authorized": False,
        },
        "artifacts": {
            "cells": {"path": str(cells_path), "sha256": file_sha256(cells_path)},
            "overlap": {
                "path": str(overlap_path),
                "sha256": file_sha256(overlap_path),
            },
            "storage": {
                "path": str(storage_path),
                "sha256": file_sha256(storage_path),
            },
        },
        "all_reported_values_finite": all_finite(summaries),
    }
    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
