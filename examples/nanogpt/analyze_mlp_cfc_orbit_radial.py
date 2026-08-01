#!/usr/bin/env python3
"""Decompose dense-minus-fresh c_fc motion into orbit and radial parts."""

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
    build_candidates,
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


SCHEMA_VERSION = "nanogpt_mlp_cfc_orbit_radial_v1"
CANDIDATES = (
    "fresh88",
    "dense_exact",
    "fresh_plus_left_orbit",
    "fresh_plus_right_orbit",
    "fresh_plus_bilateral_orbit",
    "fresh_plus_radial",
    "fresh_plus_left_orbit_radial",
    "fresh_plus_right_orbit_radial",
)


def git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def svd_orbit_radial_components(
    weight: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Orthogonally decompose residual motion around a full-column-rank W."""
    weight_f = weight.float()
    residual_f = residual.float()
    u, s, vh = torch.linalg.svd(weight_f, full_matrices=False)
    v = vh.T
    coordinates = u.T @ residual_f @ v
    radial = u @ torch.diag(torch.diagonal(coordinates)) @ vh
    bilateral = residual_f - radial
    perpendicular = residual_f - u @ coordinates @ vh

    size = coordinates.shape[0]
    indices = torch.arange(size, device=coordinates.device)
    i = indices[:, None]
    j = indices[None, :]
    denominator = s[:, None].square() + s[None, :].square()
    left_omega = (
        s[None, :] * coordinates
        - s[:, None] * coordinates.T
    ) / denominator.clamp_min(1e-30)
    right_omega = (
        s[:, None] * coordinates
        - s[None, :] * coordinates.T
    ) / denominator.clamp_min(1e-30)
    left_omega = torch.where(i == j, torch.zeros_like(left_omega), left_omega)
    right_omega = torch.where(i == j, torch.zeros_like(right_omega), right_omega)
    left = perpendicular + u @ (left_omega * s[None, :]) @ vh
    right = u @ (s[:, None] * right_omega) @ vh
    components = {
        "left_orbit": left,
        "right_orbit": right,
        "bilateral_orbit": bilateral,
        "radial": radial,
        "left_orbit_radial": left + radial,
        "right_orbit_radial": right + radial,
    }
    diagnostics = {
        "minimum_singular_value": float(s.min()),
        "maximum_singular_value": float(s.max()),
        "condition_number": float(s.max() / s.min().clamp_min(1e-30)),
        "bilateral_plus_radial_relative_error": float(
            (bilateral + radial - residual_f).norm()
            / residual_f.norm().clamp_min(1e-30)
        ),
        "left_omega_skew_error": float(
            (left_omega + left_omega.T).abs().max()
        ),
        "right_omega_skew_error": float(
            (right_omega + right_omega.T).abs().max()
        ),
    }
    return components, diagnostics


def median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def aggregate(
    rows: list[dict[str, Any]],
    *,
    windows: list[str],
    numerical_range_tolerance: float,
    sufficient_minimum_recovery: float,
    sufficient_median_recovery: float,
    radial_minimum_recovery: float,
    radial_median_recovery: float,
) -> dict[str, Any]:
    summaries: dict[str, dict[str, dict[str, float]]] = {}
    for window in windows:
        summaries[window] = {}
        for candidate in ("baseline", *CANDIDATES):
            summaries[window][candidate] = summarize(
                [
                    float(row["loss"])
                    for row in rows
                    if row["window"] == window
                    and row["candidate"] == candidate
                ]
            )
    stable = all(
        value["range"] <= numerical_range_tolerance
        for window in summaries.values()
        for value in window.values()
    )
    dense_positive = all(
        values["dense_exact"]["maximum"]
        < values["fresh88"]["minimum"]
        for values in summaries.values()
    )
    results: dict[str, dict[str, Any]] = {}
    for candidate in CANDIDATES[2:]:
        recoveries = {}
        for window, values in summaries.items():
            gap = values["fresh88"]["mean"] - values["dense_exact"]["mean"]
            recoveries[window] = (
                values["fresh88"]["mean"] - values[candidate]["mean"]
            ) / max(gap, 1e-30)
        minimum = min(recoveries.values())
        med = median(list(recoveries.values()))
        results[candidate] = {
            "recovery_by_window": recoveries,
            "minimum_recovery": minimum,
            "median_recovery": med,
            "beats_fresh_every_window": all(
                values[candidate]["maximum"]
                < values["fresh88"]["minimum"]
                for values in summaries.values()
            ),
            "sufficient": all(
                (
                    dense_positive,
                    stable,
                    minimum >= sufficient_minimum_recovery,
                    med >= sufficient_median_recovery,
                )
            ),
        }
    radial = results["fresh_plus_radial"]
    radial_material = all(
        (
            radial["minimum_recovery"] >= radial_minimum_recovery,
            radial["median_recovery"] >= radial_median_recovery,
            radial["beats_fresh_every_window"],
        )
    )
    if not dense_positive:
        decision = "DENSE_RESIDUAL_NOT_POSITIVE_CONTROL"
    elif results["fresh_plus_left_orbit"]["sufficient"]:
        decision = "LEFT_ORBIT_FAMILY_SUFFICIENT_SOLVER_CAPACITY_DEFICIT"
    elif results["fresh_plus_bilateral_orbit"]["sufficient"]:
        decision = "ADD_INPUT_SIDE_ROTATION_TO_CFC_CHART"
    elif results["fresh_plus_left_orbit_radial"]["sufficient"]:
        decision = "ADD_RADIAL_SPECTRUM_TO_LEFT_CFC_CHART"
    elif results["fresh_plus_right_orbit_radial"]["sufficient"]:
        decision = "COMPOSITE_RIGHT_ORBIT_RADIAL_CHART_REQUIRED"
    elif radial_material:
        decision = "RADIAL_SPECTRUM_IS_MATERIAL_CFC_DEFICIT"
    else:
        decision = "ORBIT_RADIAL_INTERACTION_REQUIRES_COMPOSITE_CHART"
    return {
        "decision": decision,
        "summaries": summaries,
        "candidate_results": results,
        "gates": {
            "numerically_stable": stable,
            "dense_beats_fresh_every_window": dense_positive,
            "radial_material": radial_material,
        },
        "thresholds": {
            "numerical_range_tolerance": numerical_range_tolerance,
            "sufficient_minimum_recovery": sufficient_minimum_recovery,
            "sufficient_median_recovery": sufficient_median_recovery,
            "radial_minimum_recovery": radial_minimum_recovery,
            "radial_median_recovery": radial_median_recovery,
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
    layers = [int(value) for value in protocol["layers"]]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fit_batches = fixed_batches(
        args.data_dir,
        "train",
        batch_size=int(protocol["batch_size"]),
        block_size=int(protocol["block_size"]),
        batches=int(protocol["fit_batches"]),
        seed=int(protocol["fit_train_seed"]),
    )
    windows = [f"validation_{index + 1}" for index in range(len(protocol["validation_seeds"]))]
    validation_batches = {
        window: fixed_batches(
            args.data_dir,
            "val",
            batch_size=int(protocol["batch_size"]),
            block_size=int(protocol["block_size"]),
            batches=int(protocol["validation_batches_per_window"]),
            seed=int(seed),
        )
        for window, seed in zip(windows, protocol["validation_seeds"], strict=True)
    }
    model, optimizer, checkpoint = load_model_and_optimizer(args.checkpoint, config, args.device)
    fit_loss, gradients = collect_gradient_window(
        model, fit_batches, layers, device=args.device, dtype=torch.bfloat16
    )
    updates: dict[str, dict[int, torch.Tensor]] = {candidate: {} for candidate in CANDIDATES}
    metric_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for layer in layers:
        weight = model.transformer.h[layer].mlp.c_fc.weight
        owner, group = _optimizer_and_group_for_parameter(optimizer, weight)
        buffer = owner.state[weight].get("momentum_buffer")
        if buffer is None:
            raise RuntimeError(f"missing c_fc momentum at layer {layer}")
        dense_update, descent, _diagnostics = exact_muon_update(
            weight.detach(), gradients[layer].to(weight.device), buffer,
            learning_rate=float(group["lr"]), momentum=float(group["momentum"]),
            weight_decay=float(group["weight_decay"]), ns_steps=int(group["ns_steps"]),
        )
        polar_descent = descent + float(group["weight_decay"]) * weight.detach().float()
        matched, selections = build_candidates(
            weight.detach(), dense_update, polar_descent,
            parent_stages=64, residual_stages=24,
            neighbors=int(protocol["matching_neighbors"]),
            seed=int(protocol["matching_seed"]) + layer * 1009,
            learning_rate=float(group["lr"]), weight_decay=float(group["weight_decay"]),
            native_cache=args.native_cache,
        )
        fresh = matched["fresh_expansion88"].float()
        residual = dense_update.float() - fresh
        components, diagnostics = svd_orbit_radial_components(weight.detach(), residual)
        updates["fresh88"][layer] = fresh.cpu()
        updates["dense_exact"][layer] = dense_update.float().cpu()
        mapping = {
            "fresh_plus_left_orbit": "left_orbit",
            "fresh_plus_right_orbit": "right_orbit",
            "fresh_plus_bilateral_orbit": "bilateral_orbit",
            "fresh_plus_radial": "radial",
            "fresh_plus_left_orbit_radial": "left_orbit_radial",
            "fresh_plus_right_orbit_radial": "right_orbit_radial",
        }
        for candidate, component_name in mapping.items():
            component = components[component_name]
            updates[candidate][layer] = (fresh + component).cpu()
            metric_rows.append({
                "layer": layer,
                "candidate": candidate,
                "component": component_name,
                **residual_metrics(residual, component),
            })
        diagnostic_rows.append({"layer": layer, **diagnostics})
        selection_rows.extend({"layer": layer, **selection} for selection in selections)

    repeats = int(protocol["evaluation_repeats"])
    loss_rows: list[dict[str, Any]] = []
    for window, batches in validation_batches.items():
        baseline = repeated_losses(
            model, batches, None, repeats=repeats, device=args.device, dtype=torch.float32
        )
        for repeat, loss in enumerate(baseline):
            loss_rows.append({"window": window, "candidate": "baseline", "repeat": repeat, "loss": loss})
        for candidate in CANDIDATES:
            values = repeated_losses(
                model, batches, updates[candidate], repeats=repeats,
                device=args.device, dtype=torch.float32,
            )
            for repeat, loss in enumerate(values):
                loss_rows.append({"window": window, "candidate": candidate, "repeat": repeat, "loss": loss})
    result = aggregate(
        loss_rows,
        windows=windows,
        numerical_range_tolerance=float(rule["maximum_replicate_range"]),
        sufficient_minimum_recovery=float(rule["sufficient_minimum_recovery"]),
        sufficient_median_recovery=float(rule["sufficient_median_recovery"]),
        radial_minimum_recovery=float(rule["radial_minimum_recovery"]),
        radial_median_recovery=float(rule["radial_median_recovery"]),
    )
    result["fit_gradient_loss_bfloat16"] = fit_loss
    result["parameter_updates"] = 0
    args.output.mkdir(parents=True, exist_ok=True)
    losses_path = args.output / "cfc_orbit_radial_losses.csv"
    metrics_path = args.output / "cfc_orbit_radial_metrics.csv"
    diagnostics_path = args.output / "cfc_orbit_radial_diagnostics.csv"
    selections_path = args.output / "cfc_orbit_radial_selections.json"
    aggregate_path = args.output / "cfc_orbit_radial_aggregate.json"
    write_csv(losses_path, loss_rows)
    write_csv(metrics_path, metric_rows)
    write_csv(diagnostics_path, diagnostic_rows)
    selections_path.write_text(json.dumps(selection_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    aggregate_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "decision": result["decision"],
        "parameter_updates": 0,
        "checkpoint_next_iter": int(checkpoint["next_iter"]),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "config_sha256": file_sha256(args.config),
        "dataset_manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
        "plan_sha256": file_sha256(args.plan),
        "analysis_execution": {
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": str(Path(__file__).resolve()),
            "entrypoint_sha256": file_sha256(Path(__file__).resolve()),
            "command": sys.argv,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "device": args.device,
        },
        "protocol": protocol,
        "outputs": {
            "losses_sha256": file_sha256(losses_path),
            "metrics_sha256": file_sha256(metrics_path),
            "diagnostics_sha256": file_sha256(diagnostics_path),
            "selections_sha256": file_sha256(selections_path),
            "aggregate_sha256": file_sha256(aggregate_path),
        },
        "limitations": plan["limitations"],
    }
    metadata_path = args.output / "cfc_orbit_radial_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "aggregate": str(aggregate_path), "metadata": str(metadata_path)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
