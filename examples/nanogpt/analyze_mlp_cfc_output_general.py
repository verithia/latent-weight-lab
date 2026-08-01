#!/usr/bin/env python3
"""Attribute dense-minus-fresh ``c_fc`` motion to output skew and shear.

For full-column-rank ``W``, every residual update ``E`` has the exact
one-sided representation ``E = A W`` with ``A = E W^+``.  This zero-update
diagnostic decomposes that well-conditioned output action into skew rotation,
symmetric shear, and radial singular-value motion, then measures held-out CE.
"""

from __future__ import annotations

import argparse
import json
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
from examples.nanogpt.analyze_mlp_cfc_generator_spectrum import (
    spectrum_metrics,
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


SCHEMA_VERSION = "nanogpt_mlp_cfc_output_general_v1"
CANDIDATES = (
    "fresh88",
    "dense_exact",
    "fresh_plus_output_skew",
    "fresh_plus_output_symmetric_shear",
    "fresh_plus_output_skew_shear",
)


def git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def output_general_components(
    weight: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, float], dict[str, dict[str, float | int]]]:
    """Return exact ``A W`` components and compact operator spectra."""
    weight_f = weight.float()
    residual_f = residual.float()
    if weight_f.ndim != 2 or residual_f.shape != weight_f.shape:
        raise ValueError("weight and residual must be same-shaped matrices")
    u, singular, vh = torch.linalg.svd(weight_f, full_matrices=False)
    v = vh.T
    left_factor = residual_f @ v / singular.clamp_min(1e-30)[None, :]
    reconstructed = left_factor @ (u.T @ weight_f)
    transpose_action = u @ (left_factor.T @ weight_f)
    skew = 0.5 * (reconstructed - transpose_action)
    symmetric = 0.5 * (reconstructed + transpose_action)
    coordinates = u.T @ residual_f @ v
    radial = u @ torch.diag(torch.diagonal(coordinates)) @ vh
    symmetric_shear = symmetric - radial
    skew_shear = skew + symmetric_shear

    core = u.T @ left_factor
    perpendicular = left_factor - u @ core
    _q, upper = torch.linalg.qr(perpendicular, mode="reduced")
    width = weight_f.shape[1]
    general_block = torch.zeros(
        2 * width,
        2 * width,
        device=weight_f.device,
        dtype=torch.float32,
    )
    general_block[:width, :width] = core
    general_block[width:, :width] = upper
    skew_block = 0.5 * (general_block - general_block.T)
    symmetric_block = 0.5 * (general_block + general_block.T)
    spectra = {
        "general": spectrum_metrics(torch.linalg.svdvals(general_block)),
        "skew": spectrum_metrics(torch.linalg.svdvals(skew_block)),
        "symmetric": spectrum_metrics(torch.linalg.svdvals(symmetric_block)),
    }
    diagnostics = {
        "condition_number": float(
            singular.max() / singular.min().clamp_min(1e-30)
        ),
        "general_reconstruction_error": float(
            (reconstructed - residual_f).norm()
            / residual_f.norm().clamp_min(1e-30)
        ),
        "component_reconstruction_error": float(
            (skew + symmetric - residual_f).norm()
            / residual_f.norm().clamp_min(1e-30)
        ),
        "skew_shear_bilateral_error": float(
            (skew_shear - (residual_f - radial)).norm()
            / residual_f.norm().clamp_min(1e-30)
        ),
        "general_operator_spectral_norm": float(
            spectra["general"]["top_singular_value"]
        ),
        "skew_operator_spectral_norm": float(
            spectra["skew"]["top_singular_value"]
        ),
        "symmetric_operator_spectral_norm": float(
            spectra["symmetric"]["top_singular_value"]
        ),
    }
    return {
        "output_general": reconstructed,
        "output_skew": skew,
        "output_symmetric": symmetric,
        "output_symmetric_shear": symmetric_shear,
        "output_skew_shear": skew_shear,
        "radial": radial,
    }, diagnostics, spectra


def median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def aggregate(
    loss_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
    *,
    windows: list[str],
    maximum_replicate_range: float,
    maximum_reconstruction_error: float,
    maximum_operator_norm: float,
    sufficient_minimum_recovery: float,
    sufficient_median_recovery: float,
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
        values["range"] <= maximum_replicate_range
        for window in summaries.values()
        for values in window.values()
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
                values[candidate]["maximum"] < values["fresh88"]["minimum"]
                for values in summaries.values()
            ),
            "sufficient": all(
                (
                    stable,
                    dense_positive,
                    minimum >= sufficient_minimum_recovery,
                    med >= sufficient_median_recovery,
                )
            ),
        }
    numerical_action = all(
        (
            max(row["general_reconstruction_error"] for row in diagnostic_rows)
            <= maximum_reconstruction_error,
            max(row["component_reconstruction_error"] for row in diagnostic_rows)
            <= maximum_reconstruction_error,
            max(row["general_operator_spectral_norm"] for row in diagnostic_rows)
            <= maximum_operator_norm,
        )
    )
    skew = results["fresh_plus_output_skew"]
    symmetric = results["fresh_plus_output_symmetric_shear"]
    combined = results["fresh_plus_output_skew_shear"]
    if not dense_positive:
        decision = "DENSE_RESIDUAL_NOT_POSITIVE_CONTROL"
    elif not stable or not numerical_action:
        decision = "OUTPUT_GENERAL_NUMERICAL_GATE_FAILED"
    elif skew["sufficient"]:
        decision = "OUTPUT_SKEW_FAMILY_SUFFICIENT"
    elif symmetric["sufficient"]:
        decision = "OUTPUT_SYMMETRIC_SHEAR_FAMILY_SUFFICIENT"
    elif combined["sufficient"]:
        decision = "OUTPUT_GENERAL_SKEW_SHEAR_REQUIRED"
    else:
        decision = "OUTPUT_GENERAL_DECOMPOSITION_NOT_SUFFICIENT"
    return {
        "decision": decision,
        "parameter_updates": 0,
        "summaries": summaries,
        "candidate_results": results,
        "gates": {
            "numerically_stable": stable,
            "dense_beats_fresh_every_window": dense_positive,
            "output_general_action_well_conditioned": numerical_action,
        },
        "diagnostic_extrema": {
            key: max(float(row[key]) for row in diagnostic_rows)
            for key in (
                "general_reconstruction_error",
                "component_reconstruction_error",
                "skew_shear_bilateral_error",
                "general_operator_spectral_norm",
                "skew_operator_spectral_norm",
                "symmetric_operator_spectral_norm",
                "condition_number",
            )
        },
        "thresholds": {
            "maximum_replicate_range": maximum_replicate_range,
            "maximum_reconstruction_error": maximum_reconstruction_error,
            "maximum_operator_norm": maximum_operator_norm,
            "sufficient_minimum_recovery": sufficient_minimum_recovery,
            "sufficient_median_recovery": sufficient_median_recovery,
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
        model, fit_batches, layers, device=args.device, dtype=torch.bfloat16
    )
    updates: dict[str, dict[int, torch.Tensor]] = {
        candidate: {} for candidate in CANDIDATES
    }
    metric_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []
    for layer in layers:
        weight = model.transformer.h[layer].mlp.c_fc.weight
        owner, group = _optimizer_and_group_for_parameter(optimizer, weight)
        buffer = owner.state[weight].get("momentum_buffer")
        if buffer is None:
            raise RuntimeError(f"missing c_fc momentum at layer {layer}")
        dense_update, descent, _diagnostics = exact_muon_update(
            weight.detach(),
            gradients[layer].to(weight.device),
            buffer,
            learning_rate=float(group["lr"]),
            momentum=float(group["momentum"]),
            weight_decay=float(group["weight_decay"]),
            ns_steps=int(group["ns_steps"]),
        )
        polar_descent = (
            descent + float(group["weight_decay"]) * weight.detach().float()
        )
        matched, _selections = build_candidates(
            weight.detach(),
            dense_update,
            polar_descent,
            parent_stages=int(protocol["parent_stages"]),
            residual_stages=int(protocol["residual_stages"]),
            neighbors=int(protocol["matching_neighbors"]),
            seed=int(protocol["matching_seed"]) + layer * 1009,
            learning_rate=float(group["lr"]),
            weight_decay=float(group["weight_decay"]),
            native_cache=args.native_cache,
        )
        fresh = matched["fresh_expansion88"].float()
        residual = dense_update.float() - fresh
        components, diagnostics, spectra = output_general_components(
            weight.detach(), residual
        )
        updates["fresh88"][layer] = fresh.cpu()
        updates["dense_exact"][layer] = dense_update.float().cpu()
        mapping = {
            "fresh_plus_output_skew": "output_skew",
            "fresh_plus_output_symmetric_shear": "output_symmetric_shear",
            "fresh_plus_output_skew_shear": "output_skew_shear",
        }
        for candidate, component_name in mapping.items():
            component = components[component_name]
            updates[candidate][layer] = (fresh + component).cpu()
            metric_rows.append(
                {
                    "layer": layer,
                    "candidate": candidate,
                    "component": component_name,
                    **residual_metrics(residual, component),
                }
            )
        diagnostic_rows.append({"layer": layer, **diagnostics})
        for family, values in spectra.items():
            spectrum_rows.append({"layer": layer, "family": family, **values})

    repeats = int(protocol["evaluation_repeats"])
    loss_rows: list[dict[str, Any]] = []
    for window, batches in validation_batches.items():
        for candidate, candidate_updates in (
            ("baseline", None),
            *((name, updates[name]) for name in CANDIDATES),
        ):
            losses = repeated_losses(
                model,
                batches,
                candidate_updates,
                repeats=repeats,
                device=args.device,
                dtype=torch.float32,
            )
            for repeat, loss in enumerate(losses):
                loss_rows.append(
                    {
                        "window": window,
                        "candidate": candidate,
                        "repeat": repeat,
                        "loss": loss,
                    }
                )
        print(
            json.dumps(
                {"phase_complete": "validation_window", "window": window},
                sort_keys=True,
            ),
            flush=True,
        )
    result = aggregate(
        loss_rows,
        diagnostic_rows,
        windows=windows,
        maximum_replicate_range=float(rule["maximum_replicate_range"]),
        maximum_reconstruction_error=float(rule["maximum_reconstruction_error"]),
        maximum_operator_norm=float(rule["maximum_operator_norm"]),
        sufficient_minimum_recovery=float(rule["sufficient_minimum_recovery"]),
        sufficient_median_recovery=float(rule["sufficient_median_recovery"]),
    )
    result["fit_gradient_loss_bfloat16"] = fit_loss
    args.output.mkdir(parents=True, exist_ok=True)
    paths = {
        "losses": args.output / "cfc_output_general_losses.csv",
        "metrics": args.output / "cfc_output_general_metrics.csv",
        "diagnostics": args.output / "cfc_output_general_diagnostics.csv",
        "spectra": args.output / "cfc_output_general_spectra.csv",
        "aggregate": args.output / "cfc_output_general_aggregate.json",
    }
    write_csv(paths["losses"], loss_rows)
    write_csv(paths["metrics"], metric_rows)
    write_csv(paths["diagnostics"], diagnostic_rows)
    write_csv(paths["spectra"], spectrum_rows)
    paths["aggregate"].write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
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
            "direct_foreground_polling": True,
            "watchdog": False,
            "callback": False,
        },
        "protocol": protocol,
        "outputs": {
            f"{name}_sha256": file_sha256(path)
            for name, path in paths.items()
        },
        "limitations": plan["limitations"],
    }
    metadata_path = args.output / "cfc_output_general_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "aggregate": str(paths["aggregate"]),
                "metadata": str(metadata_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
