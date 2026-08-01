#!/usr/bin/env python3
"""Held-out finite-CE gate for the selected c_fc coordinate Pareto mix."""

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
    collect_window,
    exact_muon_update,
    file_sha256,
    fixed_batches,
    load_model_and_optimizer,
)
from examples.nanogpt.analyze_mlp_cfc_functional_shear_fit import (
    CONTROL,
    WEIGHT_SHEAR,
    sample_aligned,
)
from examples.nanogpt.analyze_mlp_cfc_functional_shear_pareto import (
    PREFIX,
    build_pareto_candidates,
)
from examples.nanogpt.analyze_mlp_cfc_functional_shear_radius import scale_name
from examples.nanogpt.analyze_mlp_cfc_residual_structure import (
    validate_identity,
    write_csv,
)
from examples.nanogpt.analyze_mlp_cfc_task_shear_ce import median
from examples.nanogpt.analyze_mlp_cfc_trust_radius import repeated_losses, summarize


SCHEMA_VERSION = "nanogpt_mlp_cfc_functional_shear_ce_v1"
DENSE = "dense_exact"


def git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def aggregate(
    loss_rows: list[dict[str, Any]],
    *,
    windows: list[str],
    selected: str,
    maximum_replicate_range: float,
    minimum_recovery: float,
    median_recovery: float,
) -> dict[str, Any]:
    candidates = ("baseline", CONTROL, WEIGHT_SHEAR, selected, DENSE)
    summaries: dict[str, dict[str, dict[str, float]]] = {}
    recovery_by_window: dict[str, float] = {}
    dense_reference_valid = True
    selected_beats_weight = True
    selected_beats_fresh = True
    for window in windows:
        summaries[window] = {}
        for candidate in candidates:
            summaries[window][candidate] = summarize(
                [
                    float(row["loss"])
                    for row in loss_rows
                    if row["window"] == window and row["candidate"] == candidate
                ]
            )
        fresh = summaries[window][CONTROL]["mean"]
        weight = summaries[window][WEIGHT_SHEAR]["mean"]
        mixed = summaries[window][selected]["mean"]
        dense = summaries[window][DENSE]["mean"]
        denominator = weight - dense
        dense_reference_valid = dense_reference_valid and denominator > 0.0
        selected_beats_weight = selected_beats_weight and mixed < weight
        selected_beats_fresh = selected_beats_fresh and mixed < fresh
        recovery_by_window[window] = (
            (weight - mixed) / denominator if denominator > 0.0 else float("nan")
        )
    stable = all(
        values["range"] <= maximum_replicate_range
        for candidates_by_window in summaries.values()
        for values in candidates_by_window.values()
    )
    recoveries = list(recovery_by_window.values())
    finite = all(torch.isfinite(torch.tensor(recoveries)).tolist())
    minimum = min(recoveries) if finite else float("nan")
    central = median(recoveries) if finite else float("nan")
    sufficient = bool(
        stable
        and dense_reference_valid
        and selected_beats_weight
        and selected_beats_fresh
        and finite
        and minimum >= minimum_recovery
        and central >= median_recovery
    )
    if not stable:
        decision = "FUNCTIONAL_MIX_CE_REPLICATE_GATE_FAILED"
    elif not dense_reference_valid:
        decision = "FUNCTIONAL_MIX_CE_DENSE_REFERENCE_INVALID"
    elif sufficient:
        decision = "PROMOTE_FUNCTIONAL_COORDINATE_MIX_TO_PRODUCTION_PREFLIGHT"
    else:
        decision = "REJECT_FUNCTIONAL_COORDINATE_MIX_HELDOUT_CE"
    return {
        "decision": decision,
        "parameter_updates": 0,
        "summaries": summaries,
        "selected_recovery_over_weight_shear": {
            "by_window": recovery_by_window,
            "minimum": minimum,
            "median": central,
            "beats_weight_shear_every_window": selected_beats_weight,
            "beats_fresh88_every_window": selected_beats_fresh,
            "sufficient": sufficient,
        },
        "gates": {
            "numerically_stable": stable,
            "dense_beats_weight_shear_every_window": dense_reference_valid,
        },
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
    layers = [int(layer) for layer in protocol["layers"]]
    beta = float(protocol["selected_beta"])
    selected = scale_name(beta, prefix=PREFIX)
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
        for window, seed in zip(windows, protocol["validation_seeds"], strict=True)
    }
    model, optimizer, checkpoint = load_model_and_optimizer(
        args.checkpoint, config, args.device
    )
    fit_loss, gradients, inputs, pre_gelu = collect_window(
        model,
        fit_batches,
        layers,
        device=args.device,
        dtype=torch.bfloat16,
    )
    sampled_inputs: dict[int, torch.Tensor] = {}
    sampled_pre_gelu: dict[int, torch.Tensor] = {}
    sample_sha: dict[str, str] = {}
    for layer in layers:
        sampled_input, sampled_pre, sha = sample_aligned(
            inputs[layer],
            pre_gelu[layer],
            sample_cap=int(protocol["functional_sample_cap"]),
            seed=int(protocol["functional_sample_seed"]),
        )
        sampled_inputs[layer] = sampled_input
        sampled_pre_gelu[layer] = sampled_pre
        sample_sha[f"fit_layer{layer}"] = sha
    update_names = (CONTROL, WEIGHT_SHEAR, selected, DENSE)
    updates: dict[str, dict[int, torch.Tensor]] = {
        candidate: {} for candidate in update_names
    }
    fit_rows: list[dict[str, Any]] = []
    for layer in layers:
        weight = model.transformer.h[layer].mlp.c_fc.weight
        owner, group = _optimizer_and_group_for_parameter(optimizer, weight)
        buffer = owner.state[weight].get("momentum_buffer")
        if buffer is None:
            raise RuntimeError(f"missing c_fc momentum at layer {layer}")
        dense_update, descent, _optimizer_diag = exact_muon_update(
            weight.detach(),
            gradients[layer].to(weight.device),
            buffer,
            learning_rate=float(group["lr"]),
            momentum=float(group["momentum"]),
            weight_decay=float(group["weight_decay"]),
            ns_steps=int(group["ns_steps"]),
        )
        polar_descent = descent + float(group["weight_decay"]) * weight.detach().float()
        fitted, diagnostics = build_pareto_candidates(
            weight.detach(),
            dense_update,
            polar_descent,
            sampled_inputs[layer],
            sampled_pre_gelu[layer],
            model.transformer.h[layer].mlp.c_proj.weight.detach(),
            betas=[beta],
            neighbors=int(protocol["matching_neighbors"]),
            seed=int(protocol["matching_seed"]) + layer * 1009,
            learning_rate=float(group["lr"]),
            weight_decay=float(group["weight_decay"]),
            native_cache=args.native_cache,
        )
        updates[CONTROL][layer] = fitted[CONTROL].cpu()
        updates[WEIGHT_SHEAR][layer] = fitted[WEIGHT_SHEAR].cpu()
        updates[selected][layer] = fitted[selected].cpu()
        updates[DENSE][layer] = dense_update.float().cpu()
        fit_rows.extend({"layer": layer, **row} for row in diagnostics)
        print(json.dumps({"layer_complete": layer, "layers_total": len(layers)}), flush=True)
    loss_rows: list[dict[str, Any]] = []
    repeats = int(protocol["evaluation_repeats"])
    for window, batches in validation_batches.items():
        for candidate, candidate_updates in (
            ("baseline", None),
            *((name, updates[name]) for name in update_names),
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
        print(json.dumps({"phase_complete": "validation_window", "window": window}), flush=True)
    result = aggregate(
        loss_rows,
        windows=windows,
        selected=selected,
        maximum_replicate_range=float(rule["maximum_replicate_range"]),
        minimum_recovery=float(rule["minimum_recovery"]),
        median_recovery=float(rule["median_recovery"]),
    )
    result["selected_beta"] = beta
    result["selected_candidate"] = selected
    result["fit_gradient_loss_bfloat16"] = fit_loss
    args.output.mkdir(parents=True, exist_ok=True)
    paths = {
        "losses": args.output / "cfc_functional_shear_ce_losses.csv",
        "fits": args.output / "cfc_functional_shear_ce_fits.json",
        "aggregate": args.output / "cfc_functional_shear_ce_aggregate.json",
    }
    write_csv(paths["losses"], loss_rows)
    paths["fits"].write_text(json.dumps(fit_rows, indent=2, sort_keys=True) + "\n")
    paths["aggregate"].write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "decision": result["decision"],
        "parameter_updates": 0,
        "checkpoint_next_iter": int(checkpoint["next_iter"]),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "config_sha256": file_sha256(args.config),
        "dataset_manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
        "plan_sha256": file_sha256(args.plan),
        "sample_indices_sha256": sample_sha,
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
        "outputs": {f"{name}_sha256": file_sha256(path) for name, path in paths.items()},
        "limitations": plan["limitations"],
    }
    metadata_path = args.output / "cfc_functional_shear_ce_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": result["decision"], "aggregate": str(paths["aggregate"]), "metadata": str(metadata_path)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
