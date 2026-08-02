#!/usr/bin/env python3
"""Select a c_fc coordinate-mix beta by finite CE, then hold it out."""

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
from examples.nanogpt.analyze_mlp_cfc_functional_shear_ce import (
    DENSE,
    aggregate,
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
from examples.nanogpt.analyze_mlp_cfc_trust_radius import repeated_losses, summarize


SCHEMA_VERSION = "nanogpt_mlp_cfc_functional_shear_ce_select_v1"


def git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def select_beta(
    rows: list[dict[str, Any]],
    *,
    betas: list[float],
    tie_tolerance: float,
) -> tuple[float, dict[str, dict[str, float]]]:
    summaries = {
        scale_name(beta, prefix=PREFIX): summarize(
            [
                float(row["loss"])
                for row in rows
                if row["candidate"] == scale_name(beta, prefix=PREFIX)
            ]
        )
        for beta in betas
    }
    minimum = min(values["mean"] for values in summaries.values())
    eligible = [
        beta
        for beta in betas
        if summaries[scale_name(beta, prefix=PREFIX)]["mean"]
        <= minimum + tie_tolerance
    ]
    return min(eligible), summaries


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
    betas = [float(beta) for beta in protocol["candidate_betas"]]
    beta_names = [scale_name(beta, prefix=PREFIX) for beta in betas]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fit_batches = fixed_batches(
        args.data_dir,
        "train",
        batch_size=int(protocol["batch_size"]),
        block_size=int(protocol["block_size"]),
        batches=int(protocol["fit_batches"]),
        seed=int(protocol["fit_train_seed"]),
    )
    selection_batches = fixed_batches(
        args.data_dir,
        "val",
        batch_size=int(protocol["batch_size"]),
        block_size=int(protocol["block_size"]),
        batches=int(protocol["selection_batches"]),
        seed=int(protocol["selection_seed"]),
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
    update_names = (CONTROL, WEIGHT_SHEAR, *beta_names, DENSE)
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
            betas=betas,
            neighbors=int(protocol["matching_neighbors"]),
            seed=int(protocol["matching_seed"]) + layer * 1009,
            learning_rate=float(group["lr"]),
            weight_decay=float(group["weight_decay"]),
            native_cache=args.native_cache,
            project_to_weight_norm=bool(
                protocol.get("weight_norm_projection", False)
            ),
            max_condition_number=(
                float(protocol["max_condition_number"])
                if protocol.get("max_condition_number") is not None
                else None
            ),
        )
        updates[CONTROL][layer] = fitted[CONTROL].cpu()
        updates[WEIGHT_SHEAR][layer] = fitted[WEIGHT_SHEAR].cpu()
        for name in beta_names:
            updates[name][layer] = fitted[name].cpu()
        updates[DENSE][layer] = dense_update.float().cpu()
        fit_rows.extend({"layer": layer, **row} for row in diagnostics)
        print(json.dumps({"layer_complete": layer, "layers_total": len(layers)}), flush=True)
    repeats = int(protocol["evaluation_repeats"])
    loss_rows: list[dict[str, Any]] = []
    selection_candidates = ("baseline", CONTROL, WEIGHT_SHEAR, *beta_names, DENSE)
    for candidate in selection_candidates:
        candidate_updates = None if candidate == "baseline" else updates[candidate]
        losses = repeated_losses(
            model,
            selection_batches,
            candidate_updates,
            repeats=repeats,
            device=args.device,
            dtype=torch.float32,
        )
        for repeat, loss in enumerate(losses):
            loss_rows.append(
                {
                    "phase": "selection",
                    "window": "selection",
                    "candidate": candidate,
                    "repeat": repeat,
                    "loss": loss,
                }
            )
    selected_beta, selection_summaries = select_beta(
        loss_rows,
        betas=betas,
        tie_tolerance=float(rule["selection_tie_tolerance"]),
    )
    selected = scale_name(selected_beta, prefix=PREFIX)
    weight_selection_mean = summarize(
        [
            float(row["loss"])
            for row in loss_rows
            if row["candidate"] == WEIGHT_SHEAR
        ]
    )["mean"]
    selection_beats_weight = selection_summaries[selected]["mean"] < weight_selection_mean
    print(json.dumps({"selected_beta": selected_beta, "selected_candidate": selected}), flush=True)
    heldout_names = (CONTROL, WEIGHT_SHEAR, selected, DENSE)
    for window, batches in validation_batches.items():
        for candidate, candidate_updates in (
            ("baseline", None),
            *((name, updates[name]) for name in heldout_names),
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
                        "phase": "holdout",
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
    if not selection_beats_weight:
        result["decision"] = "REJECT_FUNCTIONAL_MIX_SELECTION_WINDOW"
    result["selected_beta"] = selected_beta
    result["selected_candidate"] = selected
    result["selection"] = {
        "summaries": selection_summaries,
        "weight_shear_mean": weight_selection_mean,
        "selected_beats_weight_shear": selection_beats_weight,
        "tie_tolerance": float(rule["selection_tie_tolerance"]),
    }
    result["fit_gradient_loss_bfloat16"] = fit_loss
    args.output.mkdir(parents=True, exist_ok=True)
    paths = {
        "losses": args.output / "cfc_functional_shear_ce_select_losses.csv",
        "fits": args.output / "cfc_functional_shear_ce_select_fits.json",
        "aggregate": args.output / "cfc_functional_shear_ce_select_aggregate.json",
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
    metadata_path = args.output / "cfc_functional_shear_ce_select_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": result["decision"], "aggregate": str(paths["aggregate"]), "metadata": str(metadata_path)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
