#!/usr/bin/env python3
"""Held-out finite-CE gate for the selected task-matched c_fc shear chart."""

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
    exact_muon_update,
    file_sha256,
    fixed_batches,
    load_model_and_optimizer,
)
from examples.nanogpt.analyze_mlp_cfc_residual_structure import (
    validate_identity,
    write_csv,
)
from examples.nanogpt.analyze_mlp_cfc_task_shear_fit import (
    build_candidates,
)
from examples.nanogpt.analyze_mlp_cfc_trust_radius import (
    collect_gradient_window,
    repeated_losses,
    summarize,
)


SCHEMA_VERSION = "nanogpt_mlp_cfc_task_shear_ce_v1"
CONTROL = "fresh88"
SELECTED = "fresh64_shear24"
DENSE = "dense_exact"
CANDIDATES = (CONTROL, SELECTED, DENSE)


def git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def aggregate(
    loss_rows: list[dict[str, Any]],
    *,
    windows: list[str],
    maximum_replicate_range: float,
    minimum_recovery: float,
    median_recovery: float,
) -> dict[str, Any]:
    summaries: dict[str, dict[str, dict[str, float]]] = {}
    recovery_by_window: dict[str, float] = {}
    dense_positive = True
    selected_positive = True
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
        fresh = summaries[window][CONTROL]["mean"]
        selected = summaries[window][SELECTED]["mean"]
        dense = summaries[window][DENSE]["mean"]
        denominator = fresh - dense
        dense_positive = dense_positive and denominator > 0.0
        selected_positive = selected_positive and selected < fresh
        recovery_by_window[window] = (
            (fresh - selected) / denominator if denominator > 0.0 else float("nan")
        )
    stable = all(
        values["range"] <= maximum_replicate_range
        for candidates in summaries.values()
        for values in candidates.values()
    )
    recoveries = list(recovery_by_window.values())
    finite = all(torch.isfinite(torch.tensor(recoveries)).tolist())
    minimum = min(recoveries) if finite else float("nan")
    central = median(recoveries) if finite else float("nan")
    sufficient = bool(
        stable
        and dense_positive
        and selected_positive
        and finite
        and minimum >= minimum_recovery
        and central >= median_recovery
    )
    if not stable:
        decision = "TASK_SHEAR_CE_REPLICATE_GATE_FAILED"
    elif not dense_positive:
        decision = "TASK_SHEAR_CE_DENSE_REFERENCE_INVALID"
    elif sufficient:
        decision = "PROMOTE_TASK_MATCHED_SHEAR_TO_PRODUCTION_PREFLIGHT"
    else:
        decision = "REJECT_TASK_MATCHED_SHEAR_HELDOUT_CE"
    return {
        "decision": decision,
        "parameter_updates": 0,
        "summaries": summaries,
        "selected_recovery": {
            "by_window": recovery_by_window,
            "minimum": minimum,
            "median": central,
            "beats_fresh_every_window": selected_positive,
            "sufficient": sufficient,
        },
        "gates": {
            "numerically_stable": stable,
            "dense_beats_fresh_every_window": dense_positive,
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
        polar_descent = (
            descent + float(group["weight_decay"]) * weight.detach().float()
        )
        fitted, diagnostics = build_candidates(
            weight.detach(),
            dense_update,
            polar_descent,
            neighbors=int(protocol["matching_neighbors"]),
            seed=int(protocol["matching_seed"]) + layer * 1009,
            learning_rate=float(group["lr"]),
            weight_decay=float(group["weight_decay"]),
            native_cache=args.native_cache,
        )
        updates[CONTROL][layer] = fitted[CONTROL].cpu()
        updates[SELECTED][layer] = fitted[SELECTED].cpu()
        updates[DENSE][layer] = dense_update.float().cpu()
        fit_rows.extend({"layer": layer, **row} for row in diagnostics)
        print(
            json.dumps(
                {"layer_complete": layer, "layers_total": len(layers)},
                sort_keys=True,
            ),
            flush=True,
        )
    loss_rows: list[dict[str, Any]] = []
    repeats = int(protocol["evaluation_repeats"])
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
        windows=windows,
        maximum_replicate_range=float(rule["maximum_replicate_range"]),
        minimum_recovery=float(rule["minimum_recovery"]),
        median_recovery=float(rule["median_recovery"]),
    )
    result["fit_gradient_loss_bfloat16"] = fit_loss
    args.output.mkdir(parents=True, exist_ok=True)
    paths = {
        "losses": args.output / "cfc_task_shear_ce_losses.csv",
        "fits": args.output / "cfc_task_shear_ce_fits.json",
        "aggregate": args.output / "cfc_task_shear_ce_aggregate.json",
    }
    write_csv(paths["losses"], loss_rows)
    paths["fits"].write_text(
        json.dumps(fit_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
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
            f"{name}_sha256": file_sha256(path) for name, path in paths.items()
        },
        "limitations": plan["limitations"],
    }
    metadata_path = args.output / "cfc_task_shear_ce_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
