#!/usr/bin/env python3
"""Separate gradient, Muon momentum, and error-feedback stability at 20TPP."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_mlp_cfc_directed_product_terminal import (
    cfc_modules,
    collect_cfc_gradients,
    directed_optimizer,
    scaled_to_dense_ratio,
)
from examples.nanogpt.analyze_mlp_cfc_directed_product_terminal_capacity import fit_schedule
from examples.nanogpt.analyze_mlp_cfc_exact_current_matcher import (
    exact_muon_update,
    file_sha256,
    fixed_batches,
    git_commit,
    load_model_and_optimizer,
)
from examples.nanogpt.analyze_mlp_dense_oracle_gap import (
    aggregate_direction_metrics,
    family_fro,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "mai_124m_qk_cfc_terminal_optimizer_stability_v1"
COMPONENTS = ("stateless_requested", "momentum_requested", "feedback_corrected")


def subset(values: dict[int, torch.Tensor], layers: list[int]) -> dict[int, torch.Tensor]:
    return {layer: values[layer] for layer in layers}


@torch.no_grad()
def optimizer_targets(owner, modules) -> tuple[dict[str, dict[int, torch.Tensor]], dict[str, Any]]:
    if len(owner.param_groups) != 1:
        raise ValueError("directed-product optimizer must have one parameter group")
    group = owner.param_groups[0]
    learning_rate = float(group["lr"])
    momentum = float(group["momentum"])
    weight_decay = float(group["weight_decay"])
    ns_steps = int(group["ns_steps"])
    targets: dict[str, dict[int, torch.Tensor]] = {
        name: {} for name in COMPONENTS
    }
    layer_rows: list[dict[str, Any]] = []
    for layer, module in enumerate(modules):
        weight = module.weight
        gradient = weight.grad
        if gradient is None:
            raise RuntimeError(f"missing c_fc gradient for layer {layer}")
        state = owner.state[weight]
        momentum_buffer = state.get("momentum_buffer")
        feedback = state.get("compression_residual")
        if momentum_buffer is None or feedback is None:
            raise RuntimeError(f"missing persisted c_fc optimizer state for layer {layer}")
        stateless, _stateless_descent, _ = exact_muon_update(
            weight,
            gradient,
            torch.zeros_like(momentum_buffer),
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
        )
        requested, _requested_descent, diagnostics = exact_muon_update(
            weight,
            gradient,
            momentum_buffer,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
        )
        corrected = requested.float() + module.error_feedback_decay * feedback.float()
        targets["stateless_requested"][layer] = stateless.detach().float().cpu()
        targets["momentum_requested"][layer] = requested.detach().float().cpu()
        targets["feedback_corrected"][layer] = corrected.detach().float().cpu()
        requested_f = requested.detach().float()
        feedback_f = feedback.detach().float()
        corrected_f = corrected.detach().float()
        layer_rows.append(
            {
                "layer": layer,
                "band": "late" if layer >= 8 else "nonlate",
                "learning_rate": learning_rate,
                "momentum": momentum,
                "error_feedback_decay": float(module.error_feedback_decay),
                "gradient_fro": diagnostics["gradient_fro"],
                "momentum_buffer_fro": diagnostics["momentum_buffer_fro"],
                "requested_fro": float(requested_f.norm()),
                "feedback_fro": float(feedback_f.norm()),
                "corrected_fro": float(corrected_f.norm()),
                "feedback_to_requested_fro_ratio": float(
                    feedback_f.norm() / requested_f.norm().clamp_min(1e-30)
                ),
                "feedback_requested_cosine": float(
                    (feedback_f * requested_f).sum()
                    / (feedback_f.norm() * requested_f.norm()).clamp_min(1e-30)
                ),
            }
        )
    group_diagnostics = {
        "learning_rate": learning_rate,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "ns_steps": ns_steps,
        "parameter_count": len(group["params"]),
    }
    return targets, {"group": group_diagnostics, "layer_rows": layer_rows}


def cross_window(
    first: dict[int, torch.Tensor],
    second: dict[int, torch.Tensor],
    late_layers: list[int],
) -> dict[str, Any]:
    return {
        "all": aggregate_direction_metrics(second, first),
        "late": aggregate_direction_metrics(
            subset(second, late_layers), subset(first, late_layers)
        ),
    }


def classify(metrics: dict[str, Any], rule: dict[str, Any]) -> dict[str, Any]:
    floor = float(rule["minimum_stable_late_cosine"])
    stateless = float(metrics["stateless_requested"]["late"]["cosine"])
    momentum = float(metrics["momentum_requested"]["late"]["cosine"])
    corrected = float(metrics["feedback_corrected"]["late"]["cosine"])
    minimum_gain = float(rule["minimum_material_cosine_gain"])
    if momentum >= floor and momentum - stateless >= minimum_gain:
        if corrected < floor and momentum - corrected >= minimum_gain:
            classification = "ERROR_FEEDBACK_DESTABILIZES_STABLE_MOMENTUM_TARGET"
        else:
            classification = "MOMENTUM_STABILIZES_TASK_TANGENT"
    elif corrected >= floor and corrected - momentum >= minimum_gain:
        classification = "ERROR_FEEDBACK_DOMINATES_TARGET_STABILITY"
    else:
        classification = "OPTIMIZER_STATE_DOES_NOT_STABILIZE_TASK_TANGENT"
    return {
        "classification": classification,
        "late_cross_window_cosine": {
            "stateless_requested": stateless,
            "momentum_requested": momentum,
            "feedback_corrected": corrected,
        },
        "thresholds": rule,
        "parameter_updates_to_checkpoint": 0,
        "training_authorized": False,
        "separate_mechanism_design_required": True,
    }


def validate(args: argparse.Namespace, plan: dict[str, Any]) -> None:
    observed = {
        "entrypoint_sha256": file_sha256(Path(__file__)),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "config_sha256": file_sha256(args.config),
        "dataset_manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
        "production_batch_result_sha256": file_sha256(args.production_batch_result),
    }
    if observed != plan["identity"]:
        raise ValueError(f"terminal optimizer-state identity mismatch: {observed}")
    parent = json.loads(args.production_batch_result.read_text())
    parent_classification = parent.get("classification") or parent.get(
        "decision", {}
    ).get("classification")
    if parent_classification != "REJECT_CAPACITY_DUE_TO_UNSTABLE_TASK_TANGENT":
        raise ValueError("parent result does not authorize optimizer-state audit")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--production-batch-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    plan = json.loads(args.plan.read_text())
    if plan.get("schema_version") != "mai_124m_qk_cfc_terminal_optimizer_stability_plan_v1":
        raise ValueError("unexpected optimizer-stability plan schema")
    validate(args, plan)
    protocol = plan["protocol"]
    config = json.loads(args.config.read_text())
    dtype = getattr(torch, str(config["dtype"]))
    started = time.time()
    model, optimizer, checkpoint = load_model_and_optimizer(
        args.checkpoint, config, args.device
    )
    if int(checkpoint["next_iter"]) != int(protocol["checkpoint_next_iter"]):
        raise ValueError("checkpoint next_iter changed")
    modules = cfc_modules(model)
    owner = directed_optimizer(optimizer)
    windows = {
        name: fixed_batches(
            args.data_dir,
            "train",
            batch_size=int(protocol["gradient_batch_size"]),
            block_size=int(config["block_size"]) + 1,
            batches=int(protocol["gradient_batches"]),
            seed=int(seed),
        )
        for name, seed in protocol["window_seeds"].items()
    }
    targets: dict[str, dict[str, dict[int, torch.Tensor]]] = {}
    diagnostics: dict[str, Any] = {}
    for name, batches in windows.items():
        print(f"collecting production window {name}", flush=True)
        mean_ce = collect_cfc_gradients(
            model, modules, batches, device=args.device, dtype=dtype
        )
        values, rows = optimizer_targets(owner, modules)
        targets[name] = values
        diagnostics[name] = {"mean_ce": mean_ce, **rows}
    names = list(windows)
    first, second = names[0], names[1]
    late_layers = [int(value) for value in protocol["late_layers"]]
    metrics = {
        component: cross_window(
            targets[first][component], targets[second][component], late_layers
        )
        for component in COMPONENTS
    }

    chart_cross_window: dict[str, Any] = {}
    schedule = [int(value) for value in protocol["current_schedule"]]
    for source_name, target_name in ((first, second), (second, first)):
        source_target = targets[source_name]["feedback_corrected"]
        target_target = targets[target_name]["feedback_corrected"]
        raw, _stages = fit_schedule(
            modules,
            source_target,
            schedule=schedule,
            ridge_ratio=float(protocol["ridge_ratio"]),
            chunk_size=int(protocol["chunk_size"]),
        )
        update = scaled_to_dense_ratio(
            raw, source_target, float(protocol["radius_ratio"])
        )
        chart_cross_window[f"{source_name}_fit_on_{target_name}"] = cross_window(
            update, target_target, late_layers
        )

    decision = classify(metrics, plan["decision_rule"])
    args.output.mkdir(parents=True)
    result = {
        "schema_version": SCHEMA_VERSION,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "decision": decision,
        "cross_window_metrics": metrics,
        "chart_cross_window_metrics": chart_cross_window,
        "optimizer_state_diagnostics": diagnostics,
        "family_norms": {
            window: {
                component: family_fro(values[component])
                for component in COMPONENTS
            }
            for window, values in targets.items()
        },
        "identity": {
            **plan["identity"],
            "plan_sha256": file_sha256(args.plan),
            "checkpoint_next_iter": int(checkpoint["next_iter"]),
        },
        "execution": {
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": str(Path(__file__).resolve()),
            "command": sys.argv,
            "device": args.device,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "direct_foreground_polling": True,
            "watchdog": False,
            "callback": False,
        },
        "authorization": {
            "language_model_training": False,
            "mfu_preflight": False,
            "larger_rung": False,
        },
    }
    result_path = args.output / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
