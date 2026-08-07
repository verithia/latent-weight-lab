#!/usr/bin/env python3
"""Test norm-balanced feedback inside the stable c_fc task gauge."""

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
    DirectedProductCfcApplier,
    cfc_modules,
    collect_cfc_gradients,
    directed_optimizer,
    scaled_to_dense_ratio,
)
from examples.nanogpt.analyze_mlp_cfc_exact_current_matcher import (
    file_sha256,
    fixed_batches,
    git_commit,
    load_model_and_optimizer,
)
from examples.nanogpt.analyze_mlp_dense_oracle_gap import (
    aggregate_direction_metrics,
    family_fro,
)
from examples.nanogpt.analyze_qk_cfc_20tpp_late_capacity_gate import paired_ce
from examples.nanogpt.analyze_qk_cfc_20tpp_phase_direction import evaluate_ce
from examples.nanogpt.analyze_qk_cfc_selector_fitter_factorial import (
    fit_with_separate_selection_target,
    support_overlap,
)
from examples.nanogpt.analyze_qk_cfc_terminal_optimizer_stability import (
    optimizer_targets,
    subset,
)
from examples.nanogpt.analyze_residual_compatibility import fixed_validation_batches


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "mai_124m_qk_cfc_balanced_feedback_gate_v1"
CANDIDATES = (
    "current_corrected",
    "task_only",
    "layer_equal_balance",
    "layer_half_balance",
    "global_equal_balance",
)
PROMOTABLE = CANDIDATES[2:]


def balanced_target(
    requested: dict[int, torch.Tensor],
    feedback: dict[int, torch.Tensor],
    *,
    gamma: float,
    layerwise: bool,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    if not 0.0 < float(gamma) <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    if layerwise:
        scales = {
            layer: float(gamma)
            * float(requested[layer].double().norm())
            / max(float(feedback[layer].double().norm()), 1e-30)
            for layer in requested
        }
    else:
        scale = (
            float(gamma)
            * family_fro(requested)
            / max(family_fro(feedback), 1e-30)
        )
        scales = {layer: scale for layer in requested}
    target = {
        layer: requested[layer] + scales[layer] * feedback[layer]
        for layer in requested
    }
    return target, {
        "gamma": float(gamma),
        "layerwise": bool(layerwise),
        "feedback_scales": scales,
        "target_to_requested_family_fro_ratio": (
            family_fro(target) / max(family_fro(requested), 1e-30)
        ),
    }


def candidate_passes(
    name: str,
    rows: dict[str, Any],
    functional: dict[str, Any],
    rule: dict[str, Any],
) -> bool:
    windows = ("fit", "holdout")
    return (
        name in PROMOTABLE
        and min(
            rows[window][name]["late_action_vs_requested"][
                "positive_line_recovery"
            ]
            - rows[window]["current_corrected"]["late_action_vs_requested"][
                "positive_line_recovery"
            ]
            for window in windows
        )
        >= float(rule["minimum_late_requested_recovery_improvement"])
        and rows["cross_window"][name]["late_action_cosine"]
        >= float(rule["minimum_late_action_cross_window_cosine"])
        and max(
            rows[window][name]["outgoing_to_incoming_feedback_fro_ratio"]
            for window in windows
        )
        <= float(rule["maximum_outgoing_to_incoming_feedback_ratio"])
        and max(
            functional[window][name]["vs_current"]["upper_confidence_bound"]
            for window in windows
        )
        <= float(rule["maximum_functional_upper_bound_regression_ce"])
        and min(
            functional[window][name]["vs_current"][
                "candidate_minus_current_mean_ce"
            ]
            for window in windows
        )
        <= -float(rule["minimum_one_window_mean_ce_improvement"])
    )


def classify(
    rows: dict[str, Any], functional: dict[str, Any], rule: dict[str, Any]
) -> dict[str, Any]:
    passing = [
        name for name in PROMOTABLE if candidate_passes(name, rows, functional, rule)
    ]
    selected = (
        min(passing, key=list(rule["candidate_priority"]).index) if passing else None
    )
    return {
        "classification": (
            "PROMOTE_BALANCED_FEEDBACK_TO_IMPLEMENTATION_MFU_GATE"
            if selected
            else "REJECT_NORM_BALANCED_FEEDBACK"
        ),
        "selected_candidate": selected,
        "passing_candidates": passing,
        "thresholds": rule,
        "parameter_updates_to_checkpoint": 0,
        "implementation_mfu_preflight_authorized": selected is not None,
        "language_model_training_authorized": False,
    }


def validate(args: argparse.Namespace, plan: dict[str, Any]) -> None:
    observed = {
        "entrypoint_sha256": file_sha256(Path(__file__)),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "config_sha256": file_sha256(args.config),
        "dataset_manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
        "factorial_result_sha256": file_sha256(args.factorial_result),
    }
    if observed != plan["identity"]:
        raise ValueError(f"balanced-feedback identity mismatch: {observed}")
    parent = json.loads(args.factorial_result.read_text())
    if parent.get("decision", {}).get("classification") != (
        "REJECT_SELECTOR_FITTER_DECOUPLING"
    ):
        raise ValueError("parent result does not authorize balanced-feedback gate")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--factorial-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    plan = json.loads(args.plan.read_text())
    if plan.get("schema_version") != "mai_124m_qk_cfc_balanced_feedback_gate_plan_v1":
        raise ValueError("unexpected balanced-feedback plan schema")
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
    schedule = [int(value) for value in protocol["schedule"]]
    late_layers = [int(value) for value in protocol["late_layers"]]
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
    validation_batches = fixed_validation_batches(
        args.data_dir,
        int(protocol["validation_batch_size"]),
        int(config["block_size"]) + 1,
        int(protocol["validation_batches"]),
        int(protocol["validation_seed"]),
    )
    rows: dict[str, Any] = {}
    actions: dict[str, dict[str, dict[int, torch.Tensor]]] = {}
    supports: dict[str, dict[str, list[torch.Tensor]]] = {}
    for window, batches in windows.items():
        print(f"collecting balanced-feedback window {window}", flush=True)
        mean_ce = collect_cfc_gradients(
            model, modules, batches, device=args.device, dtype=dtype
        )
        targets, _diagnostics = optimizer_targets(owner, modules)
        requested = targets["momentum_requested"]
        corrected = targets["feedback_corrected"]
        feedback = {
            layer: corrected[layer] - requested[layer] for layer in requested
        }
        balanced = {
            "layer_equal_balance": balanced_target(
                requested, feedback, gamma=1.0, layerwise=True
            ),
            "layer_half_balance": balanced_target(
                requested, feedback, gamma=0.5, layerwise=True
            ),
            "global_equal_balance": balanced_target(
                requested, feedback, gamma=1.0, layerwise=False
            ),
        }
        fit_targets = {
            "current_corrected": (corrected, corrected),
            "task_only": (requested, requested),
            **{
                name: (requested, value[0]) for name, value in balanced.items()
            },
        }
        actions[window] = {}
        supports[window] = {}
        rows[window] = {
            "gradient_mean_ce": mean_ce,
            "balance_diagnostics": {
                name: value[1] for name, value in balanced.items()
            },
        }
        for name, (selection, fitted) in fit_targets.items():
            raw, support = fit_with_separate_selection_target(
                modules,
                selection,
                fitted,
                schedule=schedule,
                ridge_ratio=float(protocol["ridge_ratio"]),
                chunk_size=int(protocol["chunk_size"]),
            )
            radius_target = corrected if name == "current_corrected" else requested
            action = scaled_to_dense_ratio(raw, radius_target, 1.0)
            actions[window][name] = action
            supports[window][name] = support
            outgoing = {
                layer: corrected[layer] - action[layer] for layer in action
            }
            rows[window][name] = {
                "action_vs_requested": aggregate_direction_metrics(requested, action),
                "late_action_vs_requested": aggregate_direction_metrics(
                    subset(requested, late_layers), subset(action, late_layers)
                ),
                "action_vs_corrected": aggregate_direction_metrics(corrected, action),
                "outgoing_to_incoming_feedback_fro_ratio": (
                    family_fro(outgoing) / max(family_fro(feedback), 1e-30)
                ),
            }

    first, second = tuple(windows)
    rows["cross_window"] = {}
    for name in CANDIDATES:
        rows["cross_window"][name] = {
            "action": aggregate_direction_metrics(actions[first][name], actions[second][name]),
            "late_action_cosine": aggregate_direction_metrics(
                subset(actions[first][name], late_layers),
                subset(actions[second][name], late_layers),
            )["cosine"],
            "support_overlap_late": support_overlap(
                supports[first][name], supports[second][name], late_layers
            ),
        }

    applier = DirectedProductCfcApplier(modules)
    baseline_mean, baseline_losses = evaluate_ce(
        model, validation_batches, device=args.device, dtype=dtype
    )
    functional: dict[str, Any] = {
        "baseline_no_update": {"mean_ce": baseline_mean, "batch_ce": baseline_losses}
    }
    for window in windows:
        functional[window] = {}
        for name in CANDIDATES:
            with applier.apply({"c_fc": actions[window][name]}):
                mean_ce, losses = evaluate_ce(
                    model, validation_batches, device=args.device, dtype=dtype
                )
            functional[window][name] = {"mean_ce": mean_ce, "batch_ce": losses}
        current_losses = functional[window]["current_corrected"]["batch_ce"]
        for name in CANDIDATES[1:]:
            functional[window][name]["vs_current"] = paired_ce(
                functional[window][name]["batch_ce"],
                current_losses,
                float(protocol["confidence_z"]),
            )

    decision = classify(rows, functional, plan["decision_rule"])
    args.output.mkdir(parents=True)
    result = {
        "schema_version": SCHEMA_VERSION,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "decision": decision,
        "gate_rows": rows,
        "functional": functional,
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
