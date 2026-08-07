#!/usr/bin/env python3
"""Factor c_fc topology selection, backlog fitting, and action radius."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
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
from examples.nanogpt.analyze_mlp_cfc_directed_product_terminal_capacity import (
    fit_schedule,
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
from examples.nanogpt.analyze_qk_cfc_terminal_optimizer_stability import (
    optimizer_targets,
    subset,
)
from examples.nanogpt.analyze_residual_compatibility import fixed_validation_batches


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "mai_124m_qk_cfc_selector_fitter_factorial_v1"
CANDIDATES = (
    "current_corrected",
    "task_only",
    "task_select_corrected_fit_corrected_radius",
    "task_select_corrected_fit_requested_radius",
)
PROMOTABLE = CANDIDATES[2:]


@torch.no_grad()
def _fit_on_support(
    source: torch.Tensor,
    remaining: torch.Tensor,
    indices: torch.Tensor,
    *,
    ridge_ratio: float,
    chunk_size: int,
) -> torch.Tensor:
    batch, rows, width = source.shape
    incoming = indices.shape[1]
    update = torch.empty_like(remaining)
    eye = torch.eye(incoming, device=source.device, dtype=torch.float32)[None, None]
    for start in range(0, width, int(chunk_size)):
        stop = min(start + int(chunk_size), width)
        columns = stop - start
        selected = indices[:, :, start:stop]
        dictionary = torch.gather(
            source.unsqueeze(3).expand(-1, -1, -1, columns),
            2,
            selected[:, None].expand(-1, rows, -1, -1),
        ).permute(0, 3, 1, 2).contiguous()
        targets = (
            remaining[:, :, start:stop]
            .permute(0, 2, 1)
            .contiguous()
            .unsqueeze(-1)
        )
        gram = dictionary.transpose(-1, -2) @ dictionary
        rhs = dictionary.transpose(-1, -2) @ targets
        diagonal_mean = gram.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
        gram.add_(eye * (float(ridge_ratio) * diagonal_mean)[..., None, None])
        coefficients = torch.linalg.solve(gram, rhs)
        update[:, :, start:stop] = (
            (dictionary @ coefficients).squeeze(-1).permute(0, 2, 1)
        )
    return update


@torch.no_grad()
def fit_with_separate_selection_target(
    modules,
    selection: dict[int, torch.Tensor],
    fitted: dict[int, torch.Tensor],
    *,
    schedule: list[int],
    ridge_ratio: float,
    chunk_size: int,
) -> tuple[dict[int, torch.Tensor], list[torch.Tensor]]:
    """Select supports on one target while fitting another target.

    When ``selection == fitted`` this is algebraically identical to the
    production directed-product solver.  A parallel selection prediction
    prevents every stage from re-selecting against the original task target.
    """
    source = torch.stack(
        [module.weight.float().T for module in modules], dim=0
    ).contiguous()
    selection_target = torch.stack(
        [selection[layer].T.to(source.device) for layer in range(len(modules))],
        dim=0,
    ).contiguous()
    fitted_target = torch.stack(
        [fitted[layer].T.to(source.device) for layer in range(len(modules))],
        dim=0,
    ).contiguous()
    transformed = source.clone()
    selection_prediction = torch.zeros_like(selection_target)
    fitted_prediction = torch.zeros_like(fitted_target)
    supports: list[torch.Tensor] = []
    for incoming in schedule:
        selection_remaining = selection_target - selection_prediction
        row_gram = transformed @ transformed.transpose(1, 2)
        row_scale = row_gram.diagonal(dim1=1, dim2=2).mean(dim=1)
        row_gram.diagonal(dim1=1, dim2=2).add_(
            float(ridge_ratio) * row_scale[:, None]
        )
        minimum_norm_action = transformed.transpose(1, 2) @ torch.linalg.solve(
            row_gram, selection_remaining
        )
        indices = torch.topk(
            minimum_norm_action.abs(), k=int(incoming), dim=1
        ).indices
        supports.append(indices.detach().to(device="cpu", dtype=torch.int32))
        del row_gram, minimum_norm_action
        fitted_stage = _fit_on_support(
            transformed,
            fitted_target - fitted_prediction,
            indices,
            ridge_ratio=ridge_ratio,
            chunk_size=chunk_size,
        )
        selection_stage = _fit_on_support(
            transformed,
            selection_remaining,
            indices,
            ridge_ratio=ridge_ratio,
            chunk_size=chunk_size,
        )
        transformed.add_(fitted_stage)
        fitted_prediction.add_(fitted_stage)
        selection_prediction.add_(selection_stage)
    return (
        {
            layer: value.T.contiguous().cpu()
            for layer, value in enumerate(fitted_prediction)
        },
        supports,
    )


def support_overlap(
    first: list[torch.Tensor],
    second: list[torch.Tensor],
    layers: list[int],
) -> dict[str, Any]:
    if len(first) != len(second):
        raise ValueError("support stage count changed")
    stages: list[float] = []
    numerator = 0
    denominator = 0
    for left, right in zip(first, second, strict=True):
        left = left[layers].to(torch.int64)
        right = right[layers].to(torch.int64)
        matches = (left[:, :, None, :] == right[:, None, :, :]).any(dim=2)
        shared = int(matches.sum())
        total = int(left.numel())
        stages.append(shared / max(total, 1))
        numerator += shared
        denominator += total
    return {"aggregate": numerator / max(denominator, 1), "by_stage": stages}


def candidate_passes(
    name: str,
    rows: dict[str, Any],
    functional: dict[str, Any],
    rule: dict[str, Any],
) -> bool:
    if name not in PROMOTABLE:
        return False
    windows = ("fit", "holdout")
    return (
        min(
            rows[window][name]["late_action_vs_requested"][
                "positive_line_recovery"
            ]
            - rows[window]["current_corrected"]["late_action_vs_requested"][
                "positive_line_recovery"
            ]
            for window in windows
        )
        >= float(rule["minimum_late_requested_recovery_improvement"])
        and float(rows["cross_window"][name]["late_action_cosine"])
        >= float(rule["minimum_late_action_cross_window_cosine"])
        and max(
            rows[window][name]["outgoing_to_incoming_feedback_fro_ratio"]
            for window in windows
        )
        <= float(rule["maximum_outgoing_to_incoming_feedback_ratio"])
        and max(
            rows[window][name]["outgoing_to_incoming_feedback_fro_ratio"]
            - rows[window]["current_corrected"][
                "outgoing_to_incoming_feedback_fro_ratio"
            ]
            for window in windows
        )
        <= float(rule["maximum_residual_ratio_increase_over_current"])
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
    if passing:
        priority = list(rule["candidate_priority"])
        selected = min(passing, key=priority.index)
        classification = "PROMOTE_TASK_GAUGE_SELECTOR_TO_IMPLEMENTATION_MFU_GATE"
    else:
        selected = None
        task_only_better = all(
            functional[window]["task_only"]["vs_current"][
                "candidate_minus_current_mean_ce"
            ]
            < 0.0
            for window in ("fit", "holdout")
        )
        classification = (
            "TASK_ONLY_BETTER_BUT_FEEDBACK_FACTOR_UNRESOLVED"
            if task_only_better
            else "REJECT_SELECTOR_FITTER_DECOUPLING"
        )
    return {
        "classification": classification,
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
        "optimizer_stability_result_sha256": file_sha256(
            args.optimizer_stability_result
        ),
    }
    if observed != plan["identity"]:
        raise ValueError(f"selector/fitter identity mismatch: {observed}")
    parent = json.loads(args.optimizer_stability_result.read_text())
    if parent.get("decision", {}).get("classification") != (
        "MOMENTUM_STABILIZES_TASK_TANGENT"
    ):
        raise ValueError("parent result does not authorize selector/fitter audit")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--optimizer-stability-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    plan = json.loads(args.plan.read_text())
    if plan.get("schema_version") != "mai_124m_qk_cfc_selector_fitter_factorial_plan_v1":
        raise ValueError("unexpected selector/fitter plan schema")
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
        print(f"collecting and factoring window {window}", flush=True)
        mean_ce = collect_cfc_gradients(
            model, modules, batches, device=args.device, dtype=dtype
        )
        targets, _diagnostics = optimizer_targets(owner, modules)
        requested = targets["momentum_requested"]
        corrected = targets["feedback_corrected"]
        feedback = {
            layer: corrected[layer] - requested[layer] for layer in requested
        }
        raw: dict[str, dict[int, torch.Tensor]] = {}
        supports[window] = {}
        for name, selection, fitted in (
            ("current_corrected", corrected, corrected),
            ("task_only", requested, requested),
            (
                "task_select_corrected_fit_corrected_radius",
                requested,
                corrected,
            ),
            (
                "task_select_corrected_fit_requested_radius",
                requested,
                corrected,
            ),
        ):
            raw[name], supports[window][name] = fit_with_separate_selection_target(
                modules,
                selection,
                fitted,
                schedule=schedule,
                ridge_ratio=float(protocol["ridge_ratio"]),
                chunk_size=int(protocol["chunk_size"]),
            )
        production_raw, _ = fit_schedule(
            modules,
            corrected,
            schedule=schedule,
            ridge_ratio=float(protocol["ridge_ratio"]),
            chunk_size=int(protocol["chunk_size"]),
        )
        max_control_error = max(
            float((raw["current_corrected"][layer] - production_raw[layer]).abs().max())
            for layer in production_raw
        )
        if max_control_error > float(protocol["maximum_production_control_error"]):
            raise ValueError("factorial current control does not reproduce production")
        actions[window] = {
            "current_corrected": scaled_to_dense_ratio(
                raw["current_corrected"], corrected, 1.0
            ),
            "task_only": scaled_to_dense_ratio(raw["task_only"], requested, 1.0),
            "task_select_corrected_fit_corrected_radius": scaled_to_dense_ratio(
                raw["task_select_corrected_fit_corrected_radius"], corrected, 1.0
            ),
            "task_select_corrected_fit_requested_radius": scaled_to_dense_ratio(
                raw["task_select_corrected_fit_requested_radius"], requested, 1.0
            ),
        }
        rows[window] = {
            "gradient_mean_ce": mean_ce,
            "production_control_max_abs_error": max_control_error,
        }
        for name, action in actions[window].items():
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

    names = list(windows)
    first, second = names[0], names[1]
    rows["cross_window"] = {}
    for name in CANDIDATES:
        rows["cross_window"][name] = {
            "action": aggregate_direction_metrics(actions[first][name], actions[second][name]),
            "late_action_cosine": aggregate_direction_metrics(
                subset(actions[first][name], late_layers),
                subset(actions[second][name], late_layers),
            )["cosine"],
            "support_overlap_all": support_overlap(
                supports[first][name], supports[second][name], list(range(len(modules)))
            ),
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
        "factorial_rows": rows,
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
