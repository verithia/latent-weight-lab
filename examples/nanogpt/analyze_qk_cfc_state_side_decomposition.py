#!/usr/bin/env python3
"""Decompose c_fc temporal state into output-, input-, and bilateral actions."""

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
    cfc_modules,
    directed_optimizer,
)
from examples.nanogpt.analyze_mlp_cfc_exact_current_matcher import (
    file_sha256,
    git_commit,
    load_model_and_optimizer,
)
from examples.nanogpt.analyze_qk_cfc_procedural_state_codability import (
    recovery_metrics,
    summarize,
)
from examples.nanogpt.muon_matched_givens import (
    batched_multistage_directed_sparse_update,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_SCHEMA = "mai_124m_qk_cfc_state_side_decomposition_plan_v1"
RESULT_SCHEMA = "mai_124m_qk_cfc_state_side_decomposition_result_v1"
PARENT_SCHEMA = "mai_124m_qk_cfc_procedural_state_frontier_sealed_result_v1"
STATE_NAMES = ("momentum_buffer", "compression_residual")
FAMILIES = (
    "output6",
    "input_full",
    "output6_then_input",
    "input_then_output6",
)
PROMOTABLE = FAMILIES[1:]
MATRIX_SHAPE = (3072, 768)
OUTPUT_SCHEDULE = (22, 22, 22, 22, 22, 22)
LATE_LAYERS = (8, 9, 10, 11)


@torch.no_grad()
def full_input_action_projection(
    source: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Project target onto ``source @ B`` for unrestricted input action B."""
    if source.ndim != 3 or source.shape != target.shape:
        raise ValueError("source and target must be same-shaped rank-3 tensors")
    q, _r = torch.linalg.qr(source.float(), mode="reduced")
    return q @ (q.transpose(1, 2) @ target.float())


@torch.no_grad()
def output_action_fit(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    schedule: tuple[int, ...],
    ridge_ratio: float,
    chunk_size: int,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    prediction_t, rows = batched_multistage_directed_sparse_update(
        source.transpose(1, 2).contiguous(),
        target.transpose(1, 2).contiguous(),
        incoming_schedule=schedule,
        ridge_ratio=ridge_ratio,
        chunk_size=chunk_size,
    )
    return prediction_t.transpose(1, 2).contiguous(), rows


@torch.no_grad()
def candidate_predictions(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    schedule: tuple[int, ...],
    ridge_ratio: float,
    chunk_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    output, output_rows = output_action_fit(
        source,
        target,
        schedule=schedule,
        ridge_ratio=ridge_ratio,
        chunk_size=chunk_size,
    )
    input_action = full_input_action_projection(source, target)
    output_then_input = output + full_input_action_projection(
        source, target - output
    )
    input_residual_output, residual_rows = output_action_fit(
        source,
        target - input_action,
        schedule=schedule,
        ridge_ratio=ridge_ratio,
        chunk_size=chunk_size,
    )
    return {
        "output6": output,
        "input_full": input_action,
        "output6_then_input": output_then_input,
        "input_then_output6": input_action + input_residual_output,
    }, {
        "output6_stage_rows": output_rows,
        "input_residual_output6_stage_rows": residual_rows,
    }


def family_passes(
    aggregate: dict[str, Any], family: str, rule: dict[str, Any]
) -> bool:
    return all(
        float(aggregate[state_name][family]["all"]["energy_weighted_recovery"])
        >= float(rule["minimum_aggregate_recovery"])
        and float(aggregate[state_name][family]["late"]["minimum_layer_recovery"])
        >= float(rule["minimum_late_layer_recovery"])
        for state_name in STATE_NAMES
    )


def classify(aggregate: dict[str, Any], rule: dict[str, Any]) -> dict[str, Any]:
    passing = [family for family in PROMOTABLE if family_passes(aggregate, family, rule)]
    priority = list(rule["candidate_priority"])
    selected = min(passing, key=priority.index) if passing else None
    if selected == "input_full":
        classification = "INPUT_SIDE_TEMPORAL_STATE_PLAUSIBLE"
    elif selected is not None:
        classification = "BILATERAL_TEMPORAL_STATE_PLAUSIBLE"
    else:
        classification = "WEIGHT_RELATIVE_TEMPORAL_STATE_INSUFFICIENT"
    return {
        "classification": classification,
        "family_passed": {
            family: family_passes(aggregate, family, rule) for family in FAMILIES
        },
        "selected_family": selected,
        "thresholds": rule,
        "parameter_updates_to_checkpoint": 0,
    }


def validate(args: argparse.Namespace, plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unexpected state-side plan schema")
    expected = {
        "entrypoint_sha256": file_sha256(Path(__file__)),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "config_sha256": file_sha256(args.config),
        "dataset_manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
        "parent_result_sha256": file_sha256(args.parent_result),
    }
    if plan.get("identity") != expected:
        raise ValueError(f"state-side identity mismatch: {expected}")
    parent = json.loads(args.parent_result.read_text())
    if (
        parent.get("schema_version") != PARENT_SCHEMA
        or parent.get("classification")
        != "PROCEDURAL_STATE_UNREACHABLE_AT_MAXIMUM_TESTED_BUDGET"
    ):
        raise ValueError("parent result does not authorize side decomposition")
    expected_protocol = {
        "parameter_updates": 0,
        "checkpoint_next_iter": 9489,
        "state_names": list(STATE_NAMES),
        "matrix_shape": list(MATRIX_SHAPE),
        "late_layers": list(LATE_LAYERS),
        "families": list(FAMILIES),
        "output_schedule": list(OUTPUT_SCHEDULE),
        "input_action": "unrestricted_right_multiplier_W_times_B",
        "bilateral_orders": ["output6_then_input", "input_then_output6"],
        "support_selected_from_dense_answer": True,
    }
    if plan.get("protocol") != expected_protocol:
        raise ValueError("state-side protocol changed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--parent-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    plan = json.loads(args.plan.read_text())
    validate(args, plan)
    config = json.loads(args.config.read_text())
    started = time.time()
    model, optimizer, checkpoint = load_model_and_optimizer(
        args.checkpoint, config, "cpu"
    )
    if int(checkpoint["next_iter"]) != int(plan["protocol"]["checkpoint_next_iter"]):
        raise ValueError("checkpoint next_iter changed")
    modules = cfc_modules(model)
    owner = directed_optimizer(optimizer)
    reference = modules[0]
    source = torch.stack([module.weight.float() for module in modules], dim=0).to(
        args.device
    ).contiguous()
    rows: list[dict[str, Any]] = []
    solver_rows: dict[str, Any] = {}
    for state_name in STATE_NAMES:
        tensors = []
        for layer, module in enumerate(modules):
            tensor = owner.state[module.weight].get(state_name)
            if tensor is None or tuple(tensor.shape) != MATRIX_SHAPE:
                raise ValueError(f"missing or malformed {state_name} at layer {layer}")
            tensors.append(tensor.float())
        target = torch.stack(tensors, dim=0).to(args.device).contiguous()
        candidates, state_solver_rows = candidate_predictions(
            source,
            target,
            schedule=OUTPUT_SCHEDULE,
            ridge_ratio=reference.ridge_ratio,
            chunk_size=reference.chunk_size,
        )
        solver_rows[state_name] = state_solver_rows
        for family, prediction in candidates.items():
            for layer in range(len(modules)):
                rows.append(
                    {
                        "state_name": state_name,
                        "family": family,
                        "layer": layer,
                        "band": "late" if layer in LATE_LAYERS else "nonlate",
                        **recovery_metrics(target[layer], prediction[layer]),
                    }
                )
        del target
    all_layers = list(range(len(modules)))
    aggregate = {
        state_name: {
            family: {
                "all": summarize(
                    [
                        row
                        for row in rows
                        if row["state_name"] == state_name and row["family"] == family
                    ],
                    all_layers,
                ),
                "late": summarize(
                    [
                        row
                        for row in rows
                        if row["state_name"] == state_name and row["family"] == family
                    ],
                    list(LATE_LAYERS),
                ),
            }
            for family in FAMILIES
        }
        for state_name in STATE_NAMES
    }
    decision = classify(aggregate, plan["decision_rule"])
    dense_bytes = math.prod(MATRIX_SHAPE) * 4
    output_coordinates = sum(OUTPUT_SCHEDULE) * MATRIX_SHAPE[0]
    input_coordinates = MATRIX_SHAPE[1] ** 2
    result = {
        "schema_version": RESULT_SCHEMA,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "decision": decision,
        "aggregate": aggregate,
        "rows": rows,
        "solver_rows": solver_rows,
        "accounting": {
            "dense_state_bytes_per_layer": dense_bytes,
            "output6_coordinates": output_coordinates,
            "output6_minimum_bytes_with_uint16_supports": output_coordinates * 6,
            "input_full_coordinates": input_coordinates,
            "input_full_bytes": input_coordinates * 4,
            "bilateral_minimum_bytes": output_coordinates * 6 + input_coordinates * 4,
            "bilateral_minimum_byte_ratio_to_dense": (
                output_coordinates * 6 + input_coordinates * 4
            )
            / dense_bytes,
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
            "latent_native_side_design": decision["selected_family"] is not None,
            "candidate_implementation": False,
            "mfu_preflight": False,
            "language_model_training": False,
            "larger_rung": False,
        },
    }
    args.output.mkdir(parents=True)
    path = args.output / "result.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
