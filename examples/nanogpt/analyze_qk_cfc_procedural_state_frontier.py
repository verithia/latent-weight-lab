#!/usr/bin/env python3
"""Measure c_fc temporal-state recovery versus procedural coordinate budget."""

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
from examples.nanogpt.muon_matched_givens import (
    batched_multistage_directed_sparse_update,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_SCHEMA = "mai_124m_qk_cfc_procedural_state_frontier_plan_v1"
RESULT_SCHEMA = "mai_124m_qk_cfc_procedural_state_frontier_result_v1"
PARENT_SCHEMA = "mai_124m_qk_cfc_procedural_state_codability_sealed_result_v1"
STATE_NAMES = ("momentum_buffer", "compression_residual")
MATRIX_SHAPE = (3072, 768)
INCOMING_PER_STAGE = 22
STAGE_COUNTS = (6, 12, 18, 24, 30)
LATE_LAYERS = (8, 9, 10, 11)


def coordinate_count(stage_count: int) -> int:
    return int(stage_count) * INCOMING_PER_STAGE * MATRIX_SHAPE[0]


def minimum_stored_byte_ratio(stage_count: int) -> float:
    """FP32 coefficients plus uint16 supports versus one FP32 dense state."""
    code_bytes = coordinate_count(stage_count) * (4 + 2)
    dense_bytes = math.prod(MATRIX_SHAPE) * 4
    return code_bytes / dense_bytes


def summarize_stage(
    recoveries: list[float], energies: list[float], layers: list[int]
) -> dict[str, Any]:
    total = sum(energies[layer] for layer in layers)
    return {
        "layers": layers,
        "energy_weighted_recovery": sum(
            energies[layer] * recoveries[layer] for layer in layers
        )
        / max(total, 1e-30),
        "minimum_layer_recovery": min(recoveries[layer] for layer in layers),
    }


def stage_passes(row: dict[str, Any], rule: dict[str, Any]) -> bool:
    return (
        float(row["all"]["energy_weighted_recovery"])
        >= float(rule["minimum_aggregate_recovery"])
        and float(row["late"]["minimum_layer_recovery"])
        >= float(rule["minimum_late_layer_recovery"])
    )


def classify(frontier: dict[str, list[dict[str, Any]]], rule: dict[str, Any]) -> dict[str, Any]:
    first_passing: dict[str, int | None] = {}
    for state_name in STATE_NAMES:
        matches = [
            int(row["stage_count"])
            for row in frontier[state_name]
            if stage_passes(row, rule)
        ]
        first_passing[state_name] = min(matches) if matches else None
    if all(value is not None for value in first_passing.values()):
        joint = max(int(value) for value in first_passing.values() if value is not None)
    else:
        joint = None
    credible = joint is not None and joint <= int(rule["maximum_credible_stage_count"])
    if credible:
        classification = "MODERATE_PROCEDURAL_STATE_BUDGET_PLAUSIBLE"
    elif joint is not None:
        classification = "PROCEDURAL_STATE_REQUIRES_DENSE_SCALE_BUDGET"
    else:
        classification = "PROCEDURAL_STATE_UNREACHABLE_AT_MAXIMUM_TESTED_BUDGET"
    return {
        "classification": classification,
        "first_passing_stage_by_state": first_passing,
        "joint_first_passing_stage": joint,
        "credible_compression_budget": credible,
        "thresholds": rule,
        "parameter_updates_to_checkpoint": 0,
    }


def validate(args: argparse.Namespace, plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unexpected procedural-state-frontier plan schema")
    expected = {
        "entrypoint_sha256": file_sha256(Path(__file__)),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "config_sha256": file_sha256(args.config),
        "dataset_manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
        "parent_result_sha256": file_sha256(args.parent_result),
    }
    if plan.get("identity") != expected:
        raise ValueError(f"procedural-state-frontier identity mismatch: {expected}")
    parent = json.loads(args.parent_result.read_text())
    if (
        parent.get("schema_version") != PARENT_SCHEMA
        or parent.get("classification") != "CFC_PROCEDURAL_STATE_CODE_REJECTED"
    ):
        raise ValueError("parent result does not authorize frontier audit")
    expected_protocol = {
        "parameter_updates": 0,
        "checkpoint_next_iter": 9489,
        "state_names": list(STATE_NAMES),
        "matrix_shape": list(MATRIX_SHAPE),
        "late_layers": list(LATE_LAYERS),
        "incoming_per_stage": INCOMING_PER_STAGE,
        "stage_counts": list(STAGE_COUNTS),
        "support_selected_from_dense_answer": True,
        "fit_support_index_cost_ignored": True,
        "accounting_support_dtype": "uint16",
        "accounting_coefficient_dtype": "float32",
    }
    if plan.get("protocol") != expected_protocol:
        raise ValueError("procedural-state-frontier protocol changed")


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
    source = torch.stack(
        [module.weight.float().T for module in modules], dim=0
    ).to(args.device).contiguous()
    frontier: dict[str, list[dict[str, Any]]] = {}
    raw_stage_rows: dict[str, Any] = {}
    for state_name in STATE_NAMES:
        tensors = []
        energies = []
        for layer, module in enumerate(modules):
            tensor = owner.state[module.weight].get(state_name)
            if tensor is None or tuple(tensor.shape) != MATRIX_SHAPE:
                raise ValueError(f"missing or malformed {state_name} at layer {layer}")
            tensors.append(tensor.float().T)
            energies.append(float(tensor.double().square().sum()))
        target = torch.stack(tensors, dim=0).to(args.device).contiguous()
        _prediction, stages = batched_multistage_directed_sparse_update(
            source,
            target,
            incoming_schedule=[INCOMING_PER_STAGE] * max(STAGE_COUNTS),
            ridge_ratio=reference.ridge_ratio,
            chunk_size=reference.chunk_size,
        )
        raw_stage_rows[state_name] = stages
        rows = []
        for stage_count in STAGE_COUNTS:
            recoveries = [float(value) for value in stages[stage_count - 1]["member_target_recovery"]]
            coordinates = coordinate_count(stage_count)
            row = {
                "stage_count": stage_count,
                "incoming_per_stage": INCOMING_PER_STAGE,
                "coordinates_per_layer_per_state": coordinates,
                "coordinate_fraction_of_dense": coordinates / math.prod(MATRIX_SHAPE),
                "minimum_stored_byte_ratio_with_uint16_supports": minimum_stored_byte_ratio(stage_count),
                "all": summarize_stage(recoveries, energies, list(range(len(modules)))),
                "late": summarize_stage(recoveries, energies, list(LATE_LAYERS)),
                "member_recovery": recoveries,
            }
            rows.append(row)
            print(
                f"state={state_name} stages={stage_count} "
                f"all={row['all']['energy_weighted_recovery']:.6f} "
                f"late={row['late']['energy_weighted_recovery']:.6f} "
                f"late_min={row['late']['minimum_layer_recovery']:.6f}",
                flush=True,
            )
        frontier[state_name] = rows
        del target, _prediction
    decision = classify(frontier, plan["decision_rule"])
    result = {
        "schema_version": RESULT_SCHEMA,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "decision": decision,
        "frontier": frontier,
        "raw_stage_rows": raw_stage_rows,
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
            "wider_procedural_state_candidate": decision["credible_compression_budget"],
            "paired_temporal_state_acquisition": False,
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
