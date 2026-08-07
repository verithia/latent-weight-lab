#!/usr/bin/env python3
"""Audit whether the useful c_fc temporal state is compact at 20TPP.

The accepted directed-product c_fc path still owns dense Muon momentum and
error-feedback tensors.  This zero-update audit gives both state families the
entire procedural-coordinate budget, then measures two optimistic explicit
representations: an oracle low-rank factorization and oracle unstructured
sparsity.  Passing this gate only authorizes a later functional replay; it
does not authorize training or claim that the oracle representation is causal.
"""

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


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_SCHEMA = "mai_124m_qk_cfc_optimizer_state_budget_plan_v1"
RESULT_SCHEMA = "mai_124m_qk_cfc_optimizer_state_budget_result_v1"
STATE_NAMES = ("momentum_buffer", "compression_residual")
MATRIX_SHAPE = (3072, 768)
COORDINATE_BUDGET = 405_504
ENERGY_THRESHOLDS = (0.50, 0.80, 0.90, 0.95, 0.99)


def intrinsic_rank_dimension(rank: int, rows: int, columns: int) -> int:
    if rank < 0 or rank > min(rows, columns):
        raise ValueError("rank outside matrix dimensions")
    return rank * (rows + columns - rank)


def largest_rank_within_budget(budget: int, rows: int, columns: int) -> int:
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    return max(
        rank
        for rank in range(min(rows, columns) + 1)
        if intrinsic_rank_dimension(rank, rows, columns) <= budget
    )


def rank_for_energy(eigenvalues: torch.Tensor, fraction: float) -> int:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("energy fraction must lie in (0, 1]")
    values = eigenvalues.double().flatten().clamp_min(0)
    total = values.sum().clamp_min(1e-30)
    return int(torch.searchsorted(values.cumsum(0), fraction * total).item() + 1)


def participation_rank(energy: torch.Tensor) -> float:
    values = energy.double().flatten().clamp_min(0)
    probabilities = values / values.sum().clamp_min(1e-30)
    return float(1.0 / probabilities.square().sum().clamp_min(1e-30))


def state_budget_metrics(
    state: torch.Tensor,
    *,
    coordinate_budget: int = COORDINATE_BUDGET,
) -> dict[str, float | int]:
    state = state.detach().float()
    if tuple(state.shape) != MATRIX_SHAPE:
        raise ValueError(f"unexpected state shape: {tuple(state.shape)}")
    rows, columns = state.shape
    energy = state.double().square()
    total = energy.sum().clamp_min(1e-30)
    gram = state.transpose(0, 1).double() @ state.double()
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0).flip(0)
    rank_budget = largest_rank_within_budget(coordinate_budget, rows, columns)
    sparse_budget = min(coordinate_budget, state.numel())
    sparse_energy = torch.topk(energy.flatten(), sparse_budget, sorted=False).values.sum()
    result: dict[str, float | int] = {
        "rows": rows,
        "columns": columns,
        "dense_elements": state.numel(),
        "state_fro": float(total.sqrt()),
        "coordinate_budget": coordinate_budget,
        "coordinate_budget_fraction_of_dense": coordinate_budget / state.numel(),
        "equal_budget_intrinsic_rank": rank_budget,
        "equal_budget_low_rank_recovery": float(eigenvalues[:rank_budget].sum() / total),
        "equal_budget_sparse_recovery_ignoring_indices": float(sparse_energy / total),
        "row_energy_participation_rank": participation_rank(energy.sum(dim=1)),
        "column_energy_participation_rank": participation_rank(energy.sum(dim=0)),
    }
    for fraction in ENERGY_THRESHOLDS:
        label = f"rank_{round(100 * fraction):02d}pct"
        rank = rank_for_energy(eigenvalues, fraction)
        dof = intrinsic_rank_dimension(rank, rows, columns)
        result[label] = rank
        result[f"{label}_intrinsic_dof"] = dof
        result[f"{label}_intrinsic_dof_ratio_to_dense"] = dof / state.numel()
        result[f"{label}_over_coordinate_budget"] = dof / coordinate_budget
    return result


def energy_weighted(rows: list[dict[str, Any]], field: str) -> float:
    total = sum(float(row["state_energy"]) for row in rows)
    return sum(float(row["state_energy"]) * float(row[field]) for row in rows) / max(
        total, 1e-30
    )


def summarize(rows: list[dict[str, Any]], layers: list[int]) -> dict[str, Any]:
    selected = [row for row in rows if int(row["layer"]) in layers]
    return {
        "layers": layers,
        "energy_weighted_low_rank_recovery": energy_weighted(
            selected, "equal_budget_low_rank_recovery"
        ),
        "energy_weighted_sparse_recovery_ignoring_indices": energy_weighted(
            selected, "equal_budget_sparse_recovery_ignoring_indices"
        ),
        "minimum_layer_low_rank_recovery": min(
            float(row["equal_budget_low_rank_recovery"]) for row in selected
        ),
        "minimum_layer_sparse_recovery_ignoring_indices": min(
            float(row["equal_budget_sparse_recovery_ignoring_indices"])
            for row in selected
        ),
        "energy_weighted_rank_80pct": energy_weighted(selected, "rank_80pct"),
        "energy_weighted_rank80_intrinsic_dof_ratio_to_dense": energy_weighted(
            selected, "rank_80pct_intrinsic_dof_ratio_to_dense"
        ),
    }


def classify(aggregate: dict[str, Any], rule: dict[str, Any]) -> dict[str, Any]:
    recovery = float(rule["minimum_aggregate_recovery"])
    late = float(rule["minimum_late_layer_recovery"])
    family_gates: dict[str, Any] = {}
    for state_name in STATE_NAMES:
        metrics = aggregate[state_name]
        low_rank = (
            metrics["all"]["energy_weighted_low_rank_recovery"] >= recovery
            and metrics["late"]["minimum_layer_low_rank_recovery"] >= late
        )
        sparse = (
            metrics["all"]["energy_weighted_sparse_recovery_ignoring_indices"]
            >= recovery
            and metrics["late"]["minimum_layer_sparse_recovery_ignoring_indices"]
            >= late
        )
        family_gates[state_name] = {
            "oracle_low_rank_passed": low_rank,
            "oracle_sparse_passed_ignoring_indices": sparse,
            "some_explicit_family_passed": low_rank or sparse,
        }
    all_compact = all(
        value["some_explicit_family_passed"] for value in family_gates.values()
    )
    return {
        "classification": (
            "EXPLICIT_COMPACT_CFC_TEMPORAL_STATE_PLAUSIBLE"
            if all_compact
            else "CFC_TEMPORAL_STATE_IS_DENSE_SCALE"
        ),
        "family_gates": family_gates,
        "all_state_families_compact": all_compact,
        "thresholds": rule,
        "parameter_updates_to_checkpoint": 0,
        "interpretation_boundary": (
            "Oracle terminal-state representation only. A pass requires a later "
            "same-checkpoint functional replay; a failure closes naive explicit "
            "low-rank and unstructured-sparse state, not every nonlinear temporal coder."
        ),
    }


def validate(args: argparse.Namespace, plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unexpected optimizer-state-budget plan schema")
    expected = {
        "entrypoint_sha256": file_sha256(Path(__file__)),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "config_sha256": file_sha256(args.config),
        "dataset_manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
        "parent_result_sha256": file_sha256(args.parent_result),
    }
    if plan.get("identity") != expected:
        raise ValueError(f"optimizer-state-budget identity mismatch: {expected}")
    parent = json.loads(args.parent_result.read_text())
    if parent.get("classification") != "REJECT_NORM_BALANCED_FEEDBACK":
        raise ValueError("parent result does not authorize state-budget audit")
    protocol = plan.get("protocol", {})
    if protocol != {
        "parameter_updates": 0,
        "checkpoint_next_iter": 9489,
        "state_names": list(STATE_NAMES),
        "matrix_shape": list(MATRIX_SHAPE),
        "coordinate_budget_per_layer_per_state": COORDINATE_BUDGET,
        "energy_thresholds": list(ENERGY_THRESHOLDS),
        "late_layers": [8, 9, 10, 11],
        "oracle_sparse_index_cost_ignored": True,
    }:
        raise ValueError("optimizer-state-budget protocol changed")


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
    rows: list[dict[str, Any]] = []
    for layer, module in enumerate(modules):
        state = owner.state[module.weight]
        for state_name in STATE_NAMES:
            tensor = state.get(state_name)
            if tensor is None:
                raise ValueError(f"missing {state_name} for layer {layer}")
            metrics = state_budget_metrics(tensor.to(args.device))
            rows.append(
                {
                    "state_name": state_name,
                    "layer": layer,
                    "band": "late" if layer >= 8 else "nonlate",
                    "state_energy": float(tensor.double().square().sum()),
                    **metrics,
                }
            )
            print(
                f"state={state_name} layer={layer} "
                f"rank_recovery={metrics['equal_budget_low_rank_recovery']:.6f} "
                f"sparse_recovery={metrics['equal_budget_sparse_recovery_ignoring_indices']:.6f}",
                flush=True,
            )
    all_layers = list(range(len(modules)))
    late_layers = [int(value) for value in plan["protocol"]["late_layers"]]
    aggregate = {
        state_name: {
            "all": summarize(
                [row for row in rows if row["state_name"] == state_name], all_layers
            ),
            "late": summarize(
                [row for row in rows if row["state_name"] == state_name], late_layers
            ),
        }
        for state_name in STATE_NAMES
    }
    decision = classify(aggregate, plan["decision_rule"])
    result = {
        "schema_version": RESULT_SCHEMA,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "decision": decision,
        "aggregate": aggregate,
        "rows": rows,
        "accounting": {
            "dense_elements_per_layer_per_state": math.prod(MATRIX_SHAPE),
            "dense_elements_all_layers_both_states": (
                math.prod(MATRIX_SHAPE) * len(modules) * len(STATE_NAMES)
            ),
            "oracle_coordinate_budget_all_layers_both_states": (
                COORDINATE_BUDGET * len(modules) * len(STATE_NAMES)
            ),
            "oracle_budget_fraction_of_dense_state": (
                COORDINATE_BUDGET / math.prod(MATRIX_SHAPE)
            ),
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
            "functional_compact_state_replay": decision["all_state_families_compact"],
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
