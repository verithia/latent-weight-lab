#!/usr/bin/env python3
"""Quantify oracle low-rank state needed by the dense c_proj residual.

This is a zero-update post-acquisition analysis.  It reconstructs the exact
same hidden64+24+output32 chart used by the accepted temporal-residual
diagnostic, then asks how many *task-selected* singular directions are needed
to retain fixed fractions of the remaining action.  The SVD is an oracle: the
result bounds explicit low-rank/learned-basis state only and is not evidence
that a causal nonlinear generator needs the same number of coordinates.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.nanogpt.analyze_mlp_activation_update_alignment import load_snapshot
from examples.nanogpt.analyze_mlp_cproj_activation_weighted_output_selector import (
    file_sha256,
    fit_frobenius_pass,
    git_commit,
    load_probe,
    parameter_name,
    shared_hidden_chart,
)
from examples.nanogpt.analyze_mlp_cproj_diagonal_kfac_selector import (
    acquisition_artifact_hashes,
    require_full_state_snapshot,
)
from examples.nanogpt.analyze_mlp_cproj_temporal_residual import LAYERS, PHASES
from examples.nanogpt.analyze_parameter_trajectory import write_csv


PLAN_SCHEMA = "mai_124m_mlp_cproj_residual_state_budget_plan_v1"
RESULT_SCHEMA = "mai_124m_mlp_cproj_residual_state_budget_result_v1"
ENERGY_THRESHOLDS = (0.50, 0.80, 0.90, 0.95, 0.99)
MATRIX_SHAPE = (768, 3072)
CHART_BUDGET = 147456


def intrinsic_rank_dimension(rank: int, rows: int, columns: int) -> int:
    """Dimension of the rank-r matrix manifold, r(m+n-r)."""
    if rank < 0 or rank > min(rows, columns):
        raise ValueError("rank outside matrix dimensions")
    return rank * (rows + columns - rank)


def largest_rank_within_budget(budget: int, rows: int, columns: int) -> int:
    valid = [
        rank
        for rank in range(min(rows, columns) + 1)
        if intrinsic_rank_dimension(rank, rows, columns) <= budget
    ]
    return max(valid)


def rank_for_energy(eigenvalues: torch.Tensor, fraction: float) -> int:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("energy fraction must lie in (0, 1]")
    values = eigenvalues.double().flatten().clamp_min(0)
    total = values.sum().clamp_min(1e-30)
    return int(torch.searchsorted(values.cumsum(0), fraction * total).item() + 1)


def spectrum_budget_metrics(
    residual: torch.Tensor,
    *,
    thresholds: tuple[float, ...] = ENERGY_THRESHOLDS,
    coordinate_budget: int = CHART_BUDGET,
) -> dict[str, float | int]:
    residual = residual.double()
    rows, columns = residual.shape
    eigenvalues = torch.linalg.eigvalsh(residual @ residual.T).clamp_min(0).flip(0)
    total = eigenvalues.sum().clamp_min(1e-30)
    budget_rank = largest_rank_within_budget(coordinate_budget, rows, columns)
    result: dict[str, float | int] = {
        "matrix_rows": rows,
        "matrix_columns": columns,
        "dense_elements": rows * columns,
        "coordinate_budget": coordinate_budget,
        "equal_budget_intrinsic_rank": budget_rank,
        "equal_budget_factor_rank": coordinate_budget // (rows + columns),
        "equal_budget_best_rank_recovery": float(
            eigenvalues[:budget_rank].sum() / total
        ),
    }
    for fraction in thresholds:
        label = f"rank_{int(round(100 * fraction))}pct"
        rank = rank_for_energy(eigenvalues, fraction)
        intrinsic = intrinsic_rank_dimension(rank, rows, columns)
        factor = rank * (rows + columns)
        result[label] = rank
        result[f"{label}_intrinsic_dof"] = intrinsic
        result[f"{label}_intrinsic_dof_ratio"] = intrinsic / (rows * columns)
        result[f"{label}_factor_scalars"] = factor
        result[f"{label}_factor_scalar_ratio"] = factor / (rows * columns)
        result[f"{label}_over_chart_budget"] = intrinsic / coordinate_budget
    return result


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unexpected residual-state-budget plan schema")
    analysis = plan.get("analysis", {})
    expected = {
        "parameter_updates": 0,
        "layers": list(LAYERS),
        "phases": [list(pair) for pair in PHASES],
        "matrix_shape": list(MATRIX_SHAPE),
        "energy_thresholds": list(ENERGY_THRESHOLDS),
        "chart_coordinate_budget_per_layer": CHART_BUDGET,
        "chart": {
            "hidden_parent_stages": 64,
            "hidden_residual_stages": 24,
            "output_stages": 32,
            "neighbors": 64,
            "matching_seed": 20260806,
            "weight_decay_application": "identical production ordering",
        },
    }
    if analysis != expected:
        raise ValueError("residual-state-budget analysis contract changed")
    thresholds = plan.get("decision_rule", {}).get("thresholds", {})
    if thresholds != {
        "equal_budget_best_rank_recovery_minimum": 0.80,
        "rank80_intrinsic_dof_ratio_maximum": 0.25,
    }:
        raise ValueError("residual-state-budget decision thresholds changed")
    authorization = plan.get("authorization", {})
    if authorization.get("run_zero_update_state_budget_analysis") is not True:
        raise ValueError("state-budget analysis is not authorized")
    for key in (
        "implement_candidate_structure",
        "run_exact_config_mfu",
        "run_language_model_training",
        "larger_rung",
    ):
        if authorization.get(key) is not False:
            raise ValueError(f"plan must keep {key} false")


def classify(aggregate: dict[str, float], thresholds: dict[str, float]) -> dict[str, Any]:
    gates = {
        "equal_budget_low_rank_sufficient": aggregate[
            "equal_budget_best_rank_recovery"
        ]
        >= thresholds["equal_budget_best_rank_recovery_minimum"],
        "rank80_at_most_quarter_dense": aggregate[
            "rank80_intrinsic_dof_ratio"
        ]
        <= thresholds["rank80_intrinsic_dof_ratio_maximum"],
    }
    if gates["equal_budget_low_rank_sufficient"]:
        classification = "EQUAL_BUDGET_LOW_RANK_STATE_SUFFICIENT"
    elif gates["rank80_at_most_quarter_dense"]:
        classification = "LARGER_EXPLICIT_LOW_RANK_STATE_REQUIRED"
    else:
        classification = "EXPLICIT_LOW_RANK_STATE_IS_DENSE_SCALE"
    return {
        "classification": classification,
        "gates": gates,
        "authorization": {
            "task_conditioned_procedural_selector_theory": not gates[
                "equal_budget_low_rank_sufficient"
            ],
            "explicit_low_rank_candidate": gates[
                "equal_budget_low_rank_sufficient"
            ],
            "implement_candidate_structure": False,
            "run_exact_config_mfu": False,
            "run_language_model_training": False,
            "larger_rung": False,
        },
    }


def weighted_mean(rows: list[dict[str, Any]], field: str) -> float:
    total = sum(float(row["residual_energy"]) for row in rows)
    return sum(
        float(row[field]) * float(row["residual_energy"]) for row in rows
    ) / max(total, 1e-30)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--acquisition-result", type=Path, required=True)
    parser.add_argument("--temporal-result", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists() or args.rows.exists():
        raise FileExistsError("residual-state-budget output already exists")

    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    identity = plan["identity"]
    for path, key in (
        (args.acquisition_result, "acquisition_result_sha256"),
        (args.temporal_result, "temporal_result_sha256"),
        (Path(__file__), "analyzer_sha256"),
    ):
        if file_sha256(path) != identity[key]:
            raise ValueError(f"pinned artifact SHA-256 mismatch: {path}")

    acquisition = json.loads(args.acquisition_result.read_text())
    temporal = json.loads(args.temporal_result.read_text())
    if acquisition.get("classification") != (
        "ACCEPTED_PARENT_EQUIVALENT_EXACT_FUNCTIONAL_REPLAY"
    ):
        raise ValueError("acquisition is not accepted")
    if temporal.get("classification") != "LOW_DIMENSIONAL_BUT_PHASE_TRANSPORTED":
        raise ValueError("temporal residual result is not the accepted parent")
    run_identity = acquisition["identity"]["run_identity_sha256"]
    if run_identity != identity["run_identity_sha256"]:
        raise ValueError("run identity mismatch")

    snapshot_hashes = acquisition_artifact_hashes(acquisition, "snapshots")
    probe_hashes = acquisition_artifact_hashes(acquisition, "optimizer_probes")
    weights: dict[int, dict[int, torch.Tensor]] = {}
    for step in sorted({value for phase in PHASES for value in phase}):
        path = args.snapshot_dir / f"step_{step:06d}.pt"
        if file_sha256(path) != snapshot_hashes[str(step)]:
            raise ValueError(f"snapshot SHA-256 mismatch at step {step}")
        snapshot = load_snapshot(path)
        require_full_state_snapshot(snapshot)
        if snapshot["run_identity_sha256"] != run_identity:
            raise ValueError("snapshot run identity mismatch")
        weights[step] = {
            layer: snapshot["parameters"][parameter_name(layer)].float().clone()
            for layer in LAYERS
        }

    rows: list[dict[str, Any]] = []
    chart = plan["analysis"]["chart"]
    started = time.time()
    for phase_index, (start, end) in enumerate(PHASES):
        probe_path = args.probe_dir / f"step_{start:06d}.pt"
        if file_sha256(probe_path) != probe_hashes[str(start)]:
            raise ValueError(f"probe SHA-256 mismatch at step {start}")
        probe = load_probe(probe_path, start, run_identity)
        for layer in LAYERS:
            name = parameter_name(layer)
            state = probe["parameters"][name]
            hyper = probe["hyperparameters"][name]
            weight = weights[start][layer].to(args.device)
            torch.testing.assert_close(
                state["weight_before_step"], weight.cpu(), rtol=0.0, atol=0.0
            )
            learning_rate = float(hyper["lr"])
            weight_decay = float(hyper["weight_decay"])
            applied_per_lr = state["applied_direction_per_lr"].to(args.device)
            exact_update = learning_rate * applied_per_lr
            matching_direction = applied_per_lr + weight_decay * weight
            seed = int(chart["matching_seed"]) + layer * 100000 + phase_index * 10
            hidden_weight, output_residual, hidden_diagnostics = shared_hidden_chart(
                weight,
                exact_update,
                matching_direction,
                parent_stages=int(chart["hidden_parent_stages"]),
                residual_stages=int(chart["hidden_residual_stages"]),
                neighbors=int(chart["neighbors"]),
                seed=seed,
            )
            fitted, output_diagnostics = fit_frobenius_pass(
                hidden_weight.T.contiguous(),
                output_residual.T.contiguous(),
                stages=int(chart["output_stages"]),
                neighbors=int(chart["neighbors"]),
                seed=seed + 2,
            )
            coordinates = sum(
                int(value["coordinates"]) for value in hidden_diagnostics
            ) + int(output_diagnostics["coordinates"])
            if coordinates != CHART_BUDGET:
                raise ValueError("chart coordinate budget mismatch")
            final_weight = fitted.T.contiguous() * (
                1.0 - learning_rate * weight_decay
            )
            residual = exact_update - (final_weight - weight)
            if tuple(residual.shape) != MATRIX_SHAPE:
                raise ValueError("c_proj matrix shape changed")
            rows.append(
                {
                    "phase_start": start,
                    "phase_end": end,
                    "layer": layer,
                    "residual_energy": float(residual.double().square().sum()),
                    **spectrum_budget_metrics(residual),
                }
            )

    aggregate: dict[str, float] = {}
    aggregate_fields = {
        "equal_budget_best_rank_recovery": "equal_budget_best_rank_recovery",
        "rank50": "rank_50pct",
        "rank80": "rank_80pct",
        "rank90": "rank_90pct",
        "rank95": "rank_95pct",
        "rank99": "rank_99pct",
        "rank50_intrinsic_dof_ratio": "rank_50pct_intrinsic_dof_ratio",
        "rank80_intrinsic_dof_ratio": "rank_80pct_intrinsic_dof_ratio",
        "rank90_intrinsic_dof_ratio": "rank_90pct_intrinsic_dof_ratio",
        "rank95_intrinsic_dof_ratio": "rank_95pct_intrinsic_dof_ratio",
        "rank99_intrinsic_dof_ratio": "rank_99pct_intrinsic_dof_ratio",
        "rank50_over_chart_budget": "rank_50pct_over_chart_budget",
        "rank80_over_chart_budget": "rank_80pct_over_chart_budget",
        "rank90_over_chart_budget": "rank_90pct_over_chart_budget",
        "rank95_over_chart_budget": "rank_95pct_over_chart_budget",
        "rank99_over_chart_budget": "rank_99pct_over_chart_budget",
    }
    for output_field, row_field in aggregate_fields.items():
        aggregate[output_field] = weighted_mean(rows, row_field)
    aggregate["minimum_rank80"] = min(float(row["rank_80pct"]) for row in rows)
    aggregate["maximum_rank80"] = max(float(row["rank_80pct"]) for row in rows)
    aggregate["equal_budget_intrinsic_rank"] = float(
        rows[0]["equal_budget_intrinsic_rank"]
    )
    aggregate["dense_elements_per_layer"] = float(MATRIX_SHAPE[0] * MATRIX_SHAPE[1])
    aggregate["chart_coordinate_budget_per_layer"] = float(CHART_BUDGET)
    decision = classify(aggregate, plan["decision_rule"]["thresholds"])

    output = {
        "schema_version": RESULT_SCHEMA,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "classification": decision["classification"],
        "execution": {
            "host": "PRO6",
            "device": args.device,
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": "examples.nanogpt.analyze_mlp_cproj_residual_state_budget",
            "parameter_updates": 0,
            "elapsed_seconds": time.time() - started,
            "direct_foreground_polling": True,
            "watchdog": False,
            "callback": False,
        },
        "identity": {
            "plan_sha256": file_sha256(args.plan),
            "acquisition_result_sha256": file_sha256(args.acquisition_result),
            "temporal_result_sha256": file_sha256(args.temporal_result),
            "run_identity_sha256": run_identity,
        },
        "aggregate": aggregate,
        "cell_ranges": {
            field: {
                "minimum": min(float(row[field]) for row in rows),
                "maximum": max(float(row[field]) for row in rows),
            }
            for field in (
                "equal_budget_best_rank_recovery",
                "rank_50pct",
                "rank_80pct",
                "rank_90pct",
                "rank_95pct",
                "rank_99pct",
                "rank_80pct_intrinsic_dof_ratio",
                "rank_80pct_over_chart_budget",
            )
        },
        "decision": decision,
        "limitations": [
            "SVD support is selected from the exact residual and is therefore an oracle.",
            "Rank-manifold dimension bounds explicit factor or learned-basis state, not an arbitrary nonlinear procedural generator.",
            "No causal coordinate selection, optimizer update, model update, validation selection, or architecture change is performed.",
        ],
    }
    if not all(math.isfinite(float(value)) for value in aggregate.values()):
        raise ValueError("non-finite aggregate metric")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.rows, rows)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
