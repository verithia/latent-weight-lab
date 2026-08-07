#!/usr/bin/env python3
"""Test whether c_fc's actual procedural family can encode its dense state.

The generic low-rank/sparse oracle rejected the terminal Muon momentum and
error-feedback tensors.  Those are not the only dense-output families with a
compact code, so this zero-update audit grants the exact production
directed-product solver direct access to each terminal state tensor.  Support
indices are ignored in the budget, making this an optimistic codability gate.
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
from examples.nanogpt.muon_matched_givens import (
    batched_multistage_directed_sparse_update,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_SCHEMA = "mai_124m_qk_cfc_procedural_state_codability_plan_v1"
RESULT_SCHEMA = "mai_124m_qk_cfc_procedural_state_codability_result_v1"
PARENT_SCHEMA = "mai_124m_qk_cfc_optimizer_state_budget_sealed_result_v1"
STATE_NAMES = ("momentum_buffer", "compression_residual")
MATRIX_SHAPE = (3072, 768)
COORDINATE_BUDGET = 405_504


def recovery_metrics(
    target: torch.Tensor, prediction: torch.Tensor
) -> dict[str, float]:
    target = target.double()
    prediction = prediction.double()
    if target.shape != prediction.shape:
        raise ValueError("target and prediction shapes differ")
    target_energy = target.square().sum().clamp_min(1e-30)
    prediction_energy = prediction.square().sum()
    residual_energy = (target - prediction).square().sum()
    inner = (target * prediction).sum()
    return {
        "target_energy": float(target_energy),
        "prediction_energy": float(prediction_energy),
        "residual_energy": float(residual_energy),
        "energy_recovery": float(1.0 - residual_energy / target_energy),
        "cosine": float(
            inner
            / (target_energy.sqrt() * prediction_energy.sqrt()).clamp_min(1e-30)
        ),
        "positive_line_recovery": float(
            torch.clamp_min(inner, 0.0).square()
            / (target_energy * prediction_energy).clamp_min(1e-30)
        ),
    }


def summarize(rows: list[dict[str, Any]], layers: list[int]) -> dict[str, Any]:
    selected = [row for row in rows if int(row["layer"]) in layers]
    total = sum(float(row["target_energy"]) for row in selected)
    return {
        "layers": layers,
        "energy_weighted_recovery": sum(
            float(row["target_energy"]) * float(row["energy_recovery"])
            for row in selected
        )
        / max(total, 1e-30),
        "energy_weighted_positive_line_recovery": sum(
            float(row["target_energy"])
            * float(row["positive_line_recovery"])
            for row in selected
        )
        / max(total, 1e-30),
        "minimum_layer_recovery": min(
            float(row["energy_recovery"]) for row in selected
        ),
        "minimum_layer_positive_line_recovery": min(
            float(row["positive_line_recovery"]) for row in selected
        ),
    }


def classify(aggregate: dict[str, Any], rule: dict[str, Any]) -> dict[str, Any]:
    aggregate_floor = float(rule["minimum_aggregate_recovery"])
    late_floor = float(rule["minimum_late_layer_recovery"])
    gates = {
        state_name: (
            float(aggregate[state_name]["all"]["energy_weighted_recovery"])
            >= aggregate_floor
            and float(aggregate[state_name]["late"]["minimum_layer_recovery"])
            >= late_floor
        )
        for state_name in STATE_NAMES
    }
    passed = all(gates.values())
    return {
        "classification": (
            "PROCEDURAL_CFC_STATE_CODE_PLAUSIBLE"
            if passed
            else "CFC_PROCEDURAL_STATE_CODE_REJECTED"
        ),
        "state_family_passed": gates,
        "all_state_families_passed": passed,
        "thresholds": rule,
        "parameter_updates_to_checkpoint": 0,
        "interpretation_boundary": (
            "An oracle terminal-state fit in the exact production procedural "
            "family. Supports are selected from the dense answer and their "
            "storage cost is ignored. A pass authorizes only a paired temporal "
            "acquisition; a failure closes this family as a compact state code."
        ),
    }


def validate(args: argparse.Namespace, plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unexpected procedural-state plan schema")
    expected = {
        "entrypoint_sha256": file_sha256(Path(__file__)),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "config_sha256": file_sha256(args.config),
        "dataset_manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
        "parent_result_sha256": file_sha256(args.parent_result),
    }
    if plan.get("identity") != expected:
        raise ValueError(f"procedural-state identity mismatch: {expected}")
    parent = json.loads(args.parent_result.read_text())
    if (
        parent.get("schema_version") != PARENT_SCHEMA
        or parent.get("classification") != "CFC_TEMPORAL_STATE_IS_DENSE_SCALE"
    ):
        raise ValueError("parent result does not authorize procedural-state audit")
    expected_protocol = {
        "parameter_updates": 0,
        "checkpoint_next_iter": 9489,
        "state_names": list(STATE_NAMES),
        "matrix_shape": list(MATRIX_SHAPE),
        "coordinate_budget_per_layer_per_state": COORDINATE_BUDGET,
        "late_layers": [8, 9, 10, 11],
        "production_incoming_schedule": [22, 22, 22, 22, 22, 22],
        "support_index_cost_ignored": True,
        "support_selected_from_dense_answer": True,
    }
    if plan.get("protocol") != expected_protocol:
        raise ValueError("procedural-state protocol changed")


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
    if list(reference.incoming_schedule) != plan["protocol"][
        "production_incoming_schedule"
    ]:
        raise ValueError("production incoming schedule changed")
    if reference.coordinate_count != COORDINATE_BUDGET:
        raise ValueError("production coordinate budget changed")

    source = torch.stack(
        [module.weight.float().T for module in modules], dim=0
    ).to(args.device).contiguous()
    rows: list[dict[str, Any]] = []
    stage_rows: dict[str, Any] = {}
    for state_name in STATE_NAMES:
        tensors = []
        for layer, module in enumerate(modules):
            tensor = owner.state[module.weight].get(state_name)
            if tensor is None or tuple(tensor.shape) != MATRIX_SHAPE:
                raise ValueError(f"missing or malformed {state_name} at layer {layer}")
            tensors.append(tensor.float().T)
        target = torch.stack(tensors, dim=0).to(args.device).contiguous()
        prediction, stages = batched_multistage_directed_sparse_update(
            source,
            target,
            incoming_schedule=reference.incoming_schedule,
            ridge_ratio=reference.ridge_ratio,
            chunk_size=reference.chunk_size,
        )
        stage_rows[state_name] = stages
        for layer in range(len(modules)):
            metrics = recovery_metrics(target[layer], prediction[layer])
            rows.append(
                {
                    "state_name": state_name,
                    "layer": layer,
                    "band": "late" if layer >= 8 else "nonlate",
                    **metrics,
                }
            )
            print(
                f"state={state_name} layer={layer} "
                f"recovery={metrics['energy_recovery']:.6f} "
                f"cosine={metrics['cosine']:.6f}",
                flush=True,
            )
        del target, prediction

    all_layers = list(range(len(modules)))
    late_layers = [int(value) for value in plan["protocol"]["late_layers"]]
    aggregate = {
        state_name: {
            "all": summarize(
                [row for row in rows if row["state_name"] == state_name],
                all_layers,
            ),
            "late": summarize(
                [row for row in rows if row["state_name"] == state_name],
                late_layers,
            ),
        }
        for state_name in STATE_NAMES
    }
    decision = classify(aggregate, plan["decision_rule"])
    dense_per_state = math.prod(MATRIX_SHAPE)
    result = {
        "schema_version": RESULT_SCHEMA,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "decision": decision,
        "aggregate": aggregate,
        "rows": rows,
        "stage_rows": stage_rows,
        "accounting": {
            "dense_elements_per_layer_per_state": dense_per_state,
            "coordinates_per_layer_per_state": COORDINATE_BUDGET,
            "coordinate_fraction_of_dense": COORDINATE_BUDGET / dense_per_state,
            "support_indices_per_layer_per_state": COORDINATE_BUDGET,
            "support_index_cost_ignored": True,
            "dense_elements_all_layers_both_states": (
                dense_per_state * len(modules) * len(STATE_NAMES)
            ),
            "coordinates_all_layers_both_states": (
                COORDINATE_BUDGET * len(modules) * len(STATE_NAMES)
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
            "paired_temporal_state_acquisition": decision[
                "all_state_families_passed"
            ],
            "functional_compact_state_replay": False,
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
