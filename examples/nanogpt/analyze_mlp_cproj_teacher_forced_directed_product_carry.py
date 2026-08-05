#!/usr/bin/env python3
"""Replay equal-budget non-orthogonal c_proj output products with full carry."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.nanogpt.analyze_mlp_cproj_teacher_forced_bilateral_full_carry import (
    fit_output_pass,
)
from examples.nanogpt.analyze_mlp_cproj_teacher_forced_bilateral_replay import (
    cell_metrics,
    cosine_lr,
    file_sha256,
    fit_right_pass,
    git_commit,
    load_snapshot,
)
from examples.nanogpt.analyze_parameter_trajectory import write_csv


PLAN_SCHEMA = "mai_124m_mlp_cproj_teacher_forced_directed_product_carry_plan_v1"
OUTPUT_COORDINATES = 12_288


@dataclass(frozen=True)
class Arm:
    name: str
    output_kind: str
    incoming_by_stage: tuple[int, ...] = ()


ARMS = (
    Arm("hidden88_full_carry", "none"),
    Arm("hidden88_output32_full_carry", "orthogonal"),
    Arm("hidden88_directed16_full_carry", "directed", (16,)),
    Arm("hidden88_directed8x2_full_carry", "directed", (8, 8)),
)
CONTROL = ARMS[1].name
CANDIDATES = tuple(arm.name for arm in ARMS[2:])


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unexpected directed-product carry plan schema")
    analysis = plan.get("analysis", {})
    expected = {
        "parameter_updates": 0,
        "teacher_forced_dense_updates": True,
        "layers": [0, 3, 6, 9, 11],
        "phase_boundaries": [0, 60, 120, 180, 238],
        "feedback_decay": 1.0,
        "output_coordinate_budget_per_layer": OUTPUT_COORDINATES,
    }
    observed = {key: analysis.get(key) for key in expected}
    if observed != expected:
        raise ValueError("directed-product carry plan does not match v1 contract")
    arms = analysis.get("arms_in_smallest_pass_order", [])
    if [arm.get("name") for arm in arms] != [arm.name for arm in ARMS]:
        raise ValueError("directed-product arm order mismatch")
    if analysis.get("directed_fit", {}).get("relative_ridge") != 1e-6:
        raise ValueError("directed-product ridge mismatch")
    if plan.get("authorization", {}).get("implement_and_run_zero_update_analysis") is not True:
        raise ValueError("zero-update directed-product analysis is not authorized")
    for key in (
        "bilateral_full_carry_result",
        "bilateral_fixed_eval_result",
        "hybrid_causal_result",
        "nominal_cap_result",
    ):
        path = REPO_ROOT / plan["identity"][key]
        if file_sha256(path) != plan["identity"][f"{key}_sha256"]:
            raise ValueError(f"identity hash mismatch for {key}")


def _directed_stage(
    weight: torch.Tensor,
    residual: torch.Tensor,
    *,
    incoming: int,
    relative_ridge: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Fit one sparse general output action ``delta W = B W``."""
    if weight.ndim != 2 or residual.shape != weight.shape:
        raise ValueError("weight and residual must be equal two-dimensional tensors")
    outputs = int(weight.shape[0])
    if incoming <= 0 or incoming > outputs:
        raise ValueError("invalid incoming support width")
    source = weight.T.contiguous().float()
    target = residual.T.contiguous().float()
    energy = source.square().sum(dim=0).clamp_min(1e-30)
    ridge_scalar = float(relative_ridge) * float(energy.mean())
    cross = source.T @ target
    beta = cross / (energy[:, None] + ridge_scalar)
    score = 2.0 * beta * cross - beta.square() * energy[:, None]
    supports = torch.topk(score, k=incoming, dim=0, largest=True, sorted=True).indices
    selected = source[:, supports.T].permute(1, 0, 2).contiguous()
    targets = target.T.unsqueeze(-1)
    gram = selected.transpose(1, 2).double() @ selected.double()
    rhs = selected.transpose(1, 2).double() @ targets.double()
    ridge = (
        float(relative_ridge)
        * torch.diagonal(gram, dim1=-2, dim2=-1).mean(dim=1).clamp_min(1e-30)
    )
    eye = torch.eye(incoming, device=gram.device, dtype=gram.dtype).unsqueeze(0)
    coefficients = torch.linalg.solve(
        gram + ridge[:, None, None] * eye, rhs
    ).squeeze(-1).float()
    mapping_t = torch.zeros(
        outputs, outputs, device=weight.device, dtype=torch.float32
    )
    target_indices = torch.arange(outputs, device=weight.device).unsqueeze(1)
    target_indices = target_indices.expand(-1, incoming)
    mapping_t[supports.T, target_indices] = coefficients
    delta = (source @ mapping_t).T.contiguous()
    updated = weight.float() + delta
    left_action = mapping_t.T.contiguous()
    diagnostics = {
        "incoming": float(incoming),
        "coordinate_count": float(outputs * incoming),
        "ridge_min": float(ridge.min()),
        "ridge_max": float(ridge.max()),
        "raw_delta_energy": float(delta.double().square().sum()),
        "left_action_fro": float(left_action.double().norm()),
    }
    if not torch.isfinite(updated).all() or not all(
        math.isfinite(value) for value in diagnostics.values()
    ):
        raise ValueError("directed output stage produced nonfinite values")
    return updated, left_action, diagnostics


def fit_directed_product(
    weight: torch.Tensor,
    residual: torch.Tensor,
    *,
    incoming_by_stage: tuple[int, ...],
    trust_output_energy: float,
    relative_ridge: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Fit a sequential sparse output product and apply one uniform trust scale."""
    base = weight.float()
    current = base
    remaining = residual.float()
    outputs = int(base.shape[0])
    effective = torch.eye(outputs, device=base.device, dtype=torch.float32)
    coordinate_count = 0
    minimum_ridge = float("inf")
    maximum_ridge = 0.0
    for incoming in incoming_by_stage:
        updated, left_action, stage = _directed_stage(
            current,
            remaining,
            incoming=incoming,
            relative_ridge=relative_ridge,
        )
        remaining = remaining - (updated - current)
        current = updated
        effective = (torch.eye(outputs, device=base.device) + left_action) @ effective
        coordinate_count += int(stage["coordinate_count"])
        minimum_ridge = min(minimum_ridge, stage["ridge_min"])
        maximum_ridge = max(maximum_ridge, stage["ridge_max"])
    raw_delta = current - base
    raw_energy = float(raw_delta.double().square().sum())
    energy_scale = min(
        1.0, math.sqrt(float(trust_output_energy) / max(raw_energy, 1e-30))
    )
    effective_delta = effective - torch.eye(outputs, device=base.device)
    action_fro = float(effective_delta.double().norm())
    invertibility_scale = min(1.0, 0.05 / max(action_fro, 1e-30))
    scale = min(energy_scale, invertibility_scale)
    bounded_delta = raw_delta * scale
    updated = base + bounded_delta
    bounded_energy = float(bounded_delta.double().square().sum())
    singular_lower_bound = 1.0 - scale * action_fro
    diagnostics = {
        "coordinate_count": float(coordinate_count),
        "raw_output_delta_energy": raw_energy,
        "bounded_output_delta_energy": bounded_energy,
        "trust_output_energy": float(trust_output_energy),
        "trust_scale": scale,
        "energy_scale": energy_scale,
        "invertibility_scale": invertibility_scale,
        "effective_action_fro": scale * action_fro,
        "minimum_singular_value_lower_bound": singular_lower_bound,
        "minimum_ridge": minimum_ridge,
        "maximum_ridge": maximum_ridge,
    }
    if (
        coordinate_count != outputs * sum(incoming_by_stage)
        or bounded_energy > trust_output_energy + max(1e-12, 1e-5 * trust_output_energy)
        or singular_lower_bound < 0.95 - 1e-7
        or not torch.isfinite(updated).all()
        or not all(math.isfinite(value) for value in diagnostics.values())
    ):
        raise ValueError("directed output product violated its frozen contract")
    return updated, diagnostics


def structured_step(
    weight: torch.Tensor,
    requested_update: torch.Tensor,
    feedback: torch.Tensor,
    *,
    arm: Arm,
    learning_rate: float,
    weight_decay: float,
    neighbors: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, float, dict[str, float] | None]:
    corrected = requested_update.float() + feedback.float()
    current = weight.float()
    residual = corrected
    for pass_index, stages in enumerate((64, 24)):
        updated = fit_right_pass(
            current,
            residual,
            stages=stages,
            neighbors=neighbors,
            seed=seed + pass_index,
        )
        residual = residual - (updated - current)
        current = updated
    chart: dict[str, float] | None = None
    if arm.output_kind == "orthogonal":
        updated = fit_output_pass(
            current,
            residual,
            stages=32,
            neighbors=neighbors,
            seed=seed + 2,
        )
        residual = residual - (updated - current)
        current = updated
    elif arm.output_kind == "directed":
        orthogonal = fit_output_pass(
            current,
            residual,
            stages=32,
            neighbors=neighbors,
            seed=seed + 2,
        )
        trust = float((orthogonal - current).double().square().sum())
        updated, chart = fit_directed_product(
            current,
            residual,
            incoming_by_stage=arm.incoming_by_stage,
            trust_output_energy=trust,
        )
        residual = residual - (updated - current)
        current = updated
    if weight_decay:
        current = current * (1.0 - learning_rate * weight_decay)
    actual = current - weight.float()
    new_feedback = corrected - actual
    energy = requested_update.float().square().sum().clamp_min(1e-30)
    recovery = float(1.0 - (requested_update.float() - actual).square().sum() / energy)
    return current, new_feedback.contiguous(), recovery, chart


def aggregate_rows(
    rows: list[dict[str, Any]], chart_rows: list[dict[str, Any]], plan: dict[str, Any]
) -> dict[str, Any]:
    scores: dict[str, dict[str, Any]] = {}
    steps = sorted({int(row["score_step"]) for row in rows})
    for arm in ARMS:
        for step in steps:
            selected = [
                row for row in rows
                if row["arm"] == arm.name and int(row["score_step"]) == step
            ]
            chord_energy = sum(float(row["chord_energy"]) for row in selected)
            endpoint_error = sum(float(row["endpoint_error_energy"]) for row in selected)
            gram_energy = sum(float(row["row_gram_chord_energy"]) for row in selected)
            gram_error = sum(float(row["row_gram_error_energy"]) for row in selected)
            scores[f"{arm.name}@{step}"] = {
                "endpoint_recovery": 1.0 - endpoint_error / max(chord_energy, 1e-30),
                "row_gram_recovery": 1.0 - gram_error / max(gram_energy, 1e-30),
                "endpoint_error_energy": endpoint_error,
                "row_gram_error_energy": gram_error,
                "maximum_feedback_fro": max(float(row["terminal_feedback_fro"]) for row in selected),
                "mean_requested_update_recovery": sum(float(row["mean_requested_update_recovery"]) for row in selected) / len(selected),
                "all_finite": all(
                    all(math.isfinite(float(value)) for key, value in row.items() if key not in {"arm", "layer", "score_step"})
                    for row in selected
                ),
            }
    terminal = steps[-1]
    control = scores[f"{CONTROL}@{terminal}"]
    rule = plan["decision_rule"]["candidate_pass_requirements"]
    comparisons: dict[str, Any] = {}
    for candidate in CANDIDATES:
        candidate_terminal = scores[f"{candidate}@{terminal}"]
        terminal_candidate_cells = {
            int(row["layer"]): row for row in rows
            if row["arm"] == candidate and int(row["score_step"]) == terminal
        }
        terminal_control_cells = {
            int(row["layer"]): row for row in rows
            if row["arm"] == CONTROL and int(row["score_step"]) == terminal
        }
        charts = [row for row in chart_rows if row["arm"] == candidate]
        gate = {
            "all_finite": all(scores[f"{candidate}@{step}"]["all_finite"] for step in steps),
            "equal_coordinate_budget": sorted({int(row["coordinate_count"]) for row in charts}) == [OUTPUT_COORDINATES],
            "trust_obeyed": all(float(row["bounded_output_delta_energy"]) <= float(row["trust_output_energy"]) + max(1e-12, 1e-5 * float(row["trust_output_energy"])) for row in charts),
            "minimum_singular_bound": min(float(row["minimum_singular_value_lower_bound"]) for row in charts) >= 0.95,
            "endpoint_no_worse_every_score": all(float(scores[f"{candidate}@{step}"]["endpoint_recovery"]) >= float(scores[f"{CONTROL}@{step}"]["endpoint_recovery"]) for step in steps),
            "terminal_endpoint": float(candidate_terminal["endpoint_recovery"]) >= float(rule["terminal_endpoint_recovery_minimum"]),
            "terminal_row_gram": float(candidate_terminal["row_gram_recovery"]) >= float(rule["terminal_row_gram_recovery_minimum"]),
            "terminal_feedback": float(candidate_terminal["maximum_feedback_fro"]) / max(float(control["maximum_feedback_fro"]), 1e-30) <= float(rule["terminal_feedback_ratio_to_output32_maximum"]),
            "terminal_layers": sum(float(terminal_candidate_cells[layer]["endpoint_recovery"]) > float(terminal_control_cells[layer]["endpoint_recovery"]) for layer in terminal_candidate_cells) >= int(rule["terminal_layers_beating_output32_minimum"]),
            "requested_update": float(candidate_terminal["mean_requested_update_recovery"]) >= float(control["mean_requested_update_recovery"]),
        }
        comparisons[candidate] = {
            "gate": gate,
            "passed": all(gate.values()),
            "terminal_feedback_ratio_to_output32": float(candidate_terminal["maximum_feedback_fro"]) / max(float(control["maximum_feedback_fro"]), 1e-30),
            "terminal_layers_beating_output32": sum(float(terminal_candidate_cells[layer]["endpoint_recovery"]) > float(terminal_control_cells[layer]["endpoint_recovery"]) for layer in terminal_candidate_cells),
            "minimum_singular_value_lower_bound": min(float(row["minimum_singular_value_lower_bound"]) for row in charts),
            "minimum_trust_scale": min(float(row["trust_scale"]) for row in charts),
        }
    selected = next((name for name in CANDIDATES if comparisons[name]["passed"]), None)
    return {
        "scores": scores,
        "comparisons": comparisons,
        "selected_arm": selected,
        "decision": "DIRECTED_OUTPUT_PRODUCT_PASS" if selected else "REJECT_DIRECTED_OUTPUT_PRODUCT_CARRY",
        "authorization": {
            "fixed_eval_endpoint_oracle_authorized": selected is not None,
            "language_model_training_authorized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", default="0,3,6,9,11")
    parser.add_argument("--score-steps", default="60,120,180,238")
    parser.add_argument("--neighbors", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    validate_plan(plan)
    layers = [int(value) for value in args.layers.split(",")]
    score_steps = [int(value) for value in args.score_steps.split(",")]
    if layers != plan["analysis"]["layers"] or score_steps != [60, 120, 180, 238]:
        raise ValueError("runtime layer or score-step contract mismatch")
    paths = [args.snapshot_dir / f"step_{step:06d}.pt" for step in range(239)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"missing {len(missing)} snapshots; first={missing[0]}")
    first = load_snapshot(paths[0])
    identity = first["run_identity_sha256"]
    if identity != plan["identity"]["trajectory_run_identity_sha256"]:
        raise ValueError("trajectory identity mismatch")
    config = first["run_identity"]["resolved_config"]
    if config.get("data_manifest_sha256") != plan["identity"]["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest mismatch")
    names = [f"transformer.h.{layer}.mlp.c_proj.weight" for layer in layers]
    starts = {layer: first["parameters"][name].to(args.device).float() for layer, name in zip(layers, names, strict=True)}
    dense_previous = {layer: value.clone() for layer, value in starts.items()}
    states = {(arm.name, layer): starts[layer].clone() for arm in ARMS for layer in layers}
    feedback = {key: torch.zeros_like(value) for key, value in states.items()}
    recoveries: dict[tuple[str, int], list[float]] = {key: [] for key in states}
    rows: list[dict[str, Any]] = []
    chart_rows: list[dict[str, Any]] = []
    for step in range(238):
        payload = load_snapshot(paths[step + 1])
        if payload["run_identity_sha256"] != identity:
            raise ValueError(f"trajectory identity mismatch at {step + 1}")
        lr = cosine_lr(step, learning_rate=float(config["learning_rate"]), min_lr=float(config["min_lr"]), warmup_iters=int(config["warmup_iters"]), decay_iters=int(config["lr_decay_iters"]))
        weight_decay = float(config["weight_decay"])
        for layer, name in zip(layers, names, strict=True):
            dense_before = dense_previous[layer]
            dense_after = payload["parameters"][name].to(args.device).float()
            dense_delta = dense_after - dense_before
            dense_nondecay = dense_delta + lr * weight_decay * dense_before
            for arm in ARMS:
                key = (arm.name, layer)
                candidate = states[key]
                requested = dense_nondecay - lr * weight_decay * candidate
                updated, new_feedback, recovery, chart = structured_step(candidate, requested, feedback[key], arm=arm, learning_rate=lr, weight_decay=weight_decay, neighbors=args.neighbors, seed=args.seed + layer * 100000 + step * 10)
                states[key] = updated
                feedback[key] = new_feedback
                recoveries[key].append(recovery)
                if chart is not None:
                    chart_rows.append({"arm": arm.name, "layer": layer, "step": step + 1, **chart})
            dense_previous[layer] = dense_after
        score_step = step + 1
        if score_step in score_steps:
            for layer in layers:
                for arm in ARMS:
                    key = (arm.name, layer)
                    row = {"arm": arm.name, "layer": layer, "score_step": score_step, **cell_metrics(starts[layer], dense_previous[layer], states[key], recoveries[key], feedback[key])}
                    rows.append(row)
                    print(json.dumps(row, sort_keys=True), flush=True)
        elif score_step == 1 or score_step % 10 == 0:
            print(json.dumps({"step": score_step}), flush=True)
        del payload
    aggregate = aggregate_rows(rows, chart_rows, plan)
    args.output.mkdir(parents=True, exist_ok=True)
    cells_path = args.output / "cproj_directed_product_carry_cells.csv"
    chart_path = args.output / "cproj_directed_product_carry_chart.csv"
    aggregate_path = args.output / "cproj_directed_product_carry_result.json"
    write_csv(cells_path, rows)
    write_csv(chart_path, chart_rows)
    aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_cproj_directed_product_carry_v1",
        "run_identity_sha256": identity,
        "layers": layers,
        "score_steps": score_steps,
        "snapshot_inventory": {"count": len(paths), "first_sha256": file_sha256(paths[0]), "last_sha256": file_sha256(paths[-1]), "total_bytes": sum(path.stat().st_size for path in paths)},
        "plan": {"path": str(args.plan), "sha256": file_sha256(args.plan)},
        "analysis_execution": {"git_commit": git_commit(REPO_ROOT), "entrypoint": str(script), "entrypoint_sha256": file_sha256(script), "command": sys.argv, "device": args.device, "started_at_unix": started, "finished_at_unix": time.time()},
        "outputs": {"cells_sha256": file_sha256(cells_path), "chart_sha256": file_sha256(chart_path), "aggregate_sha256": file_sha256(aggregate_path)},
        "limitations": ["Teacher-forced dense updates make this an optimistic representation oracle.", "No language-model parameter update or task-loss evaluation is performed."],
    }
    metadata_path = args.output / "cproj_directed_product_carry_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": aggregate["decision"], "selected_arm": aggregate["selected_arm"], "metadata": str(metadata_path)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
