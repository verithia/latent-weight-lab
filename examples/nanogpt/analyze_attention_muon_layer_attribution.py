#!/usr/bin/env python3
"""Attribute V and attention c_proj Muon-orbit fit across layers and phases.

The early 5-TPP probes select the best half of layers for the already-tested
target-specific Cayley charts.  The late probes are held out in time.  This is
a zero-update necessary-condition test for functional LWT, not a training
result and not an authorization to fit a learned dense basis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_SCHEMA = "mai_124m_attention_muon_layer_attribution_plan_v1"
RESULT_SCHEMA = "mai_124m_attention_muon_layer_attribution_result_v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def positive_line_recovery(target: torch.Tensor, candidate: torch.Tensor) -> float:
    target = target.double()
    candidate = candidate.double()
    dot = (target * candidate).sum().clamp_min(0.0)
    denominator = (
        target.square().sum().clamp_min(1e-30)
        * candidate.square().sum().clamp_min(1e-30)
    )
    return float(dot.square() / denominator)


def orbit_recoveries(
    weight: torch.Tensor,
    direction: torch.Tensor,
    *,
    side: str,
    active_rank: int,
) -> tuple[float, float]:
    weight = weight.float()
    direction = direction.float()
    if side == "left":
        skew = 0.5 * (
            direction @ weight.transpose(0, 1)
            - weight @ direction.transpose(0, 1)
        )
        full = skew @ weight
    elif side == "right":
        skew = 0.5 * (
            weight.transpose(0, 1) @ direction
            - direction.transpose(0, 1) @ weight
        )
        full = weight @ skew
    else:
        raise ValueError(f"unsupported orbit side: {side}")
    u, singular, vh = torch.linalg.svd(skew, full_matrices=False)
    rank = min(active_rank, singular.numel())
    truncated_skew = (u[:, :rank] * singular[:rank]) @ vh[:rank]
    truncated = (
        truncated_skew @ weight if side == "left" else weight @ truncated_skew
    )
    return (
        positive_line_recovery(direction, full),
        positive_line_recovery(direction, truncated),
    )


def rank_values(values: dict[int, float]) -> dict[int, float]:
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    return {layer: float(index) for index, (layer, _value) in enumerate(ordered)}


def spearman(first: dict[int, float], second: dict[int, float]) -> float:
    if set(first) != set(second) or len(first) < 2:
        raise ValueError("rank dictionaries must share at least two keys")
    first_rank = rank_values(first)
    second_rank = rank_values(second)
    x = torch.tensor([first_rank[key] for key in sorted(first)], dtype=torch.float64)
    y = torch.tensor([second_rank[key] for key in sorted(first)], dtype=torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    return float((x @ y) / (x.norm() * y.norm()).clamp_min(1e-30))


def weighted_recovery(rows: list[dict[str, Any]]) -> float:
    numerator = sum(float(row["direction_energy"]) * float(row["active_recovery"]) for row in rows)
    denominator = sum(float(row["direction_energy"]) for row in rows)
    return numerator / max(denominator, 1e-30)


def summarize_target(
    rows: list[dict[str, Any]],
    *,
    early_steps: set[int],
    late_steps: set[int],
    selected_layers: int,
    gates: dict[str, float],
) -> dict[str, Any]:
    layers = sorted({int(row["layer"]) for row in rows})
    if selected_layers <= 0 or selected_layers >= len(layers):
        raise ValueError("selected_layers must be between zero and layer count")

    def phase_scores(steps: set[int]) -> dict[int, float]:
        return {
            layer: weighted_recovery(
                [row for row in rows if int(row["layer"]) == layer and int(row["step"]) in steps]
            )
            for layer in layers
        }

    early = phase_scores(early_steps)
    late = phase_scores(late_steps)
    early_selected = sorted(early, key=lambda layer: (-early[layer], layer))[:selected_layers]
    late_selected = sorted(late, key=lambda layer: (-late[layer], layer))[:selected_layers]
    early_set = set(early_selected)
    late_set = set(late_selected)
    intersection = len(early_set & late_set)
    union = len(early_set | late_set)
    selected_late_rows = [
        row for row in rows
        if int(row["layer"]) in early_set and int(row["step"]) in late_steps
    ]
    unselected_late_rows = [
        row for row in rows
        if int(row["layer"]) not in early_set and int(row["step"]) in late_steps
    ]
    selected_recovery = weighted_recovery(selected_late_rows)
    unselected_recovery = weighted_recovery(unselected_late_rows)
    fraction_of_full = sum(
        float(row["direction_energy"]) * float(row["active_recovery"])
        / max(float(row["full_orbit_recovery"]), 1e-30)
        for row in selected_late_rows
    ) / max(sum(float(row["direction_energy"]) for row in selected_late_rows), 1e-30)
    metrics = {
        "early_selected_layers": early_selected,
        "late_selected_layers": late_selected,
        "top_half_jaccard": intersection / max(union, 1),
        "early_late_spearman": spearman(early, late),
        "late_selected_active_recovery": selected_recovery,
        "late_unselected_active_recovery": unselected_recovery,
        "late_selected_over_unselected": selected_recovery / max(unselected_recovery, 1e-30),
        "late_selected_active_fraction_of_full_orbit": fraction_of_full,
        "early_recovery_by_layer": {str(key): early[key] for key in layers},
        "late_recovery_by_layer": {str(key): late[key] for key in layers},
    }
    checks = {
        "jaccard": metrics["top_half_jaccard"] >= float(gates["minimum_jaccard"]),
        "spearman": metrics["early_late_spearman"] >= float(gates["minimum_spearman"]),
        "separation": metrics["late_selected_over_unselected"] >= float(gates["minimum_separation"]),
        "absolute_recovery": metrics["late_selected_active_recovery"] >= float(gates["minimum_selected_recovery"]),
        "fraction_of_full_orbit": metrics["late_selected_active_fraction_of_full_orbit"] >= float(gates["minimum_fraction_of_full_orbit"]),
    }
    return {**metrics, "checks": checks, "passed": all(checks.values())}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    plan = json.loads(args.plan.read_text())
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unexpected plan schema")
    if file_sha256(Path(__file__)) != plan["identity"]["entrypoint_sha256"]:
        raise ValueError("entrypoint hash does not match plan")
    if args.output_dir.exists():
        raise FileExistsError(f"output already exists: {args.output_dir}")
    protocol = plan["protocol"]
    steps = [int(value) for value in protocol["steps"]]
    layers = [int(value) for value in protocol["layers"]]
    probe_paths = [args.probe_dir / f"step_{step:06d}.pt" for step in steps]
    expected = plan["identity"]["probe_sha256"]
    for path in probe_paths:
        if not path.is_file() or file_sha256(path) != expected[path.name]:
            raise ValueError(f"probe identity mismatch: {path}")

    target_specs = protocol["targets"]
    rows: list[dict[str, Any]] = []
    run_identity = None
    for path in probe_paths:
        probe = torch.load(path, map_location="cpu", weights_only=False)
        if run_identity is None:
            run_identity = probe["run_identity_sha256"]
        elif probe["run_identity_sha256"] != run_identity:
            raise ValueError("optimizer probes do not share one run identity")
        n_embd = int(probe["model_config"]["n_embd"])
        for layer in layers:
            for target, spec in target_specs.items():
                parameter = probe["parameters"][
                    f"transformer.h.{layer}.{spec['parameter']}"
                ]
                weight = parameter["weight_before_step"]
                direction = parameter["applied_direction_per_lr"]
                if target == "v":
                    weight = weight[2 * n_embd :]
                    direction = direction[2 * n_embd :]
                weight = weight.to(args.device)
                direction = direction.to(args.device)
                full, active = orbit_recoveries(
                    weight,
                    direction,
                    side=str(spec["side"]),
                    active_rank=int(spec["active_skew_rank"]),
                )
                rows.append(
                    {
                        "step": int(probe["step"]),
                        "layer": layer,
                        "target": target,
                        "side": spec["side"],
                        "active_skew_rank": int(spec["active_skew_rank"]),
                        "direction_energy": float(direction.double().square().sum()),
                        "full_orbit_recovery": full,
                        "active_recovery": active,
                        "active_fraction_of_full_orbit": active / max(full, 1e-30),
                    }
                )
                del weight, direction
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    summaries = {
        target: summarize_target(
            [row for row in rows if row["target"] == target],
            early_steps=set(int(value) for value in protocol["early_steps"]),
            late_steps=set(int(value) for value in protocol["late_steps"]),
            selected_layers=int(protocol["selected_layers"]),
            gates=plan["decision_rule"]["targets"][target],
        )
        for target in target_specs
    }
    passed = [target for target, summary in summaries.items() if summary["passed"]]
    classification = (
        "AUTHORIZE_HELDOUT_FUNCTIONAL_LWT_GATE"
        if passed
        else "REJECT_ATTENTION_LAYERWISE_ORBIT_ATTRIBUTION"
    )
    args.output_dir.mkdir(parents=True)
    cells_path = args.output_dir / "attention_muon_layer_attribution_cells.csv"
    write_csv(cells_path, rows)
    result = {
        "schema_version": RESULT_SCHEMA,
        "recorded_at": "2026-08-08",
        "classification": classification,
        "identity": {
            "source_commit": git_commit(),
            "entrypoint_sha256": file_sha256(Path(__file__)),
            "plan_sha256": file_sha256(args.plan),
            "probe_run_identity_sha256": run_identity,
            "probe_sha256": expected,
        },
        "protocol": protocol,
        "summaries": summaries,
        "decision": {
            "passed_targets": passed,
            "language_model_training_authorized": False,
            "mfu_preflight_authorized": False,
            "next_action": (
                "Run a preregistered same-gauge held-out one-step functional CE gate only for passed targets."
                if passed
                else "Keep V and attention c_proj dense; no LWT training is authorized from this branch."
            ),
        },
        "limitations": [
            "This is an oracle on dense Muon-applied directions, not a trained compact model.",
            "Early and late probes are held out in time but share the same training run.",
            "A pass authorizes only a same-gauge functional gate, never direct training.",
        ],
        "artifacts": {
            "cells_csv": str(cells_path),
            "cells_csv_sha256": file_sha256(cells_path),
        },
        "elapsed_seconds": time.time() - started,
        "parameter_updates": 0,
    }
    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"classification": classification, "passed_targets": passed}, sort_keys=True))


if __name__ == "__main__":
    main()
