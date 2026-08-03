#!/usr/bin/env python3
"""Bracket terminal c_fc directed-product depth and coordinate capacity."""

from __future__ import annotations

import argparse
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
    prospective_updates,
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
from examples.nanogpt.muon_matched_givens import (
    batched_multistage_directed_sparse_update,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "nanogpt_mlp_cfc_terminal_capacity_v1"


def schedule_coordinates(schedule: list[int], output_channels: int) -> int:
    return sum(int(value) for value in schedule) * int(output_channels)


@torch.no_grad()
def fit_schedule(
    modules,
    dense: dict[int, torch.Tensor],
    *,
    schedule: list[int],
    ridge_ratio: float,
    chunk_size: int,
) -> tuple[dict[int, torch.Tensor], list[dict[str, Any]]]:
    source = torch.stack(
        [module.weight.float().T for module in modules], dim=0
    ).contiguous()
    target = torch.stack(
        [dense[layer].T.to(source.device) for layer in range(len(modules))],
        dim=0,
    ).contiguous()
    raw, stages = batched_multistage_directed_sparse_update(
        source,
        target,
        incoming_schedule=schedule,
        ridge_ratio=ridge_ratio,
        chunk_size=chunk_size,
    )
    return (
        {layer: value.T.contiguous().cpu() for layer, value in enumerate(raw)},
        stages,
    )


def direction_row(
    dense: dict[int, torch.Tensor],
    raw: dict[int, torch.Tensor],
    *,
    gradient_seed: int,
    name: str,
    schedule: list[int],
    radius_ratio: float,
    output_channels: int,
    stage_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    scaled = scaled_to_dense_ratio(raw, dense, radius_ratio)
    metrics = aggregate_direction_metrics(dense, scaled)
    per_layer = [
        aggregate_direction_metrics(
            {layer: dense[layer]}, {layer: scaled[layer]}
        )
        for layer in sorted(dense)
    ]
    radius_error = abs(
        family_fro(scaled) / max(family_fro(dense), 1e-30) - radius_ratio
    )
    return {
        "gradient_seed": int(gradient_seed),
        "candidate": name,
        "incoming_schedule": schedule,
        "stages": len(schedule),
        "coordinates_per_layer": schedule_coordinates(
            schedule, output_channels
        ),
        **metrics,
        "minimum_layer_positive_line_recovery": min(
            row["positive_line_recovery"] for row in per_layer
        ),
        "maximum_layer_positive_line_recovery": max(
            row["positive_line_recovery"] for row in per_layer
        ),
        "radius_ratio_absolute_error": radius_error,
        "stage_mean_target_cosine": [
            sum(float(value) for value in stage["member_target_cosine"])
            / len(stage["member_target_cosine"])
            for stage in stage_rows
        ],
    }


def classify_capacity(
    rows: list[dict[str, Any]],
    *,
    current_name: str,
    maximum_coordinates_per_layer: int,
    minimum_positive_line_recovery: float,
    minimum_layer_positive_line_recovery: float,
    minimum_improvement_over_current: float,
    maximum_radius_error: float,
) -> dict[str, Any]:
    names = sorted({str(row["candidate"]) for row in rows})
    seeds = sorted({int(row["gradient_seed"]) for row in rows})
    indexed = {
        (str(row["candidate"]), int(row["gradient_seed"])): row
        for row in rows
    }
    if any((current_name, seed) not in indexed for seed in seeds):
        raise ValueError("current control is incomplete")
    summaries: dict[str, dict[str, Any]] = {}
    for name in names:
        candidate_rows = [indexed[(name, seed)] for seed in seeds]
        improvements = [
            float(candidate["positive_line_recovery"])
            - float(indexed[(current_name, seed)]["positive_line_recovery"])
            for candidate, seed in zip(candidate_rows, seeds, strict=True)
        ]
        summary = {
            "coordinates_per_layer": int(
                candidate_rows[0]["coordinates_per_layer"]
            ),
            "minimum_positive_line_recovery": min(
                float(row["positive_line_recovery"])
                for row in candidate_rows
            ),
            "mean_positive_line_recovery": sum(
                float(row["positive_line_recovery"])
                for row in candidate_rows
            )
            / len(candidate_rows),
            "minimum_layer_positive_line_recovery": min(
                float(row["minimum_layer_positive_line_recovery"])
                for row in candidate_rows
            ),
            "minimum_improvement_over_current": min(improvements),
            "maximum_radius_ratio_absolute_error": max(
                float(row["radius_ratio_absolute_error"])
                for row in candidate_rows
            ),
        }
        summary["passes"] = (
            name != current_name
            and summary["coordinates_per_layer"]
            <= int(maximum_coordinates_per_layer)
            and summary["minimum_positive_line_recovery"]
            >= float(minimum_positive_line_recovery)
            and summary["minimum_layer_positive_line_recovery"]
            >= float(minimum_layer_positive_line_recovery)
            and summary["minimum_improvement_over_current"]
            >= float(minimum_improvement_over_current)
            and summary["maximum_radius_ratio_absolute_error"]
            <= float(maximum_radius_error)
        )
        summaries[name] = summary
    passing = [name for name in names if summaries[name]["passes"]]
    selected = (
        min(
            passing,
            key=lambda name: (
                summaries[name]["coordinates_per_layer"],
                -summaries[name]["minimum_positive_line_recovery"],
                name,
            ),
        )
        if passing
        else None
    )
    return {
        "classification": (
            "TERMINAL_COMPOSITIONAL_CAPACITY_PASSES"
            if selected is not None
            else "TERMINAL_COMPOSITIONAL_CAPACITY_REJECTED"
        ),
        "selected_candidate": selected,
        "candidate_summaries": summaries,
    }


def validate_plan(
    plan_path: Path, checkpoint: Path, config: Path, data_dir: Path
) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text())
    actual = {
        "checkpoint_sha256": file_sha256(checkpoint),
        "config_sha256": file_sha256(config),
        "dataset_manifest_sha256": file_sha256(data_dir / "manifest.json"),
        "entrypoint_sha256": file_sha256(Path(__file__).resolve()),
    }
    for key, value in actual.items():
        if plan["identity"][key] != value:
            raise ValueError(f"registered identity mismatch: {key}")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    plan = validate_plan(args.plan, args.checkpoint, args.config, args.data_dir)
    protocol = plan["protocol"]
    config = json.loads(args.config.read_text())
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[config["dtype"]]
    model, optimizer, checkpoint = load_model_and_optimizer(
        args.checkpoint, config, args.device
    )
    modules = cfc_modules(model)
    owner = directed_optimizer(optimizer)
    reference = modules[0]
    schedules = {
        str(name): [int(value) for value in schedule]
        for name, schedule in protocol["candidate_schedules"].items()
    }
    current_name = str(protocol["current_candidate"])
    rows: list[dict[str, Any]] = []
    gradient_window_ce: dict[str, float] = {}
    for seed_value in protocol["gradient_seeds"]:
        seed = int(seed_value)
        batches = fixed_batches(
            args.data_dir,
            "train",
            batch_size=int(protocol["gradient_batch_size"]),
            block_size=int(config["block_size"]) + 1,
            batches=int(protocol["gradient_accumulation_steps"]),
            seed=seed,
        )
        gradient_window_ce[str(seed)] = collect_cfc_gradients(
            model, modules, batches, device=args.device, dtype=dtype
        )
        dense, current_raw, prospective = prospective_updates(owner, modules)
        for name, schedule in schedules.items():
            if name == current_name:
                raw = current_raw
                stage_rows = prospective["stage_rows"]
            else:
                raw, stage_rows = fit_schedule(
                    modules,
                    dense,
                    schedule=schedule,
                    ridge_ratio=reference.ridge_ratio,
                    chunk_size=reference.chunk_size,
                )
            rows.append(
                direction_row(
                    dense,
                    raw,
                    gradient_seed=seed,
                    name=name,
                    schedule=schedule,
                    radius_ratio=float(protocol["registered_radius_ratio"]),
                    output_channels=modules[0].weight.shape[0],
                    stage_rows=stage_rows,
                )
            )
    rule = plan["decision_rule"]
    decision = classify_capacity(
        rows,
        current_name=current_name,
        maximum_coordinates_per_layer=int(
            rule["maximum_coordinates_per_layer"]
        ),
        minimum_positive_line_recovery=float(
            rule["minimum_positive_line_recovery"]
        ),
        minimum_layer_positive_line_recovery=float(
            rule["minimum_layer_positive_line_recovery"]
        ),
        minimum_improvement_over_current=float(
            rule["minimum_improvement_over_current"]
        ),
        maximum_radius_error=float(rule["maximum_radius_error"]),
    )
    args.output.mkdir(parents=True, exist_ok=False)
    rows_path = args.output / "direction_rows.json"
    rows_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "parameter_updates_to_checkpoint": 0,
        "checkpoint_next_iter": int(checkpoint["next_iter"]),
        "gradient_window_ce": gradient_window_ce,
        "identity": {
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "config_sha256": file_sha256(args.config),
            "dataset_manifest_sha256": file_sha256(
                args.data_dir / "manifest.json"
            ),
            "plan_sha256": file_sha256(args.plan),
            "direction_rows_sha256": file_sha256(rows_path),
        },
        "execution": {
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": str(Path(__file__).resolve()),
            "entrypoint_sha256": file_sha256(Path(__file__).resolve()),
            "command": sys.argv,
            "device": args.device,
            "direct_foreground_polling": True,
            "watchdog": False,
            "callback": False,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
        },
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
