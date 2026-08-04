#!/usr/bin/env python3
"""Gate projected weight-space Muon for an actual BlockFHT attention chart.

The proposed optimizer keeps momentum in compact latent coordinates, decodes
that momentum through the fixed BlockFHT Jacobian, applies Muon's matrix polar
map in generated-weight space, and pulls the result back through the exact
adjoint.  This offline oracle asks whether that nonlinear operation preserves
the dense Muon direction before any training implementation is authorized.
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

from examples.nanogpt.analyze_attention_blockfht_tangent import (
    project_attention_target,
)
from examples.nanogpt.analyze_mlp_task_gradient_direction import (
    direction_metrics,
)
from examples.nanogpt.analyze_parameter_trajectory import parse_int_list
from examples.nanogpt.muon import zeropower_via_newtonschulz5
from examples.nanogpt.parameter_trajectory import (
    OPTIMIZER_PROBE_SCHEMA_VERSION,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze_cell(
    *,
    combined_momentum: torch.Tensor,
    dense_polar: torch.Tensor,
    target: str,
    config: dict[str, Any],
    layer: int,
    ns_steps: int,
) -> dict[str, float]:
    """Score the proposed compact-state projected-Muon direction."""
    projected_momentum, _ = project_attention_target(
        combined_momentum,
        target=target,
        config=config,
        layer=layer,
    )
    weight_space_polar = zeropower_via_newtonschulz5(
        projected_momentum,
        steps=ns_steps,
    )
    proposed, _ = project_attention_target(
        weight_space_polar,
        target=target,
        config=config,
        layer=layer,
    )
    oracle, _ = project_attention_target(
        dense_polar,
        target=target,
        config=config,
        layer=layer,
    )
    proposed_metrics = direction_metrics(dense_polar, proposed)
    oracle_metrics = direction_metrics(dense_polar, oracle)
    raw_metrics = direction_metrics(dense_polar, projected_momentum)
    target_energy = float(dense_polar.double().square().sum())
    return {
        "dense_polar_fro": target_energy**0.5,
        "oracle_recovery": oracle_metrics["positive_step_line_recovery"],
        "oracle_cosine": oracle_metrics["cosine"],
        "proposed_recovery": proposed_metrics["positive_step_line_recovery"],
        "proposed_cosine": proposed_metrics["cosine"],
        "raw_projected_momentum_recovery": raw_metrics[
            "positive_step_line_recovery"
        ],
        "raw_projected_momentum_cosine": raw_metrics["cosine"],
        "proposed_over_oracle": proposed_metrics[
            "positive_step_line_recovery"
        ]
        / max(oracle_metrics["positive_step_line_recovery"], 1e-30),
        "proposed_over_raw": proposed_metrics[
            "positive_step_line_recovery"
        ]
        / max(raw_metrics["positive_step_line_recovery"], 1e-30),
    }


def weighted_summary(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("cannot summarize zero cells")
    weights = torch.tensor(
        [float(row["dense_polar_fro"]) ** 2 for row in rows],
        dtype=torch.float64,
    )

    def weighted(key: str) -> float:
        values = torch.tensor(
            [float(row[key]) for row in rows], dtype=torch.float64
        )
        return float((weights * values).sum() / weights.sum())

    result: dict[str, float | int] = {
        "cells": len(rows),
        "oracle_recovery": weighted("oracle_recovery"),
        "oracle_cosine": weighted("oracle_cosine"),
        "proposed_recovery": weighted("proposed_recovery"),
        "proposed_cosine": weighted("proposed_cosine"),
        "raw_projected_momentum_recovery": weighted(
            "raw_projected_momentum_recovery"
        ),
        "raw_projected_momentum_cosine": weighted(
            "raw_projected_momentum_cosine"
        ),
    }
    result["proposed_over_oracle"] = float(result["proposed_recovery"]) / max(
        float(result["oracle_recovery"]), 1e-30
    )
    result["proposed_over_raw"] = float(result["proposed_recovery"]) / max(
        float(result["raw_projected_momentum_recovery"]), 1e-30
    )
    return result


def gate_decision(
    aggregate: dict[str, float | int],
    by_target: dict[str, dict[str, float | int]],
    *,
    minimum_oracle_fraction: float,
    minimum_raw_multiplier: float,
    minimum_target_oracle_fraction: float,
) -> tuple[str, list[str]]:
    failures: list[str] = []
    if float(aggregate["proposed_over_oracle"]) < minimum_oracle_fraction:
        failures.append("aggregate_oracle_fraction")
    if float(aggregate["proposed_over_raw"]) < minimum_raw_multiplier:
        failures.append("aggregate_raw_multiplier")
    for target, summary in sorted(by_target.items()):
        if (
            float(summary["proposed_over_oracle"])
            < minimum_target_oracle_fraction
        ):
            failures.append(f"{target}_oracle_fraction")
        if float(summary["proposed_cosine"]) <= 0.0:
            failures.append(f"{target}_nonpositive_cosine")
    return (
        "AUTHORIZE_IMPLEMENTATION" if not failures else "REJECT_IMPLEMENTATION",
        failures,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", required=True, type=Path)
    parser.add_argument("--production-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", default="0,3,6,9,11")
    parser.add_argument("--steps", default="0,60,120,180")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--minimum-oracle-fraction", type=float, default=0.50)
    parser.add_argument("--minimum-raw-multiplier", type=float, default=1.25)
    parser.add_argument(
        "--minimum-target-oracle-fraction", type=float, default=0.35
    )
    args = parser.parse_args()

    started = time.time()
    layers = parse_int_list(args.layers)
    steps = parse_int_list(args.steps)
    if not layers or not steps:
        raise ValueError("layers and steps must be non-empty")
    config = json.loads(args.production_config.read_text())
    expected = {
        "attn.c_attn.qk_headwise",
        "attn.c_attn.v",
        "attn.c_proj",
    }
    if set(config["block_fht_targets"]) != expected:
        raise ValueError("production config is not the full-attention BlockFHT chart")

    probe_paths = [
        args.probe_dir / f"step_{step:06d}.pt" for step in steps
    ]
    missing = [str(path) for path in probe_paths if not path.is_file()]
    if missing:
        raise ValueError("missing optimizer probes: " + ", ".join(missing))

    rows: list[dict[str, Any]] = []
    run_identity_sha256: str | None = None
    for step, probe_path in zip(steps, probe_paths, strict=True):
        probe = torch.load(probe_path, map_location="cpu", weights_only=False)
        if probe.get("schema_version") != OPTIMIZER_PROBE_SCHEMA_VERSION:
            raise ValueError(f"unexpected optimizer probe schema: {probe_path}")
        if run_identity_sha256 is None:
            run_identity_sha256 = probe["run_identity_sha256"]
        elif probe["run_identity_sha256"] != run_identity_sha256:
            raise ValueError("optimizer probes do not share one run identity")
        for layer in layers:
            for target in ("attn.c_attn", "attn.c_proj"):
                name = f"transformer.h.{layer}.{target}.weight"
                record = probe["parameters"][name]
                hyperparameters = probe["hyperparameters"][name]
                metrics = analyze_cell(
                    combined_momentum=record["combined_momentum_update"].to(
                        args.device, dtype=torch.float32
                    ),
                    dense_polar=record["polar_update"].to(
                        args.device, dtype=torch.float32
                    ),
                    target=target,
                    config=config,
                    layer=layer,
                    ns_steps=int(hyperparameters["ns_steps"]),
                )
                rows.append(
                    {
                        "parameter": name,
                        "target": target,
                        "layer": layer,
                        "step": step,
                        "ns_steps": int(hyperparameters["ns_steps"]),
                        **metrics,
                    }
                )
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()

    by_target = {
        target: weighted_summary(
            [row for row in rows if row["target"] == target]
        )
        for target in ("attn.c_attn", "attn.c_proj")
    }
    aggregate = weighted_summary(rows)
    decision, failures = gate_decision(
        aggregate,
        by_target,
        minimum_oracle_fraction=args.minimum_oracle_fraction,
        minimum_raw_multiplier=args.minimum_raw_multiplier,
        minimum_target_oracle_fraction=args.minimum_target_oracle_fraction,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "projected_muon_cells.csv", rows)
    repo_root = Path(__file__).resolve().parents[2]
    result = {
        "schema_version": "mai_124m_attention_blockfht_projected_muon_v1",
        "scientific_question": (
            "Can compact latent momentum decoded through the actual fixed "
            "BlockFHT Jacobian preserve dense Muon's matrix direction after "
            "polarization and exact tangent pullback?"
        ),
        "source_commit": git_commit(repo_root),
        "source_sha256": file_sha256(Path(__file__)),
        "production_config": str(args.production_config),
        "production_config_sha256": file_sha256(args.production_config),
        "optimizer_probe_run_identity_sha256": run_identity_sha256,
        "optimizer_probe_paths": [
            {"path": str(path), "sha256": file_sha256(path)}
            for path in probe_paths
        ],
        "layers": layers,
        "steps": steps,
        "primary_target": "dense optimizer-probe polar_update (weight decay excluded)",
        "candidate_formula": "P * polar(P * combined_momentum_update)",
        "oracle_formula": "P * dense_polar_update",
        "raw_baseline_formula": "P * combined_momentum_update",
        "by_target": by_target,
        "aggregate": aggregate,
        "gate": {
            "decision": decision,
            "failures": failures,
            "minimum_oracle_fraction": args.minimum_oracle_fraction,
            "minimum_raw_multiplier": args.minimum_raw_multiplier,
            "minimum_target_oracle_fraction": (
                args.minimum_target_oracle_fraction
            ),
        },
        "interpretation": {
            "oracle_recovery": (
                "best positive-line recovery obtainable by projecting the "
                "already formed dense Muon polar direction into this tangent"
            ),
            "proposed_recovery": (
                "positive-line recovery after projecting dense momentum, "
                "applying Muon in generated-weight space, and projecting again"
            ),
            "raw_projected_momentum_recovery": (
                "control for merely using projected compact momentum without "
                "the nonlinear Muon polar map"
            ),
        },
        "limitations": [
            "This is an optimistic replay of dense-run momentum, not an end-to-end optimizer trajectory.",
            "The fixed tangent has about one percent ambient rank, so absolute recovery remains bounded by the tangent oracle.",
            "A pass authorizes implementation and MFU testing only, not a larger training rung.",
        ],
        "elapsed_seconds": time.time() - started,
    }
    (args.output / "projected_muon_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": decision,
                "failures": failures,
                "aggregate": aggregate,
                "by_target": by_target,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
