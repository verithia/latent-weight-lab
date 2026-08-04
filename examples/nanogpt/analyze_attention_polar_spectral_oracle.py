#!/usr/bin/env python3
"""Decompose dense attention directions into polar spectral coordinates.

The attention rotation oracle measured task-gradient recovery by left and
right orthogonal orbits.  Orthogonal factors cannot change singular values,
and the dense baseline actually follows Muon's applied matrix direction rather
than the clipped gradient itself.  This diagnostic therefore measures both
directions in the current weight's singular frame.

For ``W = U diag(s) V^T``, the diagonal of ``U^T D V`` is the orthogonal
projection of direction ``D`` onto singular-value-changing coordinates.  The
script reports the full spectral fraction, the fraction captured by the first
``k`` singular vectors (a causal current-weight basis), and an oracle top-k
coefficient bound.  It is a capacity diagnostic, not a trained candidate.
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

from examples.nanogpt.analyze_parameter_trajectory import parse_int_list
from examples.nanogpt.parameter_trajectory import (
    OPTIMIZER_PROBE_SCHEMA_VERSION,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def polar_spectral_metrics(
    weight: torch.Tensor,
    direction: torch.Tensor,
    ranks: list[int],
) -> dict[str, float]:
    """Return spectral-coordinate recovery in the current SVD frame."""

    weight = weight.double()
    direction = direction.double()
    u, singular, vh = torch.linalg.svd(weight, full_matrices=False)
    coordinates = u.transpose(0, 1) @ direction @ vh.transpose(0, 1)
    diagonal_energy = coordinates.diagonal().square()
    total_energy = direction.square().sum().clamp_min(1e-30)
    spectral_energy = diagonal_energy.sum()
    spectral_direction = (
        u * coordinates.diagonal().unsqueeze(0)
    ) @ vh
    orthogonality = torch.abs(
        ((direction - spectral_direction) * spectral_direction).sum()
    ) / (
        direction.norm().clamp_min(1e-30)
        * spectral_direction.norm().clamp_min(1e-30)
    )

    output: dict[str, float] = {
        "direction_fro": float(direction.norm()),
        "spectral_recovery": float(spectral_energy / total_energy),
        "spectral_projection_relative_orthogonality_error": float(
            orthogonality
        ),
        "singular_max": float(singular.max()),
        "singular_min": float(singular.min()),
        "singular_condition": float(
            singular.max() / singular.min().clamp_min(1e-30)
        ),
    }
    if singular.numel() > 1:
        adjacent = (singular[:-1] - singular[1:]).abs()
        output["minimum_adjacent_singular_gap_relative"] = float(
            adjacent.min() / singular.max().clamp_min(1e-30)
        )
    else:
        output["minimum_adjacent_singular_gap_relative"] = 1.0

    for rank in ranks:
        bounded = min(int(rank), int(diagonal_energy.numel()))
        causal_energy = diagonal_energy[:bounded].sum()
        oracle_energy = torch.topk(
            diagonal_energy, k=bounded, largest=True
        ).values.sum()
        output[f"top_singular_rank{rank}_recovery"] = float(
            causal_energy / total_energy
        )
        output[f"oracle_spectral_rank{rank}_recovery"] = float(
            oracle_energy / total_energy
        )
        output[f"top_singular_rank{rank}_spectral_share"] = float(
            causal_energy / spectral_energy.clamp_min(1e-30)
        )
        output[f"oracle_spectral_rank{rank}_spectral_share"] = float(
            oracle_energy / spectral_energy.clamp_min(1e-30)
        )
    return output


def weighted_summary(
    rows: list[dict[str, Any]],
    prefixes: tuple[str, ...],
    metric_suffixes: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {"cells": len(rows)}
    for prefix in prefixes:
        weights = torch.tensor(
            [float(row[f"{prefix}_direction_fro"]) ** 2 for row in rows],
            dtype=torch.float64,
        )
        for suffix in metric_suffixes:
            values = torch.tensor(
                [float(row[f"{prefix}_{suffix}"]) for row in rows],
                dtype=torch.float64,
            )
            result[f"{prefix}_{suffix}"] = float(
                (weights * values).sum() / weights.sum()
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", default="0,3,6,9,11")
    parser.add_argument("--steps", default="0,594,1188,1782,2372")
    parser.add_argument("--spectral-ranks", default="8,16,32,64,128,256")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    started = time.time()
    layers = parse_int_list(args.layers)
    steps = parse_int_list(args.steps)
    ranks = parse_int_list(args.spectral_ranks)
    if not layers or not steps or not ranks or any(rank <= 0 for rank in ranks):
        raise ValueError("layers, steps, and positive spectral ranks are required")
    probe_paths = [
        args.probe_dir / f"step_{step:06d}.pt" for step in steps
    ]
    missing = [str(path) for path in probe_paths if not path.is_file()]
    if missing:
        raise ValueError("missing optimizer probes: " + ", ".join(missing))

    prefixes = ("task", "muon")
    rank_suffixes = [
        suffix
        for rank in ranks
        for suffix in (
            f"top_singular_rank{rank}_recovery",
            f"oracle_spectral_rank{rank}_recovery",
            f"top_singular_rank{rank}_spectral_share",
            f"oracle_spectral_rank{rank}_spectral_share",
        )
    ]
    metric_suffixes = ["spectral_recovery", *rank_suffixes]
    rows: list[dict[str, Any]] = []
    run_identity_sha256: str | None = None
    n_embd: int | None = None
    for path in probe_paths:
        probe = torch.load(path, map_location="cpu", weights_only=False)
        if probe.get("schema_version") != OPTIMIZER_PROBE_SCHEMA_VERSION:
            raise ValueError(f"unexpected optimizer probe schema in {path}")
        if run_identity_sha256 is None:
            run_identity_sha256 = probe["run_identity_sha256"]
        elif probe["run_identity_sha256"] != run_identity_sha256:
            raise ValueError("optimizer probes do not share one identity")
        observed_n_embd = int(probe["model_config"]["n_embd"])
        if n_embd is None:
            n_embd = observed_n_embd
        elif observed_n_embd != n_embd:
            raise ValueError("optimizer probes do not share one model width")
        assert n_embd is not None
        step = int(probe["step"])
        for layer in layers:
            parameter_names = (
                (
                    "qk_shared",
                    f"transformer.h.{layer}.attn.c_attn.weight",
                    slice(0, 2 * n_embd),
                ),
                (
                    "v",
                    f"transformer.h.{layer}.attn.c_attn.weight",
                    slice(2 * n_embd, None),
                ),
                (
                    "cproj",
                    f"transformer.h.{layer}.attn.c_proj.weight",
                    slice(None),
                ),
            )
            for target, name, selection in parameter_names:
                record = probe["parameters"][name]
                weight = record["weight_before_step"][selection].to(args.device)
                directions = {
                    "task": record["gradient_after_clip"][selection].to(
                        args.device
                    ),
                    "muon": record["applied_direction_per_lr"][selection].to(
                        args.device
                    ),
                }
                row: dict[str, Any] = {
                    "step": step,
                    "layer": layer,
                    "target": target,
                }
                for prefix, direction in directions.items():
                    metrics = polar_spectral_metrics(weight, direction, ranks)
                    row.update(
                        {f"{prefix}_{key}": value for key, value in metrics.items()}
                    )
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
                del weight, directions
        del probe
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    targets = ("qk_shared", "v", "cproj")
    by_target = {
        target: weighted_summary(
            [row for row in rows if row["target"] == target],
            prefixes,
            metric_suffixes,
        )
        for target in targets
    }
    by_step = {
        str(step): weighted_summary(
            [row for row in rows if int(row["step"]) == step],
            prefixes,
            metric_suffixes,
        )
        for step in steps
    }
    aggregate = weighted_summary(rows, prefixes, metric_suffixes)

    selected_rank: int | None = None
    for rank in ranks:
        if all(
            float(by_target[target][f"muon_top_singular_rank{rank}_spectral_share"])
            >= 0.80
            for target in targets
        ):
            selected_rank = rank
            break
    material_spectral = (
        float(aggregate["task_spectral_recovery"]) >= 0.10
        or float(aggregate["muon_spectral_recovery"]) >= 0.10
    )

    args.output.mkdir(parents=True, exist_ok=True)
    cells_path = args.output / "attention_polar_spectral_cells.csv"
    write_csv(cells_path, rows)
    repo_root = Path(__file__).resolve().parents[2]
    result = {
        "schema_version": "mai_124m_attention_polar_spectral_oracle_v1",
        "scientific_question": (
            "Does the remaining attention direction require singular-value "
            "motion that orthogonal Cayley charts cannot represent?"
        ),
        "source_commit": git_commit(repo_root),
        "source_sha256": file_sha256(Path(__file__)),
        "optimizer_probe_run_identity_sha256": run_identity_sha256,
        "optimizer_probe_paths": [
            {"path": str(path), "sha256": file_sha256(path)}
            for path in probe_paths
        ],
        "layers": layers,
        "steps": steps,
        "spectral_ranks": ranks,
        "aggregate": aggregate,
        "by_target": by_target,
        "by_step": by_step,
        "decision": {
            "material_spectral_component": material_spectral,
            "material_threshold": 0.10,
            "smallest_causal_top_singular_rank_for_80pct_muon_spectral_share_all_targets": selected_rank,
            "classification": (
                "DESIGN_CURRENT_WEIGHT_SPECTRAL_CHART"
                if material_spectral
                else "KEEP_FOCUS_ON_ORBIT_OPTIMIZATION"
            ),
        },
        "interpretation": {
            "spectral_recovery": (
                "fraction of direction energy in diag(U^T D V), the singular-"
                "value-changing complement of a full bilateral orthogonal "
                "orbit under a simple full-rank spectrum"
            ),
            "top_singular_rank_recovery": (
                "direction energy captured by the first k current singular "
                "vectors, ordered causally by W's singular values"
            ),
            "oracle_spectral_rank_recovery": (
                "upper bound selecting the k largest measured spectral "
                "coefficients; not a deployable selection rule"
            ),
        },
        "limitations": [
            "This is an oracle on dense weights and exact probes, not a trained generated model.",
            "The SVD basis is computed from the current weight and introduces no learned basis, but a candidate must cache or approximate it and pass the 20% MFU gate.",
            "Near-degenerate singular values make the split between rotation and spectral coordinates ill-conditioned; minimum adjacent gaps are recorded per cell.",
            "The task-gradient and Muon-applied directions are reported separately because the trained baseline follows the latter while candidate parameter gradients originate from the former.",
        ],
        "outputs": {"cells_sha256": file_sha256(cells_path)},
        "elapsed_seconds": time.time() - started,
    }
    result_path = args.output / "attention_polar_spectral_result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["decision"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
