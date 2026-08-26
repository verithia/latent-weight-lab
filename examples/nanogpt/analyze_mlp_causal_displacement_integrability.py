#!/usr/bin/env python3
"""Test whether causal gradient row spaces can store accumulated MLP state.

The raw-gradient factor field has a locally predictable right/input side.  A
moving low-rank chart is useful only if the same causal row space can represent
the accumulated dense displacement, rather than just the next gradient.  This
script measures that missing integrability condition without training a new
model or using future information to fit the basis.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_mlp_gradient_factor_field import fit_union_basis
from examples.nanogpt.analyze_mlp_highcadence_basis import file_sha256
from examples.nanogpt.analyze_mlp_optimizer_probe_span import load_probe_inventory
from examples.nanogpt.analyze_mlp_product_fht_tangent_anchor import git_commit
from examples.nanogpt.analyze_mlp_raw_gradient_factor_transport import exact_singular_factors
from examples.nanogpt.analyze_mlp_raw_gradient_rolling_prediction import phase_for_step
from examples.nanogpt.analyze_parameter_trajectory import write_csv


def right_projection_capture(matrix: torch.Tensor, basis: torch.Tensor) -> float:
    """Frobenius-energy capture of the exact right-subspace projection."""
    total = matrix.double().square().sum().clamp_min(1e-30)
    projected_coordinates = matrix @ basis
    return float(projected_coordinates.double().square().sum() / total)


def best_rank_capture(matrix: torch.Tensor, rank: int) -> float:
    """Exact matrix-SVD energy ceiling at ``rank``."""
    values = torch.linalg.svdvals(matrix).double().square()
    return float(values[:rank].sum() / values.sum().clamp_min(1e-30))


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted(
        {
            (row["parameter"], row["split"], row["union_rank"])
            for row in rows
        }
    )
    result: list[dict[str, Any]] = []
    for parameter, split, rank in keys:
        members = [
            row
            for row in rows
            if (row["parameter"], row["split"], row["union_rank"])
            == (parameter, split, rank)
        ]
        item: dict[str, Any] = {
            "parameter": parameter,
            "split": split,
            "union_rank": rank,
            "sample_count": len(members),
            "increment_sample_count": sum(
                row["next_increment_capture"] != "" for row in members
            ),
        }
        for field in (
            "displacement_capture",
            "displacement_oracle_rank_capture",
            "next_increment_capture",
            "next_increment_oracle_rank_capture",
        ):
            values = torch.tensor(
                [float(row[field]) for row in members if row[field] != ""],
                dtype=torch.float64,
            )
            if values.numel() == 0:
                item[f"{field}_mean"] = ""
                item[f"{field}_minimum"] = ""
                item[f"{field}_p10"] = ""
            else:
                item[f"{field}_mean"] = float(values.mean())
                item[f"{field}_minimum"] = float(values.min())
                item[f"{field}_p10"] = float(torch.quantile(values, 0.10))
        result.append(item)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--targets", default="mlp.c_fc,mlp.c_proj")
    parser.add_argument("--factor-rank", type=int, default=6)
    parser.add_argument("--union-ranks", default="1,3,6,12,24,48")
    parser.add_argument("--history-probes", type=int, default=10)
    parser.add_argument("--discovery-stop", type=int, default=119)
    parser.add_argument("--validation-stop", type=int, default=179)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    union_ranks = [int(value) for value in args.union_ranks.split(",")]
    targets = {value for value in args.targets.split(",") if value}
    paths = sorted(args.probe_dir.glob("step_*.pt"))
    steps, inventory, input_metadata = load_probe_inventory(
        paths, layers={args.layer}, targets=targets
    )
    if len(steps) <= args.history_probes:
        raise ValueError("not enough probes for the requested history")

    rows: list[dict[str, Any]] = []
    for parameter, fields in sorted(inventory.items()):
        gradients = torch.stack(fields["raw_gradient_descent"]).to(
            args.device, torch.float32
        )
        weights = torch.stack(fields["weight_before_step"]).to(
            args.device, torch.float32
        )
        _left, singular, right = exact_singular_factors(
            gradients, args.factor_rank
        )
        initial = weights[0]
        matrix_size = weights.shape[1] * weights.shape[2]
        for index in range(args.history_probes, len(steps)):
            displacement = weights[index] - initial
            increment = (
                weights[index + 1] - weights[index]
                if index + 1 < len(steps)
                else None
            )
            for rank in union_ranks:
                basis = fit_union_basis(
                    right,
                    singular,
                    range(index - args.history_probes, index),
                    min(rank, args.history_probes * args.factor_rank),
                )
                row: dict[str, Any] = {
                    "parameter": parameter,
                    "probe_index": index,
                    "step": steps[index],
                    "next_step": steps[index + 1] if increment is not None else "",
                    "split": phase_for_step(
                        steps[index], args.discovery_stop, args.validation_stop
                    ),
                    "history_probe_start": index - args.history_probes,
                    "history_probe_stop": index - 1,
                    "history_step_start": steps[index - args.history_probes],
                    "history_step_stop": steps[index - 1],
                    "union_rank": rank,
                    "stored_scalar_fraction": rank
                    * (weights.shape[1] + weights.shape[2])
                    / matrix_size,
                    "displacement_capture": right_projection_capture(
                        displacement, basis
                    ),
                    "displacement_oracle_rank_capture": best_rank_capture(
                        displacement, rank
                    ),
                    "next_increment_capture": "",
                    "next_increment_oracle_rank_capture": "",
                }
                if increment is not None:
                    row["next_increment_capture"] = right_projection_capture(
                        increment, basis
                    )
                    row["next_increment_oracle_rank_capture"] = best_rank_capture(
                        increment, rank
                    )
                rows.append(row)
        del gradients, weights, right, singular
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

    summary = summarize(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    detail_path = args.output / "causal_displacement_integrability.csv"
    summary_path = args.output / "causal_displacement_integrability_summary.csv"
    write_csv(detail_path, rows)
    write_csv(summary_path, summary)
    metadata = {
        "schema_version": "nanogpt_mlp_causal_displacement_integrability_v1",
        "method": "preceding-ten raw-gradient right-basis capture of accumulated state and next dense increment",
        "source_commit": git_commit(Path(__file__).resolve().parents[2]),
        "input": input_metadata,
        "steps": steps,
        "factor_rank": args.factor_rank,
        "history_probes": args.history_probes,
        "union_ranks": union_ranks,
        "discovery_stop": args.discovery_stop,
        "validation_stop": args.validation_stop,
        "limitations": [
            "The audit is one layer, seed, schedule, horizon, and dense-Muon trajectory.",
            "Probe-to-probe increments span approximately two to three optimizer steps.",
            "The fitted basis uses only preceding gradients; oracle SVD fields are descriptive ceilings.",
        ],
        "runtime_seconds": time.time() - started,
        "detail_sha256": file_sha256(detail_path),
        "summary_sha256": file_sha256(summary_path),
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
