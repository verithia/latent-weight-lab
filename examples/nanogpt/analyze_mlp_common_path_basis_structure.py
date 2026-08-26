#!/usr/bin/env python3
"""Measure fast-code recovery of the denoised two-stream common MLP path.

The common path averages two common-initialization dense-Muon trajectories.
Five centered temporal PCs are treated as a descriptive basis, and existing
fast structure families are charged by their complete stored scalar count.
No language-model parameter is updated.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_mlp_disjoint_data_state_transfer import (
    git_commit,
    load_weight_run,
)
from examples.nanogpt.analyze_mlp_highcadence_basis import (
    file_sha256,
    parse_float_list,
)
from examples.nanogpt.analyze_mlp_state_basis_structure import analyze_parameter
from examples.nanogpt.analyze_parameter_trajectory import write_csv


def budget_frontier(
    rows: list[dict[str, Any]], budget: float
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    eligible = [
        row
        for row in rows
        if float(row["total_stored_scalar_fraction_for_all_basis_vectors"])
        <= budget + 1e-12
    ]
    parameters = sorted({str(row["parameter"]) for row in rows})
    best: dict[str, dict[str, Any]] = {}
    for parameter in parameters:
        candidates = [row for row in eligible if row["parameter"] == parameter]
        if not candidates:
            raise ValueError(f"no family fits the budget for {parameter}")
        best[parameter] = max(
            candidates, key=lambda row: float(row["weighted_basis_energy_capture"])
        )
    return eligible, best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a-probe-dir", required=True, type=Path)
    parser.add_argument("--run-b-probe-dir", required=True, type=Path)
    parser.add_argument("--run-a-name", default="stream_a")
    parser.add_argument("--run-b-name", default="stream_b")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--targets", default="mlp.c_fc,mlp.c_proj")
    parser.add_argument("--basis-rank", type=int, default=5)
    parser.add_argument("--ratios", default="0.0002,0.0005,0.001,0.002,0.01")
    parser.add_argument("--block-fht-layers", type=int, default=2)
    parser.add_argument("--block-fht-seed", type=int, default=1000)
    parser.add_argument("--total-budget", type=float, default=0.01)
    parser.add_argument("--weighted-capture-gate", type=float, default=0.90)
    parser.add_argument("--minimum-capture-gate", type=float, default=0.80)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    targets = {value for value in args.targets.split(",") if value}
    ratios = parse_float_list(args.ratios)
    if not targets or args.basis_rank != 5 or not 0 < args.total_budget <= 0.01:
        raise ValueError("the frozen audit requires targets, rank five, and <=1% budget")

    steps_a, run_a, metadata_a = load_weight_run(
        args.run_a_probe_dir, layer=args.layer, targets=targets
    )
    steps_b, run_b, metadata_b = load_weight_run(
        args.run_b_probe_dir, layer=args.layer, targets=targets
    )
    if steps_a != steps_b or set(run_a) != set(run_b):
        raise ValueError("probe steps and parameter inventories must match")
    rows: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    for parameter in sorted(run_a):
        initial_equal = torch.equal(run_a[parameter][0], run_b[parameter][0])
        identity_rows.append(
            {
                "parameter": parameter,
                "bitwise_equal": initial_equal,
                "maximum_absolute_difference": float(
                    (run_a[parameter][0].float() - run_b[parameter][0].float()).abs().max()
                ),
            }
        )
        if not initial_equal:
            raise ValueError(f"step-zero weight mismatch for {parameter}")
        common_positions = 0.5 * (
            torch.stack(run_a[parameter]).to(args.device, torch.float32)
            + torch.stack(run_b[parameter]).to(args.device, torch.float32)
        )
        parameter_rows = analyze_parameter(
            common_positions,
            parameter=parameter,
            ratios=ratios,
            basis_rank=args.basis_rank,
            block_fht_layers=args.block_fht_layers,
            block_fht_seed=args.block_fht_seed,
        )
        for row in parameter_rows:
            row["common_path"] = f"mean({args.run_a_name},{args.run_b_name})"
            row["eligible_under_total_budget"] = (
                float(row["total_stored_scalar_fraction_for_all_basis_vectors"])
                <= args.total_budget + 1e-12
            )
        rows.extend(parameter_rows)
        del common_positions
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

    eligible, best = budget_frontier(rows, args.total_budget)
    best_rows: list[dict[str, Any]] = []
    parameter_authorized: dict[str, bool] = {}
    for parameter, row in sorted(best.items()):
        minimum = row.get("minimum_basis_capture")
        minimum_gate_available = minimum is not None
        authorized = (
            float(row["weighted_basis_energy_capture"]) >= args.weighted_capture_gate
            and minimum_gate_available
            and float(minimum) >= args.minimum_capture_gate
        )
        parameter_authorized[parameter] = authorized
        best_rows.append(
            {
                **row,
                "minimum_gate_available": minimum_gate_available,
                "parameter_authorized": authorized,
            }
        )
    gate = {
        "step_zero_bitwise_equal": all(row["bitwise_equal"] for row in identity_rows),
        "total_stored_scalar_fraction_ceiling": args.total_budget,
        "weighted_basis_capture_threshold": args.weighted_capture_gate,
        "minimum_basis_capture_threshold": args.minimum_capture_gate,
        "best_under_budget": {
            parameter: {
                "family": row["family"],
                "weighted_basis_energy_capture": row["weighted_basis_energy_capture"],
                "minimum_basis_capture": row.get("minimum_basis_capture"),
                "total_stored_scalar_fraction": row[
                    "total_stored_scalar_fraction_for_all_basis_vectors"
                ],
                "authorized": parameter_authorized[parameter],
            }
            for parameter, row in sorted(best.items())
        },
        "common_path_fast_basis_authorized": all(parameter_authorized.values()),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    paths = {
        "identity": args.output / "step_zero_identity.csv",
        "structure": args.output / "common_path_basis_structure.csv",
        "eligible": args.output / "eligible_under_one_percent.csv",
        "best": args.output / "best_under_one_percent.csv",
        "gate": args.output / "gate.json",
    }
    write_csv(paths["identity"], identity_rows)
    write_csv(paths["structure"], rows)
    write_csv(paths["eligible"], eligible)
    write_csv(paths["best"], best_rows)
    paths["gate"].write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    outputs = {
        name: {"path": str(path), "sha256": file_sha256(path)}
        for name, path in paths.items()
    }
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_mlp_common_path_basis_structure_v1",
        "method": "five-PC structure audit of the common two-stream MLP weight path",
        "source_commit": git_commit(script.parents[2]),
        "entrypoint": str(script),
        "entrypoint_sha256": file_sha256(script),
        "command": [str(script), *sys.argv[1:]],
        "runs": {args.run_a_name: metadata_a, args.run_b_name: metadata_b},
        "steps": steps_a,
        "basis_rank": args.basis_rank,
        "ratios": ratios,
        "binding_gate": gate,
        "outputs": outputs,
        "runtime_seconds": time.time() - started,
        "limitations": [
            "The two-stream average is an optimistic noncausal estimate of shared task motion.",
            "Learned support and independent-SVD rows omit support indices and optimizer state.",
            "A family without per-PC minimum capture cannot be authorized by weighted capture alone.",
            "Euclidean PC recovery is not a language-model loss guarantee.",
        ],
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"gate": gate, "metadata": str(metadata_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
