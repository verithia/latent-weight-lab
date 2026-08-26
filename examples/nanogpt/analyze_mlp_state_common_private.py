#!/usr/bin/env python3
"""Attribute common MLP weight-path motion to the initial-weight scalar orbit.

The two inputs must share an exact step-zero gauge. This zero-update control
separates common and private displacement energy, removes the best scalar
multiple of W0 at every probe, and repeats cross-stream affine-basis transfer
on the residual paths.
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
    nested_bases,
    row_cosine,
    summarize_metric,
)
from examples.nanogpt.analyze_mlp_highcadence_basis import (
    chronological_splits,
    energy_capture,
    file_sha256,
)
from examples.nanogpt.analyze_mlp_raw_gradient_rolling_prediction import phase_for_step
from examples.nanogpt.analyze_parameter_trajectory import write_csv


def remove_scalar_orbit(
    rows: torch.Tensor, initial: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if rows.ndim != 2 or initial.ndim != 1 or rows.shape[1] != initial.numel():
        raise ValueError("rows and initial weight must share one flattened dimension")
    work = rows.double()
    seed = initial.double()
    denominator = seed.square().sum().clamp_min(1e-30)
    coefficients = (work @ seed) / denominator
    projection = coefficients.unsqueeze(1) * seed.unsqueeze(0)
    residual = work - projection
    row_energy = work.square().sum(dim=1).clamp_min(1e-30)
    capture = projection.square().sum(dim=1) / row_energy
    return residual.float(), coefficients, capture


def aggregate_capture(rows: torch.Tensor, initial: torch.Tensor) -> float:
    residual, _coefficients, _capture = remove_scalar_orbit(rows, initial)
    total = rows.double().square().sum().clamp_min(1e-30)
    return float(1.0 - residual.double().square().sum() / total)


def energy_fraction(numerator: torch.Tensor, denominator_parts: tuple[torch.Tensor, ...]) -> float:
    top = numerator.double().square().sum()
    bottom = sum(part.double().square().sum() for part in denominator_parts)
    return float(top / bottom.clamp_min(1e-30))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a-probe-dir", required=True, type=Path)
    parser.add_argument("--run-b-probe-dir", required=True, type=Path)
    parser.add_argument("--run-a-name", default="stream_a")
    parser.add_argument("--run-b-name", default="stream_b")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--targets", default="mlp.c_fc,mlp.c_proj")
    parser.add_argument("--ranks", default="1,3,6,12,16,24,48")
    parser.add_argument("--discovery-stop", type=int, default=119)
    parser.add_argument("--validation-stop", type=int, default=179)
    parser.add_argument("--common-scalar-capture-gate", type=float, default=0.50)
    parser.add_argument("--residual-cross-capture-gate", type=float, default=0.20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    targets = {value for value in args.targets.split(",") if value}
    ranks = [int(value) for value in args.ranks.split(",")]
    if not targets or ranks != sorted(set(ranks)) or 16 not in ranks or 48 not in ranks:
        raise ValueError("targets and ordered ranks including 16/48 are required")

    steps_a, run_a, metadata_a = load_weight_run(
        args.run_a_probe_dir, layer=args.layer, targets=targets
    )
    steps_b, run_b, metadata_b = load_weight_run(
        args.run_b_probe_dir, layer=args.layer, targets=targets
    )
    if steps_a != steps_b or set(run_a) != set(run_b):
        raise ValueError("probe steps and parameter inventories must match")
    steps = steps_a
    splits = chronological_splits(
        steps,
        discovery_stop=args.discovery_stop,
        validation_stop=args.validation_stop,
    )
    all_indices = list(range(len(steps)))
    names = (args.run_a_name, args.run_b_name)
    identity_rows: list[dict[str, Any]] = []
    orbit_rows: list[dict[str, Any]] = []
    decomposition_rows: list[dict[str, Any]] = []
    transfer_rows: list[dict[str, Any]] = []
    matched_rows: list[dict[str, Any]] = []

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
        initial = run_a[parameter][0].to(args.device, torch.float32).flatten()
        stacked = {
            names[0]: torch.stack(run_a[parameter]).to(args.device, torch.float32),
            names[1]: torch.stack(run_b[parameter]).to(args.device, torch.float32),
        }
        displacement = {
            name: (values - values[0:1]).flatten(1)
            for name, values in stacked.items()
        }
        residual: dict[str, torch.Tensor] = {}
        for run_name, rows in displacement.items():
            residual[run_name], coefficients, captures = remove_scalar_orbit(
                rows, initial
            )
            for index, step in enumerate(steps):
                orbit_rows.append(
                    {
                        "parameter": parameter,
                        "object": run_name,
                        "probe_index": index,
                        "step": step,
                        "split": phase_for_step(
                            step, args.discovery_stop, args.validation_stop
                        ),
                        "scalar_coefficient": float(coefficients[index]),
                        "row_energy_capture": float(captures[index]),
                    }
                )

        common = 0.5 * (displacement[names[0]] + displacement[names[1]])
        private = 0.5 * (displacement[names[0]] - displacement[names[1]])
        common_residual, common_coefficients, common_captures = remove_scalar_orbit(
            common, initial
        )
        private_residual, private_coefficients, private_captures = remove_scalar_orbit(
            private, initial
        )
        for label, rows, coefficients, captures in (
            ("common", common, common_coefficients, common_captures),
            ("private", private, private_coefficients, private_captures),
        ):
            for index, step in enumerate(steps):
                orbit_rows.append(
                    {
                        "parameter": parameter,
                        "object": label,
                        "probe_index": index,
                        "step": step,
                        "split": phase_for_step(
                            step, args.discovery_stop, args.validation_stop
                        ),
                        "scalar_coefficient": float(coefficients[index]),
                        "row_energy_capture": float(captures[index]),
                    }
                )

        for split_name, indices in {**splits, "all": all_indices}.items():
            a_rows = displacement[names[0]][indices]
            b_rows = displacement[names[1]][indices]
            c_rows = common[indices]
            n_rows = private[indices]
            denominator_parts = (c_rows, n_rows)
            decomposition_rows.append(
                {
                    "parameter": parameter,
                    "split": split_name,
                    "common_energy_fraction": energy_fraction(c_rows, denominator_parts),
                    "private_energy_fraction": energy_fraction(n_rows, denominator_parts),
                    "stream_a_w0_capture": aggregate_capture(a_rows, initial),
                    "stream_b_w0_capture": aggregate_capture(b_rows, initial),
                    "common_w0_capture": aggregate_capture(c_rows, initial),
                    "private_w0_capture": aggregate_capture(n_rows, initial),
                    "common_residual_energy": float(
                        common_residual[indices].double().square().sum()
                    ),
                    "private_residual_energy": float(
                        private_residual[indices].double().square().sum()
                    ),
                }
            )

        residual_cosines = row_cosine(residual[names[0]], residual[names[1]])
        for index, step in enumerate(steps):
            matched_rows.append(
                {
                    "parameter": parameter,
                    "probe_index": index,
                    "step": step,
                    "split": phase_for_step(
                        step, args.discovery_stop, args.validation_stop
                    ),
                    "residualized_displacement_cosine": float(residual_cosines[index]),
                }
            )

        for source_name, target_name in ((names[0], names[1]), (names[1], names[0])):
            source = residual[source_name]
            target = residual[target_name]
            for fit_name, fit_rows, eval_splits in (
                ("full_source", source, {"all": all_indices}),
                (
                    "discovery_source",
                    source[splits["discovery"]],
                    {**splits, "all": all_indices},
                ),
            ):
                bases = nested_bases(fit_rows, max(ranks))
                for basis_kind, maximum_basis in bases.items():
                    for rank in ranks:
                        basis = maximum_basis[:, :rank]
                        for split_name, indices in eval_splits.items():
                            transfer_rows.append(
                                {
                                    "parameter": parameter,
                                    "source_run": source_name,
                                    "target_run": target_name,
                                    "fit_window": fit_name,
                                    "basis_kind": basis_kind,
                                    "rank": rank,
                                    "eval_split": split_name,
                                    "residualized_cross_capture": energy_capture(
                                        target[indices], basis
                                    ),
                                }
                            )

    common_all = [row for row in decomposition_rows if row["split"] == "all"]
    residual_binding = [
        row
        for row in transfer_rows
        if row["fit_window"] == "full_source"
        and row["basis_kind"] == "mean_plus_centered"
        and row["rank"] == 16
        and row["eval_split"] == "all"
    ]
    common_capture_minimum = min(float(row["common_w0_capture"]) for row in common_all)
    residual_capture_maximum = max(
        float(row["residualized_cross_capture"]) for row in residual_binding
    )
    gate = {
        "step_zero_bitwise_equal": all(row["bitwise_equal"] for row in identity_rows),
        "common_w0_capture_minimum": common_capture_minimum,
        "common_w0_capture_threshold": args.common_scalar_capture_gate,
        "residualized_full_source_rank16_cross_capture_maximum": residual_capture_maximum,
        "residualized_cross_capture_ceiling": args.residual_cross_capture_gate,
        "scalar_initial_weight_orbit_is_primary_shared_core": (
            common_capture_minimum >= args.common_scalar_capture_gate
            and residual_capture_maximum < args.residual_cross_capture_gate
        ),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    paths = {
        "identity": args.output / "step_zero_identity.csv",
        "orbit": args.output / "scalar_orbit_per_step.csv",
        "orbit_summary": args.output / "scalar_orbit_summary.csv",
        "decomposition": args.output / "common_private_decomposition.csv",
        "transfer": args.output / "residualized_basis_transfer.csv",
        "matched": args.output / "residualized_matched_cosine.csv",
        "matched_summary": args.output / "residualized_matched_cosine_summary.csv",
        "gate": args.output / "gate.json",
    }
    write_csv(paths["identity"], identity_rows)
    write_csv(paths["orbit"], orbit_rows)
    write_csv(
        paths["orbit_summary"],
        summarize_metric(orbit_rows, ("parameter", "object", "split"), "row_energy_capture"),
    )
    write_csv(paths["decomposition"], decomposition_rows)
    write_csv(paths["transfer"], transfer_rows)
    write_csv(paths["matched"], matched_rows)
    write_csv(
        paths["matched_summary"],
        summarize_metric(
            matched_rows,
            ("parameter", "split"),
            "residualized_displacement_cosine",
        ),
    )
    paths["gate"].write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    outputs = {
        name: {"path": str(path), "sha256": file_sha256(path)}
        for name, path in paths.items()
    }
    metadata = {
        "schema_version": "nanogpt_mlp_state_common_private_v1",
        "method": "common/private displacement and scalar-W0 orbit attribution",
        "source_commit": git_commit(Path(__file__).resolve().parents[2]),
        "entrypoint": str(Path(__file__).resolve()),
        "entrypoint_sha256": file_sha256(Path(__file__).resolve()),
        "command": [str(Path(__file__).resolve()), *sys.argv[1:]],
        "runs": {names[0]: metadata_a, names[1]: metadata_b},
        "steps": steps,
        "ranks": ranks,
        "binding_gate": gate,
        "outputs": outputs,
        "runtime_seconds": time.time() - started,
        "limitations": [
            "Scalar-W0 attribution tests only the cheapest initialization orbit.",
            "Euclidean common/private energy is not a task-metric guarantee.",
            "Two streams cannot determine asymptotic basis growth across all data paths.",
        ],
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"gate": gate, "metadata": str(metadata_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
