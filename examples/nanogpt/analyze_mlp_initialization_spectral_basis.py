#!/usr/bin/env python3
"""Measure common MLP path bases in the exact initialization singular gauge."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_mlp_disjoint_data_gradient_transfer import git_commit
from examples.nanogpt.analyze_mlp_disjoint_data_state_transfer import load_weight_run
from examples.nanogpt.analyze_mlp_highcadence_basis import (
    file_sha256,
    parse_float_list,
)
from examples.nanogpt.analyze_mlp_tangent_drift import temporal_basis
from examples.nanogpt.analyze_parameter_trajectory import write_csv


def common_positions(
    first: list[torch.Tensor], second: list[torch.Tensor]
) -> torch.Tensor:
    if len(first) != len(second) or not torch.equal(first[0], second[0]):
        raise ValueError("common-gauge paths require equal lengths and identical W0")
    a = torch.stack(first).float()
    b = torch.stack(second).float()
    return a[0] + 0.5 * ((a - a[0]) + (b - b[0]))


def spectral_basis_metrics(
    base: torch.Tensor,
    basis_matrices: torch.Tensor,
    eigenvalues: torch.Tensor,
    *,
    total_ratios: list[float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if base.ndim != 2 or basis_matrices.ndim != 3:
        raise ValueError("base and basis matrices must be rank two and three")
    if basis_matrices.shape[1:] != base.shape:
        raise ValueError("basis/base shape mismatch")
    if len(eigenvalues) != len(basis_matrices):
        raise ValueError("one eigenvalue per basis matrix is required")
    u, _singular, vh = torch.linalg.svd(base.float(), full_matrices=False)
    v = vh.T
    cores = torch.einsum("ra,kab,bs->krs", u.T, basis_matrices.float(), v)
    full_energy = basis_matrices.double().square().sum(dim=(1, 2)).clamp_min(1e-30)
    core_energy = cores.double().square().sum(dim=(1, 2))
    diagonal_energy = torch.diagonal(cores, dim1=1, dim2=2).double().square().sum(dim=1)
    weights = eigenvalues.double() / eigenvalues.double().sum().clamp_min(1e-30)
    thin_capture = core_energy / full_energy
    diagonal_capture = diagonal_energy / full_energy
    overview = {
        "thin_frame_weighted_capture": float((weights * thin_capture).sum()),
        "thin_frame_minimum_capture": float(thin_capture.min()),
        "thin_frame_maximum_capture": float(thin_capture.max()),
        "spectral_diagonal_weighted_capture": float(
            (weights * diagonal_capture).sum()
        ),
        "spectral_diagonal_minimum_capture": float(diagonal_capture.min()),
        "spectral_diagonal_total_stored_scalar_fraction": (
            len(basis_matrices) * min(base.shape) / base.numel()
        ),
    }

    rows: list[dict[str, Any]] = []
    normalized_pc_energy = (
        cores.double().square()
        / full_energy.view(-1, 1, 1)
        * weights.view(-1, 1, 1)
    )
    shared_coordinate_energy = normalized_pc_energy.sum(dim=0).flatten()
    independent_coordinate_energy = normalized_pc_energy.flatten(1)
    for ratio in total_ratios:
        per_basis_coordinates = min(
            cores.shape[1] * cores.shape[2],
            max(1, math.floor(ratio * base.numel() / len(basis_matrices))),
        )
        shared_capture = torch.topk(
            shared_coordinate_energy, per_basis_coordinates, sorted=False
        ).values.sum()
        independent_capture = sum(
            torch.topk(row, per_basis_coordinates, sorted=False).values.sum()
            for row in independent_coordinate_energy
        )
        resolved = len(basis_matrices) * per_basis_coordinates / base.numel()
        rows.extend(
            (
                {
                    "family": "shared_initialization_spectral_support",
                    "total_ratio_requested": ratio,
                    "per_basis_coordinates": per_basis_coordinates,
                    "total_stored_scalar_fraction": resolved,
                    "weighted_basis_energy_capture": float(shared_capture),
                },
                {
                    "family": "independent_initialization_spectral_support_oracle",
                    "total_ratio_requested": ratio,
                    "per_basis_coordinates": per_basis_coordinates,
                    "total_stored_scalar_fraction": resolved,
                    "weighted_basis_energy_capture": float(independent_capture),
                },
            )
        )
    return overview, rows


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
    parser.add_argument("--total-ratios", default="0.001,0.0025,0.005,0.01")
    parser.add_argument("--weighted-capture-gate", type=float, default=0.90)
    parser.add_argument("--thin-frame-gate", type=float, default=0.95)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    targets = {value for value in args.targets.split(",") if value}
    ratios = parse_float_list(args.total_ratios)
    if args.basis_rank < 1 or not targets or any(not 0 < value <= 0.01 for value in ratios):
        raise ValueError("invalid basis rank, targets, or total ratios")

    steps_a, run_a, metadata_a = load_weight_run(
        args.run_a_probe_dir, layer=args.layer, targets=targets
    )
    steps_b, run_b, metadata_b = load_weight_run(
        args.run_b_probe_dir, layer=args.layer, targets=targets
    )
    if steps_a != steps_b or set(run_a) != set(run_b):
        raise ValueError("paired runs require identical steps and inventory")

    overview_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    for parameter in sorted(run_a):
        equal = torch.equal(run_a[parameter][0], run_b[parameter][0])
        identity_rows.append({"parameter": parameter, "bitwise_equal": equal})
        if not equal:
            raise ValueError(f"step-zero mismatch for {parameter}")
        positions = common_positions(run_a[parameter], run_b[parameter]).to(
            args.device
        )
        _centered, all_eigenvalues, basis = temporal_basis(
            positions.flatten(1), maximum_rank=args.basis_rank
        )
        available = min(args.basis_rank, basis.shape[1])
        eigenvalues = all_eigenvalues[:available]
        basis_matrices = basis[:, :available].T.reshape(
            available, *positions.shape[1:]
        )
        overview, rows = spectral_basis_metrics(
            positions[0],
            basis_matrices,
            eigenvalues,
            total_ratios=ratios,
        )
        common = {
            "parameter": parameter,
            "basis_rank": available,
            "basis_energy_fraction": float(
                eigenvalues.sum() / all_eigenvalues.sum().clamp_min(1e-30)
            ),
        }
        overview_rows.append({**common, **overview})
        support_rows.extend({**common, **row} for row in rows)
        del positions, basis_matrices
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

    one_percent = [
        row
        for row in support_rows
        if abs(float(row["total_ratio_requested"]) - 0.01) < 1e-12
    ]
    best_by_parameter = {
        parameter: max(
            (
                row
                for row in one_percent
                if row["parameter"] == parameter
            ),
            key=lambda row: float(row["weighted_basis_energy_capture"]),
        )
        for parameter in sorted(run_a)
    }
    gate = {
        "step_zero_bitwise_equal": all(row["bitwise_equal"] for row in identity_rows),
        "thin_frame_capture_minimum": min(
            float(row["thin_frame_weighted_capture"]) for row in overview_rows
        ),
        "thin_frame_capture_threshold": args.thin_frame_gate,
        "best_one_percent_capture_minimum": min(
            float(row["weighted_basis_energy_capture"])
            for row in best_by_parameter.values()
        ),
        "weighted_basis_capture_threshold": args.weighted_capture_gate,
        "best_one_percent_by_parameter": best_by_parameter,
    }
    gate["initialization_spectral_basis_authorized"] = bool(
        gate["step_zero_bitwise_equal"]
        and gate["thin_frame_capture_minimum"] >= args.thin_frame_gate
        and gate["best_one_percent_capture_minimum"] >= args.weighted_capture_gate
    )

    args.output.mkdir(parents=True, exist_ok=True)
    outputs = {
        "identity": args.output / "step_zero_identity.csv",
        "overview": args.output / "initialization_spectral_overview.csv",
        "support": args.output / "initialization_spectral_support.csv",
        "gate": args.output / "gate.json",
    }
    for path, rows in (
        (outputs["identity"], identity_rows),
        (outputs["overview"], overview_rows),
        (outputs["support"], support_rows),
    ):
        write_csv(path, rows)
    outputs["gate"].write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_mlp_initialization_spectral_basis_v1",
        "source_commit": git_commit(script.parents[2]),
        "entrypoint": str(script),
        "entrypoint_sha256": file_sha256(script),
        "command": sys.argv,
        "runs": {args.run_a_name: metadata_a, args.run_b_name: metadata_b},
        "steps": steps_a,
        "basis_rank": args.basis_rank,
        "total_ratios": ratios,
        "binding_gate": gate,
        "runtime_seconds": time.time() - started,
        "outputs": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in outputs.items()
        },
        "limitations": [
            "Exact initialization singular frames are charged as free and are not a fast decoder.",
            "Support indices and fitting compute are omitted.",
            "This is a noncausal representation ceiling, not a training result.",
        ],
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "gate": gate,
                "metadata": str(metadata_path),
                "metadata_sha256": file_sha256(metadata_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
