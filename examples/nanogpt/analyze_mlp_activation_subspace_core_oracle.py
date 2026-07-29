#!/usr/bin/env python3
"""Fit a small orthogonal core inside a fixed activation-derived subspace."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.nanogpt.analyze_mlp_activation_matched_givens_fit import (
    RANDOM_BASELINE,
    tensor_sha256,
)
from examples.nanogpt.analyze_mlp_activation_update_alignment import (
    collect_activations,
    file_sha256,
    git_commit,
    load_snapshot,
    model_from_snapshot,
    randomized_principal_basis,
)
from examples.nanogpt.analyze_mlp_global_givens_transport_fit import (
    parse_cells,
)
from examples.nanogpt.analyze_mlp_orthogonal_transport_oracle import (
    orthogonal_transport_metrics,
)
from examples.nanogpt.analyze_parameter_trajectory import (
    load_snapshots,
    parse_int_list,
    write_csv,
)
from examples.nanogpt.analyze_residual_compatibility import (
    fixed_validation_batches,
)


def fixed_subspace_core_metrics(
    source: torch.Tensor,
    target: torch.Tensor,
    basis: torch.Tensor,
) -> dict[str, float | int]:
    """Return exact orthogonal and unconstrained core endpoint recovery.

    The materialized right transform is

    ``R = I + V (G - I) V.T``

    for fixed orthonormal ``V`` and a fitted small core ``G``.
    """
    if source.ndim != 2 or source.shape != target.shape:
        raise ValueError("source and target must be same-shaped matrices")
    if basis.ndim != 2 or basis.shape[0] != source.shape[1]:
        raise ValueError("basis must have shape [source_width, rank]")
    source = source.double()
    target = target.double()
    basis = basis.double()
    orthogonality_error = (
        basis.T @ basis
        - torch.eye(basis.shape[1], device=basis.device, dtype=basis.dtype)
    ).abs().max()
    if float(orthogonality_error) > 1e-5:
        raise ValueError("basis is not orthonormal")
    delta = target - source
    chord_energy = delta.square().sum().clamp_min(1e-30)
    source_coordinates = source @ basis
    target_coordinates = target @ basis

    cross = source_coordinates.T @ target_coordinates
    left, _singular, right_h = torch.linalg.svd(
        cross, full_matrices=False
    )
    orthogonal_core = left @ right_h
    orthogonal_prediction = source + (
        source_coordinates @ orthogonal_core - source_coordinates
    ) @ basis.T
    orthogonal_residual = (
        target - orthogonal_prediction
    ).square().sum()

    unconstrained_core = torch.linalg.lstsq(
        source_coordinates, target_coordinates
    ).solution
    unconstrained_prediction = source + (
        source_coordinates @ unconstrained_core - source_coordinates
    ) @ basis.T
    unconstrained_residual = (
        target - unconstrained_prediction
    ).square().sum()

    projection_energy = (delta @ basis).square().sum()

    def recovery(residual: torch.Tensor) -> float:
        return float(1.0 - residual / chord_energy)

    orthogonal_recovery = recovery(orthogonal_residual)
    projection_upper = float(projection_energy / chord_energy)
    return {
        "rank": int(basis.shape[1]),
        "basis_values": int(basis.numel()),
        "basis_storage_fraction_of_cproj": float(
            basis.numel() / source.numel()
        ),
        "orthogonal_core_coordinates": int(
            basis.shape[1] * (basis.shape[1] - 1) // 2
        ),
        "orthogonal_core_trainable_fraction_of_cproj": float(
            (
                basis.shape[1] * (basis.shape[1] - 1) // 2
            )
            / source.numel()
        ),
        "projection_upper_recovery": projection_upper,
        "orthogonal_core_recovery": orthogonal_recovery,
        "unconstrained_core_recovery": recovery(unconstrained_residual),
        "orthogonal_fraction_of_projection_upper": (
            orthogonal_recovery / max(projection_upper, 1e-30)
        ),
        "orthogonal_residual_fro": float(orthogonal_residual.sqrt()),
        "unconstrained_residual_fro": float(
            unconstrained_residual.sqrt()
        ),
        "chord_fro": float(chord_energy.sqrt()),
        "orthogonal_core_distance_from_identity": float(
            (
                orthogonal_core
                - torch.eye(
                    basis.shape[1],
                    device=basis.device,
                    dtype=basis.dtype,
                )
            ).square().sum().sqrt()
        ),
    }


def aggregate(
    rows: list[dict[str, Any]],
    ranks: list[int],
) -> tuple[list[dict[str, Any]], str, int | None]:
    aggregates: list[dict[str, Any]] = []
    for window in ("fit", "holdout"):
        for rank in ranks:
            selected = [
                row
                for row in rows
                if row["basis_window"] == window
                and int(row["rank"]) == rank
            ]
            energy = torch.tensor(
                [float(row["chord_fro"]) ** 2 for row in selected],
                dtype=torch.float64,
            )

            def weighted(key: str) -> float:
                values = torch.tensor(
                    [float(row[key]) for row in selected],
                    dtype=torch.float64,
                )
                return float((energy * values).sum() / energy.sum())

            recovery = weighted("orthogonal_core_recovery")
            aggregates.append(
                {
                    "basis_window": window,
                    "rank": rank,
                    "cells": len(selected),
                    "basis_values_per_layer": int(
                        selected[0]["basis_values"]
                    ),
                    "basis_storage_fraction_of_cproj": float(
                        selected[0]["basis_storage_fraction_of_cproj"]
                    ),
                    "orthogonal_core_coordinates_per_layer": int(
                        selected[0]["orthogonal_core_coordinates"]
                    ),
                    "orthogonal_core_trainable_fraction_of_cproj": float(
                        selected[0][
                            "orthogonal_core_trainable_fraction_of_cproj"
                        ]
                    ),
                    "energy_weighted_orthogonal_core_recovery": recovery,
                    "energy_weighted_projection_upper_recovery": weighted(
                        "projection_upper_recovery"
                    ),
                    "energy_weighted_unconstrained_core_recovery": weighted(
                        "unconstrained_core_recovery"
                    ),
                    "energy_weighted_fraction_of_right_oracle": weighted(
                        "fraction_of_right_oracle"
                    ),
                    "recovery_over_random_givens32": (
                        recovery / RANDOM_BASELINE[32]
                    ),
                    "minimum_orthogonal_core_recovery": min(
                        float(row["orthogonal_core_recovery"])
                        for row in selected
                    ),
                    "minimum_recovery_over_random_givens32": min(
                        float(row["orthogonal_core_recovery"])
                        / RANDOM_BASELINE[32]
                        for row in selected
                    ),
                }
            )
    selected_rank: int | None = None
    for rank in ranks:
        candidates = [
            row for row in aggregates if int(row["rank"]) == rank
        ]
        if (
            len(candidates) == 2
            and min(
                float(row["recovery_over_random_givens32"])
                for row in candidates
            )
            >= 2.0
            and min(
                float(row["minimum_recovery_over_random_givens32"])
                for row in candidates
            )
            >= 1.5
        ):
            selected_rank = rank
            break
    decision = (
        f"PROMOTE_FIXED_ACTIVATION_SUBSPACE_CORE_RANK{selected_rank}"
        if selected_rank is not None
        else "REJECT_FIXED_ACTIVATION_SUBSPACE_CORE"
    )
    return aggregates, decision, selected_rank


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cells", default="0:0,0:180,6:60,11:120")
    parser.add_argument("--phase-boundaries", default="0,60,120,180,238")
    parser.add_argument("--ranks", default="64,128,256")
    parser.add_argument("--sample-cap", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--fit-seed", type=int, default=20260729)
    parser.add_argument("--holdout-seed", type=int, default=20260730)
    parser.add_argument("--pca-seed", type=int, default=424243)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    cells = parse_cells(args.cells)
    boundaries = parse_int_list(args.phase_boundaries)
    ranks = parse_int_list(args.ranks)
    end_by_start = dict(zip(boundaries[:-1], boundaries[1:], strict=True))
    layers = sorted({layer for layer, _phase in cells})
    required_steps = sorted(
        {step for _layer, phase in cells for step in (phase, end_by_start[phase])}
    )
    paths = [args.snapshot_dir / f"step_{step:06d}.pt" for step in required_steps]
    if any(not path.is_file() for path in paths):
        raise ValueError("required phase snapshots are absent")
    steps, values, snapshot_metadata = load_snapshots(
        paths,
        layers=set(layers),
        targets={"mlp.c_proj"},
    )
    step_index = {step: index for index, step in enumerate(steps)}

    initial_payload = load_snapshot(args.snapshot_dir / "step_000000.pt")
    model = model_from_snapshot(initial_payload, args.device)
    batches_needed = (
        args.sample_cap + args.batch_size * args.block_size - 1
    ) // (args.batch_size * args.block_size)
    maximum_rank = max(ranks)
    bases: dict[tuple[str, int], torch.Tensor] = {}
    basis_metadata: dict[str, Any] = {}
    try:
        for window_index, (window, seed) in enumerate(
            (
                ("fit", args.fit_seed),
                ("holdout", args.holdout_seed),
            )
        ):
            batches = fixed_validation_batches(
                args.data_dir,
                args.batch_size,
                args.block_size,
                batches_needed,
                seed,
            )
            activations = collect_activations(
                model,
                batches,
                layers,
                args.sample_cap,
                args.device,
            )
            for layer in layers:
                basis, singular, total = randomized_principal_basis(
                    activations[(layer, "post_gelu")].to(args.device),
                    maximum_rank,
                    center=True,
                    seed=args.pca_seed + 1009 * layer + window_index,
                )
                bases[(window, layer)] = basis.cpu()
                basis_metadata[f"{window}_layer{layer}"] = {
                    "sha256": tensor_sha256(basis),
                    "top256_activation_energy_fraction": float(
                        singular.square().sum() / max(total, 1e-30)
                    ),
                }
    finally:
        del model, initial_payload
        if "cuda" in args.device:
            torch.cuda.empty_cache()

    rows: list[dict[str, Any]] = []
    for layer, phase_start in cells:
        name = f"transformer.h.{layer}.mlp.c_proj.weight"
        phase_end = end_by_start[phase_start]
        source = values[name][step_index[phase_start]].to(args.device)
        target = values[name][step_index[phase_end]].to(args.device)
        right_oracle = orthogonal_transport_metrics(
            source, target
        )["right_endpoint_recovery"]
        for window in ("fit", "holdout"):
            for rank in ranks:
                selected_basis = bases[(window, layer)][:, :rank].to(
                    args.device
                )
                metrics = fixed_subspace_core_metrics(
                    source, target, selected_basis
                )
                row = {
                    "parameter": name,
                    "layer": layer,
                    "phase_start": phase_start,
                    "phase_end": phase_end,
                    "basis_source_step": 0,
                    "basis_window": window,
                    "basis_sha256": tensor_sha256(selected_basis),
                    "right_oracle_recovery": right_oracle,
                    **metrics,
                }
                row["fraction_of_right_oracle"] = (
                    float(row["orthogonal_core_recovery"])
                    / float(right_oracle)
                )
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
        del source, target

    aggregates, decision, selected_rank = aggregate(rows, ranks)
    args.output.mkdir(parents=True, exist_ok=True)
    detail_path = args.output / "activation_subspace_core_oracle.csv"
    aggregate_path = args.output / "activation_subspace_core_oracle_aggregate.csv"
    basis_path = args.output / "activation_subspace_bases.pt"
    write_csv(detail_path, rows)
    write_csv(aggregate_path, aggregates)
    torch.save(
        {
            "schema_version": "activation_subspace_bases_v1",
            "source_step": 0,
            "fit_seed": args.fit_seed,
            "holdout_seed": args.holdout_seed,
            "maximum_rank": maximum_rank,
            "bases": {
                f"{window}_layer{layer}": value
                for (window, layer), value in bases.items()
            },
            "metadata": basis_metadata,
        },
        basis_path,
    )
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_mlp_activation_subspace_core_oracle_v1",
        "decision": decision,
        "selected_rank": selected_rank,
        "decision_rule": (
            "select the smallest rank for which both disjoint step-0 basis "
            "windows recover >=2x the registered random 32-stage Givens "
            "aggregate and every pilot cell recovers >=1.5x that baseline"
        ),
        "operator": "I + V @ (G - I) @ V.T",
        "basis_policy": (
            "fixed non-trainable post-GELU PCA from deterministic step-0 "
            "activations; no learned dense basis or LoRA factor"
        ),
        "cells": [{"layer": layer, "phase_start": phase} for layer, phase in cells],
        "ranks": ranks,
        "basis_metadata": basis_metadata,
        "snapshot_metadata": snapshot_metadata,
        "analysis_execution": {
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": str(script),
            "entrypoint_sha256": file_sha256(script),
            "command": sys.argv,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "device": args.device,
        },
        "outputs": {
            "detail_sha256": file_sha256(detail_path),
            "aggregate_sha256": file_sha256(aggregate_path),
            "basis_sha256": file_sha256(basis_path),
        },
        "limitations": [
            "The orthogonal core is an exact endpoint-fit oracle, not a language-model training result.",
            "The fixed basis has conceptual storage even though it has no trainable parameters.",
            "Only four preregistered phase/layer cells are screened before implementation.",
        ],
    }
    metadata_path = args.output / "activation_subspace_core_oracle_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "rows": len(rows),
                "aggregates": aggregates,
                "decision": decision,
                "selected_rank": selected_rank,
                "metadata": str(metadata_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
