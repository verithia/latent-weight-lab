#!/usr/bin/env python3
"""Test whether c_proj follows the moving post-GELU activation subspace."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
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


def moving_frame_transport_metrics(
    source: torch.Tensor,
    target: torch.Tensor,
    start_basis: torch.Tensor,
    end_basis: torch.Tensor,
) -> dict[str, float | int]:
    """Apply the minimal principal-angle transport between two subspaces.

    If ``T`` is the minimum-displacement orthogonal map satisfying
    ``T @ start_basis ~= end_basis`` (up to principal-vector gauges), the
    forward weight prediction is ``source @ T.T``. The reverse control is
    ``source @ T``.
    """
    if source.ndim != 2 or source.shape != target.shape:
        raise ValueError("source and target must be same-shaped matrices")
    if (
        start_basis.ndim != 2
        or start_basis.shape != end_basis.shape
        or start_basis.shape[0] != source.shape[1]
    ):
        raise ValueError(
            "activation bases must share [source_width, rank] shape"
        )
    source = source.double()
    target = target.double()
    start_basis = start_basis.double()
    end_basis = end_basis.double()
    identity = torch.eye(
        start_basis.shape[1],
        device=start_basis.device,
        dtype=start_basis.dtype,
    )
    for name, basis in (
        ("start_basis", start_basis),
        ("end_basis", end_basis),
    ):
        error = (basis.T @ basis - identity).abs().max()
        if float(error) > 1e-4:
            raise ValueError(f"{name} is not orthonormal")

    left, cosine, right_h = torch.linalg.svd(
        start_basis.T @ end_basis,
        full_matrices=False,
    )
    cosine = cosine.clamp(0.0, 1.0)
    sine = (1.0 - cosine.square()).clamp_min(0.0).sqrt()
    start_principal = start_basis @ left
    end_principal = end_basis @ right_h.T
    active = sine > 1e-10
    complement = torch.zeros_like(start_principal)
    complement[:, active] = (
        end_principal[:, active]
        - start_principal[:, active] * cosine[active]
    ) / sine[active]

    source_start = source @ start_principal
    source_complement = source @ complement
    cosine_delta = cosine - 1.0
    common = (
        source
        + (source_start * cosine_delta) @ start_principal.T
        + (source_complement * cosine_delta) @ complement.T
    )
    forward = (
        common
        + (source_start * sine) @ complement.T
        - (source_complement * sine) @ start_principal.T
    )
    reverse = (
        common
        - (source_start * sine) @ complement.T
        + (source_complement * sine) @ start_principal.T
    )

    chord_energy = (target - source).square().sum().clamp_min(1e-30)

    def recovery(prediction: torch.Tensor) -> float:
        residual = (target - prediction).square().sum()
        return float(1.0 - residual / chord_energy)

    mapped_start = (
        start_principal * cosine
        + complement * sine
    )
    mapping_error = float(
        (mapped_start - end_principal).square().sum().sqrt()
    )
    plane_error = max(
        float(
            (
                start_principal.T @ complement
            ).abs().max()
        ),
        float(
            (
                complement[:, active].T @ complement[:, active]
                - torch.eye(
                    int(active.sum()),
                    device=source.device,
                    dtype=source.dtype,
                )
            ).abs().max()
        )
        if bool(active.any())
        else 0.0,
    )
    forward_recovery = recovery(forward)
    reverse_recovery = recovery(reverse)
    return {
        "rank": int(start_basis.shape[1]),
        "forward_transport_recovery": forward_recovery,
        "reverse_transport_recovery": reverse_recovery,
        "forward_minus_reverse_recovery": (
            forward_recovery - reverse_recovery
        ),
        "mean_principal_cosine": float(cosine.mean()),
        "minimum_principal_cosine": float(cosine.min()),
        "mean_subspace_overlap": float(cosine.square().mean()),
        "principal_angle_rms_radians": float(
            torch.acos(cosine).square().mean().sqrt()
        ),
        "principal_mapping_error_fro": mapping_error,
        "principal_plane_orthogonality_error": plane_error,
        "start_and_end_basis_values": int(
            start_basis.numel() + end_basis.numel()
        ),
        "start_and_end_basis_storage_fraction_of_cproj": float(
            (start_basis.numel() + end_basis.numel()) / source.numel()
        ),
        "chord_fro": float(chord_energy.sqrt()),
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

            forward = weighted("forward_transport_recovery")
            reverse = weighted("reverse_transport_recovery")
            aggregates.append(
                {
                    "basis_window": window,
                    "rank": rank,
                    "cells": len(selected),
                    "energy_weighted_forward_transport_recovery": forward,
                    "energy_weighted_reverse_transport_recovery": reverse,
                    "energy_weighted_forward_minus_reverse_recovery": (
                        forward - reverse
                    ),
                    "energy_weighted_fraction_of_right_oracle": weighted(
                        "fraction_of_right_oracle"
                    ),
                    "energy_weighted_mean_subspace_overlap": weighted(
                        "mean_subspace_overlap"
                    ),
                    "forward_recovery_over_random_givens32": (
                        forward / RANDOM_BASELINE[32]
                    ),
                    "minimum_forward_transport_recovery": min(
                        float(row["forward_transport_recovery"])
                        for row in selected
                    ),
                    "minimum_recovery_over_random_givens32": min(
                        float(row["forward_transport_recovery"])
                        / RANDOM_BASELINE[32]
                        for row in selected
                    ),
                    "forward_over_positive_reverse": (
                        forward / max(reverse, 1e-30)
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
                float(row["forward_recovery_over_random_givens32"])
                for row in candidates
            )
            >= 2.5
            and min(
                float(row["minimum_recovery_over_random_givens32"])
                for row in candidates
            )
            >= 1.5
            and all(
                float(row["energy_weighted_forward_transport_recovery"])
                >= 1.25
                * max(
                    float(row["energy_weighted_reverse_transport_recovery"]),
                    0.0,
                )
                for row in candidates
            )
        ):
            selected_rank = rank
            break
    decision = (
        f"PROMOTE_MOVING_ACTIVATION_FRAME_RANK{selected_rank}"
        if selected_rank is not None
        else "REJECT_MOVING_ACTIVATION_FRAME_TRANSPORT"
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
    parser.add_argument("--pca-seed", type=int, default=424244)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    cells = parse_cells(args.cells)
    boundaries = parse_int_list(args.phase_boundaries)
    ranks = parse_int_list(args.ranks)
    end_by_start = dict(zip(boundaries[:-1], boundaries[1:], strict=True))
    required_steps = sorted(
        {
            step
            for _layer, phase in cells
            for step in (phase, end_by_start[phase])
        }
    )
    layers_by_step: dict[int, set[int]] = defaultdict(set)
    for layer, phase in cells:
        layers_by_step[phase].add(layer)
        layers_by_step[end_by_start[phase]].add(layer)
    snapshot_paths = [
        args.snapshot_dir / f"step_{step:06d}.pt"
        for step in required_steps
    ]
    if any(not path.is_file() for path in snapshot_paths):
        raise ValueError("required phase snapshots are absent")
    steps, values, snapshot_metadata = load_snapshots(
        snapshot_paths,
        layers={layer for layer, _phase in cells},
        targets={"mlp.c_proj"},
    )
    step_index = {step: index for index, step in enumerate(steps)}

    batches_needed = (
        args.sample_cap + args.batch_size * args.block_size - 1
    ) // (args.batch_size * args.block_size)
    windows = (
        ("fit", args.fit_seed),
        ("holdout", args.holdout_seed),
    )
    batches_by_window = {
        window: fixed_validation_batches(
            args.data_dir,
            args.batch_size,
            args.block_size,
            batches_needed,
            seed,
        )
        for window, seed in windows
    }
    maximum_rank = max(ranks)
    bases: dict[tuple[str, int, int], torch.Tensor] = {}
    basis_metadata: dict[str, Any] = {}
    for step in required_steps:
        payload = load_snapshot(
            args.snapshot_dir / f"step_{step:06d}.pt"
        )
        model = model_from_snapshot(payload, args.device)
        layers = sorted(layers_by_step[step])
        try:
            for window_index, (window, _seed) in enumerate(windows):
                activations = collect_activations(
                    model,
                    batches_by_window[window],
                    layers,
                    args.sample_cap,
                    args.device,
                )
                for layer in layers:
                    basis, singular, total = randomized_principal_basis(
                        activations[(layer, "post_gelu")].to(args.device),
                        maximum_rank,
                        center=True,
                        seed=(
                            args.pca_seed
                            + 1009 * layer
                            + 100_003 * step
                            + window_index
                        ),
                    )
                    basis = torch.linalg.qr(
                        basis.double(), mode="reduced"
                    ).Q.float()
                    bases[(window, step, layer)] = basis.cpu()
                    key = f"{window}_step{step}_layer{layer}"
                    basis_metadata[key] = {
                        "sha256": tensor_sha256(basis),
                        "top256_activation_energy_fraction": float(
                            singular.square().sum() / max(total, 1e-30)
                        ),
                    }
        finally:
            del model, payload
            if "cuda" in args.device:
                torch.cuda.empty_cache()

    rows: list[dict[str, Any]] = []
    for layer, phase_start in cells:
        parameter = f"transformer.h.{layer}.mlp.c_proj.weight"
        phase_end = end_by_start[phase_start]
        source = values[parameter][step_index[phase_start]].to(args.device)
        target = values[parameter][step_index[phase_end]].to(args.device)
        right_oracle = orthogonal_transport_metrics(
            source, target
        )["right_endpoint_recovery"]
        for window, _seed in windows:
            for rank in ranks:
                start_basis = bases[
                    (window, phase_start, layer)
                ][:, :rank].to(args.device)
                end_basis = bases[
                    (window, phase_end, layer)
                ][:, :rank].to(args.device)
                metrics = moving_frame_transport_metrics(
                    source, target, start_basis, end_basis
                )
                row = {
                    "parameter": parameter,
                    "layer": layer,
                    "phase_start": phase_start,
                    "phase_end": phase_end,
                    "basis_window": window,
                    "start_basis_sha256": tensor_sha256(start_basis),
                    "end_basis_sha256": tensor_sha256(end_basis),
                    "right_oracle_recovery": right_oracle,
                    **metrics,
                }
                row["fraction_of_right_oracle"] = (
                    float(row["forward_transport_recovery"])
                    / float(right_oracle)
                )
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
        del source, target

    aggregates, decision, selected_rank = aggregate(rows, ranks)
    args.output.mkdir(parents=True, exist_ok=True)
    detail_path = args.output / "moving_activation_frame_oracle.csv"
    aggregate_path = (
        args.output / "moving_activation_frame_oracle_aggregate.csv"
    )
    basis_path = args.output / "moving_activation_frame_bases.pt"
    write_csv(detail_path, rows)
    write_csv(aggregate_path, aggregates)
    torch.save(
        {
            "schema_version": "moving_activation_frame_bases_v1",
            "fit_seed": args.fit_seed,
            "holdout_seed": args.holdout_seed,
            "maximum_rank": maximum_rank,
            "bases": {
                f"{window}_step{step}_layer{layer}": value
                for (window, step, layer), value in bases.items()
            },
            "metadata": basis_metadata,
        },
        basis_path,
    )
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_mlp_moving_activation_frame_oracle_v1",
        "decision": decision,
        "selected_rank": selected_rank,
        "decision_rule": (
            "select the smallest rank for which both disjoint activation "
            "windows recover >=2.5x the registered random 32-stage Givens "
            "aggregate, every pilot cell recovers >=1.5x, and forward "
            "transport is >=1.25x the positive reverse control"
        ),
        "operator": (
            "minimum-displacement principal-angle T carrying the phase-start "
            "post-GELU PCA subspace to the phase-end subspace; predict W1 as "
            "W0 @ T.T"
        ),
        "basis_policy": (
            "phase-local deterministic post-GELU PCA used only for diagnosis; "
            "no learned dense basis or LoRA factor"
        ),
        "cells": [
            {"layer": layer, "phase_start": phase}
            for layer, phase in cells
        ],
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
            "This is a future-endpoint activation-frame oracle, not a causal training result.",
            "The dense PCA frames are diagnostic observations, not proposed trainable parameters.",
            "Only four preregistered phase/layer cells are screened.",
        ],
    }
    metadata_path = (
        args.output / "moving_activation_frame_oracle_metadata.json"
    )
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
