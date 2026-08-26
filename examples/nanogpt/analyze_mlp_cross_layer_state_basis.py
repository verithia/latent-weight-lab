#!/usr/bin/env python3
"""Measure whether dense temporal MLP bases can be shared across layers.

The input is a registered high-cadence state trajectory.  Each layer/matrix
first receives its own descriptive temporal PCA basis.  The audit then asks:

1. how much state variance one layer's basis captures in another layer;
2. whether one shared ambient basis efficiently spans several layers; and
3. whether c_fc and transposed c_proj use a common paired basis.

All fits use the full horizon and are optimistic representation oracles, not
causal training evidence.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_mlp_highcadence_basis import (
    file_sha256,
)
from examples.nanogpt.analyze_mlp_tangent_drift import temporal_basis
from examples.nanogpt.analyze_parameter_trajectory import (
    PARAMETER_PATTERN,
    load_snapshots,
    parse_int_list,
    write_csv,
)


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def weighted_subspace_capture(
    source_basis: torch.Tensor,
    target_basis: torch.Tensor,
    target_eigenvalues: torch.Tensor,
) -> float:
    """Variance-weighted target capture by an orthonormal source basis."""
    overlaps = target_basis.T @ source_basis
    per_target_pc = overlaps.double().square().sum(dim=1)
    weights = target_eigenvalues.double()
    return float((weights * per_target_pc).sum() / weights.sum().clamp_min(1e-30))


def canonical_metrics(
    left_basis: torch.Tensor, right_basis: torch.Tensor
) -> tuple[float, float, float]:
    singular_values = torch.linalg.svdvals(left_basis.T @ right_basis).double()
    squared = singular_values.square()
    return float(squared.mean()), float(squared.min()), float(squared.max())


def component_gram(
    bases: list[torch.Tensor], eigenvalues: list[torch.Tensor]
) -> tuple[torch.Tensor, list[slice]]:
    offsets: list[slice] = []
    start = 0
    for basis in bases:
        offsets.append(slice(start, start + basis.shape[1]))
        start += basis.shape[1]
    gram = bases[0].new_zeros((start, start), dtype=torch.float64)
    for left_index, (left_basis, left_values) in enumerate(
        zip(bases, eigenvalues)
    ):
        left_slice = offsets[left_index]
        left_scale = left_values.double().sqrt()
        for right_index in range(left_index, len(bases)):
            right_basis = bases[right_index]
            right_values = eigenvalues[right_index]
            right_slice = offsets[right_index]
            overlap = (left_basis.T @ right_basis).double()
            block = left_scale.unsqueeze(1) * overlap * right_values.double().sqrt().unsqueeze(0)
            gram[left_slice, right_slice] = block
            gram[right_slice, left_slice] = block.T
    return (gram + gram.T) * 0.5, offsets


def shared_basis_rows(
    *,
    target: str,
    layers: list[int],
    bases: list[torch.Tensor],
    eigenvalues: list[torch.Tensor],
    ranks: list[int],
    parameter_size: int,
) -> list[dict[str, Any]]:
    """Evaluate a globally optimal ambient basis via a small component Gram."""
    gram, offsets = component_gram(bases, eigenvalues)
    shared_values, shared_vectors = torch.linalg.eigh(gram)
    order = torch.argsort(shared_values, descending=True)
    shared_values = shared_values[order].clamp_min(0.0)
    shared_vectors = shared_vectors[:, order]
    total = shared_values.sum().clamp_min(1e-30)
    rows: list[dict[str, Any]] = []
    for requested_rank in ranks:
        rank = min(requested_rank, int((shared_values > shared_values[0] * 1e-10).sum()))
        if rank == 0:
            continue
        vectors = shared_vectors[:, :rank]
        values = shared_values[:rank].clamp_min(1e-30)
        aggregate_capture = float(values.sum() / total)
        layer_captures: dict[int, float] = {}
        for layer_index, (layer, basis, layer_values) in enumerate(
            zip(layers, bases, eigenvalues)
        ):
            # U_l^T A, where A concatenates U_j diag(sqrt(lambda_j)).
            blocks = []
            for other_basis, other_values in zip(bases, eigenvalues):
                blocks.append(
                    (basis.T @ other_basis).double()
                    * other_values.double().sqrt().unsqueeze(0)
                )
            target_to_components = torch.cat(blocks, dim=1)
            target_to_shared = (
                target_to_components @ vectors
            ) / values.sqrt().unsqueeze(0)
            per_pc = target_to_shared.square().sum(dim=1)
            layer_captures[layer] = float(
                (layer_values.double() * per_pc).sum()
                / layer_values.double().sum().clamp_min(1e-30)
            )
        rows.append(
            {
                "target": target,
                "shared_rank_requested": requested_rank,
                "shared_rank_resolved": rank,
                "layers": ",".join(str(layer) for layer in layers),
                "aggregate_state_energy_capture": aggregate_capture,
                "minimum_layer_capture": min(layer_captures.values()),
                "maximum_layer_capture": max(layer_captures.values()),
                "mean_layer_capture": sum(layer_captures.values()) / len(layer_captures),
                "per_layer_capture_json": json.dumps(layer_captures, sort_keys=True),
                "stored_basis_scalars": rank * parameter_size,
                "stored_basis_fraction_of_one_matrix": float(rank),
                "stored_basis_fraction_of_all_target_matrices": rank / len(layers),
            }
        )
    return rows


def transpose_basis(
    basis: torch.Tensor, rows: int, columns: int
) -> torch.Tensor:
    rank = basis.shape[1]
    return (
        basis.T.reshape(rank, rows, columns)
        .transpose(1, 2)
        .contiguous()
        .reshape(rank, rows * columns)
        .T
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", default="0,6,11")
    parser.add_argument("--targets", default="mlp.c_fc,mlp.c_proj")
    parser.add_argument("--basis-rank", type=int, default=16)
    parser.add_argument("--shared-ranks", default="1,2,4,8,16,24,32,48")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    layers = sorted(set(parse_int_list(args.layers)))
    targets = {item for item in args.targets.split(",") if item}
    shared_ranks = parse_int_list(args.shared_ranks)
    paths = sorted(args.snapshot_dir.glob("step_*.pt"))
    steps, values, snapshot_metadata = load_snapshots(
        paths, layers=set(layers), targets=targets
    )

    bases: dict[tuple[int, str], torch.Tensor] = {}
    eigenvalues: dict[tuple[int, str], torch.Tensor] = {}
    shapes: dict[tuple[int, str], tuple[int, int]] = {}
    energy_fractions: dict[tuple[int, str], float] = {}
    for parameter, tensors in sorted(values.items()):
        match = PARAMETER_PATTERN.match(parameter)
        if match is None:
            raise ValueError(f"unsupported parameter {parameter}")
        key = (int(match.group("layer")), match.group("target"))
        positions = torch.stack(tensors).to(args.device, dtype=torch.float32)
        _, all_values, basis = temporal_basis(
            positions.flatten(1), maximum_rank=args.basis_rank
        )
        rank = basis.shape[1]
        bases[key] = basis[:, :rank]
        eigenvalues[key] = all_values[:rank]
        shapes[key] = tuple(positions.shape[1:])
        energy_fractions[key] = float(
            all_values[:rank].sum() / all_values.sum().clamp_min(1e-30)
        )
        del positions

    pairwise_rows: list[dict[str, Any]] = []
    shared_rows: list[dict[str, Any]] = []
    for target in sorted(targets):
        target_bases = [bases[(layer, target)] for layer in layers]
        target_values = [eigenvalues[(layer, target)] for layer in layers]
        for source_layer in layers:
            for target_layer in layers:
                source = bases[(source_layer, target)]
                target_basis = bases[(target_layer, target)]
                mean_cos2, min_cos2, max_cos2 = canonical_metrics(source, target_basis)
                pairwise_rows.append(
                    {
                        "comparison": "same_matrix_cross_layer",
                        "source": f"layer{source_layer}.{target}",
                        "target": f"layer{target_layer}.{target}",
                        "basis_rank": source.shape[1],
                        "target_state_energy_in_retained_pcs": energy_fractions[(target_layer, target)],
                        "target_variance_capture": weighted_subspace_capture(
                            source, target_basis, target_values[layers.index(target_layer)]
                        ),
                        "mean_squared_canonical_cosine": mean_cos2,
                        "minimum_squared_canonical_cosine": min_cos2,
                        "maximum_squared_canonical_cosine": max_cos2,
                    }
                )
        shared_rows.extend(
            shared_basis_rows(
                target=target,
                layers=layers,
                bases=target_bases,
                eigenvalues=target_values,
                ranks=shared_ranks,
                parameter_size=target_bases[0].shape[0],
            )
        )

    if {"mlp.c_fc", "mlp.c_proj"}.issubset(targets):
        for fc_layer in layers:
            fc = bases[(fc_layer, "mlp.c_fc")]
            for proj_layer in layers:
                proj_shape = shapes[(proj_layer, "mlp.c_proj")]
                proj_t = transpose_basis(
                    bases[(proj_layer, "mlp.c_proj")], *proj_shape
                )
                mean_cos2, min_cos2, max_cos2 = canonical_metrics(fc, proj_t)
                pairwise_rows.append(
                    {
                        "comparison": "c_fc_vs_transposed_c_proj",
                        "source": f"layer{fc_layer}.mlp.c_fc",
                        "target": f"layer{proj_layer}.mlp.c_proj_T",
                        "basis_rank": fc.shape[1],
                        "target_state_energy_in_retained_pcs": energy_fractions[(proj_layer, "mlp.c_proj")],
                        "target_variance_capture": weighted_subspace_capture(
                            fc,
                            proj_t,
                            eigenvalues[(proj_layer, "mlp.c_proj")],
                        ),
                        "mean_squared_canonical_cosine": mean_cos2,
                        "minimum_squared_canonical_cosine": min_cos2,
                        "maximum_squared_canonical_cosine": max_cos2,
                    }
                )

    args.output.mkdir(parents=True, exist_ok=True)
    pairwise_path = args.output / "pairwise_basis_overlap.csv"
    shared_path = args.output / "shared_basis_capture.csv"
    write_csv(pairwise_path, pairwise_rows)
    write_csv(shared_path, shared_rows)
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_mlp_cross_layer_state_basis_v1",
        "steps": steps,
        "snapshot_metadata": snapshot_metadata,
        "layers": layers,
        "targets": sorted(targets),
        "basis_rank": args.basis_rank,
        "shared_ranks": shared_ranks,
        "method": {
            "basis": "complete-horizon centered temporal PCA; descriptive and optimistic",
            "pairwise_capture": "target-eigenvalue-weighted projection into source temporal PC span",
            "shared_basis": "globally optimal PCA of retained per-layer temporal covariance components",
            "paired_test": "c_fc compared with the ambient transpose of c_proj",
        },
        "analysis_execution": {
            "git_commit": git_commit(script.parents[2]),
            "entrypoint": str(script),
            "entrypoint_sha256": file_sha256(script),
            "command": sys.argv,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "device": args.device,
        },
        "outputs": {
            pairwise_path.name: file_sha256(pairwise_path),
            shared_path.name: file_sha256(shared_path),
        },
        "limitations": [
            "Full-horizon fits are noncausal representation ceilings.",
            "A shared dense ambient basis still costs rank times one dense matrix.",
            "Euclidean state capture is not fixed-evaluation CE.",
        ],
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pairwise_rows": len(pairwise_rows), "shared_rows": len(shared_rows), "metadata": str(metadata_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
