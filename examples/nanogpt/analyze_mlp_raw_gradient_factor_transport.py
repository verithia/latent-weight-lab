#!/usr/bin/env python3
"""Measure chronological transport of low-rank raw-gradient factors.

Per-step raw MLP gradients are strongly low matrix-rank before Muon's polar
map.  This audit asks the stricter causal question: do their left/right
singular subspaces persist enough for a rank-r matrix-manifold tangent

    {U X + Y V^T}

to capture future gradients at a 0.1--1% factor budget?
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
    chronological_splits,
    file_sha256,
)
from examples.nanogpt.analyze_mlp_optimizer_probe_span import (
    load_probe_inventory,
)
from examples.nanogpt.analyze_parameter_trajectory import (
    PARAMETER_PATTERN,
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


def tangent_capture(
    direction: torch.Tensor,
    left_basis: torch.Tensor,
    right_basis: torch.Tensor,
) -> dict[str, float]:
    """Exact Frobenius projection energies for one matrix direction."""
    total = direction.double().square().sum().clamp_min(1e-30)
    left_coordinates = left_basis.T @ direction
    right_coordinates = direction @ right_basis
    intersection = left_coordinates @ right_basis
    left_energy = left_coordinates.double().square().sum()
    right_energy = right_coordinates.double().square().sum()
    intersection_energy = intersection.double().square().sum()
    return {
        "left_capture": float(left_energy / total),
        "right_capture": float(right_energy / total),
        "bilinear_core_capture": float(intersection_energy / total),
        "rank_manifold_tangent_capture": float(
            (left_energy + right_energy - intersection_energy) / total
        ),
    }


def exact_singular_factors(
    directions: torch.Tensor, maximum_rank: int
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    left: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    right: list[torch.Tensor] = []
    for direction in directions:
        u, s, vh = torch.linalg.svd(direction, full_matrices=False)
        left.append(u[:, :maximum_rank])
        values.append(s[:maximum_rank])
        right.append(vh[:maximum_rank].T)
    return left, values, right


def fit_shared_factors(
    left: list[torch.Tensor],
    values: list[torch.Tensor],
    right: list[torch.Tensor],
    indices: list[int],
    rank: int,
    component_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit aggregate covariance factors from truncated exact per-step SVDs."""
    left_weighted = torch.cat(
        [
            left[index][:, :component_rank]
            * values[index][:component_rank].unsqueeze(0)
            for index in indices
        ],
        dim=1,
    )
    right_weighted = torch.cat(
        [
            right[index][:, :component_rank]
            * values[index][:component_rank].unsqueeze(0)
            for index in indices
        ],
        dim=1,
    )
    left_basis = torch.linalg.svd(
        left_weighted, full_matrices=False
    ).U[:, :rank]
    right_basis = torch.linalg.svd(
        right_weighted, full_matrices=False
    ).U[:, :rank]
    return left_basis, right_basis


def aggregate_capture(
    directions: torch.Tensor,
    indices: list[int],
    left_basis: torch.Tensor,
    right_basis: torch.Tensor,
) -> dict[str, float]:
    numerators = {
        "left_capture": 0.0,
        "right_capture": 0.0,
        "bilinear_core_capture": 0.0,
        "rank_manifold_tangent_capture": 0.0,
    }
    total = 0.0
    for index in indices:
        direction = directions[index]
        energy = float(direction.double().square().sum())
        captures = tangent_capture(direction, left_basis, right_basis)
        for key, value in captures.items():
            numerators[key] += value * energy
        total += energy
    return {key: value / max(total, 1e-30) for key, value in numerators.items()}


def canonical_overlap(
    left: torch.Tensor, right: torch.Tensor
) -> tuple[float, float, float]:
    squared = torch.linalg.svdvals(left.T @ right).double().square()
    return float(squared.mean()), float(squared.min()), float(squared.max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", default="6")
    parser.add_argument("--targets", default="mlp.c_fc,mlp.c_proj")
    parser.add_argument("--ranks", default="1,2,3,4,5,6")
    parser.add_argument("--fit-component-rank", type=int, default=32)
    parser.add_argument("--discovery-stop", type=int, default=119)
    parser.add_argument("--validation-stop", type=int, default=179)
    parser.add_argument("--local-window-size", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    layers = set(parse_int_list(args.layers))
    targets = {item for item in args.targets.split(",") if item}
    ranks = parse_int_list(args.ranks)
    maximum_rank = max(max(ranks), args.fit_component_rank)
    paths = sorted(args.probe_dir.glob("step_*.pt"))
    steps, inventory, input_metadata = load_probe_inventory(
        paths, layers=layers, targets=targets
    )
    splits = chronological_splits(
        steps,
        discovery_stop=args.discovery_stop,
        validation_stop=args.validation_stop,
    )
    if args.local_window_size < 4 or args.local_window_size % 2:
        raise ValueError("local window size must be even and at least four")

    chronological_rows: list[dict[str, Any]] = []
    local_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    for parameter, fields in sorted(inventory.items()):
        match = PARAMETER_PATTERN.match(parameter)
        if match is None:
            raise ValueError(f"unsupported parameter {parameter}")
        directions = torch.stack(fields["raw_gradient_descent"]).to(
            args.device, dtype=torch.float32
        )
        left, singular_values, right = exact_singular_factors(
            directions, maximum_rank
        )
        matrix_rows, matrix_columns = directions.shape[1:]
        parameter_size = matrix_rows * matrix_columns
        common = {
            "parameter": parameter,
            "layer": int(match.group("layer")),
            "target": match.group("target"),
        }
        for rank in ranks:
            discovery_left, discovery_right = fit_shared_factors(
                left,
                singular_values,
                right,
                splits["discovery"],
                rank,
                args.fit_component_rank,
            )
            stored = rank * (matrix_rows + matrix_columns)
            tangent_dimension = rank * (
                matrix_rows + matrix_columns - rank
            )
            for split_name, indices in splits.items():
                chronological_rows.append(
                    {
                        **common,
                        "fit_split": "discovery",
                        "eval_split": split_name,
                        "rank": rank,
                        "stored_scalars": stored,
                        "stored_scalar_fraction": stored / parameter_size,
                        "rank_manifold_tangent_dimension_fraction": tangent_dimension
                        / parameter_size,
                        "fit_component_rank": args.fit_component_rank,
                        **aggregate_capture(
                            directions,
                            indices,
                            discovery_left,
                            discovery_right,
                        ),
                    }
                )

            phase_bases = {}
            for split_name, indices in splits.items():
                phase_bases[split_name] = fit_shared_factors(
                    left,
                    singular_values,
                    right,
                    indices,
                    rank,
                    args.fit_component_rank,
                )
            for source, target in (
                ("discovery", "validation"),
                ("validation", "test"),
                ("discovery", "test"),
            ):
                source_left, source_right = phase_bases[source]
                target_left, target_right = phase_bases[target]
                left_mean, left_min, left_max = canonical_overlap(
                    source_left, target_left
                )
                right_mean, right_min, right_max = canonical_overlap(
                    source_right, target_right
                )
                phase_rows.append(
                    {
                        **common,
                        "source_split": source,
                        "target_split": target,
                        "rank": rank,
                        "left_mean_squared_canonical_cosine": left_mean,
                        "left_minimum_squared_canonical_cosine": left_min,
                        "left_maximum_squared_canonical_cosine": left_max,
                        "right_mean_squared_canonical_cosine": right_mean,
                        "right_minimum_squared_canonical_cosine": right_min,
                        "right_maximum_squared_canonical_cosine": right_max,
                    }
                )

        window = args.local_window_size
        for start in range(0, len(steps) - window + 1, window):
            middle = start + window // 2
            stop = start + window
            fit_indices = list(range(start, middle))
            eval_indices = list(range(middle, stop))
            for rank in ranks:
                local_left, local_right = fit_shared_factors(
                    left,
                    singular_values,
                    right,
                    fit_indices,
                    rank,
                    args.fit_component_rank,
                )
                local_rows.append(
                    {
                        **common,
                        "window_start_probe": start,
                        "window_stop_probe": stop,
                        "fit_step_start": steps[start],
                        "fit_step_stop": steps[middle - 1],
                        "eval_step_start": steps[middle],
                        "eval_step_stop": steps[stop - 1],
                        "rank": rank,
                        **aggregate_capture(
                            directions,
                            eval_indices,
                            local_left,
                            local_right,
                        ),
                    }
                )
        del directions, left, singular_values, right
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

    args.output.mkdir(parents=True, exist_ok=True)
    chronological_path = args.output / "chronological_factor_capture.csv"
    local_path = args.output / "local_factor_capture.csv"
    phase_path = args.output / "phase_factor_overlap.csv"
    write_csv(chronological_path, chronological_rows)
    write_csv(local_path, local_rows)
    write_csv(phase_path, phase_rows)
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_mlp_raw_gradient_factor_transport_v1",
        "steps": steps,
        "sample_count": len(steps),
        "parameters": sorted(inventory),
        "ranks": ranks,
        "fit_component_rank": args.fit_component_rank,
        "splits": {name: [steps[index] for index in indices] for name, indices in splits.items()},
        "input": input_metadata,
        "method": {
            "per_step_factors": "exact SVD; top fit-component-rank factors retain covariance information",
            "shared_factors": "top eigenvectors of the aggregate truncated left/right gradient covariances",
            "tangent": "orthogonal projection into {U X + Y V^T}",
            "local": "first ten probes fit and next ten probes evaluate in each nonoverlapping 20-probe window",
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
            chronological_path.name: file_sha256(chronological_path),
            local_path.name: file_sha256(local_path),
            phase_path.name: file_sha256(phase_path),
        },
        "limitations": [
            "This is Euclidean raw-gradient capture, not fixed-evaluation CE.",
            "Full-phase factor fits are descriptive; only declared chronological rows are causal.",
            "A good raw-gradient tangent does not prove that a fixed-rank delta can represent the final dense state.",
            "The covariance fit truncates each exact SVD to fit-component-rank modes.",
        ],
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"chronological_rows": len(chronological_rows), "local_rows": len(local_rows), "phase_rows": len(phase_rows), "metadata": str(metadata_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
