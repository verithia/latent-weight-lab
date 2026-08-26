#!/usr/bin/env python3
"""Measure compact representations of exact MLP optimizer directions.

This is a zero-update representation audit.  Static dictionaries are fitted
only on discovery probes and evaluated chronologically.  Local-refresh rows
fit on the first half of each window and evaluate on its second half.  The
adaptive SVD and bilateral diagonal rows are explicitly labeled per-step
oracles rather than persistent decoder implementations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_mlp_highcadence_basis import (
    chronological_splits,
    file_sha256,
    parse_float_list,
)
from examples.nanogpt.analyze_mlp_optimizer_probe_span import (
    FIELD_ORIENTATION,
    load_probe_inventory,
)
from examples.nanogpt.analyze_parameter_trajectory import (
    PARAMETER_PATTERN,
    parse_int_list,
    write_csv,
)
from latent_weight_lab.block_fht import normalized_fht_last_dim


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def largest_power_of_two_factor(value: int) -> int:
    if value <= 0:
        raise ValueError("dimension must be positive")
    return value & -value


def blockwise_fht_2d(
    values: torch.Tensor,
    *,
    row_block: int,
    column_block: int,
) -> torch.Tensor:
    """Apply an exact orthogonal 2-D FHT independently to matrix tiles."""
    if values.ndim != 3:
        raise ValueError("expected [samples, rows, columns]")
    samples, rows, columns = values.shape
    if (
        row_block <= 0
        or column_block <= 0
        or row_block & (row_block - 1)
        or column_block & (column_block - 1)
        or rows % row_block
        or columns % column_block
    ):
        raise ValueError("FHT blocks must be power-of-two divisors")
    work = values.reshape(
        samples, rows // row_block, row_block,
        columns // column_block, column_block,
    )
    work = normalized_fht_last_dim(work)
    work = work.permute(0, 1, 3, 4, 2).contiguous()
    work = normalized_fht_last_dim(work)
    return work.permute(0, 1, 4, 2, 3).reshape_as(values)


def selected_support_capture(
    coefficients: torch.Tensor,
    *,
    fit_indices: list[int],
    eval_indices: list[int],
    coordinates: int,
) -> float:
    flat = coefficients.flatten(1)
    if not 0 < coordinates <= flat.shape[1]:
        raise ValueError("invalid coordinate count")
    discovery_energy = flat[fit_indices].double().square().sum(dim=0)
    support = torch.topk(discovery_energy, coordinates, sorted=False).indices
    selected = flat[eval_indices][:, support]
    numerator = selected.double().square().sum()
    denominator = flat[eval_indices].double().square().sum().clamp_min(1e-30)
    return float(numerator / denominator)


def static_dictionary_rows(
    values: torch.Tensor,
    *,
    parameter: str,
    field: str,
    steps: list[int],
    discovery_stop: int,
    validation_stop: int,
    ratios: list[float],
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    splits = chronological_splits(
        steps,
        discovery_stop=discovery_stop,
        validation_stop=validation_stop,
    )
    matrix = values
    row_block = largest_power_of_two_factor(matrix.shape[1])
    column_block = largest_power_of_two_factor(matrix.shape[2])
    dictionaries = {
        "ambient_coordinate_support": matrix,
        "blockwise_2d_fht_support": blockwise_fht_2d(
            matrix, row_block=row_block, column_block=column_block
        ),
    }
    rows: list[dict[str, Any]] = []
    size = matrix[0].numel()
    for family, coefficients in dictionaries.items():
        for ratio in ratios:
            coordinates = max(1, round(size * ratio))
            for split_name, indices in splits.items():
                rows.append(
                    {
                        "parameter": parameter,
                        "field": field,
                        "family": family,
                        "fit_split": "discovery",
                        "eval_split": split_name,
                        "coordinate_ratio_requested": ratio,
                        "coordinates": coordinates,
                        "coordinate_ratio_resolved": coordinates / size,
                        "energy_capture": selected_support_capture(
                            coefficients,
                            fit_indices=splits["discovery"],
                            eval_indices=indices,
                            coordinates=coordinates,
                        ),
                        "row_fht_block": row_block if "fht" in family else 0,
                        "column_fht_block": (
                            column_block if "fht" in family else 0
                        ),
                    }
                )
    return rows, dictionaries


def local_refresh_rows(
    dictionaries: dict[str, torch.Tensor],
    *,
    parameter: str,
    field: str,
    steps: list[int],
    ratios: list[float],
    window_size: int,
) -> list[dict[str, Any]]:
    if window_size < 4 or window_size % 2:
        raise ValueError("window size must be even and at least four")
    rows: list[dict[str, Any]] = []
    for start in range(0, len(steps) - window_size + 1, window_size):
        middle = start + window_size // 2
        stop = start + window_size
        fit_indices = list(range(start, middle))
        eval_indices = list(range(middle, stop))
        for family, coefficients in dictionaries.items():
            size = coefficients[0].numel()
            for ratio in ratios:
                coordinates = max(1, round(size * ratio))
                rows.append(
                    {
                        "parameter": parameter,
                        "field": field,
                        "family": family,
                        "window_start_probe": start,
                        "window_stop_probe": stop,
                        "fit_step_start": steps[start],
                        "fit_step_stop": steps[middle - 1],
                        "eval_step_start": steps[middle],
                        "eval_step_stop": steps[stop - 1],
                        "coordinate_ratio_requested": ratio,
                        "coordinates": coordinates,
                        "energy_capture": selected_support_capture(
                            coefficients,
                            fit_indices=fit_indices,
                            eval_indices=eval_indices,
                            coordinates=coordinates,
                        ),
                    }
                )
    if not rows:
        raise ValueError("no complete local-refresh windows")
    return rows


def low_rank_for_budget(rows: int, columns: int, ratio: float) -> int:
    budget = math.floor(rows * columns * ratio)
    return max(0, budget // (rows + columns))


def adaptive_svd_rows(
    directions: torch.Tensor,
    *,
    parameter: str,
    field: str,
    steps: list[int],
    ratios: list[float],
) -> list[dict[str, Any]]:
    matrix_rows, matrix_columns = directions.shape[1:]
    ranks = sorted(
        {low_rank_for_budget(matrix_rows, matrix_columns, ratio) for ratio in ratios}
    )
    maximum_rank = max(ranks)
    captures = {rank: [] for rank in ranks}
    for direction in directions:
        total = direction.double().square().sum().clamp_min(1e-30)
        if maximum_rank:
            singular_values = torch.linalg.svdvals(direction.float())
            cumulative = singular_values.double().square().cumsum(0)
        else:
            cumulative = direction.new_zeros(0, dtype=torch.float64)
        for rank in ranks:
            captures[rank].append(
                0.0 if rank == 0 else float(cumulative[rank - 1] / total)
            )
    result: list[dict[str, Any]] = []
    for ratio in ratios:
        rank = low_rank_for_budget(matrix_rows, matrix_columns, ratio)
        values = torch.tensor(captures[rank], dtype=torch.float64)
        stored = rank * (matrix_rows + matrix_columns)
        result.append(
            {
                "parameter": parameter,
                "field": field,
                "family": "per_step_adaptive_svd_oracle",
                "coordinate_ratio_requested": ratio,
                "matrix_rank": rank,
                "stored_scalars": stored,
                "stored_scalar_fraction": stored / (matrix_rows * matrix_columns),
                "mean_energy_capture": float(values.mean()),
                "minimum_energy_capture": float(values.min()),
                "maximum_energy_capture": float(values.max()),
                "samples": len(steps),
            }
        )
    return result


def bilateral_diagonal_projection(
    weight: torch.Tensor,
    direction: torch.Tensor,
    *,
    iterations: int = 12,
) -> float:
    """Project onto diag(a) W + W diag(b) by alternating exact LS."""
    weight = weight.float()
    direction = direction.float()
    square = weight.square()
    row_denominator = square.sum(dim=1).clamp_min(1e-30)
    column_denominator = square.sum(dim=0).clamp_min(1e-30)
    b = weight.new_zeros(weight.shape[1])
    for _ in range(iterations):
        a = (
            weight * (direction - weight * b.unsqueeze(0))
        ).sum(dim=1) / row_denominator
        # Fix the additive gauge before the second coordinate update.
        shift = a.mean()
        a = a - shift
        b = b + shift
        b = (
            weight * (direction - weight * a.unsqueeze(1))
        ).sum(dim=0) / column_denominator
    prediction = weight * (a.unsqueeze(1) + b.unsqueeze(0))
    return float(
        prediction.double().square().sum()
        / direction.double().square().sum().clamp_min(1e-30)
    )


def load_weights(
    paths: list[Path], parameters: set[str]
) -> dict[str, list[torch.Tensor]]:
    result = {name: [] for name in parameters}
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for name in parameters:
            result[name].append(
                payload["parameters"][name]["weight_before_step"].contiguous()
            )
        del payload
    return result


def bilateral_rows(
    weights: torch.Tensor,
    directions: torch.Tensor,
    *,
    parameter: str,
    field: str,
) -> list[dict[str, Any]]:
    captures = torch.tensor(
        [
            bilateral_diagonal_projection(weight, direction)
            for weight, direction in zip(weights, directions, strict=True)
        ],
        dtype=torch.float64,
    )
    matrix_rows, matrix_columns = directions.shape[1:]
    coordinates = matrix_rows + matrix_columns
    return [
        {
            "parameter": parameter,
            "field": field,
            "family": "moving_bilateral_diagonal_tangent_oracle",
            "coordinates": coordinates,
            "coordinate_fraction": coordinates / (matrix_rows * matrix_columns),
            "mean_energy_capture": float(captures.mean()),
            "minimum_energy_capture": float(captures.min()),
            "maximum_energy_capture": float(captures.max()),
            "samples": captures.numel(),
        }
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", default="6")
    parser.add_argument("--targets", default="mlp.c_fc,mlp.c_proj")
    parser.add_argument(
        "--fields", default="raw_gradient_descent,exact_applied_direction"
    )
    parser.add_argument("--discovery-stop", type=int, default=119)
    parser.add_argument("--validation-stop", type=int, default=179)
    parser.add_argument("--ratios", default="0.001,0.0025,0.005,0.01")
    parser.add_argument("--local-window-size", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    layers = set(parse_int_list(args.layers))
    targets = {item for item in args.targets.split(",") if item}
    fields = [item for item in args.fields.split(",") if item]
    ratios = parse_float_list(args.ratios)
    if not fields or any(field not in FIELD_ORIENTATION for field in fields):
        raise ValueError("unknown or empty optimizer field selection")
    paths = sorted(args.probe_dir.glob("step_*.pt"))
    steps, values, input_metadata = load_probe_inventory(
        paths, layers=layers, targets=targets
    )
    weights = load_weights(paths, set(values))

    static_rows: list[dict[str, Any]] = []
    refresh_rows: list[dict[str, Any]] = []
    adaptive_rows: list[dict[str, Any]] = []
    bilateral_output: list[dict[str, Any]] = []
    for parameter, stored_fields in sorted(values.items()):
        weight_rows = torch.stack(weights[parameter]).to(
            device=args.device, dtype=torch.float32
        )
        for field in fields:
            directions = torch.stack(stored_fields[field]).to(
                device=args.device, dtype=torch.float32
            )
            rows, dictionaries = static_dictionary_rows(
                directions,
                parameter=parameter,
                field=field,
                steps=steps,
                discovery_stop=args.discovery_stop,
                validation_stop=args.validation_stop,
                ratios=ratios,
            )
            static_rows.extend(rows)
            refresh_rows.extend(
                local_refresh_rows(
                    dictionaries,
                    parameter=parameter,
                    field=field,
                    steps=steps,
                    ratios=ratios,
                    window_size=args.local_window_size,
                )
            )
            adaptive_rows.extend(
                adaptive_svd_rows(
                    directions,
                    parameter=parameter,
                    field=field,
                    steps=steps,
                    ratios=ratios,
                )
            )
            bilateral_output.extend(
                bilateral_rows(
                    weight_rows,
                    directions,
                    parameter=parameter,
                    field=field,
                )
            )
            del directions, dictionaries
            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()
        del weight_rows

    args.output.mkdir(parents=True, exist_ok=True)
    output_files = {
        "static_dictionary": args.output / "static_dictionary_capture.csv",
        "local_refresh": args.output / "local_refresh_capture.csv",
        "adaptive_svd": args.output / "adaptive_svd_oracle.csv",
        "bilateral": args.output / "bilateral_diagonal_oracle.csv",
    }
    for path, rows in (
        (output_files["static_dictionary"], static_rows),
        (output_files["local_refresh"], refresh_rows),
        (output_files["adaptive_svd"], adaptive_rows),
        (output_files["bilateral"], bilateral_output),
    ):
        write_csv(path, rows)
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_mlp_applied_basis_structure_v1",
        "steps": steps,
        "sample_count": len(steps),
        "parameters": sorted(values),
        "fields": fields,
        "ratios": ratios,
        "input": input_metadata,
        "method": {
            "static_support": "top discovery-energy coordinates, frozen for validation and test",
            "blockwise_2d_fht": "exact orthogonal FHT using the largest power-of-two divisor of each matrix axis",
            "local_refresh": "support fitted on first half and evaluated on second half of each nonoverlapping probe window",
            "adaptive_svd": "per-step best low-matrix-rank oracle at scalar storage budget",
            "bilateral": "per-step LS projection onto diag(a)W + Wdiag(b)",
        },
        "cost_model": {
            "support_values": "k trainable scalars; explicit indices add ceil(log2(P))/8 bytes each unless support is baked into a fixed decoder",
            "blockwise_2d_fht_expansion": "approximately P(log2(row_block)+log2(column_block)) additions/subtractions",
            "adaptive_svd_storage": "r(m+n) scalars and approximately O(rmn) materialization FLOPs; fitting cost excluded",
            "bilateral_storage": "m+n scalars and 3mn multiply/add-scale operations to materialize both terms",
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
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in output_files.items()
        },
        "limitations": [
            "Adaptive SVD and bilateral rows use the target direction at each step and are representation oracles, not causal decoders.",
            "Static and local support selection is causal only within its declared fit/eval split.",
            "Euclidean energy capture does not establish functional-loss parity.",
        ],
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "static_rows": len(static_rows),
                "refresh_rows": len(refresh_rows),
                "adaptive_rows": len(adaptive_rows),
                "bilateral_rows": len(bilateral_output),
                "metadata": str(metadata_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
