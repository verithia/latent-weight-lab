#!/usr/bin/env python3
"""Decompose dense MLP phase chords in each phase-start singular frame.

For a phase-start matrix ``W = U diag(s) V^T`` and phase chord ``D``, the
orthogonal decomposition is

``D = U diag(diag(U^T D V)) V^T
     + U offdiag(U^T D V) V^T
     + (D - U U^T D V V^T)``.

The three terms measure singular-value/radial motion, mixing inside the
occupied singular frame, and motion outside that frame.  For the rectangular
GPT MLP matrices, the final term is expansion-output subspace rotation for
``c_fc`` and expansion-input row-space rotation for ``c_proj``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_parameter_trajectory import (
    PARAMETER_PATTERN,
    load_snapshots,
    parse_int_list,
    write_csv,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = left.norm() * right.norm()
    if float(denominator) <= 0.0:
        return 0.0
    return float(torch.dot(left.flatten(), right.flatten()) / denominator)


def singular_frame_decomposition(
    base: torch.Tensor,
    delta: torch.Tensor,
) -> dict[str, float | int | str]:
    """Return exact orthogonal phase-chord energy in the base SVD frame."""
    if base.ndim != 2 or delta.ndim != 2 or base.shape != delta.shape:
        raise ValueError("base and delta must be same-shaped matrices")
    if not base.is_floating_point() or not delta.is_floating_point():
        raise ValueError("base and delta must be floating-point tensors")
    if float(delta.norm()) <= 0.0:
        raise ValueError("phase delta must be nonzero")

    u, singular, vh = torch.linalg.svd(base, full_matrices=False)
    core = u.T @ delta @ vh.T
    diagonal_values = torch.diagonal(core)
    diagonal = (u * diagonal_values.unsqueeze(0)) @ vh
    projected = u @ core @ vh
    mixing = projected - diagonal
    residual = delta - projected

    delta_energy = delta.square().sum()
    diagonal_energy = diagonal.square().sum()
    mixing_energy = mixing.square().sum()
    residual_energy = residual.square().sum()
    component_energy = diagonal_energy + mixing_energy + residual_energy
    reconstruction = diagonal + mixing + residual

    singular_energy = singular.square().sum().clamp_min(1e-30)
    global_scale = torch.dot(diagonal_values, singular) / singular_energy
    global_scale_energy = global_scale.square() * singular_energy
    diagonal_nonuniform_energy = (
        diagonal_energy - global_scale_energy
    ).clamp_min(0.0)
    components = (diagonal, mixing, residual)
    orthogonality = max(
        abs(_cosine(components[left], components[right]))
        for left in range(len(components))
        for right in range(left + 1, len(components))
    )
    probability = singular.square() / singular_energy
    entropy = -(probability * probability.clamp_min(1e-30).log()).sum()

    if base.shape[0] > base.shape[1]:
        residual_interpretation = "left_expansion_output_subspace_rotation"
    elif base.shape[0] < base.shape[1]:
        residual_interpretation = "right_expansion_input_rowspace_rotation"
    else:
        residual_interpretation = "zero_for_full_rank_square_base_up_to_numerics"

    return {
        "matrix_axis0": base.shape[0],
        "matrix_axis1": base.shape[1],
        "base_rank_bound": min(base.shape),
        "base_fro": float(base.norm()),
        "delta_fro": float(delta_energy.sqrt()),
        "relative_delta_fro": float(delta_energy.sqrt() / base.norm().clamp_min(1e-30)),
        "delta_base_cosine": _cosine(delta, base),
        "base_spectral_norm": float(singular[0]),
        "base_stable_rank": float(singular_energy / singular[0].square().clamp_min(1e-30)),
        "base_entropy_effective_rank": float(entropy.exp()),
        "singular_value_motion_fro": float(diagonal_energy.sqrt()),
        "in_frame_mixing_fro": float(mixing_energy.sqrt()),
        "subspace_rotation_residual_fro": float(residual_energy.sqrt()),
        "singular_value_motion_energy_fraction": float(diagonal_energy / delta_energy),
        "in_frame_mixing_energy_fraction": float(mixing_energy / delta_energy),
        "subspace_rotation_residual_energy_fraction": float(residual_energy / delta_energy),
        "occupied_frame_energy_fraction": float(
            (diagonal_energy + mixing_energy) / delta_energy
        ),
        "global_scale_coefficient": float(global_scale),
        "global_scale_energy_fraction": float(global_scale_energy / delta_energy),
        "nonuniform_singular_value_energy_fraction": float(
            diagonal_nonuniform_energy / delta_energy
        ),
        "singular_change_base_cosine": _cosine(diagonal_values, singular),
        "component_energy_relative_error": float(
            (component_energy - delta_energy).abs() / delta_energy
        ),
        "reconstruction_relative_error": float(
            (reconstruction - delta).norm() / delta_energy.sqrt()
        ),
        "maximum_component_absolute_cosine": orthogonality,
        "residual_interpretation": residual_interpretation,
    }


def analyze_parameter(
    *,
    name: str,
    steps: list[int],
    tensors: list[torch.Tensor],
    device: str,
) -> list[dict[str, Any]]:
    match = PARAMETER_PATTERN.match(name)
    if match is None:
        raise ValueError(f"unsupported parameter name: {name}")
    if len(steps) != len(tensors) or len(steps) < 2:
        raise ValueError("at least two aligned phase-boundary tensors are required")
    rows: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(zip(steps[:-1], steps[1:], strict=True)):
        base = tensors[index].to(device=device, dtype=torch.float32)
        terminal = tensors[index + 1].to(device=device, dtype=torch.float32)
        delta = terminal - base
        rows.append(
            {
                "parameter": name,
                "layer": int(match.group("layer")),
                "target": match.group("target"),
                "phase_start": start,
                "phase_end": end,
                **singular_frame_decomposition(base, delta),
            }
        )
        del base, terminal, delta
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return rows


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fraction_fields = [
        "singular_value_motion_energy_fraction",
        "in_frame_mixing_energy_fraction",
        "subspace_rotation_residual_energy_fraction",
        "occupied_frame_energy_fraction",
        "global_scale_energy_fraction",
        "nonuniform_singular_value_energy_fraction",
    ]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        transition = f"{row['phase_start']}->{row['phase_end']}"
        groups.setdefault((str(row["target"]), transition), []).append(row)
        groups.setdefault((str(row["target"]), "all_phases"), []).append(row)
    result: list[dict[str, Any]] = []
    for (target, transition), selected in sorted(groups.items()):
        delta_energy = torch.tensor(
            [float(row["delta_fro"]) ** 2 for row in selected],
            dtype=torch.float64,
        )
        total_energy = delta_energy.sum().clamp_min(1e-30)
        record: dict[str, Any] = {
            "target": target,
            "transition": transition,
            "parameter_count": len(selected),
            "total_delta_fro": float(total_energy.sqrt()),
        }
        for field in fraction_fields:
            values = torch.tensor(
                [float(row[field]) for row in selected],
                dtype=torch.float64,
            )
            record[f"{field}_energy_weighted"] = float(
                (values * delta_energy).sum() / total_energy
            )
            record[f"{field}_mean"] = float(values.mean())
            record[f"{field}_median"] = float(values.median())
            record[f"{field}_minimum"] = float(values.min())
            record[f"{field}_maximum"] = float(values.max())
        result.append(record)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", default="")
    parser.add_argument("--targets", default="mlp.c_fc,mlp.c_proj")
    parser.add_argument("--phase-boundaries", default="0,60,120,180,238")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    boundaries = parse_int_list(args.phase_boundaries)
    if len(boundaries) < 2 or boundaries != sorted(set(boundaries)):
        raise ValueError("--phase-boundaries must be ordered and unique")
    paths = [args.snapshot_dir / f"step_{step:06d}.pt" for step in boundaries]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"phase-boundary snapshots are absent: {missing}")
    layers = parse_int_list(args.layers)
    targets = {item for item in args.targets.split(",") if item}
    steps, values, snapshot_metadata = load_snapshots(
        paths,
        layers=set(layers) if layers else None,
        targets=targets if targets else None,
    )
    if steps != boundaries:
        raise ValueError("loaded snapshot steps do not match requested phase boundaries")

    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for name, tensors in sorted(values.items()):
        rows.extend(
            analyze_parameter(
                name=name,
                steps=steps,
                tensors=tensors,
                device=args.device,
            )
        )
    aggregates = aggregate_rows(rows)
    write_csv(args.output / "singular_frame_motion.csv", rows)
    write_csv(args.output / "singular_frame_motion_aggregate.csv", aggregates)

    script = Path(__file__).resolve()
    repo = script.parents[2]
    metadata = {
        "schema_version": "nanogpt_mlp_singular_frame_motion_v1",
        "snapshot_metadata": snapshot_metadata,
        "phase_boundaries": boundaries,
        "parameters": sorted(values),
        "method": {
            "base_frame": "thin SVD of each phase-start dense matrix W = U diag(s) V^T",
            "singular_value_motion": "U diag(diag(U^T delta V)) V^T",
            "in_frame_mixing": "U offdiag(U^T delta V) V^T",
            "subspace_rotation_residual": "delta - U U^T delta V V^T",
            "global_scale": "projection of singular-value motion onto the base singular-value vector",
            "aggregation": "both layer mean/median/range and phase-delta-energy-weighted fractions",
        },
        "analysis_execution": {
            "git_commit": git_commit(repo),
            "entrypoint": str(script),
            "entrypoint_sha256": file_sha256(script),
            "command": sys.argv,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "device": args.device,
        },
        "snapshot_files": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in paths
        ],
        "interpretation_limits": [
            "This is a decomposition of optimizer phase chords, not a manifold-dimension estimate.",
            "The singular frame is local to the phase-start matrix and is not shared across layers.",
            "Near-degenerate singular values can rotate their bases and redistribute diagonal versus in-frame mixing.",
            "Occupied-frame total versus residual rotation is invariant to rotations within degenerate singular subspaces.",
            "One dense trajectory does not identify all low-loss MLP solutions.",
            "Component energy does not by itself establish which generator can optimize that component.",
        ],
    }
    metadata_path = args.output / "singular_frame_motion_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    all_phase_aggregates = [
        row for row in aggregates if row["transition"] == "all_phases"
    ]
    print(
        json.dumps(
            {
                "snapshots": len(steps),
                "parameters": len(values),
                "phase_chords": len(rows),
                "all_phase_aggregates": all_phase_aggregates,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
