#!/usr/bin/env python3
"""Project coherent Muon directions into the production BlockFHT tangent."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.nanogpt.analyze_mlp_bilateral_phase_capture import (
    file_sha256,
    git_commit,
)
from examples.nanogpt.analyze_mlp_task_gradient_direction import (
    direction_metrics,
)
from examples.nanogpt.analyze_parameter_trajectory import (
    load_snapshots,
    parse_int_list,
    write_csv,
)
from examples.nanogpt.parameter_trajectory import (
    OPTIMIZER_PROBE_SCHEMA_VERSION,
)
from latent_weight_lab.block_fht import (
    block_fht_grad_latent,
    block_fht_slice,
    next_power_of_two,
)


def production_cproj_geometry(config: dict[str, Any]) -> dict[str, Any]:
    target = "mlp.c_proj"
    n_embd = int(config["n_embd"])
    in_features = 4 * n_embd
    out_features = n_embd
    size = in_features * out_features
    ratios = config.get("block_fht_latent_ratios")
    ratio = float(config["block_fht_latent_ratio"])
    if isinstance(ratios, dict) and target in ratios:
        ratio = float(ratios[target])
    latent_dim = max(1, round(size * ratio))
    latent_shape: tuple[int, ...] = (latent_dim,)
    if target in config.get("block_fht_muon_latent_targets", []):
        rows = min(int(config["block_fht_muon_latent_rows"]), latent_dim)
        columns = math.ceil(latent_dim / rows)
        latent_dim = rows * columns
        latent_shape = (rows, columns)
    block_size = next_power_of_two(latent_dim)
    if size % block_size:
        raise ValueError(
            "registered exact projector requires full repeated FHT blocks"
        )
    return {
        "target": target,
        "in_features": in_features,
        "out_features": out_features,
        "size": size,
        "latent_ratio": ratio,
        "latent_dim": latent_dim,
        "latent_shape": latent_shape,
        "block_size": block_size,
        "complete_blocks": size // block_size,
        "layers": int(config["block_fht_layers"]),
        "base_seed": int(config["block_fht_seed"]),
    }


def project_blockfht_tangent(
    direction: torch.Tensor,
    *,
    latent_shape: tuple[int, ...],
    size: int,
    layers: int,
    seed: int,
) -> torch.Tensor:
    """Apply the exact Euclidean projector ``A(A^T A)^-1 A^T``."""
    if direction.numel() != size:
        raise ValueError("direction size does not match generator output")
    latent = torch.zeros(
        latent_shape,
        device=direction.device,
        dtype=torch.float32,
    )
    block_size = next_power_of_two(latent.numel())
    if size % block_size:
        raise ValueError("projection requires complete repeated FHT blocks")
    frame_bound = size // block_size
    pulled_back = block_fht_grad_latent(
        latent,
        direction.float().reshape(-1),
        size,
        layers,
        seed,
    )
    projected = block_fht_slice(
        pulled_back / float(frame_bound),
        size,
        layers,
        seed,
        0,
        size,
    )
    return projected.view_as(direction)


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    energy = torch.tensor(
        [float(row["target_chord_fro"]) ** 2 for row in rows],
        dtype=torch.float64,
    )

    def weighted(key: str) -> float:
        values = torch.tensor(
            [float(row[key]) for row in rows], dtype=torch.float64
        )
        return float((energy * values).sum() / energy.sum())

    result = {
        "cells": len(rows),
        "raw_exact_applied_recovery": weighted(
            "raw_positive_step_line_recovery"
        ),
        "projected_recovery": weighted(
            "projected_positive_step_line_recovery"
        ),
        "projected_cosine": weighted("projected_cosine"),
        "minimum_projected_cell_cosine": min(
            float(row["projected_cosine"]) for row in rows
        ),
        "positive_projected_cells": sum(
            float(row["projected_cosine"]) > 0.0 for row in rows
        ),
        "applied_direction_energy_retained": weighted(
            "applied_direction_energy_retained"
        ),
        "maximum_projector_idempotence_relative_error": max(
            float(row["projector_idempotence_relative_error"])
            for row in rows
        ),
    }
    admitted = (
        result["projected_recovery"] >= 0.10
        and result["positive_projected_cells"] == len(rows)
    )
    result["decision"] = (
        "ADMIT_WEIGHT_SPACE_MUON_PULLBACK_124M_SCREEN"
        if admitted
        else "REJECT_FIXED_BLOCKFHT_TANGENT_FOR_POLAR_PULLBACK"
    )
    result["admitted"] = admitted
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--probe-dir", required=True, type=Path)
    parser.add_argument("--production-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", default="0,3,6,9,11")
    parser.add_argument("--phase-boundaries", default="0,60,120,180,238")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    layers = parse_int_list(args.layers)
    boundaries = parse_int_list(args.phase_boundaries)
    phase_pairs = list(zip(boundaries[:-1], boundaries[1:], strict=True))
    if not layers or len(phase_pairs) < 1:
        raise ValueError("invalid layers or phase boundaries")
    config = json.loads(args.production_config.read_text())
    geometry = production_cproj_geometry(config)
    snapshot_paths = [
        args.snapshot_dir / f"step_{step:06d}.pt" for step in boundaries
    ]
    probe_paths = [
        args.probe_dir / f"step_{start:06d}.pt"
        for start, _end in phase_pairs
    ]
    if any(not path.is_file() for path in (*snapshot_paths, *probe_paths)):
        raise ValueError("required snapshot or optimizer probe is absent")
    steps, values, snapshot_metadata = load_snapshots(
        snapshot_paths,
        layers=set(layers),
        targets={"mlp.c_proj"},
    )
    if steps != boundaries:
        raise ValueError("loaded snapshot steps do not match boundaries")
    step_index = {step: index for index, step in enumerate(steps)}

    rows: list[dict[str, Any]] = []
    run_identity_sha256: str | None = None
    for phase_start, phase_end in phase_pairs:
        probe_path = args.probe_dir / f"step_{phase_start:06d}.pt"
        probe = torch.load(probe_path, map_location="cpu", weights_only=False)
        if probe.get("schema_version") != OPTIMIZER_PROBE_SCHEMA_VERSION:
            raise ValueError("unexpected optimizer probe schema")
        if run_identity_sha256 is None:
            run_identity_sha256 = probe["run_identity_sha256"]
        elif probe["run_identity_sha256"] != run_identity_sha256:
            raise ValueError("optimizer probes do not share one identity")
        for layer in layers:
            name = f"transformer.h.{layer}.mlp.c_proj.weight"
            source = values[name][step_index[phase_start]].to(args.device)
            terminal = values[name][step_index[phase_end]].to(args.device)
            chord = terminal - source
            applied = probe["parameters"][name][
                "applied_direction_per_lr"
            ].to(args.device)
            seed = geometry["base_seed"] + layer * 4 + 3
            projected = project_blockfht_tangent(
                applied,
                latent_shape=geometry["latent_shape"],
                size=geometry["size"],
                layers=geometry["layers"],
                seed=seed,
            )
            projected_twice = project_blockfht_tangent(
                projected,
                latent_shape=geometry["latent_shape"],
                size=geometry["size"],
                layers=geometry["layers"],
                seed=seed,
            )
            raw = direction_metrics(chord, applied)
            projected_metrics = direction_metrics(chord, projected)
            applied_energy = applied.double().square().sum().clamp_min(1e-30)
            projection_energy = projected.double().square().sum()
            idempotence = (
                (projected_twice - projected).double().norm()
                / projected.double().norm().clamp_min(1e-30)
            )
            row = {
                "parameter": name,
                "layer": layer,
                "phase_start": phase_start,
                "phase_end": phase_end,
                "seed": seed,
                "raw_cosine": raw["cosine"],
                "raw_positive_step_line_recovery": raw[
                    "positive_step_line_recovery"
                ],
                "projected_cosine": projected_metrics["cosine"],
                "projected_positive_step_line_recovery": projected_metrics[
                    "positive_step_line_recovery"
                ],
                "target_chord_fro": projected_metrics["target_chord_fro"],
                "applied_direction_energy_retained": float(
                    projection_energy / applied_energy
                ),
                "projector_idempotence_relative_error": float(idempotence),
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            del source, terminal, chord, applied, projected, projected_twice
        del probe
        if "cuda" in args.device:
            torch.cuda.empty_cache()

    aggregate = aggregate_rows(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    detail_path = args.output / "blockfht_polar_projection.csv"
    aggregate_path = args.output / "blockfht_polar_projection_aggregate.json"
    write_csv(detail_path, rows)
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n"
    )
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_mlp_blockfht_polar_projection_v1",
        "decision": aggregate["decision"],
        "decision_rule": (
            "admit training only with >=10% energy-weighted projected "
            "future-chord recovery and positive cosine in all 20 cells"
        ),
        "geometry": geometry,
        "layers": layers,
        "phase_boundaries": boundaries,
        "run_identity_sha256": run_identity_sha256,
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
        "inputs": {
            "production_config": {
                "path": str(args.production_config),
                "sha256": file_sha256(args.production_config),
            },
            "snapshots": [
                {"path": str(path), "sha256": file_sha256(path)}
                for path in snapshot_paths
            ],
            "optimizer_probes": [
                {"path": str(path), "sha256": file_sha256(path)}
                for path in probe_paths
            ],
        },
        "outputs": {
            "detail_sha256": file_sha256(detail_path),
            "aggregate_sha256": file_sha256(aggregate_path),
        },
        "limitations": [
            "This is an exact tangent-capacity diagnostic, not a training run.",
            "It tests the fixed production generator seeds and one-percent c_proj allocation.",
            "It does not test a task-conditioned moving generator tangent.",
        ],
    }
    metadata_path = args.output / "blockfht_polar_projection_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "aggregate": aggregate,
                "metadata": str(metadata_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
