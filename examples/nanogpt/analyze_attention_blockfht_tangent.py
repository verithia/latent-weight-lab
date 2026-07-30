#!/usr/bin/env python3
"""Measure dense attention paths against the deployed BlockFHT tangent.

This diagnostic deliberately separates three questions:

1. Is the sampled dense Muon trajectory temporally low-dimensional?
2. How much of each dense phase chord lies in the fixed production tangent?
3. How much of the exact dense Muon direction remains useful after projection?

The second question is a representability test of this chart.  The first is
not: a low-dimensional curve can still be oriented almost orthogonally to a
particular fixed low-dimensional subspace.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_mlp_blockfht_polar_projection import (
    project_blockfht_tangent,
)
from examples.nanogpt.analyze_mlp_task_gradient_direction import (
    direction_metrics,
)
from examples.nanogpt.analyze_parameter_trajectory import (
    load_snapshots,
    parse_int_list,
)
from examples.nanogpt.parameter_trajectory import (
    OPTIMIZER_PROBE_SCHEMA_VERSION,
)
from latent_weight_lab.block_fht import next_power_of_two


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def latent_geometry(size: int, ratio: float) -> dict[str, int | float]:
    latent_dim = max(1, round(size * ratio))
    block_size = next_power_of_two(latent_dim)
    if size % block_size:
        raise ValueError(
            f"exact production projector requires full repeated blocks: "
            f"{size=} {latent_dim=} {block_size=}"
        )
    return {
        "size": size,
        "latent_dim": latent_dim,
        "block_size": block_size,
        "frame_bound": size // block_size,
        "rank_fraction": latent_dim / size,
    }


def project_matrix(
    matrix: torch.Tensor,
    *,
    ratio: float,
    layers: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, int | float]]:
    geometry = latent_geometry(matrix.numel(), ratio)
    projected = project_blockfht_tangent(
        matrix,
        latent_shape=(int(geometry["latent_dim"]),),
        size=matrix.numel(),
        layers=layers,
        seed=seed,
    )
    return projected, geometry


def project_c_attn(
    matrix: torch.Tensor,
    *,
    n_embd: int,
    n_head: int,
    ratio: float,
    layers: int,
    base_seed: int,
    layer: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if matrix.shape != (3 * n_embd, n_embd):
        raise ValueError(
            f"unexpected dense c_attn shape {tuple(matrix.shape)}"
        )
    head_dim = n_embd // n_head
    output = torch.zeros_like(matrix)
    q = matrix[:n_embd]
    k = matrix[n_embd : 2 * n_embd]
    v = matrix[2 * n_embd :]
    head_geometries = []
    for head in range(n_head):
        start = head * head_dim
        stop = start + head_dim
        qk = torch.cat((q[start:stop], k[start:stop]), dim=0)
        qk_projected, geometry = project_matrix(
            qk,
            ratio=ratio,
            layers=layers,
            seed=base_seed + layer * 32 + head,
        )
        output[start:stop] = qk_projected[:head_dim]
        output[n_embd + start : n_embd + stop] = qk_projected[head_dim:]
        head_geometries.append(geometry)
    v_projected, v_geometry = project_matrix(
        v,
        ratio=ratio,
        layers=layers,
        seed=base_seed + layer * 8 + 2,
    )
    output[2 * n_embd :] = v_projected
    return output, {
        "qk_head": head_geometries[0],
        "qk_heads": n_head,
        "v": v_geometry,
        "total_latent_dim": (
            n_head * int(head_geometries[0]["latent_dim"])
            + int(v_geometry["latent_dim"])
        ),
        "total_size": matrix.numel(),
    }


def project_attention_target(
    matrix: torch.Tensor,
    *,
    target: str,
    config: dict[str, Any],
    layer: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    ratio = float(config["block_fht_latent_ratio"])
    ratios = config.get("block_fht_latent_ratios")
    if isinstance(ratios, dict):
        if target == "attn.c_attn":
            ratio = float(
                ratios.get(
                    "attn.c_attn.qk_headwise",
                    ratios.get("attn.c_attn.v", ratio),
                )
            )
        else:
            ratio = float(ratios.get(target, ratio))
    layers = int(config["block_fht_layers"])
    seed = int(config["block_fht_seed"])
    if target == "attn.c_attn":
        return project_c_attn(
            matrix,
            n_embd=int(config["n_embd"]),
            n_head=int(config["n_head"]),
            ratio=ratio,
            layers=layers,
            base_seed=seed,
            layer=layer,
        )
    if target == "attn.c_proj":
        projected, geometry = project_matrix(
            matrix,
            ratio=ratio,
            layers=layers,
            seed=seed + layer * 4 + 1,
        )
        return projected, geometry
    raise ValueError(f"unsupported attention target: {target}")


def projection_metrics(
    target_chord: torch.Tensor,
    dense_direction: torch.Tensor,
    projected_chord: torch.Tensor,
    projected_direction: torch.Tensor,
) -> dict[str, float]:
    chord = direction_metrics(target_chord, projected_chord)
    dense = direction_metrics(target_chord, dense_direction)
    projected = direction_metrics(target_chord, projected_direction)
    chord_energy = target_chord.double().square().sum().clamp_min(1e-30)
    direction_energy = dense_direction.double().square().sum().clamp_min(1e-30)
    return {
        "dense_direction_cosine": dense["cosine"],
        "dense_positive_step_line_recovery": dense[
            "positive_step_line_recovery"
        ],
        "tangent_chord_energy_fraction": float(
            projected_chord.double().square().sum() / chord_energy
        ),
        "tangent_chord_cosine": chord["cosine"],
        "projected_direction_energy_fraction": float(
            projected_direction.double().square().sum() / direction_energy
        ),
        "projected_direction_chord_cosine": projected["cosine"],
        "projected_positive_step_line_recovery": projected[
            "positive_step_line_recovery"
        ],
        "chord_fro": float(target_chord.double().norm()),
        "dense_direction_fro": float(dense_direction.double().norm()),
    }


def weighted_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    weights = torch.tensor(
        [float(row["chord_fro"]) ** 2 for row in rows], dtype=torch.float64
    )

    def weighted(key: str) -> float:
        values = torch.tensor(
            [float(row[key]) for row in rows], dtype=torch.float64
        )
        return float((weights * values).sum() / weights.sum())

    keys = [
        "dense_direction_cosine",
        "dense_positive_step_line_recovery",
        "tangent_chord_energy_fraction",
        "tangent_chord_cosine",
        "projected_direction_energy_fraction",
        "projected_direction_chord_cosine",
        "projected_positive_step_line_recovery",
        "haar_rank_fraction",
    ]
    return {
        "cells": len(rows),
        **{key: weighted(key) for key in keys},
        "minimum_tangent_chord_energy_fraction": min(
            float(row["tangent_chord_energy_fraction"]) for row in rows
        ),
        "maximum_tangent_chord_energy_fraction": max(
            float(row["tangent_chord_energy_fraction"]) for row in rows
        ),
    }


def geometry_dimensions(geometry: dict[str, Any]) -> tuple[int, int]:
    """Return coordinate and ambient dimensions for simple or QK/V charts."""
    latent_dim = (
        geometry["total_latent_dim"]
        if "total_latent_dim" in geometry
        else geometry["latent_dim"]
    )
    total_size = (
        geometry["total_size"]
        if "total_size" in geometry
        else geometry["size"]
    )
    return int(latent_dim), int(total_size)


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
    phases = list(zip(boundaries[:-1], boundaries[1:], strict=True))
    if not layers or not phases:
        raise ValueError("layers and phase boundaries must be non-empty")
    config = json.loads(args.production_config.read_text())
    expected = {
        "attn.c_attn.qk_headwise",
        "attn.c_attn.v",
        "attn.c_proj",
    }
    if set(config["block_fht_targets"]) != expected:
        raise ValueError(
            "production config must be the deployed qk-headwise/v/c_proj family"
        )

    snapshot_paths = [
        args.snapshot_dir / f"step_{step:06d}.pt" for step in boundaries
    ]
    probe_paths = [
        args.probe_dir / f"step_{start:06d}.pt" for start, _ in phases
    ]
    missing = [
        str(path)
        for path in (*snapshot_paths, *probe_paths)
        if not path.is_file()
    ]
    if missing:
        raise ValueError("missing required snapshots/probes: " + ", ".join(missing))
    steps, values, snapshot_metadata = load_snapshots(
        snapshot_paths,
        layers=set(layers),
        targets={"attn.c_attn", "attn.c_proj"},
    )
    if steps != boundaries:
        raise ValueError("loaded snapshot steps do not match phase boundaries")
    step_index = {step: index for index, step in enumerate(steps)}

    rows: list[dict[str, Any]] = []
    run_identity_sha256: str | None = None
    geometries: dict[str, Any] = {}
    for phase_start, phase_end in phases:
        probe_path = args.probe_dir / f"step_{phase_start:06d}.pt"
        probe = torch.load(probe_path, map_location="cpu", weights_only=False)
        if probe.get("schema_version") != OPTIMIZER_PROBE_SCHEMA_VERSION:
            raise ValueError("unexpected optimizer probe schema")
        if run_identity_sha256 is None:
            run_identity_sha256 = probe["run_identity_sha256"]
        elif probe["run_identity_sha256"] != run_identity_sha256:
            raise ValueError("optimizer probes do not share one identity")
        for layer in layers:
            for target in ("attn.c_attn", "attn.c_proj"):
                name = f"transformer.h.{layer}.{target}.weight"
                source = values[name][step_index[phase_start]].to(args.device)
                terminal = values[name][step_index[phase_end]].to(args.device)
                chord = terminal - source
                dense_direction = probe["parameters"][name][
                    "applied_direction_per_lr"
                ].to(args.device)
                projected_chord, geometry = project_attention_target(
                    chord,
                    target=target,
                    config=config,
                    layer=layer,
                )
                projected_direction, observed_geometry = (
                    project_attention_target(
                        dense_direction,
                        target=target,
                        config=config,
                        layer=layer,
                    )
                )
                if geometry != observed_geometry:
                    raise ValueError("projection geometry changed within one cell")
                latent_dim, total_size = geometry_dimensions(geometry)
                haar_fraction = latent_dim / total_size
                haar_sd = math.sqrt(
                    2.0
                    * latent_dim
                    * (total_size - latent_dim)
                    / (total_size * total_size * (total_size + 2))
                )
                row = {
                    "parameter": name,
                    "target": target,
                    "layer": layer,
                    "phase_start": phase_start,
                    "phase_end": phase_end,
                    "latent_dim": latent_dim,
                    "target_size": total_size,
                    "haar_rank_fraction": haar_fraction,
                    "haar_projection_sd": haar_sd,
                    **projection_metrics(
                        chord,
                        dense_direction,
                        projected_chord,
                        projected_direction,
                    ),
                }
                rows.append(row)
                geometries[f"layer{layer}.{target}"] = geometry
                del source, terminal, chord, dense_direction
                del projected_chord, projected_direction
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()

    by_target = {
        target: weighted_summary(
            [row for row in rows if row["target"] == target]
        )
        for target in ("attn.c_attn", "attn.c_proj")
    }
    aggregate = weighted_summary(rows)
    aggregate["decision"] = (
        "BLOCKFHT_TANGENT_ALIGNED_ABOVE_EQUAL_RANK_HAAR"
        if aggregate["tangent_chord_energy_fraction"]
        > aggregate["haar_rank_fraction"] + 3.0 * max(
            float(row["haar_projection_sd"]) for row in rows
        )
        else "BLOCKFHT_TANGENT_NOT_ALIGNED_ABOVE_EQUAL_RANK_HAAR"
    )

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "attention_tangent_cells.csv", rows)
    repo_root = Path(__file__).resolve().parents[2]
    result = {
        "schema_version": "mai_124m_attention_blockfht_tangent_v1",
        "scientific_question": (
            "Does the deployed one-percent qk-headwise/v/c_proj BlockFHT "
            "tangent contain dense Muon attention phase chords and directions?"
        ),
        "production_config": str(args.production_config),
        "production_config_sha256": file_sha256(args.production_config),
        "source_commit": git_commit(repo_root),
        "source_sha256": file_sha256(Path(__file__)),
        "snapshot_run_identity_sha256": snapshot_metadata[
            "run_identity_sha256"
        ],
        "optimizer_probe_run_identity_sha256": run_identity_sha256,
        "snapshot_paths": [
            {"path": str(path), "sha256": file_sha256(path)}
            for path in snapshot_paths
        ],
        "optimizer_probe_paths": [
            {"path": str(path), "sha256": file_sha256(path)}
            for path in probe_paths
        ],
        "layers": layers,
        "phase_boundaries": boundaries,
        "geometry": geometries,
        "by_target": by_target,
        "aggregate": aggregate,
        "interpretation": {
            "tangent_chord_energy_fraction": (
                "best Euclidean fit of the realized dense phase chord inside "
                "the fixed production tangent"
            ),
            "projected_positive_step_line_recovery": (
                "phase-chord energy recovered by the positive line through "
                "the exact dense Muon direction after tangent projection"
            ),
            "haar_rank_fraction": (
                "expected projection energy for an equal-rank Haar-random "
                "subspace; comparison isolates BlockFHT orientation/locality"
            ),
        },
        "limitations": [
            "One optimizer path is not the global manifold of good solutions.",
            "Euclidean weight-space capture does not include the attention functional metric.",
            "The test evaluates the fixed tangent, not nonlinear modulation or Mapping Loss.",
        ],
        "elapsed_seconds": time.time() - started,
    }
    (args.output / "attention_tangent_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()
