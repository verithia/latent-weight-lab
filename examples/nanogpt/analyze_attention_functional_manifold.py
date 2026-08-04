#!/usr/bin/env python3
"""Analyze gauge-invariant attention-operator trajectories.

For every sampled layer and head this applies the Mapping-Networks paper's
trajectory-PCA idea to the operators that actually determine attention:

* score kernel: ``Q_h.T @ K_h``;
* value/output kernel: ``V_h.T @ O_h.T``.

The comparison is the joint raw factor trajectory for the same head.  This
separates a genuinely simpler functional manifold from motion that merely
changes the Q/K or V/O gauge.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_attention_fht_block_skew_tangent import (
    file_sha256,
    write_csv,
)
from examples.nanogpt.analyze_parameter_trajectory import (
    energy_dimension,
    load_snapshots,
    parse_int_list,
    pca_from_rows,
)


def git_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def trajectory_metrics(rows: torch.Tensor) -> dict[str, float | int]:
    if rows.ndim != 2 or rows.shape[0] < 3:
        raise ValueError("trajectory rows must be [time, coordinates]")
    centered = rows - rows.mean(dim=0, keepdim=True)
    eigenvalues, _ = pca_from_rows(centered)
    total = eigenvalues.sum().clamp_min(1e-30)
    probabilities = eigenvalues / total
    participation = total.square() / eigenvalues.square().sum().clamp_min(1e-30)

    displacements = rows - rows[0]
    chord = displacements[-1]
    chord_energy = chord.double().square().sum().clamp_min(1e-30)
    chord_norm = chord_energy.sqrt()
    progress = (displacements.double() @ chord.double()) / chord_energy
    residual = (
        displacements.double()
        - progress.unsqueeze(1) * chord.double().unsqueeze(0)
    )
    relative_residual = residual.norm(dim=1) / displacements.double().norm(
        dim=1
    ).clamp_min(1e-30)

    increments = rows[1:] - rows[:-1]
    increment_norms = increments.double().norm(dim=1)
    path_length = increment_norms.sum()
    cosines = torch.nn.functional.cosine_similarity(
        increments[:-1].double(),
        increments[1:].double(),
        dim=1,
        eps=1e-30,
    )
    turns = torch.rad2deg(torch.acos(cosines.clamp(-1.0, 1.0)))
    displacement_energy = displacements.double().square().sum(dim=1)
    ray_dot = displacements.double() @ chord.double()
    ray_recovery = ray_dot.clamp_min(0.0).square() / (
        displacement_energy * chord_energy
    ).clamp_min(1e-30)
    ray_recovery = ray_recovery[1:]
    return {
        "pc1_energy": float(probabilities[0]),
        "pc1_pc2_energy": float(probabilities[:2].sum()),
        "dimension_90pct": energy_dimension(eigenvalues, 0.90),
        "dimension_95pct": energy_dimension(eigenvalues, 0.95),
        "dimension_99pct": energy_dimension(eigenvalues, 0.99),
        "participation_dimension": float(participation),
        "path_length_over_chord": float(path_length / chord_norm),
        "median_relative_terminal_ray_residual": float(
            relative_residual[1:].median()
        ),
        "mean_terminal_ray_recovery": float(ray_recovery.mean()),
        "minimum_terminal_ray_recovery": float(ray_recovery.min()),
        "mean_consecutive_increment_cosine": float(cosines.mean()),
        "median_turn_degrees": float(turns.median()),
        "maximum_turn_degrees": float(turns.max()),
        "monotone_terminal_progress_fraction": float(
            (progress[1:] >= progress[:-1] - 1e-7).float().mean()
        ),
        "terminal_displacement_fro": float(chord_norm),
    }


def product_kernel(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    if first.ndim != 2 or second.shape != first.shape:
        raise ValueError("kernel factors must be same-shaped matrices")
    return first.transpose(0, 1) @ second


def product_delta_singular_values(
    first_before: torch.Tensor,
    second_before: torch.Tensor,
    first_after: torch.Tensor,
    second_after: torch.Tensor,
) -> torch.Tensor:
    """Return exact nonzero singular values of a product-kernel delta."""
    if not (
        first_before.shape
        == second_before.shape
        == first_after.shape
        == second_after.shape
    ):
        raise ValueError("all product factors must share one shape")
    first_delta = first_after - first_before
    second_delta = second_after - second_before
    left = torch.cat(
        (first_after.transpose(0, 1), first_delta.transpose(0, 1)), dim=1
    )
    right = torch.cat(
        (second_delta.transpose(0, 1), second_before.transpose(0, 1)),
        dim=1,
    )
    q_left, r_left = torch.linalg.qr(left, mode="reduced")
    q_right, r_right = torch.linalg.qr(right, mode="reduced")
    del q_left, q_right
    return torch.linalg.svdvals(r_left @ r_right.transpose(0, 1))


def spectrum_metrics(values: torch.Tensor) -> dict[str, float | int]:
    energy = values.double().square()
    total = energy.sum().clamp_min(1e-30)
    probability = energy / total
    entropy = -(probability * probability.clamp_min(1e-30).log()).sum()
    return {
        "rank90": energy_dimension(energy, 0.90),
        "rank95": energy_dimension(energy, 0.95),
        "stable_rank": float(total / energy[0].clamp_min(1e-30)),
        "entropy_effective_rank": float(entropy.exp()),
        "top1_energy": float(probability[0]),
    }


def weighted_summary(
    rows: list[dict[str, Any]], prefix: str
) -> dict[str, float | int]:
    if not rows:
        raise ValueError("cannot summarize zero rows")
    weights = torch.tensor(
        [float(row["functional_terminal_displacement_fro"]) ** 2 for row in rows],
        dtype=torch.float64,
    )
    keys = (
        "pc1_energy",
        "pc1_pc2_energy",
        "dimension_90pct",
        "dimension_95pct",
        "dimension_99pct",
        "participation_dimension",
        "path_length_over_chord",
        "median_relative_terminal_ray_residual",
        "mean_terminal_ray_recovery",
        "minimum_terminal_ray_recovery",
        "mean_consecutive_increment_cosine",
        "median_turn_degrees",
        "maximum_turn_degrees",
        "monotone_terminal_progress_fraction",
    )

    def weighted(key: str) -> float:
        values = torch.tensor(
            [float(row[f"{prefix}_{key}"]) for row in rows],
            dtype=torch.float64,
        )
        return float((weights * values).sum() / weights.sum())

    return {"cells": len(rows), **{key: weighted(key) for key in keys}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    plan = json.loads(args.plan.read_text())
    if (
        plan.get("schema_version")
        != "mai_124m_attention_functional_manifold_plan_v1"
    ):
        raise ValueError("unexpected plan schema")
    protocol = plan["protocol"]
    layers = [int(value) for value in protocol["layers"]]
    steps = [int(value) for value in protocol["steps"]]
    snapshot_paths = [
        args.snapshot_dir / f"step_{step:06d}.pt" for step in steps
    ]
    missing = [str(path) for path in snapshot_paths if not path.is_file()]
    if missing:
        raise ValueError("missing snapshots: " + ", ".join(missing))
    loaded_steps, values, metadata = load_snapshots(
        snapshot_paths,
        layers=set(layers),
        targets={"attn.c_attn", "attn.c_proj"},
    )
    if loaded_steps != steps:
        raise ValueError("loaded snapshot steps do not match the plan")
    config = metadata["model_config"]
    n_embd = int(config["n_embd"])
    n_head = int(config["n_head"])
    head_dim = n_embd // n_head
    rows: list[dict[str, Any]] = []
    for layer in layers:
        c_attn_name = f"transformer.h.{layer}.attn.c_attn.weight"
        c_proj_name = f"transformer.h.{layer}.attn.c_proj.weight"
        c_attn_steps = [
            value.to(args.device, dtype=torch.float32)
            for value in values[c_attn_name]
        ]
        c_proj_steps = [
            value.to(args.device, dtype=torch.float32)
            for value in values[c_proj_name]
        ]
        for head in range(n_head):
            start = head * head_dim
            stop = start + head_dim
            factor_sequences = {
                "score_qk": (
                    [value[start:stop] for value in c_attn_steps],
                    [value[n_embd + start : n_embd + stop] for value in c_attn_steps],
                ),
                "value_output": (
                    [value[2 * n_embd + start : 2 * n_embd + stop] for value in c_attn_steps],
                    [value[:, start:stop].transpose(0, 1) for value in c_proj_steps],
                ),
            }
            for family, (first, second) in factor_sequences.items():
                kernels = [
                    product_kernel(left, right)
                    for left, right in zip(first, second, strict=True)
                ]
                functional_rows = torch.stack(kernels).flatten(1)
                factor_rows = torch.stack(
                    [
                        torch.cat((left.flatten(), right.flatten()))
                        for left, right in zip(first, second, strict=True)
                    ]
                )
                functional = trajectory_metrics(functional_rows)
                factors = trajectory_metrics(factor_rows)
                terminal_spectrum = spectrum_metrics(
                    product_delta_singular_values(
                        first[0], second[0], first[-1], second[-1]
                    )
                )
                increment_spectra = [
                    spectrum_metrics(
                        product_delta_singular_values(
                            first[index],
                            second[index],
                            first[index + 1],
                            second[index + 1],
                        )
                    )
                    for index in range(len(steps) - 1)
                ]
                row: dict[str, Any] = {
                    "layer": layer,
                    "head": head,
                    "family": family,
                    "snapshots": len(steps),
                    **{f"functional_{key}": value for key, value in functional.items()},
                    **{f"factor_{key}": value for key, value in factors.items()},
                    **{
                        f"terminal_delta_{key}": value
                        for key, value in terminal_spectrum.items()
                    },
                }
                for key in increment_spectra[0]:
                    row[f"mean_increment_{key}"] = sum(
                        float(item[key]) for item in increment_spectra
                    ) / len(increment_spectra)
                rows.append(row)
                del kernels, functional_rows, factor_rows
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
        del c_attn_steps, c_proj_steps

    aggregate = {
        "functional": weighted_summary(rows, "functional"),
        "factors": weighted_summary(rows, "factor"),
        "by_family": {
            family: {
                "functional": weighted_summary(
                    [row for row in rows if row["family"] == family],
                    "functional",
                ),
                "factors": weighted_summary(
                    [row for row in rows if row["family"] == family],
                    "factor",
                ),
            }
            for family in ("score_qk", "value_output")
        },
        "terminal_delta_rank95_mean": sum(
            float(row["terminal_delta_rank95"]) for row in rows
        )
        / len(rows),
        "increment_rank95_mean": sum(
            float(row["mean_increment_rank95"]) for row in rows
        )
        / len(rows),
    }
    thresholds = plan["interpretation_rule"]
    path_ratio = float(aggregate["functional"]["path_length_over_chord"]) / float(
        aggregate["factors"]["path_length_over_chord"]
    )
    pc1_gain = float(aggregate["functional"]["pc1_energy"]) - float(
        aggregate["factors"]["pc1_energy"]
    )
    simpler = (
        path_ratio <= float(thresholds["maximum_path_length_ratio"])
        and pc1_gain >= float(thresholds["minimum_pc1_energy_gain"])
    )
    low_rank_delta = (
        float(aggregate["terminal_delta_rank95_mean"])
        <= float(thresholds["maximum_terminal_delta_rank95_mean"])
    )
    classification = (
        "FUNCTIONAL_QUOTIENT_SIMPLER_AND_LOW_RANK"
        if simpler and low_rank_delta
        else "FUNCTIONAL_QUOTIENT_NOT_SUFFICIENTLY_SIMPLER"
    )

    args.output.mkdir(parents=True, exist_ok=True)
    cells_path = args.output / "attention_functional_manifold_cells.csv"
    write_csv(cells_path, rows)
    repo_root = Path(__file__).resolve().parents[2]
    result = {
        "schema_version": "mai_124m_attention_functional_manifold_v1",
        "scientific_question": (
            "Does quotienting Q/K and V/O gauge motion reveal a straighter, "
            "lower-dimensional attention-operator trajectory?"
        ),
        "source_commit": git_commit(repo_root),
        "source_sha256": file_sha256(Path(__file__)),
        "plan": {"path": str(args.plan), "sha256": file_sha256(args.plan)},
        "snapshot_run_identity_sha256": metadata["run_identity_sha256"],
        "snapshot_paths": [
            {"path": str(path), "sha256": file_sha256(path)}
            for path in snapshot_paths
        ],
        "layers": layers,
        "steps": steps,
        "operators": {
            "score_qk": "Q_h.T @ K_h",
            "value_output": "V_h.T @ O_h.T",
        },
        "aggregate": aggregate,
        "comparison": {
            "functional_over_factor_path_length_ratio": path_ratio,
            "functional_minus_factor_pc1_energy": pc1_gain,
        },
        "decision": {
            "classification": classification,
            "functional_quotient_materially_simpler": simpler,
            "functional_terminal_delta_low_rank": low_rank_delta,
            "thresholds": thresholds,
            "automatic_training_authorized": False,
        },
        "cells_csv": {
            "path": str(cells_path),
            "sha256": file_sha256(cells_path),
        },
        "limitations": [
            "Temporal PCA rank is bounded by the 17 sampled checkpoints and is not a global solution-manifold dimension.",
            "Kernel products quotient exact linear factor gauges but not softmax state dependence or residual-stream geometry.",
            "A simpler quotient trajectory would motivate a direct operator chart, not authorize a training run by itself.",
        ],
        "elapsed_seconds": time.time() - started,
    }
    result_path = args.output / "attention_functional_manifold_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
