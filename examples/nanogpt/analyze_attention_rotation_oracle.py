#!/usr/bin/env python3
"""Measure radial and orthogonal attention charts on dense task gradients.

The deployed one-percent affine BlockFHT tangent captures only its nominal
random share of the clipped task gradient.  This diagnostic asks which
state-dependent matrix orbit contains the missing descent direction without
fitting a dense additive residual:

* output/input channel gains (radial directions);
* left/right orthogonal orbits;
* low-rank truncations of the left- and right-orbit skew generators.

Q/K are evaluated as one layer-shared input rotation, matching the proposed
implementation.  V and c_proj receive their own input rotations.  The result
is an oracle over measured dense gradients, not a trained candidate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_parameter_trajectory import parse_int_list
from examples.nanogpt.parameter_trajectory import (
    OPTIMIZER_PROBE_SCHEMA_VERSION,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
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
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def positive_line_recovery(
    target: torch.Tensor,
    direction: torch.Tensor,
) -> float:
    target = target.double()
    direction = direction.double()
    target_energy = target.square().sum().clamp_min(1e-30)
    direction_energy = direction.square().sum().clamp_min(1e-30)
    dot = (target * direction).sum().clamp_min(0.0)
    return float(dot.square() / (target_energy * direction_energy))


def row_gain_direction(
    weight: torch.Tensor,
    gradient: torch.Tensor,
) -> torch.Tensor:
    coefficient = (gradient * weight).sum(dim=1) / weight.square().sum(
        dim=1
    ).clamp_min(1e-30)
    return coefficient[:, None] * weight


def column_gain_direction(
    weight: torch.Tensor,
    gradient: torch.Tensor,
) -> torch.Tensor:
    coefficient = (gradient * weight).sum(dim=0) / weight.square().sum(
        dim=0
    ).clamp_min(1e-30)
    return weight * coefficient[None, :]


def left_orthogonal_direction(
    weight: torch.Tensor,
    gradient: torch.Tensor,
) -> torch.Tensor:
    """Return ``skew(G W^T) W`` without retaining the square skew."""

    return 0.5 * (
        gradient @ (weight.transpose(0, 1) @ weight)
        - weight @ (gradient.transpose(0, 1) @ weight)
    )


def right_orthogonal_direction(
    weight: torch.Tensor,
    gradient: torch.Tensor,
) -> torch.Tensor:
    """Return ``W skew(W^T G)`` without retaining the square skew."""

    return 0.5 * (
        (weight @ weight.transpose(0, 1)) @ gradient
        - (weight @ gradient.transpose(0, 1)) @ weight
    )


def direction_span_recovery(
    target: torch.Tensor,
    directions: list[torch.Tensor],
) -> float:
    target_flat = target.double().reshape(-1)
    matrix = torch.stack(
        [direction.double().reshape(-1) for direction in directions],
        dim=1,
    )
    coefficients = torch.linalg.lstsq(matrix, target_flat).solution
    prediction = matrix @ coefficients
    return float(
        prediction.square().sum()
        / target_flat.square().sum().clamp_min(1e-30)
    )


def low_rank_right_recovery(
    weight: torch.Tensor,
    gradient: torch.Tensor,
    ranks: list[int],
) -> dict[int, float]:
    skew = 0.5 * (
        weight.transpose(0, 1) @ gradient
        - gradient.transpose(0, 1) @ weight
    )
    left, singular, right_h = torch.linalg.svd(
        skew, full_matrices=False
    )
    output: dict[int, float] = {}
    for rank in ranks:
        truncated = (
            (left[:, :rank] * singular[:rank]) @ right_h[:rank]
        )
        output[rank] = positive_line_recovery(
            gradient,
            weight @ truncated,
        )
    return output


def low_rank_left_recovery(
    weight: torch.Tensor,
    gradient: torch.Tensor,
    ranks: list[int],
) -> dict[int, float]:
    skew = 0.5 * (
        gradient @ weight.transpose(0, 1)
        - weight @ gradient.transpose(0, 1)
    )
    left, singular, right_h = torch.linalg.svd(
        skew, full_matrices=False
    )
    output: dict[int, float] = {}
    for rank in ranks:
        truncated = (
            (left[:, :rank] * singular[:rank]) @ right_h[:rank]
        )
        output[rank] = positive_line_recovery(
            gradient,
            truncated @ weight,
        )
    return output


def cell_metrics(
    *,
    step: int,
    layer: int,
    target: str,
    weight: torch.Tensor,
    gradient: torch.Tensor,
    ranks: list[int],
) -> dict[str, Any]:
    weight = weight.float()
    gradient = gradient.float()
    row = row_gain_direction(weight, gradient)
    column = column_gain_direction(weight, gradient)
    left = left_orthogonal_direction(weight, gradient)
    right = right_orthogonal_direction(weight, gradient)
    low_rank_left = low_rank_left_recovery(weight, gradient, ranks)
    low_rank = low_rank_right_recovery(weight, gradient, ranks)
    return {
        "step": step,
        "layer": layer,
        "target": target,
        "gradient_fro": float(gradient.double().norm()),
        "row_gain_recovery": positive_line_recovery(gradient, row),
        "column_gain_recovery": positive_line_recovery(gradient, column),
        "left_orbit_recovery": positive_line_recovery(gradient, left),
        "right_orbit_recovery": positive_line_recovery(gradient, right),
        "radial_bilateral_orbit_span_recovery": direction_span_recovery(
            gradient,
            [row, column, left, right],
        ),
        **{
            f"left_skew_rank{rank}_recovery": value
            for rank, value in low_rank_left.items()
        },
        **{
            f"right_skew_rank{rank}_recovery": value
            for rank, value in low_rank.items()
        },
    }


def weighted_summary(
    rows: list[dict[str, Any]],
    metric_names: list[str],
) -> dict[str, Any]:
    weights = torch.tensor(
        [float(row["gradient_fro"]) ** 2 for row in rows],
        dtype=torch.float64,
    )
    output: dict[str, Any] = {"cells": len(rows)}
    for name in metric_names:
        values = torch.tensor(
            [float(row[name]) for row in rows], dtype=torch.float64
        )
        output[name] = float((weights * values).sum() / weights.sum())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", default="0,3,6,9,11")
    parser.add_argument("--steps", default="0,60,120,180")
    parser.add_argument("--skew-ranks", default="2,4,8,16,32,64,128")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    started = time.time()
    layers = parse_int_list(args.layers)
    steps = parse_int_list(args.steps)
    ranks = parse_int_list(args.skew_ranks)
    if not layers or not steps or not ranks:
        raise ValueError("layers, steps, and skew ranks must be non-empty")
    if any(rank <= 0 or rank % 2 for rank in ranks):
        raise ValueError("skew ranks must be positive and even")

    probe_paths = [
        args.probe_dir / f"step_{step:06d}.pt" for step in steps
    ]
    missing = [str(path) for path in probe_paths if not path.is_file()]
    if missing:
        raise ValueError("missing optimizer probes: " + ", ".join(missing))

    rows: list[dict[str, Any]] = []
    run_identity_sha256: str | None = None
    n_embd: int | None = None
    for path in probe_paths:
        probe = torch.load(path, map_location="cpu", weights_only=False)
        if probe.get("schema_version") != OPTIMIZER_PROBE_SCHEMA_VERSION:
            raise ValueError(f"unexpected optimizer probe schema in {path}")
        if run_identity_sha256 is None:
            run_identity_sha256 = probe["run_identity_sha256"]
        elif probe["run_identity_sha256"] != run_identity_sha256:
            raise ValueError("optimizer probes do not share one identity")
        observed_n_embd = int(probe["model_config"]["n_embd"])
        if n_embd is None:
            n_embd = observed_n_embd
        elif observed_n_embd != n_embd:
            raise ValueError("optimizer probes do not share one model width")
        assert n_embd is not None
        step = int(probe["step"])
        for layer in layers:
            c_attn = probe["parameters"][
                f"transformer.h.{layer}.attn.c_attn.weight"
            ]
            c_attn_weight = c_attn["weight_before_step"].to(args.device)
            c_attn_gradient = c_attn["gradient_after_clip"].to(args.device)
            rows.append(
                cell_metrics(
                    step=step,
                    layer=layer,
                    target="qk_shared_input",
                    weight=c_attn_weight[: 2 * n_embd],
                    gradient=c_attn_gradient[: 2 * n_embd],
                    ranks=ranks,
                )
            )
            rows.append(
                cell_metrics(
                    step=step,
                    layer=layer,
                    target="v_input",
                    weight=c_attn_weight[2 * n_embd :],
                    gradient=c_attn_gradient[2 * n_embd :],
                    ranks=ranks,
                )
            )
            c_proj = probe["parameters"][
                f"transformer.h.{layer}.attn.c_proj.weight"
            ]
            rows.append(
                cell_metrics(
                    step=step,
                    layer=layer,
                    target="cproj_input",
                    weight=c_proj["weight_before_step"].to(args.device),
                    gradient=c_proj["gradient_after_clip"].to(args.device),
                    ranks=ranks,
                )
            )
            del c_attn_weight, c_attn_gradient
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    metric_names = [
        "row_gain_recovery",
        "column_gain_recovery",
        "left_orbit_recovery",
        "right_orbit_recovery",
        "radial_bilateral_orbit_span_recovery",
        *[f"left_skew_rank{rank}_recovery" for rank in ranks],
        *[f"right_skew_rank{rank}_recovery" for rank in ranks],
    ]
    targets = ("qk_shared_input", "v_input", "cproj_input")
    by_target = {
        target: weighted_summary(
            [row for row in rows if row["target"] == target],
            metric_names,
        )
        for target in targets
    }
    aggregate = weighted_summary(rows, metric_names)
    selected_skew_rank = next(
        (
            rank
            for rank in ranks
            if min(
                float(
                    by_target[target][
                        f"right_skew_rank{rank}_recovery"
                    ]
                )
                for target in targets
            )
            >= 0.10
        ),
        None,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "attention_rotation_oracle_cells.csv"
    write_csv(csv_path, rows)
    repo_root = Path(__file__).resolve().parents[2]
    result = {
        "schema_version": "mai_124m_attention_rotation_oracle_v1",
        "scientific_question": (
            "Do radial gains or compact low-rank orthogonal input rotations "
            "contain the clipped dense attention task-gradient direction?"
        ),
        "source_commit": git_commit(repo_root),
        "source_sha256": file_sha256(Path(__file__)),
        "optimizer_probe_run_identity_sha256": run_identity_sha256,
        "optimizer_probe_paths": [
            {"path": str(path), "sha256": file_sha256(path)}
            for path in probe_paths
        ],
        "layers": layers,
        "steps": steps,
        "skew_ranks": ranks,
        "by_target": by_target,
        "aggregate": aggregate,
        "decision": {
            "selected_skew_rank": selected_skew_rank,
            "selected_cayley_pair_rank": (
                selected_skew_rank // 2
                if selected_skew_rank is not None
                else None
            ),
            "classification": (
                "PROMOTE_LOW_RANK_CAYLEY_SMALLEST_SCREEN"
                if selected_skew_rank is not None
                else "NO_COMPACT_ORTHOGONAL_DIRECTION_CLEARS_THRESHOLD"
            ),
            "threshold": 0.10,
        },
        "interpretation": {
            "gain_recovery": (
                "best positive line recovery along the exact per-row or "
                "per-column radial task-gradient projection"
            ),
            "orbit_recovery": (
                "positive line recovery of the canonical task-gradient-induced "
                "left or right orthogonal tangent direction"
            ),
            "right_skew_rank_recovery": (
                "recovery after truncating skew(W^T G) to the stated even "
                "matrix rank; skew rank 2r maps to r Cayley vector pairs"
            ),
            "left_skew_rank_recovery": (
                "recovery after truncating skew(G W^T) to the stated even "
                "matrix rank; skew rank 2r maps to r Cayley vector pairs"
            ),
        },
        "limitations": [
            "This is an oracle on dense Muon weights and gradients, not a trained generated model.",
            "A learned low-rank Cayley chart must still demonstrate that its moving coordinates reach the oracle direction.",
            "The chart preserves singular values; radial gains remain a separate structural degree of freedom.",
        ],
        "elapsed_seconds": time.time() - started,
    }
    result_path = args.output / "attention_rotation_oracle_result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
