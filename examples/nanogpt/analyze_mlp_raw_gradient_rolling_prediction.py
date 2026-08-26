#!/usr/bin/env python3
"""Measure strictly causal next-probe prediction by raw-gradient factors.

For every optimizer probe after a fixed history window, fit shared left/right
singular spaces using only the preceding probes and score the immediately
following raw clipped gradient in the exact low-rank-manifold tangent

    {U X + Y V^T}.

Unlike a previous-window/next-window aggregate, this never uses a future
gradient to construct or average the evaluation target.
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

from examples.nanogpt.analyze_mlp_highcadence_basis import file_sha256
from examples.nanogpt.analyze_mlp_optimizer_probe_span import load_probe_inventory
from examples.nanogpt.analyze_mlp_raw_gradient_factor_transport import (
    canonical_overlap,
    exact_singular_factors,
    fit_shared_factors,
    tangent_capture,
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


def phase_for_step(step: int, discovery_stop: int, validation_stop: int) -> str:
    if step <= discovery_stop:
        return "discovery"
    if step <= validation_stop:
        return "validation"
    return "test"


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["parameter"]),
            str(row["target"]),
            int(row["rank"]),
            str(row["eval_phase"]),
        )
        groups.setdefault(key, []).append(row)
        overall_key = (key[0], key[1], key[2], "all")
        groups.setdefault(overall_key, []).append(row)

    summaries: list[dict[str, Any]] = []
    metric_names = (
        "left_capture",
        "right_capture",
        "bilinear_core_capture",
        "rank_manifold_tangent_capture",
    )
    for (parameter, target, rank, phase), members in sorted(groups.items()):
        energies = torch.tensor(
            [float(row["direction_energy"]) for row in members],
            dtype=torch.float64,
        )
        total_energy = energies.sum().clamp_min(1e-30)
        summary: dict[str, Any] = {
            "parameter": parameter,
            "target": target,
            "rank": rank,
            "eval_phase": phase,
            "sample_count": len(members),
            "eval_step_start": min(int(row["eval_step"]) for row in members),
            "eval_step_stop": max(int(row["eval_step"]) for row in members),
        }
        for metric in metric_names:
            values = torch.tensor(
                [float(row[metric]) for row in members], dtype=torch.float64
            )
            summary[f"{metric}_energy_weighted_mean"] = float(
                (values * energies).sum() / total_energy
            )
            summary[f"{metric}_median"] = float(values.quantile(0.5))
            summary[f"{metric}_p10"] = float(values.quantile(0.1))
            summary[f"{metric}_minimum"] = float(values.min())
        for side in ("left", "right"):
            values = torch.tensor(
                [
                    float(row[f"{side}_current_overlap_mean_squared_cosine"])
                    for row in members
                ],
                dtype=torch.float64,
            )
            summary[f"{side}_current_overlap_mean"] = float(values.mean())
            summary[f"{side}_current_overlap_median"] = float(values.quantile(0.5))
        summaries.append(summary)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", default="6")
    parser.add_argument("--targets", default="mlp.c_fc,mlp.c_proj")
    parser.add_argument("--ranks", default="1,2,3,4,5,6")
    parser.add_argument("--fit-component-rank", type=int, default=32)
    parser.add_argument("--history-probes", type=int, default=10)
    parser.add_argument("--discovery-stop", type=int, default=119)
    parser.add_argument("--validation-stop", type=int, default=179)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    ranks = parse_int_list(args.ranks)
    maximum_rank = max(max(ranks), args.fit_component_rank)
    if args.history_probes < 2:
        raise ValueError("history-probes must be at least two")
    paths = sorted(args.probe_dir.glob("step_*.pt"))
    steps, inventory, input_metadata = load_probe_inventory(
        paths,
        layers=set(parse_int_list(args.layers)),
        targets={item for item in args.targets.split(",") if item},
    )
    if len(steps) <= args.history_probes:
        raise ValueError("not enough probes for the requested history")

    rows: list[dict[str, Any]] = []
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
            "history_probes": args.history_probes,
            "fit_component_rank": args.fit_component_rank,
        }
        for eval_index in range(args.history_probes, len(steps)):
            fit_indices = list(
                range(eval_index - args.history_probes, eval_index)
            )
            fitted_left, fitted_right = fit_shared_factors(
                left,
                singular_values,
                right,
                fit_indices,
                max(ranks),
                args.fit_component_rank,
            )
            direction = directions[eval_index]
            direction_energy = float(direction.double().square().sum())
            for rank in ranks:
                left_basis = fitted_left[:, :rank]
                right_basis = fitted_right[:, :rank]
                capture = tangent_capture(direction, left_basis, right_basis)
                left_overlap = canonical_overlap(
                    left_basis, left[eval_index][:, :rank]
                )
                right_overlap = canonical_overlap(
                    right_basis, right[eval_index][:, :rank]
                )
                stored = rank * (matrix_rows + matrix_columns)
                rows.append(
                    {
                        **common,
                        "rank": rank,
                        "stored_scalars": stored,
                        "stored_scalar_fraction": stored / parameter_size,
                        "fit_probe_start": eval_index - args.history_probes,
                        "fit_probe_stop": eval_index - 1,
                        "fit_step_start": steps[eval_index - args.history_probes],
                        "fit_step_stop": steps[eval_index - 1],
                        "eval_probe": eval_index,
                        "eval_step": steps[eval_index],
                        "eval_phase": phase_for_step(
                            steps[eval_index],
                            args.discovery_stop,
                            args.validation_stop,
                        ),
                        "direction_energy": direction_energy,
                        **capture,
                        "left_current_overlap_mean_squared_cosine": left_overlap[0],
                        "left_current_overlap_minimum_squared_cosine": left_overlap[1],
                        "left_current_overlap_maximum_squared_cosine": left_overlap[2],
                        "right_current_overlap_mean_squared_cosine": right_overlap[0],
                        "right_current_overlap_minimum_squared_cosine": right_overlap[1],
                        "right_current_overlap_maximum_squared_cosine": right_overlap[2],
                    }
                )
        del directions, left, singular_values, right
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

    summaries = summarize_rows(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    rolling_path = args.output / "rolling_next_probe_capture.csv"
    summary_path = args.output / "rolling_next_probe_summary.csv"
    write_csv(rolling_path, rows)
    write_csv(summary_path, summaries)
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_mlp_raw_gradient_rolling_prediction_v1",
        "steps": steps,
        "sample_count": len(steps),
        "prediction_count_per_parameter": len(steps) - args.history_probes,
        "parameters": sorted(inventory),
        "ranks": ranks,
        "history_probes": args.history_probes,
        "fit_component_rank": args.fit_component_rank,
        "input": input_metadata,
        "method": {
            "fit": "preceding history-probes only; aggregate truncated raw-gradient row/column covariance",
            "evaluation": "immediately following single raw clipped gradient",
            "tangent": "orthogonal projection into {U X + Y V^T}",
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
            rolling_path.name: file_sha256(rolling_path),
            summary_path.name: file_sha256(summary_path),
        },
        "limitations": [
            "Probe spacing is approximately 2--3 optimizer steps, so this predicts the next saved probe rather than every optimizer step.",
            "This scores raw-gradient Euclidean energy, not terminal validation CE.",
            "The fitted chart may rotate between predictions; a trainable low-rank state must realize such transport without retaining dense state.",
        ],
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "rolling_rows": len(rows),
                "summary_rows": len(summaries),
                "metadata": str(metadata_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
