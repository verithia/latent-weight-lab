#!/usr/bin/env python3
"""Isolate optimizer-induced description complexity on sealed MLP probes.

The same sampled gradient field is replayed through norm-matched raw-gradient,
pre-polar momentum, Muon-polar, exact-Muon, and AdamW update maps.  Each
counterfactual path is then passed through the empirical temporal-PC
rate--distortion oracle.  This is an offline operator audit: it neither trains
a model nor claims that the sampled Muon-path gradients equal gradients on a
counterfactual optimizer trajectory.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_mlp_highcadence_basis import file_sha256
from examples.nanogpt.analyze_mlp_optimizer_probe_span import load_probe_inventory
from examples.nanogpt.analyze_mlp_state_basis_rate_distortion import (
    analyze_parameter,
)
from examples.nanogpt.analyze_parameter_trajectory import parse_int_list, write_csv


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_float_list(value: str) -> list[float]:
    result = [float(item) for item in value.split(",") if item]
    if not result:
        raise ValueError("at least one threshold is required")
    return result


def load_probe_learning_rates(
    paths: list[Path], parameters: set[str]
) -> dict[str, list[float]]:
    result = {parameter: [] for parameter in parameters}
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        hyperparameters = payload.get("hyperparameters", {})
        for parameter in sorted(parameters):
            if parameter not in hyperparameters:
                raise ValueError(f"missing hyperparameters for {parameter}: {path}")
            result[parameter].append(float(hyperparameters[parameter]["lr"]))
    return result


def normalized_like(direction: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Scale every matrix direction to its paired reference Frobenius norm."""
    if direction.shape != reference.shape or direction.ndim < 2:
        raise ValueError("direction/reference shape mismatch")
    flat_direction = direction.flatten(1)
    flat_reference = reference.flatten(1)
    scale = flat_reference.norm(dim=1) / flat_direction.norm(dim=1).clamp_min(1e-30)
    return direction * scale.reshape((-1,) + (1,) * (direction.ndim - 1))


def cumulative_path(
    directions: torch.Tensor,
    *,
    steps: list[int],
    learning_rates: list[float],
) -> torch.Tensor:
    """Zero-order-hold integration from each probe to the next probe."""
    if directions.shape[0] != len(steps) or len(steps) != len(learning_rates):
        raise ValueError("direction/step/LR length mismatch")
    state = torch.zeros_like(directions[0], dtype=torch.float32)
    states = [state.clone()]
    for index in range(len(steps) - 1):
        interval = steps[index + 1] - steps[index]
        if interval <= 0:
            raise ValueError("steps must be strictly increasing")
        state = state + (
            float(interval)
            * float(learning_rates[index])
            * directions[index].float()
        )
        states.append(state.clone())
    return torch.stack(states)


def adamw_replay_path(
    gradient_descent: torch.Tensor,
    *,
    steps: list[int],
    learning_rates: list[float],
    beta1: float,
    beta2: float,
    epsilon: float,
) -> torch.Tensor:
    """Replay AdamW's adaptive direction under probe-wise zero-order hold.

    Weight decay is omitted because the audit targets optimizer-direction
    entropy rather than a particular counterfactual endpoint scale.
    """
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError("AdamW betas must be in [0,1)")
    gradient = -gradient_descent.float()
    first = torch.zeros_like(gradient[0])
    second = torch.zeros_like(gradient[0])
    state = torch.zeros_like(gradient[0])
    states = [state.clone()]
    optimizer_step = 0
    for index in range(len(steps) - 1):
        interval = steps[index + 1] - steps[index]
        if interval <= 0:
            raise ValueError("steps must be strictly increasing")
        for _ in range(interval):
            optimizer_step += 1
            first.mul_(beta1).add_(gradient[index], alpha=1.0 - beta1)
            second.mul_(beta2).addcmul_(
                gradient[index], gradient[index], value=1.0 - beta2
            )
            first_hat = first / (1.0 - beta1**optimizer_step)
            second_hat = second / (1.0 - beta2**optimizer_step)
            descent = -first_hat / (second_hat.sqrt() + epsilon)
            state = state + float(learning_rates[index]) * descent
        states.append(state.clone())
    return torch.stack(states)


def relative_path_error(candidate: torch.Tensor, observed: torch.Tensor) -> float:
    candidate = candidate.float() - candidate[0].float()
    observed = observed.float() - observed[0].float()
    candidate = candidate / candidate.flatten(1).norm(dim=1).mean().clamp_min(1e-30)
    observed = observed / observed.flatten(1).norm(dim=1).mean().clamp_min(1e-30)
    return float((candidate - observed).norm() / observed.norm().clamp_min(1e-30))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", default="6")
    parser.add_argument("--targets", default="mlp.c_fc,mlp.c_proj")
    parser.add_argument("--basis-rank", type=int, default=16)
    parser.add_argument("--thresholds", default="0.9,0.95")
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    paths = sorted(args.probe_dir.glob("step_*.pt"))
    layers = set(parse_int_list(args.layers))
    targets = {item for item in args.targets.split(",") if item}
    thresholds = parse_float_list(args.thresholds)
    started = time.time()
    steps, values, input_metadata = load_probe_inventory(
        paths, layers=layers, targets=targets
    )
    learning_rates = load_probe_learning_rates(paths, set(values))
    rows: list[dict[str, Any]] = []
    path_summary: list[dict[str, Any]] = []
    for parameter, fields in sorted(values.items()):
        observed = torch.stack(
            [
                torch.load(path, map_location="cpu", weights_only=False)[
                    "parameters"
                ][parameter]["weight_before_step"]
                for path in paths
            ]
        ).to(args.device, dtype=torch.float32)
        exact = torch.stack(fields["exact_applied_direction"]).to(
            args.device, dtype=torch.float32
        )
        paths_by_name = {"observed_muon_weights": observed}
        for field in (
            "raw_gradient_descent",
            "combined_momentum_descent",
            "muon_polar_descent",
            "exact_applied_direction",
        ):
            directions = torch.stack(fields[field]).to(
                args.device, dtype=torch.float32
            )
            if field != "exact_applied_direction":
                directions = normalized_like(directions, exact)
            paths_by_name[f"replay_{field}"] = cumulative_path(
                directions,
                steps=steps,
                learning_rates=learning_rates[parameter],
            )
        paths_by_name["replay_adamw"] = adamw_replay_path(
            torch.stack(fields["raw_gradient_descent"]).to(
                args.device, dtype=torch.float32
            ),
            steps=steps,
            learning_rates=learning_rates[parameter],
            beta1=args.beta1,
            beta2=args.beta2,
            epsilon=args.adam_epsilon,
        )
        for path_name, positions in paths_by_name.items():
            analyzed = analyze_parameter(
                positions,
                parameter=parameter,
                basis_rank=args.basis_rank,
                thresholds=thresholds,
            )
            for row in analyzed:
                row["path"] = path_name
            rows.extend(analyzed)
            path_summary.append(
                {
                    "parameter": parameter,
                    "path": path_name,
                    "relative_shape_error_vs_observed": (
                        0.0
                        if path_name == "observed_muon_weights"
                        else relative_path_error(positions, observed)
                    ),
                }
            )
            del positions
            torch.cuda.empty_cache()
        del observed, exact
        torch.cuda.empty_cache()
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / "optimizer_path_rate_distortion.csv"
    summary_path = args.output / "optimizer_path_summary.csv"
    write_csv(result_path, rows)
    write_csv(summary_path, path_summary)
    metadata = {
        "schema_version": "nanogpt_mlp_optimizer_path_rate_distortion_v1",
        "method": "same-gradient-field zero-order-hold counterfactual operator audit",
        "steps": steps,
        "basis_rank": args.basis_rank,
        "thresholds": thresholds,
        "adamw": {
            "beta1": args.beta1,
            "beta2": args.beta2,
            "epsilon": args.adam_epsilon,
            "weight_decay_included": False,
        },
        "input": input_metadata,
        "runtime_seconds": time.time() - started,
        "source_commit": git_commit(Path(__file__).resolve().parents[2]),
        "result_sha256": file_sha256(result_path),
        "summary_sha256": file_sha256(summary_path),
        "limitations": [
            "Counterfactual paths use Muon-trajectory gradients and are local operator audits, not trained AdamW/SGD trajectories.",
            "Missing steps are approximated by zero-order hold over each exact probe interval.",
            "Raw, momentum, and polar paths are Frobenius-norm matched to the exact applied Muon direction at each probe.",
            "The AdamW replay omits weight decay and reuses the stored Muon LR schedule because only path geometry is tested.",
        ],
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"metadata_sha256={file_sha256(metadata_path)}")


if __name__ == "__main__":
    main()
