#!/usr/bin/env python3
"""Audit a task-selected sparse grouped-Hadamard atlas for dense MLP paths.

The proposed compact chart is

    W = W0 + H_out S H_in^T,

where H_out/H_in are exact grouped normalized Hadamard transforms and only a
fixed fraction of the entries of S may be active.  Current-gradient top-k
support is causally usable before an optimizer update; support selected from
the preceding history tests temporal stability.  Dense-path displacement
support tests whether the same sparse state can remember the path while
remaining tangent to the current task field.  This is an oracle only: it does
not update language-model parameters.
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
from examples.nanogpt.analyze_mlp_optimizer_probe_span import select_parameter
from examples.nanogpt.analyze_mlp_raw_gradient_rolling_prediction import phase_for_step
from examples.nanogpt.analyze_parameter_trajectory import parse_int_list, write_csv
from examples.nanogpt.parameter_trajectory import OPTIMIZER_PROBE_SCHEMA_VERSION
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


def largest_power_of_two_divisor(value: int) -> int:
    if value <= 0:
        raise ValueError("dimension must be positive")
    return value & -value


def grouped_hadamard_2d(
    matrix: torch.Tensor,
    row_group: int | None = None,
    column_group: int | None = None,
) -> torch.Tensor:
    """Apply an involutory separable normalized FHT within exact 2-D panels."""
    if matrix.ndim != 2:
        raise ValueError("grouped_hadamard_2d expects one matrix")
    rows, columns = matrix.shape
    row_group = row_group or largest_power_of_two_divisor(rows)
    column_group = column_group or largest_power_of_two_divisor(columns)
    if rows % row_group or columns % column_group:
        raise ValueError("Hadamard group must divide its matrix dimension")
    if row_group & (row_group - 1) or column_group & (column_group - 1):
        raise ValueError("Hadamard groups must be powers of two")
    panels = matrix.reshape(
        rows // row_group, row_group, columns // column_group, column_group
    )
    panels = normalized_fht_last_dim(panels)
    panels = panels.permute(0, 2, 3, 1)
    panels = normalized_fht_last_dim(panels)
    return panels.permute(0, 3, 1, 2).reshape(rows, columns)


def support_capture(square: torch.Tensor, support: torch.Tensor) -> float:
    total = square.double().sum().clamp_min(1e-30)
    return float(square.reshape(-1)[support].double().sum() / total)


def top_support(square: torch.Tensor, count: int) -> torch.Tensor:
    return torch.topk(square.reshape(-1), k=count, sorted=False).indices


def support_jaccard(first: torch.Tensor, second: torch.Tensor, size: int) -> float:
    first_mask = torch.zeros(size, dtype=torch.bool, device=first.device)
    second_mask = torch.zeros(size, dtype=torch.bool, device=second.device)
    first_mask[first] = True
    second_mask[second] = True
    intersection = torch.logical_and(first_mask, second_mask).sum()
    union = torch.logical_or(first_mask, second_mask).sum().clamp_min(1)
    return float(intersection.double() / union.double())


def balanced_joint_support(
    state_square: torch.Tensor, gradient_square: torch.Tensor, count: int
) -> torch.Tensor:
    state_normalized = state_square / state_square.sum().clamp_min(1e-30)
    gradient_normalized = gradient_square / gradient_square.sum().clamp_min(1e-30)
    return top_support(state_normalized + gradient_normalized, count)


def load_probe_states(
    paths: list[Path], layers: set[int], targets: set[str]
) -> tuple[list[int], dict[str, dict[str, list[torch.Tensor]]], dict[str, Any]]:
    if len(paths) < 3:
        raise ValueError("at least three probes are required")
    steps: list[int] = []
    inventory: dict[str, dict[str, list[torch.Tensor]]] = {}
    run_identity: str | None = None
    provenance: dict[str, Any] | None = None
    files: list[dict[str, Any]] = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != OPTIMIZER_PROBE_SCHEMA_VERSION:
            raise ValueError(f"unexpected probe schema: {path}")
        step = int(payload.get("step", -1))
        if steps and step <= steps[-1]:
            raise ValueError("probe steps must be strictly increasing")
        identity = payload.get("run_identity_sha256")
        if run_identity is None:
            run_identity = identity
            provenance = payload.get("execution_provenance")
        elif identity != run_identity:
            raise ValueError("optimizer probes do not share one run identity")
        selected = {
            name: value
            for name, value in payload.get("parameters", {}).items()
            if select_parameter(name, layers, targets)
        }
        if not selected:
            raise ValueError(f"probe has no selected parameters: {path}")
        if inventory and set(selected) != set(inventory):
            raise ValueError("selected parameter inventory changed")
        for name, value in selected.items():
            destination = inventory.setdefault(name, {"weight": [], "gradient": []})
            weight = value["weight_before_step"].contiguous()
            gradient = value["gradient_after_clip"].contiguous()
            if weight.shape != gradient.shape:
                raise ValueError(f"weight/gradient shape mismatch: {name}")
            destination["weight"].append(weight)
            destination["gradient"].append(gradient)
        steps.append(step)
        files.append({"path": str(path), "bytes": path.stat().st_size})
        del payload
    return steps, inventory, {
        "run_identity_sha256": run_identity,
        "execution_provenance": provenance,
        "files": files,
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        for phase in (str(row["eval_phase"]), "all"):
            grouped.setdefault((str(row["target"]), phase), []).append(row)
    metrics = (
        "instantaneous_gradient_capture",
        "previous_gradient_support_jaccard",
        "history_support_gradient_capture",
        "state_best_capture",
        "state_support_gradient_capture",
        "state_gradient_support_jaccard",
        "joint_support_state_capture",
        "joint_support_gradient_capture",
    )
    result: list[dict[str, Any]] = []
    for (target, phase), members in sorted(grouped.items()):
        item: dict[str, Any] = {
            "target": target,
            "eval_phase": phase,
            "sample_count": len(members),
            "active_scalars": int(members[0]["active_scalars"]),
            "active_scalar_fraction": float(members[0]["active_scalar_fraction"]),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in members if row[metric] is not None]
            if not values:
                continue
            tensor = torch.tensor(values, dtype=torch.float64)
            item[f"{metric}_mean"] = float(tensor.mean())
            item[f"{metric}_median"] = float(tensor.median())
            item[f"{metric}_minimum"] = float(tensor.min())
        result.append(item)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", default="6")
    parser.add_argument("--targets", default="mlp.c_fc,mlp.c_proj")
    parser.add_argument("--scalar-fraction", type=float, default=0.01)
    parser.add_argument("--history-probes", type=int, default=10)
    parser.add_argument("--discovery-stop", type=int, default=119)
    parser.add_argument("--validation-stop", type=int, default=179)
    parser.add_argument("--maximum-probes", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0 < args.scalar_fraction <= 0.01:
        raise ValueError("scalar-fraction must be in (0, 0.01]")
    if args.history_probes < 2:
        raise ValueError("history-probes must be at least two")
    started = time.time()
    paths = sorted(args.probe_dir.glob("step_*.pt"))
    if args.maximum_probes:
        paths = paths[: args.maximum_probes]
    steps, inventory, input_metadata = load_probe_states(
        paths,
        set(parse_int_list(args.layers)),
        {item for item in args.targets.split(",") if item},
    )

    rows: list[dict[str, Any]] = []
    representation: list[dict[str, Any]] = []
    for parameter, fields in sorted(inventory.items()):
        target = "mlp.c_fc" if ".mlp.c_fc." in parameter else "mlp.c_proj"
        initial = fields["weight"][0].to(args.device, dtype=torch.float32)
        matrix_rows, matrix_columns = initial.shape
        row_group = largest_power_of_two_divisor(matrix_rows)
        column_group = largest_power_of_two_divisor(matrix_columns)
        scalar_count = matrix_rows * matrix_columns
        active = max(1, math.floor(args.scalar_fraction * scalar_count))
        coefficient_bits = math.ceil(math.log2(scalar_count))
        ideal_macs = (
            matrix_rows * int(math.log2(row_group))
            + matrix_columns * int(math.log2(column_group))
            + active
        )
        representation.append(
            {
                "parameter": parameter,
                "target": target,
                "shape": [matrix_rows, matrix_columns],
                "row_group": row_group,
                "column_group": column_group,
                "panel_count": (matrix_rows // row_group) * (matrix_columns // column_group),
                "active_scalars": active,
                "active_scalar_fraction": active / scalar_count,
                "coefficient_index_bits": coefficient_bits,
                "packed_fp16_value_plus_index_fraction_of_dense_bf16_bytes": (
                    active * (16 + coefficient_bits) / (scalar_count * 16)
                ),
                "ideal_sparse_decode_macs_or_additions_per_token": ideal_macs,
                "dense_macs_per_token": scalar_count,
                "ideal_arithmetic_reduction": scalar_count / ideal_macs,
            }
        )
        gradient_coefficients = torch.stack(
            [
                grouped_hadamard_2d(value.to(args.device, dtype=torch.float32))
                for value in fields["gradient"]
            ]
        )
        gradient_squares = gradient_coefficients.square()
        previous_instant_support: torch.Tensor | None = None
        for index, step in enumerate(steps):
            gradient_square = gradient_squares[index]
            instant_support = top_support(gradient_square, active)
            instant_capture = support_capture(gradient_square, instant_support)
            previous_jaccard = (
                None
                if previous_instant_support is None
                else support_jaccard(previous_instant_support, instant_support, scalar_count)
            )
            history_capture = None
            if index >= args.history_probes:
                history_energy = gradient_squares[
                    index - args.history_probes : index
                ].sum(dim=0)
                history_support = top_support(history_energy, active)
                history_capture = support_capture(gradient_square, history_support)

            displacement = fields["weight"][index].to(args.device, dtype=torch.float32) - initial
            displacement_square = grouped_hadamard_2d(displacement).square()
            state_total = float(displacement_square.double().sum())
            if state_total > 1e-30:
                state_support = top_support(displacement_square, active)
                state_best_capture = support_capture(displacement_square, state_support)
                state_gradient_capture = support_capture(gradient_square, state_support)
                state_gradient_jaccard = support_jaccard(
                    state_support, instant_support, scalar_count
                )
                joint_support = balanced_joint_support(
                    displacement_square, gradient_square, active
                )
                joint_state_capture = support_capture(displacement_square, joint_support)
                joint_gradient_capture = support_capture(gradient_square, joint_support)
            else:
                state_best_capture = None
                state_gradient_capture = None
                state_gradient_jaccard = None
                joint_state_capture = None
                joint_gradient_capture = None
            rows.append(
                {
                    "parameter": parameter,
                    "target": target,
                    "eval_step": step,
                    "eval_phase": phase_for_step(
                        step, args.discovery_stop, args.validation_stop
                    ),
                    "active_scalars": active,
                    "active_scalar_fraction": active / scalar_count,
                    "gradient_energy": float(gradient_square.double().sum()),
                    "state_displacement_energy": state_total,
                    "instantaneous_gradient_capture": instant_capture,
                    "previous_gradient_support_jaccard": previous_jaccard,
                    "history_support_gradient_capture": history_capture,
                    "state_best_capture": state_best_capture,
                    "state_support_gradient_capture": state_gradient_capture,
                    "state_gradient_support_jaccard": state_gradient_jaccard,
                    "joint_support_state_capture": joint_state_capture,
                    "joint_support_gradient_capture": joint_gradient_capture,
                }
            )
            previous_instant_support = instant_support
            del displacement, displacement_square
        del initial, gradient_coefficients, gradient_squares
        torch.cuda.empty_cache()

    summary = summarize(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    rows_path = args.output / "probe_metrics.csv"
    summary_path = args.output / "summary.csv"
    write_csv(rows_path, rows)
    write_csv(summary_path, summary)
    metadata = {
        "schema_version": "nanogpt_mlp_sparse_hadamard_atlas_v1",
        "method": "grouped separable normalized Hadamard basis with task-selected top-k support",
        "sample_count": len(steps),
        "steps": steps,
        "history_probes": args.history_probes,
        "scalar_fraction_ceiling": args.scalar_fraction,
        "input": input_metadata,
        "representation": representation,
        "summary": summary,
        "promotion_gate": {
            "late_history_support_gradient_capture_minimum": 0.40,
            "state_and_gradient_support_must_coexist": True,
            "no_language_model_run_if_gate_fails": True,
        },
        "runtime_seconds": time.time() - started,
        "source_commit": git_commit(Path(__file__).resolve().parents[2]),
        "rows_sha256": file_sha256(rows_path),
        "summary_sha256": file_sha256(summary_path),
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"metadata_sha256={file_sha256(metadata_path)}")


if __name__ == "__main__":
    main()
