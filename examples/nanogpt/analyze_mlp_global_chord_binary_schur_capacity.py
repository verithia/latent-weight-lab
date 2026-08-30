#!/usr/bin/env python3
"""H53a continuous ceiling for a global chord plus private Schur factors."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from examples.nanogpt.analyze_mlp_lowbit_global_frame_dct_carrier_capacity import (
    DENSE_REPLACED_MLP_FP16_BYTES,
    DEPLOYED_NODES,
    ROWS,
    WIDTH,
    file_sha256,
    load_node_pc_inventory,
    role_summary,
    tensor_sha256,
    write_json,
)
from examples.nanogpt.analyze_mlp_eight_binary_global_frame_dct_capacity import (
    component_captures,
)


SCHEMA_VERSION = "nanogpt_mlp_global_chord_binary_schur_capacity_v1"
PLAN_SCHEMA_VERSION = "nanogpt_mlp_global_chord_binary_schur_capacity_plan_v1"
RANK = 35
ROBUST_EPSILON = 1e-4


def deployment_accounting(
    *,
    rank: int = RANK,
    width: int = WIDTH,
    rows: int = ROWS,
    deployed_nodes: int = DEPLOYED_NODES,
) -> dict[str, int | float]:
    dense_bytes = deployed_nodes * rows * width * 2
    global_values = rows * width
    if global_values % 8:
        raise ValueError("global chord code must be byte aligned")
    global_bytes = global_values // 8
    private_values = deployed_nodes * 2 * rank * (rows + width)
    if private_values % 8:
        raise ValueError("private factor code must be byte aligned")
    private_bytes = private_values // 8
    rank_amplitude_values = deployed_nodes * 2 * rank
    rank_amplitude_bytes = 2 * rank_amplitude_values
    global_amplitude_values = deployed_nodes
    global_amplitude_bytes = 2 * global_amplitude_values
    total = (
        global_bytes
        + private_bytes
        + rank_amplitude_bytes
        + global_amplitude_bytes
    )
    return {
        "dense_replaced_mlp_fp16_bytes": dense_bytes,
        "global_chord_code_values": global_values,
        "global_chord_code_bytes": global_bytes,
        "private_factor_code_values": private_values,
        "private_factor_code_bytes": private_bytes,
        "fp16_rank_amplitude_values": rank_amplitude_values,
        "fp16_rank_amplitude_bytes": rank_amplitude_bytes,
        "fp16_global_amplitude_values": global_amplitude_values,
        "fp16_global_amplitude_bytes": global_amplitude_bytes,
        "total_checkpoint_bytes": total,
        "checkpoint_byte_fraction": total / dense_bytes,
        "persistent_pca_or_dense_basis_values": 0,
    }


def initialize_relaxation(
    *,
    nodes: int,
    components: int,
    rows: int,
    width: int,
    rank: int,
    seed: int,
    device: torch.device,
) -> tuple[
    torch.nn.Parameter,
    torch.nn.ParameterList,
    torch.nn.ParameterList,
    torch.nn.ParameterList,
    torch.nn.ParameterList,
]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    chord = torch.nn.Parameter(
        torch.randn(rows, width, generator=generator, dtype=torch.float32).to(device)
    )
    left = torch.nn.ParameterList()
    right = torch.nn.ParameterList()
    scales = torch.nn.ParameterList()
    alpha = torch.nn.ParameterList()
    for _ in range(nodes):
        left.append(
            torch.nn.Parameter(
                torch.randn(
                    components,
                    2,
                    rows,
                    rank,
                    generator=generator,
                    dtype=torch.float32,
                ).to(device)
            )
        )
        right.append(
            torch.nn.Parameter(
                torch.randn(
                    components,
                    2,
                    width,
                    rank,
                    generator=generator,
                    dtype=torch.float32,
                ).to(device)
            )
        )
        scales.append(
            torch.nn.Parameter(
                torch.full(
                    (components, 2, rank),
                    1.0 / math.sqrt(rank),
                    device=device,
                    dtype=torch.float32,
                )
            )
        )
        alpha.append(
            torch.nn.Parameter(
                torch.full(
                    (components,), 0.05, device=device, dtype=torch.float32
                )
            )
        )
    return chord, left, right, scales, alpha


def normalized_chord(chord: torch.Tensor) -> torch.Tensor:
    return chord / chord.norm().clamp_min(1e-12)


def generated_node(
    chord: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    scales: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    if left.ndim != 4 or left.shape[1] != 2:
        raise ValueError("left factors must have shape [components,2,rows,rank]")
    if right.ndim != 4 or right.shape[1] != 2:
        raise ValueError("right factors must have shape [components,2,width,rank]")
    left_unit = F.normalize(left, dim=2)
    right_unit = F.normalize(right, dim=2)
    fields = []
    for branch in range(2):
        weighted_left = left_unit[:, branch] * scales[:, branch, None, :]
        fields.append(
            torch.bmm(weighted_left, right_unit[:, branch].transpose(1, 2))
        )
    rows, width = left.shape[2], right.shape[2]
    schur = math.sqrt(rows * width) * fields[0] * fields[1]
    return schur + alpha[:, None, None] * normalized_chord(chord)[None, :, :]


def metric_rows(
    targets: tuple[torch.Tensor, ...],
    weights: tuple[torch.Tensor, ...],
    *,
    chord: torch.Tensor,
    left: torch.nn.ParameterList | list[torch.Tensor],
    right: torch.nn.ParameterList | list[torch.Tensor],
    scales: torch.nn.ParameterList | list[torch.Tensor],
    alpha: torch.nn.ParameterList | list[torch.Tensor],
) -> dict[str, Any]:
    rows = []
    robust_values = []
    with torch.no_grad():
        for node, (target, weight) in enumerate(zip(targets, weights, strict=True)):
            prediction = generated_node(
                chord, left[node], right[node], scales[node], alpha[node]
            )
            capture = component_captures(prediction, target)
            robust_values.append(torch.log(capture + ROBUST_EPSILON).mean())
            rows.append(
                {
                    "index": node,
                    "weighted_top16_capture": float((capture * weight).sum()),
                    "uniform_mean_capture": float(capture.mean()),
                    "geometric_mean_capture": float(
                        torch.exp(torch.log(capture + ROBUST_EPSILON).mean())
                        - ROBUST_EPSILON
                    ),
                    "minimum_pc_capture": float(capture.min()),
                    "median_pc_capture": float(capture.median()),
                    "maximum_pc_capture": float(capture.max()),
                    "component_captures": [float(value) for value in capture],
                }
            )
    return {
        "mean_weighted_capture": sum(
            row["weighted_top16_capture"] for row in rows
        )
        / len(rows),
        "mean_log_capture_plus_epsilon": float(torch.stack(robust_values).mean()),
        "rows": rows,
        "role_summaries": {
            "c_fc": role_summary(rows, (0, 2, 4)),
            "c_proj": role_summary(rows, (1, 3, 5)),
        },
    }


def chord_only_metrics(
    targets: tuple[torch.Tensor, ...],
    weights: tuple[torch.Tensor, ...],
    chord: torch.Tensor,
) -> dict[str, Any]:
    unit = normalized_chord(chord).flatten()
    rows = []
    for node, (target, weight) in enumerate(zip(targets, weights, strict=True)):
        with torch.no_grad():
            capture = (target.flatten(1) @ unit).square().clamp(0.0, 1.0)
        rows.append(
            {
                "index": node,
                "weighted_top16_capture": float((capture * weight).sum()),
                "uniform_mean_capture": float(capture.mean()),
                "minimum_pc_capture": float(capture.min()),
                "median_pc_capture": float(capture.median()),
                "maximum_pc_capture": float(capture.max()),
                "component_captures": [float(value) for value in capture],
            }
        )
    return {
        "mean_weighted_capture": sum(
            row["weighted_top16_capture"] for row in rows
        )
        / len(rows),
        "rows": rows,
        "role_summaries": {
            "c_fc": role_summary(rows, (0, 2, 4)),
            "c_proj": role_summary(rows, (1, 3, 5)),
        },
    }


def fit_relaxation(
    targets: tuple[torch.Tensor, ...],
    weights: tuple[torch.Tensor, ...],
    *,
    rank: int,
    steps: int,
    learning_rate: float,
    gradient_clip_norm: float,
    seed: int,
    progress_callback: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    components, rows, width = targets[0].shape
    chord, left, right, scales, alpha = initialize_relaxation(
        nodes=len(targets),
        components=components,
        rows=rows,
        width=width,
        rank=rank,
        seed=seed,
        device=targets[0].device,
    )
    parameters: list[torch.Tensor] = [chord]
    for values in (left, right, scales, alpha):
        parameters.extend(list(values))
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=0.0)
    history = []
    record_steps = {0, 1, 2, 3, 7, 15, 31, 63, 95, 127, steps - 1}
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        objective_sum = 0.0
        uniform_sum = 0.0
        for node, target in enumerate(targets):
            prediction = generated_node(
                chord, left[node], right[node], scales[node], alpha[node]
            )
            capture = component_captures(prediction, target)
            objective = torch.log(capture + ROBUST_EPSILON).mean()
            (-(objective / len(targets))).backward()
            objective_sum += float(objective.detach()) / len(targets)
            uniform_sum += float(capture.mean().detach()) / len(targets)
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(parameters, gradient_clip_norm)
        )
        optimizer.step()
        with torch.no_grad():
            for value in scales:
                value.clamp_(-4.0, 4.0)
            for value in alpha:
                value.clamp_(-4.0, 4.0)
        if progress_callback is not None:
            progress_callback(step + 1, steps)
        if step in record_steps:
            history.append(
                {
                    "step": step + 1,
                    "mean_log_capture_plus_epsilon": objective_sum,
                    "uniform_mean_capture": uniform_sum,
                    "gradient_norm": gradient_norm,
                }
            )
    result = metric_rows(
        targets,
        weights,
        chord=chord.detach(),
        left=[value.detach() for value in left],
        right=[value.detach() for value in right],
        scales=[value.detach() for value in scales],
        alpha=[value.detach() for value in alpha],
    )
    result["history"] = history
    controls = {
        "global_chord_only": chord_only_metrics(targets, weights, chord.detach()),
        "global_chord_sha256": tensor_sha256(normalized_chord(chord.detach())),
    }
    return result, controls


def self_test(device_name: str = "cpu") -> dict[str, Any]:
    device = torch.device(device_name)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(53)
    components, rows, width, rank = 3, 11, 7, 2
    chord = torch.randn(rows, width, generator=generator).to(device)
    left = torch.randn(components, 2, rows, rank, generator=generator).to(device)
    right = torch.randn(components, 2, width, rank, generator=generator).to(device)
    scales = torch.randn(components, 2, rank, generator=generator).to(device)
    alpha = torch.randn(components, generator=generator).to(device)
    target = generated_node(chord, left, right, scales, alpha)
    capture = component_captures(target, target)
    field0 = torch.bmm(
        F.normalize(left[:, 0], dim=1) * scales[:, 0, None, :],
        F.normalize(right[:, 0], dim=1).transpose(1, 2),
    )[0]
    field1 = torch.bmm(
        F.normalize(left[:, 1], dim=1) * scales[:, 1, None, :],
        F.normalize(right[:, 1], dim=1).transpose(1, 2),
    )[0]
    observed_rank = int(torch.linalg.matrix_rank(field0 * field1))
    accounting = deployment_accounting()
    if float(capture.min()) < 0.999999:
        raise AssertionError({"synthetic_capture": [float(value) for value in capture]})
    if not (rank < observed_rank <= rank * rank):
        raise AssertionError(
            {"observed_schur_rank": observed_rank, "rank_bound": rank * rank}
        )
    if accounting["total_checkpoint_bytes"] != 1_104_720:
        raise AssertionError(accounting)
    if accounting["dense_replaced_mlp_fp16_bytes"] != DENSE_REPLACED_MLP_FP16_BYTES:
        raise AssertionError(accounting)
    return {
        "status": "passed",
        "synthetic_minimum_capture": float(capture.min()),
        "factor_rank": rank,
        "observed_schur_rank": observed_rank,
        "schur_rank_bound": rank * rank,
        "accounting": accounting,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--trajectory-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(args.device), sort_keys=True))
        return

    plan = json.loads(args.plan.read_text())
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unexpected H53a plan schema")
    accounting = deployment_accounting(rank=int(plan["frozen_decoder"]["private_factor_rank"]))
    if accounting != plan["exact_deployment_accounting"]:
        raise ValueError({"computed": accounting, "planned": plan["exact_deployment_accounting"]})
    if accounting["checkpoint_byte_fraction"] > 0.01:
        raise ValueError("H53 exceeds one-percent checkpoint budget")
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device.index or 0)
        torch.backends.cuda.matmul.allow_tf32 = True
    started = time.time()

    fit_plan = plan["continuous_relaxation_ceiling"]
    systems = plan["systems_preflight"]
    components = (
        int(systems["preflight_components_per_node"])
        if args.preflight
        else int(plan["frozen_inventory"]["components_per_node"])
    )
    steps = (
        int(systems["preflight_relaxation_steps"])
        if args.preflight
        else int(fit_plan["steps"])
    )
    targets, weights, inventory, _ = load_node_pc_inventory(
        args.trajectory_dir, components=components, device=args.device
    )
    if inventory["trajectory_identity_sha256"] != plan["frozen_inventory"]["trajectory_identity_sha256"]:
        raise ValueError("H53 trajectory identity mismatch")
    progress_path = args.output / "progress.json"

    def progress(step: int, total: int) -> None:
        write_json(
            progress_path,
            {
                "schema_version": f"{SCHEMA_VERSION}_progress_v1",
                "stage": "continuous_relaxation_fit",
                "stage_step": step,
                "stage_steps": total,
                "completed_updates": step,
                "total_updates": total,
                "fraction": step / total,
            },
        )

    progress(0, steps)
    relaxation, controls = fit_relaxation(
        targets,
        weights,
        rank=int(plan["frozen_decoder"]["private_factor_rank"]),
        steps=steps,
        learning_rate=float(fit_plan["learning_rate"]),
        gradient_clip_norm=float(fit_plan["gradient_clip_norm"]),
        seed=202608530,
        progress_callback=progress,
    )
    gates = plan["capacity_gates"]
    weighted_pass = all(
        row["weighted_top16_capture"]
        >= float(gates["relaxation_weighted_top16_capture_min_every_node"])
        for row in relaxation["rows"]
    )
    minimum_pass = all(
        row["minimum_pc_capture"]
        >= float(gates["relaxation_minimum_pc_capture_every_node"])
        for row in relaxation["rows"]
    )
    finite_pass = all(
        math.isfinite(row["weighted_top16_capture"])
        and math.isfinite(row["minimum_pc_capture"])
        for row in relaxation["rows"]
    )
    relaxation_pass = weighted_pass and minimum_pass and finite_pass
    classification = (
        "PREFLIGHT"
        if args.preflight
        else (
            "RELAXATION_PASSED_BINARY_PENDING"
            if relaxation_pass
            else "RELAXATION_REJECTED"
        )
    )
    gate = {
        "classification": classification,
        "relaxation_pass": relaxation_pass,
        "weighted_capture_every_node_pass": weighted_pass,
        "minimum_pc_every_node_pass": minimum_pass,
        "finite_pass": finite_pass,
        "binary_stage_authorized": (not args.preflight) and relaxation_pass,
    }
    accounting_path = args.output / "accounting.json"
    write_json(accounting_path, accounting)
    metrics = {
        "continuous_relaxation": relaxation,
        "controls": controls,
        "gate": gate,
    }
    metrics_path = args.output / "metrics.json"
    write_json(metrics_path, metrics)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime = time.time() - started
    peak = (
        torch.cuda.max_memory_allocated(device.index or 0)
        if device.type == "cuda"
        else 0
    )
    projected = (
        runtime
        * int(fit_plan["steps"])
        / max(1, steps)
        * int(plan["frozen_inventory"]["components_per_node"])
        / components
        if args.preflight
        else runtime
    )
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "classification": classification,
        "preflight": args.preflight,
        "plan": plan,
        "inventory": inventory,
        "accounting": accounting,
        "metrics": metrics,
        "self_test": self_test(args.device),
        "execution": {
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip(),
            "source_status": subprocess.check_output(
                ["git", "status", "--short"], cwd=REPO_ROOT, text=True
            ).splitlines(),
            "entrypoint": str(script),
            "entrypoint_sha256": file_sha256(script),
            "plan_path": str(args.plan),
            "plan_sha256": file_sha256(args.plan),
            "command": [str(script), *sys.argv[1:]],
            "runtime_seconds": runtime,
            "projected_binding_runtime_seconds": projected,
            "peak_cuda_allocated_bytes": peak,
            "device": args.device,
        },
        "outputs": {
            "accounting": {"path": str(accounting_path), "sha256": file_sha256(accounting_path)},
            "metrics": {"path": str(metrics_path), "sha256": file_sha256(metrics_path)},
            "progress": {"path": str(progress_path), "sha256": file_sha256(progress_path)},
        },
        "limitations": [
            "This all-PC continuous fit is a necessary topology ceiling, not a compact checkpoint.",
            "Per-PC factors are alternative manifold points and are not stored simultaneously.",
            "No binary candidate is fitted unless the frozen continuous gates pass.",
            "No function, CE, attention, or scale result is produced by H53a.",
        ],
    }
    metadata_path = args.output / "metadata.json"
    write_json(metadata_path, metadata)
    print(
        json.dumps(
            {
                "classification": classification,
                "metadata": str(metadata_path),
                "relaxation_mean_weighted_capture": relaxation["mean_weighted_capture"],
                "relaxation_role_summaries": relaxation["role_summaries"],
                "runtime_seconds": runtime,
                "projected_binding_runtime_seconds": projected,
                "peak_cuda_allocated_bytes": peak,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
