#!/usr/bin/env python3
"""H52a robust eight-plane binary global-frame MLP capacity audit."""

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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from examples.nanogpt.analyze_mlp_lowbit_global_frame_dct_carrier_capacity import (
    BLOCKS,
    DENSE_REPLACED_MLP_FP16_BYTES,
    DEPLOYED_NODES,
    ROWS,
    WIDTH,
    apply_carrier,
    file_sha256,
    load_node_pc_inventory,
    make_carrier_geometry,
    role_summary,
    tensor_sha256,
    write_json,
)


SCHEMA_VERSION = "nanogpt_mlp_eight_binary_global_frame_dct_capacity_v1"
PLAN_SCHEMA_VERSION = "nanogpt_mlp_eight_binary_global_frame_dct_capacity_plan_v1"
FRAMES = 8
ROBUST_EPSILON = 1e-4


def deployment_accounting(
    *,
    width: int = WIDTH,
    rows: int = ROWS,
    deployed_nodes: int = DEPLOYED_NODES,
    frames: int = FRAMES,
) -> dict[str, int | float]:
    code_values = frames * width * width
    if code_values % 8:
        raise ValueError("binary frame count must be byte aligned")
    code_bytes = code_values // 8
    diagonal_values = deployed_nodes * frames * width
    diagonal_bytes = 2 * diagonal_values
    amplitude_values = deployed_nodes * rows
    amplitude_bytes = 2 * amplitude_values
    total = code_bytes + diagonal_bytes + amplitude_bytes
    dense_bytes = deployed_nodes * rows * width * 2
    return {
        "dense_replaced_mlp_fp16_bytes": dense_bytes,
        "binary_frame_count": frames,
        "binary_frame_code_values": code_values,
        "binary_frame_code_bytes": code_bytes,
        "fp16_node_diagonal_values": diagonal_values,
        "fp16_node_diagonal_bytes": diagonal_bytes,
        "fp16_node_row_amplitude_values": amplitude_values,
        "fp16_node_row_amplitude_bytes": amplitude_bytes,
        "total_checkpoint_bytes": total,
        "checkpoint_byte_fraction": total / dense_bytes,
        "continuous_coordinate_values": diagonal_values + amplitude_values,
        "continuous_coordinate_fraction": (diagonal_values + amplitude_values)
        / (dense_bytes // 2),
        "persistent_pca_or_carrier_values": 0,
    }


def pack_binary(codes: torch.Tensor) -> torch.Tensor:
    flat = codes.detach().cpu().to(torch.int16).flatten()
    if not bool(((flat == -1) | (flat == 1)).all()):
        raise ValueError("binary codes must lie in {-1,+1}")
    encoded = (flat > 0).to(torch.uint8)
    padding = (-encoded.numel()) % 8
    if padding:
        encoded = torch.cat((encoded, torch.zeros(padding, dtype=torch.uint8)))
    encoded = encoded.reshape(-1, 8).to(torch.int16)
    packed = sum(encoded[:, bit] << bit for bit in range(8))
    return packed.to(torch.uint8).contiguous()


def unpack_binary(packed: torch.Tensor, values: int) -> torch.Tensor:
    packed = packed.detach().cpu().to(torch.int16).flatten()
    decoded = torch.stack(
        [((packed >> bit) & 1) for bit in range(8)], dim=1
    ).flatten()[:values]
    return torch.where(decoded > 0, 1, -1).to(torch.int8)


def initial_frame_logits(
    *, frames: int, width: int, seed: int, device: torch.device
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    signs = (
        2
        * torch.randint(
            0, 2, (frames, width, width), generator=generator, dtype=torch.int64
        )
        - 1
    ).float()
    return (0.05 * signs).to(device)


def binary_frames(logits: torch.Tensor, *, straight_through: bool) -> torch.Tensor:
    decoded = torch.where(logits >= 0, 1.0, -1.0) / math.sqrt(logits.shape[-1])
    return logits + (decoded - logits).detach() if straight_through else decoded


def initialize_coordinates(
    *,
    nodes: int,
    components: int,
    rows: int,
    width: int,
    frames: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    amplitude = torch.nn.Parameter(
        torch.ones(nodes, components, rows, device=device, dtype=torch.float32)
    )
    diagonal = torch.nn.Parameter(
        (
            torch.randn(
                nodes,
                components,
                frames,
                width,
                generator=generator,
                dtype=torch.float32,
            )
            / math.sqrt(width)
        ).to(device)
    )
    return amplitude, diagonal


def generated_node(
    frames: torch.Tensor,
    amplitude: torch.Tensor,
    diagonal: torch.Tensor,
    geometry: tuple[tuple[dict[str, torch.Tensor], ...], ...],
) -> torch.Tensor:
    result: torch.Tensor | None = None
    for branch in range(frames.shape[0]):
        branch_input = diagonal[:, branch, :, None] * frames[branch][None, :, :]
        branch_output = apply_carrier(branch_input, geometry[branch])
        result = branch_output if result is None else result + branch_output
    assert result is not None
    return amplitude[:, :, None] * result


def component_captures(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    prediction_flat = prediction.flatten(1)
    target_flat = target.flatten(1)
    numerator = (prediction_flat * target_flat).sum(dim=1).square()
    denominator = prediction_flat.square().sum(dim=1).clamp_min(1e-30)
    target_denominator = target_flat.square().sum(dim=1).clamp_min(1e-30)
    return (numerator / (denominator * target_denominator)).clamp(0.0, 1.0)


def metrics_from_coordinates(
    targets: tuple[torch.Tensor, ...],
    weights: tuple[torch.Tensor, ...],
    *,
    frames: torch.Tensor,
    amplitude: torch.Tensor,
    diagonal: torch.Tensor,
    geometry: tuple[tuple[tuple[dict[str, torch.Tensor], ...], ...], ...],
) -> dict[str, Any]:
    rows = []
    robust_values = []
    with torch.no_grad():
        for node, (target, weight) in enumerate(zip(targets, weights, strict=True)):
            prediction = generated_node(
                frames, amplitude[node], diagonal[node], geometry[node]
            )
            capture = component_captures(prediction, target)
            weighted = (capture * weight).sum()
            robust_values.append(torch.log(capture + ROBUST_EPSILON).mean())
            rows.append(
                {
                    "index": node,
                    "weighted_top16_capture": float(weighted),
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
    mean_weighted = sum(row["weighted_top16_capture"] for row in rows) / len(rows)
    return {
        "mean_weighted_capture": mean_weighted,
        "mean_log_capture_plus_epsilon": float(torch.stack(robust_values).mean()),
        "rows": rows,
        "role_summaries": {
            "c_fc": role_summary(rows, (0, 2, 4)),
            "c_proj": role_summary(rows, (1, 3, 5)),
        },
    }


def fit_capacity(
    targets: tuple[torch.Tensor, ...],
    weights: tuple[torch.Tensor, ...],
    *,
    geometry: tuple[tuple[tuple[dict[str, torch.Tensor], ...], ...], ...],
    logits: torch.Tensor,
    steps: int,
    learning_rate: float,
    seed: int,
    learn_frames: bool,
    frame_logit_clip: float,
    gradient_clip_norm: float,
    progress_callback: Any | None = None,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    nodes = len(targets)
    components, rows, width = targets[0].shape
    amplitude, diagonal = initialize_coordinates(
        nodes=nodes,
        components=components,
        rows=rows,
        width=width,
        frames=logits.shape[0],
        seed=seed,
        device=targets[0].device,
    )
    if learn_frames:
        logits_value = torch.nn.Parameter(logits.detach().clone())
        parameters: list[torch.Tensor] = [logits_value, amplitude, diagonal]
    else:
        logits_value = logits.detach()
        parameters = [amplitude, diagonal]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=0.0)
    history = []
    record_steps = {0, 1, 2, 3, 7, 15, 31, 63, 95, steps - 1}
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        objective_sum = 0.0
        mean_capture_sum = 0.0
        for node, target in enumerate(targets):
            frames = binary_frames(logits_value, straight_through=learn_frames)
            prediction = generated_node(
                frames, amplitude[node], diagonal[node], geometry[node]
            )
            capture = component_captures(prediction, target)
            objective = torch.log(capture + ROBUST_EPSILON).mean()
            (-(objective / nodes)).backward()
            objective_sum += float(objective.detach()) / nodes
            mean_capture_sum += float(capture.mean().detach()) / nodes
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(parameters, gradient_clip_norm)
        )
        optimizer.step()
        if learn_frames:
            with torch.no_grad():
                logits_value.clamp_(-frame_logit_clip, frame_logit_clip)
        if progress_callback is not None:
            progress_callback(step + 1, steps)
        if step in record_steps:
            history.append(
                {
                    "step": step + 1,
                    "mean_log_capture_plus_epsilon": objective_sum,
                    "uniform_mean_capture": mean_capture_sum,
                    "gradient_norm": gradient_norm,
                    "frame_sign_positive_fraction": float(
                        (logits_value.detach() >= 0).float().mean()
                    ),
                }
            )
    frames = binary_frames(logits_value, straight_through=False).detach()
    metrics = metrics_from_coordinates(
        targets,
        weights,
        frames=frames,
        amplitude=amplitude.detach(),
        diagonal=diagonal.detach(),
        geometry=geometry,
    )
    metrics["history"] = history
    return metrics, logits_value.detach(), amplitude.detach(), diagonal.detach()


def refine_coordinates(
    targets: tuple[torch.Tensor, ...],
    weights: tuple[torch.Tensor, ...],
    *,
    geometry: tuple[tuple[tuple[dict[str, torch.Tensor], ...], ...], ...],
    frames: torch.Tensor,
    amplitude: torch.Tensor,
    diagonal: torch.Tensor,
    steps: int,
    learning_rate: float,
    gradient_clip_norm: float,
    progress_callback: Any | None = None,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    amplitude_value = torch.nn.Parameter(amplitude.detach().clone())
    diagonal_value = torch.nn.Parameter(diagonal.detach().clone())
    parameters = [amplitude_value, diagonal_value]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=0.0)
    history = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        objective_sum = 0.0
        mean_capture_sum = 0.0
        for node, target in enumerate(targets):
            prediction = generated_node(
                frames,
                amplitude_value[node],
                diagonal_value[node],
                geometry[node],
            )
            capture = component_captures(prediction, target)
            objective = torch.log(capture + ROBUST_EPSILON).mean()
            (-(objective / len(targets))).backward()
            objective_sum += float(objective.detach()) / len(targets)
            mean_capture_sum += float(capture.mean().detach()) / len(targets)
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(parameters, gradient_clip_norm)
        )
        optimizer.step()
        if progress_callback is not None:
            progress_callback(step + 1, steps)
        if step in {0, 1, 3, 7, 15, 31, steps - 1}:
            history.append(
                {
                    "step": step + 1,
                    "mean_log_capture_plus_epsilon": objective_sum,
                    "uniform_mean_capture": mean_capture_sum,
                    "gradient_norm": gradient_norm,
                }
            )
    metrics = metrics_from_coordinates(
        targets,
        weights,
        frames=frames,
        amplitude=amplitude_value.detach(),
        diagonal=diagonal_value.detach(),
        geometry=geometry,
    )
    metrics["history"] = history
    return metrics, amplitude_value.detach(), diagonal_value.detach()


def checkpoint_payload(
    frame_codes: torch.Tensor,
    *,
    width: int = WIDTH,
    rows: int = ROWS,
    deployed_nodes: int = DEPLOYED_NODES,
) -> bytes:
    coordinates = torch.zeros(
        deployed_nodes * (rows + frame_codes.shape[0] * width),
        dtype=torch.float16,
    )
    payload = b"".join(
        (pack_binary(frame_codes).numpy().tobytes(), coordinates.numpy().tobytes())
    )
    expected = deployment_accounting(
        width=width,
        rows=rows,
        deployed_nodes=deployed_nodes,
        frames=frame_codes.shape[0],
    )["total_checkpoint_bytes"]
    if len(payload) != expected:
        raise ValueError(f"checkpoint has {len(payload)} bytes, expected {expected}")
    return payload


def self_test(device_name: str = "cpu") -> dict[str, Any]:
    device = torch.device(device_name)
    logits = initial_frame_logits(frames=3, width=8, seed=29, device=device)
    logits.requires_grad_(True)
    decoded = binary_frames(logits, straight_through=True)
    decoded.sum().backward()
    codes = torch.where(decoded.detach().cpu() >= 0, 1, -1).to(torch.int8)
    restored = unpack_binary(pack_binary(codes), codes.numel()).reshape(codes.shape)
    accounting = deployment_accounting()
    forward_values = sorted(set(float(value) for value in decoded.detach().cpu().unique()))
    expected_values = [-1.0 / math.sqrt(8), 1.0 / math.sqrt(8)]
    if not torch.equal(codes, restored):
        raise AssertionError("binary packing roundtrip failed")
    if max(abs(a - b) for a, b in zip(forward_values, expected_values, strict=True)) > 1e-6:
        raise AssertionError({"forward_values": forward_values})
    if not torch.equal(logits.grad, torch.ones_like(logits.grad)):
        raise AssertionError("straight-through gradient is not identity")
    if accounting["total_checkpoint_bytes"] != 1_032_192:
        raise AssertionError(accounting)
    if accounting["dense_replaced_mlp_fp16_bytes"] != DENSE_REPLACED_MLP_FP16_BYTES:
        raise AssertionError(accounting)
    return {
        "status": "passed",
        "forward_values": forward_values,
        "straight_through_gradient_sha256": tensor_sha256(logits.grad),
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
        raise ValueError("unexpected H52a plan schema")
    accounting = deployment_accounting()
    if accounting != plan["exact_deployment_accounting"]:
        raise ValueError({"computed": accounting, "planned": plan["exact_deployment_accounting"]})
    if accounting["checkpoint_byte_fraction"] > 0.01:
        raise ValueError("H52 exceeds one-percent checkpoint budget")
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device.index or 0)
        torch.backends.cuda.matmul.allow_tf32 = True
    started = time.time()

    fit_plan = plan["optimistic_capacity_fit"]
    systems = plan["systems_preflight"]
    components = (
        int(systems["preflight_components_per_node"])
        if args.preflight
        else int(plan["frozen_inventory"]["components_per_node"])
    )
    joint_steps = (
        int(systems["preflight_joint_steps"])
        if args.preflight
        else int(fit_plan["joint_ste_steps"])
    )
    local_steps = (
        int(systems["preflight_local_steps"])
        if args.preflight
        else int(fit_plan["post_sign_local_only_steps"])
    )
    targets, weights, inventory, _ = load_node_pc_inventory(
        args.trajectory_dir, components=components, device=args.device
    )
    if inventory["trajectory_identity_sha256"] != plan["frozen_inventory"]["trajectory_identity_sha256"]:
        raise ValueError("H52 trajectory identity mismatch")
    decoder = plan["frozen_decoder"]
    geometry = make_carrier_geometry(
        nodes=len(targets),
        width=WIDTH,
        branches=FRAMES,
        blocks=BLOCKS,
        seed_base=int(decoder["carrier_seed_base"]),
        node_stride=int(decoder["carrier_node_stride"]),
        branch_stride=int(decoder["carrier_branch_stride"]),
        device=device,
    )
    total_updates = 2 * (joint_steps + local_steps)
    progress_path = args.output / "progress.json"

    def make_progress(stage: str, completed_before: int) -> Any:
        def record(step: int, stage_steps: int) -> None:
            write_json(
                progress_path,
                {
                    "schema_version": f"{SCHEMA_VERSION}_progress_v1",
                    "stage": stage,
                    "stage_step": step,
                    "stage_steps": stage_steps,
                    "completed_updates": completed_before + step,
                    "total_updates": total_updates,
                    "fraction": (completed_before + step) / total_updates,
                },
            )

        return record

    write_json(
        progress_path,
        {
            "schema_version": f"{SCHEMA_VERSION}_progress_v1",
            "stage": "inventory_loaded",
            "stage_step": 0,
            "stage_steps": joint_steps,
            "completed_updates": 0,
            "total_updates": total_updates,
            "fraction": 0.0,
        },
    )
    initial_logits = initial_frame_logits(
        frames=FRAMES, width=WIDTH, seed=202608521, device=device
    )
    joint, learned_logits, amplitude, diagonal = fit_capacity(
        targets,
        weights,
        geometry=geometry,
        logits=initial_logits,
        steps=joint_steps,
        learning_rate=float(fit_plan["joint_learning_rate"]),
        seed=202608522,
        learn_frames=True,
        frame_logit_clip=float(fit_plan["frame_logit_clip"]),
        gradient_clip_norm=float(fit_plan["gradient_clip_norm"]),
        progress_callback=make_progress("joint_binary_frame_fit", 0),
    )
    learned_frames = binary_frames(learned_logits, straight_through=False)
    candidate, candidate_amplitude, candidate_diagonal = refine_coordinates(
        targets,
        weights,
        geometry=geometry,
        frames=learned_frames,
        amplitude=amplitude,
        diagonal=diagonal,
        steps=local_steps,
        learning_rate=float(fit_plan["post_sign_learning_rate"]),
        gradient_clip_norm=float(fit_plan["gradient_clip_norm"]),
        progress_callback=make_progress("signed_local_refine", joint_steps),
    )
    two_frame = metrics_from_coordinates(
        targets,
        weights,
        frames=learned_frames[:2],
        amplitude=candidate_amplitude,
        diagonal=candidate_diagonal[:, :, :2],
        geometry=tuple(tuple(node[:2]) for node in geometry),
    )
    procedural, _, _, _ = fit_capacity(
        targets,
        weights,
        geometry=geometry,
        logits=initial_logits,
        steps=joint_steps + local_steps,
        learning_rate=float(fit_plan["joint_learning_rate"]),
        seed=202608523,
        learn_frames=False,
        frame_logit_clip=float(fit_plan["frame_logit_clip"]),
        gradient_clip_norm=float(fit_plan["gradient_clip_norm"]),
        progress_callback=make_progress(
            "procedural_control_fit", joint_steps + local_steps
        ),
    )
    margins = [
        candidate["rows"][index]["weighted_top16_capture"]
        - procedural["rows"][index]["weighted_top16_capture"]
        for index in range(len(targets))
    ]
    gates = plan["capacity_gates"]
    weighted_pass = all(
        row["weighted_top16_capture"]
        >= float(gates["binary_weighted_top16_capture_min_every_node"])
        for row in candidate["rows"]
    )
    role_pass = all(
        summary["median_weighted_top16_capture"]
        >= float(gates["binary_weighted_top16_capture_median_each_role"])
        for summary in candidate["role_summaries"].values()
    )
    minimum_pass = all(
        row["minimum_pc_capture"]
        >= float(gates["binary_minimum_pc_capture_every_node"])
        for row in candidate["rows"]
    )
    margin_pass = all(
        margin
        >= float(gates["binary_minus_procedural_weighted_capture_min_every_node"])
        for margin in margins
    )
    finite = all(
        math.isfinite(row["weighted_top16_capture"])
        and math.isfinite(row["minimum_pc_capture"])
        for row in candidate["rows"]
    )
    retained = (
        (not args.preflight)
        and weighted_pass
        and role_pass
        and minimum_pass
        and margin_pass
        and finite
    )
    gate = {
        "classification": "PREFLIGHT" if args.preflight else ("RETAINED" if retained else "REJECTED"),
        "retained": retained,
        "weighted_capture_every_node_pass": weighted_pass,
        "role_median_pass": role_pass,
        "minimum_pc_pass": minimum_pass,
        "procedural_margin_pass": margin_pass,
        "finite_pass": finite,
        "per_node_procedural_margins": margins,
    }

    codes = torch.where(learned_frames.detach().cpu() >= 0, 1, -1).to(torch.int8)
    checkpoint_path = args.output / "compact_checkpoint.bin"
    checkpoint_path.write_bytes(checkpoint_payload(codes))
    accounting_path = args.output / "accounting.json"
    write_json(accounting_path, accounting)
    metrics = {
        "joint_signed_fit": joint,
        "signed_local_refined": candidate,
        "first_two_learned_frames": two_frame,
        "procedural_unfitted_frames": procedural,
        "gate": gate,
        "frame_sign_change_fraction": float(
            ((learned_logits >= 0) != (initial_logits >= 0)).float().mean()
        ),
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
        * int(fit_plan["joint_ste_steps"] + fit_plan["post_sign_local_only_steps"])
        / max(1, joint_steps + local_steps)
        * int(plan["frozen_inventory"]["components_per_node"])
        / components
        if args.preflight
        else runtime
    )
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "classification": gate["classification"],
        "preflight": args.preflight,
        "plan": plan,
        "inventory": inventory,
        "accounting": accounting,
        "metrics": metrics,
        "self_test": self_test(args.device),
        "frame_code_sha256": tensor_sha256(codes),
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
            "compact_checkpoint": {"path": str(checkpoint_path), "sha256": file_sha256(checkpoint_path)},
            "progress": {"path": str(progress_path), "sha256": file_sha256(progress_path)},
        },
        "limitations": [
            "This optimistic all-PC fit is a capacity ceiling, not chronological transfer.",
            "Per-PC nuisance coordinates are alternate manifold points and are not stored together.",
            "FP32 frame logits are offline acquisition state and are absent from the artifact.",
            "No function, CE, or scale result is produced by H52a.",
        ],
    }
    metadata_path = args.output / "metadata.json"
    write_json(metadata_path, metadata)
    print(
        json.dumps(
            {
                "classification": gate["classification"],
                "metadata": str(metadata_path),
                "candidate_mean_weighted_capture": candidate["mean_weighted_capture"],
                "candidate_role_summaries": candidate["role_summaries"],
                "minimum_procedural_margin": min(margins),
                "runtime_seconds": runtime,
                "projected_binding_runtime_seconds": projected,
                "peak_cuda_allocated_bytes": peak,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
