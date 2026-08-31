#!/usr/bin/env python3
"""Frozen H59 equal-budget parallel residual-MLP functional audit."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from examples.nanogpt.analyze_mlp_activation_update_alignment import (
    file_sha256,
    load_snapshot,
    model_from_snapshot,
)
from examples.nanogpt.analyze_mlp_lowbit_complete_neuron_functional import (
    FunctionalBank,
    canonical_sha256,
    collect_functional_bank,
    evaluate_function,
    extract_teacher_atoms,
    gelu_derivative,
    git_commit,
    rademacher_direction,
    summarize,
    teacher_jvp,
    tensor_sha256,
    write_json,
)
from examples.nanogpt.analyze_mlp_token_router_residual_carrier_functional import (
    dense_teacher_function,
)


SCHEMA_VERSION = "nanogpt_mlp_parallel_residual_mlp_functional_v1"
PLAN_SCHEMA_VERSION = "nanogpt_mlp_parallel_residual_mlp_functional_plan_v1"
CANDIDATE_NAME = "learned_parallel_residual_mlp"


def deployment_accounting(
    *, layers: int = 12, width: int = 768, residual_width: int = 288
) -> dict[str, int | float]:
    shared_values = 2 * width * residual_width
    private_values = layers * 2 * residual_width
    total_values = shared_values + private_values
    dense_values = layers * 2 * 4 * width * width
    return {
        "dense_replaced_mlp_fp16_values": dense_values,
        "dense_replaced_mlp_fp16_bytes": 2 * dense_values,
        "fp16_shared_residual_values": shared_values,
        "fp16_shared_residual_bytes": 2 * shared_values,
        "fp16_private_coordinate_values": private_values,
        "fp16_private_coordinate_bytes": 2 * private_values,
        "total_latent_values": total_values,
        "total_checkpoint_payload_bytes": 2 * total_values,
        "latent_value_fraction": total_values / dense_values,
        "checkpoint_byte_fraction": total_values / dense_values,
        "extra_mlp_matmul_fraction": 2 * width * residual_width / (8 * width * width),
        "cached_procedural_endpoint_bytes": 2 * dense_values,
    }


def deterministic_detector(width: int, residual_width: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return (
        torch.randn(residual_width, width, generator=generator)
        / math.sqrt(width)
    ).contiguous()


class ParallelResidualMLP(torch.nn.Module):
    def __init__(
        self,
        base_detector: dict[int, torch.Tensor],
        base_write: dict[int, torch.Tensor],
        layers: list[int],
        *,
        residual_width: int,
        detector_seed: int,
        gain_initial: float,
        bias_initial: float,
    ) -> None:
        super().__init__()
        self.layers = list(layers)
        self.layer_to_row = {layer: row for row, layer in enumerate(layers)}
        first = base_detector[layers[0]]
        _hidden, width = first.shape
        self.width = int(width)
        self.residual_width = int(residual_width)
        stacked_detector = torch.stack(
            [base_detector[layer].detach().float() for layer in layers]
        )
        stacked_write = torch.stack(
            [base_write[layer].detach().float() for layer in layers]
        )
        if stacked_detector.shape != stacked_write.shape:
            raise ValueError("procedural endpoint shapes disagree")
        self.register_buffer("base_detector", stacked_detector)
        self.register_buffer("base_write", stacked_write)
        initial_detector = deterministic_detector(width, residual_width, detector_seed)
        self.register_buffer("initial_detector", initial_detector.clone())
        self.detector = torch.nn.Parameter(initial_detector)
        self.write = torch.nn.Parameter(torch.zeros(width, residual_width))
        self.gain = torch.nn.Parameter(
            torch.full((len(layers), residual_width), float(gain_initial))
        )
        self.bias = torch.nn.Parameter(
            torch.full((len(layers), residual_width), float(bias_initial))
        )

    def _coordinates(
        self, row: int, mode: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        detector = self.detector
        write = self.write
        gain = self.gain[row]
        bias = self.bias[row]
        linear = False
        if mode == "step_zero_parent":
            write = torch.zeros_like(write)
        elif mode == "global_coordinates":
            gain = self.gain.mean(dim=0)
            bias = self.bias.mean(dim=0)
        elif mode == "initial_random_detector":
            detector = self.initial_detector
        elif mode == "linear_residual":
            linear = True
        elif mode != "full":
            raise ValueError(f"unknown H59 mode: {mode}")
        return detector, write, gain, bias, linear

    def forward_function(
        self,
        layer: int,
        inputs: torch.Tensor,
        directions: torch.Tensor | None,
        *,
        mode: str = "full",
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        row = self.layer_to_row[layer]
        base_detector = self.base_detector[row]
        base_write = self.base_write[row]
        detector, write, gain, bias, linear = self._coordinates(row, mode)
        base_hidden = F.gelu(inputs @ base_detector.T)
        base_output = base_hidden @ base_write
        pre = inputs @ detector.T + bias
        residual_hidden = pre if linear else F.gelu(pre)
        output = base_output + (residual_hidden * gain) @ write.T
        if directions is None:
            return output, None
        base_action = teacher_jvp(inputs, directions, base_detector, base_write)
        pre_action = directions @ detector.T
        residual_action = pre_action if linear else gelu_derivative(pre) * pre_action
        action = base_action + (residual_action * gain) @ write.T
        return output, action


def optimizer_for(
    student: ParallelResidualMLP, plan: dict[str, Any]
) -> torch.optim.Optimizer:
    fit = plan["fit"]
    return torch.optim.AdamW(
        [
            {
                "params": [student.detector, student.write],
                "lr": float(fit["learning_rate_shared_factors"]),
                "weight_decay": float(fit["weight_decay_shared"]),
            },
            {
                "params": [student.gain, student.bias],
                "lr": float(fit["learning_rate_private_coordinates"]),
                "weight_decay": float(fit["weight_decay_private"]),
            },
        ]
    )


def fit_student(
    student: ParallelResidualMLP,
    bank: FunctionalBank,
    teacher_detector: dict[int, torch.Tensor],
    teacher_write: dict[int, torch.Tensor],
    plan: dict[str, Any],
    *,
    layers: list[int],
    iterations: int,
    minibatch_rows: int,
    jvp_seed: int,
    device: str,
) -> list[dict[str, float | int]]:
    student.to(device)
    optimizer = optimizer_for(student, plan)
    gradient_clip_norm = float(plan["fit"]["gradient_clip_norm"])
    history: list[dict[str, float | int]] = []
    for iteration in range(iterations):
        layer = layers[iteration % len(layers)]
        rows = bank.inputs[layer].shape[0]
        cycle = iteration // len(layers)
        start = (cycle * minibatch_rows) % rows
        index = (torch.arange(minibatch_rows) + start) % rows
        inputs = bank.inputs[layer][index].to(device=device, dtype=torch.float32)
        target = bank.outputs[layer][index].to(device=device, dtype=torch.float32)
        direction_index = cycle % 4
        direction = rademacher_direction(
            tuple(inputs.shape),
            jvp_seed
            + 1009 * layer
            + 100_003 * direction_index
            + 1_000_003 * start,
            device,
        )
        with torch.no_grad():
            target_action = teacher_jvp(
                inputs,
                direction,
                teacher_detector[layer].to(device=device, dtype=torch.float32),
                teacher_write[layer].to(device=device, dtype=torch.float32),
            )
        prediction, action = student.forward_function(layer, inputs, direction)
        if action is None:
            raise RuntimeError("H59 student JVP is missing")
        output_loss = (prediction - target).square().mean() / target.square().mean().clamp_min(1e-30)
        jvp_loss = (action - target_action).square().mean() / target_action.square().mean().clamp_min(1e-30)
        loss = output_loss + 0.25 * jvp_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            student.parameters(), gradient_clip_norm
        )
        optimizer.step()
        if iteration in {
            0,
            iterations // 4,
            iterations // 2,
            3 * iterations // 4,
            iterations - 1,
        }:
            record = {
                "iteration": iteration + 1,
                "layer": layer,
                "loss": float(loss.detach()),
                "relative_output_mse": float(output_loss.detach()),
                "relative_jvp_mse": float(jvp_loss.detach()),
                "gradient_norm": float(gradient_norm.detach()),
            }
            history.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
    return history


def terminal_artifact(
    student: ParallelResidualMLP, accounting: dict[str, Any]
) -> dict[str, Any]:
    values = {
        "detector": student.detector.detach().to(torch.float16).cpu(),
        "write": student.write.detach().to(torch.float16).cpu(),
        "gain": student.gain.detach().to(torch.float16).cpu(),
        "bias": student.bias.detach().to(torch.float16).cpu(),
    }
    payload_bytes = 2 * sum(value.numel() for value in values.values())
    if payload_bytes != accounting["total_checkpoint_payload_bytes"]:
        raise AssertionError((payload_bytes, accounting))
    return {
        "schema_version": "parallel_residual_mlp_checkpoint_v1",
        **values,
        "layers": student.layers,
        "accounted_payload_bytes": payload_bytes,
        "base_detector_sha256": tensor_sha256(student.base_detector),
        "base_write_sha256": tensor_sha256(student.base_write),
    }


def artifact_student(
    artifact: dict[str, Any],
    base_detector: dict[int, torch.Tensor],
    base_write: dict[int, torch.Tensor],
    plan: dict[str, Any],
    device: str,
) -> ParallelResidualMLP:
    frozen = plan["frozen_representation"]
    initialization = plan["initialization"]
    student = ParallelResidualMLP(
        base_detector,
        base_write,
        [int(value) for value in artifact["layers"]],
        residual_width=int(frozen["residual_width"]),
        detector_seed=int(initialization["detector_seed"]),
        gain_initial=float(initialization["gain"]),
        bias_initial=float(initialization["bias"]),
    )
    with torch.no_grad():
        for name in ("detector", "write", "gain", "bias"):
            getattr(student, name).copy_(artifact[name].float())
    student.to(device)
    student.eval()
    return student


def student_function(
    student: ParallelResidualMLP, mode: str
) -> Callable[
    [int, torch.Tensor, torch.Tensor | None],
    tuple[torch.Tensor, torch.Tensor | None],
]:
    def function(
        layer: int, inputs: torch.Tensor, directions: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return student.forward_function(layer, inputs, directions, mode=mode)

    return function


def gate_outcome(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["candidate"] == CANDIDATE_NAME and row["split"] == "holdout"
    ]
    layer_gates = [
        {
            "layer": row["layer"],
            "output_pass": row["relative_output_rmse"] <= 0.10,
            "jvp_pass": row["relative_jvp_rmse"] <= 0.15,
            "covariance_pass": row[
                "retained_centered_output_covariance_energy"
            ]
            >= 0.90,
            "finite_pass": bool(row["finite"]),
        }
        for row in selected
    ]
    representation_pass = len(layer_gates) == 12 and all(
        all(value for key, value in row.items() if key != "layer")
        for row in layer_gates
    )
    return {
        "layer_gates": layer_gates,
        "representation_pass": representation_pass,
        "decision": (
            "PROMOTE_H59_TO_EXACT_SYSTEMS_GATE"
            if representation_pass
            else "REJECT_H59_PARALLEL_RESIDUAL_MLP"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-cap", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=3072)
    parser.add_argument("--minibatch-rows", type=int, default=128)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    started = time.time()
    plan = json.loads(args.plan.read_text())
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unexpected H59 plan schema")
    if args.preflight:
        args.sample_cap = min(args.sample_cap, 128)
        args.iterations = min(args.iterations, 24)
    else:
        fit = plan["fit"]
        inventory = plan["frozen_function_inventory"]
        if args.sample_cap != int(inventory["samples_per_layer_per_split"]):
            raise ValueError("binding sample cap differs from frozen plan")
        if args.iterations != int(fit["iterations"]):
            raise ValueError("binding iterations differ from frozen plan")
        if args.minibatch_rows != int(fit["minibatch_rows"]):
            raise ValueError("binding minibatch differs from frozen plan")

    teacher_plan = plan["frozen_teacher"]
    layers = [int(value) for value in teacher_plan["layers"]]
    width = int(teacher_plan["model_width"])
    residual_width = int(plan["frozen_representation"]["residual_width"])
    accounting = deployment_accounting(
        layers=len(layers), width=width, residual_width=residual_width
    )
    expected = plan["exact_deployment_accounting"]
    for key in (
        "dense_replaced_mlp_fp16_values",
        "dense_replaced_mlp_fp16_bytes",
        "fp16_shared_residual_values",
        "fp16_shared_residual_bytes",
        "fp16_private_coordinate_values",
        "fp16_private_coordinate_bytes",
        "total_latent_values",
        "total_checkpoint_bytes",
        "cached_procedural_endpoint_bytes",
    ):
        actual_key = (
            "total_checkpoint_payload_bytes"
            if key == "total_checkpoint_bytes"
            else key
        )
        if accounting[actual_key] != expected[key]:
            raise AssertionError((key, accounting[actual_key], expected[key]))

    initial_snapshot = args.snapshot_dir / teacher_plan["initial_snapshot"]
    terminal_snapshot = args.snapshot_dir / teacher_plan["terminal_snapshot"]
    if file_sha256(initial_snapshot) != teacher_plan["initial_snapshot_sha256"]:
        raise ValueError("step-zero snapshot identity mismatch")
    if file_sha256(terminal_snapshot) != teacher_plan["terminal_snapshot_sha256"]:
        raise ValueError("terminal snapshot identity mismatch")
    if file_sha256(args.data_dir / "manifest.json") != plan[
        "frozen_function_inventory"
    ]["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest identity mismatch")

    terminal_payload = load_snapshot(terminal_snapshot)
    terminal_model = model_from_snapshot(terminal_payload, args.device)
    terminal_model.eval()
    teacher_detector, teacher_write = extract_teacher_atoms(terminal_model, layers)
    inventory = plan["frozen_function_inventory"]
    banks = {
        "train": collect_functional_bank(
            terminal_model,
            args.data_dir,
            layers,
            args.sample_cap,
            int(inventory["train_seed"]),
            2,
            int(terminal_model.config.block_size),
            args.device,
        ),
        "holdout": collect_functional_bank(
            terminal_model,
            args.data_dir,
            layers,
            args.sample_cap,
            int(inventory["holdout_seed"]),
            2,
            int(terminal_model.config.block_size),
            args.device,
        ),
    }
    del terminal_model
    initial_payload = load_snapshot(initial_snapshot)
    initial_model = model_from_snapshot(initial_payload, args.device)
    initial_model.eval()
    base_detector, base_write = extract_teacher_atoms(initial_model, layers)
    del initial_model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    initialization = plan["initialization"]
    student = ParallelResidualMLP(
        base_detector,
        base_write,
        layers,
        residual_width=residual_width,
        detector_seed=int(initialization["detector_seed"]),
        gain_initial=float(initialization["gain"]),
        bias_initial=float(initialization["bias"]),
    )
    history = fit_student(
        student,
        banks["train"],
        teacher_detector,
        teacher_write,
        plan,
        layers=layers,
        iterations=args.iterations,
        minibatch_rows=args.minibatch_rows,
        jvp_seed=int(inventory["jvp_seed"]),
        device=args.device,
    )
    artifact = terminal_artifact(student, accounting)
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint_path = args.output / "parallel_residual_mlp_checkpoint.pt"
    torch.save(artifact, checkpoint_path)
    fitted = artifact_student(
        artifact, base_detector, base_write, plan, args.device
    )
    candidates = {
        CANDIDATE_NAME: student_function(fitted, "full"),
        "step_zero_parent": student_function(fitted, "step_zero_parent"),
        "global_coordinates": student_function(fitted, "global_coordinates"),
        "initial_random_detector": student_function(
            fitted, "initial_random_detector"
        ),
        "linear_residual": student_function(fitted, "linear_residual"),
        "dense_teacher_identity": dense_teacher_function(
            teacher_detector, teacher_write, args.device
        ),
    }
    rows: list[dict[str, Any]] = []
    for name, function in candidates.items():
        rows.extend(
            evaluate_function(
                name,
                function,
                banks,
                teacher_detector,
                teacher_write,
                layers=layers,
                jvp_seed=int(inventory["jvp_seed"]),
                directions=int(inventory["jvp_directions_per_layer_per_split"]),
                device=args.device,
            )
        )
    summary = summarize(rows)
    gate = gate_outcome(rows)
    gate["all_gates_pass"] = bool(gate["representation_pass"])
    detail_path = args.output / "functional_metrics.json"
    summary_path = args.output / "functional_summary.json"
    history_path = args.output / "fit_history.json"
    gate_path = args.output / "gate.json"
    write_json(detail_path, rows)
    write_json(summary_path, summary)
    write_json(history_path, history)
    write_json(gate_path, gate)
    source = Path(__file__).resolve()
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "preflight": bool(args.preflight),
        "source_commit": git_commit(source.parents[2]),
        "source_sha256": file_sha256(source),
        "plan": str(args.plan),
        "plan_sha256": file_sha256(args.plan),
        "initial_snapshot": str(initial_snapshot),
        "initial_snapshot_sha256": file_sha256(initial_snapshot),
        "initial_snapshot_run_identity_sha256": str(
            initial_payload["run_identity_sha256"]
        ),
        "initial_snapshot_tensor_inventory_sha256": canonical_sha256(
            initial_payload["tensor_inventory"]
        ),
        "terminal_snapshot": str(terminal_snapshot),
        "terminal_snapshot_sha256": file_sha256(terminal_snapshot),
        "terminal_snapshot_run_identity_sha256": str(
            terminal_payload["run_identity_sha256"]
        ),
        "terminal_snapshot_tensor_inventory_sha256": canonical_sha256(
            terminal_payload["tensor_inventory"]
        ),
        "dataset_manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
        "sample_cap": args.sample_cap,
        "iterations": args.iterations,
        "layers": layers,
        "base_detector_sha256": tensor_sha256(
            torch.stack([base_detector[layer] for layer in layers])
        ),
        "base_write_sha256": tensor_sha256(
            torch.stack([base_write[layer] for layer in layers])
        ),
        "accounting": accounting,
        "temporary_training_state": {
            "fp32_coordinate_master_bytes": accounting["total_latent_values"] * 4,
            "fp32_coordinate_gradient_bytes": accounting["total_latent_values"] * 4,
            "adam_coordinate_moment_bytes": accounting["total_latent_values"] * 8,
            "fixed_fp32_procedural_endpoint_bytes": accounting[
                "dense_replaced_mlp_fp16_bytes"
            ]
            * 2,
            "optimizer_state_is_not_deployment_state": True,
        },
        "checkpoint_file_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "detail_sha256": file_sha256(detail_path),
        "summary_sha256": file_sha256(summary_path),
        "history_sha256": file_sha256(history_path),
        "gate_sha256": file_sha256(gate_path),
        "runtime_seconds": time.time() - started,
        "peak_cuda_bytes": (
            int(torch.cuda.max_memory_allocated())
            if args.device.startswith("cuda")
            else 0
        ),
        "gate": gate,
        "limitations": [
            "Single 124M step-zero parent, terminal teacher, and dataset.",
            "Function/JVP representation audit, not CE training.",
            "Step-zero endpoints are exact audit inputs but excluded from the compact artifact; regeneration remains a later mandatory gate.",
            "FP32 compact-coordinate optimizer state is transient training state and is reported separately.",
        ],
    }
    metadata_path = args.output / "metadata.json"
    write_json(metadata_path, metadata)
    result_bytes = sum(
        path.stat().st_size for path in args.output.rglob("*") if path.is_file()
    )
    if result_bytes > int(plan["runtime_gates"]["maximum_result_directory_bytes"]):
        raise RuntimeError("H59 result directory exceeds frozen storage gate")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
