#!/usr/bin/env python3
"""Frozen H57 compact-native token-router/residual-carrier functional audit."""

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


SCHEMA_VERSION = "nanogpt_mlp_token_router_residual_carrier_functional_v1"
PLAN_SCHEMA_VERSION = (
    "nanogpt_mlp_token_router_residual_carrier_functional_plan_v1"
)
CANDIDATE_NAME = "learned_token_router_rank256_residual_carrier"


def deployment_accounting(
    *,
    layers: int = 12,
    width: int = 768,
    hidden: int = 3072,
    group_width: int = 32,
    carrier_rank: int = 256,
) -> dict[str, int | float]:
    if hidden % group_width:
        raise ValueError("hidden width must be divisible by group width")
    groups = hidden // group_width
    carrier_values = 2 * width * carrier_rank
    router_values = groups * width
    private_values = layers * 2 * groups
    total_values = carrier_values + router_values + private_values
    dense_values = layers * 2 * 4 * width * width
    return {
        "dense_replaced_mlp_fp16_values": dense_values,
        "dense_replaced_mlp_fp16_bytes": 2 * dense_values,
        "fp16_carrier_factor_values": carrier_values,
        "fp16_carrier_factor_bytes": 2 * carrier_values,
        "fp16_router_values": router_values,
        "fp16_router_bytes": 2 * router_values,
        "fp16_private_gate_values": private_values,
        "fp16_private_gate_bytes": 2 * private_values,
        "total_latent_values": total_values,
        "total_checkpoint_payload_bytes": 2 * total_values,
        "latent_value_fraction": total_values / dense_values,
        "checkpoint_byte_fraction": total_values / dense_values,
        "extra_mlp_matmul_fraction": (
            2 * width * carrier_rank + groups * width
        )
        / (8 * width * width),
        "cached_procedural_endpoint_bytes": 2 * dense_values,
    }


def deterministic_router(groups: int, width: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(groups, width, generator=generator) / math.sqrt(width)


def deterministic_orthonormal(
    width: int, rank: int, seed: int
) -> torch.Tensor:
    if rank > width:
        raise ValueError("carrier rank exceeds width")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    value = torch.randn(width, rank, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(value, mode="reduced")
    sign = torch.where(torch.diagonal(r) < 0, -1.0, 1.0)
    return (q * sign).float().contiguous()


class TokenRouterResidualCarrier(torch.nn.Module):
    def __init__(
        self,
        base_detector: dict[int, torch.Tensor],
        base_write: dict[int, torch.Tensor],
        layers: list[int],
        *,
        group_width: int,
        carrier_rank: int,
        router_seed: int,
        carrier_v_seed: int,
        alpha_initial: float,
        bias_initial: float,
    ) -> None:
        super().__init__()
        self.layers = list(layers)
        self.layer_to_row = {layer: row for row, layer in enumerate(layers)}
        self.group_width = int(group_width)
        first = base_detector[layers[0]]
        hidden, width = first.shape
        if hidden % group_width:
            raise ValueError("hidden width is not group aligned")
        self.hidden = int(hidden)
        self.width = int(width)
        self.groups = hidden // group_width
        self.carrier_rank = int(carrier_rank)
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
        initial_router = deterministic_router(self.groups, width, router_seed)
        self.register_buffer("initial_router", initial_router.clone())
        self.router = torch.nn.Parameter(initial_router)
        self.alpha = torch.nn.Parameter(
            torch.full((len(layers), self.groups), float(alpha_initial))
        )
        self.bias = torch.nn.Parameter(
            torch.full((len(layers), self.groups), float(bias_initial))
        )
        self.carrier_u = torch.nn.Parameter(torch.zeros(width, carrier_rank))
        self.carrier_v = torch.nn.Parameter(
            deterministic_orthonormal(width, carrier_rank, carrier_v_seed)
        )

    def _coordinates(
        self,
        row: int,
        mode: str,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        router = self.router
        alpha = self.alpha[row]
        bias = self.bias[row]
        carrier_u = self.carrier_u
        carrier_v = self.carrier_v
        if mode == "step_zero_parent":
            alpha = torch.zeros_like(alpha)
            bias = torch.zeros_like(bias)
            carrier_u = torch.zeros_like(carrier_u)
        elif mode == "carrier_only":
            alpha = torch.zeros_like(alpha)
            bias = torch.zeros_like(bias)
        elif mode == "router_only":
            carrier_u = torch.zeros_like(carrier_u)
        elif mode == "initial_random_router":
            router = self.initial_router
        elif mode not in {"full", "static_layer_gates"}:
            raise ValueError(f"unknown H57 mode: {mode}")
        return router, alpha, bias, carrier_u, carrier_v

    def forward_function(
        self,
        layer: int,
        inputs: torch.Tensor,
        directions: torch.Tensor | None,
        *,
        mode: str = "full",
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        row = self.layer_to_row[layer]
        detector = self.base_detector[row]
        write = self.base_write[row]
        router, alpha, bias, carrier_u, carrier_v = self._coordinates(row, mode)
        pre = inputs @ detector.T
        hidden = F.gelu(pre)
        if mode == "static_layer_gates":
            router_pre = bias.expand(inputs.shape[0], -1)
        else:
            router_pre = inputs @ router.T + bias
        router_hidden = torch.tanh(router_pre)
        gate = 1.0 + alpha * router_hidden
        gated_hidden = (
            hidden.reshape(inputs.shape[0], self.groups, self.group_width)
            * gate[:, :, None]
        ).reshape(inputs.shape[0], self.hidden)
        base_output = gated_hidden @ write
        output = base_output + (base_output @ carrier_v) @ carrier_u.T
        if directions is None:
            return output, None
        pre_jvp = directions @ detector.T
        hidden_jvp = gelu_derivative(pre) * pre_jvp
        if mode == "static_layer_gates":
            router_pre_jvp = torch.zeros_like(router_pre)
        else:
            router_pre_jvp = directions @ router.T
        gate_jvp = alpha * (1.0 - router_hidden.square()) * router_pre_jvp
        hidden_groups = hidden.reshape(
            inputs.shape[0], self.groups, self.group_width
        )
        hidden_jvp_groups = hidden_jvp.reshape_as(hidden_groups)
        gated_jvp = (
            hidden_jvp_groups * gate[:, :, None]
            + hidden_groups * gate_jvp[:, :, None]
        ).reshape(inputs.shape[0], self.hidden)
        base_action = gated_jvp @ write
        action = base_action + (base_action @ carrier_v) @ carrier_u.T
        return output, action


def optimizer_for(
    student: TokenRouterResidualCarrier, plan: dict[str, Any]
) -> torch.optim.Optimizer:
    fit = plan["fit"]
    return torch.optim.AdamW(
        [
            {
                "params": [student.carrier_u, student.carrier_v],
                "lr": float(fit["learning_rate_carrier_factors"]),
                "weight_decay": float(fit["weight_decay_shared"]),
            },
            {
                "params": [student.router],
                "lr": float(fit["learning_rate_router"]),
                "weight_decay": float(fit["weight_decay_shared"]),
            },
            {
                "params": [student.alpha, student.bias],
                "lr": float(fit["learning_rate_private_gates"]),
                "weight_decay": float(fit["weight_decay_private"]),
            },
        ]
    )


def fit_student(
    student: TokenRouterResidualCarrier,
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
            raise RuntimeError("H57 student JVP is missing")
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
    student: TokenRouterResidualCarrier,
    accounting: dict[str, Any],
) -> dict[str, Any]:
    values = {
        "carrier_u": student.carrier_u.detach().to(torch.float16).cpu(),
        "carrier_v": student.carrier_v.detach().to(torch.float16).cpu(),
        "router": student.router.detach().to(torch.float16).cpu(),
        "alpha": student.alpha.detach().to(torch.float16).cpu(),
        "bias": student.bias.detach().to(torch.float16).cpu(),
    }
    payload_bytes = 2 * sum(value.numel() for value in values.values())
    if payload_bytes != accounting["total_checkpoint_payload_bytes"]:
        raise AssertionError((payload_bytes, accounting))
    return {
        "schema_version": "token_router_residual_carrier_checkpoint_v1",
        **values,
        "layers": student.layers,
        "group_width": student.group_width,
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
) -> TokenRouterResidualCarrier:
    frozen = plan["frozen_representation"]
    initialization = plan["initialization"]
    student = TokenRouterResidualCarrier(
        base_detector,
        base_write,
        [int(value) for value in artifact["layers"]],
        group_width=int(frozen["group_width"]),
        carrier_rank=int(frozen["carrier_rank"]),
        router_seed=int(initialization["router_seed"]),
        carrier_v_seed=int(initialization["carrier_v_seed"]),
        alpha_initial=float(initialization["alpha"]),
        bias_initial=float(initialization["bias"]),
    )
    with torch.no_grad():
        for name in ("carrier_u", "carrier_v", "router", "alpha", "bias"):
            getattr(student, name).copy_(artifact[name].float())
    student.to(device)
    student.eval()
    return student


def student_function(
    student: TokenRouterResidualCarrier,
    mode: str,
) -> Callable[
    [int, torch.Tensor, torch.Tensor | None],
    tuple[torch.Tensor, torch.Tensor | None],
]:
    def function(
        layer: int,
        inputs: torch.Tensor,
        directions: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return student.forward_function(layer, inputs, directions, mode=mode)

    return function


def dense_teacher_function(
    teacher_detector: dict[int, torch.Tensor],
    teacher_write: dict[int, torch.Tensor],
    device: str,
) -> Callable[
    [int, torch.Tensor, torch.Tensor | None],
    tuple[torch.Tensor, torch.Tensor | None],
]:
    def function(
        layer: int,
        inputs: torch.Tensor,
        directions: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        detector = teacher_detector[layer].to(device=device, dtype=torch.float32)
        write = teacher_write[layer].to(device=device, dtype=torch.float32)
        hidden = F.gelu(inputs @ detector.T)
        output = hidden @ write
        action = (
            teacher_jvp(inputs, directions, detector, write)
            if directions is not None
            else None
        )
        return output, action

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
            "PROMOTE_H57_TO_EXACT_SYSTEMS_GATE"
            if representation_pass
            else "REJECT_H57_TOKEN_ROUTER_RESIDUAL_CARRIER"
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
        raise ValueError("unexpected H57 plan schema")
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
    hidden = int(teacher_plan["hidden_width"])
    frozen = plan["frozen_representation"]
    accounting = deployment_accounting(
        layers=len(layers),
        width=width,
        hidden=hidden,
        group_width=int(frozen["group_width"]),
        carrier_rank=int(frozen["carrier_rank"]),
    )
    expected = plan["exact_deployment_accounting"]
    for key in (
        "dense_replaced_mlp_fp16_values",
        "dense_replaced_mlp_fp16_bytes",
        "fp16_carrier_factor_values",
        "fp16_carrier_factor_bytes",
        "fp16_router_values",
        "fp16_router_bytes",
        "fp16_private_gate_values",
        "fp16_private_gate_bytes",
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
    student = TokenRouterResidualCarrier(
        base_detector,
        base_write,
        layers,
        group_width=int(frozen["group_width"]),
        carrier_rank=int(frozen["carrier_rank"]),
        router_seed=int(initialization["router_seed"]),
        carrier_v_seed=int(initialization["carrier_v_seed"]),
        alpha_initial=float(initialization["alpha"]),
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
    checkpoint_path = args.output / "token_router_residual_carrier_checkpoint.pt"
    torch.save(artifact, checkpoint_path)
    fitted = artifact_student(
        artifact, base_detector, base_write, plan, args.device
    )
    candidates = {
        CANDIDATE_NAME: student_function(fitted, "full"),
        "step_zero_parent": student_function(fitted, "step_zero_parent"),
        "carrier_only": student_function(fitted, "carrier_only"),
        "router_only": student_function(fitted, "router_only"),
        "static_layer_gates": student_function(fitted, "static_layer_gates"),
        "initial_random_router": student_function(
            fitted, "initial_random_router"
        ),
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
            "Step-zero endpoints are loaded for exact audit identity but excluded from the compact artifact; regeneration remains a later mandatory gate.",
            "FP32 compact-coordinate optimizer state is transient training state and is reported separately.",
        ],
    }
    metadata_path = args.output / "metadata.json"
    write_json(metadata_path, metadata)
    result_bytes = sum(
        path.stat().st_size for path in args.output.rglob("*") if path.is_file()
    )
    if result_bytes > int(plan["runtime_gates"]["maximum_result_directory_bytes"]):
        raise RuntimeError("H57 result directory exceeds frozen storage gate")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
