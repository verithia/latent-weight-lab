#!/usr/bin/env python3
"""Frozen H68 procedural complete-neuron orbit functional audit."""

from __future__ import annotations

import argparse
import json
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
from examples.nanogpt.analyze_mlp_layer_private_diagonal_mixer_transport_functional import (
    apply_diagonal_mixer_transport,
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


SCHEMA_VERSION = "nanogpt_mlp_procedural_neuron_orbit_functional_v1"
PLAN_SCHEMA_VERSION = "nanogpt_mlp_procedural_neuron_orbit_functional_plan_v1"
CANDIDATE_NAME = "procedural_complete_neuron_orbit"


def deployment_accounting(
    *,
    layers: int = 12,
    width: int = 768,
    hidden_width: int = 3072,
    transport_diagonals_per_side: int = 3,
) -> dict[str, int | float]:
    neuron_modulation = layers * 3 * hidden_width
    transport_diagonal = layers * 2 * transport_diagonals_per_side * width
    output_offset = layers * width
    total_values = neuron_modulation + transport_diagonal + output_offset
    dense_values = layers * 2 * hidden_width * width
    return {
        "dense_replaced_mlp_fp16_values": dense_values,
        "dense_replaced_mlp_fp16_bytes": 2 * dense_values,
        "fp16_neuron_modulation_values": neuron_modulation,
        "fp16_transport_diagonal_values": transport_diagonal,
        "fp16_output_offset_values": output_offset,
        "total_latent_values": total_values,
        "total_checkpoint_payload_bytes": 2 * total_values,
        "latent_value_fraction": total_values / dense_values,
        "checkpoint_byte_fraction": total_values / dense_values,
        "cached_generated_endpoint_bytes": 2 * dense_values,
        "token_time_mlp_matmul_fraction": 1.0,
        "extra_token_elementwise_operations_per_layer": hidden_width + width,
        "persistent_procedural_parent_bytes": 0,
    }


class ProceduralCompleteNeuronOrbit(torch.nn.Module):
    def __init__(
        self,
        base_detector: dict[int, torch.Tensor],
        base_write: dict[int, torch.Tensor],
        layers: list[int],
        *,
        transport_diagonals_per_side: int,
        mixer_groups: int,
        mixer_group_width: int,
        pre_gain_initial: float,
        pre_bias_initial: float,
        post_gain_initial: float,
        transport_diagonal_initial: float,
        output_offset_initial: float,
    ) -> None:
        super().__init__()
        self.layers = list(layers)
        self.layer_to_row = {layer: row for row, layer in enumerate(layers)}
        detector = torch.stack(
            [base_detector[layer].detach().float() for layer in layers]
        )
        write = torch.stack(
            [base_write[layer].detach().float() for layer in layers]
        )
        if detector.shape != write.shape:
            raise ValueError("H68 procedural endpoints disagree")
        self.register_buffer("base_detector", detector)
        self.register_buffer("base_write", write)
        self.hidden_width = int(detector.shape[1])
        self.width = int(detector.shape[2])
        if transport_diagonals_per_side != 3:
            raise ValueError("H68 freezes three transport diagonals per side")
        if mixer_groups * mixer_group_width != self.width:
            raise ValueError("H68 mixer does not cover residual width")
        self.transport_diagonals_per_side = int(
            transport_diagonals_per_side
        )
        self.mixer_groups = int(mixer_groups)
        self.mixer_group_width = int(mixer_group_width)
        hidden_shape = (len(layers), self.hidden_width)
        transport_shape = (
            len(layers),
            self.transport_diagonals_per_side,
            self.width,
        )
        self.pre_gain = torch.nn.Parameter(
            torch.full(hidden_shape, float(pre_gain_initial))
        )
        self.pre_bias = torch.nn.Parameter(
            torch.full(hidden_shape, float(pre_bias_initial))
        )
        self.post_gain = torch.nn.Parameter(
            torch.full(hidden_shape, float(post_gain_initial))
        )
        self.input_transport_diagonals = torch.nn.Parameter(
            torch.full(transport_shape, float(transport_diagonal_initial))
        )
        self.output_transport_diagonals = torch.nn.Parameter(
            torch.full(transport_shape, float(transport_diagonal_initial))
        )
        self.output_offset = torch.nn.Parameter(
            torch.full(
                (len(layers), self.width), float(output_offset_initial)
            )
        )

    def _parts(self, row: int, mode: str) -> dict[str, Any]:
        parts: dict[str, Any] = {
            "pre_gain": self.pre_gain[row],
            "pre_bias": self.pre_bias[row],
            "post_gain": self.post_gain[row],
            "input_diagonals": self.input_transport_diagonals[row],
            "output_diagonals": self.output_transport_diagonals[row],
            "input_mode": "full",
            "output_mode": "full",
            "offset": self.output_offset[row],
        }
        if mode == "step_zero_parent":
            parts["pre_gain"] = torch.ones_like(self.pre_gain[row])
            parts["pre_bias"] = torch.zeros_like(self.pre_bias[row])
            parts["post_gain"] = torch.ones_like(self.post_gain[row])
            parts["input_mode"] = "identity"
            parts["output_mode"] = "identity"
            parts["offset"] = torch.zeros_like(self.output_offset[row])
        elif mode == "neuron_modulation_only":
            parts["input_mode"] = "identity"
            parts["output_mode"] = "identity"
        elif mode == "residual_gauges_only":
            parts["pre_gain"] = torch.ones_like(self.pre_gain[row])
            parts["pre_bias"] = torch.zeros_like(self.pre_bias[row])
            parts["post_gain"] = torch.ones_like(self.post_gain[row])
            parts["offset"] = torch.zeros_like(self.output_offset[row])
        elif mode in {"pre_gain_only", "pre_bias_only", "post_gain_only"}:
            if mode != "pre_gain_only":
                parts["pre_gain"] = torch.ones_like(self.pre_gain[row])
            if mode != "pre_bias_only":
                parts["pre_bias"] = torch.zeros_like(self.pre_bias[row])
            if mode != "post_gain_only":
                parts["post_gain"] = torch.ones_like(self.post_gain[row])
            parts["input_mode"] = "identity"
            parts["output_mode"] = "identity"
            parts["offset"] = torch.zeros_like(self.output_offset[row])
        elif mode == "input_gauge_only":
            parts["output_mode"] = "identity"
        elif mode == "output_gauge_only":
            parts["input_mode"] = "identity"
        elif mode == "shared_gauges":
            parts["input_diagonals"] = self.input_transport_diagonals.mean(dim=0)
            parts["output_diagonals"] = self.output_transport_diagonals.mean(dim=0)
        elif mode != "full":
            raise ValueError(f"unknown H68 mode: {mode}")
        return parts

    def _transport(
        self,
        values: torch.Tensor,
        diagonals: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        return apply_diagonal_mixer_transport(
            values,
            diagonals,
            groups=self.mixer_groups,
            group_width=self.mixer_group_width,
            mode=mode,
        )

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
        p = self._parts(row, mode)
        transported_input = self._transport(
            inputs, p["input_diagonals"], p["input_mode"]
        )
        pre = (transported_input @ detector.T) * p["pre_gain"] + p[
            "pre_bias"
        ]
        hidden = F.gelu(pre) * p["post_gain"]
        raw_output = hidden @ write
        output = self._transport(
            raw_output, p["output_diagonals"], p["output_mode"]
        ) + p["offset"]
        if directions is None:
            return output, None
        transported_direction = self._transport(
            directions, p["input_diagonals"], p["input_mode"]
        )
        pre_action = (transported_direction @ detector.T) * p["pre_gain"]
        hidden_action = gelu_derivative(pre) * pre_action * p["post_gain"]
        raw_action = hidden_action @ write
        action = self._transport(
            raw_action, p["output_diagonals"], p["output_mode"]
        )
        return output, action


def optimizer_for(
    student: ProceduralCompleteNeuronOrbit, plan: dict[str, Any]
) -> torch.optim.Optimizer:
    fit = plan["fit"]
    return torch.optim.AdamW(
        [
            {
                "params": [
                    student.pre_gain,
                    student.pre_bias,
                    student.post_gain,
                ],
                "lr": float(fit["learning_rate_neuron_modulation"]),
            },
            {
                "params": [
                    student.input_transport_diagonals,
                    student.output_transport_diagonals,
                    student.output_offset,
                ],
                "lr": float(fit["learning_rate_transport"]),
            },
        ],
        weight_decay=float(fit["weight_decay"]),
    )


def fit_student(
    student: ProceduralCompleteNeuronOrbit,
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
    student.train()
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
            raise RuntimeError("H68 student JVP is missing")
        output_loss = (
            (prediction - target).square().mean()
            / target.square().mean().clamp_min(1e-30)
        )
        jvp_loss = (
            (action - target_action).square().mean()
            / target_action.square().mean().clamp_min(1e-30)
        )
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
    student.eval()
    return history


def terminal_artifact(
    student: ProceduralCompleteNeuronOrbit,
    accounting: dict[str, Any],
) -> dict[str, Any]:
    values = {
        "pre_gain": student.pre_gain.detach().to(torch.float16).cpu(),
        "pre_bias": student.pre_bias.detach().to(torch.float16).cpu(),
        "post_gain": student.post_gain.detach().to(torch.float16).cpu(),
        "input_transport_diagonals": student.input_transport_diagonals.detach().to(torch.float16).cpu(),
        "output_transport_diagonals": student.output_transport_diagonals.detach().to(torch.float16).cpu(),
        "output_offset": student.output_offset.detach().to(torch.float16).cpu(),
    }
    payload_bytes = 2 * sum(value.numel() for value in values.values())
    if payload_bytes != accounting["total_checkpoint_payload_bytes"]:
        raise AssertionError((payload_bytes, accounting))
    return {
        "schema_version": "procedural_complete_neuron_orbit_checkpoint_v1",
        **values,
        "layers": student.layers,
        "transport_diagonals_per_side": student.transport_diagonals_per_side,
        "mixer_groups": student.mixer_groups,
        "mixer_group_width": student.mixer_group_width,
        "accounted_payload_bytes": payload_bytes,
        "base_detector_sha256": tensor_sha256(student.base_detector),
        "base_write_sha256": tensor_sha256(student.base_write),
    }


def make_student(
    base_detector: dict[int, torch.Tensor],
    base_write: dict[int, torch.Tensor],
    layers: list[int],
    plan: dict[str, Any],
) -> ProceduralCompleteNeuronOrbit:
    frozen = plan["frozen_representation"]
    init = plan["initialization"]
    return ProceduralCompleteNeuronOrbit(
        base_detector,
        base_write,
        layers,
        transport_diagonals_per_side=int(
            frozen["transport_diagonals_per_side"]
        ),
        mixer_groups=int(frozen["mixer_groups"]),
        mixer_group_width=int(frozen["mixer_group_width"]),
        pre_gain_initial=float(init["pre_gain"]),
        pre_bias_initial=float(init["pre_bias"]),
        post_gain_initial=float(init["post_gain"]),
        transport_diagonal_initial=float(init["input_transport_diagonals"]),
        output_offset_initial=float(init["output_offset"]),
    )


def artifact_student(
    artifact: dict[str, Any],
    base_detector: dict[int, torch.Tensor],
    base_write: dict[int, torch.Tensor],
    plan: dict[str, Any],
    device: str,
) -> ProceduralCompleteNeuronOrbit:
    student = make_student(
        base_detector,
        base_write,
        [int(value) for value in artifact["layers"]],
        plan,
    )
    with torch.no_grad():
        for name in (
            "pre_gain",
            "pre_bias",
            "post_gain",
            "input_transport_diagonals",
            "output_transport_diagonals",
            "output_offset",
        ):
            getattr(student, name).copy_(artifact[name].float())
    student.to(device)
    student.eval()
    return student


def student_function(
    student: ProceduralCompleteNeuronOrbit, mode: str
) -> Callable[
    [int, torch.Tensor, torch.Tensor | None],
    tuple[torch.Tensor, torch.Tensor | None],
]:
    def function(
        layer: int, inputs: torch.Tensor, directions: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return student.forward_function(layer, inputs, directions, mode=mode)

    return function


def parent_identity_error(
    student: ProceduralCompleteNeuronOrbit,
    bank: FunctionalBank,
    *,
    layers: list[int],
    jvp_seed: int,
    device: str,
) -> float:
    maximum = 0.0
    for layer in layers:
        inputs = bank.inputs[layer].to(device=device, dtype=torch.float32)
        direction = rademacher_direction(
            tuple(inputs.shape), jvp_seed + 99_991 * layer, device
        )
        output, action = student.forward_function(
            layer, inputs, direction, mode="step_zero_parent"
        )
        row = student.layer_to_row[layer]
        detector = student.base_detector[row]
        write = student.base_write[row]
        expected_output = F.gelu(inputs @ detector.T) @ write
        expected_action = teacher_jvp(
            inputs, direction, detector, write
        )
        assert action is not None
        maximum = max(
            maximum,
            float((output - expected_output).abs().max()),
            float((action - expected_action).abs().max()),
        )
    return maximum


def gate_outcome(
    rows: list[dict[str, Any]], parent_error: float
) -> dict[str, Any]:
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
    parent_pass = parent_error <= 5e-5
    representation_pass = (
        len(layer_gates) == 12
        and all(
            all(value for key, value in row.items() if key != "layer")
            for row in layer_gates
        )
        and parent_pass
    )
    return {
        "layer_gates": layer_gates,
        "parent_identity_max_absolute_error": parent_error,
        "parent_identity_pass": parent_pass,
        "representation_pass": representation_pass,
        "decision": (
            "PROMOTE_H68_TO_EXACT_SYSTEMS_GATE"
            if representation_pass
            else "REJECT_H68_PROCEDURAL_COMPLETE_NEURON_ORBIT"
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
        raise ValueError("unexpected H68 plan schema")
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
    hidden_width = int(teacher_plan["hidden_width"])
    frozen = plan["frozen_representation"]
    accounting = deployment_accounting(
        layers=len(layers),
        width=width,
        hidden_width=hidden_width,
        transport_diagonals_per_side=int(
            frozen["transport_diagonals_per_side"]
        ),
    )
    expected = plan["exact_deployment_accounting"]
    for key in (
        "dense_replaced_mlp_fp16_values",
        "dense_replaced_mlp_fp16_bytes",
        "fp16_neuron_modulation_values",
        "fp16_transport_diagonal_values",
        "fp16_output_offset_values",
        "total_latent_values",
        "cached_generated_endpoint_bytes",
        "persistent_procedural_parent_bytes",
    ):
        if accounting[key] != expected[key]:
            raise AssertionError((key, accounting[key], expected[key]))
    if accounting["total_checkpoint_payload_bytes"] != expected[
        "total_checkpoint_bytes"
    ]:
        raise AssertionError("H68 checkpoint-byte accounting mismatch")

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
    banks_data = {
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

    student = make_student(base_detector, base_write, layers, plan)
    initial_parent_error = parent_identity_error(
        student.to(args.device),
        banks_data["train"],
        layers=layers,
        jvp_seed=int(inventory["jvp_seed"]),
        device=args.device,
    )
    history = fit_student(
        student,
        banks_data["train"],
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
    checkpoint_path = args.output / "procedural_neuron_orbit_checkpoint.pt"
    torch.save(artifact, checkpoint_path)
    fitted = artifact_student(artifact, base_detector, base_write, plan, args.device)
    terminal_parent_error = parent_identity_error(
        fitted,
        banks_data["holdout"],
        layers=layers,
        jvp_seed=int(inventory["jvp_seed"]),
        device=args.device,
    )
    parent_error = max(initial_parent_error, terminal_parent_error)
    modes = (
        "full",
        "step_zero_parent",
        "neuron_modulation_only",
        "residual_gauges_only",
        "pre_gain_only",
        "pre_bias_only",
        "post_gain_only",
        "input_gauge_only",
        "output_gauge_only",
        "shared_gauges",
    )
    candidates = {
        (CANDIDATE_NAME if mode == "full" else mode): student_function(
            fitted, mode
        )
        for mode in modes
    }
    candidates["dense_teacher_identity"] = dense_teacher_function(
        teacher_detector, teacher_write, args.device
    )
    rows: list[dict[str, Any]] = []
    for name, function in candidates.items():
        rows.extend(
            evaluate_function(
                name,
                function,
                banks_data,
                teacher_detector,
                teacher_write,
                layers=layers,
                jvp_seed=int(inventory["jvp_seed"]),
                directions=int(inventory["jvp_directions_per_layer_per_split"]),
                device=args.device,
            )
        )
    summary = summarize(rows)
    gate = gate_outcome(rows, parent_error)
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
            "fp32_coordinate_gradient_upper_bound_bytes": accounting[
                "total_latent_values"
            ]
            * 4,
            "adam_coordinate_moment_upper_bound_bytes": accounting[
                "total_latent_values"
            ]
            * 8,
            "cached_fp16_generated_endpoint_bytes": accounting[
                "cached_generated_endpoint_bytes"
            ],
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
            "Single 124M parent, terminal teacher, and dataset.",
            "Function/JVP representation audit, not CE training.",
            "Step-zero endpoints are exact audit inputs and excluded from compact state.",
            "Generated dense endpoints are runtime cache, not checkpoint state.",
            "FP32 compact optimizer state is transient.",
        ],
    }
    metadata_path = args.output / "metadata.json"
    write_json(metadata_path, metadata)
    result_bytes = sum(
        path.stat().st_size for path in args.output.rglob("*") if path.is_file()
    )
    if result_bytes > int(plan["runtime_gates"]["maximum_result_directory_bytes"]):
        raise RuntimeError("H68 result directory exceeds frozen storage gate")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
