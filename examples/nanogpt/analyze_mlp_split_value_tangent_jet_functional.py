#!/usr/bin/env python3
"""Frozen H60 split conditional-value and affine-tangent functional audit."""

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
from examples.nanogpt.analyze_mlp_lowbit_complete_neuron_functional import (
    FunctionalBank,
    canonical_sha256,
    collect_functional_bank,
    evaluate_function,
    extract_teacher_atoms,
    git_commit,
    rademacher_direction,
    summarize,
    teacher_jvp,
    tensor_sha256,
    write_json,
)
from examples.nanogpt.analyze_mlp_token_router_residual_carrier_functional import (
    dense_teacher_function,
    deterministic_orthonormal,
    deterministic_router,
)


SCHEMA_VERSION = "nanogpt_mlp_split_value_tangent_jet_functional_v1"
PLAN_SCHEMA_VERSION = "nanogpt_mlp_split_value_tangent_jet_functional_plan_v1"
CANDIDATE_NAME = "learned_split_value_tangent_jet"


def deployment_accounting(
    *,
    layers: int = 12,
    width: int = 768,
    value_rank: int = 128,
    tangent_rank: int = 96,
) -> dict[str, int | float]:
    value_shared = 3 * width * value_rank
    value_private = layers * 3 * value_rank
    tangent_shared = 2 * width * tangent_rank
    tangent_private = layers * tangent_rank
    anchor_offset = layers * 2 * width
    total_values = (
        value_shared
        + value_private
        + tangent_shared
        + tangent_private
        + anchor_offset
    )
    dense_values = layers * 2 * 4 * width * width
    return {
        "dense_replaced_mlp_fp16_values": dense_values,
        "dense_replaced_mlp_fp16_bytes": 2 * dense_values,
        "fp16_value_shared_values": value_shared,
        "fp16_value_private_values": value_private,
        "fp16_tangent_shared_values": tangent_shared,
        "fp16_tangent_private_values": tangent_private,
        "fp16_anchor_offset_values": anchor_offset,
        "total_latent_values": total_values,
        "total_checkpoint_payload_bytes": 2 * total_values,
        "latent_value_fraction": total_values / dense_values,
        "checkpoint_byte_fraction": total_values / dense_values,
        "extra_mlp_matmul_fraction": (
            3 * width * value_rank + 2 * width * tangent_rank
        )
        / (8 * width * width),
        "cached_procedural_endpoint_bytes": 2 * dense_values,
    }


class SplitValueTangentJet(torch.nn.Module):
    def __init__(
        self,
        base_detector: dict[int, torch.Tensor],
        base_write: dict[int, torch.Tensor],
        layers: list[int],
        anchors: torch.Tensor,
        *,
        value_rank: int,
        tangent_rank: int,
        router_seed: int,
        value_v_seed: int,
        tangent_v_seed: int,
        static_initial: float,
        amplitude_initial: float,
        bias_initial: float,
        tangent_initial: float,
        offset_initial: float,
    ) -> None:
        super().__init__()
        self.layers = list(layers)
        self.layer_to_row = {layer: row for row, layer in enumerate(layers)}
        first = base_detector[layers[0]]
        hidden, width = first.shape
        self.hidden = int(hidden)
        self.width = int(width)
        self.value_rank = int(value_rank)
        self.tangent_rank = int(tangent_rank)
        stacked_detector = torch.stack(
            [base_detector[layer].detach().float() for layer in layers]
        )
        stacked_write = torch.stack(
            [base_write[layer].detach().float() for layer in layers]
        )
        if stacked_detector.shape != stacked_write.shape:
            raise ValueError("procedural endpoint shapes disagree")
        if tuple(anchors.shape) != (len(layers), width):
            raise ValueError("activation-anchor shape disagrees")
        self.register_buffer("base_detector", stacked_detector)
        self.register_buffer("base_write", stacked_write)
        self.register_buffer("anchors", anchors.detach().float().clone())

        self.value_router = torch.nn.Parameter(
            deterministic_router(value_rank, width, router_seed)
        )
        self.value_static = torch.nn.Parameter(
            torch.full((len(layers), value_rank), float(static_initial))
        )
        self.value_amplitude = torch.nn.Parameter(
            torch.full((len(layers), value_rank), float(amplitude_initial))
        )
        self.value_bias = torch.nn.Parameter(
            torch.full((len(layers), value_rank), float(bias_initial))
        )
        self.value_u = torch.nn.Parameter(torch.zeros(width, value_rank))
        self.value_v = torch.nn.Parameter(
            deterministic_orthonormal(width, value_rank, value_v_seed)
        )

        self.tangent_u = torch.nn.Parameter(torch.zeros(width, tangent_rank))
        self.tangent_v = torch.nn.Parameter(
            deterministic_orthonormal(width, tangent_rank, tangent_v_seed)
        )
        self.tangent_coordinate = torch.nn.Parameter(
            torch.full((len(layers), tangent_rank), float(tangent_initial))
        )
        self.output_offset = torch.nn.Parameter(
            torch.full((len(layers), width), float(offset_initial))
        )

    def _parts(self, row: int, mode: str) -> dict[str, torch.Tensor]:
        parts = {
            "value_u": self.value_u,
            "value_v": self.value_v,
            "value_router": self.value_router,
            "value_static": self.value_static[row],
            "value_amplitude": self.value_amplitude[row],
            "value_bias": self.value_bias[row],
            "tangent_u": self.tangent_u,
            "tangent_v": self.tangent_v,
            "tangent_coordinate": self.tangent_coordinate[row],
            "anchor": self.anchors[row],
            "offset": self.output_offset[row],
        }
        if mode == "step_zero_parent":
            parts["value_u"] = torch.zeros_like(self.value_u)
            parts["tangent_u"] = torch.zeros_like(self.tangent_u)
            parts["offset"] = torch.zeros_like(self.output_offset[row])
        elif mode == "value_carrier_only":
            parts["tangent_u"] = torch.zeros_like(self.tangent_u)
            parts["offset"] = torch.zeros_like(self.output_offset[row])
        elif mode == "affine_jet_only":
            parts["value_u"] = torch.zeros_like(self.value_u)
        elif mode == "global_tangent_coordinates":
            parts["tangent_coordinate"] = self.tangent_coordinate.mean(dim=0)
        elif mode == "zero_anchor_offset":
            parts["anchor"] = torch.zeros_like(self.anchors[row])
            parts["offset"] = torch.zeros_like(self.output_offset[row])
        elif mode != "full":
            raise ValueError(f"unknown H60 mode: {mode}")
        return parts

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

        hidden = F.gelu(inputs @ detector.T)
        base_output = hidden @ write
        router_hidden = torch.tanh(
            inputs @ p["value_router"].T + p["value_bias"]
        )
        value_coordinate = (
            p["value_static"] + p["value_amplitude"] * router_hidden
        )
        value_projection = base_output @ p["value_v"]
        value_correction = (
            value_projection * value_coordinate
        ) @ p["value_u"].T
        tangent_projection = (inputs - p["anchor"]) @ p["tangent_v"]
        tangent_correction = (
            tangent_projection * p["tangent_coordinate"]
        ) @ p["tangent_u"].T
        output = (
            base_output + value_correction + p["offset"] + tangent_correction
        )
        if directions is None:
            return output, None

        base_action = teacher_jvp(inputs, directions, detector, write)
        router_action = (
            p["value_amplitude"]
            * (1.0 - router_hidden.square())
            * (directions @ p["value_router"].T)
        )
        value_projection_action = base_action @ p["value_v"]
        value_action = (
            value_projection_action * value_coordinate
            + value_projection * router_action
        ) @ p["value_u"].T
        tangent_action = (
            (directions @ p["tangent_v"]) * p["tangent_coordinate"]
        ) @ p["tangent_u"].T
        return output, base_action + value_action + tangent_action


def optimizer_for(
    student: SplitValueTangentJet, plan: dict[str, Any]
) -> torch.optim.Optimizer:
    fit = plan["fit"]
    return torch.optim.AdamW(
        [
            {
                "params": [
                    student.value_u,
                    student.value_v,
                    student.value_router,
                    student.tangent_u,
                    student.tangent_v,
                ],
                "lr": float(fit["learning_rate_shared_factors"]),
                "weight_decay": float(fit["weight_decay_shared"]),
            },
            {
                "params": [
                    student.value_static,
                    student.value_amplitude,
                    student.value_bias,
                    student.tangent_coordinate,
                    student.output_offset,
                ],
                "lr": float(fit["learning_rate_private_coordinates"]),
                "weight_decay": float(fit["weight_decay_private"]),
            },
        ]
    )


def fit_student(
    student: SplitValueTangentJet,
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
            raise RuntimeError("H60 student JVP is missing")
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
    return history


def terminal_artifact(
    student: SplitValueTangentJet, accounting: dict[str, Any]
) -> dict[str, Any]:
    values = {
        "value_u": student.value_u.detach().to(torch.float16).cpu(),
        "value_v": student.value_v.detach().to(torch.float16).cpu(),
        "value_router": student.value_router.detach().to(torch.float16).cpu(),
        "value_static": student.value_static.detach().to(torch.float16).cpu(),
        "value_amplitude": student.value_amplitude.detach().to(torch.float16).cpu(),
        "value_bias": student.value_bias.detach().to(torch.float16).cpu(),
        "tangent_u": student.tangent_u.detach().to(torch.float16).cpu(),
        "tangent_v": student.tangent_v.detach().to(torch.float16).cpu(),
        "tangent_coordinate": student.tangent_coordinate.detach().to(torch.float16).cpu(),
        "anchors": student.anchors.detach().to(torch.float16).cpu(),
        "output_offset": student.output_offset.detach().to(torch.float16).cpu(),
    }
    payload_bytes = 2 * sum(value.numel() for value in values.values())
    if payload_bytes != accounting["total_checkpoint_payload_bytes"]:
        raise AssertionError((payload_bytes, accounting))
    return {
        "schema_version": "split_value_tangent_jet_checkpoint_v1",
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
) -> SplitValueTangentJet:
    frozen = plan["frozen_representation"]
    init = plan["initialization"]
    student = SplitValueTangentJet(
        base_detector,
        base_write,
        [int(value) for value in artifact["layers"]],
        artifact["anchors"].float(),
        value_rank=int(frozen["value_rank"]),
        tangent_rank=int(frozen["tangent_rank"]),
        router_seed=int(init["value_router_seed"]),
        value_v_seed=int(init["value_v_seed"]),
        tangent_v_seed=int(init["tangent_v_seed"]),
        static_initial=float(init["value_static_coordinate"]),
        amplitude_initial=float(init["value_token_amplitude"]),
        bias_initial=float(init["value_token_bias"]),
        tangent_initial=float(init["tangent_coordinate"]),
        offset_initial=float(init["output_offset"]),
    )
    with torch.no_grad():
        for name in (
            "value_u",
            "value_v",
            "value_router",
            "value_static",
            "value_amplitude",
            "value_bias",
            "tangent_u",
            "tangent_v",
            "tangent_coordinate",
            "anchors",
            "output_offset",
        ):
            getattr(student, name).copy_(artifact[name].float())
    student.to(device)
    student.eval()
    return student


def student_function(
    student: SplitValueTangentJet, mode: str
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
            "PROMOTE_H60_TO_EXACT_SYSTEMS_GATE"
            if representation_pass
            else "REJECT_H60_SPLIT_VALUE_TANGENT_JET"
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
        raise ValueError("unexpected H60 plan schema")
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
    frozen = plan["frozen_representation"]
    value_rank = int(frozen["value_rank"])
    tangent_rank = int(frozen["tangent_rank"])
    accounting = deployment_accounting(
        layers=len(layers),
        width=width,
        value_rank=value_rank,
        tangent_rank=tangent_rank,
    )
    expected = plan["exact_deployment_accounting"]
    for key in (
        "dense_replaced_mlp_fp16_values",
        "dense_replaced_mlp_fp16_bytes",
        "fp16_value_shared_values",
        "fp16_value_private_values",
        "fp16_tangent_shared_values",
        "fp16_tangent_private_values",
        "fp16_anchor_offset_values",
        "total_latent_values",
        "cached_procedural_endpoint_bytes",
    ):
        if accounting[key] != expected[key]:
            raise AssertionError((key, accounting[key], expected[key]))
    if (
        accounting["total_checkpoint_payload_bytes"]
        != expected["total_checkpoint_bytes"]
    ):
        raise AssertionError("H60 checkpoint-byte accounting mismatch")

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
    anchors = torch.stack(
        [banks["train"].inputs[layer].float().mean(dim=0) for layer in layers]
    )
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    init = plan["initialization"]
    student = SplitValueTangentJet(
        base_detector,
        base_write,
        layers,
        anchors,
        value_rank=value_rank,
        tangent_rank=tangent_rank,
        router_seed=int(init["value_router_seed"]),
        value_v_seed=int(init["value_v_seed"]),
        tangent_v_seed=int(init["tangent_v_seed"]),
        static_initial=float(init["value_static_coordinate"]),
        amplitude_initial=float(init["value_token_amplitude"]),
        bias_initial=float(init["value_token_bias"]),
        tangent_initial=float(init["tangent_coordinate"]),
        offset_initial=float(init["output_offset"]),
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
    checkpoint_path = args.output / "split_value_tangent_jet_checkpoint.pt"
    torch.save(artifact, checkpoint_path)
    fitted = artifact_student(
        artifact, base_detector, base_write, plan, args.device
    )
    candidates = {
        CANDIDATE_NAME: student_function(fitted, "full"),
        "step_zero_parent": student_function(fitted, "step_zero_parent"),
        "value_carrier_only": student_function(fitted, "value_carrier_only"),
        "affine_jet_only": student_function(fitted, "affine_jet_only"),
        "global_tangent_coordinates": student_function(
            fitted, "global_tangent_coordinates"
        ),
        "zero_anchor_offset": student_function(fitted, "zero_anchor_offset"),
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
        "activation_anchor_sha256": tensor_sha256(anchors),
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
            "Activation anchors are data-derived but fully charged in persistent state.",
            "FP32 compact-coordinate optimizer state is transient training state and is reported separately.",
        ],
    }
    metadata_path = args.output / "metadata.json"
    write_json(metadata_path, metadata)
    result_bytes = sum(
        path.stat().st_size for path in args.output.rglob("*") if path.is_file()
    )
    if result_bytes > int(plan["runtime_gates"]["maximum_result_directory_bytes"]):
        raise RuntimeError("H60 result directory exceeds frozen storage gate")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
