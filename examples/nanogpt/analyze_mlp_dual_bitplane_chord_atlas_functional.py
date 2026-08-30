#!/usr/bin/env python3
"""Frozen H50 dual-bitplane per-neuron chord-atlas functional audit."""

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
    complete_neuron_jvp,
    complete_neuron_output,
    evaluate_function,
    extract_teacher_atoms,
    git_commit,
    rademacher_direction,
    summarize,
    teacher_jvp,
    tensor_sha256,
    write_json,
)


SCHEMA_VERSION = "nanogpt_mlp_dual_bitplane_chord_atlas_functional_v1"
PLAN_SCHEMA_VERSION = "nanogpt_mlp_dual_bitplane_chord_atlas_functional_plan_v1"
CANDIDATE_NAME = "learned_dual_bitplane_per_neuron_chord"


def encode_binary(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    value = value.detach().float().cpu().contiguous()
    scale = value.abs().mean(dim=-1).clamp_min(1e-30)
    codes = torch.where(value >= 0, 1, -1).to(torch.int8)
    return codes, scale.to(torch.float16)


def pack_binary(codes: torch.Tensor) -> torch.Tensor:
    flat = codes.detach().cpu().to(torch.int16).flatten()
    if not bool(((flat == -1) | (flat == 1)).all()):
        raise ValueError("binary codes must lie in {-1,+1}")
    encoded = (flat > 0).to(torch.uint8)
    padding = (-encoded.numel()) % 8
    if padding:
        encoded = torch.cat([encoded, torch.zeros(padding, dtype=torch.uint8)])
    encoded = encoded.reshape(-1, 8).to(torch.int16)
    packed = sum(encoded[:, bit] << bit for bit in range(8))
    return packed.to(torch.uint8).contiguous()


def unpack_binary(packed: torch.Tensor, values: int) -> torch.Tensor:
    packed = packed.detach().cpu().to(torch.int16).flatten()
    decoded = torch.stack(
        [((packed >> bit) & 1) for bit in range(8)], dim=1
    ).flatten()[:values]
    return torch.where(decoded > 0, 1, -1).to(torch.int8)


def decode_binary(
    packed: torch.Tensor,
    scales: torch.Tensor,
    shape: tuple[int, int, int],
    *,
    device: str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    codes = unpack_binary(packed, math.prod(shape)).reshape(shape)
    return codes.to(device=device, dtype=dtype) * scales.to(
        device=device, dtype=dtype
    )[:, :, None]


def deployment_accounting(
    *, layers: int = 12, width: int = 768, atoms: int = 2304, planes: int = 2
) -> dict[str, int | float]:
    roles = 2
    binary_values = roles * planes * atoms * width
    if binary_values % 8:
        raise ValueError("binary payload must contain a multiple of eight values")
    binary_bytes = binary_values // 8
    scale_values = roles * planes * atoms
    scale_bytes = 2 * scale_values
    coordinate_values = layers * roles * planes * atoms
    coordinate_bytes = 2 * coordinate_values
    total = binary_bytes + scale_bytes + coordinate_bytes
    dense_values = layers * 2 * 4 * width * width
    dense_bytes = 2 * dense_values
    cache_bytes = layers * roles * atoms * width * 2
    return {
        "binary_endpoint_values": binary_values,
        "binary_endpoint_bytes": binary_bytes,
        "fp16_endpoint_scale_values": scale_values,
        "fp16_endpoint_scale_bytes": scale_bytes,
        "fp16_chord_coordinate_values": coordinate_values,
        "fp16_chord_coordinate_bytes": coordinate_bytes,
        "total_checkpoint_payload_bytes": total,
        "dense_replaced_mlp_fp16_bytes": dense_bytes,
        "checkpoint_byte_fraction": total / dense_bytes,
        "continuous_coordinate_values": coordinate_values,
        "continuous_coordinate_fraction": coordinate_values / dense_values,
        "cached_all_layer_fp16_endpoint_bytes": cache_bytes,
        "cached_weight_memory_fraction": cache_bytes / dense_bytes,
        "candidate_dense_matmul_fraction": atoms / (4 * width),
    }


def rank_teacher_atoms(
    inputs: torch.Tensor,
    detector: torch.Tensor,
    write: torch.Tensor,
    count: int,
    device: str,
) -> torch.Tensor:
    if count > detector.shape[0]:
        raise ValueError("requested more teacher atoms than exist")
    with torch.no_grad():
        activation_norm = F.gelu(
            inputs.to(device=device, dtype=torch.float32)
            @ detector.to(device=device, dtype=torch.float32).T
        ).norm(dim=0)
        score = activation_norm * write.to(
            device=device, dtype=torch.float32
        ).norm(dim=1)
        return torch.argsort(score, descending=True)[:count].cpu()


def acquire_cross_layer_planes(
    inputs: dict[int, torch.Tensor],
    teacher_u: dict[int, torch.Tensor],
    teacher_v: dict[int, torch.Tensor],
    layers: list[int],
    *,
    atoms_per_layer_per_plane: int,
    device: str,
) -> dict[str, Any]:
    if layers != list(range(len(layers))):
        raise ValueError("H50 frozen cross-layer assignment requires contiguous layers")
    groups = len(layers)
    group_width = atoms_per_layer_per_plane
    atoms = groups * group_width
    width = teacher_u[layers[0]].shape[1]
    raw_u = torch.empty(2, atoms, width, dtype=torch.float32)
    raw_v = torch.empty_like(raw_u)
    source_layers = torch.empty(2, atoms, dtype=torch.int64)
    source_indices = torch.empty(2, atoms, dtype=torch.int64)
    selected: dict[int, torch.Tensor] = {}
    for layer in layers:
        selected[layer] = rank_teacher_atoms(
            inputs[layer],
            teacher_u[layer],
            teacher_v[layer],
            2 * group_width,
            device,
        )
    for group, layer in enumerate(layers):
        start, stop = group * group_width, (group + 1) * group_width
        first = selected[layer][:group_width]
        next_layer = layers[(group + 1) % groups]
        second = selected[next_layer][group_width : 2 * group_width]
        raw_u[0, start:stop] = teacher_u[layer][first]
        raw_v[0, start:stop] = teacher_v[layer][first]
        raw_u[1, start:stop] = teacher_u[next_layer][second]
        raw_v[1, start:stop] = teacher_v[next_layer][second]
        source_layers[0, start:stop] = layer
        source_layers[1, start:stop] = next_layer
        source_indices[0, start:stop] = first
        source_indices[1, start:stop] = second
    initial_u = torch.zeros(groups, 2, atoms, dtype=torch.float32)
    initial_v = torch.zeros_like(initial_u)
    for plane in range(2):
        for row in range(atoms):
            source = int(source_layers[plane, row])
            initial_u[source, plane, row] = 1.0
            initial_v[source, plane, row] = 1.0
    return {
        "raw_u": raw_u,
        "raw_v": raw_v,
        "source_layers": source_layers,
        "source_indices": source_indices,
        "initial_u": initial_u,
        "initial_v": initial_v,
    }


class DualBitplaneChordBank(torch.nn.Module):
    def __init__(
        self,
        base_u: torch.Tensor,
        base_v: torch.Tensor,
        initial_u: torch.Tensor,
        initial_v: torch.Tensor,
        layers: list[int],
    ) -> None:
        super().__init__()
        if base_u.shape != base_v.shape or base_u.shape[0] != 2:
            raise ValueError("H50 requires two equal-shape endpoint planes")
        if initial_u.shape != initial_v.shape:
            raise ValueError("coordinate shapes differ")
        self.layers = list(layers)
        self.layer_to_row = {layer: row for row, layer in enumerate(layers)}
        self.register_buffer("base_u", base_u.detach().float().clone())
        self.register_buffer("base_v", base_v.detach().float().clone())
        self.coordinate_u = torch.nn.Parameter(initial_u.detach().float().clone())
        self.coordinate_v = torch.nn.Parameter(initial_v.detach().float().clone())

    def factors(
        self, layer: int, *, row_override: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.layer_to_row[layer] if row_override is None else row_override
        detector = torch.sum(
            self.coordinate_u[row, :, :, None] * self.base_u, dim=0
        )
        write = torch.sum(
            self.coordinate_v[row, :, :, None] * self.base_v, dim=0
        )
        return detector, write

    def forward_function(
        self,
        layer: int,
        inputs: torch.Tensor,
        directions: torch.Tensor | None,
        *,
        quantized: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del quantized
        detector, write = self.factors(layer)
        gain = torch.ones(detector.shape[0], device=detector.device)
        output = complete_neuron_output(inputs, detector, write, gain)
        action = (
            complete_neuron_jvp(inputs, directions, detector, write, gain)
            if directions is not None
            else None
        )
        return output, action


def fit_coordinates(
    student: DualBitplaneChordBank,
    bank: FunctionalBank,
    teacher_u: dict[int, torch.Tensor],
    teacher_v: dict[int, torch.Tensor],
    *,
    layers: list[int],
    iterations: int,
    minibatch_rows: int,
    jvp_seed: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    device: str,
) -> list[dict[str, float | int]]:
    student.to(device)
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
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
            jvp_seed + 1009 * layer + 100_003 * direction_index + 1_000_003 * start,
            device,
        )
        with torch.no_grad():
            target_action = teacher_jvp(
                inputs,
                direction,
                teacher_u[layer].to(device=device, dtype=torch.float32),
                teacher_v[layer].to(device=device, dtype=torch.float32),
            )
        prediction, action = student.forward_function(layer, inputs, direction)
        if action is None:
            raise RuntimeError("student JVP missing")
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
    student: DualBitplaneChordBank,
    u_codes: torch.Tensor,
    v_codes: torch.Tensor,
    u_scales: torch.Tensor,
    v_scales: torch.Tensor,
    accounting: dict[str, Any],
) -> dict[str, Any]:
    packed_u = pack_binary(u_codes)
    packed_v = pack_binary(v_codes)
    coordinate_u = student.coordinate_u.detach().to(torch.float16).cpu()
    coordinate_v = student.coordinate_v.detach().to(torch.float16).cpu()
    payload_bytes = packed_u.numel() + packed_v.numel()
    payload_bytes += 2 * (u_scales.numel() + v_scales.numel())
    payload_bytes += 2 * (coordinate_u.numel() + coordinate_v.numel())
    if payload_bytes != accounting["total_checkpoint_payload_bytes"]:
        raise AssertionError((payload_bytes, accounting))
    return {
        "schema_version": "dual_bitplane_chord_atlas_checkpoint_v1",
        "u_shape": list(u_codes.shape),
        "v_shape": list(v_codes.shape),
        "packed_u": packed_u,
        "packed_v": packed_v,
        "u_scales": u_scales.detach().cpu(),
        "v_scales": v_scales.detach().cpu(),
        "coordinate_u": coordinate_u,
        "coordinate_v": coordinate_v,
        "layers": student.layers,
        "accounted_payload_bytes": payload_bytes,
    }


def artifact_function(
    artifact: dict[str, Any],
    device: str,
    *,
    mode: str = "full",
) -> Callable[
    [int, torch.Tensor, torch.Tensor | None],
    tuple[torch.Tensor, torch.Tensor | None],
]:
    shape = tuple(int(value) for value in artifact["u_shape"])
    base_u = decode_binary(
        artifact["packed_u"], artifact["u_scales"], shape, device=device
    )
    base_v = decode_binary(
        artifact["packed_v"], artifact["v_scales"], shape, device=device
    )
    if mode == "random_shared_planes":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(202608504)
        random_u = torch.where(
            torch.rand(shape, generator=generator) >= 0.5, 1.0, -1.0
        )
        random_v = torch.where(
            torch.rand(shape, generator=generator) >= 0.5, 1.0, -1.0
        )
        base_u = random_u.to(device) * artifact["u_scales"].to(device).float()[:, :, None]
        base_v = random_v.to(device) * artifact["v_scales"].to(device).float()[:, :, None]
    coordinate_u = artifact["coordinate_u"].to(device=device, dtype=torch.float32)
    coordinate_v = artifact["coordinate_v"].to(device=device, dtype=torch.float32)
    layer_to_row = {int(layer): row for row, layer in enumerate(artifact["layers"])}

    def function(
        layer: int, inputs: torch.Tensor, directions: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        row = layer_to_row[layer]
        if mode == "shuffled_layer_coordinates":
            row = (row + 1) % len(layer_to_row)
        local_u = coordinate_u[row]
        local_v = coordinate_v[row]
        if mode == "plane_zero_only":
            local_u = local_u.clone()
            local_v = local_v.clone()
            local_u[1].zero_()
            local_v[1].zero_()
        if mode == "row_shared_scalar_coordinates":
            local_u = local_u.mean(dim=1, keepdim=True).expand_as(local_u)
            local_v = local_v.mean(dim=1, keepdim=True).expand_as(local_v)
        detector = torch.sum(local_u[:, :, None] * base_u, dim=0)
        write = torch.sum(local_v[:, :, None] * base_v, dim=0)
        gain = torch.ones(detector.shape[0], device=device)
        output = complete_neuron_output(inputs, detector, write, gain)
        action = (
            complete_neuron_jvp(inputs, directions, detector, write, gain)
            if directions is not None
            else None
        )
        return output, action

    return function


def dense_teacher_function(
    teacher_u: dict[int, torch.Tensor],
    teacher_v: dict[int, torch.Tensor],
    device: str,
) -> Callable[
    [int, torch.Tensor, torch.Tensor | None],
    tuple[torch.Tensor, torch.Tensor | None],
]:
    def function(
        layer: int, inputs: torch.Tensor, directions: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        detector = teacher_u[layer].to(device=device, dtype=torch.float32)
        write = teacher_v[layer].to(device=device, dtype=torch.float32)
        gain = torch.ones(detector.shape[0], device=device)
        output = complete_neuron_output(inputs, detector, write, gain)
        action = (
            complete_neuron_jvp(inputs, directions, detector, write, gain)
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
            "covariance_pass": row["retained_centered_output_covariance_energy"] >= 0.90,
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
            "PROMOTE_H50_TO_124M_CAUSAL_PREFLIGHT"
            if representation_pass
            else "REJECT_H50_DUAL_BITPLANE_CHORD_ATLAS"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--systems-result", required=True, type=Path)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-cap", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=1536)
    parser.add_argument("--minibatch-rows", type=int, default=128)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    started = time.time()
    plan = json.loads(args.plan.read_text())
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unexpected H50 functional plan schema")
    systems = json.loads(args.systems_result.read_text())
    if file_sha256(args.systems_result) != plan["sealed_systems_gate"]["result_sha256"]:
        raise ValueError("H50 systems result identity mismatch")
    if systems.get("decision") != "PASS_H50_SYSTEMS_AUTHORIZE_REPRESENTATION_AUDIT":
        raise ValueError("H50 systems gate did not pass")
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
    atoms = int(teacher_plan["candidate_hidden_width"])
    width = int(teacher_plan["model_width"])
    planes = int(plan["frozen_representation"]["planes_per_endpoint_role"])
    accounting = deployment_accounting(
        layers=len(layers), width=width, atoms=atoms, planes=planes
    )
    expected = plan["exact_deployment_accounting"]
    for key in (
        "dense_replaced_mlp_fp16_bytes",
        "binary_endpoint_values",
        "binary_endpoint_bytes",
        "fp16_endpoint_scale_values",
        "fp16_endpoint_scale_bytes",
        "fp16_chord_coordinate_values",
        "fp16_chord_coordinate_bytes",
        "total_checkpoint_bytes",
        "continuous_coordinate_values",
        "cached_all_layer_fp16_endpoint_bytes",
    ):
        actual_key = "total_checkpoint_payload_bytes" if key == "total_checkpoint_bytes" else key
        if accounting[actual_key] != expected[key]:
            raise AssertionError((key, accounting[actual_key], expected[key]))

    snapshot = args.snapshot_dir / teacher_plan["terminal_snapshot"]
    payload = load_snapshot(snapshot)
    snapshot_run_identity_sha256 = str(payload["run_identity_sha256"])
    snapshot_inventory_sha256 = canonical_sha256(payload["tensor_inventory"])
    model = model_from_snapshot(payload, args.device)
    model.eval()
    if int(model.config.n_layer) != len(layers) or int(model.config.n_embd) != width:
        raise ValueError("teacher architecture does not match frozen H50 plan")
    teacher_u, teacher_v = extract_teacher_atoms(model, layers)
    inventory = plan["frozen_function_inventory"]
    banks = {
        "train": collect_functional_bank(
            model,
            args.data_dir,
            layers,
            args.sample_cap,
            int(inventory["train_seed"]),
            2,
            int(model.config.block_size),
            args.device,
        ),
        "holdout": collect_functional_bank(
            model,
            args.data_dir,
            layers,
            args.sample_cap,
            int(inventory["holdout_seed"]),
            2,
            int(model.config.block_size),
            args.device,
        ),
    }
    del model, payload
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    acquisition = acquire_cross_layer_planes(
        banks["train"].inputs,
        teacher_u,
        teacher_v,
        layers,
        atoms_per_layer_per_plane=int(
            plan["teacher_derived_initialization"]["plane_atoms_per_layer"]
        ),
        device=args.device,
    )
    u_codes, u_scales = encode_binary(acquisition["raw_u"])
    v_codes, v_scales = encode_binary(acquisition["raw_v"])
    base_u = u_codes.float() * u_scales.float()[:, :, None]
    base_v = v_codes.float() * v_scales.float()[:, :, None]
    student = DualBitplaneChordBank(
        base_u,
        base_v,
        acquisition["initial_u"],
        acquisition["initial_v"],
        layers,
    )
    fit = plan["fit"]
    history = fit_coordinates(
        student,
        banks["train"],
        teacher_u,
        teacher_v,
        layers=layers,
        iterations=args.iterations,
        minibatch_rows=args.minibatch_rows,
        jvp_seed=int(inventory["jvp_seed"]),
        learning_rate=float(fit["learning_rate_coordinates"]),
        weight_decay=float(fit["weight_decay"]),
        gradient_clip_norm=float(fit["gradient_clip_norm"]),
        device=args.device,
    )
    artifact = terminal_artifact(
        student, u_codes, v_codes, u_scales, v_scales, accounting
    )
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint_path = args.output / "dual_bitplane_chord_atlas_checkpoint.pt"
    torch.save(artifact, checkpoint_path)

    candidates = {
        CANDIDATE_NAME: artifact_function(artifact, args.device),
        "plane_zero_only": artifact_function(
            artifact, args.device, mode="plane_zero_only"
        ),
        "random_shared_planes": artifact_function(
            artifact, args.device, mode="random_shared_planes"
        ),
        "row_shared_scalar_coordinates": artifact_function(
            artifact, args.device, mode="row_shared_scalar_coordinates"
        ),
        "shuffled_layer_coordinates": artifact_function(
            artifact, args.device, mode="shuffled_layer_coordinates"
        ),
        "dense_teacher_identity": dense_teacher_function(
            teacher_u, teacher_v, args.device
        ),
    }
    rows: list[dict[str, Any]] = []
    for name, function in candidates.items():
        rows.extend(
            evaluate_function(
                name,
                function,
                banks,
                teacher_u,
                teacher_v,
                layers=layers,
                jvp_seed=int(inventory["jvp_seed"]),
                directions=int(inventory["jvp_directions_per_layer_per_split"]),
                device=args.device,
            )
        )
    summary = summarize(rows)
    gate = gate_outcome(rows)
    gate["sealed_systems_result_sha256"] = file_sha256(args.systems_result)
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
        "systems_result": str(args.systems_result),
        "systems_result_sha256": file_sha256(args.systems_result),
        "snapshot": str(snapshot),
        "snapshot_sha256": file_sha256(snapshot),
        "snapshot_run_identity_sha256": snapshot_run_identity_sha256,
        "snapshot_tensor_inventory_sha256": snapshot_inventory_sha256,
        "data_dir": str(args.data_dir),
        "dataset_manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
        "sample_cap": args.sample_cap,
        "iterations": args.iterations,
        "layers": layers,
        "source_layer_sha256": tensor_sha256(acquisition["source_layers"]),
        "source_index_sha256": tensor_sha256(acquisition["source_indices"]),
        "accounting": accounting,
        "temporary_training_state": {
            "fixed_decoded_fp32_plane_bytes": accounting["binary_endpoint_values"] * 4,
            "fp32_coordinate_master_bytes": accounting["continuous_coordinate_values"] * 4,
            "fp32_coordinate_gradient_bytes": accounting["continuous_coordinate_values"] * 4,
            "adam_coordinate_moment_bytes": accounting["continuous_coordinate_values"] * 8,
            "optimizer_state_is_not_deployment_state": True,
        },
        "checkpoint_file_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "detail_sha256": file_sha256(detail_path),
        "summary_sha256": file_sha256(summary_path),
        "history_sha256": file_sha256(history_path),
        "gate_sha256": file_sha256(gate_path),
        "runtime_seconds": time.time() - started,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()) if args.device.startswith("cuda") else 0,
        "gate": gate,
        "limitations": [
            "Single 124M terminal dense parent and dataset.",
            "Function/JVP representation audit, not CE training.",
            "The two frozen binary planes are teacher-derived and charged in full.",
            "Materialized FP16 layer endpoints are reproducible runtime cache, not checkpoint state."
        ]
    }
    metadata_path = args.output / "metadata.json"
    write_json(metadata_path, metadata)
    result_bytes = sum(
        path.stat().st_size for path in args.output.rglob("*") if path.is_file()
    )
    if result_bytes > int(plan["runtime_gates"]["maximum_result_directory_bytes"]):
        raise RuntimeError("H50 result directory exceeds frozen storage gate")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
