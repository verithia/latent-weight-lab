#!/usr/bin/env python3
"""Audit a shared low-bit bank of complete GELU neurons in function space.

This is the frozen H47 representation audit.  It deliberately stops trying to
reconstruct the dense optimizer trajectory.  Instead, it fits paired detector
and residual-write atoms to terminal dense MLP functions and exact input JVPs
on deterministic residual-stream samples.  No CE training occurs here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
import torch.nn.functional as F

from examples.nanogpt.analyze_mlp_activation_chart_oracle import (
    ActivationCollector,
)
from examples.nanogpt.analyze_mlp_activation_update_alignment import (
    file_sha256,
    load_snapshot,
    model_from_snapshot,
)
from examples.nanogpt.analyze_residual_compatibility import (
    fixed_validation_batches,
)


SCHEMA_VERSION = "nanogpt_mlp_lowbit_complete_neuron_functional_v1"
PLAN_SCHEMA_VERSION = "nanogpt_mlp_lowbit_complete_neuron_functional_plan_v1"


def git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def tensor_sha256(value: torch.Tensor) -> str:
    digest = hashlib.sha256()
    array = value.detach().cpu().contiguous().numpy()
    digest.update(memoryview(array))
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def gelu_derivative(value: torch.Tensor) -> torch.Tensor:
    return (
        0.5 * (1.0 + torch.erf(value / math.sqrt(2.0)))
        + value
        * torch.exp(-0.5 * value.square())
        / math.sqrt(2.0 * math.pi)
    )


def complete_neuron_output(
    inputs: torch.Tensor,
    detector: torch.Tensor,
    write: torch.Tensor,
    gain: torch.Tensor,
) -> torch.Tensor:
    hidden = F.gelu(inputs @ detector.transpose(0, 1))
    return (hidden * gain) @ write


def complete_neuron_jvp(
    inputs: torch.Tensor,
    directions: torch.Tensor,
    detector: torch.Tensor,
    write: torch.Tensor,
    gain: torch.Tensor,
) -> torch.Tensor:
    pre = inputs @ detector.transpose(0, 1)
    pre_jvp = directions @ detector.transpose(0, 1)
    hidden_jvp = gelu_derivative(pre) * pre_jvp
    return (hidden_jvp * gain) @ write


def encode_symmetric_int4(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return signed codes and FP16 per-row scales for range [-7, 7]."""
    if value.ndim != 2:
        raise ValueError("int4 encoder requires a rank-two tensor")
    maximum = value.detach().abs().amax(dim=1, keepdim=True)
    scale = (maximum / 7.0).clamp_min(torch.finfo(torch.float16).tiny)
    scale = scale.to(torch.float16)
    codes = torch.round(value.detach() / scale.float()).clamp(-7, 7)
    return codes.to(torch.int8), scale


def pack_signed_int4(codes: torch.Tensor) -> torch.Tensor:
    flat = codes.detach().to(torch.int16).reshape(-1)
    if bool(((flat < -7) | (flat > 7)).any()):
        raise ValueError("signed int4 codes must lie in [-7,7]")
    if flat.numel() % 2:
        flat = torch.cat((flat, torch.zeros(1, dtype=flat.dtype)))
    nibbles = torch.bitwise_and(flat, 15).to(torch.uint8)
    return nibbles[0::2] | (nibbles[1::2] << 4)


def unpack_signed_int4(packed: torch.Tensor, values: int) -> torch.Tensor:
    packed = packed.detach().to(torch.uint8).reshape(-1)
    low = torch.bitwise_and(packed, 15).to(torch.int16)
    high = torch.bitwise_and(packed >> 4, 15).to(torch.int16)
    interleaved = torch.empty(
        packed.numel() * 2, dtype=torch.int16, device=packed.device
    )
    interleaved[0::2] = low
    interleaved[1::2] = high
    interleaved = interleaved[:values]
    interleaved = torch.where(interleaved >= 8, interleaved - 16, interleaved)
    return interleaved.to(torch.int8)


def decode_symmetric_int4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    shape: tuple[int, int],
    *,
    device: str | torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    codes = unpack_signed_int4(packed, math.prod(shape)).reshape(shape)
    return codes.to(device=device, dtype=dtype) * scales.to(
        device=device, dtype=dtype
    )


def ste_symmetric_int4(value: torch.Tensor) -> torch.Tensor:
    maximum = value.detach().abs().amax(dim=1, keepdim=True)
    scale = (maximum / 7.0).clamp_min(torch.finfo(value.dtype).tiny)
    quantized = torch.round(value / scale).clamp(-7, 7) * scale
    return value + (quantized - value).detach()


def relative_rmse(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    reference = reference.detach().double()
    candidate = candidate.detach().double()
    numerator = (candidate - reference).square().sum()
    denominator = reference.square().sum().clamp_min(1e-30)
    return float((numerator / denominator).sqrt())


def retained_centered_energy(
    reference: torch.Tensor, candidate: torch.Tensor
) -> float:
    reference = reference.detach().double()
    error = candidate.detach().double() - reference
    reference = reference - reference.mean(dim=0, keepdim=True)
    error = error - error.mean(dim=0, keepdim=True)
    denominator = reference.square().sum().clamp_min(1e-30)
    return float(1.0 - error.square().sum() / denominator)


def deployment_accounting(atoms: int, layers: int, width: int) -> dict[str, Any]:
    atom_values = atoms * width * 2
    atom_bytes = (atom_values + 1) // 2
    scale_values = atoms * 2
    gain_values = layers * atoms
    scale_bytes = scale_values * 2
    gain_bytes = gain_values * 2
    total = atom_bytes + scale_bytes + gain_bytes
    dense_bytes = layers * 2 * 4 * width * width * 2
    return {
        "atoms": atoms,
        "layers": layers,
        "width": width,
        "int4_atom_values": atom_values,
        "int4_atom_bytes": atom_bytes,
        "fp16_scale_values": scale_values,
        "fp16_scale_bytes": scale_bytes,
        "fp16_gain_values": gain_values,
        "fp16_gain_bytes": gain_bytes,
        "total_checkpoint_payload_bytes": total,
        "dense_replaced_mlp_fp16_bytes": dense_bytes,
        "checkpoint_byte_fraction": total / dense_bytes,
        "cached_fp16_bank_bytes": atom_values * 2,
        "fp32_master_bank_bytes": atom_values * 4,
    }


@dataclass
class FunctionalBank:
    inputs: dict[int, torch.Tensor]
    outputs: dict[int, torch.Tensor]


def collect_functional_bank(
    model: torch.nn.Module,
    data_dir: Path,
    layers: list[int],
    sample_cap: int,
    seed: int,
    batch_size: int,
    block_size: int,
    device: str,
) -> FunctionalBank:
    batches_needed = (
        sample_cap + batch_size * block_size - 1
    ) // (batch_size * block_size)
    batches = fixed_validation_batches(
        data_dir,
        batch_size=batch_size,
        block_size=block_size,
        batches=batches_needed,
        seed=seed,
    )
    collector = ActivationCollector(
        model,
        layers,
        sample_cap,
        collect_pre_gelu=False,
        collect_mlp_input=True,
    )
    try:
        with torch.no_grad():
            for batch in batches:
                model(batch.to(device), None)
                if collector.complete():
                    break
        if not collector.complete():
            raise RuntimeError("functional activation collection is incomplete")
        return FunctionalBank(
            inputs={
                layer: collector.tensor(layer, "mlp_input").contiguous()
                for layer in layers
            },
            outputs={
                layer: collector.tensor(layer, "mlp_out").contiguous()
                for layer in layers
            },
        )
    finally:
        collector.close()


def extract_teacher_atoms(
    model: torch.nn.Module, layers: Iterable[int]
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    detector: dict[int, torch.Tensor] = {}
    write: dict[int, torch.Tensor] = {}
    for layer in layers:
        mlp = model.transformer.h[layer].mlp
        if getattr(mlp.c_fc, "bias", None) is not None:
            raise ValueError("H47 requires the frozen bias-free teacher")
        if getattr(mlp.c_proj, "bias", None) is not None:
            raise ValueError("H47 requires the frozen bias-free teacher")
        detector[layer] = mlp.c_fc.weight.detach().float().cpu().contiguous()
        write[layer] = (
            mlp.c_proj.weight.detach().float().cpu().transpose(0, 1).contiguous()
        )
        if detector[layer].shape != write[layer].shape:
            raise ValueError("teacher detector/write atom shapes disagree")
    return detector, write


def balanced_atom_quotas(atoms: int, layers: list[int]) -> dict[int, int]:
    quotient, remainder = divmod(atoms, len(layers))
    return {
        layer: quotient + int(index < remainder)
        for index, layer in enumerate(layers)
    }


def select_complete_atoms(
    inputs: dict[int, torch.Tensor],
    detector: dict[int, torch.Tensor],
    write: dict[int, torch.Tensor],
    quotas: dict[int, int],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[int, list[int]]]:
    selected_u: list[torch.Tensor] = []
    selected_v: list[torch.Tensor] = []
    sources: list[int] = []
    indices: dict[int, list[int]] = {}
    with torch.no_grad():
        for layer in sorted(quotas):
            x = inputs[layer].to(device=device, dtype=torch.float32)
            u = detector[layer].to(device=device, dtype=torch.float32)
            activation_energy = torch.zeros(u.shape[0], device=device)
            for chunk in x.split(128):
                activation_energy += F.gelu(chunk @ u.T).square().sum(dim=0)
            score = activation_energy.sqrt() * write[layer].float().norm(dim=1).to(
                device
            )
            order = torch.argsort(score, descending=True, stable=True)
            chosen = order[: quotas[layer]].cpu().tolist()
            indices[layer] = [int(value) for value in chosen]
            selected_u.append(detector[layer][chosen])
            selected_v.append(write[layer][chosen])
            sources.extend([layer] * len(chosen))
    return (
        torch.cat(selected_u, dim=0),
        torch.cat(selected_v, dim=0),
        torch.tensor(sources, dtype=torch.long),
        indices,
    )


class SharedCompleteNeuronBank(torch.nn.Module):
    def __init__(
        self,
        detector: torch.Tensor,
        write: torch.Tensor,
        source_layers: torch.Tensor,
        layers: list[int],
        *,
        train_atoms: bool,
    ) -> None:
        super().__init__()
        self.layers = list(layers)
        self.layer_to_row = {layer: row for row, layer in enumerate(layers)}
        if train_atoms:
            self.detector = torch.nn.Parameter(detector.clone())
            self.write = torch.nn.Parameter(write.clone())
        else:
            self.register_buffer("detector", detector.clone())
            self.register_buffer("write", write.clone())
        gain = torch.zeros(len(layers), detector.shape[0], dtype=torch.float32)
        for atom, source in enumerate(source_layers.tolist()):
            gain[self.layer_to_row[int(source)], atom] = 1.0
        self.gain = torch.nn.Parameter(gain)
        self.train_atoms = bool(train_atoms)

    def factors(
        self, layer: int, *, quantized: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        detector = ste_symmetric_int4(self.detector) if quantized else self.detector
        write = ste_symmetric_int4(self.write) if quantized else self.write
        return detector, write, self.gain[self.layer_to_row[layer]]

    def forward_function(
        self,
        layer: int,
        inputs: torch.Tensor,
        directions: torch.Tensor | None,
        *,
        quantized: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        detector, write, gain = self.factors(layer, quantized=quantized)
        output = complete_neuron_output(inputs, detector, write, gain)
        action = (
            complete_neuron_jvp(inputs, directions, detector, write, gain)
            if directions is not None
            else None
        )
        return output, action


class PrivateCompleteNeuronBank(torch.nn.Module):
    def __init__(
        self,
        detector: dict[int, torch.Tensor],
        write: dict[int, torch.Tensor],
        layers: list[int],
    ) -> None:
        super().__init__()
        self.layers = list(layers)
        self.layer_to_row = {layer: row for row, layer in enumerate(layers)}
        self.detectors = torch.nn.ParameterList(
            [torch.nn.Parameter(detector[layer].clone()) for layer in layers]
        )
        self.writes = torch.nn.ParameterList(
            [torch.nn.Parameter(write[layer].clone()) for layer in layers]
        )
        self.gains = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.ones(detector[layer].shape[0]))
                for layer in layers
            ]
        )

    def forward_function(
        self,
        layer: int,
        inputs: torch.Tensor,
        directions: torch.Tensor | None,
        *,
        quantized: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        row = self.layer_to_row[layer]
        detector = self.detectors[row]
        write = self.writes[row]
        if quantized:
            detector = ste_symmetric_int4(detector)
            write = ste_symmetric_int4(write)
        gain = self.gains[row]
        output = complete_neuron_output(inputs, detector, write, gain)
        action = (
            complete_neuron_jvp(inputs, directions, detector, write, gain)
            if directions is not None
            else None
        )
        return output, action


def rademacher_direction(
    shape: tuple[int, ...], seed: int, device: str
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    value = torch.randint(0, 2, shape, generator=generator, dtype=torch.float32)
    value = (value * 2.0 - 1.0) / math.sqrt(shape[-1])
    return value.to(device)


def teacher_jvp(
    inputs: torch.Tensor,
    directions: torch.Tensor,
    detector: torch.Tensor,
    write: torch.Tensor,
) -> torch.Tensor:
    gain = torch.ones(detector.shape[0], device=inputs.device)
    return complete_neuron_jvp(inputs, directions, detector, write, gain)


def optimizer_for(
    model: torch.nn.Module,
    learning_rate_atoms: float,
    learning_rate_gains: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    atom_parameters: list[torch.nn.Parameter] = []
    gain_parameters: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if "gain" in name:
            gain_parameters.append(parameter)
        else:
            atom_parameters.append(parameter)
    groups: list[dict[str, Any]] = []
    if atom_parameters:
        groups.append(
            {
                "params": atom_parameters,
                "lr": learning_rate_atoms,
                "weight_decay": weight_decay,
            }
        )
    groups.append(
        {
            "params": gain_parameters,
            "lr": learning_rate_gains,
            "weight_decay": 0.0,
        }
    )
    return torch.optim.AdamW(groups)


def fit_student(
    student: torch.nn.Module,
    bank: FunctionalBank,
    teacher_detector: dict[int, torch.Tensor],
    teacher_write: dict[int, torch.Tensor],
    *,
    layers: list[int],
    iterations: int,
    minibatch_rows: int,
    jvp_seed: int,
    learning_rate_atoms: float,
    learning_rate_gains: float,
    weight_decay: float,
    gradient_clip_norm: float,
    device: str,
) -> list[dict[str, float | int]]:
    student.to(device)
    optimizer = optimizer_for(
        student, learning_rate_atoms, learning_rate_gains, weight_decay
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
        dense_u = teacher_detector[layer].to(device=device, dtype=torch.float32)
        dense_v = teacher_write[layer].to(device=device, dtype=torch.float32)
        with torch.no_grad():
            target_action = teacher_jvp(inputs, direction, dense_u, dense_v)
        prediction, action = student.forward_function(
            layer, inputs, direction, quantized=True
        )
        if action is None:
            raise RuntimeError("student did not produce a JVP")
        output_loss = (prediction - target).square().mean() / target.square().mean().clamp_min(
            1e-30
        )
        jvp_loss = (action - target_action).square().mean() / target_action.square().mean().clamp_min(
            1e-30
        )
        loss = output_loss + 0.25 * jvp_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            student.parameters(), gradient_clip_norm
        )
        optimizer.step()
        if iteration in {0, iterations // 4, iterations // 2, 3 * iterations // 4, iterations - 1}:
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


def terminal_shared_artifact(
    student: SharedCompleteNeuronBank,
    accounting: dict[str, Any],
) -> dict[str, Any]:
    u_codes, u_scales = encode_symmetric_int4(student.detector)
    v_codes, v_scales = encode_symmetric_int4(student.write)
    packed_u = pack_signed_int4(u_codes).cpu()
    packed_v = pack_signed_int4(v_codes).cpu()
    gains = student.gain.detach().to(torch.float16).cpu()
    payload_bytes = (
        packed_u.numel()
        + packed_v.numel()
        + 2 * u_scales.numel()
        + 2 * v_scales.numel()
        + 2 * gains.numel()
    )
    if payload_bytes != accounting["total_checkpoint_payload_bytes"]:
        raise AssertionError((payload_bytes, accounting))
    return {
        "schema_version": "complete_neuron_int4_checkpoint_v1",
        "u_shape": list(student.detector.shape),
        "v_shape": list(student.write.shape),
        "packed_u": packed_u,
        "packed_v": packed_v,
        "u_scales": u_scales.cpu(),
        "v_scales": v_scales.cpu(),
        "gains": gains,
        "layers": student.layers,
        "accounted_payload_bytes": payload_bytes,
    }


def shared_artifact_function(
    artifact: dict[str, Any], device: str
) -> Callable[[int, torch.Tensor, torch.Tensor | None], tuple[torch.Tensor, torch.Tensor | None]]:
    detector = decode_symmetric_int4(
        artifact["packed_u"],
        artifact["u_scales"],
        tuple(artifact["u_shape"]),
        device=device,
    )
    write = decode_symmetric_int4(
        artifact["packed_v"],
        artifact["v_scales"],
        tuple(artifact["v_shape"]),
        device=device,
    )
    gains = artifact["gains"].to(device=device, dtype=torch.float32)
    layer_to_row = {
        int(layer): row for row, layer in enumerate(artifact["layers"])
    }

    def function(
        layer: int, inputs: torch.Tensor, directions: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        gain = gains[layer_to_row[layer]]
        output = complete_neuron_output(inputs, detector, write, gain)
        action = (
            complete_neuron_jvp(inputs, directions, detector, write, gain)
            if directions is not None
            else None
        )
        return output, action

    return function


def live_student_function(
    student: torch.nn.Module, *, quantized: bool
) -> Callable[[int, torch.Tensor, torch.Tensor | None], tuple[torch.Tensor, torch.Tensor | None]]:
    def function(
        layer: int, inputs: torch.Tensor, directions: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return student.forward_function(
            layer, inputs, directions, quantized=quantized
        )

    return function


def evaluate_function(
    name: str,
    function: Callable[
        [int, torch.Tensor, torch.Tensor | None],
        tuple[torch.Tensor, torch.Tensor | None],
    ],
    banks: dict[str, FunctionalBank],
    teacher_detector: dict[int, torch.Tensor],
    teacher_write: dict[int, torch.Tensor],
    *,
    layers: list[int],
    jvp_seed: int,
    directions: int,
    device: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for split_index, (split, bank) in enumerate(banks.items()):
            for layer in layers:
                inputs = bank.inputs[layer].to(device=device, dtype=torch.float32)
                reference = bank.outputs[layer].to(device=device, dtype=torch.float32)
                candidate, _ = function(layer, inputs, None)
                error_energy = 0.0
                reference_energy = 0.0
                for direction_index in range(directions):
                    direction = rademacher_direction(
                        tuple(inputs.shape),
                        jvp_seed
                        + 1009 * layer
                        + 100_003 * direction_index
                        + 10_000_019 * split_index,
                        device,
                    )
                    reference_action = teacher_jvp(
                        inputs,
                        direction,
                        teacher_detector[layer].to(device),
                        teacher_write[layer].to(device),
                    )
                    _ignored, candidate_action = function(layer, inputs, direction)
                    if candidate_action is None:
                        raise RuntimeError("candidate JVP is missing")
                    error_energy += float(
                        (candidate_action.double() - reference_action.double())
                        .square()
                        .sum()
                    )
                    reference_energy += float(reference_action.double().square().sum())
                row = {
                    "candidate": name,
                    "split": split,
                    "layer": layer,
                    "relative_output_rmse": relative_rmse(reference, candidate),
                    "relative_jvp_rmse": math.sqrt(
                        error_energy / max(reference_energy, 1e-30)
                    ),
                    "retained_centered_output_covariance_energy": retained_centered_energy(
                        reference, candidate
                    ),
                    "finite": bool(torch.isfinite(candidate).all()),
                }
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for candidate in sorted({row["candidate"] for row in rows}):
        for split in ("train", "holdout"):
            selected = [
                row
                for row in rows
                if row["candidate"] == candidate and row["split"] == split
            ]
            output.append(
                {
                    "candidate": candidate,
                    "split": split,
                    "layers": len(selected),
                    "mean_relative_output_rmse": sum(
                        row["relative_output_rmse"] for row in selected
                    )
                    / len(selected),
                    "maximum_relative_output_rmse": max(
                        row["relative_output_rmse"] for row in selected
                    ),
                    "mean_relative_jvp_rmse": sum(
                        row["relative_jvp_rmse"] for row in selected
                    )
                    / len(selected),
                    "maximum_relative_jvp_rmse": max(
                        row["relative_jvp_rmse"] for row in selected
                    ),
                    "mean_retained_centered_output_covariance_energy": sum(
                        row["retained_centered_output_covariance_energy"]
                        for row in selected
                    )
                    / len(selected),
                    "minimum_retained_centered_output_covariance_energy": min(
                        row["retained_centered_output_covariance_energy"]
                        for row in selected
                    ),
                    "finite": all(row["finite"] for row in selected),
                }
            )
    return output


def gate_outcome(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["candidate"] == "learned_shared_int4" and row["split"] == "holdout"
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
            "PROMOTE_H47_TO_SYSTEMS_GATE"
            if representation_pass
            else "REJECT_H47_SHARED_COMPLETE_NEURON_BANK"
        ),
    }


def benchmark_mlp(
    teacher_detector: torch.Tensor,
    teacher_write: torch.Tensor,
    artifact: dict[str, Any],
    *,
    layer: int,
    tokens: int,
    trials: int,
    device: str,
) -> dict[str, float | int]:
    if not device.startswith("cuda"):
        return {"available": 0, "reason": "CUDA required"}
    dense_u = teacher_detector.to(device=device, dtype=torch.float16)
    dense_v = teacher_write.to(device=device, dtype=torch.float16)
    compact_u = decode_symmetric_int4(
        artifact["packed_u"],
        artifact["u_scales"],
        tuple(artifact["u_shape"]),
        device=device,
        dtype=torch.float16,
    )
    compact_v = decode_symmetric_int4(
        artifact["packed_v"],
        artifact["v_scales"],
        tuple(artifact["v_shape"]),
        device=device,
        dtype=torch.float16,
    )
    layer_row = artifact["layers"].index(layer)
    gain = artifact["gains"][layer_row].to(device=device, dtype=torch.float16)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(202608314)
    inputs = torch.randn(tokens, dense_u.shape[1], generator=generator).to(
        device=device, dtype=torch.float16
    )

    def dense() -> torch.Tensor:
        return F.gelu(inputs @ dense_u.T) @ dense_v

    def compact() -> torch.Tensor:
        return (F.gelu(inputs @ compact_u.T) * gain) @ compact_v

    def measure(function: Callable[[], torch.Tensor]) -> float:
        for _ in range(5):
            function()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(trials):
            function()
        end.record()
        torch.cuda.synchronize()
        return float(start.elapsed_time(end) / trials)

    dense_ms = measure(dense)
    compact_ms = measure(compact)
    return {
        "available": 1,
        "tokens": tokens,
        "trials": trials,
        "dense_ms": dense_ms,
        "compact_ms": compact_ms,
        "compact_over_dense_latency": compact_ms / dense_ms,
        "latency_gate_max": 0.65,
        "latency_gate_pass": compact_ms / dense_ms <= 0.65,
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
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
        raise ValueError("unexpected H47 plan schema")
    if args.preflight:
        args.sample_cap = min(args.sample_cap, 128)
        args.iterations = min(args.iterations, 24)
    layers = [int(value) for value in plan["frozen_teacher"]["layers"]]
    if layers != list(range(12)):
        raise ValueError("H47 frozen layer inventory changed")
    atoms = int(plan["frozen_representation"]["complete_atoms"])
    width = int(plan["frozen_teacher"]["model_width"])
    accounting = deployment_accounting(atoms, len(layers), width)
    expected = plan["exact_deployment_accounting"]
    for key in (
        "int4_atom_values",
        "int4_atom_bytes",
        "fp16_scale_values",
        "fp16_scale_bytes",
        "total_checkpoint_bytes",
        "dense_replaced_mlp_fp16_bytes",
    ):
        actual_key = (
            "total_checkpoint_payload_bytes" if key == "total_checkpoint_bytes" else key
        )
        if accounting[actual_key] != expected[key]:
            raise AssertionError((key, accounting[actual_key], expected[key]))

    snapshot = args.snapshot_dir / "step_000238.pt"
    payload = load_snapshot(snapshot)
    snapshot_run_identity_sha256 = str(payload["run_identity_sha256"])
    snapshot_inventory_sha256 = canonical_sha256(payload["tensor_inventory"])
    model = model_from_snapshot(payload, args.device)
    model.eval()
    if int(model.config.n_layer) != 12 or int(model.config.n_embd) != width:
        raise ValueError("teacher architecture does not match frozen H47 plan")
    teacher_detector, teacher_write = extract_teacher_atoms(model, layers)
    train_seed = int(plan["frozen_function_inventory"]["train_seed"])
    holdout_seed = int(plan["frozen_function_inventory"]["holdout_seed"])
    jvp_seed = int(plan["frozen_function_inventory"]["jvp_seed"])
    block_size = int(model.config.block_size)
    banks = {
        "train": collect_functional_bank(
            model,
            args.data_dir,
            layers,
            args.sample_cap,
            train_seed,
            2,
            block_size,
            args.device,
        ),
        "holdout": collect_functional_bank(
            model,
            args.data_dir,
            layers,
            args.sample_cap,
            holdout_seed,
            2,
            block_size,
            args.device,
        ),
    }
    del model, payload
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    quotas = balanced_atom_quotas(atoms, layers)
    selected_u, selected_v, source_layers, selected_indices = select_complete_atoms(
        banks["train"].inputs,
        teacher_detector,
        teacher_write,
        quotas,
        args.device,
    )
    learned = SharedCompleteNeuronBank(
        selected_u, selected_v, source_layers, layers, train_atoms=True
    )

    random_generator = torch.Generator(device="cpu")
    random_generator.manual_seed(202608315)
    random_u = torch.randn(selected_u.shape, generator=random_generator)
    random_v = torch.randn(selected_v.shape, generator=random_generator)
    random_u = random_u / random_u.norm(dim=1, keepdim=True).clamp_min(1e-30)
    random_v = random_v / random_v.norm(dim=1, keepdim=True).clamp_min(1e-30)
    random_u *= selected_u.norm(dim=1).median()
    random_v *= selected_v.norm(dim=1).median()
    random_source = torch.tensor(
        [layers[index % len(layers)] for index in range(atoms)], dtype=torch.long
    )
    random_shared = SharedCompleteNeuronBank(
        random_u, random_v, random_source, layers, train_atoms=False
    )

    private_u = {
        layer: teacher_detector[layer][selected_indices[layer]] for layer in layers
    }
    private_v = {
        layer: teacher_write[layer][selected_indices[layer]] for layer in layers
    }
    private = PrivateCompleteNeuronBank(private_u, private_v, layers)

    fit_arguments = {
        "bank": banks["train"],
        "teacher_detector": teacher_detector,
        "teacher_write": teacher_write,
        "layers": layers,
        "iterations": args.iterations,
        "minibatch_rows": args.minibatch_rows,
        "jvp_seed": jvp_seed,
        "learning_rate_atoms": float(plan["fit"]["learning_rate_atoms"]),
        "learning_rate_gains": float(plan["fit"]["learning_rate_gains"]),
        "weight_decay": float(plan["fit"]["weight_decay"]),
        "gradient_clip_norm": float(plan["fit"]["gradient_clip_norm"]),
        "device": args.device,
    }
    histories = {
        "learned_shared_int4": fit_student(learned, **fit_arguments),
        "random_shared_int4": fit_student(random_shared, **fit_arguments),
        "private_narrow_int4": fit_student(private, **fit_arguments),
    }
    artifact = terminal_shared_artifact(learned, accounting)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output / "complete_neuron_int4_checkpoint.pt"
    torch.save(artifact, checkpoint_path)

    learned_function = shared_artifact_function(artifact, args.device)
    candidates = {
        "learned_shared_int4": learned_function,
        "learned_shared_fp32_master": live_student_function(
            learned, quantized=False
        ),
        "random_shared_int4": live_student_function(
            random_shared, quantized=True
        ),
        "private_narrow_int4": live_student_function(private, quantized=True),
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
                jvp_seed=jvp_seed,
                directions=int(
                    plan["frozen_function_inventory"][
                        "jvp_directions_per_layer_per_split"
                    ]
                ),
                device=args.device,
            )
        )
    summary = summarize(rows)
    gate = gate_outcome(rows)
    benchmark = benchmark_mlp(
        teacher_detector[6],
        teacher_write[6],
        artifact,
        layer=6,
        tokens=4096 if not args.preflight else 512,
        trials=20 if not args.preflight else 5,
        device=args.device,
    )
    gate["systems_preflight"] = benchmark
    gate["all_gates_pass"] = bool(
        gate["representation_pass"] and benchmark.get("latency_gate_pass", False)
    )
    gate["decision"] = (
        "PROMOTE_H47_TO_124M_CAUSAL_PREFLIGHT"
        if gate["all_gates_pass"]
        else "REJECT_H47_SHARED_COMPLETE_NEURON_BANK"
    )

    detail_path = args.output / "functional_metrics.json"
    summary_path = args.output / "functional_summary.json"
    history_path = args.output / "fit_history.json"
    gate_path = args.output / "gate.json"
    write_json(detail_path, rows)
    write_json(summary_path, summary)
    write_json(history_path, histories)
    write_json(gate_path, gate)
    source = Path(__file__).resolve()
    runtime = time.time() - started
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "preflight": bool(args.preflight),
        "source_commit": git_commit(source.parents[2]),
        "source_sha256": file_sha256(source),
        "plan": str(args.plan),
        "plan_sha256": file_sha256(args.plan),
        "snapshot": str(snapshot),
        "snapshot_sha256": file_sha256(snapshot),
        "snapshot_run_identity_sha256": snapshot_run_identity_sha256,
        "snapshot_tensor_inventory_sha256": snapshot_inventory_sha256,
        "data_dir": str(args.data_dir),
        "dataset_manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
        "sample_cap": args.sample_cap,
        "iterations_per_candidate": args.iterations,
        "layers": layers,
        "quotas": quotas,
        "selected_index_sha256": tensor_sha256(
            torch.tensor(
                [index for layer in layers for index in selected_indices[layer]],
                dtype=torch.int64,
            )
        ),
        "accounting": accounting,
        "temporary_training_state": {
            "fp32_master_bank_bytes": accounting["fp32_master_bank_bytes"],
            "fp32_master_bank_gradient_bytes": accounting[
                "fp32_master_bank_bytes"
            ],
            "adam_bank_moment_bytes": 2 * accounting["fp32_master_bank_bytes"],
            "fp32_gain_master_bytes": len(layers) * atoms * 4,
            "optimizer_state_is_not_deployment_state": True,
        },
        "checkpoint_file_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "detail_sha256": file_sha256(detail_path),
        "summary_sha256": file_sha256(summary_path),
        "history_sha256": file_sha256(history_path),
        "gate_sha256": file_sha256(gate_path),
        "runtime_seconds": runtime,
        "peak_cuda_bytes": (
            int(torch.cuda.max_memory_allocated()) if args.device.startswith("cuda") else 0
        ),
        "gate": gate,
        "limitations": [
            "Single 124M terminal dense parent, dataset, initialization, and schedule.",
            "This is a function/JVP representation audit, not CE training.",
            "The checkpoint budget is measured in deployed bytes, not trainable latent scalars.",
            "The cached FP16 dequantized bank is runtime memory and is accounted separately.",
        ],
    }
    metadata_path = args.output / "metadata.json"
    write_json(metadata_path, metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
