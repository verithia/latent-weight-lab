#!/usr/bin/env python3
"""H51a low-bit global-frame/DCT-carrier MLP manifold capacity audit.

This is a representation oracle, not a language-model training run.  It fits
two globally shared dense frames and per-target nuisance coordinates to the
top temporal PCs of six 239-state MLP paths, quantizes the frames to int8 and
int4, and then refines only the nuisance coordinates.  The nuisance
coordinates represent alternative points on a manifold and are not counted
simultaneously in the deployment checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from examples.nanogpt.analyze_mlp_role_wide_givens_atlas import (
    load_node_pc_inventory,
)
from examples.nanogpt.analyze_mlp_synthetic_muon_program_joint import (
    FROZEN_PARAMETERS,
)


SCHEMA_VERSION = "nanogpt_mlp_lowbit_global_frame_dct_carrier_capacity_v1"
PLAN_SCHEMA_VERSION = (
    "nanogpt_mlp_lowbit_global_frame_dct_carrier_capacity_plan_v1"
)
WIDTH = 768
ROWS = 3072
BLOCKS = 4
DEPLOYED_NODES = 24
DENSE_REPLACED_MLP_FP16_BYTES = 113_246_208


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def deployment_accounting(
    *, width: int = WIDTH, rows: int = ROWS, deployed_nodes: int = DEPLOYED_NODES
) -> dict[str, int | float]:
    int8_bytes = width * width
    int4_values = width * width
    if int4_values % 2:
        raise ValueError("int4 frame must contain an even number of values")
    int4_bytes = int4_values // 2
    scale_values = 2 * width
    scale_bytes = 2 * scale_values
    diagonal_values = deployed_nodes * 2 * width
    diagonal_bytes = 2 * diagonal_values
    amplitude_values = deployed_nodes * rows
    amplitude_bytes = 2 * amplitude_values
    total = int8_bytes + int4_bytes + scale_bytes + diagonal_bytes + amplitude_bytes
    dense_bytes = deployed_nodes * rows * width * 2
    return {
        "int8_frame_code_bytes": int8_bytes,
        "int4_frame_code_bytes": int4_bytes,
        "fp16_frame_scale_values": scale_values,
        "fp16_frame_scale_bytes": scale_bytes,
        "fp16_node_diagonal_values": diagonal_values,
        "fp16_node_diagonal_bytes": diagonal_bytes,
        "fp16_node_row_amplitude_values": amplitude_values,
        "fp16_node_row_amplitude_bytes": amplitude_bytes,
        "total_checkpoint_bytes": total,
        "dense_replaced_mlp_fp16_bytes": dense_bytes,
        "checkpoint_byte_fraction": total / dense_bytes,
        "continuous_coordinate_values": scale_values
        + diagonal_values
        + amplitude_values,
        "continuous_coordinate_fraction": (
            scale_values + diagonal_values + amplitude_values
        )
        / (dense_bytes // 2),
        "persistent_pca_or_carrier_values": 0,
    }


def dct_ii_ortho_last(value: torch.Tensor) -> torch.Tensor:
    """Apply an orthonormal DCT-II along the final dimension using one FFT."""
    size = value.shape[-1]
    if size < 2:
        raise ValueError("DCT dimension must be at least two")
    reordered = torch.cat(
        (value[..., ::2], value[..., 1::2].flip(-1)), dim=-1
    )
    spectrum = torch.fft.fft(reordered, dim=-1)
    index = torch.arange(size, device=value.device, dtype=value.dtype)
    angle = -math.pi * index / (2.0 * size)
    transformed = spectrum.real * angle.cos() - spectrum.imag * angle.sin()
    transformed[..., 0] /= math.sqrt(size)
    transformed[..., 1:] /= math.sqrt(size / 2.0)
    return transformed


def dct_ii_ortho_rows(value: torch.Tensor) -> torch.Tensor:
    """Apply an orthonormal DCT-II to the penultimate (row) dimension."""
    return dct_ii_ortho_last(value.transpose(-2, -1)).transpose(-2, -1)


def _cpu_randperm(size: int, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    return torch.randperm(size, generator=generator).to(device)


def _cpu_signs(size: int, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    return (
        2 * torch.randint(0, 2, (size,), generator=generator, dtype=torch.int64) - 1
    ).to(device=device, dtype=torch.float32)


def make_carrier_geometry(
    *,
    nodes: int,
    width: int,
    branches: int,
    blocks: int,
    seed_base: int,
    node_stride: int,
    branch_stride: int,
    device: torch.device,
) -> tuple[tuple[tuple[dict[str, torch.Tensor], ...], ...], ...]:
    geometry = []
    for node in range(nodes):
        node_rows = []
        for branch in range(branches):
            block_rows = []
            for block in range(blocks):
                generator = torch.Generator(device="cpu")
                generator.manual_seed(
                    seed_base
                    + node * node_stride
                    + branch * branch_stride
                    + block * 1_000_003
                )
                block_rows.append(
                    {
                        "input_sign": _cpu_signs(width, generator, device),
                        "output_sign": _cpu_signs(width, generator, device),
                        "output_permutation": _cpu_randperm(width, generator, device),
                    }
                )
            node_rows.append(tuple(block_rows))
        geometry.append(tuple(node_rows))
    return tuple(geometry)


def apply_carrier(
    value: torch.Tensor,
    blocks: tuple[dict[str, torch.Tensor], ...],
) -> torch.Tensor:
    """Apply four procedural signed/permuted DCT blocks to batched matrices."""
    rows = []
    for geometry in blocks:
        transformed = dct_ii_ortho_rows(
            value * geometry["input_sign"][None, :, None]
        )
        transformed = transformed[:, geometry["output_permutation"], :]
        transformed = transformed * geometry["output_sign"][None, :, None]
        rows.append(transformed)
    return torch.cat(rows, dim=1)


def initial_orthogonal_frame(width: int, *, seed: int, device: torch.device) -> torch.Tensor:
    identity = torch.eye(width, device=device, dtype=torch.float32)
    frame = dct_ii_ortho_rows(identity.unsqueeze(0))[0]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    row_perm = _cpu_randperm(width, generator, device)
    column_perm = _cpu_randperm(width, generator, device)
    row_sign = _cpu_signs(width, generator, device)
    column_sign = _cpu_signs(width, generator, device)
    return (
        frame[row_perm][:, column_perm]
        * row_sign[:, None]
        * column_sign[None, :]
    ).contiguous()


def quantize_per_row(
    value: torch.Tensor, qmax: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale = value.detach().abs().amax(dim=1).clamp_min(1e-30) / qmax
    code = torch.round(value.detach() / scale[:, None]).clamp(-qmax, qmax).to(
        torch.int8
    )
    decoded = code.to(device=value.device, dtype=torch.float32) * scale[:, None]
    return code.cpu().contiguous(), scale.half().cpu().contiguous(), decoded


def pack_signed_int4(codes: torch.Tensor) -> torch.Tensor:
    flat = codes.detach().cpu().to(torch.int16).flatten()
    if bool(((flat < -7) | (flat > 7)).any()):
        raise ValueError("signed int4 code lies outside [-7,7]")
    encoded = (flat + 8).to(torch.uint8)
    if encoded.numel() % 2:
        encoded = torch.cat((encoded, torch.full((1,), 8, dtype=torch.uint8)))
    return (encoded[::2] | (encoded[1::2] << 4)).contiguous()


def unpack_signed_int4(packed: torch.Tensor, values: int) -> torch.Tensor:
    packed = packed.detach().cpu().to(torch.int16).flatten()
    decoded = torch.stack((packed & 15, (packed >> 4) & 15), dim=1).flatten()
    return (decoded[:values] - 8).to(torch.int8)


def generated_targets(
    frame_zero: torch.Tensor,
    frame_one: torch.Tensor,
    amplitude: torch.Tensor,
    diagonal: torch.Tensor,
    geometry: tuple[tuple[dict[str, torch.Tensor], ...], ...],
) -> tuple[torch.Tensor, ...]:
    predictions = []
    for node in range(amplitude.shape[0]):
        zero_input = diagonal[node, :, 0, :, None] * frame_zero[None, :, :]
        one_input = diagonal[node, :, 1, :, None] * frame_one[None, :, :]
        zero = apply_carrier(zero_input, geometry[node][0])
        one = apply_carrier(one_input, geometry[node][1])
        predictions.append(amplitude[node, :, :, None] * (zero + one))
    return tuple(predictions)


def capture_rows(
    predictions: tuple[torch.Tensor, ...],
    targets: tuple[torch.Tensor, ...],
    weights: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    objective = torch.zeros((), device=predictions[0].device)
    rows = []
    for index, (prediction, target, weight) in enumerate(
        zip(predictions, targets, weights, strict=True)
    ):
        flat_prediction = prediction.flatten(1)
        flat_target = target.flatten(1)
        numerator = (flat_prediction * flat_target).sum(dim=1).square()
        denominator = flat_prediction.square().sum(dim=1).clamp_min(1e-30)
        target_denominator = flat_target.square().sum(dim=1).clamp_min(1e-30)
        capture = (numerator / (denominator * target_denominator)).clamp(0.0, 1.0)
        weighted_capture = (capture * weight).sum()
        objective = objective + weighted_capture / len(predictions)
        rows.append(
            {
                "index": index,
                "weighted_top16_capture": float(weighted_capture.detach()),
                "minimum_pc_capture": float(capture.min().detach()),
                "median_pc_capture": float(capture.median().detach()),
                "maximum_pc_capture": float(capture.max().detach()),
                "component_captures": [float(value) for value in capture.detach()],
            }
        )
    return objective, rows


def initialize_coordinates(
    *,
    nodes: int,
    components: int,
    rows: int,
    width: int,
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
                nodes, components, 2, width, generator=generator, dtype=torch.float32
            )
            / math.sqrt(width)
        ).to(device)
    )
    return amplitude, diagonal


def fit_capacity(
    targets: tuple[torch.Tensor, ...],
    weights: tuple[torch.Tensor, ...],
    *,
    geometry: tuple[tuple[tuple[dict[str, torch.Tensor], ...], ...], ...],
    frame_zero: torch.Tensor,
    frame_one: torch.Tensor,
    steps: int,
    learning_rate: float,
    seed: int,
    learn_frames: bool,
    gradient_clip_norm: float,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    nodes = len(targets)
    components, rows, width = targets[0].shape
    amplitude, diagonal = initialize_coordinates(
        nodes=nodes,
        components=components,
        rows=rows,
        width=width,
        seed=seed,
        device=targets[0].device,
    )
    if learn_frames:
        frame_zero_value = torch.nn.Parameter(frame_zero.detach().clone())
        frame_one_value = torch.nn.Parameter(frame_one.detach().clone())
        parameters: list[torch.Tensor] = [
            frame_zero_value,
            frame_one_value,
            amplitude,
            diagonal,
        ]
    else:
        frame_zero_value = frame_zero.detach()
        frame_one_value = frame_one.detach()
        parameters = [amplitude, diagonal]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=0.0)
    history = []
    record = {0, 1, 2, 3, 7, 15, 31, 63, steps - 1}
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        predictions = generated_targets(
            frame_zero_value,
            frame_one_value,
            amplitude,
            diagonal,
            geometry,
        )
        objective, _ = capture_rows(predictions, targets, weights)
        (-objective).backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(parameters, gradient_clip_norm)
        )
        optimizer.step()
        if learn_frames:
            with torch.no_grad():
                frame_zero_value.div_(
                    frame_zero_value.norm(dim=1, keepdim=True).clamp_min(1e-20)
                )
                frame_one_value.div_(
                    frame_one_value.norm(dim=1, keepdim=True).clamp_min(1e-20)
                )
        if step in record:
            history.append(
                {
                    "step": step + 1,
                    "mean_weighted_capture": float(objective.detach()),
                    "gradient_norm": gradient_norm,
                }
            )
    with torch.no_grad():
        predictions = generated_targets(
            frame_zero_value,
            frame_one_value,
            amplitude,
            diagonal,
            geometry,
        )
        objective, rows_payload = capture_rows(predictions, targets, weights)
    return (
        {
            "mean_weighted_capture": float(objective),
            "rows": rows_payload,
            "history": history,
        },
        frame_zero_value.detach(),
        frame_one_value.detach(),
        amplitude.detach(),
        diagonal.detach(),
    )


def refine_coordinates(
    targets: tuple[torch.Tensor, ...],
    weights: tuple[torch.Tensor, ...],
    *,
    geometry: tuple[tuple[tuple[dict[str, torch.Tensor], ...], ...], ...],
    frame_zero: torch.Tensor,
    frame_one: torch.Tensor,
    amplitude: torch.Tensor,
    diagonal: torch.Tensor,
    steps: int,
    learning_rate: float,
    gradient_clip_norm: float,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    amplitude_parameter = torch.nn.Parameter(amplitude.detach().clone())
    diagonal_parameter = torch.nn.Parameter(diagonal.detach().clone())
    parameters = [amplitude_parameter, diagonal_parameter]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=0.0)
    history = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        predictions = generated_targets(
            frame_zero,
            frame_one,
            amplitude_parameter,
            diagonal_parameter,
            geometry,
        )
        objective, _ = capture_rows(predictions, targets, weights)
        (-objective).backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(parameters, gradient_clip_norm)
        )
        optimizer.step()
        if step in {0, 1, 3, 7, 15, 31, steps - 1}:
            history.append(
                {
                    "step": step + 1,
                    "mean_weighted_capture": float(objective.detach()),
                    "gradient_norm": gradient_norm,
                }
            )
    with torch.no_grad():
        predictions = generated_targets(
            frame_zero,
            frame_one,
            amplitude_parameter,
            diagonal_parameter,
            geometry,
        )
        objective, rows_payload = capture_rows(predictions, targets, weights)
    return (
        {
            "mean_weighted_capture": float(objective),
            "rows": rows_payload,
            "history": history,
        },
        amplitude_parameter.detach(),
        diagonal_parameter.detach(),
    )


def role_summary(rows: list[dict[str, Any]], indices: tuple[int, ...]) -> dict[str, float]:
    weighted = [rows[index]["weighted_top16_capture"] for index in indices]
    minimum = [rows[index]["minimum_pc_capture"] for index in indices]
    return {
        "minimum_weighted_top16_capture": min(weighted),
        "median_weighted_top16_capture": statistics.median(weighted),
        "maximum_weighted_top16_capture": max(weighted),
        "minimum_pc_capture": min(minimum),
    }


def evaluate_with_coordinates(
    targets: tuple[torch.Tensor, ...],
    weights: tuple[torch.Tensor, ...],
    *,
    geometry: tuple[tuple[tuple[dict[str, torch.Tensor], ...], ...], ...],
    frame_zero: torch.Tensor,
    frame_one: torch.Tensor,
    amplitude: torch.Tensor,
    diagonal: torch.Tensor,
) -> dict[str, Any]:
    with torch.no_grad():
        predictions = generated_targets(
            frame_zero, frame_one, amplitude, diagonal, geometry
        )
        objective, rows = capture_rows(predictions, targets, weights)
    return {
        "mean_weighted_capture": float(objective),
        "rows": rows,
        "role_summaries": {
            "c_fc": role_summary(rows, (0, 2, 4)),
            "c_proj": role_summary(rows, (1, 3, 5)),
        },
    }


def checkpoint_payload(
    frame_zero_codes: torch.Tensor,
    frame_one_codes: torch.Tensor,
    frame_zero_scales: torch.Tensor,
    frame_one_scales: torch.Tensor,
    *,
    width: int = WIDTH,
    rows: int = ROWS,
    deployed_nodes: int = DEPLOYED_NODES,
) -> bytes:
    coordinates = torch.zeros(
        deployed_nodes * (rows + 2 * width), dtype=torch.float16
    )
    chunks = [
        frame_zero_codes.detach().cpu().contiguous().numpy().tobytes(),
        pack_signed_int4(frame_one_codes).numpy().tobytes(),
        frame_zero_scales.detach().cpu().contiguous().numpy().tobytes(),
        frame_one_scales.detach().cpu().contiguous().numpy().tobytes(),
        coordinates.numpy().tobytes(),
    ]
    payload = b"".join(chunks)
    expected = deployment_accounting(
        width=width, rows=rows, deployed_nodes=deployed_nodes
    )["total_checkpoint_bytes"]
    if len(payload) != expected:
        raise ValueError(f"checkpoint has {len(payload)} bytes, expected {expected}")
    return payload


def self_test(device_name: str = "cpu") -> dict[str, Any]:
    device = torch.device(device_name)
    width = 8
    rows = 4 * width
    identity = torch.eye(width, device=device).unsqueeze(0)
    dct = dct_ii_ortho_rows(identity)[0]
    dct_check = dct.detach().double().cpu()
    orthogonality_error = float(
        (dct_check @ dct_check.T - torch.eye(width, dtype=torch.float64))
        .abs()
        .max()
    )
    geometry = make_carrier_geometry(
        nodes=2,
        width=width,
        branches=2,
        blocks=4,
        seed_base=73,
        node_stride=11,
        branch_stride=101,
        device=device,
    )
    sample = torch.randn(3, width, width, device=device)
    transformed = apply_carrier(sample, geometry[0][0])
    norm_ratio = float(transformed.square().sum() / sample.square().sum())
    codes = torch.arange(-7, 8, dtype=torch.int8).repeat(3)
    restored = unpack_signed_int4(pack_signed_int4(codes), codes.numel())
    accounting = deployment_accounting()
    if orthogonality_error > 2e-5:
        raise AssertionError({"orthogonality_error": orthogonality_error})
    if abs(norm_ratio - 4.0) > 2e-4:
        raise AssertionError({"carrier_norm_ratio": norm_ratio})
    if not torch.equal(codes, restored):
        raise AssertionError("signed int4 roundtrip failed")
    if accounting["total_checkpoint_bytes"] != 1_108_992:
        raise AssertionError(accounting)
    if accounting["dense_replaced_mlp_fp16_bytes"] != DENSE_REPLACED_MLP_FP16_BYTES:
        raise AssertionError(accounting)
    return {
        "status": "passed",
        "dct_orthogonality_max_error": orthogonality_error,
        "four_block_norm_ratio": norm_ratio,
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
        raise ValueError("unexpected H51a plan schema")
    accounting = deployment_accounting()
    if accounting != plan["exact_deployment_accounting"]:
        raise ValueError({"computed": accounting, "planned": plan["exact_deployment_accounting"]})
    if accounting["checkpoint_byte_fraction"] > 0.01:
        raise ValueError("H51 exceeds one-percent checkpoint budget")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device.index or 0)
        torch.backends.cuda.matmul.allow_tf32 = True

    fit_plan = plan["optimistic_capacity_fit"]
    system_plan = plan["systems_preflight"]
    components = (
        int(system_plan["preflight_components_per_node"])
        if args.preflight
        else int(plan["frozen_inventory"]["components_per_node"])
    )
    joint_steps = (
        int(system_plan["preflight_joint_steps"])
        if args.preflight
        else int(fit_plan["joint_fp32_steps"])
    )
    local_steps = (
        int(system_plan["preflight_local_steps"])
        if args.preflight
        else int(fit_plan["post_quantization_local_only_steps"])
    )
    targets, weights, inventory, _ = load_node_pc_inventory(
        args.trajectory_dir, components=components, device=args.device
    )
    decoder = plan["frozen_decoder"]
    geometry = make_carrier_geometry(
        nodes=len(targets),
        width=WIDTH,
        branches=2,
        blocks=BLOCKS,
        seed_base=int(decoder["carrier_seed_base"]),
        node_stride=int(decoder["carrier_node_stride"]),
        branch_stride=int(decoder["carrier_branch_stride"]),
        device=device,
    )
    initial_zero = initial_orthogonal_frame(WIDTH, seed=202608511, device=device)
    initial_one = initial_orthogonal_frame(WIDTH, seed=202608512, device=device)
    fitted, learned_zero, learned_one, amplitude, diagonal = fit_capacity(
        targets,
        weights,
        geometry=geometry,
        frame_zero=initial_zero,
        frame_one=initial_one,
        steps=joint_steps,
        learning_rate=float(fit_plan["joint_learning_rate"]),
        seed=202608513,
        learn_frames=True,
        gradient_clip_norm=float(fit_plan["gradient_clip_norm"]),
    )
    fitted["role_summaries"] = {
        "c_fc": role_summary(fitted["rows"], (0, 2, 4)),
        "c_proj": role_summary(fitted["rows"], (1, 3, 5)),
    }
    code_zero, scale_zero, quantized_zero = quantize_per_row(learned_zero, 127)
    code_one, scale_one, quantized_one = quantize_per_row(learned_one, 7)
    quantized, quantized_amplitude, quantized_diagonal = refine_coordinates(
        targets,
        weights,
        geometry=geometry,
        frame_zero=quantized_zero,
        frame_one=quantized_one,
        amplitude=amplitude,
        diagonal=diagonal,
        steps=local_steps,
        learning_rate=float(fit_plan["post_quantization_learning_rate"]),
        gradient_clip_norm=float(fit_plan["gradient_clip_norm"]),
    )
    quantized["role_summaries"] = {
        "c_fc": role_summary(quantized["rows"], (0, 2, 4)),
        "c_proj": role_summary(quantized["rows"], (1, 3, 5)),
    }
    single_frame = evaluate_with_coordinates(
        targets,
        weights,
        geometry=geometry,
        frame_zero=quantized_zero,
        frame_one=torch.zeros_like(quantized_one),
        amplitude=quantized_amplitude,
        diagonal=quantized_diagonal,
    )
    procedural, _, _, _, _ = fit_capacity(
        targets,
        weights,
        geometry=geometry,
        frame_zero=initial_zero,
        frame_one=initial_one,
        steps=joint_steps + local_steps,
        learning_rate=float(fit_plan["joint_learning_rate"]),
        seed=202608514,
        learn_frames=False,
        gradient_clip_norm=float(fit_plan["gradient_clip_norm"]),
    )
    procedural["role_summaries"] = {
        "c_fc": role_summary(procedural["rows"], (0, 2, 4)),
        "c_proj": role_summary(procedural["rows"], (1, 3, 5)),
    }

    margins = [
        quantized["rows"][index]["weighted_top16_capture"]
        - procedural["rows"][index]["weighted_top16_capture"]
        for index in range(len(targets))
    ]
    gates = plan["capacity_gates"]
    weighted_every_node = all(
        row["weighted_top16_capture"]
        >= float(gates["quantized_weighted_top16_capture_min_every_node"])
        for row in quantized["rows"]
    )
    role_medians = all(
        summary["median_weighted_top16_capture"]
        >= float(gates["quantized_weighted_top16_capture_median_each_role"])
        for summary in quantized["role_summaries"].values()
    )
    pc_minimum = all(
        row["minimum_pc_capture"]
        >= float(gates["quantized_minimum_pc_capture_every_node"])
        for row in quantized["rows"]
    )
    procedural_margin = all(
        margin
        >= float(gates["quantized_minus_procedural_weighted_capture_min_every_node"])
        for margin in margins
    )
    finite = all(
        math.isfinite(row["weighted_top16_capture"])
        and math.isfinite(row["minimum_pc_capture"])
        for row in quantized["rows"]
    )
    retained = (
        (not args.preflight)
        and weighted_every_node
        and role_medians
        and pc_minimum
        and procedural_margin
        and finite
    )
    gate = {
        "classification": "PREFLIGHT" if args.preflight else ("RETAINED" if retained else "REJECTED"),
        "retained": retained,
        "weighted_capture_every_node_pass": weighted_every_node,
        "role_median_pass": role_medians,
        "minimum_pc_pass": pc_minimum,
        "procedural_margin_pass": procedural_margin,
        "finite_pass": finite,
        "per_node_procedural_margins": margins,
    }

    payload = checkpoint_payload(code_zero, code_one, scale_zero, scale_one)
    checkpoint_path = args.output / "compact_checkpoint.bin"
    checkpoint_path.write_bytes(payload)
    accounting_path = args.output / "accounting.json"
    write_json(accounting_path, accounting)
    metrics = {
        "unquantized_fitted": fitted,
        "quantized_local_refined": quantized,
        "single_int8_frame": single_frame,
        "procedural_unfitted_frames": procedural,
        "gate": gate,
        "quantization": {
            "int8_relative_frobenius_error": float(
                (quantized_zero - learned_zero).norm() / learned_zero.norm()
            ),
            "int4_relative_frobenius_error": float(
                (quantized_one - learned_one).norm() / learned_one.norm()
            ),
        },
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
        * int(fit_plan["joint_fp32_steps"] + fit_plan["post_quantization_local_only_steps"])
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
        "frame_hashes": {
            "int8_codes": tensor_sha256(code_zero),
            "int8_scales": tensor_sha256(scale_zero),
            "int4_codes": tensor_sha256(code_one),
            "int4_scales": tensor_sha256(scale_one),
        },
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
        },
        "limitations": [
            "This optimistic all-PC fit is a capacity ceiling, not chronological transfer.",
            "Per-PC nuisance coordinates are alternate manifold points and are not stored together.",
            "The offline FP32 frame fit is acquisition cost, not latent-only training.",
            "No function, CE, or scale result is produced by H51a.",
        ],
    }
    metadata_path = args.output / "metadata.json"
    write_json(metadata_path, metadata)
    print(
        json.dumps(
            {
                "classification": gate["classification"],
                "metadata": str(metadata_path),
                "quantized_mean_weighted_capture": quantized["mean_weighted_capture"],
                "quantized_role_summaries": quantized["role_summaries"],
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
