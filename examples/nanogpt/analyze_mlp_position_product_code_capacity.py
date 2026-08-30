#!/usr/bin/env python3
"""H55a capacity gate for a position-conditioned subrow product code."""

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


SCHEMA_VERSION = "nanogpt_mlp_position_product_code_capacity_v1"
PLAN_SCHEMA_VERSION = "nanogpt_mlp_position_product_code_capacity_plan_v1"
BLOCK_WIDTH = 32
POSITIONS = WIDTH // BLOCK_WIDTH
CODEWORDS = 32
CODE_BITS = 5


def deployment_accounting(
    *,
    block_width: int = BLOCK_WIDTH,
    positions: int = POSITIONS,
    codewords: int = CODEWORDS,
    code_bits: int = CODE_BITS,
    rows: int = ROWS,
    deployed_nodes: int = DEPLOYED_NODES,
) -> dict[str, int | float]:
    if positions * block_width != WIDTH:
        raise ValueError("product-code blocks must cover the canonical width")
    if codewords > 2**code_bits:
        raise ValueError("code width cannot address the requested codewords")
    dense_bytes = deployed_nodes * rows * WIDTH * 2
    deployed_blocks = deployed_nodes * rows * positions
    private_bits = deployed_blocks * code_bits
    if private_bits % 8:
        raise ValueError("packed product codes must be byte aligned")
    private_bytes = private_bits // 8
    dictionary_values = positions * codewords * block_width
    dictionary_bits = dictionary_values * 4
    if dictionary_bits % 8:
        raise ValueError("int4 dictionary must be byte aligned")
    dictionary_bytes = dictionary_bits // 8
    scale_values = positions * codewords
    scale_bytes = scale_values * 2
    total = private_bytes + dictionary_bytes + scale_bytes
    return {
        "dense_replaced_mlp_fp16_bytes": dense_bytes,
        "deployed_block_values": deployed_blocks,
        "private_code_bits": private_bits,
        "private_code_bytes": private_bytes,
        "int4_dictionary_values": dictionary_values,
        "int4_dictionary_bytes": dictionary_bytes,
        "fp16_dictionary_scale_values": scale_values,
        "fp16_dictionary_scale_bytes": scale_bytes,
        "total_checkpoint_bytes": total,
        "checkpoint_byte_fraction": total / dense_bytes,
        "persistent_pca_or_per_node_basis_values": 0,
    }


def pack_unsigned_codes(codes: torch.Tensor, *, bits: int) -> torch.Tensor:
    """Pack unsigned fixed-width codes without logical per-code padding."""
    flat = codes.detach().cpu().to(torch.int64).flatten()
    if bool(((flat < 0) | (flat >= 2**bits)).any()):
        raise ValueError("code outside fixed-width range")
    total_bits = flat.numel() * bits
    packed = torch.zeros((total_bits + 7) // 8, dtype=torch.uint8)
    for shift in range(bits):
        bit_positions = torch.arange(flat.numel(), dtype=torch.int64) * bits + shift
        values = ((flat >> shift) & 1).to(torch.uint8)
        byte_positions = bit_positions // 8
        bit_offsets = bit_positions % 8
        packed.index_add_(
            0, byte_positions, (values << bit_offsets).to(torch.uint8)
        )
    return packed


def unpack_unsigned_codes(
    packed: torch.Tensor, *, values: int, bits: int
) -> torch.Tensor:
    packed = packed.detach().cpu().to(torch.int64).flatten()
    result = torch.zeros(values, dtype=torch.int64)
    for shift in range(bits):
        bit_positions = torch.arange(values, dtype=torch.int64) * bits + shift
        result |= ((packed[bit_positions // 8] >> (bit_positions % 8)) & 1) << shift
    return result


def _stack_target_blocks(
    targets: tuple[torch.Tensor, ...], *, block_width: int
) -> torch.Tensor:
    stacked = torch.cat(targets, dim=0)
    return stacked.reshape(stacked.shape[0], ROWS, WIDTH // block_width, block_width)


def initialize_position_codebook(
    target_blocks: torch.Tensor,
    *,
    codewords: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    targets, _, positions, width = target_blocks.shape
    target_order = torch.randperm(targets, generator=generator)
    rows = []
    for position in range(positions):
        entries = []
        for code in range(codewords):
            target_index = int(target_order[(position * codewords + code) % targets])
            energy = (
                target_blocks[target_index, :, position]
                .square()
                .sum(dim=1)
                .detach()
                .cpu()
            )
            row = int(
                torch.multinomial(
                    energy.clamp_min(1e-30),
                    1,
                    replacement=False,
                    generator=generator,
                )
            )
            entries.append(target_blocks[target_index, row, position])
        rows.append(torch.stack(entries))
    result = torch.stack(rows)
    if result.shape != (positions, codewords, width):
        raise AssertionError(result.shape)
    return result.contiguous()


def initialize_global_codebook(
    target_blocks: torch.Tensor,
    *,
    codewords: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    targets, _, positions, _ = target_blocks.shape
    entries = []
    for code in range(codewords):
        target_index = int(torch.randint(targets, (1,), generator=generator))
        position = int(torch.randint(positions, (1,), generator=generator))
        energy = (
            target_blocks[target_index, :, position]
            .square()
            .sum(dim=1)
            .detach()
            .cpu()
        )
        row = int(
            torch.multinomial(
                energy.clamp_min(1e-30),
                1,
                replacement=False,
                generator=generator,
            )
        )
        entries.append(target_blocks[target_index, row, position])
    return torch.stack(entries)[None].contiguous()


def make_random_codebook(
    initial: torch.Tensor, *, seed: int
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    value = torch.randn(initial.shape, generator=generator).to(initial.device)
    mean_norm = initial.norm(dim=2).mean(dim=1, keepdim=True)
    return value / value.norm(dim=2, keepdim=True).clamp_min(1e-30) * mean_norm[:, :, None]


def _sample_blocks(
    target_blocks: torch.Tensor,
    energy_cpu: torch.Tensor,
    *,
    batch_blocks: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    samples_per_pair = 4
    if batch_blocks % samples_per_pair:
        raise ValueError("batch_blocks must be divisible by four")
    pair_count = batch_blocks // samples_per_pair
    target_count, _, position_count, _ = target_blocks.shape
    target_indices = torch.randint(
        target_count, (pair_count,), generator=generator
    )
    positions = torch.randint(
        position_count, (pair_count,), generator=generator
    )
    pair_energy = energy_cpu[target_indices, :, positions]
    row_indices = torch.multinomial(
        pair_energy.clamp_min(1e-30),
        samples_per_pair,
        replacement=True,
        generator=generator,
    )
    target_indices = target_indices[:, None].expand(-1, samples_per_pair).reshape(-1)
    positions = positions[:, None].expand(-1, samples_per_pair).reshape(-1)
    row_indices = row_indices.reshape(-1)
    device = target_blocks.device
    blocks = target_blocks[
        target_indices.to(device), row_indices.to(device), positions.to(device)
    ]
    return blocks, positions.to(device)


def assign_codewords(
    blocks: torch.Tensor,
    positions: torch.Tensor,
    codebook: torch.Tensor,
    *,
    position_conditioned: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    lookup_positions = positions if position_conditioned else torch.zeros_like(positions)
    selected = codebook[lookup_positions]
    distance = (
        blocks.square().sum(dim=1, keepdim=True)
        + selected.square().sum(dim=2)
        - 2.0 * torch.einsum("nd,nkd->nk", blocks, selected)
    )
    codes = distance.argmin(dim=1)
    prediction = selected[torch.arange(blocks.shape[0], device=blocks.device), codes]
    return prediction, codes


def fit_codebook(
    target_blocks: torch.Tensor,
    *,
    initial: torch.Tensor,
    steps: int,
    batch_blocks: int,
    ema_coefficient: float,
    seed: int,
    position_conditioned: bool,
    progress_callback: Any | None = None,
) -> tuple[torch.Tensor, list[dict[str, float | int]]]:
    value = initial.detach().clone()
    energy_cpu = target_blocks.square().sum(dim=3).detach().cpu()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    history = []
    record_steps = {0, 1, 2, 3, 7, 15, 31, 63, 127, 191, 255, steps - 1}
    for step in range(steps):
        blocks, positions = _sample_blocks(
            target_blocks,
            energy_cpu,
            batch_blocks=batch_blocks,
            generator=generator,
        )
        prediction, codes = assign_codewords(
            blocks,
            positions,
            value,
            position_conditioned=position_conditioned,
        )
        lookup_positions = positions if position_conditioned else torch.zeros_like(positions)
        combined = lookup_positions * value.shape[1] + codes
        flat_size = value.shape[0] * value.shape[1]
        sums = torch.zeros(
            flat_size, value.shape[2], device=value.device, dtype=value.dtype
        )
        counts = torch.zeros(flat_size, device=value.device, dtype=value.dtype)
        sums.index_add_(0, combined, blocks)
        counts.index_add_(0, combined, torch.ones_like(combined, dtype=value.dtype))
        active = counts > 0
        flat = value.reshape(flat_size, value.shape[2])
        means = sums[active] / counts[active, None]
        flat[active] = (1.0 - ema_coefficient) * flat[active] + ema_coefficient * means
        if progress_callback is not None:
            progress_callback(step + 1, steps)
        if step in record_steps:
            residual = prediction - blocks
            history.append(
                {
                    "step": step + 1,
                    "sample_relative_squared_error": float(
                        residual.square().sum() / blocks.square().sum().clamp_min(1e-30)
                    ),
                    "sample_capture": float(
                        (blocks.flatten() @ prediction.flatten()).square()
                        / (
                            blocks.square().sum()
                            * prediction.square().sum()
                        ).clamp_min(1e-30)
                    ),
                    "active_code_fraction": float(active.float().mean()),
                }
            )
    return value, history


def make_sign_carrier(
    *, positions: int, width: int, seed: int, device: torch.device
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    signs = 2 * torch.randint(0, 2, (positions, width), generator=generator) - 1
    return signs.to(device=device, dtype=torch.float32) / math.sqrt(width)


def fit_scalar_sign_codebook(
    target_blocks: torch.Tensor,
    *,
    signs: torch.Tensor,
    initial: torch.Tensor,
    steps: int,
    batch_blocks: int,
    ema_coefficient: float,
    seed: int,
    progress_callback: Any | None = None,
) -> tuple[torch.Tensor, list[dict[str, float | int]]]:
    levels = torch.einsum("pkd,pd->pk", initial, signs).detach().clone()
    energy_cpu = target_blocks.square().sum(dim=3).detach().cpu()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    history = []
    record_steps = {0, 1, 2, 3, 7, 15, 31, 63, 127, 191, 255, steps - 1}
    for step in range(steps):
        blocks, positions = _sample_blocks(
            target_blocks,
            energy_cpu,
            batch_blocks=batch_blocks,
            generator=generator,
        )
        projection = (blocks * signs[positions]).sum(dim=1)
        codes = (projection[:, None] - levels[positions]).square().argmin(dim=1)
        combined = positions * levels.shape[1] + codes
        flat_size = levels.numel()
        sums = torch.zeros(flat_size, device=levels.device)
        counts = torch.zeros(flat_size, device=levels.device)
        sums.index_add_(0, combined, projection)
        counts.index_add_(0, combined, torch.ones_like(projection))
        active = counts > 0
        flat = levels.flatten()
        flat[active] = (
            (1.0 - ema_coefficient) * flat[active]
            + ema_coefficient * sums[active] / counts[active]
        )
        if progress_callback is not None:
            progress_callback(step + 1, steps)
        if step in record_steps:
            prediction = levels[positions, codes, None] * signs[positions]
            history.append(
                {
                    "step": step + 1,
                    "sample_relative_squared_error": float(
                        (prediction - blocks).square().sum()
                        / blocks.square().sum().clamp_min(1e-30)
                    ),
                    "active_code_fraction": float(active.float().mean()),
                }
            )
    return levels[:, :, None] * signs[:, None, :], history


def _squared_cosine(target: torch.Tensor, prediction: torch.Tensor) -> float:
    target = target.flatten()
    prediction = prediction.flatten()
    denominator = (
        target.square().sum() * prediction.square().sum()
    ).clamp_min(1e-30)
    return float((target @ prediction).square() / denominator)


def evaluate_codebook(
    targets: tuple[torch.Tensor, ...],
    weights: tuple[torch.Tensor, ...],
    *,
    codebook: torch.Tensor,
    block_width: int,
    block_batch: int,
    position_conditioned: bool,
) -> dict[str, Any]:
    rows_out = []
    utilization = torch.zeros(
        codebook.shape[:2], device=codebook.device, dtype=torch.int64
    )
    energy_sums = torch.zeros(4, device=codebook.device, dtype=torch.float64)
    position_sums = torch.zeros(POSITIONS, device=codebook.device, dtype=torch.float64)
    strata_count = 0
    positions = torch.arange(POSITIONS, device=codebook.device).repeat(ROWS)
    with torch.no_grad():
        for node, (target_components, weight) in enumerate(
            zip(targets, weights, strict=True)
        ):
            captures = []
            for target in target_components:
                blocks = target.reshape(ROWS, POSITIONS, block_width)
                flat_blocks = blocks.reshape(-1, block_width)
                predictions = []
                code_rows = []
                for start in range(0, flat_blocks.shape[0], block_batch):
                    prediction, codes = assign_codewords(
                        flat_blocks[start : start + block_batch],
                        positions[start : start + block_batch],
                        codebook,
                        position_conditioned=position_conditioned,
                    )
                    predictions.append(prediction)
                    code_rows.append(codes)
                prediction = torch.cat(predictions).reshape_as(blocks)
                codes = torch.cat(code_rows)
                captures.append(_squared_cosine(blocks, prediction))
                lookup_positions = (
                    positions if position_conditioned else torch.zeros_like(positions)
                )
                combined = lookup_positions * codebook.shape[1] + codes
                utilization += torch.bincount(
                    combined, minlength=utilization.numel()
                ).reshape_as(utilization)
                block_energy = blocks.square().sum(dim=2).flatten()
                order = block_energy.argsort()
                flat_prediction = prediction.reshape(-1, block_width)
                for quartile, group in enumerate(torch.tensor_split(order, 4)):
                    energy_sums[quartile] += _squared_cosine(
                        flat_blocks[group], flat_prediction[group]
                    )
                for position in range(POSITIONS):
                    position_sums[position] += _squared_cosine(
                        blocks[:, position], prediction[:, position]
                    )
                strata_count += 1
            capture = torch.tensor(captures, device=weight.device)
            rows_out.append(
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
    active_fractions = []
    entropies = []
    maximum_loads = []
    for position in range(utilization.shape[0]):
        counts = utilization[position].double()
        probabilities = counts / counts.sum().clamp_min(1)
        nonzero = probabilities > 0
        active_fractions.append(float(nonzero.float().mean()))
        entropies.append(
            float(
                -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
                / math.log(codebook.shape[1])
            )
        )
        maximum_loads.append(float(probabilities.max()))
    return {
        "mean_weighted_capture": sum(
            row["weighted_top16_capture"] for row in rows_out
        )
        / len(rows_out),
        "rows": rows_out,
        "role_summaries": {
            "c_fc": role_summary(rows_out, (0, 2, 4)),
            "c_proj": role_summary(rows_out, (1, 3, 5)),
        },
        "code_utilization": {
            "mean_active_fraction": sum(active_fractions) / len(active_fractions),
            "minimum_active_fraction": min(active_fractions),
            "mean_normalized_entropy": sum(entropies) / len(entropies),
            "minimum_normalized_entropy": min(entropies),
            "maximum_load_fraction": max(maximum_loads),
            "assignment_count": int(utilization.sum()),
        },
        "mean_capture_by_block_energy_quartile": [
            float(value) for value in energy_sums / max(1, strata_count)
        ],
        "mean_capture_by_block_position": [
            float(value) for value in position_sums / max(1, strata_count)
        ],
    }


def self_test(device_name: str = "cpu") -> dict[str, Any]:
    device = torch.device(device_name)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(55)
    positions, codewords, width = 4, 8, 4
    codebook = torch.randn(
        positions, codewords, width, generator=generator
    ).to(device)
    codes = torch.randint(0, codewords, (2, 7, positions), generator=generator)
    decoded = codebook[
        torch.arange(positions, device=device)[None, None], codes.to(device)
    ]
    own_capture = _squared_cosine(decoded, decoded)
    packed = pack_unsigned_codes(codes, bits=3)
    unpacked = unpack_unsigned_codes(
        packed, values=codes.numel(), bits=3
    ).reshape_as(codes)
    accounting = deployment_accounting()
    if not torch.equal(unpacked, codes):
        raise AssertionError("fixed-width pack round trip failed")
    if own_capture < 0.999999:
        raise AssertionError(own_capture)
    if accounting["total_checkpoint_bytes"] != 1_119_744:
        raise AssertionError(accounting)
    if accounting["dense_replaced_mlp_fp16_bytes"] != DENSE_REPLACED_MLP_FP16_BYTES:
        raise AssertionError(accounting)
    return {
        "status": "passed",
        "synthetic_own_family_capture": own_capture,
        "packed_roundtrip": True,
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
        raise ValueError("unexpected H55a plan schema")
    decoder = plan["frozen_decoder"]
    accounting = deployment_accounting(
        block_width=int(decoder["block_width"]),
        positions=int(decoder["block_positions"]),
        codewords=int(decoder["codewords_per_position"]),
        code_bits=int(decoder["private_code_bits_per_block"]),
    )
    if accounting != plan["exact_deployment_accounting"]:
        raise ValueError({"computed": accounting, "planned": plan["exact_deployment_accounting"]})
    if accounting["checkpoint_byte_fraction"] > 0.01:
        raise ValueError("H55 exceeds one-percent checkpoint budget")
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device.index or 0)
        torch.backends.cuda.matmul.allow_tf32 = True
    started = time.time()

    fit_plan = plan["unquantized_capacity_fit"]
    systems = plan["systems_preflight"]
    components = (
        int(systems["preflight_components_per_node"])
        if args.preflight
        else int(plan["frozen_inventory"]["components_per_node"])
    )
    steps = int(systems["preflight_steps"]) if args.preflight else int(fit_plan["steps"])
    batch_blocks = (
        int(systems["preflight_batch_blocks"])
        if args.preflight
        else int(fit_plan["batch_blocks"])
    )
    targets, weights, inventory, _ = load_node_pc_inventory(
        args.trajectory_dir, components=components, device=args.device
    )
    if inventory["trajectory_identity_sha256"] != plan["frozen_inventory"]["trajectory_identity_sha256"]:
        raise ValueError("H55 trajectory identity mismatch")
    target_blocks = _stack_target_blocks(
        targets, block_width=int(decoder["block_width"])
    )
    progress_path = args.output / "progress.json"
    total_updates = 3 * steps
    completed_updates = 0

    def progress(stage: str, step: int, total: int) -> None:
        nonlocal completed_updates
        stage_offset = {
            "position_codebook_fit": 0,
            "global_codebook_control_fit": steps,
            "scalar_sign_control_fit": 2 * steps,
        }[stage]
        completed_updates = stage_offset + step
        write_json(
            progress_path,
            {
                "schema_version": f"{SCHEMA_VERSION}_progress_v1",
                "stage": stage,
                "stage_step": step,
                "stage_steps": total,
                "completed_updates": completed_updates,
                "total_updates": total_updates,
                "fraction": completed_updates / total_updates,
            },
        )

    progress("position_codebook_fit", 0, steps)
    seed = int(fit_plan["seed"])
    initial = initialize_position_codebook(
        target_blocks,
        codewords=int(decoder["codewords_per_position"]),
        seed=seed,
    )
    candidate_codebook, candidate_history = fit_codebook(
        target_blocks,
        initial=initial,
        steps=steps,
        batch_blocks=batch_blocks,
        ema_coefficient=0.25,
        seed=seed + 1,
        position_conditioned=True,
        progress_callback=lambda step, total: progress(
            "position_codebook_fit", step, total
        ),
    )
    global_initial = initialize_global_codebook(
        target_blocks,
        codewords=int(decoder["codewords_per_position"]),
        seed=seed + 2,
    )
    global_codebook, global_history = fit_codebook(
        target_blocks,
        initial=global_initial,
        steps=steps,
        batch_blocks=batch_blocks,
        ema_coefficient=0.25,
        seed=seed + 3,
        position_conditioned=False,
        progress_callback=lambda step, total: progress(
            "global_codebook_control_fit", step, total
        ),
    )
    signs = make_sign_carrier(
        positions=POSITIONS,
        width=BLOCK_WIDTH,
        seed=seed + 4,
        device=device,
    )
    scalar_codebook, scalar_history = fit_scalar_sign_codebook(
        target_blocks,
        signs=signs,
        initial=initial,
        steps=steps,
        batch_blocks=batch_blocks,
        ema_coefficient=0.25,
        seed=seed + 5,
        progress_callback=lambda step, total: progress(
            "scalar_sign_control_fit", step, total
        ),
    )
    block_batch = int(fit_plan["evaluation_block_batch"])
    candidate = evaluate_codebook(
        targets,
        weights,
        codebook=candidate_codebook,
        block_width=BLOCK_WIDTH,
        block_batch=block_batch,
        position_conditioned=True,
    )
    initial_control = evaluate_codebook(
        targets,
        weights,
        codebook=initial,
        block_width=BLOCK_WIDTH,
        block_batch=block_batch,
        position_conditioned=True,
    )
    random_control = evaluate_codebook(
        targets,
        weights,
        codebook=make_random_codebook(initial, seed=seed + 6),
        block_width=BLOCK_WIDTH,
        block_batch=block_batch,
        position_conditioned=True,
    )
    global_control = evaluate_codebook(
        targets,
        weights,
        codebook=global_codebook,
        block_width=BLOCK_WIDTH,
        block_batch=block_batch,
        position_conditioned=False,
    )
    scalar_control = evaluate_codebook(
        targets,
        weights,
        codebook=scalar_codebook,
        block_width=BLOCK_WIDTH,
        block_batch=block_batch,
        position_conditioned=True,
    )
    candidate["history"] = candidate_history
    global_control["history"] = global_history
    scalar_control["history"] = scalar_history
    gaussian_ceiling = float(
        plan["rate_distortion_reference"]["isotropic_gaussian_capture_ceiling"]
    )
    candidate["rate_efficiency_vs_gaussian_ceiling"] = (
        candidate["mean_weighted_capture"] / gaussian_ceiling
    )

    gates = plan["capacity_gates"]
    weighted_pass = all(
        row["weighted_top16_capture"]
        >= float(gates["unquantized_weighted_top16_capture_min_every_node"])
        for row in candidate["rows"]
    )
    role_pass = all(
        summary["median_weighted_top16_capture"]
        >= float(gates["unquantized_weighted_top16_capture_median_each_role"])
        for summary in candidate["role_summaries"].values()
    )
    minimum_pass = all(
        row["minimum_pc_capture"]
        >= float(gates["unquantized_minimum_pc_capture_every_node"])
        for row in candidate["rows"]
    )
    finite_pass = all(
        math.isfinite(row["weighted_top16_capture"])
        and math.isfinite(row["minimum_pc_capture"])
        for row in candidate["rows"]
    )
    capacity_pass = weighted_pass and role_pass and minimum_pass and finite_pass
    classification = (
        "PREFLIGHT"
        if args.preflight
        else (
            "UNQUANTIZED_PASSED_INT4_PENDING"
            if capacity_pass
            else "UNQUANTIZED_REJECTED"
        )
    )
    gate = {
        "classification": classification,
        "unquantized_capacity_pass": capacity_pass,
        "weighted_capture_every_node_pass": weighted_pass,
        "role_median_pass": role_pass,
        "minimum_pc_every_node_pass": minimum_pass,
        "finite_pass": finite_pass,
        "int4_stage_authorized": (not args.preflight) and capacity_pass,
    }
    accounting_path = args.output / "accounting.json"
    write_json(accounting_path, accounting)
    metrics = {
        "position_conditioned_product_code": candidate,
        "fixed_initial_product_code": initial_control,
        "fixed_random_product_code": random_control,
        "global_unpositioned_product_code": global_control,
        "equal_rate_scalar_sign": scalar_control,
        "gate": gate,
        "candidate_codebook_sha256": tensor_sha256(candidate_codebook),
        "initial_codebook_sha256": tensor_sha256(initial),
        "global_codebook_sha256": tensor_sha256(global_codebook),
        "scalar_codebook_sha256": tensor_sha256(scalar_codebook),
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
        * int(fit_plan["batch_blocks"])
        / batch_blocks
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
            "This all-PC FP32 fit is a necessary product-code capacity gate, not a compact checkpoint.",
            "Per-PC hard codes are alternative manifold points and are not stored simultaneously.",
            "No int4 candidate is fitted unless every frozen unquantized gate passes.",
            "No function, CE, attention, or scale result is produced by H55a.",
        ],
    }
    metadata_path = args.output / "metadata.json"
    write_json(metadata_path, metadata)
    print(
        json.dumps(
            {
                "classification": classification,
                "metadata": str(metadata_path),
                "candidate_mean_weighted_capture": candidate["mean_weighted_capture"],
                "candidate_role_summaries": candidate["role_summaries"],
                "rate_efficiency_vs_gaussian_ceiling": candidate["rate_efficiency_vs_gaussian_ceiling"],
                "runtime_seconds": runtime,
                "projected_binding_runtime_seconds": projected,
                "peak_cuda_allocated_bytes": peak,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
