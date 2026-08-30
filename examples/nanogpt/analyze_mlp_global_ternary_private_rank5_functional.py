#!/usr/bin/env python3
"""Frozen H48 global-ternary/private-rank5 complete-neuron audit.

This audit keeps one charged low-bit global bank of paired GELU detector/write
endpoints and learns only layer-private rank-five endpoint transports plus
gains.  It evaluates functions and exact input JVPs; it does not train on CE.
"""

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
    collect_functional_bank,
    complete_neuron_jvp,
    complete_neuron_output,
    evaluate_function,
    extract_teacher_atoms,
    fit_student,
    git_commit,
    select_complete_atoms,
    summarize,
    tensor_sha256,
    canonical_sha256,
    write_json,
)


SCHEMA_VERSION = "nanogpt_mlp_global_ternary_private_rank5_functional_v1"
PLAN_SCHEMA_VERSION = (
    "nanogpt_mlp_global_ternary_private_rank5_functional_plan_v1"
)
CANDIDATE_NAME = "learned_global_ternary_private_rank5"


def encode_ternary(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    value = value.detach().float().cpu().contiguous()
    scale = value.abs().mean(dim=1).clamp_min(1e-30)
    codes = torch.round(value / scale[:, None]).clamp(-1, 1).to(torch.int8)
    return codes, scale.to(torch.float16)


def pack_ternary_2bit(codes: torch.Tensor) -> torch.Tensor:
    flat = codes.detach().cpu().to(torch.int16).flatten()
    if not bool(((flat >= -1) & (flat <= 1)).all()):
        raise ValueError("ternary codes must lie in {-1,0,+1}")
    encoded = (flat + 1).to(torch.uint8)
    padding = (-encoded.numel()) % 4
    if padding:
        encoded = torch.cat([encoded, torch.ones(padding, dtype=torch.uint8)])
    encoded = encoded.reshape(-1, 4).to(torch.int16)
    packed = (
        encoded[:, 0]
        | (encoded[:, 1] << 2)
        | (encoded[:, 2] << 4)
        | (encoded[:, 3] << 6)
    )
    return packed.to(torch.uint8).contiguous()


def unpack_ternary_2bit(packed: torch.Tensor, values: int) -> torch.Tensor:
    packed = packed.detach().cpu().to(torch.int16).flatten()
    decoded = torch.stack(
        [
            packed & 3,
            (packed >> 2) & 3,
            (packed >> 4) & 3,
            (packed >> 6) & 3,
        ],
        dim=1,
    ).flatten()[:values]
    if bool((decoded > 2).any()):
        raise ValueError("packed ternary payload contains reserved code 3")
    return (decoded - 1).to(torch.int8)


def decode_ternary(
    packed: torch.Tensor,
    scales: torch.Tensor,
    shape: tuple[int, int],
    *,
    device: str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    codes = unpack_ternary_2bit(packed, math.prod(shape)).reshape(shape)
    return codes.to(device=device, dtype=dtype) * scales.to(
        device=device, dtype=dtype
    )[:, None]


def deployment_accounting(
    atoms: int, layers: int, width: int, rank: int
) -> dict[str, int | float]:
    atom_values = 2 * atoms * width
    if atom_values % 4:
        raise ValueError("two-bit payload must contain a multiple of four values")
    atom_bytes = atom_values // 4
    scale_values = 2 * atoms
    scale_bytes = 2 * scale_values
    factor_values = layers * 2 * rank * (atoms + width)
    factor_bytes = 2 * factor_values
    gain_values = layers * atoms
    gain_bytes = 2 * gain_values
    total = atom_bytes + scale_bytes + factor_bytes + gain_bytes
    dense_bytes = layers * 2 * 4 * width * width * 2
    continuous = factor_values + gain_values
    cached = layers * 2 * atoms * width * 2
    return {
        "atoms": atoms,
        "layers": layers,
        "width": width,
        "rank": rank,
        "ternary_atom_values": atom_values,
        "ternary_atom_bytes": atom_bytes,
        "fp16_scale_values": scale_values,
        "fp16_scale_bytes": scale_bytes,
        "fp16_private_factor_values": factor_values,
        "fp16_private_factor_bytes": factor_bytes,
        "fp16_layer_gain_values": gain_values,
        "fp16_layer_gain_bytes": gain_bytes,
        "total_checkpoint_payload_bytes": total,
        "dense_replaced_mlp_fp16_bytes": dense_bytes,
        "checkpoint_byte_fraction": total / dense_bytes,
        "continuous_coordinate_values": continuous,
        "continuous_coordinate_fraction": continuous / (dense_bytes // 2),
        "cached_all_layer_fp16_endpoint_bytes": cached,
        "cached_weight_memory_fraction": cached / dense_bytes,
        "candidate_dense_matmul_fraction": atoms / (4 * width),
    }


class GlobalTernaryPrivateRankBank(torch.nn.Module):
    def __init__(
        self,
        base_u: torch.Tensor,
        base_v: torch.Tensor,
        source_layers: torch.Tensor,
        layers: list[int],
        rank: int,
        *,
        seed: int,
    ) -> None:
        super().__init__()
        self.layers = list(layers)
        self.layer_to_row = {layer: row for row, layer in enumerate(layers)}
        self.rank = int(rank)
        self.register_buffer("base_u", base_u.detach().float().clone())
        self.register_buffer("base_v", base_v.detach().float().clone())
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        layer_count, atoms, width = len(layers), *base_u.shape
        initial_a_u = 1e-3 * torch.randn(
            layer_count, atoms, rank, generator=generator
        )
        initial_a_v = 1e-3 * torch.randn(
            layer_count, atoms, rank, generator=generator
        )
        self.a_u = torch.nn.Parameter(initial_a_u)
        self.b_u = torch.nn.Parameter(torch.zeros(layer_count, rank, width))
        self.a_v = torch.nn.Parameter(initial_a_v)
        self.b_v = torch.nn.Parameter(torch.zeros(layer_count, rank, width))
        gain = torch.zeros(layer_count, atoms, dtype=torch.float32)
        for atom, source in enumerate(source_layers.tolist()):
            gain[self.layer_to_row[int(source)], atom] = 1.0
        self.gain = torch.nn.Parameter(gain)

    def factors(
        self,
        layer: int,
        *,
        include_base: bool = True,
        row_override: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.layer_to_row[layer] if row_override is None else row_override
        correction_u = self.a_u[row] @ self.b_u[row]
        correction_v = self.a_v[row] @ self.b_v[row]
        base_u = self.base_u if include_base else torch.zeros_like(self.base_u)
        base_v = self.base_v if include_base else torch.zeros_like(self.base_v)
        return base_u + correction_u, base_v + correction_v, self.gain[row]

    def forward_function(
        self,
        layer: int,
        inputs: torch.Tensor,
        directions: torch.Tensor | None,
        *,
        quantized: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del quantized
        detector, write, gain = self.factors(layer)
        output = complete_neuron_output(inputs, detector, write, gain)
        action = (
            complete_neuron_jvp(inputs, directions, detector, write, gain)
            if directions is not None
            else None
        )
        return output, action


def terminal_artifact(
    student: GlobalTernaryPrivateRankBank,
    base_u_codes: torch.Tensor,
    base_v_codes: torch.Tensor,
    u_scales: torch.Tensor,
    v_scales: torch.Tensor,
    accounting: dict[str, Any],
) -> dict[str, Any]:
    packed_u = pack_ternary_2bit(base_u_codes)
    packed_v = pack_ternary_2bit(base_v_codes)
    factors = {
        "a_u": student.a_u.detach().to(torch.float16).cpu(),
        "b_u": student.b_u.detach().to(torch.float16).cpu(),
        "a_v": student.a_v.detach().to(torch.float16).cpu(),
        "b_v": student.b_v.detach().to(torch.float16).cpu(),
    }
    gains = student.gain.detach().to(torch.float16).cpu()
    payload_bytes = packed_u.numel() + packed_v.numel()
    payload_bytes += 2 * (u_scales.numel() + v_scales.numel())
    payload_bytes += 2 * sum(value.numel() for value in factors.values())
    payload_bytes += 2 * gains.numel()
    if payload_bytes != accounting["total_checkpoint_payload_bytes"]:
        raise AssertionError((payload_bytes, accounting))
    return {
        "schema_version": "global_ternary_private_rank5_checkpoint_v1",
        "u_shape": list(base_u_codes.shape),
        "v_shape": list(base_v_codes.shape),
        "packed_u": packed_u,
        "packed_v": packed_v,
        "u_scales": u_scales.detach().cpu(),
        "v_scales": v_scales.detach().cpu(),
        **factors,
        "gains": gains,
        "layers": student.layers,
        "rank": student.rank,
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
    base_u = decode_ternary(
        artifact["packed_u"], artifact["u_scales"], shape, device=device
    )
    base_v = decode_ternary(
        artifact["packed_v"], artifact["v_scales"], shape, device=device
    )
    if mode == "random_global_bank":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(202608318)
        random_u = torch.randint(-1, 2, shape, generator=generator)
        random_v = torch.randint(-1, 2, shape, generator=generator)
        base_u = random_u.to(device=device, dtype=torch.float32) * artifact[
            "u_scales"
        ].to(device=device, dtype=torch.float32)[:, None]
        base_v = random_v.to(device=device, dtype=torch.float32) * artifact[
            "v_scales"
        ].to(device=device, dtype=torch.float32)[:, None]
    a_u = artifact["a_u"].to(device=device, dtype=torch.float32)
    b_u = artifact["b_u"].to(device=device, dtype=torch.float32)
    a_v = artifact["a_v"].to(device=device, dtype=torch.float32)
    b_v = artifact["b_v"].to(device=device, dtype=torch.float32)
    gains = artifact["gains"].to(device=device, dtype=torch.float32)
    layer_to_row = {
        int(layer): row for row, layer in enumerate(artifact["layers"])
    }

    def function(
        layer: int, inputs: torch.Tensor, directions: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        row = layer_to_row[layer]
        if mode == "shuffled_layer_transport":
            row = (row + 1) % len(layer_to_row)
        correction_u = a_u[row] @ b_u[row]
        correction_v = a_v[row] @ b_v[row]
        if mode == "global_bank_only":
            correction_u = torch.zeros_like(correction_u)
            correction_v = torch.zeros_like(correction_v)
        local_u = base_u + correction_u
        local_v = base_v + correction_v
        if mode == "private_transport_only":
            local_u = correction_u
            local_v = correction_v
        gain = gains[row]
        output = complete_neuron_output(inputs, local_u, local_v, gain)
        action = (
            complete_neuron_jvp(inputs, directions, local_u, local_v, gain)
            if directions is not None
            else None
        )
        return output, action

    return function


def materialize_layer(
    artifact: dict[str, Any], layer: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = tuple(int(value) for value in artifact["u_shape"])
    base_u = decode_ternary(
        artifact["packed_u"],
        artifact["u_scales"],
        shape,
        device=device,
        dtype=torch.float16,
    )
    base_v = decode_ternary(
        artifact["packed_v"],
        artifact["v_scales"],
        shape,
        device=device,
        dtype=torch.float16,
    )
    row = artifact["layers"].index(layer)
    local_u = base_u + artifact["a_u"][row].to(device) @ artifact["b_u"][row].to(
        device
    )
    local_v = base_v + artifact["a_v"][row].to(device) @ artifact["b_v"][row].to(
        device
    )
    local_v = local_v * artifact["gains"][row].to(device)[:, None]
    return local_u.contiguous(), local_v.contiguous()


def benchmark_mlp(
    teacher_u: torch.Tensor,
    teacher_v: torch.Tensor,
    artifact: dict[str, Any],
    *,
    layer: int,
    tokens: int,
    trials: int,
    device: str,
) -> dict[str, float | int]:
    if not device.startswith("cuda"):
        return {"available": 0, "reason": "CUDA required"}
    dense_u = teacher_u.to(device=device, dtype=torch.float16)
    dense_v = teacher_v.to(device=device, dtype=torch.float16)
    compact_u, compact_v = materialize_layer(artifact, layer, device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(202608319)
    inputs = torch.randn(tokens, dense_u.shape[1], generator=generator).to(
        device=device, dtype=torch.float16
    )

    def dense() -> torch.Tensor:
        return F.gelu(inputs @ dense_u.T) @ dense_v

    def compact() -> torch.Tensor:
        return F.gelu(inputs @ compact_u.T) @ compact_v

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
    }


def exact_quotas(layers: list[int]) -> dict[int, int]:
    quotas = {layer: (118 if layer < 8 else 116) for layer in layers}
    if sum(quotas.values()) != 1408:
        raise AssertionError(quotas)
    return quotas


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
        raise ValueError("unexpected H48 plan schema")
    if args.preflight:
        args.sample_cap = min(args.sample_cap, 128)
        args.iterations = min(args.iterations, 24)
    layers = [int(value) for value in plan["frozen_teacher"]["layers"]]
    atoms = int(plan["frozen_representation"]["complete_atoms"])
    rank = int(plan["frozen_representation"]["transport_rank"])
    width = int(plan["frozen_teacher"]["model_width"])
    accounting = deployment_accounting(atoms, len(layers), width, rank)
    expected = plan["exact_deployment_accounting"]
    for key in (
        "dense_replaced_mlp_fp16_bytes",
        "ternary_atom_values",
        "ternary_atom_bytes",
        "fp16_scale_values",
        "fp16_scale_bytes",
        "fp16_private_factor_values",
        "fp16_private_factor_bytes",
        "fp16_layer_gain_values",
        "fp16_layer_gain_bytes",
        "total_checkpoint_bytes",
        "continuous_coordinate_values",
        "cached_all_layer_fp16_endpoint_bytes",
    ):
        actual_key = (
            "total_checkpoint_payload_bytes"
            if key == "total_checkpoint_bytes"
            else key
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
        raise ValueError("teacher architecture does not match frozen H48 plan")
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

    quotas = exact_quotas(layers)
    selected_u, selected_v, source_layers, selected_indices = select_complete_atoms(
        banks["train"].inputs, teacher_u, teacher_v, quotas, args.device
    )
    u_codes, u_scales = encode_ternary(selected_u)
    v_codes, v_scales = encode_ternary(selected_v)
    base_u = u_codes.float() * u_scales.float()[:, None]
    base_v = v_codes.float() * v_scales.float()[:, None]
    student = GlobalTernaryPrivateRankBank(
        base_u,
        base_v,
        source_layers,
        layers,
        rank,
        seed=202608317,
    )
    history = fit_student(
        student,
        bank=banks["train"],
        teacher_detector=teacher_u,
        teacher_write=teacher_v,
        layers=layers,
        iterations=args.iterations,
        minibatch_rows=args.minibatch_rows,
        jvp_seed=int(inventory["jvp_seed"]),
        learning_rate_atoms=float(plan["fit"]["learning_rate_factors"]),
        learning_rate_gains=float(plan["fit"]["learning_rate_gains"]),
        weight_decay=float(plan["fit"]["weight_decay"]),
        gradient_clip_norm=float(plan["fit"]["gradient_clip_norm"]),
        device=args.device,
    )
    artifact = terminal_artifact(
        student, u_codes, v_codes, u_scales, v_scales, accounting
    )
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output / "global_ternary_private_rank5_checkpoint.pt"
    torch.save(artifact, checkpoint_path)

    candidates = {
        CANDIDATE_NAME: artifact_function(artifact, args.device),
        "global_bank_only": artifact_function(
            artifact, args.device, mode="global_bank_only"
        ),
        "private_transport_only": artifact_function(
            artifact, args.device, mode="private_transport_only"
        ),
        "shuffled_layer_transport": artifact_function(
            artifact, args.device, mode="shuffled_layer_transport"
        ),
        "random_global_bank": artifact_function(
            artifact, args.device, mode="random_global_bank"
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
    benchmark = benchmark_mlp(
        teacher_u[6],
        teacher_v[6],
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
        "PROMOTE_H48_TO_124M_CAUSAL_PREFLIGHT"
        if gate["all_gates_pass"]
        else "REJECT_H48_GLOBAL_TERNARY_PRIVATE_RANK5"
    )

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
        "snapshot": str(snapshot),
        "snapshot_sha256": file_sha256(snapshot),
        "snapshot_run_identity_sha256": snapshot_run_identity_sha256,
        "snapshot_tensor_inventory_sha256": snapshot_inventory_sha256,
        "data_dir": str(args.data_dir),
        "dataset_manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
        "sample_cap": args.sample_cap,
        "iterations": args.iterations,
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
            "fixed_decoded_fp32_bank_bytes": 2 * atoms * width * 4,
            "fp32_continuous_master_bytes": accounting[
                "continuous_coordinate_values"
            ]
            * 4,
            "fp32_continuous_gradient_bytes": accounting[
                "continuous_coordinate_values"
            ]
            * 4,
            "adam_continuous_moment_bytes": accounting[
                "continuous_coordinate_values"
            ]
            * 8,
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
            "Single 124M terminal dense parent and dataset.",
            "Function/JVP representation audit, not CE training.",
            "The frozen global ternary bank is teacher-derived and charged in full.",
            "Materialized FP16 layer endpoints are reproducible runtime cache, not checkpoint state.",
        ],
    }
    metadata_path = args.output / "metadata.json"
    write_json(metadata_path, metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
