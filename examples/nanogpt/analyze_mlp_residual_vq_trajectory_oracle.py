#!/usr/bin/env python3
"""Zero-update gate for a compact shared residual-VQ MLP weight field.

The oracle learns codebook atoms only from a chronological discovery prefix.
Every evaluated dense displacement is encoded with two block codes and two
scalar gains; no ambient residual, dense basis, or dense optimizer state is
retained.  Held-out metrics use both weight-space chords and the change in the
expert function/input-JVP relative to the exact seeded initialization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


PLAN_SCHEMA = "mai_124m_mlp_residual_vq_trajectory_gate_plan_v1"
RESULT_SCHEMA = "mai_124m_mlp_residual_vq_trajectory_gate_result_v1"
SNAPSHOT_SCHEMAS = {
    "nanogpt_parameter_trajectory_v1",
    "nanogpt_parameter_trajectory_v2",
}
TARGET_SUFFIXES = ("expert_c_fc", "expert_c_proj")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def inventory(directory: Path) -> tuple[list[dict[str, Any]], str]:
    rows = [
        {"name": path.name, "bytes": path.stat().st_size}
        for path in sorted(directory.glob("step_*.pt"))
    ]
    return rows, canonical_sha256(rows)


def load_snapshot(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") not in SNAPSHOT_SCHEMAS:
        raise ValueError(f"unsupported trajectory snapshot: {path}")
    parameters = {
        name: value.float().contiguous()
        for name, value in payload["parameters"].items()
        if name.endswith(TARGET_SUFFIXES)
    }
    if not parameters:
        raise ValueError(f"snapshot contains no expert MLP matrices: {path}")
    return {
        "step": int(payload["step"]),
        "run_identity_sha256": str(payload["run_identity_sha256"]),
        "parameters": parameters,
    }


def blocks(tensor: torch.Tensor, block_size: int) -> torch.Tensor:
    flat = tensor.reshape(-1)
    if flat.numel() % block_size:
        raise ValueError(f"tensor size {flat.numel()} is not divisible by block size {block_size}")
    return flat.reshape(-1, block_size)


def normalize_rows(values: torch.Tensor) -> torch.Tensor:
    return values / values.norm(dim=1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def nearest_atom(
    values: torch.Tensor,
    codebook: torch.Tensor,
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices: list[torch.Tensor] = []
    gains: list[torch.Tensor] = []
    for start in range(0, values.shape[0], chunk_size):
        chunk = values[start : start + chunk_size]
        scores = chunk @ codebook.T
        index = scores.abs().argmax(dim=1)
        gain = scores.gather(1, index[:, None]).squeeze(1)
        indices.append(index)
        gains.append(gain)
    return torch.cat(indices), torch.cat(gains)


@torch.no_grad()
def spherical_kmeans(
    samples: torch.Tensor,
    *,
    atoms: int,
    iterations: int,
    seed: int,
    chunk_size: int,
) -> torch.Tensor:
    if samples.shape[0] < atoms:
        raise ValueError("fewer samples than codebook atoms")
    samples = normalize_rows(samples)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    initial = torch.randperm(samples.shape[0], generator=generator)[:atoms]
    codebook = samples.index_select(0, initial.to(samples.device)).clone()
    for _ in range(iterations):
        sums = torch.zeros_like(codebook)
        counts = torch.zeros(atoms, device=samples.device, dtype=torch.float32)
        for start in range(0, samples.shape[0], chunk_size):
            chunk = samples[start : start + chunk_size]
            scores = chunk @ codebook.T
            index = scores.abs().argmax(dim=1)
            sign = scores.gather(1, index[:, None]).sign()
            sums.index_add_(0, index, chunk * sign)
            counts.index_add_(0, index, torch.ones_like(index, dtype=torch.float32))
        empty = counts == 0
        if empty.any():
            replacement = torch.randperm(
                samples.shape[0], generator=generator
            )[: int(empty.sum())].to(samples.device)
            sums[empty] = samples.index_select(0, replacement)
            counts[empty] = 1.0
        codebook = normalize_rows(sums / counts[:, None])
    return codebook


@torch.no_grad()
def fit_codebooks(
    samples: torch.Tensor,
    *,
    stages: int,
    atoms: int,
    iterations: int,
    seed: int,
    chunk_size: int,
) -> list[torch.Tensor]:
    residual = samples
    result = []
    for stage in range(stages):
        codebook = spherical_kmeans(
            residual,
            atoms=atoms,
            iterations=iterations,
            seed=seed + stage,
            chunk_size=chunk_size,
        )
        index, gain = nearest_atom(residual, codebook, chunk_size=chunk_size)
        residual = residual - codebook.index_select(0, index) * gain[:, None]
        result.append(codebook)
    return result


@torch.no_grad()
def encode_decode(
    values: torch.Tensor,
    codebooks: list[torch.Tensor],
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    residual = values
    decoded = torch.zeros_like(values)
    indices, gains = [], []
    for codebook in codebooks:
        index, gain = nearest_atom(residual, codebook, chunk_size=chunk_size)
        contribution = codebook.index_select(0, index) * gain[:, None]
        decoded.add_(contribution)
        residual = residual - contribution
        indices.append(index)
        gains.append(gain)
    return decoded, indices, gains


def recovery(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    denominator = reference.float().square().sum().clamp_min(1e-30)
    error = (candidate.float() - reference.float()).square().sum()
    return float(1.0 - error / denominator)


def cosine(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    return float(F.cosine_similarity(
        reference.float().reshape(1, -1),
        candidate.float().reshape(1, -1),
        dim=1,
        eps=1e-30,
    ))


def gelu_derivative(values: torch.Tensor) -> torch.Tensor:
    inv_sqrt_two = 1.0 / math.sqrt(2.0)
    inv_sqrt_two_pi = 1.0 / math.sqrt(2.0 * math.pi)
    return 0.5 * (1.0 + torch.erf(values * inv_sqrt_two)) + (
        values * torch.exp(-0.5 * values.square()) * inv_sqrt_two_pi
    )


@torch.no_grad()
def expert_function_and_jvp(
    c_fc: torch.Tensor,
    c_proj: torch.Tensor,
    inputs: torch.Tensor,
    directions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pre = torch.einsum("esd,ehd->esh", inputs, c_fc)
    pre_jvp = torch.einsum("esd,ehd->esh", directions, c_fc)
    hidden = F.gelu(pre)
    hidden_jvp = gelu_derivative(pre) * pre_jvp
    output = torch.einsum("esh,edh->esd", hidden, c_proj)
    output_jvp = torch.einsum("esh,edh->esd", hidden_jvp, c_proj)
    return output, output_jvp


def layer_from_name(name: str) -> int:
    parts = name.split(".")
    if len(parts) < 3 or not parts[2].isdigit():
        raise ValueError(f"cannot parse layer from {name}")
    return int(parts[2])


def validate_plan(plan: dict[str, Any], plan_path: Path, trajectory_dir: Path) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("plan schema mismatch")
    identity = plan["identity"]
    if Path(identity["entrypoint"]) != Path(__file__).resolve().relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError("entrypoint path mismatch")
    expected_entrypoint = identity.get("entrypoint_sha256")
    if expected_entrypoint and file_sha256(Path(__file__)) != expected_entrypoint:
        raise ValueError("entrypoint hash drift")
    if trajectory_dir.resolve() != Path(identity["trajectory_directory"]).resolve():
        raise ValueError("trajectory directory differs from plan")
    rows, digest = inventory(trajectory_dir)
    if len(rows) != int(identity["trajectory_file_count"]):
        raise ValueError("trajectory file-count mismatch")
    if sum(row["bytes"] for row in rows) != int(identity["trajectory_total_bytes"]):
        raise ValueError("trajectory byte-count mismatch")
    if digest != identity["trajectory_inventory_sha256"]:
        raise ValueError("trajectory inventory mismatch")
    if not file_sha256(plan_path):
        raise ValueError("empty plan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text())
    validate_plan(plan, args.plan, args.trajectory_dir)
    protocol, spec = plan["protocol"], plan["candidate"]
    steps = [int(value) for value in protocol["trajectory_steps"]]
    discovery_steps = set(int(value) for value in protocol["discovery_steps"])
    heldout_steps = [int(value) for value in protocol["heldout_steps"]]
    block_size = int(spec["block_size"])
    stages = int(spec["stages"])
    atoms = int(spec["atoms_per_stage"])
    chunk_size = int(protocol["similarity_chunk_size"])
    device = torch.device(args.device)
    if args.preflight:
        steps = steps[:3]
        discovery_steps = {steps[1]}
        heldout_steps = [steps[2]]

    started = time.time()
    base = load_snapshot(args.trajectory_dir / f"step_{steps[0]:06d}.pt")
    if base["run_identity_sha256"] != plan["identity"]["trajectory_run_identity_sha256"]:
        raise ValueError("trajectory run identity mismatch")
    names = sorted(base["parameters"])
    parameter_numel = sum(base["parameters"][name].numel() for name in names)

    sample_cap = int(protocol["codebook_sample_blocks"])
    generator = torch.Generator(device="cpu").manual_seed(int(protocol["sample_seed"]))
    sample_parts: list[torch.Tensor] = []
    discovery_paths = [
        args.trajectory_dir / f"step_{step:06d}.pt"
        for step in steps[1:]
        if step in discovery_steps
    ]
    per_source = max(1, sample_cap // max(len(discovery_paths) * len(names), 1))
    for path in discovery_paths:
        snapshot = load_snapshot(path)
        for name in names:
            delta = blocks(snapshot["parameters"][name] - base["parameters"][name], block_size)
            count = min(per_source, delta.shape[0])
            selected = torch.randperm(delta.shape[0], generator=generator)[:count]
            sample_parts.append(delta.index_select(0, selected))
    samples = torch.cat(sample_parts)[:sample_cap].to(device)
    codebooks = fit_codebooks(
        samples,
        stages=stages,
        atoms=atoms,
        iterations=int(protocol["kmeans_iterations"] if not args.preflight else 2),
        seed=int(protocol["codebook_seed"]),
        chunk_size=chunk_size,
    )
    del samples, sample_parts

    input_samples = int(protocol["functional_samples_per_expert"])
    input_generator = torch.Generator(device="cpu").manual_seed(int(protocol["functional_seed"]))
    decoded_by_step: dict[int, dict[str, torch.Tensor]] = {}
    rows = []
    aggregate_weight_reference = torch.tensor(0.0, device=device)
    aggregate_weight_error = torch.tensor(0.0, device=device)
    aggregate_output_reference = torch.tensor(0.0, device=device)
    aggregate_output_error = torch.tensor(0.0, device=device)
    aggregate_jvp_reference = torch.tensor(0.0, device=device)
    aggregate_jvp_error = torch.tensor(0.0, device=device)
    layer_output_recoveries: list[float] = []

    for step in heldout_steps:
        snapshot = load_snapshot(args.trajectory_dir / f"step_{step:06d}.pt")
        decoded_parameters: dict[str, torch.Tensor] = {}
        step_reference = torch.tensor(0.0, device=device)
        step_error = torch.tensor(0.0, device=device)
        for name in names:
            dense_delta = blocks(
                snapshot["parameters"][name] - base["parameters"][name], block_size
            ).to(device)
            decoded_blocks, _, _ = encode_decode(
                dense_delta, codebooks, chunk_size=chunk_size
            )
            step_reference += dense_delta.square().sum()
            step_error += (decoded_blocks - dense_delta).square().sum()
            decoded_parameters[name] = (
                base["parameters"][name].to(device) + decoded_blocks.reshape_as(dense_delta).reshape_as(base["parameters"][name])
            )
        aggregate_weight_reference += step_reference
        aggregate_weight_error += step_error
        decoded_by_step[step] = decoded_parameters

        layers = sorted({layer_from_name(name) for name in names})
        step_outputs, step_jvps = [], []
        for layer in layers:
            cfc_name = next(name for name in names if layer_from_name(name) == layer and name.endswith("expert_c_fc"))
            cproj_name = next(name for name in names if layer_from_name(name) == layer and name.endswith("expert_c_proj"))
            dense_fc = snapshot["parameters"][cfc_name].to(device)
            dense_proj = snapshot["parameters"][cproj_name].to(device)
            base_fc = base["parameters"][cfc_name].to(device)
            base_proj = base["parameters"][cproj_name].to(device)
            candidate_fc = decoded_parameters[cfc_name]
            candidate_proj = decoded_parameters[cproj_name]
            experts, _, width = dense_fc.shape
            inputs = torch.randn(experts, input_samples, width, generator=input_generator).to(device)
            directions = torch.randn(experts, input_samples, width, generator=input_generator).to(device)
            base_output, base_jvp = expert_function_and_jvp(base_fc, base_proj, inputs, directions)
            dense_output, dense_jvp = expert_function_and_jvp(dense_fc, dense_proj, inputs, directions)
            candidate_output, candidate_jvp = expert_function_and_jvp(candidate_fc, candidate_proj, inputs, directions)
            output_reference = dense_output - base_output
            output_candidate = candidate_output - base_output
            jvp_reference = dense_jvp - base_jvp
            jvp_candidate = candidate_jvp - base_jvp
            output_rec = recovery(output_reference, output_candidate)
            jvp_rec = recovery(jvp_reference, jvp_candidate)
            layer_output_recoveries.append(output_rec)
            step_outputs.append(output_rec)
            step_jvps.append(jvp_rec)
            aggregate_output_reference += output_reference.square().sum()
            aggregate_output_error += (output_candidate - output_reference).square().sum()
            aggregate_jvp_reference += jvp_reference.square().sum()
            aggregate_jvp_error += (jvp_candidate - jvp_reference).square().sum()
        rows.append({
            "step": step,
            "weight_displacement_recovery": float(1.0 - step_error / step_reference.clamp_min(1e-30)),
            "mean_layer_output_change_recovery": sum(step_outputs) / len(step_outputs),
            "mean_layer_input_jvp_change_recovery": sum(step_jvps) / len(step_jvps),
        })

    chord_cosines, chord_recoveries = [], []
    for previous, current in zip(heldout_steps[:-1], heldout_steps[1:]):
        dense_previous = load_snapshot(args.trajectory_dir / f"step_{previous:06d}.pt")
        dense_current = load_snapshot(args.trajectory_dir / f"step_{current:06d}.pt")
        for name in names:
            dense_chord = dense_current["parameters"][name] - dense_previous["parameters"][name]
            decoded_chord = decoded_by_step[current][name].cpu() - decoded_by_step[previous][name].cpu()
            chord_cosines.append(cosine(dense_chord, decoded_chord))
            chord_recoveries.append(recovery(dense_chord, decoded_chord))

    full = plan["full_model_accounting"]
    dense_entries = int(full["dense_paired_mlp_entries"])
    codebook_coordinates = stages * atoms * block_size
    gain_coordinates = stages * (dense_entries // block_size)
    continuous_coordinates = codebook_coordinates + gain_coordinates
    packed_code_bytes = math.ceil(stages * (dense_entries // block_size) * math.log2(atoms) / 8.0)
    persistent_bytes = 2 * continuous_coordinates + packed_code_bytes
    coordinate_compression = dense_entries / continuous_coordinates
    bf16_state_compression = (2 * dense_entries) / persistent_bytes

    metrics = {
        "aggregate_weight_displacement_recovery": float(
            1.0 - aggregate_weight_error / aggregate_weight_reference.clamp_min(1e-30)
        ),
        "mean_local_chord_cosine": sum(chord_cosines) / max(len(chord_cosines), 1),
        "mean_local_chord_recovery": sum(chord_recoveries) / max(len(chord_recoveries), 1),
        "aggregate_output_change_recovery": float(
            1.0 - aggregate_output_error / aggregate_output_reference.clamp_min(1e-30)
        ),
        "aggregate_input_jvp_change_recovery": float(
            1.0 - aggregate_jvp_error / aggregate_jvp_reference.clamp_min(1e-30)
        ),
        "minimum_layer_output_change_recovery": min(layer_output_recoveries),
    }
    gates = plan["gates"]
    outcomes = {
        "weight_displacement": metrics["aggregate_weight_displacement_recovery"] >= float(gates["minimum_weight_displacement_recovery"]),
        "local_chord_cosine": metrics["mean_local_chord_cosine"] >= float(gates["minimum_mean_local_chord_cosine"]),
        "output_change": metrics["aggregate_output_change_recovery"] >= float(gates["minimum_output_change_recovery"]),
        "input_jvp_change": metrics["aggregate_input_jvp_change_recovery"] >= float(gates["minimum_input_jvp_change_recovery"]),
        "every_layer_output": metrics["minimum_layer_output_change_recovery"] >= float(gates["minimum_every_layer_output_change_recovery"]),
        "continuous_coordinate_compression": coordinate_compression >= float(gates["minimum_continuous_coordinate_compression"]),
        "persistent_bf16_state_compression": bf16_state_compression >= float(gates["minimum_persistent_bf16_state_compression"]),
        "finite": all(math.isfinite(value) for value in metrics.values()),
    }
    result = {
        "schema_version": RESULT_SCHEMA,
        "classification": "PREFLIGHT" if args.preflight else ("ACCEPTED" if all(outcomes.values()) else "REJECTED"),
        "passed": bool(all(outcomes.values())) if not args.preflight else None,
        "plan_sha256": file_sha256(args.plan),
        "trajectory_run_identity_sha256": base["run_identity_sha256"],
        "trajectory_subset_entries": parameter_numel,
        "candidate": spec,
        "accounting": {
            "dense_paired_mlp_entries": dense_entries,
            "codebook_coordinates": codebook_coordinates,
            "gain_coordinates": gain_coordinates,
            "continuous_coordinates": continuous_coordinates,
            "packed_code_bytes": packed_code_bytes,
            "persistent_model_bytes": persistent_bytes,
            "continuous_coordinate_compression": coordinate_compression,
            "persistent_bf16_state_compression": bf16_state_compression,
            "decode_multiply_adds_per_materialization": stages * dense_entries,
        },
        "metrics": metrics,
        "per_snapshot": rows,
        "gate_outcomes": outcomes,
        "wall_seconds": time.time() - started,
        "maximum_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
        "interpretation": (
            "This is an optimistic representability oracle: held-out codes and gains are selected with dense targets. "
            "A pass authorizes a causal/STE training design, not an LM performance claim; a failure rejects this exact fixed RVQ budget."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
