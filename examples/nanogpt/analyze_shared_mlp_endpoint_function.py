#!/usr/bin/env python3
"""Compare a compact MLP endpoint with a dense teacher on both state banks.

The models see identical token batches.  We collect the residual-stream input
to every MLP from each model, then evaluate both MLPs on both the candidate and
teacher input banks.  Direct output recovery measures endpoint-value mismatch;
input-JVP recovery measures local functional mismatch without taking an
optimizer update or fitting an auxiliary map.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from examples.nanogpt.analyze_cproj_manifold import load_model
from examples.nanogpt.analyze_mlp_activation_chart_oracle import (
    prepare_inference_cache,
)
from examples.nanogpt.analyze_residual_compatibility import (
    fixed_validation_batches,
)


CORE_CONFIG_FIELDS = (
    "n_layer",
    "n_head",
    "n_embd",
    "block_size",
    "vocab_size",
    "bias",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(values: Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(memoryview(values.detach().cpu().contiguous().numpy()))
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def pair_metrics(target: Tensor, prediction: Tensor) -> dict[str, float]:
    """Measure prediction against the dense-teacher target."""

    target = target.detach().float()
    prediction = prediction.detach().float()
    if target.shape != prediction.shape:
        raise ValueError("target and prediction shapes must match")
    residual = target - prediction
    target_energy = target.square().sum().clamp_min(1e-30)
    denominator = target.norm() * prediction.norm()
    target_rms = target.square().mean().sqrt().clamp_min(1e-30)
    residual_rms = residual.square().mean().sqrt()
    return {
        "explained_target_energy": float(
            1.0 - residual.square().sum() / target_energy
        ),
        "cosine": float(
            (target * prediction).sum() / denominator.clamp_min(1e-30)
        ),
        "target_rms": float(target_rms),
        "prediction_rms": float(prediction.square().mean().sqrt()),
        "residual_rms": float(residual_rms),
        "relative_residual_rms": float(residual_rms / target_rms),
        "target_energy": float(target_energy),
        "residual_energy": float(residual.square().sum()),
    }


def summarize(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        raise ValueError("at least one metric record is required")
    recovery = [record["explained_target_energy"] for record in records]
    target_energy = sum(record["target_energy"] for record in records)
    residual_energy = sum(record["residual_energy"] for record in records)
    return {
        "mean_explained_target_energy": sum(recovery) / len(recovery),
        "minimum_explained_target_energy": min(recovery),
        "global_explained_target_energy": 1.0
        - residual_energy / max(target_energy, 1e-30),
        "mean_cosine": sum(record["cosine"] for record in records)
        / len(records),
        "maximum_relative_residual_rms": max(
            record["relative_residual_rms"] for record in records
        ),
    }


def classify_endpoint(
    output_summary: dict[str, float],
    jvp_summary: dict[str, float],
    *,
    output_mean_threshold: float,
    output_minimum_threshold: float,
    jvp_mean_threshold: float,
    jvp_minimum_threshold: float,
) -> str:
    output_passes = (
        output_summary["mean_explained_target_energy"] >= output_mean_threshold
        and output_summary["minimum_explained_target_energy"]
        >= output_minimum_threshold
    )
    if not output_passes:
        return "ENDPOINT_VALUE_MISMATCH"
    jvp_passes = (
        jvp_summary["mean_explained_target_energy"] >= jvp_mean_threshold
        and jvp_summary["minimum_explained_target_energy"]
        >= jvp_minimum_threshold
    )
    if not jvp_passes:
        return "INPUT_JACOBIAN_MISMATCH"
    return "UPSTREAM_OR_TEMPORAL_COMPOUNDING"


class MLPInputCollector:
    def __init__(self, model: nn.Module, sample_cap: int) -> None:
        self.sample_cap = int(sample_cap)
        self.values: dict[int, list[Tensor]] = {
            layer: [] for layer in range(model.config.n_layer)
        }
        self.counts = {layer: 0 for layer in range(model.config.n_layer)}
        self.handles = [
            block.mlp.register_forward_pre_hook(self._hook(layer))
            for layer, block in enumerate(model.transformer.h)
        ]

    def _hook(self, layer: int):
        def hook(_module: nn.Module, inputs: tuple[Tensor, ...]) -> None:
            remaining = self.sample_cap - self.counts[layer]
            if remaining <= 0:
                return
            rows = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
            rows = rows[:remaining].cpu()
            self.values[layer].append(rows)
            self.counts[layer] += int(rows.shape[0])

        return hook

    def complete(self) -> bool:
        return all(count >= self.sample_cap for count in self.counts.values())

    def tensors(self) -> dict[int, Tensor]:
        if not self.complete():
            raise RuntimeError("insufficient MLP input samples")
        return {
            layer: torch.cat(parts, dim=0)[: self.sample_cap]
            for layer, parts in self.values.items()
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def collect_inputs(
    model: nn.Module,
    batches: list[Tensor],
    sample_cap: int,
    device: str,
) -> dict[int, Tensor]:
    collector = MLPInputCollector(model, sample_cap)
    prepare_inference_cache(model)
    try:
        with torch.no_grad():
            for batch in batches:
                with torch.amp.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                ):
                    model(batch.to(device), None)
                if collector.complete():
                    break
        return collector.tensors()
    finally:
        collector.close()
        model.flush_block_fht_cache()


def rademacher_tangent(
    shape: torch.Size,
    *,
    device: str,
    seed: int,
) -> Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    values = torch.randint(
        0, 2, shape, generator=generator, device=device, dtype=torch.int64
    )
    return (values.float().mul_(2.0).sub_(1.0)) / math.sqrt(shape[-1])


def module_jvp(module: nn.Module, inputs: Tensor, tangent: Tensor) -> Tensor:
    with torch.enable_grad():
        _, jvp = torch.autograd.functional.jvp(
            module,
            inputs,
            tangent,
            create_graph=False,
            strict=False,
        )
    return jvp.detach()


def validate_core_configs(candidate: nn.Module, teacher: nn.Module) -> None:
    mismatches = {
        field: (getattr(candidate.config, field), getattr(teacher.config, field))
        for field in CORE_CONFIG_FIELDS
        if getattr(candidate.config, field) != getattr(teacher.config, field)
    }
    if mismatches:
        raise ValueError(f"candidate/teacher core config mismatch: {mismatches}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-checkpoint", required=True, type=Path)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--teacher-checkpoint", required=True, type=Path)
    parser.add_argument("--teacher-sha256", required=True)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--data-manifest-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--sample-cap", type=int, default=512)
    parser.add_argument("--jvp-directions", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--output-mean-threshold", type=float, default=0.9)
    parser.add_argument("--output-minimum-threshold", type=float, default=0.75)
    parser.add_argument("--jvp-mean-threshold", type=float, default=0.8)
    parser.add_argument("--jvp-minimum-threshold", type=float, default=0.5)
    args = parser.parse_args()

    for value in (
        args.batch_size,
        args.batches,
        args.sample_cap,
        args.jvp_directions,
    ):
        if value <= 0:
            raise ValueError("batch and sample counts must be positive")
    manifest = args.data_dir / "manifest.json"
    if sha256_file(manifest) != args.data_manifest_sha256:
        raise ValueError("dataset manifest identity mismatch")
    if sha256_file(args.candidate_checkpoint) != args.candidate_sha256:
        raise ValueError("candidate checkpoint identity mismatch")
    if sha256_file(args.teacher_checkpoint) != args.teacher_sha256:
        raise ValueError("teacher checkpoint identity mismatch")

    device = "cuda"
    started = time.time()
    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    candidate = load_model(args.candidate_checkpoint, device)
    teacher = load_model(args.teacher_checkpoint, device)
    validate_core_configs(candidate, teacher)
    batches = fixed_validation_batches(
        args.data_dir,
        batch_size=args.batch_size,
        block_size=candidate.config.block_size,
        batches=args.batches,
        seed=args.seed,
    )
    batch_digest = tensor_sha256(torch.stack(batches))
    candidate_inputs = collect_inputs(
        candidate, batches, args.sample_cap, device
    )
    teacher_inputs = collect_inputs(teacher, batches, args.sample_cap, device)

    per_bank: dict[str, dict[str, Any]] = {}
    output_records: list[dict[str, float]] = []
    jvp_records: list[dict[str, float]] = []
    input_records: list[dict[str, float]] = []
    for bank_index, (bank_name, bank) in enumerate(
        (("candidate_states", candidate_inputs), ("teacher_states", teacher_inputs))
    ):
        per_layer: dict[str, Any] = {}
        for layer in range(candidate.config.n_layer):
            inputs = bank[layer].to(device=device, dtype=torch.float32)
            candidate_mlp = candidate.transformer.h[layer].mlp
            teacher_mlp = teacher.transformer.h[layer].mlp
            with torch.no_grad():
                candidate_output = candidate_mlp(inputs)
                teacher_output = teacher_mlp(inputs)
            output_metrics = pair_metrics(teacher_output, candidate_output)
            output_records.append(output_metrics)

            candidate_jvps: list[Tensor] = []
            teacher_jvps: list[Tensor] = []
            for direction in range(args.jvp_directions):
                tangent = rademacher_tangent(
                    inputs.shape,
                    device=device,
                    seed=(
                        args.seed
                        + bank_index * 100_000
                        + layer * 1_000
                        + direction
                    ),
                )
                candidate_jvps.append(
                    module_jvp(candidate_mlp, inputs, tangent).cpu()
                )
                teacher_jvps.append(
                    module_jvp(teacher_mlp, inputs, tangent).cpu()
                )
            jvp_metrics = pair_metrics(
                torch.stack(teacher_jvps), torch.stack(candidate_jvps)
            )
            jvp_records.append(jvp_metrics)
            per_layer[str(layer)] = {
                "input_sha256": tensor_sha256(bank[layer]),
                "output": output_metrics,
                "input_jvp": jvp_metrics,
            }
        per_bank[bank_name] = {"per_layer": per_layer}

    for layer in range(candidate.config.n_layer):
        input_records.append(
            pair_metrics(teacher_inputs[layer], candidate_inputs[layer])
        )

    output_summary = summarize(output_records)
    jvp_summary = summarize(jvp_records)
    input_summary = summarize(input_records)
    classification = classify_endpoint(
        output_summary,
        jvp_summary,
        output_mean_threshold=args.output_mean_threshold,
        output_minimum_threshold=args.output_minimum_threshold,
        jvp_mean_threshold=args.jvp_mean_threshold,
        jvp_minimum_threshold=args.jvp_minimum_threshold,
    )
    payload = {
        "schema_version": "mai_shared_mlp_endpoint_function_v1",
        "classification": classification,
        "candidate": {
            "path": str(args.candidate_checkpoint),
            "sha256": args.candidate_sha256,
        },
        "teacher": {
            "path": str(args.teacher_checkpoint),
            "sha256": args.teacher_sha256,
        },
        "dataset_manifest": {
            "path": str(manifest),
            "sha256": args.data_manifest_sha256,
        },
        "measurement": {
            "definition": "two-sided same-input direct MLP output and analytic input-JVP comparison; no fit and no optimizer update",
            "split": "validation",
            "seed": args.seed,
            "batch_size": args.batch_size,
            "batches": args.batches,
            "sample_cap_per_layer_per_bank": args.sample_cap,
            "jvp_directions": args.jvp_directions,
            "fixed_batch_sha256": batch_digest,
        },
        "frozen_gate": {
            "output_mean_threshold": args.output_mean_threshold,
            "output_minimum_threshold": args.output_minimum_threshold,
            "jvp_mean_threshold": args.jvp_mean_threshold,
            "jvp_minimum_threshold": args.jvp_minimum_threshold,
        },
        "summaries": {
            "aligned_state_input": input_summary,
            "mlp_output": output_summary,
            "mlp_input_jvp": jvp_summary,
        },
        "banks": per_bank,
        "maximum_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "wall_seconds": time.time() - started,
    }
    atomic_json(args.output, payload)
    print(json.dumps({
        "classification": classification,
        "summaries": payload["summaries"],
        "output": str(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
