#!/usr/bin/env python3
"""Gate blockwise ambient quantization of the accepted full-MLP state.

The accepted full-MLP replacement preserves missing dense directions through
Muon momentum and compression-error feedback.  Earlier low-rank, sparse, and
procedural-chart state codes changed the direction family and failed.  This
zero-update oracle instead quantizes every ambient coordinate independently,
with one max-absolute scale per contiguous block.  It measures raw state
reconstruction and the induced state-only Muon polar direction before any
optimizer or model update is allowed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from examples.nanogpt.muon import zeropower_via_newtonschulz5


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_SCHEMA = "mai_124m_full_mlp_temporal_state_quantization_plan_v1"
RESULT_SCHEMA = "mai_124m_full_mlp_temporal_state_quantization_result_v1"
STATE_NAMES = ("momentum_buffer", "compression_residual")
OPTIMIZER_ROLES = ((0, "c_fc"), (1, "c_proj"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def blockwise_symmetric_quantize(
    source: torch.Tensor,
    *,
    bits: int,
    block_size: int,
) -> tuple[torch.Tensor, dict[str, int | float]]:
    """Return FP32 reconstruction using packed-width values and FP16 scales."""
    if bits not in {4, 6, 8}:
        raise ValueError("bits must be one of 4, 6, or 8")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    flat = source.detach().float().reshape(-1)
    blocks = math.ceil(flat.numel() / block_size)
    padded_elements = blocks * block_size - flat.numel()
    padded = F.pad(flat, (0, padded_elements)).reshape(blocks, block_size)
    qmax = (1 << (bits - 1)) - 1
    exact_scale = padded.abs().amax(dim=1) / qmax
    stored_scale = exact_scale.to(torch.float16)
    restored_scale = stored_scale.float()
    nonzero = exact_scale > 0
    underflow = int((nonzero & (restored_scale == 0)).sum())
    overflow = int(torch.isinf(stored_scale).sum())
    if underflow or overflow:
        raise FloatingPointError(
            f"FP16 scale failure: underflow={underflow}, overflow={overflow}"
        )
    divisor = torch.where(nonzero, restored_scale, torch.ones_like(restored_scale))
    quantized = (
        torch.round(padded / divisor[:, None]).clamp(-qmax, qmax).to(torch.int8)
    )
    reconstructed = (
        quantized.float() * restored_scale[:, None]
    ).reshape(-1)[: flat.numel()]
    packed_value_bytes = math.ceil(flat.numel() * bits / 8)
    scale_bytes = blocks * torch.tensor([], dtype=torch.float16).element_size()
    return reconstructed.reshape_as(source), {
        "elements": flat.numel(),
        "blocks": blocks,
        "padded_elements": padded_elements,
        "qmax": qmax,
        "packed_value_bytes": packed_value_bytes,
        "scale_bytes": scale_bytes,
        "persistent_bytes": packed_value_bytes + scale_bytes,
        "dense_fp32_bytes": flat.numel() * 4,
        "prototype_container": "signed int8 values plus FP16 block scales",
        "packed_storage_assumption": bits < 8,
        "scale_underflow_blocks": underflow,
        "scale_overflow_blocks": overflow,
    }


def tensor_metrics(target: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    target = target.detach().double().reshape(-1)
    candidate = candidate.detach().double().reshape(-1)
    target_energy = target.square().sum().clamp_min(1e-30)
    candidate_energy = candidate.square().sum().clamp_min(1e-30)
    squared_error = (target - candidate).square().sum()
    dot = (target * candidate).sum()
    return {
        "target_energy": float(target_energy),
        "candidate_energy": float(candidate_energy),
        "squared_error": float(squared_error),
        "relative_fro_error": float((squared_error / target_energy).sqrt()),
        "energy_recovery": float(1.0 - squared_error / target_energy),
        "cosine": float(dot / (target_energy * candidate_energy).sqrt()),
    }


def aggregate_metrics(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    target_energy = sum(float(row[f"{prefix}_target_energy"]) for row in rows)
    candidate_energy = sum(float(row[f"{prefix}_candidate_energy"]) for row in rows)
    squared_error = sum(float(row[f"{prefix}_squared_error"]) for row in rows)
    dot = sum(float(row[f"{prefix}_dot"]) for row in rows)
    denominator = max(target_energy, 1e-30)
    return {
        "relative_fro_error": math.sqrt(squared_error / denominator),
        "energy_recovery": 1.0 - squared_error / denominator,
        "cosine": dot / math.sqrt(max(target_energy * candidate_energy, 1e-30)),
        "minimum_layer_energy_recovery": min(
            float(row[f"{prefix}_energy_recovery"]) for row in rows
        ),
        "minimum_layer_cosine": min(float(row[f"{prefix}_cosine"]) for row in rows),
        "maximum_layer_relative_fro_error": max(
            float(row[f"{prefix}_relative_fro_error"]) for row in rows
        ),
    }


def flatten_metric(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    result = {f"{prefix}_{key}": value for key, value in metrics.items()}
    target_energy = float(metrics["target_energy"])
    candidate_energy = float(metrics["candidate_energy"])
    cosine = float(metrics["cosine"])
    result[f"{prefix}_dot"] = cosine * math.sqrt(target_energy * candidate_energy)
    return result


def classify(
    summaries: list[dict[str, Any]], rule: dict[str, Any]
) -> dict[str, Any]:
    passing: list[dict[str, Any]] = []
    for summary in summaries:
        raw = summary["raw"]
        polar = summary["momentum_state_only_polar_proxy"]
        passed = (
            summary["persistent_storage_ratio_to_dense_fp32"]
            <= float(rule["maximum_persistent_storage_ratio"])
            and raw["momentum_buffer"]["energy_recovery"]
            >= float(rule["minimum_global_raw_energy_recovery"])
            and raw["compression_residual"]["energy_recovery"]
            >= float(rule["minimum_global_raw_energy_recovery"])
            and raw["momentum_buffer"]["minimum_layer_cosine"]
            >= float(rule["minimum_layer_raw_cosine"])
            and raw["compression_residual"]["minimum_layer_cosine"]
            >= float(rule["minimum_layer_raw_cosine"])
            and polar["energy_recovery"]
            >= float(rule["minimum_global_polar_energy_recovery"])
            and polar["minimum_layer_cosine"]
            >= float(rule["minimum_layer_polar_cosine"])
        )
        summary["gate_passed"] = passed
        if passed:
            passing.append(summary)
    selected = min(
        passing,
        key=lambda row: (
            int(row["persistent_storage_bytes"]),
            -float(row["momentum_state_only_polar_proxy"]["energy_recovery"]),
        ),
        default=None,
    )
    return {
        "classification": (
            "BLOCKWISE_AMBIENT_STATE_QUANTIZATION_PLAUSIBLE"
            if selected is not None
            else "BLOCKWISE_AMBIENT_STATE_QUANTIZATION_REJECTED"
        ),
        "selected_candidate": (
            None
            if selected is None
            else {"bits": selected["bits"], "block_size": selected["block_size"]}
        ),
        "passing_candidate_count": len(passing),
        "thresholds": rule,
        "parameter_updates_to_checkpoint": 0,
        "interpretation_boundary": (
            "The polar metric is a state-only Muon proxy because the sealed terminal "
            "checkpoint has no next-step task gradient. Passing authorizes an exact-resume "
            "codec implementation, tests, and an MFU/VRAM gate, not a larger rung."
        ),
    }


def validate(args: argparse.Namespace, plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unexpected temporal-state quantization plan schema")
    observed = {
        "entrypoint_sha256": file_sha256(Path(__file__)),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "config_sha256": file_sha256(args.config),
        "dataset_manifest_sha256": file_sha256(args.data_manifest),
        "parent_result_sha256": file_sha256(args.parent_result),
    }
    if observed != plan.get("identity"):
        raise ValueError(f"temporal-state quantization identity mismatch: {observed}")
    parent = json.loads(args.parent_result.read_text())
    if parent.get("classification") != "STABLE_FULL_MLP_ERROR_FEEDBACK_0P5TPP_PASSED":
        raise ValueError("parent result does not authorize state quantization")
    expected = {
        "parameter_updates": 0,
        "checkpoint_next_iter": 238,
        "optimizer_roles": [list(value) for value in OPTIMIZER_ROLES],
        "state_names": list(STATE_NAMES),
        "bits": [8, 6, 4],
        "block_sizes": [256, 1024, 4096],
        "scale_dtype": "float16",
        "quantizer": "symmetric_per_contiguous_flat_block_maxabs",
        "muon_state_only_polar_proxy_steps": 5,
        "lower_bit_storage_is_theoretical_packed": True,
    }
    if plan.get("protocol") != expected:
        raise ValueError("temporal-state quantization protocol changed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--parent-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    plan = json.loads(args.plan.read_text())
    validate(args, plan)
    started = time.time()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if int(checkpoint["next_iter"]) != int(plan["protocol"]["checkpoint_next_iter"]):
        raise ValueError("checkpoint next_iter changed")
    optimizers = checkpoint["optimizer"]["optimizers"]
    candidates = [
        (int(bits), int(block_size))
        for bits in plan["protocol"]["bits"]
        for block_size in plan["protocol"]["block_sizes"]
    ]
    rows: list[dict[str, Any]] = []
    for optimizer_index, role in OPTIMIZER_ROLES:
        state_dict = optimizers[optimizer_index]["state"]
        for layer, parameter_id in enumerate(sorted(state_dict, key=int)):
            parameter_state = state_dict[parameter_id]
            for state_name in STATE_NAMES:
                source_cpu = parameter_state[state_name].detach().float()
                source = source_cpu.to(args.device)
                original_polar = None
                if state_name == "momentum_buffer":
                    original_polar = zeropower_via_newtonschulz5(source, steps=5).float()
                for bits, block_size in candidates:
                    reconstructed, storage = blockwise_symmetric_quantize(
                        source, bits=bits, block_size=block_size
                    )
                    raw = tensor_metrics(source, reconstructed)
                    row: dict[str, Any] = {
                        "optimizer_index": optimizer_index,
                        "role": role,
                        "layer": layer,
                        "state_name": state_name,
                        "shape": list(source.shape),
                        "bits": bits,
                        "block_size": block_size,
                        **storage,
                        **flatten_metric("raw", raw),
                    }
                    if original_polar is not None:
                        quantized_polar = zeropower_via_newtonschulz5(
                            reconstructed, steps=5
                        ).float()
                        row.update(
                            flatten_metric(
                                "polar", tensor_metrics(original_polar, quantized_polar)
                            )
                        )
                    rows.append(row)
                    print(
                        f"role={role} layer={layer} state={state_name} bits={bits} "
                        f"block={block_size} recovery={raw['energy_recovery']:.8f} "
                        f"cosine={raw['cosine']:.8f}",
                        flush=True,
                    )
                del source, original_polar
    summaries: list[dict[str, Any]] = []
    for bits, block_size in candidates:
        selected = [
            row for row in rows
            if row["bits"] == bits and row["block_size"] == block_size
        ]
        raw = {
            state_name: aggregate_metrics(
                [row for row in selected if row["state_name"] == state_name], "raw"
            )
            for state_name in STATE_NAMES
        }
        momentum = [row for row in selected if row["state_name"] == "momentum_buffer"]
        persistent_bytes = sum(int(row["persistent_bytes"]) for row in selected)
        dense_bytes = sum(int(row["dense_fp32_bytes"]) for row in selected)
        summaries.append(
            {
                "bits": bits,
                "block_size": block_size,
                "raw": raw,
                "momentum_state_only_polar_proxy": aggregate_metrics(momentum, "polar"),
                "persistent_storage_bytes": persistent_bytes,
                "dense_fp32_bytes": dense_bytes,
                "persistent_storage_ratio_to_dense_fp32": persistent_bytes / dense_bytes,
                "storage_reduction_factor": dense_bytes / persistent_bytes,
            }
        )
    decision = classify(summaries, plan["decision_rule"])
    result = {
        "schema_version": RESULT_SCHEMA,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "decision": decision,
        "summaries": summaries,
        "rows": rows,
        "identity": {
            **plan["identity"],
            "plan_sha256": file_sha256(args.plan),
            "checkpoint_next_iter": int(checkpoint["next_iter"]),
        },
        "execution": {
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": str(Path(__file__).resolve()),
            "command": sys.argv,
            "device": args.device,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "direct_foreground_polling": True,
            "watchdog": False,
            "callback": False,
        },
        "authorization": {
            "exact_resume_codec_implementation": decision["selected_candidate"] is not None,
            "focused_tests": decision["selected_candidate"] is not None,
            "exact_config_mfu_vram_gate": False,
            "language_model_training": False,
            "larger_rung": False,
        },
    }
    args.output.mkdir(parents=True)
    result_path = args.output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": decision, "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
