#!/usr/bin/env python3
"""Gate a compact low-bit residual over the dense attention c_proj path.

This is a zero-update trajectory oracle.  Each saved dense ``attn.c_proj``
state is represented as an exactly reproducible GPT-initialization base plus a
blockwise low-bit displacement.  The gate measures both state reconstruction
and the motion induced by differences of consecutive decoded states.  Good
state fits alone are insufficient: a persistent compact state must also follow
the dense training chords rather than jitter between independently good codes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_attention_cproj_fresh_residual_gate import (
    aggregate,
    file_sha256,
    metrics,
    parameter_name,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_SCHEMA = "mai_124m_attention_cproj_lowbit_trajectory_gate_plan_v1"
RESULT_SCHEMA = "mai_124m_attention_cproj_lowbit_trajectory_gate_result_v1"
SNAPSHOT_SCHEMA = "nanogpt_parameter_trajectory_v1"


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def trajectory_inventory(directory: Path) -> tuple[list[dict[str, Any]], str]:
    items: list[dict[str, Any]] = []
    for path in sorted(directory.glob("step_*.pt")):
        items.append(
            {
                "name": path.name,
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return items, canonical_sha256(items)


def encode_blocks(
    values: torch.Tensor,
    *,
    codec: str,
    block_size: int,
    ternary_threshold_rms: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return integer codes, FP32 scales, and their deterministic decode."""

    if values.ndim != 3:
        raise ValueError("values must be [members, rows, columns]")
    flattened = values.float().reshape(values.shape[0], -1)
    if flattened.shape[1] % int(block_size) != 0:
        raise ValueError("registered block size must divide each matrix")
    blocks = flattened.reshape(flattened.shape[0], -1, int(block_size))
    if codec == "binary":
        codes = torch.where(blocks >= 0, 1, -1).to(torch.int8)
        scales = blocks.abs().mean(dim=-1, keepdim=True)
    elif codec == "ternary":
        rms = blocks.square().mean(dim=-1, keepdim=True).sqrt()
        active = blocks.abs() >= float(ternary_threshold_rms) * rms
        codes = (torch.sign(blocks) * active).to(torch.int8)
        scales = (
            (blocks.abs() * active).sum(dim=-1, keepdim=True)
            / active.sum(dim=-1, keepdim=True).clamp_min(1)
        )
    elif codec == "int4":
        scales = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30) / 7.0
        codes = torch.round(blocks / scales).clamp(-7, 7).to(torch.int8)
    else:
        raise ValueError(f"unsupported codec: {codec}")
    decoded = (codes.float() * scales).reshape_as(values)
    return codes, scales, decoded


def theoretical_storage(
    *, elements: int, bits: int, block_size: int
) -> dict[str, int | float]:
    blocks = math.ceil(int(elements) / int(block_size))
    code_bytes = math.ceil(int(elements) * int(bits) / 8)
    scale_bytes = blocks * 2
    dense_fp32_bytes = int(elements) * 4
    compact_bytes = code_bytes + scale_bytes
    return {
        "elements": int(elements),
        "bits_per_code": int(bits),
        "blocks": blocks,
        "code_bytes": code_bytes,
        "fp16_scale_bytes": scale_bytes,
        "compact_bytes": compact_bytes,
        "dense_fp32_bytes": dense_fp32_bytes,
        "storage_ratio": compact_bytes / dense_fp32_bytes,
        "storage_reduction_factor": dense_fp32_bytes / compact_bytes,
    }


def phase_name(step: int, boundaries: list[int]) -> str:
    for index, boundary in enumerate(boundaries):
        if step <= boundary:
            return f"phase_{index}"
    raise ValueError(f"step {step} exceeds registered phase boundaries")


def minimum_group_recovery(
    rows: list[dict[str, Any]], field: str
) -> float:
    groups = sorted({str(row[field]) for row in rows})
    return min(
        float(aggregate([row for row in rows if str(row[field]) == group])[
            "fixed_scale_recovery"
        ])
        for group in groups
    )


def classify_candidate(
    summary: dict[str, Any], thresholds: dict[str, float]
) -> tuple[bool, dict[str, bool]]:
    checks = {
        "state_aggregate": float(summary["state"]["fixed_scale_recovery"])
        >= float(thresholds["state_aggregate_recovery_minimum"]),
        "state_endpoint": float(summary["endpoint_state"]["fixed_scale_recovery"])
        >= float(thresholds["state_endpoint_recovery_minimum"]),
        "state_every_layer": float(summary["minimum_layer_state_recovery"])
        >= float(thresholds["state_minimum_layer_recovery"]),
        "state_every_snapshot": float(summary["minimum_snapshot_state_recovery"])
        >= float(thresholds["state_minimum_snapshot_recovery"]),
        "chord_aggregate": float(summary["chord"]["fixed_scale_recovery"])
        >= float(thresholds["chord_aggregate_recovery_minimum"]),
        "chord_cosine": float(summary["chord"]["cosine"])
        >= float(thresholds["chord_aggregate_cosine_minimum"]),
        "chord_every_phase": float(summary["minimum_phase_chord_recovery"])
        >= float(thresholds["chord_minimum_phase_recovery"]),
        "chord_every_layer": float(summary["minimum_layer_chord_recovery"])
        >= float(thresholds["chord_minimum_layer_recovery"]),
        "deterministic_decode": bool(summary["deterministic_decode"]),
        "storage": float(summary["storage"]["storage_ratio"])
        <= float(thresholds["maximum_storage_ratio"]),
    }
    return all(checks.values()), checks


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_plan(plan: dict[str, Any], args: argparse.Namespace) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unexpected trajectory-gate plan schema")
    if int(plan["protocol"]["parameter_updates"]) != 0:
        raise ValueError("this gate must make zero parameter updates")
    if any(bool(value) for value in plan["authorization"].values()):
        raise ValueError("preregistration must not pre-authorize downstream work")
    identity = plan["identity"]
    if Path(identity["trajectory_directory"]) != args.trajectory_dir:
        raise ValueError("trajectory directory differs from immutable plan")
    if file_sha256(Path(__file__)) != identity["entrypoint_sha256"]:
        raise ValueError("entrypoint hash differs from immutable plan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--trajectory-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    validate_plan(plan, args)
    if args.output_dir.exists():
        raise FileExistsError(f"output already exists: {args.output_dir}")
    started = time.time()
    protocol = plan["protocol"]
    identity = plan["identity"]
    steps = [int(value) for value in protocol["trajectory_steps"]]
    layers = [int(value) for value in protocol["layers"]]
    boundaries = [int(value) for value in protocol["phase_end_steps"]]

    inventory, inventory_sha = trajectory_inventory(args.trajectory_dir)
    if len(inventory) != int(identity["trajectory_file_count"]):
        raise ValueError("trajectory file count mismatch")
    if sum(int(item["size"]) for item in inventory) != int(
        identity["trajectory_total_bytes"]
    ):
        raise ValueError("trajectory byte count mismatch")
    if inventory_sha != str(identity["trajectory_inventory_sha256"]):
        raise ValueError("trajectory inventory hash mismatch")

    candidates = protocol["candidates"]
    rows: list[dict[str, Any]] = []
    codec_rows: list[dict[str, Any]] = []
    previous_dense: torch.Tensor | None = None
    previous_decoded: dict[str, torch.Tensor] = {}
    previous_codes: dict[str, torch.Tensor] = {}
    previous_scales: dict[str, torch.Tensor] = {}
    initial: torch.Tensor | None = None
    observed_identity: str | None = None
    deterministic = {name: True for name in candidates}

    for step_index, step in enumerate(steps):
        path = args.trajectory_dir / f"step_{step:06d}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != SNAPSHOT_SCHEMA:
            raise ValueError(f"unexpected snapshot schema: {path}")
        if int(payload.get("step")) != step:
            raise ValueError(f"snapshot step mismatch: {path}")
        run_identity = str(payload.get("run_identity_sha256"))
        if observed_identity is None:
            observed_identity = run_identity
        elif observed_identity != run_identity:
            raise ValueError("trajectory snapshots do not share one identity")
        current = torch.stack(
            [payload["parameters"][parameter_name(layer)].float() for layer in layers]
        ).to(args.device)
        del payload
        if initial is None:
            initial = current.clone()
        displacement = current - initial

        for candidate, spec in candidates.items():
            codes, scales, decoded = encode_blocks(
                displacement,
                codec=str(spec["codec"]),
                block_size=int(spec["block_size"]),
                ternary_threshold_rms=float(protocol["ternary_threshold_rms"]),
            )
            codes2, scales2, decoded2 = encode_blocks(
                displacement,
                codec=str(spec["codec"]),
                block_size=int(spec["block_size"]),
                ternary_threshold_rms=float(protocol["ternary_threshold_rms"]),
            )
            deterministic[candidate] = deterministic[candidate] and bool(
                torch.equal(codes, codes2)
                and torch.equal(scales, scales2)
                and torch.equal(decoded, decoded2)
            )
            if step_index > 0:
                for index, layer in enumerate(layers):
                    rows.append(
                        {
                            "kind": "state",
                            "candidate": candidate,
                            "layer": layer,
                            "step_start": 0,
                            "step_end": step,
                            "phase": phase_name(step, boundaries),
                            **metrics(displacement[index], decoded[index]),
                        }
                    )
                assert previous_dense is not None
                chord = current - previous_dense
                decoded_chord = decoded - previous_decoded[candidate]
                for index, layer in enumerate(layers):
                    rows.append(
                        {
                            "kind": "chord",
                            "candidate": candidate,
                            "layer": layer,
                            "step_start": steps[step_index - 1],
                            "step_end": step,
                            "phase": phase_name(step, boundaries),
                            **metrics(chord[index], decoded_chord[index]),
                        }
                    )
                if step_index > 1:
                    churn = float((codes != previous_codes[candidate]).float().mean())
                    relative_scale_change = float(
                        (
                            (scales - previous_scales[candidate]).abs()
                            / previous_scales[candidate].abs().clamp_min(1e-30)
                        ).mean()
                    )
                    codec_rows.append(
                        {
                            "candidate": candidate,
                            "step_start": steps[step_index - 1],
                            "step_end": step,
                            "code_churn_fraction": churn,
                            "mean_relative_scale_change": relative_scale_change,
                        }
                    )
                else:
                    codec_rows.append(
                        {
                            "candidate": candidate,
                            "step_start": 0,
                            "step_end": step,
                            "code_churn_fraction": None,
                            "mean_relative_scale_change": None,
                        }
                    )
            else:
                codec_rows.append(
                    {
                        "candidate": candidate,
                        "step_start": 0,
                        "step_end": 0,
                        "code_churn_fraction": None,
                        "mean_relative_scale_change": None,
                    }
                )
            previous_decoded[candidate] = decoded.detach().clone()
            previous_codes[candidate] = codes.detach().clone()
            previous_scales[candidate] = scales.detach().clone()
        previous_dense = current.detach().clone()
        print(json.dumps({"loaded_step": step, "snapshots": len(steps)}), flush=True)

    if observed_identity != str(identity["trajectory_run_identity_sha256"]):
        raise ValueError("trajectory run identity mismatch")

    summaries: dict[str, Any] = {}
    decisions: dict[str, Any] = {}
    passing: list[str] = []
    elements = len(layers) * int(protocol["matrix_elements_per_layer"])
    terminal = steps[-1]
    for candidate, spec in candidates.items():
        selected = [row for row in rows if row["candidate"] == candidate]
        state = [row for row in selected if row["kind"] == "state"]
        chord = [row for row in selected if row["kind"] == "chord"]
        endpoint = [row for row in state if int(row["step_end"]) == terminal]
        codec_selected = [
            row
            for row in codec_rows
            if row["candidate"] == candidate and row["code_churn_fraction"] is not None
        ]
        summary = {
            "codec": str(spec["codec"]),
            "block_size": int(spec["block_size"]),
            "state": aggregate(state),
            "endpoint_state": aggregate(endpoint),
            "chord": aggregate(chord),
            "minimum_layer_state_recovery": minimum_group_recovery(state, "layer"),
            "minimum_snapshot_state_recovery": minimum_group_recovery(state, "step_end"),
            "minimum_layer_chord_recovery": minimum_group_recovery(chord, "layer"),
            "minimum_phase_chord_recovery": minimum_group_recovery(chord, "phase"),
            "mean_code_churn_fraction": sum(
                float(row["code_churn_fraction"]) for row in codec_selected
            )
            / max(len(codec_selected), 1),
            "maximum_code_churn_fraction": max(
                (float(row["code_churn_fraction"]) for row in codec_selected),
                default=0.0,
            ),
            "mean_relative_scale_change": sum(
                float(row["mean_relative_scale_change"]) for row in codec_selected
            )
            / max(len(codec_selected), 1),
            "deterministic_decode": deterministic[candidate],
            "storage": theoretical_storage(
                elements=elements,
                bits=int(spec["bits"]),
                block_size=int(spec["block_size"]),
            ),
        }
        passed, checks = classify_candidate(
            summary, plan["decision_rule"]["thresholds"]
        )
        summaries[candidate] = summary
        decisions[candidate] = {"passed": passed, "checks": checks}
        if passed:
            passing.append(candidate)

    selected_candidate = (
        min(
            passing,
            key=lambda name: (
                float(summaries[name]["storage"]["storage_ratio"]),
                -float(summaries[name]["chord"]["fixed_scale_recovery"]),
            ),
        )
        if passing
        else None
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "classification": (
            "PROMOTE_ATTENTION_CPROJ_LOWBIT_TERMINAL_REPLAY"
            if selected_candidate is not None
            else "REJECT_ATTENTION_CPROJ_LOWBIT_PERSISTENT_STATE"
        ),
        "parameter_updates": 0,
        "source_commit": git_commit(),
        "elapsed_seconds": time.time() - started,
        "identity": {
            "trajectory_run_identity_sha256": observed_identity,
            "trajectory_inventory_sha256": inventory_sha,
            "plan_sha256": file_sha256(args.plan),
        },
        "summaries": summaries,
        "decisions": decisions,
        "selected_candidate": selected_candidate,
        "authorization": {
            "terminal_fixed_checkpoint_replay": selected_candidate is not None,
            "model_implementation": False,
            "mfu_preflight": False,
            "language_model_training": False,
            "larger_rung": False,
        },
    }
    args.output_dir.mkdir(parents=True)
    write_rows(args.output_dir / "cells.csv", rows)
    write_rows(args.output_dir / "codec_rows.csv", codec_rows)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
