#!/usr/bin/env python3
"""Gate persistent fixed-scale integer lattices for attention c_proj.

The preceding oracle showed that independently optimal low-bit codes reconstruct
individual states but inject temporally uncorrelated error into state
differences.  This zero-update follow-up uses one persistent block scale: an
oracle trajectory maximum is the feasibility ceiling, while a monotone running
maximum is the causal candidate.  No dense residual or error buffer is kept.
"""

from __future__ import annotations

import argparse
import json
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
from examples.nanogpt.analyze_attention_cproj_lowbit_trajectory_gate import (
    RESULT_SCHEMA as PREVIOUS_RESULT_SCHEMA,
    canonical_sha256,
    classify_candidate,
    git_commit,
    minimum_group_recovery,
    phase_name,
    theoretical_storage,
    trajectory_inventory,
    write_rows,
)


PLAN_SCHEMA = "mai_124m_attention_cproj_integer_lattice_gate_plan_v1"
RESULT_SCHEMA = "mai_124m_attention_cproj_integer_lattice_gate_result_v1"
SNAPSHOT_SCHEMA = "nanogpt_parameter_trajectory_v1"


def block_absmax(values: torch.Tensor, block_size: int) -> torch.Tensor:
    flattened = values.float().reshape(values.shape[0], -1)
    if flattened.shape[1] % int(block_size) != 0:
        raise ValueError("registered block size must divide each matrix")
    return flattened.reshape(flattened.shape[0], -1, int(block_size)).abs().amax(
        dim=-1, keepdim=True
    )


def fp16_scales(absmax: torch.Tensor, qmax: int) -> torch.Tensor:
    return (absmax / float(qmax)).to(torch.float16).float()


def quantize_on_lattice(
    values: torch.Tensor, scales: torch.Tensor, qmax: int
) -> tuple[torch.Tensor, torch.Tensor]:
    flattened = values.float().reshape(values.shape[0], -1)
    blocks = flattened.reshape(scales.shape[0], scales.shape[1], -1)
    safe = scales.clamp_min(torch.finfo(torch.float32).tiny)
    codes = torch.round(blocks / safe).clamp(-int(qmax), int(qmax)).to(torch.int8)
    codes = torch.where(scales > 0, codes, torch.zeros_like(codes))
    decoded = (codes.float() * scales).reshape_as(values)
    return codes, decoded


def load_state(
    path: Path,
    *,
    expected_step: int,
    layers: list[int],
    expected_identity: str,
) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != SNAPSHOT_SCHEMA:
        raise ValueError(f"unexpected snapshot schema: {path}")
    if int(payload.get("step")) != int(expected_step):
        raise ValueError(f"snapshot step mismatch: {path}")
    if str(payload.get("run_identity_sha256")) != expected_identity:
        raise ValueError(f"snapshot identity mismatch: {path}")
    return torch.stack(
        [payload["parameters"][parameter_name(layer)].float() for layer in layers]
    )


def validate_plan(plan: dict[str, Any], args: argparse.Namespace) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unexpected integer-lattice plan schema")
    if int(plan["protocol"]["parameter_updates"]) != 0:
        raise ValueError("this gate must make zero parameter updates")
    if any(bool(value) for value in plan["authorization"].values()):
        raise ValueError("preregistration must not pre-authorize downstream work")
    identity = plan["identity"]
    if Path(identity["trajectory_directory"]) != args.trajectory_dir:
        raise ValueError("trajectory directory differs from immutable plan")
    if file_sha256(Path(__file__)) != identity["entrypoint_sha256"]:
        raise ValueError("entrypoint hash differs from immutable plan")
    previous = json.loads(
        (Path(__file__).resolve().parents[2] / identity["previous_result"]).read_text()
    )
    if previous.get("schema_version") != PREVIOUS_RESULT_SCHEMA:
        raise ValueError("unexpected previous-result schema")
    if file_sha256(
        Path(__file__).resolve().parents[2] / identity["previous_result"]
    ) != identity["previous_result_sha256"]:
        raise ValueError("previous-result hash mismatch")


def summarize(
    *,
    rows: list[dict[str, Any]],
    codec_rows: list[dict[str, Any]],
    candidate: str,
    spec: dict[str, Any],
    terminal_step: int,
    elements: int,
    deterministic: bool,
) -> dict[str, Any]:
    selected = [row for row in rows if row["candidate"] == candidate]
    states = [row for row in selected if row["kind"] == "state"]
    chords = [row for row in selected if row["kind"] == "chord"]
    endpoint = [row for row in states if int(row["step_end"]) == terminal_step]
    changes = [
        row
        for row in codec_rows
        if row["candidate"] == candidate and row["code_churn_fraction"] is not None
    ]
    return {
        "bits": int(spec["bits"]),
        "qmax": int(spec["qmax"]),
        "block_size": int(spec["block_size"]),
        "scale_policy": str(spec["scale_policy"]),
        "causal": bool(spec["causal"]),
        "state": aggregate(states),
        "endpoint_state": aggregate(endpoint),
        "chord": aggregate(chords),
        "minimum_layer_state_recovery": minimum_group_recovery(states, "layer"),
        "minimum_snapshot_state_recovery": minimum_group_recovery(
            states, "step_end"
        ),
        "minimum_layer_chord_recovery": minimum_group_recovery(chords, "layer"),
        "minimum_phase_chord_recovery": minimum_group_recovery(chords, "phase"),
        "mean_code_churn_fraction": sum(
            float(row["code_churn_fraction"]) for row in changes
        )
        / max(len(changes), 1),
        "maximum_code_churn_fraction": max(
            (float(row["code_churn_fraction"]) for row in changes), default=0.0
        ),
        "mean_scale_growth_fraction": sum(
            float(row["scale_growth_fraction"]) for row in changes
        )
        / max(len(changes), 1),
        "deterministic_decode": bool(deterministic),
        "storage": theoretical_storage(
            elements=elements,
            bits=int(spec["bits"]),
            block_size=int(spec["block_size"]),
        ),
    }


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
    run_identity = str(identity["trajectory_run_identity_sha256"])
    candidates = protocol["candidates"]

    inventory, inventory_sha = trajectory_inventory(args.trajectory_dir)
    if len(inventory) != int(identity["trajectory_file_count"]):
        raise ValueError("trajectory file count mismatch")
    if sum(int(item["size"]) for item in inventory) != int(
        identity["trajectory_total_bytes"]
    ):
        raise ValueError("trajectory byte count mismatch")
    if inventory_sha != str(identity["trajectory_inventory_sha256"]):
        raise ValueError("trajectory inventory hash mismatch")

    initial_cpu = load_state(
        args.trajectory_dir / f"step_{steps[0]:06d}.pt",
        expected_step=steps[0],
        layers=layers,
        expected_identity=run_identity,
    )
    block_size = int(protocol["block_size"])
    trajectory_absmax = torch.zeros(
        len(layers), initial_cpu[0].numel() // block_size, 1
    )
    for step in steps[1:]:
        current = load_state(
            args.trajectory_dir / f"step_{step:06d}.pt",
            expected_step=step,
            layers=layers,
            expected_identity=run_identity,
        )
        trajectory_absmax.copy_(
            torch.maximum(
                trajectory_absmax,
                block_absmax(current - initial_cpu, block_size),
            )
        )
    oracle_scales = {
        name: fp16_scales(trajectory_absmax.to(args.device), int(spec["qmax"]))
        for name, spec in candidates.items()
        if str(spec["scale_policy"]) == "trajectory_max_oracle"
    }

    rows: list[dict[str, Any]] = []
    codec_rows: list[dict[str, Any]] = []
    previous_dense: torch.Tensor | None = None
    previous_decoded: dict[str, torch.Tensor] = {}
    previous_codes: dict[str, torch.Tensor] = {}
    running_absmax: dict[str, torch.Tensor] = {}
    previous_scales: dict[str, torch.Tensor] = {}
    deterministic = {name: True for name in candidates}
    initial = initial_cpu.to(args.device)

    for step_index, step in enumerate(steps):
        current = load_state(
            args.trajectory_dir / f"step_{step:06d}.pt",
            expected_step=step,
            layers=layers,
            expected_identity=run_identity,
        ).to(args.device)
        displacement = current - initial
        current_absmax = block_absmax(displacement, block_size)
        for candidate, spec in candidates.items():
            qmax = int(spec["qmax"])
            policy = str(spec["scale_policy"])
            if policy == "trajectory_max_oracle":
                scales = oracle_scales[candidate]
            elif policy == "running_max":
                if candidate not in running_absmax:
                    running_absmax[candidate] = torch.zeros_like(current_absmax)
                running_absmax[candidate].copy_(
                    torch.maximum(running_absmax[candidate], current_absmax)
                )
                scales = fp16_scales(running_absmax[candidate], qmax)
            else:
                raise ValueError(f"unsupported scale policy: {policy}")
            codes, decoded = quantize_on_lattice(displacement, scales, qmax)
            codes2, decoded2 = quantize_on_lattice(displacement, scales, qmax)
            deterministic[candidate] = deterministic[candidate] and bool(
                torch.equal(codes, codes2) and torch.equal(decoded, decoded2)
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
                churn = float((codes != previous_codes[candidate]).float().mean())
                scale_growth = float(
                    (scales > previous_scales[candidate]).float().mean()
                )
                codec_rows.append(
                    {
                        "candidate": candidate,
                        "step_start": steps[step_index - 1],
                        "step_end": step,
                        "code_churn_fraction": churn,
                        "scale_growth_fraction": scale_growth,
                    }
                )
            else:
                codec_rows.append(
                    {
                        "candidate": candidate,
                        "step_start": 0,
                        "step_end": 0,
                        "code_churn_fraction": None,
                        "scale_growth_fraction": None,
                    }
                )
            previous_decoded[candidate] = decoded.detach().clone()
            previous_codes[candidate] = codes.detach().clone()
            previous_scales[candidate] = scales.detach().clone()
        previous_dense = current.detach().clone()
        print(json.dumps({"loaded_step": step, "snapshots": len(steps)}), flush=True)

    summaries: dict[str, Any] = {}
    decisions: dict[str, Any] = {}
    causal_passing: list[str] = []
    oracle_passing: list[str] = []
    elements = len(layers) * int(protocol["matrix_elements_per_layer"])
    for candidate, spec in candidates.items():
        summary = summarize(
            rows=rows,
            codec_rows=codec_rows,
            candidate=candidate,
            spec=spec,
            terminal_step=steps[-1],
            elements=elements,
            deterministic=deterministic[candidate],
        )
        passed, checks = classify_candidate(
            summary, plan["decision_rule"]["thresholds"]
        )
        eligible = bool(spec["causal"])
        decisions[candidate] = {
            "passed_measurement_gate": passed,
            "eligible_for_selection": eligible,
            "passed_selection_gate": passed and eligible,
            "checks": checks,
        }
        summaries[candidate] = summary
        if passed and eligible:
            causal_passing.append(candidate)
        if passed and not eligible:
            oracle_passing.append(candidate)

    selected = (
        min(
            causal_passing,
            key=lambda name: (
                int(summaries[name]["bits"]),
                -float(summaries[name]["chord"]["fixed_scale_recovery"]),
            ),
        )
        if causal_passing
        else None
    )
    if selected is not None:
        classification = "PROMOTE_ATTENTION_CPROJ_CAUSAL_INTEGER_LATTICE"
    elif oracle_passing:
        classification = "ORACLE_ONLY_ATTENTION_CPROJ_INTEGER_LATTICE"
    else:
        classification = "REJECT_ATTENTION_CPROJ_INTEGER_LATTICE"
    result = {
        "schema_version": RESULT_SCHEMA,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "classification": classification,
        "parameter_updates": 0,
        "source_commit": git_commit(),
        "elapsed_seconds": time.time() - started,
        "identity": {
            "trajectory_run_identity_sha256": run_identity,
            "trajectory_inventory_sha256": inventory_sha,
            "plan_sha256": file_sha256(args.plan),
            "previous_result_sha256": identity["previous_result_sha256"],
            "oracle_scale_sha256": canonical_sha256(
                {
                    name: canonical_sha256(scales.cpu().numpy().tobytes().hex())
                    for name, scales in oracle_scales.items()
                }
            ),
        },
        "summaries": summaries,
        "decisions": decisions,
        "selected_candidate": selected,
        "oracle_only_passing": oracle_passing,
        "authorization": {
            "prototype_implementation": selected is not None,
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
