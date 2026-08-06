#!/usr/bin/env python3
"""Replay mature c_proj phase chords through a bilateral capacity oracle."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.nanogpt.analyze_mlp_activation_update_alignment import (
    load_snapshot,
    model_from_snapshot,
)
from examples.nanogpt.analyze_mlp_cproj_activation_weighted_output_selector import (
    file_sha256,
    git_commit,
    parameter_name,
)
from examples.nanogpt.analyze_mlp_cproj_diagonal_kfac_selector import (
    acquisition_artifact_hashes,
    require_full_state_snapshot,
)
from examples.nanogpt.analyze_mlp_cproj_teacher_forced_bilateral_full_carry import (
    structured_step,
)
from examples.nanogpt.analyze_parameter_trajectory import write_csv
from examples.nanogpt.train import (
    fixed_eval_indices_digest,
    make_fixed_eval_indices,
    require_block_fht_native_extension,
)
from examples.nanogpt.verify_full_state_functional_replay import (
    evaluate_validation_ce,
)


PLAN_SCHEMA = "mai_124m_mlp_cproj_5tpp_integrated_trajectory_capacity_plan_v1"
RESULT_SCHEMA = "mai_124m_mlp_cproj_5tpp_integrated_trajectory_capacity_result_v1"
PHASES = ((0, 594), (594, 1188), (1188, 1782), (1782, 2373))
LAYERS = tuple(range(8))
CADENCES = (1, 8, 32)


def validate_plan(plan: dict[str, Any]) -> None:
    analysis = plan.get("analysis", {})
    observed = {
        "schema_version": plan.get("schema_version"),
        "parameter_updates": analysis.get("parameter_updates"),
        "layers": analysis.get("layers"),
        "phases": analysis.get("phases"),
        "cadences": analysis.get("straight_chord_substeps"),
        "chart": analysis.get("chart"),
        "eval": analysis.get("fixed_validation"),
        "thresholds": plan.get("decision_rule", {}).get("thresholds"),
    }
    expected = {
        "schema_version": PLAN_SCHEMA,
        "parameter_updates": 0,
        "layers": list(LAYERS),
        "phases": [list(value) for value in PHASES],
        "cadences": list(CADENCES),
        "chart": {
            "hidden_parent_stages": 64,
            "hidden_residual_stages": 24,
            "output_stages": 32,
            "neighbors": 64,
            "matching_seed": 20260807,
            "feedback": "full within each oracle path",
            "weight_decay": 0.0,
            "learning_rate": 1.0,
        },
        "eval": {
            "split": "validation",
            "eval_iters": 400,
            "eval_batch_size": 16,
            "block_size": 1024,
            "eval_seed": 20260715,
            "fixed_eval_indices_sha256": "5ca31b59768e43de808ad5e206ed152a4a0a3515ad68d29a0b2338c4db140747",
        },
        "thresholds": {
            "phase_straight32_geometric_recovery_minimum": 0.85,
            "phase_straight32_maximum_validation_ce_gap": 0.01,
            "phase_straight32_terminal_validation_ce_gap": 0.005,
            "sequential_straight32_terminal_validation_ce_gap": 0.02,
            "straight32_terminal_ce_improvement_over_direct_minimum": 0.01,
        },
    }
    if observed != expected:
        raise ValueError("integrated-trajectory plan does not match v1 contract")
    authorization = plan.get("authorization", {})
    if authorization.get("run_zero_update_integrated_trajectory_analysis") is not True:
        raise ValueError("integrated-trajectory analysis is not authorized")
    for key in (
        "acquire_higher_cadence_trajectory",
        "implement_candidate_structure",
        "run_exact_config_mfu",
        "run_language_model_training",
        "larger_rung",
    ):
        if authorization.get(key) is not False:
            raise ValueError(f"plan must keep {key} false")


def geometric_metrics(
    start: torch.Tensor,
    dense_end: torch.Tensor,
    candidate_end: torch.Tensor,
) -> dict[str, float]:
    chord = dense_end.float() - start.float()
    candidate = candidate_end.float() - start.float()
    error = dense_end.float() - candidate_end.float()
    energy = chord.double().square().sum().clamp_min(1e-30)
    candidate_norm = candidate.double().norm().clamp_min(1e-30)
    cosine = (chord.double() * candidate.double()).sum() / (
        chord.double().norm().clamp_min(1e-30) * candidate_norm
    )
    return {
        "chord_energy": float(energy),
        "endpoint_error_energy": float(error.double().square().sum()),
        "endpoint_recovery": float(
            1.0 - error.double().square().sum() / energy
        ),
        "endpoint_cosine": float(cosine),
    }


def fit_straight_chord(
    start: torch.Tensor,
    chord: torch.Tensor,
    *,
    substeps: int,
    layer: int,
    phase_index: int,
    neighbors: int,
    seed: int,
    initial_feedback: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    current = start.float().clone()
    feedback = (
        torch.zeros_like(current)
        if initial_feedback is None
        else initial_feedback.float().clone()
    )
    requested = chord.float() / float(substeps)
    recoveries: list[float] = []
    for substep in range(substeps):
        current, feedback, recovery = structured_step(
            current,
            requested,
            feedback,
            output_stages=32,
            learning_rate=1.0,
            weight_decay=0.0,
            neighbors=neighbors,
            seed=seed + layer * 100000 + phase_index * 1000 + substep * 10,
        )
        recoveries.append(recovery)
    return current, feedback, recoveries


def classify(metrics: dict[str, float], thresholds: dict[str, float]) -> dict[str, Any]:
    capacity_gates = {
        "geometric_recovery": metrics["phase_straight32_geometric_recovery"]
        >= thresholds["phase_straight32_geometric_recovery_minimum"],
        "maximum_validation_gap": metrics[
            "phase_straight32_maximum_validation_ce_gap"
        ]
        <= thresholds["phase_straight32_maximum_validation_ce_gap"],
        "terminal_validation_gap": metrics[
            "phase_straight32_terminal_validation_ce_gap"
        ]
        <= thresholds["phase_straight32_terminal_validation_ce_gap"],
        "cadence_improvement": metrics[
            "straight32_terminal_ce_improvement_over_direct"
        ]
        >= thresholds[
            "straight32_terminal_ce_improvement_over_direct_minimum"
        ],
    }
    transport_gate = metrics[
        "sequential_straight32_terminal_validation_ce_gap"
    ] <= thresholds["sequential_straight32_terminal_validation_ce_gap"]
    capacity_passed = all(capacity_gates.values())
    if capacity_passed and transport_gate:
        classification = "MATURE_PHASE_CAPACITY_AND_COARSE_TRANSPORT_PASS"
    elif capacity_passed:
        classification = "MATURE_PHASE_CAPACITY_PASS_TRANSPORT_FAIL"
    else:
        classification = "REJECT_MATURE_PHASE_BILATERAL_CAPACITY_AT_TESTED_CADENCE"
    return {
        "classification": classification,
        "capacity_passed": capacity_passed,
        "transport_passed": transport_gate,
        "capacity_gates": capacity_gates,
        "transport_gate": transport_gate,
        "authorization": {
            "acquire_higher_cadence_trajectory": bool(
                capacity_passed and not transport_gate
            ),
            "candidate_structure_theory": bool(capacity_passed and transport_gate),
            "implement_candidate_structure": False,
            "run_exact_config_mfu": False,
            "run_language_model_training": False,
            "larger_rung": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--acquisition-result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    acquisition = json.loads(args.acquisition_result.read_text())
    if file_sha256(args.acquisition_result) != plan["identity"][
        "acquisition_result_sha256"
    ]:
        raise ValueError("acquisition result SHA-256 mismatch")
    if acquisition.get("classification") != (
        "ACCEPTED_PARENT_EQUIVALENT_EXACT_FUNCTIONAL_REPLAY"
    ):
        raise ValueError("acquisition is not functionally accepted")
    if acquisition["functional_replay"]["result_sha256"] != plan["identity"][
        "functional_replay_result_sha256"
    ]:
        raise ValueError("functional replay SHA-256 mismatch")
    if file_sha256(args.config) != plan["identity"]["config_sha256"]:
        raise ValueError("config SHA-256 mismatch")
    config = json.loads(args.config.read_text())
    require_block_fht_native_extension(
        bool(config["block_fht_native_extension_required"])
    )
    snapshot_hashes = acquisition_artifact_hashes(acquisition, "snapshots")
    snapshots: dict[int, dict[str, Any]] = {}
    weights: dict[int, dict[int, torch.Tensor]] = {}
    identity = plan["identity"]["run_identity_sha256"]
    for step in sorted({value for phase in PHASES for value in phase}):
        path = args.snapshot_dir / f"step_{step:06d}.pt"
        if file_sha256(path) != snapshot_hashes[str(step)]:
            raise ValueError(f"snapshot SHA-256 mismatch at step {step}")
        snapshot = load_snapshot(path)
        require_full_state_snapshot(snapshot)
        if snapshot["run_identity_sha256"] != identity:
            raise ValueError("snapshot run identity mismatch")
        snapshots[step] = snapshot
        weights[step] = {
            layer: snapshot["parameters"][parameter_name(layer)].float().clone()
            for layer in LAYERS
        }

    started = time.time()
    chart = plan["analysis"]["chart"]
    neighbors = int(chart["neighbors"])
    seed = int(chart["matching_seed"])
    cells: list[dict[str, Any]] = []
    phase_states: dict[tuple[int, str, int], torch.Tensor] = {}
    sequential = {
        layer: weights[PHASES[0][0]][layer].to(args.device)
        for layer in LAYERS
    }
    sequential_feedback = {
        layer: torch.zeros_like(sequential[layer]) for layer in LAYERS
    }
    for phase_index, (start_step, end_step) in enumerate(PHASES):
        for layer in LAYERS:
            dense_start = weights[start_step][layer].to(args.device)
            dense_end = weights[end_step][layer].to(args.device)
            chord = dense_end - dense_start
            for cadence in CADENCES:
                candidate, feedback, recoveries = fit_straight_chord(
                    dense_start,
                    chord,
                    substeps=cadence,
                    layer=layer,
                    phase_index=phase_index,
                    neighbors=neighbors,
                    seed=seed,
                )
                variant = f"phase_straight{cadence}"
                phase_states[(end_step, variant, layer)] = candidate.cpu()
                cells.append(
                    {
                        "phase_start": start_step,
                        "phase_end": end_step,
                        "layer": layer,
                        "variant": variant,
                        "substeps": cadence,
                        "mean_requested_update_recovery": sum(recoveries)
                        / len(recoveries),
                        "terminal_feedback_fro": float(feedback.norm()),
                        **geometric_metrics(dense_start, dense_end, candidate),
                    }
                )

            candidate, feedback, recoveries = fit_straight_chord(
                sequential[layer],
                chord,
                substeps=32,
                layer=layer,
                phase_index=phase_index,
                neighbors=neighbors,
                seed=seed + 50000000,
                initial_feedback=sequential_feedback[layer],
            )
            sequential[layer] = candidate
            sequential_feedback[layer] = feedback
            phase_states[(end_step, "sequential_straight32", layer)] = candidate.cpu()
            cells.append(
                {
                    "phase_start": start_step,
                    "phase_end": end_step,
                    "layer": layer,
                    "variant": "sequential_straight32",
                    "substeps": 32,
                    "mean_requested_update_recovery": sum(recoveries)
                    / len(recoveries),
                    "terminal_feedback_fro": float(feedback.norm()),
                    **geometric_metrics(
                        weights[PHASES[0][0]][layer].to(args.device),
                        dense_end,
                        candidate,
                    ),
                }
            )

    fixed_spec = plan["analysis"]["fixed_validation"]
    fixed = make_fixed_eval_indices(
        Path(config["data_dir"]),
        int(fixed_spec["eval_batch_size"]),
        int(fixed_spec["block_size"]),
        int(fixed_spec["eval_iters"]),
        int(fixed_spec["eval_seed"]),
    )
    if fixed_eval_indices_digest(fixed) != fixed_spec["fixed_eval_indices_sha256"]:
        raise ValueError("fixed validation indices SHA-256 mismatch")
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[config["dtype"]]
    eval_args = SimpleNamespace(**config)
    eval_args._ptdtype = dtype
    ctx = (
        contextlib.nullcontext()
        if "cuda" not in args.device
        else torch.amp.autocast(device_type="cuda", dtype=dtype)
    )
    parent_ce = {
        int(row["step"]): float(row["replayed_validation_ce"])
        for row in acquisition["functional_replay"]["rows"]
    }
    evaluation_rows: list[dict[str, Any]] = []
    for _start_step, end_step in PHASES:
        variants = ["phase_straight32"]
        if end_step == PHASES[-1][1]:
            variants.extend(
                ["phase_straight1", "phase_straight8", "sequential_straight32"]
            )
        model = model_from_snapshot(snapshots[end_step], args.device)
        for variant in variants:
            with torch.no_grad():
                for layer in LAYERS:
                    target = model.transformer.h[layer].mlp.c_proj.weight
                    target.copy_(
                        phase_states[(end_step, variant, layer)].to(
                            device=target.device, dtype=target.dtype
                        )
                    )
            value = evaluate_validation_ce(
                model,
                data_dir=Path(config["data_dir"]),
                args=eval_args,
                indices=fixed["val"],
                ctx=ctx,
            )
            evaluation_rows.append(
                {
                    "step": end_step,
                    "variant": variant,
                    "validation_ce": value,
                    "parent_validation_ce": parent_ce[end_step],
                    "validation_ce_gap": value - parent_ce[end_step],
                }
            )
        del model
        torch.cuda.empty_cache()

    selected = [row for row in cells if row["variant"] == "phase_straight32"]
    total_energy = sum(float(row["chord_energy"]) for row in selected)
    total_error = sum(float(row["endpoint_error_energy"]) for row in selected)
    phase_eval = [
        row for row in evaluation_rows if row["variant"] == "phase_straight32"
    ]
    terminal = PHASES[-1][1]
    eval_index = {
        (int(row["step"]), str(row["variant"])): row for row in evaluation_rows
    }
    metrics = {
        "phase_straight32_geometric_recovery": 1.0
        - total_error / max(total_energy, 1e-30),
        "phase_straight32_minimum_layer_phase_recovery": min(
            float(row["endpoint_recovery"]) for row in selected
        ),
        "phase_straight32_maximum_validation_ce_gap": max(
            float(row["validation_ce_gap"]) for row in phase_eval
        ),
        "phase_straight32_terminal_validation_ce_gap": float(
            eval_index[(terminal, "phase_straight32")]["validation_ce_gap"]
        ),
        "sequential_straight32_terminal_validation_ce_gap": float(
            eval_index[(terminal, "sequential_straight32")]["validation_ce_gap"]
        ),
        "straight32_terminal_ce_improvement_over_direct": float(
            eval_index[(terminal, "phase_straight1")]["validation_ce"]
            - eval_index[(terminal, "phase_straight32")]["validation_ce"]
        ),
        "straight32_terminal_ce_improvement_over_straight8": float(
            eval_index[(terminal, "phase_straight8")]["validation_ce"]
            - eval_index[(terminal, "phase_straight32")]["validation_ce"]
        ),
    }
    decision = classify(metrics, plan["decision_rule"]["thresholds"])
    args.output.mkdir(parents=True)
    cells_path = args.output / "integrated_trajectory_cells.csv"
    eval_path = args.output / "integrated_trajectory_fixed_ce.csv"
    result_path = args.output / "integrated_trajectory_result.json"
    write_csv(cells_path, cells)
    write_csv(eval_path, evaluation_rows)
    result = {
        "schema_version": RESULT_SCHEMA,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "classification": decision["classification"],
        "execution": {
            "host": "PRO6",
            "device": args.device,
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": "examples.nanogpt.analyze_mlp_cproj_integrated_trajectory_capacity",
            "parameter_updates": 0,
            "direct_foreground_polling": True,
            "watchdog": False,
            "callback": False,
            "elapsed_seconds": time.time() - started,
        },
        "identity": {
            "plan_path": str(args.plan),
            "plan_sha256": file_sha256(args.plan),
            "acquisition_result_sha256": file_sha256(args.acquisition_result),
            "run_identity_sha256": identity,
        },
        "metrics": metrics,
        "evaluations": evaluation_rows,
        "decision": decision,
    }
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
