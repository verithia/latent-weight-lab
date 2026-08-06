#!/usr/bin/env python3
"""Test bounded residual carry on mature integrated c_proj chords."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import shlex
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
from examples.nanogpt.analyze_mlp_cproj_integrated_trajectory_capacity import (
    geometric_metrics,
)
from examples.nanogpt.analyze_mlp_cproj_teacher_forced_bilateral_full_carry import (
    fit_output_pass,
)
from examples.nanogpt.analyze_mlp_cproj_teacher_forced_bilateral_replay import (
    fit_right_pass,
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


PLAN_SCHEMA = "mai_124m_mlp_cproj_5tpp_bounded_integrated_trajectory_plan_v1"
RESULT_SCHEMA = "mai_124m_mlp_cproj_5tpp_bounded_integrated_trajectory_result_v1"
PHASES = ((0, 594), (594, 1188), (1188, 1782), (1782, 2373))
LAYERS = tuple(range(8))
SUBSTEPS = 8
PHASE_VARIANTS = {
    "phase_zero_feedback": 0.0,
    "phase_decay0p5": 0.5,
}
SEQUENTIAL_VARIANT = "sequential_decay0p5"


def validate_plan(plan: dict[str, Any]) -> None:
    analysis = plan.get("analysis", {})
    observed = {
        "schema_version": plan.get("schema_version"),
        "parameter_updates": analysis.get("parameter_updates"),
        "layers": analysis.get("layers"),
        "phases": analysis.get("phases"),
        "substeps": analysis.get("straight_chord_substeps"),
        "variants": analysis.get("variants"),
        "chart": analysis.get("chart"),
        "eval": analysis.get("fixed_validation"),
        "thresholds": plan.get("decision_rule", {}).get("thresholds"),
    }
    expected = {
        "schema_version": PLAN_SCHEMA,
        "parameter_updates": 0,
        "layers": list(LAYERS),
        "phases": [list(value) for value in PHASES],
        "substeps": SUBSTEPS,
        "variants": {
            "phase_zero_feedback": {"feedback_decay": 0.0},
            "phase_decay0p5": {"feedback_decay": 0.5},
            "sequential_decay0p5": {"feedback_decay": 0.5},
        },
        "chart": {
            "hidden_parent_stages": 64,
            "hidden_residual_stages": 24,
            "output_stages": 32,
            "neighbors": 64,
            "matching_seed": 20260807,
            "weight_decay": 0.0,
            "learning_rate": 1.0,
        },
        "eval": {
            "split": "validation",
            "eval_iters": 400,
            "eval_batch_size": 16,
            "block_size": 1024,
            "eval_seed": 20260715,
            "fixed_eval_indices_sha256": (
                "5ca31b59768e43de808ad5e206ed152a4a0a3515ad68d29a0b2338c4db140747"
            ),
        },
        "thresholds": {
            "phase_decay0p5_geometric_recovery_minimum": 0.55,
            "phase_decay0p5_minimum_layer_phase_recovery": 0.0,
            "phase_decay0p5_maximum_validation_ce_gap": 0.01,
            "phase_decay0p5_terminal_validation_ce_gap": 0.005,
            "sequential_decay0p5_terminal_validation_ce_gap": 0.02,
            "phase_decay0p5_maximum_feedback_fro": 2.6153,
        },
    }
    if observed != expected:
        raise ValueError("bounded integrated-trajectory plan does not match v1 contract")
    authorization = plan.get("authorization", {})
    if authorization.get("run_zero_update_bounded_feedback_analysis") is not True:
        raise ValueError("bounded-feedback analysis is not authorized")
    for key in (
        "implement_candidate_structure",
        "run_exact_config_mfu",
        "run_language_model_training",
        "larger_rung",
    ):
        if authorization.get(key) is not False:
            raise ValueError(f"plan must keep {key} false")


def bounded_structured_step(
    weight: torch.Tensor,
    requested_update: torch.Tensor,
    feedback: torch.Tensor,
    *,
    feedback_decay: float,
    output_stages: int,
    learning_rate: float,
    weight_decay: float,
    neighbors: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Apply the accepted chart with an explicit residual-memory coefficient."""
    corrected = requested_update.float() + feedback_decay * feedback.float()
    current = weight.float()
    residual = corrected
    for pass_index, stages in enumerate((64, 24)):
        updated = fit_right_pass(
            current,
            residual,
            stages=stages,
            neighbors=neighbors,
            seed=seed + pass_index,
        )
        residual = residual - (updated - current)
        current = updated
    if output_stages:
        updated = fit_output_pass(
            current,
            residual,
            stages=output_stages,
            neighbors=neighbors,
            seed=seed + 2,
        )
        residual = residual - (updated - current)
        current = updated
    if weight_decay:
        current = current * (1.0 - learning_rate * weight_decay)
    actual = current - weight.float()
    new_feedback = corrected - actual
    energy = requested_update.float().square().sum().clamp_min(1e-30)
    recovery = float(
        1.0 - (requested_update.float() - actual).square().sum() / energy
    )
    return current, new_feedback.contiguous(), recovery


def fit_straight_chord(
    start: torch.Tensor,
    chord: torch.Tensor,
    *,
    feedback_decay: float,
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
    requested = chord.float() / float(SUBSTEPS)
    recoveries: list[float] = []
    for substep in range(SUBSTEPS):
        current, feedback, recovery = bounded_structured_step(
            current,
            requested,
            feedback,
            feedback_decay=feedback_decay,
            output_stages=32,
            learning_rate=1.0,
            weight_decay=0.0,
            neighbors=neighbors,
            seed=seed + layer * 100000 + phase_index * 1000 + substep * 10,
        )
        recoveries.append(recovery)
    return current, feedback, recoveries


def classify(metrics: dict[str, float], thresholds: dict[str, float]) -> dict[str, Any]:
    phase_gates = {
        "geometric_recovery": metrics["phase_decay0p5_geometric_recovery"]
        >= thresholds["phase_decay0p5_geometric_recovery_minimum"],
        "minimum_cell_recovery": metrics[
            "phase_decay0p5_minimum_layer_phase_recovery"
        ]
        >= thresholds["phase_decay0p5_minimum_layer_phase_recovery"],
        "maximum_validation_gap": metrics[
            "phase_decay0p5_maximum_validation_ce_gap"
        ]
        <= thresholds["phase_decay0p5_maximum_validation_ce_gap"],
        "terminal_validation_gap": metrics[
            "phase_decay0p5_terminal_validation_ce_gap"
        ]
        <= thresholds["phase_decay0p5_terminal_validation_ce_gap"],
        "bounded_feedback": metrics["phase_decay0p5_maximum_feedback_fro"]
        <= thresholds["phase_decay0p5_maximum_feedback_fro"],
    }
    transport_gate = metrics[
        "sequential_decay0p5_terminal_validation_ce_gap"
    ] <= thresholds["sequential_decay0p5_terminal_validation_ce_gap"]
    phase_passed = all(phase_gates.values())
    if phase_passed and transport_gate:
        classification = "BOUNDED_CARRY_MATURE_CAPACITY_AND_TRANSPORT_PASS"
    elif phase_passed:
        classification = "BOUNDED_CARRY_PHASE_CAPACITY_PASS_TRANSPORT_FAIL"
    else:
        classification = "REJECT_BILATERAL_MATURE_PATH_AFTER_BOUNDED_CARRY"
    return {
        "classification": classification,
        "phase_capacity_passed": phase_passed,
        "transport_passed": transport_gate,
        "phase_gates": phase_gates,
        "transport_gate": transport_gate,
        "zero_feedback_terminal_closes": metrics[
            "phase_zero_feedback_terminal_validation_ce_gap"
        ]
        <= thresholds["phase_decay0p5_terminal_validation_ce_gap"],
        "authorization": {
            "bounded_transport_theory": bool(phase_passed and transport_gate),
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
    states: dict[tuple[int, str, int], torch.Tensor] = {}
    sequential = {
        layer: weights[PHASES[0][0]][layer].to(args.device) for layer in LAYERS
    }
    sequential_feedback = {
        layer: torch.zeros_like(sequential[layer]) for layer in LAYERS
    }
    for phase_index, (start_step, end_step) in enumerate(PHASES):
        for layer in LAYERS:
            dense_start = weights[start_step][layer].to(args.device)
            dense_end = weights[end_step][layer].to(args.device)
            chord = dense_end - dense_start
            for variant, decay in PHASE_VARIANTS.items():
                candidate, feedback, recoveries = fit_straight_chord(
                    dense_start,
                    chord,
                    feedback_decay=decay,
                    layer=layer,
                    phase_index=phase_index,
                    neighbors=neighbors,
                    seed=seed,
                )
                states[(end_step, variant, layer)] = candidate.cpu()
                cells.append(
                    {
                        "phase_start": start_step,
                        "phase_end": end_step,
                        "layer": layer,
                        "variant": variant,
                        "substeps": SUBSTEPS,
                        "feedback_decay": decay,
                        "mean_requested_update_recovery": sum(recoveries)
                        / len(recoveries),
                        "terminal_feedback_fro": float(feedback.norm()),
                        **geometric_metrics(dense_start, dense_end, candidate),
                    }
                )

            candidate, feedback, recoveries = fit_straight_chord(
                sequential[layer],
                chord,
                feedback_decay=0.5,
                layer=layer,
                phase_index=phase_index,
                neighbors=neighbors,
                seed=seed + 50000000,
                initial_feedback=sequential_feedback[layer],
            )
            sequential[layer] = candidate
            sequential_feedback[layer] = feedback
            states[(end_step, SEQUENTIAL_VARIANT, layer)] = candidate.cpu()
            cells.append(
                {
                    "phase_start": start_step,
                    "phase_end": end_step,
                    "layer": layer,
                    "variant": SEQUENTIAL_VARIANT,
                    "substeps": SUBSTEPS,
                    "feedback_decay": 0.5,
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
    terminal = PHASES[-1][1]
    for _start_step, end_step in PHASES:
        variants = ["phase_decay0p5"]
        if end_step == terminal:
            variants.extend(["phase_zero_feedback", SEQUENTIAL_VARIANT])
        model = model_from_snapshot(snapshots[end_step], args.device)
        for variant in variants:
            with torch.no_grad():
                for layer in LAYERS:
                    target = model.transformer.h[layer].mlp.c_proj.weight
                    target.copy_(
                        states[(end_step, variant, layer)].to(
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

    selected = [row for row in cells if row["variant"] == "phase_decay0p5"]
    zero = [row for row in cells if row["variant"] == "phase_zero_feedback"]
    total_energy = sum(float(row["chord_energy"]) for row in selected)
    total_error = sum(float(row["endpoint_error_energy"]) for row in selected)
    zero_energy = sum(float(row["chord_energy"]) for row in zero)
    zero_error = sum(float(row["endpoint_error_energy"]) for row in zero)
    eval_index = {
        (int(row["step"]), str(row["variant"])): row for row in evaluation_rows
    }
    phase_eval = [
        row for row in evaluation_rows if row["variant"] == "phase_decay0p5"
    ]
    metrics = {
        "phase_decay0p5_geometric_recovery": 1.0
        - total_error / max(total_energy, 1e-30),
        "phase_decay0p5_minimum_layer_phase_recovery": min(
            float(row["endpoint_recovery"]) for row in selected
        ),
        "phase_decay0p5_maximum_feedback_fro": max(
            float(row["terminal_feedback_fro"]) for row in selected
        ),
        "phase_decay0p5_maximum_validation_ce_gap": max(
            float(row["validation_ce_gap"]) for row in phase_eval
        ),
        "phase_decay0p5_terminal_validation_ce_gap": float(
            eval_index[(terminal, "phase_decay0p5")]["validation_ce_gap"]
        ),
        "phase_zero_feedback_geometric_recovery": 1.0
        - zero_error / max(zero_energy, 1e-30),
        "phase_zero_feedback_terminal_validation_ce_gap": float(
            eval_index[(terminal, "phase_zero_feedback")]["validation_ce_gap"]
        ),
        "sequential_decay0p5_terminal_validation_ce_gap": float(
            eval_index[(terminal, SEQUENTIAL_VARIANT)]["validation_ce_gap"]
        ),
        "decay0p5_terminal_ce_improvement_over_zero": float(
            eval_index[(terminal, "phase_zero_feedback")]["validation_ce"]
            - eval_index[(terminal, "phase_decay0p5")]["validation_ce"]
        ),
    }
    decision = classify(metrics, plan["decision_rule"]["thresholds"])
    args.output.mkdir(parents=True)
    cells_path = args.output / "bounded_integrated_trajectory_cells.csv"
    eval_path = args.output / "bounded_integrated_trajectory_fixed_ce.csv"
    result_path = args.output / "bounded_integrated_trajectory_result.json"
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
            "entrypoint": (
                "examples.nanogpt.analyze_mlp_cproj_bounded_integrated_trajectory"
            ),
            "command": " ".join(shlex.quote(value) for value in sys.argv),
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
            "dataset_manifest_sha256": plan["identity"][
                "dataset_manifest_sha256"
            ],
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
