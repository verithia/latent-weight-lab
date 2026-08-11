#!/usr/bin/env python3
"""Test a model-wide orthogonal residual gauge on rank-nine sparse-MoE MPOs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_cproj_manifold import load_model
from examples.nanogpt.analyze_mlp_activation_update_alignment import git_commit
from examples.nanogpt.analyze_sparse_moe_cproj_context_modulated_fht_oracle import (
    action_cosine,
    cproj_target_action,
    routed_hidden_frames,
)
from examples.nanogpt.analyze_sparse_moe_cproj_kronecker_oracle import (
    apply_delta,
    fixed_inputs,
    parameter_recovery,
)
from examples.nanogpt.analyze_sparse_moe_cproj_mpo_oracle import (
    coordinate_count,
    fit_functional_cores,
    materialize_mpo,
    truncated_mpo_svd,
)
from examples.nanogpt.analyze_sparse_moe_paired_alignment import file_sha256
from examples.nanogpt.analyze_sparse_moe_paired_atom_oracle import union_fieldnames
from examples.nanogpt.analyze_sparse_moe_rolling_tangent_oracle import recovery_fraction
from examples.nanogpt.analyze_sparse_moe_stepzero_task_gradient_oracle import (
    layer_state_from_mapping,
    load_terminal_snapshot,
    model_from_exact_stepzero,
    selected_stepzero_hashes,
)


PLAN_SCHEMA = "nanogpt_sparse_moe_shared_orthogonal_gauge_mpo_plan_v1"


def apply_output_gauge(action: torch.Tensor, gauge: torch.Tensor) -> torch.Tensor:
    """Apply Delta W = R M to a row-major routed action x M^T."""
    return action.float() @ gauge.float().T


def orthogonal_procrustes_right(
    predictions: list[torch.Tensor],
    targets: list[torch.Tensor],
) -> torch.Tensor:
    """Return Q minimizing sum ||P_l Q - T_l||^2 / ||T_l||^2."""
    if len(predictions) != len(targets) or not predictions:
        raise ValueError("predictions and targets must be nonempty and aligned")
    normalized_predictions: list[torch.Tensor] = []
    normalized_targets: list[torch.Tensor] = []
    width = int(targets[0].shape[1])
    for prediction, target in zip(predictions, targets):
        if prediction.shape != target.shape or prediction.ndim != 2:
            raise ValueError("every Procrustes action pair must have the same 2D shape")
        if int(target.shape[1]) != width:
            raise ValueError("all Procrustes actions must share output width")
        scale = target.float().square().sum().sqrt().clamp_min(1e-30)
        normalized_predictions.append(prediction.float() / scale)
        normalized_targets.append(target.float() / scale)
    prediction_matrix = torch.cat(normalized_predictions, dim=0)
    target_matrix = torch.cat(normalized_targets, dim=0)
    u, _, vh = torch.linalg.svd(
        prediction_matrix.T @ target_matrix,
        full_matrices=False,
    )
    return u @ vh


def relative_error_squared(prediction: torch.Tensor, target: torch.Tensor) -> float:
    denominator = target.float().square().sum().clamp_min(1e-30)
    return float((prediction.float() - target.float()).square().sum() / denominator)


def clone_core_sets(core_sets: dict[int, list[torch.Tensor]]) -> dict[int, list[torch.Tensor]]:
    return {
        int(layer): [core.detach().clone() for core in cores]
        for layer, cores in core_sets.items()
    }


def treatment_actions(
    problems: dict[int, dict[str, Any]],
    core_sets: dict[int, list[torch.Tensor]],
    gauge: torch.Tensor,
    output_modes: list[int],
    input_modes: list[int],
) -> dict[int, torch.Tensor]:
    result: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for layer, problem in problems.items():
            delta = materialize_mpo(core_sets[layer], output_modes, input_modes)
            ungauged = apply_delta(problem["frames"], delta, problem["target"].shape[0])
            result[layer] = apply_output_gauge(ungauged, gauge)
    return result


def combined_relative_error(
    actions: dict[int, torch.Tensor],
    problems: dict[int, dict[str, Any]],
) -> float:
    return sum(
        relative_error_squared(actions[layer], problems[layer]["target"])
        for layer in problems
    ) / len(problems)


def fit_shared_gauge_mpo(
    problems: dict[int, dict[str, Any]],
    initial_core_sets: dict[int, list[torch.Tensor]],
    output_modes: list[int],
    input_modes: list[int],
    *,
    outer_rounds: int,
    steps_per_round: int,
    learning_rate: float,
    gradient_clip: float,
) -> tuple[dict[int, list[torch.Tensor]], torch.Tensor, dict[str, Any]]:
    """Alternate MPO core Adam steps and one exact shared Procrustes update."""
    parameters = {
        layer: [torch.nn.Parameter(core.detach().float().clone()) for core in cores]
        for layer, cores in initial_core_sets.items()
    }
    optimizers = {
        layer: torch.optim.Adam(layer_parameters, lr=float(learning_rate))
        for layer, layer_parameters in parameters.items()
    }
    first_target = next(iter(problems.values()))["target"]
    gauge = torch.eye(
        int(first_target.shape[1]),
        device=first_target.device,
        dtype=torch.float32,
    )
    current = clone_core_sets(parameters)
    initial_actions = treatment_actions(
        problems, current, gauge, output_modes, input_modes
    )
    initial_error = combined_relative_error(initial_actions, problems)
    best_error = initial_error
    best_cores = clone_core_sets(parameters)
    best_gauge = gauge.detach().clone()
    maximum_gradient = 0.0
    rounds: list[dict[str, Any]] = []

    for outer in range(int(outer_rounds)):
        for layer, problem in problems.items():
            layer_parameters = parameters[layer]
            optimizer = optimizers[layer]
            denominator = problem["target"].float().square().sum().detach().clamp_min(1e-30)
            for _ in range(int(steps_per_round)):
                optimizer.zero_grad(set_to_none=True)
                delta = materialize_mpo(layer_parameters, output_modes, input_modes)
                ungauged = apply_delta(
                    problem["frames"], delta, problem["target"].shape[0]
                )
                prediction = apply_output_gauge(ungauged, gauge)
                loss = (
                    (prediction - problem["target"].float()).square().sum()
                    / denominator
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite shared-gauge MPO objective")
                loss.backward()
                gradient = float(
                    torch.nn.utils.clip_grad_norm_(
                        layer_parameters, float(gradient_clip)
                    )
                )
                maximum_gradient = max(maximum_gradient, gradient)
                optimizer.step()

        current = clone_core_sets(parameters)
        pregauge_actions: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        with torch.no_grad():
            for layer, problem in problems.items():
                delta = materialize_mpo(current[layer], output_modes, input_modes)
                pregauge_actions.append(
                    apply_delta(
                        problem["frames"], delta, problem["target"].shape[0]
                    )
                )
                targets.append(problem["target"])
            right_transform = orthogonal_procrustes_right(
                pregauge_actions, targets
            )
            gauge = right_transform.T.contiguous()
            actions = {
                layer: apply_output_gauge(action, gauge)
                for layer, action in zip(problems, pregauge_actions)
            }
            error = combined_relative_error(actions, problems)
        rounds.append(
            {
                "outer_round": outer + 1,
                "combined_relative_error_squared": error,
            }
        )
        if error < best_error:
            best_error = error
            best_cores = clone_core_sets(parameters)
            best_gauge = gauge.detach().clone()

    return best_cores, best_gauge, {
        "outer_rounds": int(outer_rounds),
        "steps_per_round": int(steps_per_round),
        "total_core_steps_per_layer": int(outer_rounds) * int(steps_per_round),
        "initial_combined_relative_error_squared": initial_error,
        "best_combined_relative_error_squared": best_error,
        "maximum_preclip_gradient_norm": maximum_gradient,
        "rounds": rounds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--terminal-snapshot", required=True, type=Path)
    parser.add_argument("--mpo-seal", required=True, type=Path)
    parser.add_argument("--depth-group-seal", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    if args.output.exists():
        raise FileExistsError(f"registered output path already exists: {args.output}")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("shared orthogonal gauge MPO plan schema mismatch")
    source = plan["source"]
    for path, expected, label in (
        (args.terminal_snapshot, source["terminal_snapshot_sha256"], "terminal snapshot"),
        (args.mpo_seal, source["mpo_seal_sha256"], "MPO seal"),
        (args.depth_group_seal, source["depth_group_seal_sha256"], "depth-group seal"),
        (args.data_dir / "manifest.json", source["dataset_manifest_sha256"], "dataset manifest"),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"{label} hash disagrees with frozen plan")

    mechanism = plan["mechanism"]
    output_modes = [int(value) for value in mechanism["output_modes"]]
    input_modes = [int(value) for value in mechanism["input_modes"]]
    rank = int(mechanism["bond_rank"])
    coordinates = coordinate_count(output_modes, input_modes, rank)
    if coordinates != int(mechanism["mpo_coordinates_per_expert"]):
        raise ValueError("registered MPO coordinate count is incorrect")
    dense_values = 12 * 8 * math.prod(output_modes) * math.prod(input_modes)
    mpo_total = 12 * 8 * coordinates
    width = math.prod(output_modes)
    orthogonal_dof = width * (width - 1) // 2
    if dense_values != int(mechanism["dense_cproj_values_all_12_layers_8_experts"]):
        raise ValueError("registered dense state count is incorrect")
    if mpo_total != int(mechanism["all_96_mpo_coordinates"]):
        raise ValueError("registered all-MPO state count is incorrect")
    if orthogonal_dof != int(mechanism["orthogonal_intrinsic_degrees_of_freedom"]):
        raise ValueError("registered orthogonal state count is incorrect")

    payload = load_terminal_snapshot(args.terminal_snapshot)
    layers = [int(value) for value in source["layers"]]
    stepzero_model = model_from_exact_stepzero(
        payload, int(source["model_seed"]), args.device
    )
    stepzero_hashes = selected_stepzero_hashes(stepzero_model, layers)
    initial_mapping = dict(stepzero_model.named_parameters())
    initial = {
        layer: layer_state_from_mapping(initial_mapping, layer) for layer in layers
    }
    terminal = {
        layer: layer_state_from_mapping(payload["model"], layer) for layer in layers
    }
    del initial_mapping, stepzero_model
    torch.cuda.empty_cache()
    terminal_model = load_model(args.terminal_snapshot, args.device)
    inputs = fixed_inputs(terminal_model, plan, args.data_dir, layers, args.device)
    del terminal_model
    torch.cuda.empty_cache()

    protocol = plan["functional_protocol"]
    fit = plan["fit"]
    initial_core_sets: dict[int, list[torch.Tensor]] = {}
    heldout: dict[int, dict[str, Any]] = {}
    discovery: dict[str, dict[int, dict[str, Any]]] = {
        spec["name"]: {} for spec in protocol["discovery_banks"]
    }
    deltas: dict[int, torch.Tensor] = {}

    for layer in layers:
        delta = (
            terminal[layer].c_proj.to(args.device).float()
            - initial[layer].c_proj.to(args.device).float()
        )
        deltas[layer] = delta
        initial_core_sets[layer] = truncated_mpo_svd(
            delta, output_modes, input_modes, rank
        )
        heldout_x = inputs["heldout"][layer].to(args.device)
        heldout_frames, heldout_counts = routed_hidden_frames(
            terminal[layer], heldout_x, int(protocol["top_k"]), args.device
        )
        heldout[layer] = {
            "frames": heldout_frames,
            "counts": heldout_counts,
            "target": cproj_target_action(
                heldout_frames,
                initial[layer].c_proj,
                terminal[layer].c_proj,
                heldout_x.shape[0],
                heldout_x.shape[1],
                args.device,
            ),
        }
        for bank_spec in protocol["discovery_banks"]:
            bank = bank_spec["name"]
            x = inputs[bank][layer].to(args.device)
            frames, counts = routed_hidden_frames(
                terminal[layer], x, int(protocol["top_k"]), args.device
            )
            discovery[bank][layer] = {
                "frames": frames,
                "counts": counts,
                "target": cproj_target_action(
                    frames,
                    initial[layer].c_proj,
                    terminal[layer].c_proj,
                    x.shape[0],
                    x.shape[1],
                    args.device,
                ),
            }

    rows: list[dict[str, Any]] = []
    heldout_actions: dict[tuple[str, int], torch.Tensor] = {}
    fitted_states: dict[str, Any] = {}
    bank_diagnostics: dict[str, dict[str, Any]] = {}
    bank_gauges: dict[str, torch.Tensor] = {}

    for bank_spec in protocol["discovery_banks"]:
        bank = bank_spec["name"]
        problems = discovery[bank]
        treatment_cores, gauge, treatment_diagnostics = fit_shared_gauge_mpo(
            problems,
            initial_core_sets,
            output_modes,
            input_modes,
            outer_rounds=int(fit["outer_rounds"]),
            steps_per_round=int(fit["mpo_adam_steps_per_round"]),
            learning_rate=float(fit["learning_rate"]),
            gradient_clip=float(fit["gradient_clip"]),
        )
        bank_gauges[bank] = gauge.detach().cpu()
        fitted_states[f"{bank}:gauge"] = gauge.detach().cpu()
        bank_diagnostics[bank] = treatment_diagnostics
        for layer in layers:
            control_cores, control_diagnostics = fit_functional_cores(
                problems[layer]["frames"],
                problems[layer]["target"],
                initial_core_sets[layer],
                output_modes,
                input_modes,
                steps=int(fit["matched_control_total_adam_steps"]),
                learning_rate=float(fit["learning_rate"]),
                gradient_clip=float(fit["gradient_clip"]),
            )
            control_delta = materialize_mpo(
                control_cores, output_modes, input_modes
            )
            treatment_delta_ungauged = materialize_mpo(
                treatment_cores[layer], output_modes, input_modes
            )
            treatment_delta = torch.einsum(
                "oi,eij->eoj", gauge.float(), treatment_delta_ungauged.float()
            )
            control_discovery = apply_delta(
                problems[layer]["frames"],
                control_delta,
                problems[layer]["target"].shape[0],
            )
            treatment_discovery_ungauged = apply_delta(
                problems[layer]["frames"],
                treatment_delta_ungauged,
                problems[layer]["target"].shape[0],
            )
            treatment_discovery = apply_output_gauge(
                treatment_discovery_ungauged, gauge
            )
            control_heldout = apply_delta(
                heldout[layer]["frames"],
                control_delta,
                heldout[layer]["target"].shape[0],
            )
            treatment_heldout_ungauged = apply_delta(
                heldout[layer]["frames"],
                treatment_delta_ungauged,
                heldout[layer]["target"].shape[0],
            )
            treatment_heldout = apply_output_gauge(
                treatment_heldout_ungauged, gauge
            )
            heldout_actions[(bank, layer)] = treatment_heldout.detach().cpu()
            for index, core in enumerate(treatment_cores[layer]):
                fitted_states[f"{bank}:treatment:layer{layer}:core{index}"] = core.detach().cpu()
            for index, core in enumerate(control_cores):
                fitted_states[f"{bank}:control:layer{layer}:core{index}"] = core.detach().cpu()
            rows.append(
                {
                    "bank": bank,
                    "layer": layer,
                    "minimum_discovery_assignments": min(problems[layer]["counts"]),
                    "minimum_heldout_assignments": min(heldout[layer]["counts"]),
                    "control_parameter_recovery": parameter_recovery(
                        control_delta, deltas[layer]
                    ),
                    "treatment_parameter_recovery": parameter_recovery(
                        treatment_delta, deltas[layer]
                    ),
                    "control_discovery_recovery": recovery_fraction(
                        control_discovery, problems[layer]["target"]
                    ),
                    "treatment_discovery_recovery": recovery_fraction(
                        treatment_discovery, problems[layer]["target"]
                    ),
                    "control_heldout_recovery": recovery_fraction(
                        control_heldout, heldout[layer]["target"]
                    ),
                    "treatment_heldout_recovery": recovery_fraction(
                        treatment_heldout, heldout[layer]["target"]
                    ),
                    "treatment_minus_control_heldout": recovery_fraction(
                        treatment_heldout, heldout[layer]["target"]
                    )
                    - recovery_fraction(control_heldout, heldout[layer]["target"]),
                    "control_initial_relative_error_squared": control_diagnostics[
                        "initial_relative_error_squared"
                    ],
                    "control_best_relative_error_squared": control_diagnostics[
                        "best_relative_error_squared"
                    ],
                }
            )

    banks = [spec["name"] for spec in protocol["discovery_banks"]]
    for row in rows:
        other = banks[1] if row["bank"] == banks[0] else banks[0]
        row["heldout_bank_action_cosine"] = action_cosine(
            heldout_actions[(row["bank"], int(row["layer"]))],
            heldout_actions[(other, int(row["layer"]))],
        )

    summaries: dict[str, Any] = {}
    gate_results: dict[str, Any] = {}
    gates = plan["frozen_gates"]
    for bank in banks:
        selected = [row for row in rows if row["bank"] == bank]
        treatment_mean = sum(
            float(row["treatment_heldout_recovery"]) for row in selected
        ) / len(selected)
        control_mean = sum(
            float(row["control_heldout_recovery"]) for row in selected
        ) / len(selected)
        summary = {
            "treatment_heldout_recovery_mean": treatment_mean,
            "treatment_heldout_recovery_minimum": min(
                float(row["treatment_heldout_recovery"]) for row in selected
            ),
            "treatment_heldout_recovery_by_layer": [
                float(row["treatment_heldout_recovery"]) for row in selected
            ],
            "control_heldout_recovery_mean": control_mean,
            "control_heldout_recovery_by_layer": [
                float(row["control_heldout_recovery"]) for row in selected
            ],
            "treatment_minus_control_mean": treatment_mean - control_mean,
            "heldout_bank_action_cosine_mean": sum(
                float(row["heldout_bank_action_cosine"]) for row in selected
            )
            / len(selected),
            "minimum_discovery_assignments": min(
                int(row["minimum_discovery_assignments"]) for row in selected
            ),
            "treatment_fit_non_degrading": bank_diagnostics[bank][
                "best_combined_relative_error_squared"
            ]
            <= bank_diagnostics[bank]["initial_combined_relative_error_squared"]
            + 1e-8,
            "fit_diagnostics": bank_diagnostics[bank],
        }
        summaries[bank] = summary
        gate_results[bank] = {
            "mean_recovery": treatment_mean
            >= float(gates["heldout_recovery_mean_min"]),
            "minimum_layer_recovery": summary[
                "treatment_heldout_recovery_minimum"
            ]
            >= float(gates["heldout_recovery_every_layer_min"]),
            "improvement_over_matched_control": summary[
                "treatment_minus_control_mean"
            ]
            >= float(gates["improvement_over_matched_ungauged_mpo_min"]),
            "bank_action_cosine": summary[
                "heldout_bank_action_cosine_mean"
            ]
            >= float(gates["heldout_bank_action_cosine_mean_min"]),
            "occupancy": summary["minimum_discovery_assignments"]
            >= int(gates["minimum_expert_assignments"]),
            "fit_non_degrading": bool(summary["treatment_fit_non_degrading"]),
        }
        gate_results[bank]["all_pass"] = all(gate_results[bank].values())

    gauge_agreement = float(
        torch.trace(bank_gauges[banks[0]].T @ bank_gauges[banks[1]]) / width
    )
    finite_keys = (
        "control_parameter_recovery",
        "treatment_parameter_recovery",
        "control_discovery_recovery",
        "treatment_discovery_recovery",
        "control_heldout_recovery",
        "treatment_heldout_recovery",
        "treatment_minus_control_heldout",
        "heldout_bank_action_cosine",
    )
    finite = all(
        math.isfinite(float(row[key])) for row in rows for key in finite_keys
    ) and math.isfinite(gauge_agreement)
    passed = finite and all(result["all_pass"] for result in gate_results.values())
    decision = (
        "PASS_SHARED_ORTHOGONAL_GAUGE_MPO_UPPER_BOUND"
        if passed
        else "REJECT_SHARED_ORTHOGONAL_GAUGE_AS_SUFFICIENT_MPO_REPAIR"
    )

    args.output.mkdir(parents=True, exist_ok=False)
    rows_path = args.output / "shared_gauge_mpo_rows.csv"
    with rows_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=union_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)
    state_path = args.output / "shared_gauge_mpo_state.pt"
    torch.save(fitted_states, state_path)
    result = {
        "schema_version": "nanogpt_sparse_moe_shared_orthogonal_gauge_mpo_result_v1",
        "decision": decision,
        "all_values_finite": finite,
        "summary": summaries,
        "gates": gate_results,
        "independent_gauge_normalized_trace": gauge_agreement,
        "mechanism": mechanism,
        "stepzero_selected_tensor_sha256": stepzero_hashes,
        "source": {
            "terminal_snapshot_sha256": file_sha256(args.terminal_snapshot),
            "mpo_seal_sha256": file_sha256(args.mpo_seal),
            "depth_group_seal_sha256": file_sha256(args.depth_group_seal),
            "plan_sha256": file_sha256(args.plan),
            "dataset_manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
        },
        "authorization": {
            "compress_shared_gauge": bool(passed),
            "training": False,
            "mfu_preflight": False,
            "larger_rung": False,
        },
        "execution": {
            "git_commit": git_commit(Path(__file__).resolve().parents[2]),
            "entrypoint": str(Path(__file__).resolve()),
            "entrypoint_sha256": file_sha256(Path(__file__).resolve()),
            "command": sys.argv,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "device": args.device,
        },
    }
    result_path = args.output / "shared_gauge_mpo_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    status = {
        "state": "finished",
        "exit_code": 0,
        "decision": decision,
        "result_sha256": file_sha256(result_path),
        "rows_sha256": file_sha256(rows_path),
        "state_sha256": file_sha256(state_path),
        "wall_seconds": time.time() - started,
    }
    status_path = args.output / "shared_gauge_mpo_status.json"
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": status, "summary": summaries}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
