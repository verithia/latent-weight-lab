#!/usr/bin/env python3
"""Functionally fit a left-canonical global TT for every sparse-MoE c_fc."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from examples.nanogpt.analyze_cproj_manifold import load_model
from examples.nanogpt.analyze_mlp_activation_update_alignment import git_commit
from examples.nanogpt.analyze_sparse_moe_cfc_global_tt_oracle import (
    capped_bond_ranks,
    collect_heldout_inputs,
    coordinate_count,
    dense_to_physical,
    materialize_expert_matrix,
    materialize_tt,
    randomized_tt_svd,
    routed_outputs,
)
from examples.nanogpt.analyze_sparse_moe_cfc_spectral_feature_oracle import (
    action_cosine,
    collect_protocol_inputs,
    dense_targets,
    normalized_fit_loss,
    route_and_sample,
)
from examples.nanogpt.analyze_sparse_moe_paired_alignment import file_sha256
from examples.nanogpt.analyze_sparse_moe_rolling_tangent_oracle import (
    recovery_fraction,
)
from examples.nanogpt.analyze_sparse_moe_stepzero_task_gradient_oracle import (
    layer_state_from_mapping,
    load_terminal_snapshot,
)


PLAN_SCHEMA = "nanogpt_sparse_moe_cfc_functional_tt_oracle_plan_v1"


def left_canonicalize(cores: list[torch.Tensor]) -> list[torch.Tensor]:
    """QR-sweep TT cores left-to-right while preserving their dense tensor."""
    if not cores:
        raise ValueError("TT core list is empty")
    values = [core.detach().clone() for core in cores]
    for index in range(len(values) - 1):
        left_rank, mode, right_rank = values[index].shape
        matrix = values[index].reshape(left_rank * mode, right_rank)
        q, r = torch.linalg.qr(matrix, mode="reduced")
        values[index] = q.reshape(left_rank, mode, right_rank)
        values[index + 1] = torch.tensordot(
            r, values[index + 1], dims=([1], [0])
        )
    return values


class CanonicalTT(torch.nn.Module):
    """A TT chart with differentiable left-QR gauge fixing."""

    def __init__(self, cores: list[torch.Tensor]) -> None:
        super().__init__()
        canonical = left_canonicalize(cores)
        self.raw_cores = torch.nn.ParameterList(
            [torch.nn.Parameter(core) for core in canonical]
        )

    def canonical_cores(self) -> list[torch.Tensor]:
        values: list[torch.Tensor] = []
        for raw in self.raw_cores[:-1]:
            left_rank, mode, right_rank = raw.shape
            q, _r = torch.linalg.qr(
                raw.reshape(left_rank * mode, right_rank), mode="reduced"
            )
            values.append(q.reshape(left_rank, mode, right_rank))
        values.append(self.raw_cores[-1])
        return values

    def detached_cores(self) -> list[torch.Tensor]:
        return [core.detach().cpu() for core in self.canonical_cores()]


def materialize_layer(
    cores: list[torch.Tensor],
    layer: int,
    experts: int,
    hidden_modes: list[int],
    input_modes: list[int],
) -> torch.Tensor:
    return torch.stack(
        [
            materialize_expert_matrix(
                cores,
                layer,
                expert,
                hidden_modes,
                input_modes,
                interleaved=True,
            )
            for expert in range(int(experts))
        ]
    )


def functional_loss(
    candidate_c_fc: torch.Tensor,
    inputs: torch.Tensor,
    dense_c_fc: torch.Tensor,
    dense_c_proj: torch.Tensor,
    device: str,
) -> torch.Tensor:
    target_pre, target_output = dense_targets(
        inputs, dense_c_fc, dense_c_proj, device
    )
    x = inputs.to(device=device, dtype=torch.float32)
    candidate_pre = torch.bmm(
        x, candidate_c_fc.to(device=device, dtype=torch.float32).transpose(1, 2)
    )
    candidate_output = torch.bmm(
        F.gelu(candidate_pre),
        dense_c_proj.to(device=device, dtype=torch.float32).transpose(1, 2),
    )
    return normalized_fit_loss(
        candidate_pre, candidate_output, target_pre, target_output
    )


def aggregate_fit_objective(
    module: CanonicalTT,
    tasks: list[tuple[str, int, torch.Tensor]],
    states: dict[int, Any],
    *,
    experts: int,
    hidden_modes: list[int],
    input_modes: list[int],
    device: str,
) -> float:
    values: list[float] = []
    with torch.no_grad():
        for _bank, layer, inputs in tasks:
            candidate = materialize_layer(
                module.canonical_cores(),
                layer,
                experts,
                hidden_modes,
                input_modes,
            )
            values.append(
                float(
                    functional_loss(
                        candidate,
                        inputs,
                        states[layer].c_fc,
                        states[layer].c_proj,
                        device,
                    )
                )
            )
    return sum(values) / len(values)


def fit_module(
    initial_cores: list[torch.Tensor],
    tasks: list[tuple[str, int, torch.Tensor]],
    states: dict[int, Any],
    plan: dict[str, Any],
    *,
    steps: int,
    device: str,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    mechanism = plan["mechanism"]
    protocol = plan["fit_protocol"]
    hidden_modes = [int(value) for value in mechanism["hidden_modes"]]
    input_modes = [int(value) for value in mechanism["input_modes"]]
    experts = int(plan["source"]["num_experts"])
    module = CanonicalTT([core.to(device) for core in initial_cores]).to(device)
    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=float(protocol["learning_rate"]),
        weight_decay=float(protocol["weight_decay"]),
    )
    initial_objective = aggregate_fit_objective(
        module,
        tasks,
        states,
        experts=experts,
        hidden_modes=hidden_modes,
        input_modes=input_modes,
        device=device,
    )
    losses: list[float] = []
    maximum_gradient = 0.0
    for step in range(int(steps)):
        _bank, layer, inputs = tasks[step % len(tasks)]
        optimizer.zero_grad(set_to_none=True)
        candidate = materialize_layer(
            module.canonical_cores(),
            layer,
            experts,
            hidden_modes,
            input_modes,
        )
        loss = functional_loss(
            candidate,
            inputs,
            states[layer].c_fc,
            states[layer].c_proj,
            device,
        )
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite functional TT objective")
        loss.backward()
        parameters = list(module.parameters())
        if any(
            parameter.grad is None or not torch.isfinite(parameter.grad).all()
            for parameter in parameters
        ):
            raise RuntimeError("non-finite or missing functional TT gradient")
        gradient = float(
            torch.nn.utils.clip_grad_norm_(
                parameters, float(protocol["gradient_clip"])
            )
        )
        maximum_gradient = max(maximum_gradient, gradient)
        optimizer.step()
        losses.append(float(loss.detach()))
    final_objective = aggregate_fit_objective(
        module,
        tasks,
        states,
        experts=experts,
        hidden_modes=hidden_modes,
        input_modes=input_modes,
        device=device,
    )
    return module.detached_cores(), {
        "steps": int(steps),
        "initial_aggregate_objective": initial_objective,
        "final_aggregate_objective": final_objective,
        "relative_objective_decrease": (
            (initial_objective - final_objective) / max(initial_objective, 1e-30)
        ),
        "scheduled_step_loss_first": losses[0],
        "scheduled_step_loss_last": losses[-1],
        "scheduled_step_loss_minimum": min(losses),
        "maximum_preclip_gradient_norm": maximum_gradient,
    }


def initialization_relative_error(
    original: list[torch.Tensor], canonical: list[torch.Tensor]
) -> float:
    target = materialize_tt(original)
    candidate = materialize_tt(canonical)
    return float(
        (candidate - target).square().sum()
        / target.square().sum().clamp_min(1e-30)
    )


def score_cores(
    cores: list[torch.Tensor],
    heldout_inputs: dict[str, dict[int, torch.Tensor]],
    states: dict[int, Any],
    plan: dict[str, Any],
    *,
    device: str,
) -> tuple[dict[str, Any], dict[tuple[str, int], torch.Tensor]]:
    source = plan["source"]
    mechanism = plan["mechanism"]
    hidden_modes = [int(value) for value in mechanism["hidden_modes"]]
    input_modes = [int(value) for value in mechanism["input_modes"]]
    layers = [int(value) for value in source["fit_and_probe_layers"]]
    experts = int(source["num_experts"])
    actions: dict[tuple[str, int], torch.Tensor] = {}
    summaries: dict[str, Any] = {}
    device_cores = [core.to(device) for core in cores]
    for bank in heldout_inputs:
        summaries[bank] = {}
        for layer in layers:
            candidate = materialize_layer(
                device_cores,
                layer,
                experts,
                hidden_modes,
                input_modes,
            )
            predicted, target, expert_recovery, pre_recovery = routed_outputs(
                states[layer],
                heldout_inputs[bank][layer],
                candidate,
                top_k=int(plan["fit_protocol"]["top_k"]),
                chunk_size=int(plan["evaluation"]["heldout_chunk_size"]),
                device=device,
            )
            summaries[bank][str(layer)] = {
                "mixture_recovery": recovery_fraction(predicted, target),
                "expert_recovery": expert_recovery,
                "minimum_expert_recovery": min(expert_recovery),
                "pregelu_recovery": pre_recovery,
                "minimum_pregelu_recovery": min(pre_recovery),
            }
            actions[(bank, layer)] = predicted
    return summaries, actions


def validate_plan(plan: dict[str, Any], plan_path: Path) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("functional TT plan schema mismatch")
    identity = plan["identity"]
    if identity.get("entrypoint_sha256") != file_sha256(Path(__file__)):
        raise ValueError("entrypoint hash is not sealed in the plan")
    root = Path(__file__).resolve().parents[2]
    for relative, expected in identity["helper_sha256"].items():
        if file_sha256(root / relative) != expected:
            raise ValueError(f"helper hash drift: {relative}")
    mechanism = plan["mechanism"]
    modes = [int(value) for value in mechanism["interleaved_modes"]]
    ranks = capped_bond_ranks(modes, int(mechanism["bond_rank_cap"]))
    if ranks != [int(value) for value in mechanism["actual_bond_ranks"]]:
        raise ValueError("TT bond ranks drift")
    if coordinate_count(modes, ranks) != int(mechanism["coordinates"]):
        raise ValueError("TT coordinate count drift")
    if float(mechanism["cfc_compression_ratio"]) < 200.0:
        raise ValueError("functional TT violates compression floor")
    if not file_sha256(plan_path):
        raise AssertionError("unreachable empty plan hash")


def prepare(
    plan: dict[str, Any],
    fit_input_plan: dict[str, Any],
    terminal_snapshot: Path,
    data_dir: Path,
    device: str,
) -> tuple[
    torch.Tensor,
    dict[int, Any],
    list[tuple[str, int, torch.Tensor]],
    dict[str, dict[int, torch.Tensor]],
]:
    source = plan["source"]
    payload = load_terminal_snapshot(terminal_snapshot)
    if int(payload["next_iter"]) != int(source["next_iter"]):
        raise ValueError("snapshot step disagrees with frozen plan")
    states = {
        layer: layer_state_from_mapping(payload["model"], layer)
        for layer in range(int(source["tensor_layers"]))
    }
    dense = torch.stack(
        [states[layer].c_fc for layer in range(int(source["tensor_layers"]))]
    ).to(device=device, dtype=torch.float32)
    model = load_model(terminal_snapshot, device)
    model.eval()
    fit_inputs = collect_protocol_inputs(model, fit_input_plan, data_dir, device)
    heldout_plan = {
        "source": {"functional_probe_layers": source["fit_and_probe_layers"]},
        "functional_protocol": {"heldout_banks": source["heldout_banks"]},
    }
    heldout_inputs = collect_heldout_inputs(model, heldout_plan, data_dir, device)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    protocol = plan["fit_protocol"]
    layers = [int(value) for value in source["fit_and_probe_layers"]]
    tasks: list[tuple[str, int, torch.Tensor]] = []
    for bank_index, bank in enumerate(source["fit_banks"]):
        for layer in layers:
            sampled, _counts = route_and_sample(
                states[layer],
                fit_inputs[bank][layer],
                top_k=int(protocol["top_k"]),
                samples_per_expert=int(
                    protocol["fit_samples_per_expert_per_bank_layer"]
                ),
                seed=(
                    int(protocol["sampling_seed_base"])
                    + int(protocol["sampling_seed_bank_stride"]) * bank_index
                    + int(protocol["sampling_seed_layer_stride"]) * layer
                ),
            )
            tasks.append((bank, layer, sampled))
    return dense, states, tasks, heldout_inputs


def initialize_cores(
    dense: torch.Tensor,
    plan: dict[str, Any],
    seed: int,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    mechanism = plan["mechanism"]
    initialization = plan["initialization"]
    hidden_modes = [int(value) for value in mechanism["hidden_modes"]]
    input_modes = [int(value) for value in mechanism["input_modes"]]
    modes = [int(value) for value in mechanism["interleaved_modes"]]
    ranks = [int(value) for value in mechanism["actual_bond_ranks"]]
    physical = dense_to_physical(
        dense, hidden_modes, input_modes, interleaved=True
    )
    cores, split_diagnostics = randomized_tt_svd(
        physical,
        modes,
        ranks,
        seed=int(seed),
        oversample=int(initialization["tt_svd_oversample"]),
        power_iterations=int(initialization["tt_svd_power_iterations"]),
    )
    canonical = left_canonicalize(cores)
    relative_error = initialization_relative_error(cores, canonical)
    if relative_error > float(
        initialization[
            "initial_materialization_relative_error_after_canonicalization_max"
        ]
    ):
        raise RuntimeError("left canonicalization changed the initial TT")
    return canonical, {
        "split_diagnostics": split_diagnostics,
        "canonicalization_relative_error_squared": relative_error,
    }


def run_preflight(
    dense: torch.Tensor,
    states: dict[int, Any],
    tasks: list[tuple[str, int, torch.Tensor]],
    plan: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    started = time.time()
    cores, initialization = initialize_cores(
        dense, plan, int(plan["initialization"]["independent_seeds"][0])
    )
    setup_seconds = time.time() - started
    fit_started = time.time()
    _fitted, diagnostics = fit_module(
        cores, tasks, states, plan, steps=5, device=device
    )
    fit_seconds = time.time() - fit_started
    projected = setup_seconds * 2 + (fit_seconds / 5.0) * (
        int(plan["fit_protocol"]["steps_per_seed"]) * 2
    )
    return {
        "schema_version": "nanogpt_sparse_moe_cfc_functional_tt_preflight_v1",
        "device": device,
        "setup_seconds_one_seed": setup_seconds,
        "five_step_fit_seconds": fit_seconds,
        "projected_two_seed_fit_seconds_before_evaluation": projected,
        "maximum_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated())
            if device.startswith("cuda")
            else 0
        ),
        "initialization": initialization,
        "fit_diagnostics": diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--fit-input-plan", required=True, type=Path)
    parser.add_argument("--terminal-snapshot", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    started = time.time()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    validate_plan(plan, args.plan)
    source = plan["source"]
    if file_sha256(args.fit_input_plan) != source["fit_input_plan_sha256"]:
        raise ValueError("fit input plan hash disagrees with frozen plan")
    if file_sha256(args.terminal_snapshot) != source[
        "terminal_manifold_snapshot_sha256"
    ]:
        raise ValueError("terminal snapshot hash disagrees with frozen plan")
    if file_sha256(args.data_dir / "manifest.json") != source[
        "dataset_manifest_sha256"
    ]:
        raise ValueError("dataset manifest hash disagrees with frozen plan")
    fit_input_plan = json.loads(args.fit_input_plan.read_text(encoding="utf-8"))
    dense, states, tasks, heldout_inputs = prepare(
        plan, fit_input_plan, args.terminal_snapshot, args.data_dir, args.device
    )
    if args.preflight_only:
        print(
            json.dumps(
                run_preflight(dense, states, tasks, plan, args.device),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.output is None:
        parser.error("scientific oracle requires --output")

    seeds = [int(value) for value in plan["initialization"]["independent_seeds"]]
    fitted: dict[str, list[torch.Tensor]] = {}
    initial: dict[str, list[torch.Tensor]] = {}
    diagnostics: dict[str, Any] = {}
    evaluations: dict[str, Any] = {}
    final_actions: dict[tuple[str, str, int], torch.Tensor] = {}
    for seed in seeds:
        seed_key = str(seed)
        initial_cores, initialization = initialize_cores(dense, plan, seed)
        initial[seed_key] = [core.detach().cpu() for core in initial_cores]
        fitted_cores, fit_diagnostics = fit_module(
            initial_cores,
            tasks,
            states,
            plan,
            steps=int(plan["fit_protocol"]["steps_per_seed"]),
            device=args.device,
        )
        fitted[seed_key] = fitted_cores
        diagnostics[seed_key] = {
            "initialization": initialization,
            "fit": fit_diagnostics,
        }
        initial_scores, _initial_actions = score_cores(
            initial[seed_key], heldout_inputs, states, plan, device=args.device
        )
        final_scores, actions = score_cores(
            fitted[seed_key], heldout_inputs, states, plan, device=args.device
        )
        evaluations[seed_key] = {
            "initialization": initial_scores,
            "fitted": final_scores,
        }
        for key, value in actions.items():
            final_actions[(seed_key, *key)] = value

    layers = [int(value) for value in source["fit_and_probe_layers"]]
    bank_names = [row["name"] for row in source["heldout_banks"]]
    thresholds = plan["frozen_gates"]
    gates: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        seed_key = str(seed)
        gates[seed_key] = {
            "fit_objective_relative_decrease_pass": float(
                diagnostics[seed_key]["fit"]["relative_objective_decrease"]
            )
            >= float(thresholds["fit_objective_relative_decrease_min_each_seed"])
        }
        for bank in bank_names:
            final_rows = [
                evaluations[seed_key]["fitted"][bank][str(layer)]
                for layer in layers
            ]
            initial_rows = [
                evaluations[seed_key]["initialization"][bank][str(layer)]
                for layer in layers
            ]
            recoveries = [float(row["mixture_recovery"]) for row in final_rows]
            gains = [
                float(final_row["mixture_recovery"])
                - float(initial_row["mixture_recovery"])
                for final_row, initial_row in zip(final_rows, initial_rows)
            ]
            aggregate = {
                "mixture_recovery_mean": sum(recoveries) / len(recoveries),
                "mixture_recovery_minimum_layer": min(recoveries),
                "minimum_expert_recovery": min(
                    float(row["minimum_expert_recovery"]) for row in final_rows
                ),
                "candidate_minus_initialization_recovery_mean": sum(gains)
                / len(gains),
            }
            evaluations[seed_key]["fitted"][bank]["aggregate"] = aggregate
            gates[seed_key][bank] = {
                "mean_recovery_pass": aggregate["mixture_recovery_mean"]
                >= float(
                    thresholds[
                        "heldout_mixture_recovery_mean_min_each_seed_and_bank"
                    ]
                ),
                "every_layer_pass": aggregate["mixture_recovery_minimum_layer"]
                >= float(
                    thresholds[
                        "heldout_mixture_recovery_every_layer_min_each_seed_and_bank"
                    ]
                ),
                "every_expert_pass": aggregate["minimum_expert_recovery"]
                >= float(
                    thresholds["heldout_expert_recovery_min_each_seed_and_bank"]
                ),
                "gain_pass": aggregate[
                    "candidate_minus_initialization_recovery_mean"
                ]
                >= float(
                    thresholds[
                        "candidate_minus_initialization_recovery_mean_min_each_seed_and_bank"
                    ]
                ),
            }

    agreement_by_bank_layer: dict[str, dict[str, float]] = {}
    agreements: list[float] = []
    for bank in bank_names:
        agreement_by_bank_layer[bank] = {}
        for layer in layers:
            value = action_cosine(
                final_actions[(str(seeds[0]), bank, layer)],
                final_actions[(str(seeds[1]), bank, layer)],
            )
            agreement_by_bank_layer[bank][str(layer)] = value
            agreements.append(value)
    agreement_mean = sum(agreements) / len(agreements)
    agreement_pass = agreement_mean >= float(
        thresholds["same_bank_action_cosine_between_fitted_seeds_mean_min"]
    )
    finite = all(
        torch.isfinite(core).all()
        for seed_key in fitted
        for core in fitted[seed_key]
    ) and all(
        torch.isfinite(core).all()
        for seed_key in initial
        for core in initial[seed_key]
    )
    for seed_key in gates:
        gates[seed_key]["action_agreement_pass"] = agreement_pass
        gates[seed_key]["finite_pass"] = bool(finite)
        gates[seed_key]["all_pass"] = all(
            value
            for key, value in gates[seed_key].items()
            if key.endswith("_pass")
        ) and all(all(row.values()) for row in gates[seed_key].values() if isinstance(row, dict))
    fit_conditioned = all(
        gates[str(seed)]["fit_objective_relative_decrease_pass"] for seed in seeds
    )
    passed = fit_conditioned and all(gates[str(seed)]["all_pass"] for seed in seeds)
    classification = (
        "FUNCTIONAL_TT_CFC_REPRESENTABILITY_PASSES"
        if passed
        else (
            "FUNCTIONAL_TT_OPTIMIZATION_CONDITIONING_FAILED"
            if not fit_conditioned
            else "FUNCTIONAL_TT_CFC_REPRESENTABILITY_REJECTED"
        )
    )

    args.output.mkdir(parents=True, exist_ok=False)
    coordinates_path = args.output / "functional_tt_cores.pt"
    torch.save(
        {
            "schema_version": "nanogpt_sparse_moe_cfc_functional_tt_coordinates_v1",
            "initial": initial,
            "fitted": fitted,
        },
        coordinates_path,
    )
    result = {
        "schema_version": "nanogpt_sparse_moe_cfc_functional_tt_oracle_result_v1",
        "classification": classification,
        "passed": passed,
        "identity": {
            "git_commit": git_commit(Path(__file__).resolve().parents[2]),
            "plan_sha256": file_sha256(args.plan),
            "entrypoint_sha256": file_sha256(Path(__file__)),
            "fit_input_plan_sha256": file_sha256(args.fit_input_plan),
            "terminal_snapshot_sha256": file_sha256(args.terminal_snapshot),
            "dataset_manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
        },
        "execution": {
            "device": args.device,
            "wall_seconds": time.time() - started,
            "checkpoint_updates": 0,
            "maximum_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated())
                if args.device.startswith("cuda")
                else 0
            ),
            "coordinates_path": str(coordinates_path),
            "coordinates_sha256": file_sha256(coordinates_path),
        },
        "accounting": {
            "coordinates": int(plan["mechanism"]["coordinates"]),
            "dense_cfc_parameters": int(plan["mechanism"]["dense_cfc_parameters"]),
            "cfc_compression_ratio": float(
                plan["mechanism"]["cfc_compression_ratio"]
            ),
            "materialized_dense_cfc_in_candidate": False,
            "dense_cproj_retained_as_exception": True,
        },
        "diagnostics": diagnostics,
        "evaluations": evaluations,
        "same_bank_action_cosine_between_fitted_seeds": {
            "mean": agreement_mean,
            "by_bank_and_layer": agreement_by_bank_layer,
        },
        "gates": gates,
        "all_values_and_gradients_finite": bool(finite),
        "authorization": {
            "implementation": bool(passed),
            "initialization_mapping_loss_shadow": bool(passed),
            "mfu_preflight": False,
            "language_model_training": False,
            "larger_rung": False,
            "generated_cproj": False,
        },
    }
    result_path = args.output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
