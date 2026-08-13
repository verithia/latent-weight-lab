#!/usr/bin/env python3
"""Decompose sparse-MoE c_fc tangent error into normal and GELU-boundary terms."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from examples.nanogpt.analyze_cproj_manifold import load_model
from examples.nanogpt.analyze_mlp_activation_update_alignment import git_commit
from examples.nanogpt.analyze_sparse_moe_cfc_spectral_feature_oracle import (
    collect_protocol_inputs,
    route_and_sample,
)
from examples.nanogpt.analyze_sparse_moe_input_atlas_ceiling import (
    BANKS,
    CAUSAL_BANKS,
    gelu_derivative,
    rademacher,
    ridge_coefficients,
)
from examples.nanogpt.analyze_sparse_moe_paired_alignment import file_sha256
from examples.nanogpt.analyze_sparse_moe_paired_coordinate_field_oracle import (
    function_and_jvp as dense_function_and_jvp,
)
from examples.nanogpt.analyze_sparse_moe_rolling_tangent_oracle import (
    LayerState,
)
from examples.nanogpt.analyze_sparse_moe_stepzero_task_gradient_oracle import (
    all_finite,
    layer_state_from_mapping,
    load_terminal_snapshot,
)


PLAN_SCHEMA = "nanogpt_sparse_moe_gelu_boundary_tangent_decomposition_plan_v1"
INPUT_BASIS_SCHEMA = "nanogpt_sparse_moe_input_atlas_ceiling_bases_v1"
ARMS = (
    "teacher_gate_projected_normal",
    "projected_gate_dense_normal",
    "self_consistent",
    "first_order_normal_plus_boundary",
)


def gelu_second_derivative(values: torch.Tensor) -> torch.Tensor:
    density = torch.exp(-0.5 * values.square()) / math.sqrt(2.0 * math.pi)
    return (2.0 - values.square()) * density


def load_input_bases(path: Path, plan: dict[str, Any]) -> dict[str, torch.Tensor]:
    if file_sha256(path) != plan["source"]["input_bases_artifact_sha256"]:
        raise ValueError("input-basis artifact hash drift")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != INPUT_BASIS_SCHEMA:
        raise ValueError("input-basis artifact schema mismatch")
    width = int(plan["source"]["input_width"])
    rank = int(plan["source"]["input_atlas_rank"])
    result = {}
    for bank in BANKS:
        basis = payload["bases"][bank]
        if tuple(basis.shape) != (rank, width):
            raise ValueError("input-basis shape drift")
        result[bank] = basis.float().contiguous()
    return result


def tangent_counterfactuals(
    inputs: torch.Tensor,
    directions: torch.Tensor,
    c_fc: torch.Tensor,
    c_proj: torch.Tensor,
    basis: torch.Tensor,
    coefficients: torch.Tensor,
) -> dict[str, torch.Tensor]:
    p = inputs @ c_fc.T
    t = directions @ c_fc.T
    z = inputs @ basis.T
    z_tangent = directions @ basis.T
    p_hat = z @ coefficients
    t_hat = z_tangent @ coefficients
    dense_gate = gelu_derivative(p)
    fitted_gate = gelu_derivative(p_hat)
    target = (dense_gate * t) @ c_proj.T
    teacher_gate = (dense_gate * t_hat) @ c_proj.T
    projected_gate = (fitted_gate * t) @ c_proj.T
    self_consistent = (fitted_gate * t_hat) @ c_proj.T
    normal_term = (dense_gate * (t_hat - t)) @ c_proj.T
    boundary_term = (
        gelu_second_derivative(p) * (p_hat - p) * t
    ) @ c_proj.T
    first_order = target + normal_term + boundary_term
    return {
        "output": F.gelu(p_hat) @ c_proj.T,
        "target_output": F.gelu(p) @ c_proj.T,
        "dense_target": target,
        "teacher_gate_projected_normal": teacher_gate,
        "projected_gate_dense_normal": projected_gate,
        "self_consistent": self_consistent,
        "first_order_normal_plus_boundary": first_order,
        "normal_term": normal_term,
        "boundary_term": boundary_term,
    }


def _new_arm(experts: int) -> dict[str, Any]:
    return {
        "error": 0.0,
        "energy": 0.0,
        "expert_error": torch.zeros(experts, dtype=torch.float64),
        "expert_energy": torch.zeros(experts, dtype=torch.float64),
    }


def _new_decomposition() -> dict[str, float]:
    return {
        "exact_error_energy": 0.0,
        "normal_energy": 0.0,
        "boundary_energy": 0.0,
        "normal_boundary_dot": 0.0,
        "normal_exact_dot": 0.0,
        "boundary_exact_dot": 0.0,
        "first_order_residual_energy": 0.0,
    }


@torch.no_grad()
def evaluate_decomposition(
    states: dict[int, LayerState],
    heldout_inputs: dict[int, torch.Tensor],
    bases: dict[str, torch.Tensor],
    coefficients: dict[str, dict[int, torch.Tensor]],
    plan: dict[str, Any],
    device: str,
    chunk_size: int = 1024,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, float]]:
    source = plan["source"]
    experts = int(source["num_experts"])
    top_k = int(source["outer_moe_top_k"])
    live_bases = {bank: bases[bank].to(device) for bank in BANKS}
    live_coefficients = {
        bank: {layer: value.to(device) for layer, value in rows.items()}
        for bank, rows in coefficients.items()
    }
    layer_rows = {bank: {arm: {} for arm in ARMS} for bank in BANKS}
    output_rows = {bank: {} for bank in BANKS}
    decomposition = {bank: _new_decomposition() for bank in BANKS}
    agreement = {
        arm: {"dot": 0.0, "left": 0.0, "right": 0.0} for arm in ARMS
    }
    for layer in [int(value) for value in source["layers"]]:
        state = states[layer].to(device)
        activations = heldout_inputs[layer]
        directions = rademacher(
            tuple(activations.shape),
            int(plan["data_protocol"]["analytic_jvp_probe_seed"]) + 17 * layer,
            "cpu",
        )
        accumulators = {
            bank: {arm: _new_arm(experts) for arm in ARMS} for bank in BANKS
        }
        output_acc = {
            bank: {"error": 0.0, "energy": 0.0} for bank in BANKS
        }
        for start in range(0, activations.shape[0], int(chunk_size)):
            stop = min(activations.shape[0], start + int(chunk_size))
            x = activations[start:stop].to(device=device, dtype=torch.float32)
            direction = directions[start:stop].to(device=device)
            logits = x @ state.router.T
            tie = torch.arange(experts, device=device, dtype=x.dtype)
            selected = torch.topk(
                logits - tie * torch.finfo(x.dtype).eps,
                top_k,
                dim=-1,
                largest=True,
                sorted=True,
            ).indices
            probabilities = F.softmax(logits.gather(-1, selected), dim=-1)
            target_output = torch.zeros_like(x)
            target_jvp = torch.zeros_like(x)
            predicted_output = {bank: torch.zeros_like(x) for bank in BANKS}
            predicted = {
                bank: {arm: torch.zeros_like(x) for arm in ARMS}
                for bank in BANKS
            }
            normal = {bank: torch.zeros_like(x) for bank in BANKS}
            boundary = {bank: torch.zeros_like(x) for bank in BANKS}
            for expert in range(experts):
                locations = (selected == expert).nonzero(as_tuple=False)
                if not locations.numel():
                    continue
                token, slot = locations[:, 0], locations[:, 1]
                expert_input = x.index_select(0, token)
                expert_direction = direction.index_select(0, token)
                target, target_tangent = dense_function_and_jvp(
                    expert_input[None],
                    expert_direction[None],
                    state.c_fc[expert : expert + 1],
                    state.c_proj[expert : expert + 1].transpose(1, 2),
                )
                target, target_tangent = target[0], target_tangent[0]
                weight = probabilities[token, slot, None]
                target_output.index_add_(0, token, target * weight)
                target_jvp.index_add_(0, token, target_tangent * weight)
                for bank in BANKS:
                    rows = tangent_counterfactuals(
                        expert_input,
                        expert_direction,
                        state.c_fc[expert],
                        state.c_proj[expert],
                        live_bases[bank],
                        live_coefficients[bank][layer][expert],
                    )
                    predicted_output[bank].index_add_(
                        0, token, rows["output"] * weight
                    )
                    normal[bank].index_add_(
                        0, token, rows["normal_term"] * weight
                    )
                    boundary[bank].index_add_(
                        0, token, rows["boundary_term"] * weight
                    )
                    for arm in ARMS:
                        live = rows[arm]
                        predicted[bank][arm].index_add_(0, token, live * weight)
                        accumulators[bank][arm]["expert_error"][expert] += float(
                            (live - target_tangent).square().sum()
                        )
                        accumulators[bank][arm]["expert_energy"][expert] += float(
                            target_tangent.square().sum()
                        )
            output_energy = float(target_output.square().sum())
            tangent_energy = float(target_jvp.square().sum())
            for bank in BANKS:
                output_acc[bank]["error"] += float(
                    (predicted_output[bank] - target_output).square().sum()
                )
                output_acc[bank]["energy"] += output_energy
                exact_error = predicted[bank]["self_consistent"] - target_jvp
                normal_term = normal[bank]
                boundary_term = boundary[bank]
                residual = exact_error - normal_term - boundary_term
                row = decomposition[bank]
                row["exact_error_energy"] += float(exact_error.square().sum())
                row["normal_energy"] += float(normal_term.square().sum())
                row["boundary_energy"] += float(boundary_term.square().sum())
                row["normal_boundary_dot"] += float((normal_term * boundary_term).sum())
                row["normal_exact_dot"] += float((normal_term * exact_error).sum())
                row["boundary_exact_dot"] += float((boundary_term * exact_error).sum())
                row["first_order_residual_energy"] += float(residual.square().sum())
                for arm in ARMS:
                    live = predicted[bank][arm]
                    arm_row = accumulators[bank][arm]
                    arm_row["error"] += float((live - target_jvp).square().sum())
                    arm_row["energy"] += tangent_energy
            for arm in ARMS:
                left = predicted[CAUSAL_BANKS[0]][arm]
                right = predicted[CAUSAL_BANKS[1]][arm]
                row = agreement[arm]
                row["dot"] += float((left * right).sum())
                row["left"] += float(left.square().sum())
                row["right"] += float(right.square().sum())
        for bank in BANKS:
            output_rows[bank][str(layer)] = 1.0 - output_acc[bank]["error"] / max(
                output_acc[bank]["energy"], 1e-30
            )
            for arm in ARMS:
                row = accumulators[bank][arm]
                expert_recovery = 1.0 - row["expert_error"] / row[
                    "expert_energy"
                ].clamp_min(1e-30)
                layer_rows[bank][arm][str(layer)] = {
                    "jvp_recovery": 1.0 - row["error"] / max(row["energy"], 1e-30),
                    "minimum_expert_recovery": float(expert_recovery.min()),
                    "expert_recovery": expert_recovery.tolist(),
                }
    summaries = {}
    for bank in BANKS:
        summaries[bank] = {
            "output_recovery_mean": sum(output_rows[bank].values()) / len(output_rows[bank]),
            "output_recovery_by_layer": output_rows[bank],
            "arms": {},
        }
        for arm in ARMS:
            values = list(layer_rows[bank][arm].values())
            summaries[bank]["arms"][arm] = {
                "by_layer": layer_rows[bank][arm],
                "jvp_recovery_mean": sum(row["jvp_recovery"] for row in values) / len(values),
                "jvp_recovery_minimum_layer": min(row["jvp_recovery"] for row in values),
                "minimum_expert_recovery": min(row["minimum_expert_recovery"] for row in values),
            }
    decomposition_summaries = {}
    for bank, row in decomposition.items():
        exact = max(row["exact_error_energy"], 1e-30)
        normal = max(row["normal_energy"], 1e-30)
        boundary_energy = max(row["boundary_energy"], 1e-30)
        decomposition_summaries[bank] = {
            **row,
            "normal_fraction_of_exact_error_energy": row["normal_energy"] / exact,
            "boundary_fraction_of_exact_error_energy": row["boundary_energy"] / exact,
            "normal_boundary_cosine": row["normal_boundary_dot"] / math.sqrt(normal * boundary_energy),
            "normal_exact_error_cosine": row["normal_exact_dot"] / math.sqrt(normal * exact),
            "boundary_exact_error_cosine": row["boundary_exact_dot"] / math.sqrt(boundary_energy * exact),
            "first_order_exact_error_recovery": 1.0 - row["first_order_residual_energy"] / exact,
            "higher_order_residual_fraction": row["first_order_residual_energy"] / exact,
        }
    agreement_summaries = {
        arm: row["dot"] / math.sqrt(max(row["left"] * row["right"], 1e-30))
        for arm, row in agreement.items()
    }
    return summaries, decomposition_summaries, agreement_summaries


def replay_checks(summaries: dict[str, Any], plan: dict[str, Any]) -> dict[str, bool]:
    tolerance = plan["coefficient_replay"]["required_replay_checks"]
    keys = {
        "discovery_a_input_only_output_tolerance": summaries["discovery_a"]["output_recovery_mean"],
        "discovery_b_input_only_output_tolerance": summaries["discovery_b"]["output_recovery_mean"],
        "heldout_input_only_output_tolerance": summaries["heldout"]["output_recovery_mean"],
        "discovery_a_self_jvp_tolerance": summaries["discovery_a"]["arms"]["self_consistent"]["jvp_recovery_mean"],
        "discovery_b_self_jvp_tolerance": summaries["discovery_b"]["arms"]["self_consistent"]["jvp_recovery_mean"],
        "heldout_self_jvp_tolerance": summaries["heldout"]["arms"]["self_consistent"]["jvp_recovery_mean"],
    }
    return {
        key: float(tolerance[key][0]) <= value <= float(tolerance[key][1])
        for key, value in keys.items()
    }


def validate_plan(plan: dict[str, Any], path: Path) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("GELU-boundary decomposition plan schema mismatch")
    identity = plan["identity"]
    if identity.get("entrypoint_sha256") != file_sha256(Path(__file__)):
        raise ValueError("entrypoint hash is not sealed")
    helpers = identity.get("helper_sha256")
    if not isinstance(helpers, dict) or not helpers:
        raise ValueError("helper hashes are not sealed")
    root = Path(__file__).resolve().parents[2]
    for relative, expected in helpers.items():
        if file_sha256(root / relative) != expected:
            raise ValueError(f"helper hash drift: {relative}")
    source = plan["source"]
    if file_sha256(root / source["input_atlas_result"]) != source[
        "input_atlas_result_sha256"
    ]:
        raise ValueError("input-atlas result hash drift")
    if not file_sha256(path):
        raise AssertionError("unreachable empty plan hash")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--input-bases", required=True, type=Path)
    parser.add_argument("--terminal-snapshot", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    validate_plan(plan, args.plan)
    source = plan["source"]
    if file_sha256(args.terminal_snapshot) != source["terminal_manifold_snapshot_sha256"]:
        raise ValueError("terminal snapshot hash drift")
    manifest = args.data_dir / "manifest.json"
    if file_sha256(manifest) != source["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest hash drift")
    payload = load_terminal_snapshot(args.terminal_snapshot)
    if int(payload["next_iter"]) != int(source["next_iter"]):
        raise ValueError("terminal snapshot step drift")
    bases = load_input_bases(args.input_bases, plan)
    model = load_model(args.terminal_snapshot, args.device)
    model.eval()
    inputs = collect_protocol_inputs(model, plan, args.data_dir, args.device)
    mapping = dict(model.named_parameters())
    layers = [int(value) for value in source["layers"]]
    states = {layer: layer_state_from_mapping(mapping, layer) for layer in layers}
    del mapping, model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    protocol = plan["data_protocol"]
    split_index = {bank: index for index, bank in enumerate(BANKS)}
    coefficients: dict[str, dict[int, torch.Tensor]] = {bank: {} for bank in BANKS}
    coefficient_diagnostics: dict[str, dict[str, Any]] = {bank: {} for bank in BANKS}
    occupancy: dict[str, dict[str, list[int]]] = {bank: {} for bank in BANKS}
    for bank in BANKS:
        for layer in layers:
            sampled, counts = route_and_sample(
                states[layer],
                inputs[bank][layer],
                top_k=int(source["outer_moe_top_k"]),
                samples_per_expert=int(protocol["samples_per_expert_each_split"]),
                seed=int(protocol["sample_selection_seed_base"]) + 1009 * split_index[bank] + 17 * layer,
            )
            occupancy[bank][str(layer)] = counts
            fitted, diagnostics = ridge_coefficients(
                sampled,
                states[layer].c_fc,
                bases[bank],
                ridge_scale=1e-4,
                device=args.device,
            )
            coefficients[bank][layer] = fitted
            coefficient_diagnostics[bank][str(layer)] = diagnostics

    summaries, decomposition, agreements = evaluate_decomposition(
        states, inputs["heldout"], bases, coefficients, plan, args.device
    )
    replay = replay_checks(summaries, plan)
    if not all(replay.values()):
        raise RuntimeError(f"sealed predecessor replay drift: {replay}")

    frozen = plan["frozen_gates"]
    gates: dict[str, Any] = {"replay": replay}
    for bank in BANKS:
        gates[bank] = {}
        teacher = summaries[bank]["arms"]["teacher_gate_projected_normal"]
        moved = summaries[bank]["arms"]["projected_gate_dense_normal"]
        gates[bank]["teacher_gate_projected_normal_pass"] = (
            teacher["jvp_recovery_mean"] >= float(frozen["teacher_gate_projected_normal_jvp_recovery_mean_min_each_bank"])
            and teacher["jvp_recovery_minimum_layer"] >= float(frozen["teacher_gate_projected_normal_every_layer_min_each_bank"])
            and teacher["minimum_expert_recovery"] >= float(frozen["teacher_gate_projected_normal_minimum_expert_min_each_bank"])
        )
        gates[bank]["projected_gate_dense_normal_pass"] = (
            moved["jvp_recovery_mean"] >= float(frozen["projected_gate_dense_normal_jvp_recovery_mean_min_each_bank"])
            and moved["jvp_recovery_minimum_layer"] >= float(frozen["projected_gate_dense_normal_every_layer_min_each_bank"])
            and moved["minimum_expert_recovery"] >= float(frozen["projected_gate_dense_normal_minimum_expert_min_each_bank"])
        )
        gates[bank]["first_order_pass"] = decomposition[bank][
            "first_order_exact_error_recovery"
        ] >= float(frozen["first_order_exact_error_recovery_min_each_bank"])
        gates[bank]["action_agreement_pass"] = all(
            agreements[arm] >= float(frozen["cross_bank_counterfactual_action_cosine_min"])
            for arm in ARMS
        )
    self_pass = all(
        summaries[bank]["arms"]["self_consistent"]["jvp_recovery_mean"] >= 0.6
        for bank in CAUSAL_BANKS
    )
    teacher_pass = all(
        gates[bank]["teacher_gate_projected_normal_pass"] for bank in CAUSAL_BANKS
    )
    moved_pass = all(
        gates[bank]["projected_gate_dense_normal_pass"] for bank in CAUSAL_BANKS
    )
    first_order_pass = all(gates[bank]["first_order_pass"] for bank in CAUSAL_BANKS)
    if self_pass:
        classification = "PREDECESSOR_REPLAY_CONTRADICTION"
    elif not teacher_pass:
        classification = "RANK480_NORMAL_FIELD_BINDING_FULL_INPUT_RANK_REQUIRED"
    elif not moved_pass:
        classification = "GELU_BOUNDARY_PHASE_BINDING"
    elif not first_order_pass:
        classification = "HIGHER_ORDER_GATE_INTERACTION_BINDING"
    else:
        classification = "JOINT_NORMAL_GATE_INTERACTION_BINDING"

    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    result = {
        "schema_version": "nanogpt_sparse_moe_gelu_boundary_tangent_decomposition_result_v1",
        "classification": classification,
        "identity": {
            "git_commit": git_commit(root),
            "plan_sha256": file_sha256(args.plan),
            "entrypoint_sha256": file_sha256(Path(__file__)),
            "input_bases_sha256": file_sha256(args.input_bases),
            "terminal_snapshot_sha256": file_sha256(args.terminal_snapshot),
            "dataset_manifest_sha256": file_sha256(manifest),
        },
        "execution": {
            "device": args.device,
            "wall_seconds": time.time() - started,
            "checkpoint_updates": 0,
            "maximum_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()) if args.device.startswith("cuda") else 0,
        },
        "occupancy": occupancy,
        "coefficient_diagnostics": coefficient_diagnostics,
        "summaries": summaries,
        "decomposition": decomposition,
        "cross_bank_action_cosine": agreements,
        "gates": gates,
        "all_values_finite": all_finite({
            "summaries": summaries,
            "decomposition": decomposition,
            "agreements": agreements,
            "coefficient_diagnostics": coefficient_diagnostics,
        }),
        "authorization": {
            "full_input_rank_theory": not teacher_pass,
            "boundary_phase_theory": teacher_pass and not moved_pass,
            "direct_exact_jvp_oracle_theory": teacher_pass and moved_pass and not self_pass,
            "new_parameter_candidate": False,
            "language_model_training": False,
            "larger_rung": False,
            "full_attention_work": False,
            "automatic_retry_or_sweep": False,
        },
    }
    path = args.output / "result.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
