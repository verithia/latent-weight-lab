#!/usr/bin/env python3
"""Measure algebraic write-subspace ceilings for complete sparse-MoE experts."""
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
from examples.nanogpt.analyze_sparse_moe_paired_alignment import file_sha256
from examples.nanogpt.analyze_sparse_moe_paired_coordinate_field_oracle import (
    function_and_jvp as dense_function_and_jvp,
    rademacher,
)
from examples.nanogpt.analyze_sparse_moe_rolling_tangent_oracle import LayerState
from examples.nanogpt.analyze_sparse_moe_stepzero_task_gradient_oracle import (
    all_finite,
    layer_state_from_mapping,
    load_terminal_snapshot,
)


PLAN_SCHEMA = "nanogpt_sparse_moe_write_subspace_ceiling_plan_v1"
TOPOLOGY_ORDER = (
    "global_shared_rank619",
    "layer_shared_rank60",
    "expert_local_rank7",
)
SPLITS = ("discovery_a", "discovery_b", "heldout")


def topology_groups(
    topology: str, layers: list[int], experts: int,
) -> list[list[tuple[int, int]]]:
    nodes = [(layer, expert) for layer in layers for expert in range(experts)]
    if topology.startswith("global_shared"):
        return [nodes]
    if topology.startswith("layer_shared"):
        return [[(layer, expert) for expert in range(experts)] for layer in layers]
    if topology.startswith("expert_local"):
        return [[node] for node in nodes]
    raise ValueError(f"unknown topology: {topology}")


def group_index(
    topology: str, layer_index: int, expert: int, experts: int,
) -> int:
    if topology.startswith("global_shared"):
        return 0
    if topology.startswith("layer_shared"):
        return layer_index
    if topology.startswith("expert_local"):
        return layer_index * experts + expert
    raise ValueError(f"unknown topology: {topology}")


def aggregate_covariances(
    node_covariance: torch.Tensor,
    topology: str,
) -> torch.Tensor:
    """Aggregate [layers, experts, d, d] covariance into sharing groups."""
    if node_covariance.ndim != 4:
        raise ValueError("node covariance must be [layers, experts, d, d]")
    if topology.startswith("global_shared"):
        return node_covariance.sum(dim=(0, 1), keepdim=False)[None]
    if topology.startswith("layer_shared"):
        return node_covariance.sum(dim=1)
    if topology.startswith("expert_local"):
        return node_covariance.flatten(0, 1)
    raise ValueError(f"unknown topology: {topology}")


def leading_eigensystem(
    covariance: torch.Tensor, device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    values, vectors = torch.linalg.eigh(covariance.to(device=device))
    return values.flip(-1).cpu(), vectors.flip(-1).cpu()


def subspace_overlap(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("subspaces must have equal [d, rank] shape")
    singular = torch.linalg.svdvals(left.double().T @ right.double())
    return float(singular.square().mean())


def project_actions(actions: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Project [experts, samples, d] through [groups, d, rank] bases."""
    if basis.shape[0] == 1:
        return (actions @ basis[0]) @ basis[0].T
    if basis.shape[0] == actions.shape[0]:
        coefficients = torch.einsum("end,edr->enr", actions, basis)
        return torch.einsum("enr,edr->end", coefficients, basis)
    raise ValueError("action/basis sharing shape mismatch")


def minimum_rank(curve: torch.Tensor, threshold: float) -> int | None:
    indices = torch.nonzero(curve >= float(threshold), as_tuple=False).flatten()
    return None if not indices.numel() else int(indices[0]) + 1


def spectral_curve(
    basis: torch.Tensor,
    heldout_output_covariance: torch.Tensor,
    heldout_jvp_covariance: torch.Tensor,
    registered_rank: int,
) -> dict[str, Any]:
    basis = basis.double()
    output_covariance = heldout_output_covariance.double()
    jvp_covariance = heldout_jvp_covariance.double()
    output_diagonal = torch.einsum(
        "gdr,gde,ger->gr", basis, output_covariance, basis
    ).clamp_min(0.0)
    jvp_diagonal = torch.einsum(
        "gdr,gde,ger->gr", basis, jvp_covariance, basis
    ).clamp_min(0.0)
    output_energy = output_covariance.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-30)
    jvp_energy = jvp_covariance.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-30)
    output_group = output_diagonal.cumsum(-1) / output_energy[:, None]
    jvp_group = jvp_diagonal.cumsum(-1) / jvp_energy[:, None]
    output_mean = output_diagonal.sum(0).cumsum(0) / output_energy.sum()
    jvp_mean = jvp_diagonal.sum(0).cumsum(0) / jvp_energy.sum()
    joint_both = (output_mean >= 0.8) & (jvp_mean >= 0.6)
    registered = min(int(registered_rank), basis.shape[-1]) - 1
    return {
        "registered_rank": int(registered_rank),
        "registered_output_recovery_energy_weighted": float(output_mean[registered]),
        "registered_output_recovery_minimum_group": float(output_group[:, registered].min()),
        "registered_jvp_recovery_energy_weighted": float(jvp_mean[registered]),
        "registered_jvp_recovery_minimum_group": float(jvp_group[:, registered].min()),
        "minimum_rank_output_mean_0p8": minimum_rank(output_mean, 0.8),
        "minimum_rank_output_every_group_0p5": minimum_rank(output_group.min(0).values, 0.5),
        "minimum_rank_jvp_mean_0p6": minimum_rank(jvp_mean, 0.6),
        "minimum_rank_joint_output0p8_jvp0p6": (
            None if not joint_both.any() else int(torch.nonzero(joint_both)[0]) + 1
        ),
    }


@torch.no_grad()
def acquire_sample_actions(
    states: dict[int, LayerState],
    inputs: dict[str, dict[int, torch.Tensor]],
    plan: dict[str, Any],
    device: str,
) -> tuple[
    dict[str, dict[int, dict[str, torch.Tensor]]],
    dict[str, dict[str, list[int]]],
]:
    source = plan["source"]
    layers = [int(value) for value in source["layers"]]
    samples_per_expert = int(plan["data_protocol"]["samples_per_expert_each_split"])
    probe_seeds = plan["data_protocol"]["analytic_jvp_probe_seeds"]
    actions: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    occupancy: dict[str, dict[str, list[int]]] = {}
    for split_index, split in enumerate(SPLITS):
        actions[split], occupancy[split] = {}, {}
        for layer in layers:
            sampled, counts = route_and_sample(
                states[layer], inputs[split][layer],
                top_k=int(source["outer_moe_top_k"]),
                samples_per_expert=samples_per_expert,
                seed=20261311 + 1009 * split_index + 17 * layer,
            )
            occupancy[split][str(layer)] = counts
            live = sampled.to(device=device, dtype=torch.float32)
            direction = rademacher(
                tuple(live.shape), int(probe_seeds[split]) + 17 * layer, device
            )
            state = states[layer].to(device)
            output, jvp = dense_function_and_jvp(
                live, direction, state.c_fc, state.c_proj.transpose(1, 2)
            )
            actions[split][layer] = {
                "output": output.cpu(),
                "jvp": jvp.cpu(),
            }
    return actions, occupancy


@torch.no_grad()
def node_covariances(
    actions: dict[str, dict[int, dict[str, torch.Tensor]]],
    layers: list[int],
    experts: int,
    input_width: int,
    device: str,
) -> dict[str, dict[str, torch.Tensor]]:
    result: dict[str, dict[str, torch.Tensor]] = {}
    for split in SPLITS:
        output_rows, jvp_rows = [], []
        for layer in layers:
            output = actions[split][layer]["output"].to(device=device, dtype=torch.float64)
            jvp = actions[split][layer]["jvp"].to(device=device, dtype=torch.float64)
            output_rows.append(torch.einsum("esd,esf->edf", output, output).cpu())
            jvp_rows.append(torch.einsum("esd,esf->edf", jvp, jvp).cpu())
        result[split] = {
            "output": torch.stack(output_rows).reshape(
                len(layers), experts, input_width, input_width
            ),
            "jvp": torch.stack(jvp_rows).reshape(
                len(layers), experts, input_width, input_width
            ),
        }
    return result


def derive_bases_and_curves(
    covariances: dict[str, dict[str, torch.Tensor]],
    plan: dict[str, Any],
    device: str,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    source = plan["source"]
    layers = [int(value) for value in source["layers"]]
    experts = int(source["num_experts"])
    fixed_bases: dict[str, dict[str, torch.Tensor]] = {}
    curves: dict[str, Any] = {}
    for topology in TOPOLOGY_ORDER:
        rank = int(plan["topologies"][topology]["rank"])
        heldout_output = aggregate_covariances(covariances["heldout"]["output"], topology)
        heldout_jvp = aggregate_covariances(covariances["heldout"]["jvp"], topology)
        fixed_bases[topology], curves[topology] = {}, {}
        for split in SPLITS:
            output_covariance = aggregate_covariances(covariances[split]["output"], topology)
            jvp_covariance = aggregate_covariances(covariances[split]["jvp"], topology)
            joint_covariance = output_covariance + 0.10 * jvp_covariance
            _values, joint_vectors = leading_eigensystem(joint_covariance, device)
            fixed_bases[topology][split] = joint_vectors[..., :rank].float().contiguous()
            curves[topology][split] = {
                "joint_basis": spectral_curve(
                    joint_vectors, heldout_output, heldout_jvp, rank
                )
            }
            _values, output_vectors = leading_eigensystem(output_covariance, device)
            curves[topology][split]["optimistic_output_basis"] = spectral_curve(
                output_vectors, heldout_output, heldout_jvp, rank
            )
            del output_vectors
            _values, jvp_vectors = leading_eigensystem(jvp_covariance, device)
            curves[topology][split]["optimistic_jvp_basis"] = spectral_curve(
                jvp_vectors, heldout_output, heldout_jvp, rank
            )
            del jvp_vectors, joint_vectors
        groups = topology_groups(topology, layers, experts)
        if len(groups) != fixed_bases[topology]["heldout"].shape[0]:
            raise RuntimeError("topology group inventory drift")
    return fixed_bases, curves


def _new_accumulator(layers: list[int], experts: int) -> dict[str, Any]:
    return {
        "error": 0.0, "energy": 0.0, "jvp_error": 0.0, "jvp_energy": 0.0,
        "layers": {
            str(layer): {"error": 0.0, "energy": 0.0, "jvp_error": 0.0, "jvp_energy": 0.0}
            for layer in layers
        },
        "expert_error": torch.zeros(len(layers), experts, dtype=torch.float64),
        "expert_energy": torch.zeros(len(layers), experts, dtype=torch.float64),
    }


@torch.no_grad()
def evaluate_full_routed(
    states: dict[int, LayerState],
    heldout_inputs: dict[int, torch.Tensor],
    bases: dict[str, dict[str, torch.Tensor]],
    plan: dict[str, Any],
    device: str,
    chunk_size: int = 1024,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = plan["source"]
    layers = [int(value) for value in source["layers"]]
    experts = int(source["num_experts"])
    top_k = int(source["outer_moe_top_k"])
    accumulators = {
        topology: {split: _new_accumulator(layers, experts) for split in SPLITS}
        for topology in TOPOLOGY_ORDER
    }
    agreement = {
        topology: {"dot": 0.0, "left": 0.0, "right": 0.0}
        for topology in TOPOLOGY_ORDER
    }
    for layer_index, layer in enumerate(layers):
        state = states[layer].to(device)
        activations = heldout_inputs[layer]
        directions = rademacher(
            tuple(activations.shape),
            int(plan["data_protocol"]["analytic_jvp_probe_seeds"]["heldout"]) + 17 * layer,
            "cpu",
        )
        for start in range(0, activations.shape[0], int(chunk_size)):
            stop = min(activations.shape[0], start + int(chunk_size))
            x = activations[start:stop].to(device=device, dtype=torch.float32)
            direction = directions[start:stop].to(device=device)
            logits = x @ state.router.T
            tie = torch.arange(experts, device=device, dtype=x.dtype)
            selected = torch.topk(
                logits - tie * torch.finfo(x.dtype).eps,
                top_k, dim=-1, largest=True, sorted=True,
            ).indices
            probabilities = F.softmax(logits.gather(-1, selected), dim=-1)
            expert_input = x[None].expand(experts, -1, -1)
            expert_direction = direction[None].expand(experts, -1, -1)
            target_expert, target_expert_jvp = dense_function_and_jvp(
                expert_input, expert_direction,
                state.c_fc, state.c_proj.transpose(1, 2),
            )
            target_by_token = target_expert.permute(1, 0, 2)
            target_jvp_by_token = target_expert_jvp.permute(1, 0, 2)
            gather_index = selected[..., None].expand(-1, -1, x.shape[-1])
            target = (
                target_by_token.gather(1, gather_index)
                * probabilities[..., None]
            ).sum(1)
            target_jvp = (
                target_jvp_by_token.gather(1, gather_index)
                * probabilities[..., None]
            ).sum(1)
            energy = float(target.square().sum())
            jvp_energy = float(target_jvp.square().sum())
            for topology in TOPOLOGY_ORDER:
                predictions: dict[str, torch.Tensor] = {}
                for split in SPLITS:
                    live_basis = bases[topology][split].to(device)
                    if topology.startswith("global_shared"):
                        selected_basis = live_basis
                    elif topology.startswith("layer_shared"):
                        selected_basis = live_basis[layer_index : layer_index + 1]
                    else:
                        first = layer_index * experts
                        selected_basis = live_basis[first : first + experts]
                    projected = project_actions(target_expert, selected_basis)
                    projected_jvp = project_actions(target_expert_jvp, selected_basis)
                    predicted = (
                        projected.permute(1, 0, 2).gather(1, gather_index)
                        * probabilities[..., None]
                    ).sum(1)
                    predicted_jvp = (
                        projected_jvp.permute(1, 0, 2).gather(1, gather_index)
                        * probabilities[..., None]
                    ).sum(1)
                    predictions[split] = predicted
                    row = accumulators[topology][split]
                    error = float((predicted - target).square().sum())
                    jvp_error = float((predicted_jvp - target_jvp).square().sum())
                    row["error"] += error
                    row["energy"] += energy
                    row["jvp_error"] += jvp_error
                    row["jvp_energy"] += jvp_energy
                    layer_row = row["layers"][str(layer)]
                    layer_row["error"] += error
                    layer_row["energy"] += energy
                    layer_row["jvp_error"] += jvp_error
                    layer_row["jvp_energy"] += jvp_energy
                    for expert in range(experts):
                        locations = (selected == expert).nonzero(as_tuple=False)
                        if not locations.numel():
                            continue
                        token = locations[:, 0]
                        expert_prediction = projected[expert].index_select(0, token)
                        expert_target = target_expert[expert].index_select(0, token)
                        row["expert_error"][layer_index, expert] += float(
                            (expert_prediction - expert_target).square().sum()
                        )
                        row["expert_energy"][layer_index, expert] += float(
                            expert_target.square().sum()
                        )
                cross = agreement[topology]
                cross["dot"] += float((predictions["discovery_a"] * predictions["discovery_b"]).sum())
                cross["left"] += float(predictions["discovery_a"].square().sum())
                cross["right"] += float(predictions["discovery_b"].square().sum())
    summaries: dict[str, Any] = {}
    agreements: dict[str, Any] = {}
    for topology in TOPOLOGY_ORDER:
        summaries[topology] = {}
        for split in SPLITS:
            row = accumulators[topology][split]
            layer_recovery = {
                layer: 1.0 - values["error"] / max(values["energy"], 1e-30)
                for layer, values in row["layers"].items()
            }
            layer_jvp = {
                layer: 1.0 - values["jvp_error"] / max(values["jvp_energy"], 1e-30)
                for layer, values in row["layers"].items()
            }
            expert_recovery = 1.0 - row["expert_error"] / row["expert_energy"].clamp_min(1e-30)
            summaries[topology][split] = {
                "mixture_recovery_mean": 1.0 - row["error"] / max(row["energy"], 1e-30),
                "mixture_recovery_by_layer": layer_recovery,
                "mixture_recovery_minimum_layer": min(layer_recovery.values()),
                "jvp_recovery_mean": 1.0 - row["jvp_error"] / max(row["jvp_energy"], 1e-30),
                "jvp_recovery_by_layer": layer_jvp,
                "minimum_expert_recovery": float(expert_recovery.min()),
                "expert_recovery": expert_recovery.tolist(),
            }
        cross = agreement[topology]
        agreements[topology] = cross["dot"] / math.sqrt(
            max(cross["left"] * cross["right"], 1e-30)
        )
    return summaries, agreements


def validate_plan(plan: dict[str, Any], plan_path: Path) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("write-subspace ceiling plan schema mismatch")
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
    if file_sha256(root / plan["source"]["parent_result"]) != plan["source"]["parent_result_sha256"]:
        raise ValueError("parent result hash drift")
    if file_sha256(plan_path) == "":
        raise AssertionError("unreachable empty plan hash")


def run_preflight(plan: dict[str, Any], device: str) -> dict[str, Any]:
    source = plan["source"]
    width = int(source["input_width"])
    layers = len(source["layers"])
    experts = int(source["num_experts"])
    generator = torch.Generator(device="cpu").manual_seed(20261321)
    diagonal = torch.rand(layers, experts, width, generator=generator, dtype=torch.float64) + 0.01
    node = torch.diag_embed(diagonal).contiguous()
    started = time.time()
    rows = {}
    for topology in TOPOLOGY_ORDER:
        covariance = aggregate_covariances(node, topology)
        values, vectors = leading_eigensystem(covariance, device)
        rows[topology] = {
            "groups": int(covariance.shape[0]),
            "minimum_eigenvalue": float(values.min()),
            "maximum_eigenvalue": float(values.max()),
            "vectors_finite": bool(torch.isfinite(vectors).all()),
        }
    elapsed = time.time() - started
    return {
        "schema_version": "nanogpt_sparse_moe_write_subspace_ceiling_preflight_v1",
        "device": device,
        "exact_width": width,
        "exact_layers": layers,
        "exact_experts": experts,
        "one_split_joint_eigensystem_seconds": elapsed,
        "projected_three_split_three_metric_eigensystem_seconds": elapsed * 9.0,
        "maximum_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else 0,
        "all_values_finite": all_finite(rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--terminal-snapshot", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    validate_plan(plan, args.plan)
    if args.preflight_only:
        print(json.dumps(run_preflight(plan, args.device), indent=2, sort_keys=True))
        return
    if args.terminal_snapshot is None or args.data_dir is None or args.output is None:
        parser.error("audit requires --terminal-snapshot, --data-dir, and --output")

    started = time.time()
    source = plan["source"]
    if file_sha256(args.terminal_snapshot) != source["terminal_manifold_snapshot_sha256"]:
        raise ValueError("terminal snapshot hash drift")
    manifest = args.data_dir / "manifest.json"
    if file_sha256(manifest) != source["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest hash drift")
    payload = load_terminal_snapshot(args.terminal_snapshot)
    if int(payload["next_iter"]) != int(source["next_iter"]):
        raise ValueError("terminal snapshot step drift")
    model = load_model(args.terminal_snapshot, args.device)
    model.eval()
    inputs = collect_protocol_inputs(model, plan, args.data_dir, args.device)
    mapping = dict(model.named_parameters())
    layers = [int(value) for value in source["layers"]]
    states = {layer: layer_state_from_mapping(mapping, layer) for layer in layers}
    del mapping, model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    actions, occupancy = acquire_sample_actions(states, inputs, plan, args.device)
    covariances = node_covariances(
        actions, layers, int(source["num_experts"]), int(source["input_width"]),
        args.device,
    )
    bases, curves = derive_bases_and_curves(covariances, plan, args.device)
    summaries, agreements = evaluate_full_routed(
        states, inputs["heldout"], bases, plan, args.device
    )
    overlaps = {}
    for topology in TOPOLOGY_ORDER:
        left, right = bases[topology]["discovery_a"], bases[topology]["discovery_b"]
        values = [subspace_overlap(left[index], right[index]) for index in range(left.shape[0])]
        overlaps[topology] = {"mean": sum(values) / len(values), "by_group": values}

    frozen = plan["frozen_gates"]
    gates: dict[str, Any] = {}
    outcomes: dict[str, str] = {}
    for topology in TOPOLOGY_ORDER:
        gates[topology] = {}
        for split in ("discovery_a", "discovery_b"):
            row = summaries[topology][split]
            bank = {
                "mean_output_pass": row["mixture_recovery_mean"] >= float(frozen["heldout_mixture_recovery_mean_min_each_discovery_basis"]),
                "every_layer_pass": row["mixture_recovery_minimum_layer"] >= float(frozen["heldout_mixture_recovery_every_layer_min_each_discovery_basis"]),
                "every_expert_pass": row["minimum_expert_recovery"] >= float(frozen["heldout_expert_recovery_min_each_discovery_basis"]),
                "jvp_pass": row["jvp_recovery_mean"] >= float(frozen["heldout_jvp_recovery_mean_min_each_discovery_basis"]),
                "action_agreement_pass": agreements[topology] >= float(frozen["heldout_cross_bank_projected_action_cosine_mean_min"]),
                "occupancy_pass": min(min(values) for values in occupancy[split].values()) >= int(frozen["minimum_assignments_per_expert_each_split"]),
                "finite_pass": all_finite(row),
            }
            bank["all_pass"] = all(bank.values())
            gates[topology][split] = bank
        gates[topology]["all_pass"] = all(
            gates[topology][split]["all_pass"]
            for split in ("discovery_a", "discovery_b")
        )
        same = summaries[topology]["heldout"]
        same_pass = (
            same["mixture_recovery_mean"] >= float(frozen["heldout_mixture_recovery_mean_min_each_discovery_basis"])
            and same["mixture_recovery_minimum_layer"] >= float(frozen["heldout_mixture_recovery_every_layer_min_each_discovery_basis"])
            and same["minimum_expert_recovery"] >= float(frozen["heldout_expert_recovery_min_each_discovery_basis"])
            and same["jvp_recovery_mean"] >= float(frozen["heldout_jvp_recovery_mean_min_each_discovery_basis"])
        )
        gates[topology]["same_split_ceiling_pass"] = same_pass
        if not same_pass:
            outcomes[topology] = "WRITE_CAPACITY_REJECTED_AT_REGISTERED_RANK"
        elif not gates[topology]["all_pass"]:
            outcomes[topology] = "WRITE_SUBSPACE_EXISTS_BUT_CAUSAL_IDENTIFICATION_REJECTED"
        else:
            outcomes[topology] = "WRITE_NECESSARY_CONDITION_PASSES_FEATURE_SIDE_LOCALIZED"

    args.output.mkdir(parents=True, exist_ok=False)
    bases_path = args.output / "write_bases.pt"
    torch.save({
        "schema_version": "nanogpt_sparse_moe_write_subspace_ceiling_bases_v1",
        "bases": bases,
    }, bases_path)
    root = Path(__file__).resolve().parents[2]
    result = {
        "schema_version": "nanogpt_sparse_moe_write_subspace_ceiling_result_v1",
        "classification": "WRITE_SUBSPACE_NECESSARY_CONDITION_AUDITED",
        "identity": {
            "git_commit": git_commit(root),
            "plan_sha256": file_sha256(args.plan),
            "entrypoint_sha256": file_sha256(Path(__file__)),
            "terminal_snapshot_sha256": file_sha256(args.terminal_snapshot),
            "dataset_manifest_sha256": file_sha256(manifest),
        },
        "execution": {
            "device": args.device,
            "wall_seconds": time.time() - started,
            "checkpoint_updates": 0,
            "bases_path": str(bases_path),
            "bases_sha256": file_sha256(bases_path),
            "maximum_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()) if args.device.startswith("cuda") else 0,
        },
        "occupancy": occupancy,
        "spectral_curves": curves,
        "fixed_rank_routed_summaries": summaries,
        "cross_bank_projected_action_cosine": agreements,
        "cross_bank_subspace_overlap": overlaps,
        "gates": gates,
        "outcomes": outcomes,
        "all_values_finite": all_finite({
            "curves": curves, "summaries": summaries,
            "agreements": agreements, "overlaps": overlaps,
        }),
        "authorization": {
            "feature_side_theory_for_passing_topology": any(
                value == "WRITE_NECESSARY_CONDITION_PASSES_FEATURE_SIDE_LOCALIZED"
                for value in outcomes.values()
            ),
            "causal_metric_or_community_theory": any(
                value == "WRITE_SUBSPACE_EXISTS_BUT_CAUSAL_IDENTIFICATION_REJECTED"
                for value in outcomes.values()
            ),
            "learned_candidate": False,
            "language_model_training": False,
            "larger_rung": False,
            "full_attention_work": False,
            "automatic_retry_or_sweep": False,
        },
    }
    result_path = args.output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
