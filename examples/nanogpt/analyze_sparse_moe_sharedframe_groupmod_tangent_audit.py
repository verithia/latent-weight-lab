#!/usr/bin/env python3
"""Audit node-gradient conflict and budgeted group modulation of a shared frame."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from examples.nanogpt.analyze_cproj_manifold import load_model
from examples.nanogpt.analyze_mlp_activation_update_alignment import git_commit
from examples.nanogpt.analyze_sparse_moe_cfc_global_conditional_tangent_audit import (
    all_tensor_values_finite,
)
from examples.nanogpt.analyze_sparse_moe_cfc_spectral_feature_oracle import (
    collect_protocol_inputs,
    route_and_sample,
)
from examples.nanogpt.analyze_sparse_moe_global_write_givens_feature_oracle import BANKS
from examples.nanogpt.analyze_sparse_moe_paired_alignment import file_sha256
from examples.nanogpt.analyze_sparse_moe_paired_coordinate_field_oracle import (
    function_and_jvp as dense_function_and_jvp,
    normalized_expert_loss,
    rademacher,
)
from examples.nanogpt.analyze_sparse_moe_sharedframe_fullrank_pregelu_oracle import (
    SharedFrameFullRankPreGelu,
    load_write_bases,
    make_module,
)
from examples.nanogpt.analyze_sparse_moe_shared_nonlinear_dictionary_oracle import (
    gelu_derivative,
)
from examples.nanogpt.analyze_sparse_moe_stepzero_task_gradient_oracle import (
    layer_state_from_mapping,
    load_terminal_snapshot,
)


PLAN_SCHEMA = "nanogpt_sparse_moe_sharedframe_groupmod_tangent_audit_plan_v1"
COORDINATE_SCHEMA = "nanogpt_sparse_moe_sharedframe_fullrank_pregelu_coordinates_v1"


def conflict_metrics(gradients: torch.Tensor) -> dict[str, Any]:
    """Summarize expert conflict for one [expert, coordinate] gradient matrix."""
    if gradients.ndim != 2 or gradients.shape[0] < 2:
        raise ValueError("gradients must be [expert, coordinate]")
    gradients = gradients.float()
    norms = gradients.norm(dim=-1)
    finite = torch.isfinite(gradients).all(dim=-1) & torch.isfinite(norms)
    nonzero = finite & (norms > 1e-12)
    normalized = gradients / norms[:, None].clamp_min(1e-30)
    cosines = torch.stack([
        normalized[left] @ normalized[right]
        for left, right in itertools.combinations(range(gradients.shape[0]), 2)
    ])
    cancellation = gradients.sum(dim=0).square().sum() / (
        gradients.shape[0] * gradients.square().sum()
    ).clamp_min(1e-30)
    return {
        "pairwise_cosine_mean": float(cosines.mean()),
        "pairwise_cosine_minimum": float(cosines.min()),
        "pairwise_cosine_maximum": float(cosines.max()),
        "negative_pair_fraction": float((cosines < 0).float().mean()),
        "cancellation_ratio": float(cancellation),
        "finite_nonzero_gradient_count": int(nonzero.sum()),
        "gradient_norm_mean": float(norms.mean()),
        "gradient_norm_minimum": float(norms.min()),
        "gradient_norm_maximum": float(norms.max()),
    }


def corresponding_cosines(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    if left.shape != right.shape or left.ndim != 3:
        raise ValueError("crossed tensors must be [layer, expert, coordinate]")
    values = F.cosine_similarity(left.float(), right.float(), dim=-1, eps=1e-12)
    return {
        "mean": float(values.mean()),
        "minimum_layer_mean": float(values.mean(dim=1).min()),
        "minimum_node": float(values.min()),
        "by_layer_mean": [float(value) for value in values.mean(dim=1)],
    }


def group_tangent_projection(
    gradient: torch.Tensor,
    frame: torch.Tensor,
    *,
    groups: int,
    side: str,
) -> tuple[torch.Tensor, float]:
    """Project a full frame gradient onto orthonormal grouped-gain tangents."""
    if gradient.shape != frame.shape or gradient.ndim != 2:
        raise ValueError("gradient and frame must be equal square matrices")
    if gradient.shape[0] != gradient.shape[1]:
        raise ValueError("group audit expects a square frame")
    if gradient.shape[0] % int(groups):
        raise ValueError("group count must divide frame width")
    width = gradient.shape[0] // int(groups)
    coefficients = []
    for group in range(int(groups)):
        selected = slice(group * width, (group + 1) * width)
        if side == "left":
            tangent = frame[selected, :]
            value = gradient[selected, :]
        elif side == "right":
            tangent = frame[:, selected]
            value = gradient[:, selected]
        else:
            raise ValueError("side must be left or right")
        coefficients.append(
            (value * tangent).sum() / tangent.norm().clamp_min(1e-30)
        )
    result = torch.stack(coefficients)
    explained = result.square().sum() / gradient.square().sum().clamp_min(1e-30)
    return result, float(explained)


def effective_coefficients_and_jvp(
    module: SharedFrameFullRankPreGelu,
    frame: torch.Tensor,
    inputs: torch.Tensor,
    directions: torch.Tensor,
    *,
    layer: int,
    expert: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Parent forward map with an explicit differentiable effective frame."""
    selected = module._selection(expert)
    compact = torch.einsum("esd,rd->esr", inputs.float(), frame)
    compact_jvp = torch.einsum("esd,rd->esr", directions.float(), frame)
    pre, pre_jvp = module._fht_pair(
        compact,
        compact_jvp,
        layer=layer,
        selected=selected,
        input_sign_index=0,
        output_sign_index=1,
        output_width=module.hidden_width,
        scale=module.expansion_scale,
    )
    pre = pre + module.hidden_bias[layer, selected, None, :].detach()
    hidden = F.gelu(pre)
    hidden_jvp = gelu_derivative(pre) * pre_jvp
    compact, compact_jvp = module._fht_pair(
        hidden,
        hidden_jvp,
        layer=layer,
        selected=selected,
        input_sign_index=2,
        output_sign_index=3,
        output_width=module.write_rank,
        scale=module.contraction_scale,
    )
    modulation = module.output_modulation[layer, selected, None, :].detach()
    return compact * modulation, compact_jvp * modulation


def node_gradient(
    module: SharedFrameFullRankPreGelu,
    inputs: torch.Tensor,
    directions: torch.Tensor,
    target: torch.Tensor,
    target_jvp: torch.Tensor,
    *,
    layer: int,
    expert: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    frame = module.input_frame().detach().clone().requires_grad_(True)
    predicted, predicted_jvp = effective_coefficients_and_jvp(
        module,
        frame,
        inputs[expert : expert + 1],
        directions[expert : expert + 1],
        layer=layer,
        expert=expert,
    )
    output_loss = normalized_expert_loss(
        predicted, target[expert : expert + 1]
    )
    jvp_loss = normalized_expert_loss(
        predicted_jvp, target_jvp[expert : expert + 1]
    )
    loss = output_loss + jvp_loss
    gradient = torch.autograd.grad(loss, frame)[0].detach()
    if not torch.isfinite(gradient).all() or float(gradient.norm()) <= 1e-12:
        raise RuntimeError("non-finite or zero effective-frame gradient")
    return gradient.cpu(), {
        "objective": float(loss.detach()),
        "output_loss": float(output_loss.detach()),
        "jvp_loss": float(jvp_loss.detach()),
    }


def summarize_energy(values: torch.Tensor) -> dict[str, Any]:
    if values.ndim != 2:
        raise ValueError("energy values must be [layer, expert]")
    return {
        "mean": float(values.mean()),
        "minimum_layer_mean": float(values.mean(dim=1).min()),
        "minimum_node": float(values.min()),
        "by_layer_mean": [float(value) for value in values.mean(dim=1)],
    }


def side_passes(
    diagonal: list[dict[str, Any]],
    stability: list[dict[str, Any]],
    gates: dict[str, Any],
) -> bool:
    return all(
        row["mean"] >= float(gates["group_tangent_explained_energy_mean_min_each_diagonal_cell"])
        and row["minimum_layer_mean"] >= float(
            gates["group_tangent_explained_energy_minimum_layer_mean_min_each_diagonal_cell"]
        )
        for row in diagonal
    ) and all(
        row["mean"] >= float(
            gates["fixed_endpoint_cross_bank_group_coefficient_cosine_mean_min"]
        )
        and row["minimum_layer_mean"] >= float(
            gates[
                "fixed_endpoint_cross_bank_group_coefficient_cosine_minimum_layer_mean_min"
            ]
        )
        for row in stability
    )


def validate_plan(plan: dict[str, Any], path: Path) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("group-modulation tangent plan schema mismatch")
    source = plan["source"]
    root = Path(__file__).resolve().parents[2]
    if file_sha256(root / plan["causal_parent"]["normalized_result"]) != (
        plan["causal_parent"]["normalized_result_sha256"]
    ):
        raise ValueError("causal parent result hash drift")
    if file_sha256(root / source["parent_plan"]) != source["parent_plan_sha256"]:
        raise ValueError("parent plan hash drift")
    budget = plan["coordinate_budget"]
    if not (
        float(budget["compression_ratio_if_one_side_is_used"]) >= 200.0
        and int(budget["candidate_coordinates_if_one_side_is_used"])
        == int(budget["parent_coordinates"])
        + int(budget["node_group_modulation_coordinates"])
    ):
        raise ValueError("group-modulation coordinate budget drift")
    identity = plan["identity"]
    if identity.get("entrypoint_sha256") is not None:
        if identity["entrypoint_sha256"] != file_sha256(Path(__file__)):
            raise ValueError("entrypoint hash drift")
        for relative, expected in identity["helper_sha256"].items():
            if file_sha256(root / relative) != expected:
                raise ValueError(f"helper hash drift: {relative}")
    if not file_sha256(path):
        raise AssertionError("unreachable empty plan hash")


def preflight(plan: dict[str, Any], device: str) -> dict[str, Any]:
    source = plan["source"]
    generator = torch.Generator(device="cpu").manual_seed(20261711)
    basis, _ = torch.linalg.qr(torch.randn(
        int(source["input_width"]), int(source["write_rank"]), generator=generator
    ))
    module = make_module(
        json.loads((Path(__file__).resolve().parents[2] / source["parent_plan"]).read_text()),
        basis,
        candidate=True,
        device=device,
    )
    samples = 1024
    inputs = torch.randn(1, samples, int(source["input_width"]), generator=generator).to(device)
    directions = torch.randn_like(inputs)
    target = torch.randn(1, samples, int(source["write_rank"]), generator=generator).to(device)
    target_jvp = torch.randn_like(target)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    gradient, diagnostics = node_gradient(
        module, inputs, directions, target, target_jvp, layer=0, expert=0
    )
    frame = module.input_frame().detach().cpu()
    projections = {
        side: group_tangent_projection(
            gradient, frame, groups=int(plan["replay_protocol"]["group_count"]), side=side
        )[1]
        for side in ("left", "right")
    }
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    cells = len(BANKS) * len(BANKS)
    nodes = len(source["layers"]) * int(source["num_experts"])
    return {
        "schema_version": "nanogpt_sparse_moe_sharedframe_groupmod_tangent_preflight_v1",
        "device": device,
        "one_exact_node_seconds": elapsed,
        "projected_gradient_audit_seconds": elapsed * cells * nodes,
        "one_gradient_shape": list(gradient.shape),
        "diagnostics": diagnostics,
        "projection_smoke": projections,
        "all_values_finite": all_tensor_values_finite(
            {"gradient": gradient, "diagnostics": diagnostics, "projections": projections}
        ),
        "maximum_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else 0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--parent-plan", type=Path)
    parser.add_argument("--parent-coordinates", type=Path)
    parser.add_argument("--write-bases", type=Path)
    parser.add_argument("--terminal-snapshot", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    validate_plan(plan, args.plan)
    if args.preflight_only:
        print(json.dumps(preflight(plan, args.device), indent=2, sort_keys=True))
        return
    required = (
        args.parent_plan, args.parent_coordinates, args.write_bases,
        args.terminal_snapshot, args.data_dir, args.output,
    )
    if any(value is None for value in required):
        parser.error("full audit requires all source and output paths")

    started = time.time()
    source = plan["source"]
    for path, expected, label in (
        (args.parent_plan, source["parent_plan_sha256"], "parent plan"),
        (args.parent_coordinates, source["parent_coordinates_sha256"], "coordinates"),
        (args.write_bases, source["write_basis_artifact_sha256"], "write bases"),
        (args.terminal_snapshot, source["terminal_manifold_snapshot_sha256"], "snapshot"),
        (args.data_dir / "manifest.json", source["dataset_manifest_sha256"], "manifest"),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"{label} hash drift")
    parent_plan = json.loads(args.parent_plan.read_text(encoding="utf-8"))
    coordinates = torch.load(args.parent_coordinates, map_location="cpu", weights_only=False)
    if coordinates.get("schema_version") != COORDINATE_SCHEMA:
        raise ValueError("parent coordinate schema drift")
    payload = load_terminal_snapshot(args.terminal_snapshot)
    if int(payload["next_iter"]) != int(source["next_iter"]):
        raise ValueError("terminal step drift")
    write_bases = load_write_bases(args.write_bases, parent_plan)
    model = load_model(args.terminal_snapshot, args.device)
    model.eval()
    inputs = collect_protocol_inputs(model, parent_plan, args.data_dir, args.device)
    mapping = dict(model.named_parameters())
    layers = [int(value) for value in source["layers"]]
    states = {layer: layer_state_from_mapping(mapping, layer) for layer in layers}
    del mapping, model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    sampled: dict[str, dict[int, torch.Tensor]] = {}
    occupancy: dict[str, dict[str, list[int]]] = {}
    replay = plan["replay_protocol"]
    for bank_index, bank in enumerate(BANKS):
        sampled[bank], occupancy[bank] = {}, {}
        for layer in layers:
            values, counts = route_and_sample(
                states[layer], inputs[bank][layer],
                top_k=int(source["outer_moe_top_k"]),
                samples_per_expert=int(replay["fit_samples_per_expert"]),
                seed=(
                    int(replay["sample_selection_seed_base"])
                    + int(replay["sample_selection_seed_bank_stride"]) * bank_index
                    + int(replay["sample_selection_seed_layer_stride"]) * layer
                ),
            )
            sampled[bank][layer] = values
            occupancy[bank][str(layer)] = counts

    full_gradients: dict[str, dict[str, torch.Tensor]] = {}
    group_coefficients: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    group_energy: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    objectives: dict[str, dict[str, dict[str, Any]]] = {}
    groups = int(replay["group_count"])
    for endpoint in BANKS:
        module = make_module(
            parent_plan, write_bases[endpoint], candidate=True, device=args.device
        )
        module.load_state_dict(coordinates["states"][endpoint]["candidate"], strict=True)
        module.eval()
        frame_cpu = module.input_frame().detach().cpu()
        full_gradients[endpoint] = {}
        group_coefficients[endpoint] = {}
        group_energy[endpoint] = {}
        objectives[endpoint] = {}
        for bank_index, data_bank in enumerate(BANKS):
            layer_gradients, layer_objectives = [], {}
            side_coefficients = {"left": [], "right": []}
            side_energies = {"left": [], "right": []}
            for layer in layers:
                live = sampled[data_bank][layer].to(args.device, dtype=torch.float32)
                directions = rademacher(
                    tuple(live.shape),
                    int(replay["jvp_probe_seed_base"])
                    + int(replay["jvp_probe_seed_bank_stride"]) * bank_index
                    + int(replay["jvp_probe_seed_layer_stride"]) * layer,
                    args.device,
                )
                state = states[layer].to(args.device)
                with torch.no_grad():
                    target, target_jvp = dense_function_and_jvp(
                        live, directions, state.c_fc, state.c_proj.transpose(1, 2)
                    )
                    target = target @ module.write_basis
                    target_jvp = target_jvp @ module.write_basis
                expert_gradients, expert_objectives = [], []
                layer_side_coefficients = {"left": [], "right": []}
                layer_side_energies = {"left": [], "right": []}
                for expert in range(int(source["num_experts"])):
                    gradient, diagnostic = node_gradient(
                        module, live, directions, target, target_jvp,
                        layer=layer, expert=expert,
                    )
                    expert_gradients.append(gradient)
                    expert_objectives.append(diagnostic)
                    for side in ("left", "right"):
                        coefficients, energy = group_tangent_projection(
                            gradient, frame_cpu, groups=groups, side=side
                        )
                        layer_side_coefficients[side].append(coefficients)
                        layer_side_energies[side].append(energy)
                layer_gradients.append(torch.stack(expert_gradients))
                layer_objectives[str(layer)] = expert_objectives
                for side in ("left", "right"):
                    side_coefficients[side].append(
                        torch.stack(layer_side_coefficients[side])
                    )
                    side_energies[side].append(
                        torch.tensor(layer_side_energies[side], dtype=torch.float32)
                    )
                del live, directions, state, target, target_jvp
            full_gradients[endpoint][data_bank] = torch.stack(layer_gradients)
            group_coefficients[endpoint][data_bank] = {
                side: torch.stack(side_coefficients[side]) for side in ("left", "right")
            }
            group_energy[endpoint][data_bank] = {
                side: torch.stack(side_energies[side]) for side in ("left", "right")
            }
            objectives[endpoint][data_bank] = layer_objectives
        del module
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    conflict = {}
    conflict_pass = True
    frozen = plan["frozen_gates"]
    for endpoint in BANKS:
        conflict[endpoint] = {}
        gradients = full_gradients[endpoint][endpoint]
        for layer_index, layer in enumerate(layers):
            row = conflict_metrics(gradients[layer_index])
            row["passed"] = (
                row["pairwise_cosine_mean"] <= float(
                    frozen["conflict_pairwise_cosine_mean_max_each_diagonal_cell_layer"]
                )
                and row["cancellation_ratio"] <= float(
                    frozen["conflict_cancellation_ratio_max_each_diagonal_cell_layer"]
                )
                and row["finite_nonzero_gradient_count"] >= int(
                    frozen["minimum_finite_nonzero_node_gradients_each_layer"]
                )
            )
            conflict[endpoint][str(layer)] = row
            conflict_pass = conflict_pass and row["passed"]

    raw_crossed = {
        "fixed_endpoint_a_data_a_vs_b": corresponding_cosines(
            full_gradients[BANKS[0]][BANKS[0]], full_gradients[BANKS[0]][BANKS[1]]
        ),
        "fixed_endpoint_b_data_a_vs_b": corresponding_cosines(
            full_gradients[BANKS[1]][BANKS[0]], full_gradients[BANKS[1]][BANKS[1]]
        ),
        "fixed_data_a_endpoint_a_vs_b": corresponding_cosines(
            full_gradients[BANKS[0]][BANKS[0]], full_gradients[BANKS[1]][BANKS[0]]
        ),
        "fixed_data_b_endpoint_a_vs_b": corresponding_cosines(
            full_gradients[BANKS[0]][BANKS[1]], full_gradients[BANKS[1]][BANKS[1]]
        ),
    }
    energy_summaries = {
        endpoint: {
            data_bank: {
                side: summarize_energy(group_energy[endpoint][data_bank][side])
                for side in ("left", "right")
            }
            for data_bank in BANKS
        }
        for endpoint in BANKS
    }
    group_stability = {
        endpoint: {
            side: corresponding_cosines(
                group_coefficients[endpoint][BANKS[0]][side],
                group_coefficients[endpoint][BANKS[1]][side],
            )
            for side in ("left", "right")
        }
        for endpoint in BANKS
    }
    side_gates = {}
    for side in ("left", "right"):
        diagonal = [energy_summaries[endpoint][endpoint][side] for endpoint in BANKS]
        stability = [group_stability[endpoint][side] for endpoint in BANKS]
        side_gates[side] = {
            "diagonal_energy_pass": all(
                row["mean"] >= float(
                    frozen["group_tangent_explained_energy_mean_min_each_diagonal_cell"]
                )
                and row["minimum_layer_mean"] >= float(
                    frozen[
                        "group_tangent_explained_energy_minimum_layer_mean_min_each_diagonal_cell"
                    ]
                )
                for row in diagonal
            ),
            "fixed_endpoint_stability_pass": all(
                row["mean"] >= float(
                    frozen["fixed_endpoint_cross_bank_group_coefficient_cosine_mean_min"]
                )
                and row["minimum_layer_mean"] >= float(
                    frozen[
                        "fixed_endpoint_cross_bank_group_coefficient_cosine_minimum_layer_mean_min"
                    ]
                )
                for row in stability
            ),
            "all_pass": side_passes(diagonal, stability, frozen),
            "worst_diagonal_mean_energy": min(row["mean"] for row in diagonal),
        }
    supported = [side for side in ("left", "right") if side_gates[side]["all_pass"]]
    selected_side = None
    if supported:
        selected_side = max(
            supported,
            key=lambda side: (side_gates[side]["worst_diagonal_mean_energy"], side == "left"),
        )
    any_energy = any(row["diagonal_energy_pass"] for row in side_gates.values())
    if not conflict_pass:
        classification = "SHARED_FRAME_NODE_GRADIENTS_NOT_CANCELLED"
    elif selected_side is not None:
        classification = "BUDGETED_NODE_GROUP_MODULATION_SUPPORTED"
    elif any_energy:
        classification = "ACTIVATION_CONDITIONED_GROUP_TANGENT_INSTABILITY"
    else:
        classification = "NODE_CONFLICT_BUT_GROUP_MODULATION_INSUFFICIENT"

    finite = all_tensor_values_finite({
        "full_gradients": full_gradients,
        "group_coefficients": group_coefficients,
        "group_energy": group_energy,
        "objectives": objectives,
        "conflict": conflict,
        "raw_crossed": raw_crossed,
        "energy_summaries": energy_summaries,
        "group_stability": group_stability,
    })
    if not finite:
        raise RuntimeError("non-finite tangent audit result")
    args.output.mkdir(parents=True, exist_ok=False)
    gradients_path = args.output / "sharedframe_groupmod_gradients.pt"
    torch.save({
        "schema_version": "nanogpt_sparse_moe_sharedframe_groupmod_gradients_v1",
        "layers": layers,
        "endpoints": list(BANKS),
        "data_banks": list(BANKS),
        "full_effective_frame_gradients": full_gradients,
        "group_coefficients": group_coefficients,
        "group_explained_energy": group_energy,
    }, gradients_path)
    root = Path(__file__).resolve().parents[2]
    result = {
        "schema_version": "nanogpt_sparse_moe_sharedframe_groupmod_tangent_audit_result_v1",
        "classification": classification,
        "selected_side": selected_side,
        "identity": {
            "git_commit": git_commit(root),
            "plan_sha256": file_sha256(args.plan),
            "entrypoint_sha256": file_sha256(Path(__file__)),
            "parent_plan_sha256": file_sha256(args.parent_plan),
            "parent_coordinates_sha256": file_sha256(args.parent_coordinates),
            "write_basis_artifact_sha256": file_sha256(args.write_bases),
            "terminal_snapshot_sha256": file_sha256(args.terminal_snapshot),
            "dataset_manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
        },
        "execution": {
            "device": args.device,
            "wall_seconds": time.time() - started,
            "checkpoint_updates": 0,
            "maximum_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated()) if args.device.startswith("cuda") else 0
            ),
            "gradients_path": str(gradients_path),
            "gradients_sha256": file_sha256(gradients_path),
        },
        "coordinate_budget": plan["coordinate_budget"],
        "occupancy": occupancy,
        "objectives": objectives,
        "conflict": conflict,
        "raw_crossed_cosine": raw_crossed,
        "group_energy": energy_summaries,
        "group_coefficient_stability": group_stability,
        "gates": {
            "shared_node_gradient_conflict_pass": conflict_pass,
            "side_gates": side_gates,
            "all_values_and_gradients_finite": finite,
        },
        "authorization": {
            "group_modulation_functional_oracle": (
                classification == "BUDGETED_NODE_GROUP_MODULATION_SUPPORTED"
            ),
            "selected_group_modulation_side": selected_side,
            "node_private_frame_ceiling": (
                classification == "NODE_CONFLICT_BUT_GROUP_MODULATION_INSUFFICIENT"
            ),
            "activation_conditioned_audit": (
                classification == "ACTIVATION_CONDITIONED_GROUP_TANGENT_INSTABILITY"
            ),
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
