#!/usr/bin/env python3
"""Audit task-functional global input-atlas ceilings for complete sparse MoE experts."""
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
)
from examples.nanogpt.analyze_sparse_moe_rolling_tangent_oracle import (
    LayerState,
    recovery_fraction,
)
from examples.nanogpt.analyze_sparse_moe_stepzero_task_gradient_oracle import (
    all_finite,
    layer_state_from_mapping,
    load_terminal_snapshot,
)


PLAN_SCHEMA = "nanogpt_sparse_moe_input_atlas_ceiling_plan_v1"
BASIS_SCHEMA = "nanogpt_sparse_moe_write_subspace_ceiling_bases_v1"
BANKS = ("discovery_a", "discovery_b", "heldout")
CAUSAL_BANKS = BANKS[:2]
ARMS = ("input_only", "paired")


def gelu_derivative(values: torch.Tensor) -> torch.Tensor:
    inv_sqrt_two = 1.0 / math.sqrt(2.0)
    inv_sqrt_two_pi = 1.0 / math.sqrt(2.0 * math.pi)
    return (
        0.5 * (1.0 + torch.erf(values * inv_sqrt_two))
        + values * torch.exp(-0.5 * values.square()) * inv_sqrt_two_pi
    )


def rademacher(
    shape: tuple[int, ...], seed: int, device: str,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    bits = torch.randint(0, 2, shape, generator=generator, dtype=torch.int8)
    return bits.mul(2).sub(1).to(device=device, dtype=torch.float32)


def subspace_overlap(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("input atlases must have equal [rank,width] shape")
    singular = torch.linalg.svdvals(left.double() @ right.double().T)
    return float(singular.square().mean())


def action_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.double().reshape(-1)
    right = right.double().reshape(-1)
    return float((left @ right) / (left.norm() * right.norm()).clamp_min(1e-30))


def load_write_bases(path: Path, plan: dict[str, Any]) -> dict[str, torch.Tensor]:
    source = plan["source"]
    if file_sha256(path) != source["write_basis_artifact_sha256"]:
        raise ValueError("write-basis artifact hash drift")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != BASIS_SCHEMA:
        raise ValueError("write-basis artifact schema mismatch")
    rank = int(source["write_atlas_rank"])
    width = int(source["input_width"])
    result: dict[str, torch.Tensor] = {}
    for bank in BANKS:
        basis = payload["bases"]["global_shared_rank619"][bank]
        if tuple(basis.shape[:2]) != (1, width):
            raise ValueError("global write-basis shape drift")
        result[bank] = basis[0, :, :rank].float().contiguous()
    return result


@torch.no_grad()
def jacobian_node_grams(
    samples: torch.Tensor,
    state: LayerState,
    *,
    probes: int,
    seed: int,
    device: str,
) -> torch.Tensor:
    """Return exact-probe estimates [experts,width,width] of J(x)^T J(x)."""
    if samples.ndim != 3:
        raise ValueError("samples must be [experts,samples,width]")
    live_state = state.to(device)
    experts, sample_count, width = samples.shape
    grams = []
    for expert in range(experts):
        x = samples[expert].to(device=device, dtype=torch.float32)
        c_fc = live_state.c_fc[expert]
        c_proj = live_state.c_proj[expert]
        pre = x @ c_fc.T
        probe = rademacher(
            (sample_count, int(probes), width),
            int(seed) + 1009 * expert,
            device,
        )
        hidden_pullback = torch.einsum("spd,dh->sph", probe, c_proj)
        input_pullback = torch.einsum(
            "sph,hd->spd",
            hidden_pullback * gelu_derivative(pre)[:, None, :],
            c_fc,
        )
        rows = input_pullback.reshape(-1, width).double()
        grams.append((rows.T @ rows / float(probes)).cpu())
    return torch.stack(grams)


def equal_node_input_atlas(
    node_grams: dict[int, torch.Tensor], rank: int, device: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    aggregate: torch.Tensor | None = None
    traces = []
    for layer in sorted(node_grams):
        for gram in node_grams[layer]:
            live = gram.to(device=device, dtype=torch.float64)
            trace = live.diagonal().sum().clamp_min(1e-30)
            traces.append(float(trace))
            normalized = live / trace
            aggregate = normalized if aggregate is None else aggregate + normalized
    if aggregate is None:
        raise ValueError("empty node-Gram inventory")
    values, vectors = torch.linalg.eigh(aggregate)
    basis = vectors[:, -int(rank) :].flip(1).T.contiguous().cpu().float()
    return basis, {
        "minimum_node_trace": min(traces),
        "maximum_node_trace": max(traces),
        "leading_eigenvalue": float(values[-1]),
        "rank_boundary_eigenvalue": float(values[-int(rank)]),
        "minimum_eigenvalue": float(values[0]),
    }


def metric_recovery(
    node_grams: dict[int, torch.Tensor], basis: torch.Tensor,
) -> dict[str, Any]:
    live_basis = basis.double()
    captured, energy = 0.0, 0.0
    by_node: dict[str, float] = {}
    for layer in sorted(node_grams):
        for expert, gram in enumerate(node_grams[layer]):
            gram = gram.double()
            node_energy = float(gram.diagonal().sum())
            node_captured = float(torch.einsum(
                "rd,de,re->", live_basis, gram, live_basis
            ))
            by_node[f"{layer}:{expert}"] = node_captured / max(node_energy, 1e-30)
            captured += node_captured
            energy += node_energy
    return {
        "jacobian_energy_recovery": captured / max(energy, 1e-30),
        "minimum_node_recovery": min(by_node.values()),
        "by_node": by_node,
    }


@torch.no_grad()
def ridge_coefficients(
    samples: torch.Tensor,
    c_fc: torch.Tensor,
    basis: torch.Tensor,
    *,
    ridge_scale: float,
    device: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Fit A in preactivation=(x U^T) A for every expert."""
    live_basis = basis.to(device=device, dtype=torch.float64)
    coefficients, diagnostics = [], []
    for expert in range(samples.shape[0]):
        x = samples[expert].to(device=device, dtype=torch.float64)
        target = x @ c_fc[expert].to(device=device, dtype=torch.float64).T
        z = x @ live_basis.T
        gram = z.T @ z
        ridge = float(ridge_scale) * float(gram.diagonal().sum()) / gram.shape[0]
        system = gram + ridge * torch.eye(
            gram.shape[0], device=device, dtype=torch.float64
        )
        rhs = z.T @ target
        factor = torch.linalg.cholesky(system)
        solution = torch.cholesky_solve(rhs, factor)
        predicted = z @ solution
        diagonal = system.diagonal()
        diagnostics.append({
            "ridge": ridge,
            "preactivation_recovery": recovery_fraction(predicted, target),
            "normal_diagonal_dynamic_range": float(
                diagonal.max() / diagonal.min().clamp_min(1e-30)
            ),
            "relative_normal_equation_residual": float(
                (system @ solution - rhs).norm() / rhs.norm().clamp_min(1e-30)
            ),
        })
        coefficients.append(solution.cpu().float())
    return torch.stack(coefficients), {
        "mean_preactivation_recovery": sum(
            row["preactivation_recovery"] for row in diagnostics
        ) / len(diagnostics),
        "minimum_preactivation_recovery": min(
            row["preactivation_recovery"] for row in diagnostics
        ),
        "maximum_normal_diagonal_dynamic_range": max(
            row["normal_diagonal_dynamic_range"] for row in diagnostics
        ),
        "maximum_relative_normal_equation_residual": max(
            row["relative_normal_equation_residual"] for row in diagnostics
        ),
        "by_expert": diagnostics,
    }


def oracle_function_and_jvp(
    inputs: torch.Tensor,
    directions: torch.Tensor,
    basis: torch.Tensor,
    coefficients: torch.Tensor,
    c_proj: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if inputs.shape != directions.shape:
        raise ValueError("oracle input/JVP shapes disagree")
    z = inputs @ basis.T
    z_jvp = directions @ basis.T
    pre = z @ coefficients
    pre_jvp = z_jvp @ coefficients
    output = F.gelu(pre) @ c_proj.T
    output_jvp = (gelu_derivative(pre) * pre_jvp) @ c_proj.T
    return output, output_jvp


def _empty_accumulator(experts: int) -> dict[str, Any]:
    return {
        "error": 0.0,
        "energy": 0.0,
        "jvp_error": 0.0,
        "jvp_energy": 0.0,
        "expert_error": torch.zeros(experts, dtype=torch.float64),
        "expert_energy": torch.zeros(experts, dtype=torch.float64),
    }


@torch.no_grad()
def evaluate_heldout(
    states: dict[int, LayerState],
    heldout_inputs: dict[int, torch.Tensor],
    input_bases: dict[str, torch.Tensor],
    coefficients: dict[str, dict[int, torch.Tensor]],
    write_bases: dict[str, torch.Tensor],
    plan: dict[str, Any],
    device: str,
    chunk_size: int = 1024,
) -> tuple[dict[str, Any], float]:
    source = plan["source"]
    experts = int(source["num_experts"])
    top_k = int(source["outer_moe_top_k"])
    per_layer: dict[str, dict[str, dict[str, Any]]] = {
        bank: {arm: {} for arm in ARMS} for bank in BANKS
    }
    live_input_bases = {
        bank: input_bases[bank].to(device=device, dtype=torch.float32)
        for bank in BANKS
    }
    live_write_bases = {
        bank: write_bases[bank].to(device=device, dtype=torch.float32)
        for bank in BANKS
    }
    live_coefficients = {
        bank: {
            layer: coefficients[bank][layer].to(device=device, dtype=torch.float32)
            for layer in coefficients[bank]
        }
        for bank in BANKS
    }
    cross = {"dot": 0.0, "left": 0.0, "right": 0.0}
    for layer in [int(value) for value in source["layers"]]:
        state = states[layer].to(device)
        activations = heldout_inputs[layer]
        directions = rademacher(
            tuple(activations.shape),
            int(plan["data_protocol"]["analytic_jvp_probe_seed"]) + 17 * layer,
            "cpu",
        )
        accumulators = {
            bank: {arm: _empty_accumulator(experts) for arm in ARMS}
            for bank in BANKS
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
            target = torch.zeros_like(x)
            target_jvp = torch.zeros_like(x)
            predicted = {
                bank: {arm: torch.zeros_like(x) for arm in ARMS}
                for bank in BANKS
            }
            predicted_jvp = {
                bank: {arm: torch.zeros_like(x) for arm in ARMS}
                for bank in BANKS
            }
            for expert in range(experts):
                locations = (selected == expert).nonzero(as_tuple=False)
                if not locations.numel():
                    continue
                token, slot = locations[:, 0], locations[:, 1]
                expert_input = x.index_select(0, token)[None]
                expert_direction = direction.index_select(0, token)[None]
                target_output, target_expert_jvp = dense_function_and_jvp(
                    expert_input,
                    expert_direction,
                    state.c_fc[expert : expert + 1],
                    state.c_proj[expert : expert + 1].transpose(1, 2),
                )
                target_output, target_expert_jvp = target_output[0], target_expert_jvp[0]
                weight = probabilities[token, slot, None]
                target.index_add_(0, token, target_output * weight)
                target_jvp.index_add_(0, token, target_expert_jvp * weight)
                for bank in BANKS:
                    basis = live_input_bases[bank]
                    coeff = live_coefficients[bank][layer][expert]
                    output, output_jvp = oracle_function_and_jvp(
                        expert_input[0],
                        expert_direction[0],
                        basis,
                        coeff,
                        state.c_proj[expert],
                    )
                    write = live_write_bases[bank]
                    paired = (output @ write) @ write.T
                    paired_jvp = (output_jvp @ write) @ write.T
                    for arm, live_output, live_jvp in (
                        ("input_only", output, output_jvp),
                        ("paired", paired, paired_jvp),
                    ):
                        predicted[bank][arm].index_add_(
                            0, token, live_output * weight
                        )
                        predicted_jvp[bank][arm].index_add_(
                            0, token, live_jvp * weight
                        )
                        row = accumulators[bank][arm]
                        row["expert_error"][expert] += float(
                            (live_output - target_output).square().sum()
                        )
                        row["expert_energy"][expert] += float(
                            target_output.square().sum()
                        )
            target_energy = float(target.square().sum())
            target_jvp_energy = float(target_jvp.square().sum())
            for bank in BANKS:
                for arm in ARMS:
                    row = accumulators[bank][arm]
                    row["error"] += float(
                        (predicted[bank][arm] - target).square().sum()
                    )
                    row["energy"] += target_energy
                    row["jvp_error"] += float(
                        (predicted_jvp[bank][arm] - target_jvp).square().sum()
                    )
                    row["jvp_energy"] += target_jvp_energy
            left = predicted[CAUSAL_BANKS[0]]["paired"]
            right = predicted[CAUSAL_BANKS[1]]["paired"]
            cross["dot"] += float((left * right).sum())
            cross["left"] += float(left.square().sum())
            cross["right"] += float(right.square().sum())
        for bank in BANKS:
            for arm in ARMS:
                row = accumulators[bank][arm]
                expert_recovery = 1.0 - row["expert_error"] / row[
                    "expert_energy"
                ].clamp_min(1e-30)
                per_layer[bank][arm][str(layer)] = {
                    "mixture_recovery": 1.0 - row["error"] / max(row["energy"], 1e-30),
                    "jvp_recovery": 1.0 - row["jvp_error"] / max(
                        row["jvp_energy"], 1e-30
                    ),
                    "expert_recovery": expert_recovery.tolist(),
                }
    summaries: dict[str, Any] = {}
    for bank in BANKS:
        summaries[bank] = {}
        for arm in ARMS:
            rows = per_layer[bank][arm]
            values = list(rows.values())
            summaries[bank][arm] = {
                "by_layer": rows,
                "aggregate": {
                    "mixture_recovery_mean": sum(
                        float(row["mixture_recovery"]) for row in values
                    ) / len(values),
                    "mixture_recovery_minimum_layer": min(
                        float(row["mixture_recovery"]) for row in values
                    ),
                    "jvp_recovery_mean": sum(
                        float(row["jvp_recovery"]) for row in values
                    ) / len(values),
                    "minimum_expert_recovery": min(
                        min(row["expert_recovery"]) for row in values
                    ),
                },
            }
    agreement = cross["dot"] / math.sqrt(
        max(cross["left"] * cross["right"], 1e-30)
    )
    return summaries, agreement


def absolute_gates(
    row: dict[str, Any], *, prefix: str, frozen: dict[str, Any],
) -> dict[str, bool]:
    return {
        "mean_output_pass": row["mixture_recovery_mean"] >= float(
            frozen[f"{prefix}_mixture_recovery_mean_min_each_causal_bank"]
        ),
        "every_layer_pass": row["mixture_recovery_minimum_layer"] >= float(
            frozen[f"{prefix}_mixture_recovery_every_layer_min_each_causal_bank"]
        ),
        "every_expert_pass": row["minimum_expert_recovery"] >= float(
            frozen[f"{prefix}_expert_recovery_min_each_causal_bank"]
        ),
        "jvp_pass": row["jvp_recovery_mean"] >= float(
            frozen[f"{prefix}_jvp_recovery_mean_min_each_causal_bank"]
        ),
    }


def validate_plan(plan: dict[str, Any], plan_path: Path) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("input-atlas ceiling plan schema mismatch")
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
    for key in ("fullwidth_result", "write_ceiling_result"):
        if file_sha256(root / source[key]) != source[f"{key}_sha256"]:
            raise ValueError(f"source result hash drift: {key}")
    if not file_sha256(plan_path):
        raise AssertionError("unreachable empty plan hash")


def run_preflight(device: str) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(20261691)
    experts, samples, width, hidden, rank = 2, 64, 32, 48, 20
    basis, _ = torch.linalg.qr(torch.randn(width, rank, generator=generator))
    basis = basis.T.contiguous()
    coefficients = torch.randn(experts, rank, hidden, generator=generator) * 0.1
    c_fc = torch.einsum("erh,rd->ehd", coefficients, basis)
    c_proj = torch.randn(experts, width, hidden, generator=generator) * 0.1
    state = LayerState(
        torch.randn(experts, width, generator=generator), c_fc, c_proj
    )
    live = torch.randn(experts, samples, width, generator=generator)
    grams = jacobian_node_grams(
        live, state, probes=2, seed=20261692, device=device
    )
    fitted, diagnostics = ridge_coefficients(
        live, c_fc, basis, ridge_scale=1e-8, device=device
    )
    x = live[0].to(device)
    direction = torch.randn(x.shape, generator=generator).to(device)
    target, target_jvp = dense_function_and_jvp(
        x[None], direction[None], c_fc[0:1].to(device),
        c_proj[0:1].to(device).transpose(1, 2),
    )
    predicted, predicted_jvp = oracle_function_and_jvp(
        x, direction, basis.to(device), fitted[0].to(device), c_proj[0].to(device)
    )
    return {
        "schema_version": "nanogpt_sparse_moe_input_atlas_ceiling_preflight_v1",
        "gram_shape": list(grams.shape),
        "all_finite": all_finite({"grams": grams, "diagnostics": diagnostics}),
        "output_recovery": recovery_fraction(predicted, target[0]),
        "jvp_recovery": recovery_fraction(predicted_jvp, target_jvp[0]),
        "maximum_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else 0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
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
        print(json.dumps(run_preflight(args.device), indent=2, sort_keys=True))
        return
    if any(value is None for value in (
        args.write_bases, args.terminal_snapshot, args.data_dir, args.output
    )):
        parser.error(
            "audit requires --write-bases, --terminal-snapshot, --data-dir, and --output"
        )

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
    write_bases = load_write_bases(args.write_bases, plan)
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
    samples: dict[str, dict[int, torch.Tensor]] = {bank: {} for bank in BANKS}
    occupancy: dict[str, dict[str, list[int]]] = {bank: {} for bank in BANKS}
    node_grams: dict[str, dict[int, torch.Tensor]] = {bank: {} for bank in BANKS}
    split_index = {bank: index for index, bank in enumerate(BANKS)}
    for bank in BANKS:
        for layer in layers:
            sampled, counts = route_and_sample(
                states[layer],
                inputs[bank][layer],
                top_k=int(source["outer_moe_top_k"]),
                samples_per_expert=int(protocol["samples_per_expert_each_split"]),
                seed=(
                    int(protocol["sample_selection_seed_base"])
                    + 1009 * split_index[bank]
                    + 17 * layer
                ),
            )
            samples[bank][layer] = sampled
            occupancy[bank][str(layer)] = counts
            node_grams[bank][layer] = jacobian_node_grams(
                sampled,
                states[layer],
                probes=int(protocol["jacobian_output_probes_per_sample"]),
                seed=int(protocol["jacobian_probe_seeds"][bank]) + 17 * layer,
                device=args.device,
            )

    input_bases: dict[str, torch.Tensor] = {}
    atlas_diagnostics: dict[str, Any] = {}
    metric_ceilings: dict[str, Any] = {}
    for bank in BANKS:
        input_bases[bank], atlas_diagnostics[bank] = equal_node_input_atlas(
            node_grams[bank], int(source["input_atlas_rank"]), args.device
        )
        metric_ceilings[bank] = metric_recovery(
            node_grams["heldout"], input_bases[bank]
        )
    overlap = subspace_overlap(
        input_bases[CAUSAL_BANKS[0]], input_bases[CAUSAL_BANKS[1]]
    )

    coefficients: dict[str, dict[int, torch.Tensor]] = {
        bank: {} for bank in BANKS
    }
    coefficient_diagnostics: dict[str, dict[str, Any]] = {
        bank: {} for bank in BANKS
    }
    for bank in BANKS:
        for layer in layers:
            fitted, diagnostics = ridge_coefficients(
                samples[bank][layer],
                states[layer].c_fc,
                input_bases[bank],
                ridge_scale=1e-4,
                device=args.device,
            )
            coefficients[bank][layer] = fitted
            coefficient_diagnostics[bank][str(layer)] = diagnostics

    summaries, paired_agreement = evaluate_heldout(
        states,
        inputs["heldout"],
        input_bases,
        coefficients,
        write_bases,
        plan,
        args.device,
    )

    frozen = plan["frozen_gates"]
    occupancy_pass = {
        bank: min(min(row) for row in occupancy[bank].values()) >= int(
            frozen["minimum_assignments_per_expert_each_split"]
        )
        for bank in BANKS
    }
    gates: dict[str, Any] = {
        "cross_bank_input_atlas_overlap_pass": overlap >= float(
            frozen["cross_bank_input_atlas_overlap_min"]
        ),
        "cross_bank_paired_action_cosine_pass": paired_agreement >= float(
            frozen["cross_bank_paired_action_cosine_min"]
        ),
    }
    for bank in BANKS:
        gates[bank] = {
            "metric_pass": metric_ceilings[bank]["jacobian_energy_recovery"] >= float(
                frozen["metric_jacobian_energy_recovery_mean_min_each_causal_atlas"]
            ),
            "occupancy_pass": occupancy_pass[bank],
        }
        for arm in ARMS:
            arm_gates = absolute_gates(
                summaries[bank][arm]["aggregate"],
                prefix=arm,
                frozen=frozen,
            )
            arm_gates["all_pass"] = all(arm_gates.values())
            gates[bank][arm] = arm_gates

    same_metric = gates["heldout"]["metric_pass"]
    same_input = gates["heldout"]["input_only"]["all_pass"]
    same_paired = gates["heldout"]["paired"]["all_pass"]
    causal_input = all(
        gates[bank]["metric_pass"]
        and gates[bank]["input_only"]["all_pass"]
        and gates[bank]["occupancy_pass"]
        for bank in CAUSAL_BANKS
    ) and gates["cross_bank_input_atlas_overlap_pass"]
    causal_paired = causal_input and all(
        gates[bank]["paired"]["all_pass"] for bank in CAUSAL_BANKS
    ) and gates["cross_bank_paired_action_cosine_pass"]
    if not same_metric:
        classification = "GLOBAL_RANK480_INPUT_TANGENT_CAPACITY_REJECTED"
    elif not same_input:
        classification = "GLOBAL_RANK480_INPUT_REALIZATION_REJECTED"
    elif not same_paired:
        classification = "INPUT_WRITE_SAME_SPLIT_INTERACTION_REJECTED"
    elif not causal_input:
        classification = "INPUT_ATLAS_CAUSAL_ACQUISITION_REJECTED"
    elif not causal_paired:
        classification = "INPUT_WRITE_CAUSAL_COADAPTATION_REJECTED"
    else:
        classification = "COMPACT_PSEUDORANDOM_COEFFICIENT_MAP_LOCALIZED"

    args.output.mkdir(parents=True, exist_ok=False)
    bases_path = args.output / "input_bases.pt"
    torch.save({
        "schema_version": "nanogpt_sparse_moe_input_atlas_ceiling_bases_v1",
        "bases": input_bases,
    }, bases_path)
    root = Path(__file__).resolve().parents[2]
    result = {
        "schema_version": "nanogpt_sparse_moe_input_atlas_ceiling_result_v1",
        "classification": classification,
        "identity": {
            "git_commit": git_commit(root),
            "plan_sha256": file_sha256(args.plan),
            "entrypoint_sha256": file_sha256(Path(__file__)),
            "write_basis_artifact_sha256": file_sha256(args.write_bases),
            "terminal_snapshot_sha256": file_sha256(args.terminal_snapshot),
            "dataset_manifest_sha256": file_sha256(manifest),
        },
        "execution": {
            "device": args.device,
            "wall_seconds": time.time() - started,
            "checkpoint_updates": 0,
            "input_bases_path": str(bases_path),
            "input_bases_sha256": file_sha256(bases_path),
            "maximum_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated())
                if args.device.startswith("cuda") else 0
            ),
        },
        "oracle_accounting": {
            "input_atlas_coordinates": int(source["input_atlas_rank"]) * int(
                source["input_width"]
            ),
            "dense_node_coefficient_coordinates": (
                len(layers) * int(source["num_experts"])
                * int(source["input_atlas_rank"])
                * int(source["expert_hidden_width"])
            ),
            "input_only_retains_dense_cproj": True,
            "paired_uses_terminal_derived_write_atlas": True,
            "deployable_candidate": False,
        },
        "occupancy": occupancy,
        "atlas_diagnostics": atlas_diagnostics,
        "metric_ceilings": metric_ceilings,
        "cross_bank_input_atlas_overlap": overlap,
        "coefficient_diagnostics": coefficient_diagnostics,
        "summaries": summaries,
        "cross_bank_paired_action_cosine": paired_agreement,
        "gates": gates,
        "all_values_finite": all_finite({
            "atlas_diagnostics": atlas_diagnostics,
            "metric_ceilings": metric_ceilings,
            "coefficient_diagnostics": coefficient_diagnostics,
            "summaries": summaries,
            "overlap": overlap,
            "paired_agreement": paired_agreement,
        }),
        "authorization": {
            "compact_coefficient_map_theory": causal_paired,
            "causal_atlas_acquisition_theory": same_input and same_paired and not causal_input,
            "joint_input_write_atlas_theory": causal_input and not causal_paired,
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
