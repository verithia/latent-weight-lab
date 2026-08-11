#!/usr/bin/env python3
"""Gate an activation-conditioned fixed-FHT latent function for sparse-MoE c_proj."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from examples.nanogpt.analyze_mlp_activation_update_alignment import git_commit
from examples.nanogpt.analyze_residual_compatibility import fixed_validation_batches
from examples.nanogpt.analyze_sparse_moe_paired_alignment import (
    collect_inputs,
    file_sha256,
    tensor_sha256,
)
from examples.nanogpt.analyze_sparse_moe_paired_atom_oracle import union_fieldnames
from examples.nanogpt.analyze_sparse_moe_rolling_tangent_oracle import (
    LayerState,
    recovery_fraction,
)
from examples.nanogpt.analyze_sparse_moe_stepzero_task_gradient_oracle import (
    all_finite,
    layer_state_from_mapping,
    load_terminal_snapshot,
    model_from_exact_stepzero,
    selected_stepzero_hashes,
)
from latent_weight_lab.block_fht import (
    next_power_of_two,
    normalized_fht_last_dim,
    signs_for,
)


PLAN_SCHEMA = "nanogpt_sparse_moe_cproj_context_modulated_fht_oracle_plan_v1"


@dataclass
class ExpertFrame:
    tokens: torch.Tensor
    probabilities: torch.Tensor
    hidden: torch.Tensor


def routed_hidden_frames(
    state: LayerState,
    activations: torch.Tensor,
    top_k: int,
    device: str,
) -> tuple[list[ExpertFrame], list[int]]:
    state = state.to(device)
    x = activations.to(device=device, dtype=torch.float32)
    logits = x @ state.router.T
    tie = torch.arange(logits.shape[-1], device=x.device, dtype=x.dtype)
    selected = torch.topk(
        logits - tie * torch.finfo(x.dtype).eps,
        top_k,
        dim=-1,
        sorted=True,
    ).indices
    probabilities = F.softmax(logits.gather(-1, selected), dim=-1)
    frames: list[ExpertFrame] = []
    counts: list[int] = []
    for expert in range(state.c_fc.shape[0]):
        locations = (selected == expert).nonzero(as_tuple=False)
        if locations.shape[0] == 0:
            raise RuntimeError(f"expert {expert} has no routed assignments")
        token = locations[:, 0]
        slot = locations[:, 1]
        hidden = F.gelu(x.index_select(0, token) @ state.c_fc[expert].T)
        frames.append(
            ExpertFrame(
                tokens=token,
                probabilities=probabilities[token, slot],
                hidden=hidden,
            )
        )
        counts.append(int(token.numel()))
    return frames, counts


def cproj_target_action(
    frames: list[ExpertFrame],
    left_c_proj: torch.Tensor,
    right_c_proj: torch.Tensor,
    tokens: int,
    output_width: int,
    device: str,
) -> torch.Tensor:
    output = torch.zeros(tokens, output_width, device=device)
    delta = right_c_proj.to(device=device).float() - left_c_proj.to(device=device).float()
    for expert, frame in enumerate(frames):
        action = frame.hidden @ delta[expert].T
        output.index_add_(
            0,
            frame.tokens,
            action * frame.probabilities[:, None],
        )
    return output


class ContextModulatedCProjOperator:
    """Linear-in-coordinate, activation-conditioned c_proj residual operator."""

    def __init__(
        self,
        frames: list[ExpertFrame],
        token_count: int,
        output_width: int,
        factors: int,
        beta: float,
        seed: int,
        layer: int,
        device: str,
    ) -> None:
        if not frames or factors <= 0 or output_width <= 0:
            raise ValueError("context operator dimensions must be positive")
        self.frames = frames
        self.token_count = int(token_count)
        self.output_width = int(output_width)
        self.factors = int(factors)
        self.beta = float(beta)
        self.device = device
        self.experts = len(frames)
        self.hidden_width = int(frames[0].hidden.shape[-1])
        self.padded_width = next_power_of_two(
            max(self.hidden_width, self.output_width)
        )
        reference = frames[0].hidden
        self.hidden_padded: list[torch.Tensor] = []
        self.gates: list[torch.Tensor] = []
        self.input_signs: list[list[torch.Tensor]] = []
        self.output_signs: list[list[torch.Tensor]] = []
        for expert, frame in enumerate(frames):
            if frame.hidden.shape[-1] != self.hidden_width:
                raise ValueError("expert hidden widths disagree")
            hidden = F.pad(frame.hidden.float(), (0, self.padded_width - self.hidden_width))
            self.hidden_padded.append(hidden)
            base_seed = int(seed) + 1009 * int(layer) + 131 * expert
            context_sign = signs_for(
                reference, expert, 0, base_seed, self.padded_width
            )
            if self.beta == 0.0:
                gate = torch.ones_like(hidden)
            else:
                context = normalized_fht_last_dim(hidden * context_sign)
                context = context / context.square().mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
                gate = (1.0 + self.beta * context) / math.sqrt(1.0 + self.beta * self.beta)
            self.gates.append(gate)
            self.input_signs.append(
                [
                    signs_for(reference, expert, 1 + k, base_seed, self.padded_width)
                    for k in range(self.factors)
                ]
            )
            self.output_signs.append(
                [
                    signs_for(reference, expert, 17 + k, base_seed, self.padded_width)
                    for k in range(self.factors)
                ]
            )

    @property
    def coordinate_shape(self) -> tuple[int, int, int]:
        return self.experts, self.factors, self.hidden_width

    def apply(self, coordinates: torch.Tensor) -> torch.Tensor:
        if tuple(coordinates.shape) != self.coordinate_shape:
            raise ValueError("coordinate shape disagrees with context operator")
        output = torch.zeros(
            self.token_count,
            self.output_width,
            device=self.device,
            dtype=torch.float32,
        )
        coordinates = coordinates.to(device=self.device, dtype=torch.float32)
        for expert, frame in enumerate(self.frames):
            hidden = self.hidden_padded[expert]
            gate = self.gates[expert]
            expert_output = torch.zeros(
                hidden.shape[0], self.output_width, device=self.device
            )
            for factor in range(self.factors):
                coordinate = F.pad(
                    coordinates[expert, factor],
                    (0, self.padded_width - self.hidden_width),
                )
                mixed = normalized_fht_last_dim(
                    hidden * coordinate * self.input_signs[expert][factor]
                )
                mixed = normalized_fht_last_dim(
                    mixed * gate * self.output_signs[expert][factor]
                )
                expert_output.add_(mixed[:, : self.output_width])
            output.index_add_(
                0,
                frame.tokens,
                expert_output * frame.probabilities[:, None],
            )
        return output

    def adjoint(self, output_cotangent: torch.Tensor) -> torch.Tensor:
        if tuple(output_cotangent.shape) != (self.token_count, self.output_width):
            raise ValueError("cotangent shape disagrees with context operator")
        cotangent = output_cotangent.to(device=self.device, dtype=torch.float32)
        result = torch.zeros(self.coordinate_shape, device=self.device)
        for expert, frame in enumerate(self.frames):
            selected = cotangent.index_select(0, frame.tokens)
            selected = selected * frame.probabilities[:, None]
            selected = F.pad(
                selected, (0, self.padded_width - self.output_width)
            )
            after_output = normalized_fht_last_dim(selected)
            hidden = self.hidden_padded[expert]
            gate = self.gates[expert]
            for factor in range(self.factors):
                after_gate = (
                    after_output * gate * self.output_signs[expert][factor]
                )
                after_input = normalized_fht_last_dim(after_gate)
                gradient = (
                    after_input * hidden * self.input_signs[expert][factor]
                ).sum(dim=0)
                result[expert, factor] = gradient[: self.hidden_width]
        return result


def coordinate_dot(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left.float() * right.float()).sum()


def action_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = left.float().norm() * right.float().norm()
    return float((left.float() * right.float()).sum() / denominator.clamp_min(1e-30))


def estimate_average_normal_diagonal(
    operator: ContextModulatedCProjOperator,
    probe_seed: int,
    probes: int,
) -> float:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(probe_seed))
    estimates = []
    for _ in range(probes):
        probe = torch.randint(
            0,
            2,
            operator.coordinate_shape,
            generator=generator,
            dtype=torch.int8,
        ).float()
        probe = (2.0 * probe - 1.0).to(operator.device)
        estimates.append(float(operator.apply(probe).square().sum() / probe.numel()))
    return sum(estimates) / len(estimates)


def cgls_fit(
    operator: ContextModulatedCProjOperator,
    target: torch.Tensor,
    *,
    maximum_iterations: int,
    tolerance: float,
    ridge_ratio: float,
    probe_seed: int,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    rhs = operator.adjoint(target)
    average_diagonal = estimate_average_normal_diagonal(
        operator, probe_seed, probes=2
    )
    ridge = float(ridge_ratio) * max(average_diagonal, 1e-30)
    solution = torch.zeros_like(rhs)
    residual = rhs.clone()
    search = residual.clone()
    residual_squared = coordinate_dot(residual, residual)
    initial_squared = residual_squared.clamp_min(1e-30)
    relative = 1.0
    iterations = 0
    for iteration in range(int(maximum_iterations)):
        normal = operator.adjoint(operator.apply(search)) + ridge * search
        denominator = coordinate_dot(search, normal).clamp_min(1e-30)
        step = residual_squared / denominator
        solution = solution + step * search
        updated = residual - step * normal
        updated_squared = coordinate_dot(updated, updated)
        relative = float(torch.sqrt(updated_squared / initial_squared))
        iterations = iteration + 1
        residual = updated
        if relative <= float(tolerance):
            residual_squared = updated_squared
            break
        beta = updated_squared / residual_squared.clamp_min(1e-30)
        search = residual + beta * search
        residual_squared = updated_squared
    return solution, {
        "iterations": iterations,
        "relative_normal_residual": relative,
        "average_normal_diagonal": average_diagonal,
        "ridge": ridge,
    }


def split_inputs(
    model: torch.nn.Module,
    plan: dict[str, Any],
    data_dir: Path,
    layers: list[int],
    device: str,
) -> dict[str, dict[int, torch.Tensor]]:
    result: dict[str, dict[int, torch.Tensor]] = {}
    for spec in plan["discovery_banks"]:
        batches = fixed_validation_batches(
            data_dir,
            int(spec["batch_size"]),
            int(spec["block_size"]),
            int(spec["batches"]),
            int(spec["seed"]),
        )
        result[spec["name"]] = collect_inputs(
            model, batches, layers, int(spec["tokens"]), device
        )
    evaluation = plan["evaluation"]
    heldout_batches = fixed_validation_batches(
        data_dir,
        int(evaluation["heldout_batch_size"]),
        int(evaluation["heldout_block_size"]),
        int(evaluation["heldout_batches"]),
        int(evaluation["heldout_seed"]),
    )
    result["heldout"] = collect_inputs(
        model, heldout_batches, layers, int(evaluation["heldout_tokens"]), device
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--terminal-snapshot", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("context-modulated oracle plan schema mismatch")
    target_spec = plan["target_and_gauge"]
    if file_sha256(args.terminal_snapshot) != target_spec["terminal_manifold_snapshot_sha256"]:
        raise ValueError("terminal snapshot hash disagrees with frozen plan")
    payload = load_terminal_snapshot(args.terminal_snapshot)
    layers = [int(value) for value in target_spec["layers"]]
    model = model_from_exact_stepzero(
        payload, int(target_spec["model_seed"]), args.device
    )
    stepzero_hashes = selected_stepzero_hashes(model, layers)
    initial_mapping = dict(model.named_parameters())
    initial = {
        layer: layer_state_from_mapping(initial_mapping, layer) for layer in layers
    }
    terminal = {
        layer: layer_state_from_mapping(payload["model"], layer) for layer in layers
    }
    inputs = split_inputs(model, plan, args.data_dir, layers, args.device)
    del model
    torch.cuda.empty_cache()

    evaluation = plan["evaluation"]
    fixed_seed = int(plan["mechanism"]["fixed_operator_seed"])
    rows: list[dict[str, Any]] = []
    coordinate_payload: dict[str, torch.Tensor] = {}
    heldout_actions: dict[tuple[str, str, int], torch.Tensor] = {}
    occupancies: dict[str, dict[str, Any]] = {}
    families = {
        spec["name"]: int(spec["factors"])
        for spec in plan["coordinate_families"]
    }
    for bank in [spec["name"] for spec in plan["discovery_banks"]]:
        occupancies[bank] = {}
        for layer in layers:
            discovery_x = inputs[bank][layer].to(args.device)
            heldout_x = inputs["heldout"][layer].to(args.device)
            discovery_frames, discovery_counts = routed_hidden_frames(
                LayerState(
                    terminal[layer].router,
                    terminal[layer].c_fc,
                    terminal[layer].c_proj,
                ),
                discovery_x,
                top_k=2,
                device=args.device,
            )
            heldout_frames, heldout_counts = routed_hidden_frames(
                LayerState(
                    terminal[layer].router,
                    terminal[layer].c_fc,
                    terminal[layer].c_proj,
                ),
                heldout_x,
                top_k=2,
                device=args.device,
            )
            occupancies[bank][str(layer)] = {
                "discovery": discovery_counts,
                "heldout": heldout_counts,
                "minimum_discovery": min(discovery_counts),
            }
            discovery_target = cproj_target_action(
                discovery_frames,
                initial[layer].c_proj,
                terminal[layer].c_proj,
                discovery_x.shape[0],
                discovery_x.shape[1],
                args.device,
            )
            heldout_target = cproj_target_action(
                heldout_frames,
                initial[layer].c_proj,
                terminal[layer].c_proj,
                heldout_x.shape[0],
                heldout_x.shape[1],
                args.device,
            )
            for family, factors in families.items():
                for mode, beta in (("dynamic", 1.0), ("beta_zero", 0.0)):
                    discovery_operator = ContextModulatedCProjOperator(
                        discovery_frames,
                        discovery_x.shape[0],
                        discovery_x.shape[1],
                        factors,
                        beta,
                        fixed_seed,
                        layer,
                        args.device,
                    )
                    coordinates, diagnostics = cgls_fit(
                        discovery_operator,
                        discovery_target,
                        maximum_iterations=int(evaluation["maximum_cgls_iterations"]),
                        tolerance=float(evaluation["relative_normal_residual_tolerance"]),
                        ridge_ratio=float(evaluation["ridge_ratio"]),
                        probe_seed=20260924 + 1009 * layer + factors,
                    )
                    discovery_action = discovery_operator.apply(coordinates)
                    heldout_operator = ContextModulatedCProjOperator(
                        heldout_frames,
                        heldout_x.shape[0],
                        heldout_x.shape[1],
                        factors,
                        beta,
                        fixed_seed,
                        layer,
                        args.device,
                    )
                    heldout_action = heldout_operator.apply(coordinates)
                    key = f"{bank}:{family}:{mode}:layer{layer}"
                    coordinate_payload[key] = coordinates.detach().cpu()
                    heldout_actions[(bank, family + ":" + mode, layer)] = (
                        heldout_action.detach().cpu()
                    )
                    rows.append(
                        {
                            "bank": bank,
                            "layer": layer,
                            "family": family,
                            "factors": factors,
                            "mode": mode,
                            "coordinates": coordinates.numel(),
                            "c_proj_compression_ratio": discovery_x.shape[1] / factors,
                            "minimum_expert_assignments": min(discovery_counts),
                            "discovery_recovery": recovery_fraction(
                                discovery_action, discovery_target
                            ),
                            "heldout_recovery": recovery_fraction(
                                heldout_action, heldout_target
                            ),
                            "coordinate_l2": float(coordinates.norm()),
                            **diagnostics,
                        }
                    )
                    del discovery_operator, heldout_operator

    summary: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    frozen = plan["frozen_gates"]
    bank_names = [spec["name"] for spec in plan["discovery_banks"]]
    for family in families:
        agreement_by_layer = {}
        for layer in layers:
            left = heldout_actions[(bank_names[0], family + ":dynamic", layer)]
            right = heldout_actions[(bank_names[1], family + ":dynamic", layer)]
            agreement_by_layer[str(layer)] = action_cosine(left, right)
        agreement_mean = sum(agreement_by_layer.values()) / len(layers)
        summary[family] = {
            "heldout_bank_action_cosine": {
                "mean": agreement_mean,
                "by_layer": agreement_by_layer,
            }
        }
        gates[family] = {}
        for bank in bank_names:
            dynamic = [
                row for row in rows
                if row["family"] == family and row["bank"] == bank
                and row["mode"] == "dynamic"
            ]
            static = {
                int(row["layer"]): row for row in rows
                if row["family"] == family and row["bank"] == bank
                and row["mode"] == "beta_zero"
            }
            heldout = [float(row["heldout_recovery"]) for row in dynamic]
            discovery = [float(row["discovery_recovery"]) for row in dynamic]
            gains = [
                float(row["heldout_recovery"])
                - float(static[int(row["layer"])]["heldout_recovery"])
                for row in dynamic
            ]
            drops = [fit - test for fit, test in zip(discovery, heldout)]
            minimum_occupancy = min(
                int(row["minimum_expert_assignments"]) for row in dynamic
            )
            bank_summary = {
                "heldout_mean": sum(heldout) / len(heldout),
                "heldout_minimum": min(heldout),
                "heldout_by_layer": {
                    str(row["layer"]): float(row["heldout_recovery"])
                    for row in dynamic
                },
                "discovery_mean": sum(discovery) / len(discovery),
                "dynamic_minus_beta_zero_mean": sum(gains) / len(gains),
                "maximum_discovery_to_heldout_drop": max(drops),
                "minimum_expert_assignments": minimum_occupancy,
                "maximum_relative_normal_residual": max(
                    float(row["relative_normal_residual"]) for row in dynamic
                ),
            }
            summary[family][bank] = bank_summary
            bank_gates = {
                "mean_pass": bank_summary["heldout_mean"]
                >= float(frozen["heldout_exact_recovery_mean_min_each_bank"]),
                "every_layer_pass": bank_summary["heldout_minimum"]
                >= float(frozen["heldout_exact_recovery_every_layer_min_each_bank"]),
                "dynamic_gain_pass": bank_summary["dynamic_minus_beta_zero_mean"]
                >= float(frozen["dynamic_minus_beta_zero_mean_min_each_bank"]),
                "generalization_pass": bank_summary["maximum_discovery_to_heldout_drop"]
                <= float(frozen["discovery_to_heldout_drop_max_each_layer"]),
                "action_agreement_pass": agreement_mean
                >= float(frozen["heldout_bank_action_cosine_mean_min"]),
                "occupancy_pass": minimum_occupancy
                >= int(frozen["minimum_expert_assignments"]),
            }
            bank_gates["all_pass"] = all(bank_gates.values())
            gates[family][bank] = bank_gates

    finite = all_finite({"rows": rows, "summary": summary})
    passing = [
        family for family in families
        if finite and all(gates[family][bank]["all_pass"] for bank in bank_names)
    ]
    if "two_factor" in passing:
        decision = "PASS_384X_AUTHORIZE_MULTIPHASE_SHADOW_ONLY"
    elif "three_factor" in passing:
        decision = "PASS_256X_AUTHORIZE_MULTIPHASE_SHADOW_ONLY"
    else:
        decision = "REJECT_CONTEXT_MODULATED_FHT_AT_TESTED_BUDGET"

    args.output.mkdir(parents=True, exist_ok=True)
    rows_path = args.output / "context_modulated_oracle_rows.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=union_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)
    coordinates_path = args.output / "context_modulated_coordinates.pt"
    torch.save(coordinate_payload, coordinates_path)
    result = {
        "schema_version": "nanogpt_sparse_moe_cproj_context_modulated_fht_oracle_result_v1",
        "decision": decision,
        "passing_families": passing,
        "all_values_finite": finite,
        "summary": summary,
        "gates": gates,
        "expert_occupancy": occupancies,
        "stepzero_selected_tensor_sha256": stepzero_hashes,
        "source": {
            "terminal_snapshot_sha256": file_sha256(args.terminal_snapshot),
            "plan_sha256": file_sha256(args.plan),
            "dataset_manifest_sha256": target_spec["dataset_manifest_sha256"],
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
    result_path = args.output / "context_modulated_oracle_result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    status = {
        "state": "finished",
        "exit_code": 0,
        "decision": decision,
        "result_sha256": file_sha256(result_path),
        "rows_sha256": file_sha256(rows_path),
        "coordinates_sha256": file_sha256(coordinates_path),
        "wall_seconds": time.time() - started,
    }
    status_path = args.output / "context_modulated_oracle_status.json"
    status_path.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": status, "summary": summary}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
