#!/usr/bin/env python3
"""Gate a full-rank multilinear Kronecker latent map for sparse-MoE c_proj.

This is an optimistic, zero-update representability oracle.  The exact
step-zero c_proj is retained as a fixed base and its terminal displacement is
represented as a sum of Kronecker products.  Both Kronecker factors are the
compact latent state; there is no learned dense decoder or low-rank matrix
adapter.  Terminal routed activations define the functional fit and untouched
tokens define the score.
"""
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

from examples.nanogpt.analyze_cproj_manifold import load_model
from examples.nanogpt.analyze_mlp_activation_update_alignment import git_commit
from examples.nanogpt.analyze_residual_compatibility import fixed_validation_batches
from examples.nanogpt.analyze_sparse_moe_cproj_context_modulated_fht_oracle import (
    action_cosine,
    cproj_target_action,
    routed_hidden_frames,
)
from examples.nanogpt.analyze_sparse_moe_paired_alignment import (
    collect_inputs,
    file_sha256,
)
from examples.nanogpt.analyze_sparse_moe_paired_atom_oracle import union_fieldnames
from examples.nanogpt.analyze_sparse_moe_rolling_tangent_oracle import (
    LayerState,
    recovery_fraction,
)
from examples.nanogpt.analyze_sparse_moe_stepzero_task_gradient_oracle import (
    layer_state_from_mapping,
    load_terminal_snapshot,
    model_from_exact_stepzero,
    selected_stepzero_hashes,
)


PLAN_SCHEMA = "nanogpt_sparse_moe_cproj_kronecker_oracle_plan_v1"


@dataclass(frozen=True)
class KroneckerShape:
    output_groups: int
    output_channels: int
    input_groups: int
    input_channels: int

    @property
    def output_width(self) -> int:
        return self.output_groups * self.output_channels

    @property
    def input_width(self) -> int:
        return self.input_groups * self.input_channels

    def coordinates_per_expert(self, rank: int) -> int:
        return int(rank) * (
            self.output_groups * self.input_groups
            + self.output_channels * self.input_channels
        )


def rearrange_for_kronecker(delta: torch.Tensor, shape: KroneckerShape) -> torch.Tensor:
    if delta.ndim != 3:
        raise ValueError("expected [experts, output, input] delta")
    if tuple(delta.shape[1:]) != (shape.output_width, shape.input_width):
        raise ValueError("delta does not match the registered Kronecker shape")
    return (
        delta.reshape(
            delta.shape[0],
            shape.output_groups,
            shape.output_channels,
            shape.input_groups,
            shape.input_channels,
        )
        .permute(0, 1, 3, 2, 4)
        .reshape(
            delta.shape[0],
            shape.output_groups * shape.input_groups,
            shape.output_channels * shape.input_channels,
        )
    )


def materialize_kronecker(
    group_factors: torch.Tensor,
    channel_factors: torch.Tensor,
) -> torch.Tensor:
    if group_factors.ndim != 4 or channel_factors.ndim != 4:
        raise ValueError("Kronecker factors must have [expert, rank, rows, cols]")
    if group_factors.shape[:2] != channel_factors.shape[:2]:
        raise ValueError("expert and rank dimensions must agree")
    # W[(i,j),(k,l)] = sum_r A[r,i,k] B[r,j,l].
    return torch.einsum(
        "erik,erjl->eijkl", group_factors, channel_factors
    ).reshape(
        group_factors.shape[0],
        group_factors.shape[2] * channel_factors.shape[2],
        group_factors.shape[3] * channel_factors.shape[3],
    )


def truncated_kronecker_svd(
    delta: torch.Tensor,
    shape: KroneckerShape,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rearranged = rearrange_for_kronecker(delta.float(), shape)
    u, singular, vh = torch.linalg.svd(rearranged, full_matrices=False)
    root = singular[:, :rank].clamp_min(0).sqrt()
    group = (u[:, :, :rank] * root[:, None, :]).permute(0, 2, 1)
    channel = root[:, :, None] * vh[:, :rank, :]
    return (
        group.reshape(
            delta.shape[0], rank, shape.output_groups, shape.input_groups
        ),
        channel.reshape(
            delta.shape[0], rank, shape.output_channels, shape.input_channels
        ),
    )


def parameter_recovery(predicted: torch.Tensor, target: torch.Tensor) -> float:
    denominator = target.float().square().sum().clamp_min(1e-30)
    return float(
        1.0 - (predicted.float() - target.float()).square().sum() / denominator
    )


def apply_delta(frames: list[Any], delta: torch.Tensor, token_count: int) -> torch.Tensor:
    output = torch.zeros(
        token_count,
        delta.shape[1],
        device=delta.device,
        dtype=torch.float32,
    )
    for expert, frame in enumerate(frames):
        action = frame.hidden.float() @ delta[expert].float().T
        output.index_add_(
            0,
            frame.tokens,
            action * frame.probabilities.float()[:, None],
        )
    return output


def fit_functional_factors(
    frames: list[Any],
    target: torch.Tensor,
    initial_group: torch.Tensor,
    initial_channel: torch.Tensor,
    *,
    steps: int,
    learning_rate: float,
    gradient_clip: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    group = torch.nn.Parameter(initial_group.detach().float().clone())
    channel = torch.nn.Parameter(initial_channel.detach().float().clone())
    optimizer = torch.optim.Adam((group, channel), lr=float(learning_rate))
    denominator = target.float().square().sum().detach().clamp_min(1e-30)
    best_loss = math.inf
    best_group = group.detach().clone()
    best_channel = channel.detach().clone()
    initial_loss = math.nan
    gradient_maximum = 0.0
    for step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        delta = materialize_kronecker(group, channel)
        prediction = apply_delta(frames, delta, target.shape[0])
        loss = (prediction - target.float()).square().sum() / denominator
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite functional Kronecker objective")
        if step == 0:
            initial_loss = float(loss.detach())
        loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_((group, channel), float(gradient_clip))
        )
        gradient_maximum = max(gradient_maximum, gradient_norm)
        optimizer.step()
        current = float(loss.detach())
        if current < best_loss:
            best_loss = current
            best_group = group.detach().clone()
            best_channel = channel.detach().clone()
    with torch.no_grad():
        final_delta = materialize_kronecker(group, channel)
        final_prediction = apply_delta(frames, final_delta, target.shape[0])
        final_loss = float(
            (final_prediction - target.float()).square().sum() / denominator
        )
        best_delta = materialize_kronecker(best_group, best_channel)
        best_prediction = apply_delta(frames, best_delta, target.shape[0])
        best_loss_recomputed = float(
            (best_prediction - target.float()).square().sum() / denominator
        )
    return best_group, best_channel, {
        "steps": int(steps),
        "initial_relative_error_squared": initial_loss,
        "final_relative_error_squared": final_loss,
        "best_relative_error_squared": best_loss_recomputed,
        "maximum_preclip_gradient_norm": gradient_maximum,
    }


def fixed_inputs(
    model: torch.nn.Module,
    plan: dict[str, Any],
    data_dir: Path,
    layers: list[int],
    device: str,
) -> dict[str, dict[int, torch.Tensor]]:
    result: dict[str, dict[int, torch.Tensor]] = {}
    protocol = plan["functional_protocol"]
    for spec in protocol["discovery_banks"]:
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
    heldout = protocol["heldout"]
    batches = fixed_validation_batches(
        data_dir,
        int(heldout["batch_size"]),
        int(heldout["block_size"]),
        int(heldout["batches"]),
        int(heldout["seed"]),
    )
    result["heldout"] = collect_inputs(
        model, batches, layers, int(heldout["tokens"]), device
    )
    return result


def parent_static_means(parent: dict[str, Any]) -> dict[str, float]:
    result = parent["results"]["three_factor_256x"]
    dynamic = result["dynamic_heldout_mean_a_b"]
    difference = result["dynamic_minus_static_a_b"]
    return {
        "discovery_a": float(dynamic[0]) - float(difference[0]),
        "discovery_b": float(dynamic[1]) - float(difference[1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--terminal-snapshot", required=True, type=Path)
    parser.add_argument("--parent-result", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("Kronecker oracle plan schema mismatch")
    source = plan["source"]
    if file_sha256(args.terminal_snapshot) != source["terminal_snapshot_sha256"]:
        raise ValueError("terminal snapshot hash disagrees with frozen plan")
    if file_sha256(args.parent_result) != source["parent_result_sha256"]:
        raise ValueError("parent result hash disagrees with frozen plan")
    manifest = args.data_dir / "manifest.json"
    if file_sha256(manifest) != source["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest hash disagrees with frozen plan")

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

    mechanism = plan["mechanism"]
    shape = KroneckerShape(
        int(mechanism["output_groups"]),
        int(mechanism["output_channels"]),
        int(mechanism["input_groups"]),
        int(mechanism["input_channels"]),
    )
    rank = int(mechanism["kronecker_rank"])
    coordinates = shape.coordinates_per_expert(rank)
    compression = shape.output_width * shape.input_width / coordinates
    if abs(compression - float(mechanism["coordinate_compression_ratio"])) > 1e-9:
        raise ValueError("registered coordinate compression ratio is incorrect")
    parent = json.loads(args.parent_result.read_text(encoding="utf-8"))
    static_means = parent_static_means(parent)
    protocol = plan["functional_protocol"]
    fit = plan["fit"]
    rows: list[dict[str, Any]] = []
    factors: dict[str, torch.Tensor] = {}
    heldout_actions: dict[tuple[str, int], torch.Tensor] = {}
    occupancy: dict[str, dict[str, list[int]]] = {}

    for layer in layers:
        delta = (
            terminal[layer].c_proj.to(args.device).float()
            - initial[layer].c_proj.to(args.device).float()
        )
        initial_group, initial_channel = truncated_kronecker_svd(
            delta, shape, rank
        )
        svd_delta = materialize_kronecker(initial_group, initial_channel)
        heldout_x = inputs["heldout"][layer].to(args.device)
        heldout_frames, heldout_counts = routed_hidden_frames(
            terminal[layer], heldout_x, int(protocol["top_k"]), args.device
        )
        heldout_target = cproj_target_action(
            heldout_frames,
            initial[layer].c_proj,
            terminal[layer].c_proj,
            heldout_x.shape[0],
            heldout_x.shape[1],
            args.device,
        )
        svd_heldout = apply_delta(heldout_frames, svd_delta, heldout_x.shape[0])
        for bank_spec in protocol["discovery_banks"]:
            bank = bank_spec["name"]
            discovery_x = inputs[bank][layer].to(args.device)
            discovery_frames, discovery_counts = routed_hidden_frames(
                terminal[layer], discovery_x, int(protocol["top_k"]), args.device
            )
            occupancy.setdefault(bank, {})[str(layer)] = discovery_counts
            discovery_target = cproj_target_action(
                discovery_frames,
                initial[layer].c_proj,
                terminal[layer].c_proj,
                discovery_x.shape[0],
                discovery_x.shape[1],
                args.device,
            )
            group, channel, diagnostics = fit_functional_factors(
                discovery_frames,
                discovery_target,
                initial_group,
                initial_channel,
                steps=int(fit["adam_steps"]),
                learning_rate=float(fit["learning_rate"]),
                gradient_clip=float(fit["gradient_clip"]),
            )
            fitted_delta = materialize_kronecker(group, channel)
            discovery_action = apply_delta(
                discovery_frames, fitted_delta, discovery_x.shape[0]
            )
            heldout_action = apply_delta(
                heldout_frames, fitted_delta, heldout_x.shape[0]
            )
            heldout_actions[(bank, layer)] = heldout_action.detach().cpu()
            factors[f"{bank}:layer{layer}:group"] = group.detach().cpu()
            factors[f"{bank}:layer{layer}:channel"] = channel.detach().cpu()
            rows.append(
                {
                    "bank": bank,
                    "layer": layer,
                    "coordinates_per_expert": coordinates,
                    "compression_ratio": compression,
                    "minimum_discovery_assignments": min(discovery_counts),
                    "minimum_heldout_assignments": min(heldout_counts),
                    "svd_parameter_recovery": parameter_recovery(svd_delta, delta),
                    "fitted_parameter_recovery": parameter_recovery(fitted_delta, delta),
                    "svd_heldout_recovery": recovery_fraction(
                        svd_heldout, heldout_target
                    ),
                    "discovery_recovery": recovery_fraction(
                        discovery_action, discovery_target
                    ),
                    "heldout_recovery": recovery_fraction(
                        heldout_action, heldout_target
                    ),
                    **diagnostics,
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
    for bank in banks:
        selected = [row for row in rows if row["bank"] == bank]
        summaries[bank] = {
            "heldout_recovery_mean": sum(float(row["heldout_recovery"]) for row in selected)
            / len(selected),
            "heldout_recovery_minimum": min(float(row["heldout_recovery"]) for row in selected),
            "svd_heldout_recovery_mean": sum(float(row["svd_heldout_recovery"]) for row in selected)
            / len(selected),
            "fitted_parameter_recovery_mean": sum(float(row["fitted_parameter_recovery"]) for row in selected)
            / len(selected),
            "heldout_bank_action_cosine_mean": sum(
                float(row["heldout_bank_action_cosine"]) for row in selected
            )
            / len(selected),
            "minimum_discovery_assignments": min(
                int(row["minimum_discovery_assignments"]) for row in selected
            ),
            "improvement_over_parent_static_256x": (
                sum(float(row["heldout_recovery"]) for row in selected) / len(selected)
                - static_means[bank]
            ),
            "all_fits_non_degrading": all(
                float(row["best_relative_error_squared"])
                <= float(row["initial_relative_error_squared"]) + 1e-8
                for row in selected
            ),
        }
    gates = plan["frozen_gates"]
    gate_results: dict[str, Any] = {}
    for bank in banks:
        summary = summaries[bank]
        gate_results[bank] = {
            "mean_recovery": summary["heldout_recovery_mean"]
            >= float(gates["heldout_recovery_mean_min"]),
            "minimum_layer_recovery": summary["heldout_recovery_minimum"]
            >= float(gates["heldout_recovery_every_layer_min"]),
            "improvement_over_static": summary["improvement_over_parent_static_256x"]
            >= float(gates["improvement_over_parent_static_min"]),
            "bank_action_cosine": summary["heldout_bank_action_cosine_mean"]
            >= float(gates["heldout_bank_action_cosine_mean_min"]),
            "occupancy": summary["minimum_discovery_assignments"]
            >= int(gates["minimum_expert_assignments"]),
            "fit_non_degrading": bool(summary["all_fits_non_degrading"]),
        }
        gate_results[bank]["all_pass"] = all(gate_results[bank].values())
    finite = all(
        math.isfinite(float(row[key]))
        for row in rows
        for key in (
            "svd_parameter_recovery",
            "fitted_parameter_recovery",
            "svd_heldout_recovery",
            "discovery_recovery",
            "heldout_recovery",
            "heldout_bank_action_cosine",
            "initial_relative_error_squared",
            "best_relative_error_squared",
        )
    )
    passed = finite and all(value["all_pass"] for value in gate_results.values())
    decision = (
        "PASS_KRONECKER_REPRESENTABILITY_UPPER_BOUND"
        if passed
        else "REJECT_RANK2_KRONECKER_CPROJ_AT_TESTED_LAYOUT_AND_BUDGET"
    )

    args.output.mkdir(parents=True, exist_ok=False)
    rows_path = args.output / "kronecker_oracle_rows.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=union_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)
    factors_path = args.output / "kronecker_factors.pt"
    torch.save(factors, factors_path)
    result = {
        "schema_version": "nanogpt_sparse_moe_cproj_kronecker_oracle_result_v1",
        "decision": decision,
        "all_values_finite": finite,
        "summary": summaries,
        "gates": gate_results,
        "mechanism": {
            **mechanism,
            "coordinates_per_expert": coordinates,
            "coordinate_compression_ratio": compression,
            "ordinary_matrix_rank_ceiling": min(shape.output_width, shape.input_width),
        },
        "expert_occupancy": occupancy,
        "stepzero_selected_tensor_sha256": stepzero_hashes,
        "source": {
            "terminal_snapshot_sha256": file_sha256(args.terminal_snapshot),
            "parent_result_sha256": file_sha256(args.parent_result),
            "plan_sha256": file_sha256(args.plan),
            "dataset_manifest_sha256": file_sha256(manifest),
        },
        "authorization": {
            "training": False,
            "mfu_preflight": False,
            "generated_experts": False,
            "larger_rung": False,
            "causal_multiphase_shadow_oracle": bool(passed),
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
    result_path = args.output / "kronecker_oracle_result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    status = {
        "state": "finished",
        "exit_code": 0,
        "decision": decision,
        "result_sha256": file_sha256(result_path),
        "rows_sha256": file_sha256(rows_path),
        "factors_sha256": file_sha256(factors_path),
        "wall_seconds": time.time() - started,
    }
    status_path = args.output / "kronecker_oracle_status.json"
    status_path.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": status, "summary": summaries}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
