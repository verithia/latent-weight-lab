#!/usr/bin/env python3
"""Gate an equal-budget tensor-train/MPO latent map for sparse-MoE c_proj."""
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
    parent_static_means,
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


PLAN_SCHEMA = "nanogpt_sparse_moe_cproj_mpo_oracle_plan_v1"


def coordinate_count(output_modes: list[int], input_modes: list[int], rank: int) -> int:
    if len(output_modes) != len(input_modes) or not output_modes:
        raise ValueError("MPO mode lists must be nonempty and equal length")
    bonds = [1] + [int(rank)] * (len(output_modes) - 1) + [1]
    return sum(
        bonds[index]
        * int(output_modes[index])
        * int(input_modes[index])
        * bonds[index + 1]
        for index in range(len(output_modes))
    )


def matrix_to_physical(
    matrix: torch.Tensor,
    output_modes: list[int],
    input_modes: list[int],
) -> torch.Tensor:
    experts, output_width, input_width = matrix.shape
    if math.prod(output_modes) != output_width or math.prod(input_modes) != input_width:
        raise ValueError("MPO modes do not match matrix dimensions")
    modes = len(output_modes)
    reshaped = matrix.reshape(experts, *output_modes, *input_modes)
    permutation = [0]
    for index in range(modes):
        permutation.extend((1 + index, 1 + modes + index))
    return reshaped.permute(permutation).reshape(
        experts, *(output_modes[index] * input_modes[index] for index in range(modes))
    )


def physical_to_matrix(
    physical: torch.Tensor,
    output_modes: list[int],
    input_modes: list[int],
) -> torch.Tensor:
    experts = physical.shape[0]
    interleaved: list[int] = []
    for output, input_ in zip(output_modes, input_modes):
        interleaved.extend((int(output), int(input_)))
    expanded = physical.reshape(experts, *interleaved)
    output_positions = [1 + 2 * index for index in range(len(output_modes))]
    input_positions = [2 + 2 * index for index in range(len(input_modes))]
    return expanded.permute([0, *output_positions, *input_positions]).reshape(
        experts, math.prod(output_modes), math.prod(input_modes)
    )


def truncated_mpo_svd(
    matrix: torch.Tensor,
    output_modes: list[int],
    input_modes: list[int],
    rank: int,
) -> list[torch.Tensor]:
    physical = matrix_to_physical(matrix.float(), output_modes, input_modes)
    physical_modes = list(physical.shape[1:])
    per_expert: list[list[torch.Tensor]] = []
    for expert in range(matrix.shape[0]):
        remainder = physical[expert]
        left_rank = 1
        cores: list[torch.Tensor] = []
        for index, physical_width in enumerate(physical_modes[:-1]):
            flattened = remainder.reshape(left_rank * physical_width, -1)
            u, singular, vh = torch.linalg.svd(flattened, full_matrices=False)
            next_rank = min(int(rank), u.shape[1])
            core = u[:, :next_rank].reshape(
                left_rank,
                int(output_modes[index]),
                int(input_modes[index]),
                next_rank,
            )
            cores.append(core)
            remainder = (singular[:next_rank, None] * vh[:next_rank]).reshape(
                next_rank, *physical_modes[index + 1 :]
            )
            left_rank = next_rank
        cores.append(
            remainder.reshape(
                left_rank,
                int(output_modes[-1]),
                int(input_modes[-1]),
                1,
            )
        )
        per_expert.append(cores)
    return [
        torch.stack([per_expert[expert][index] for expert in range(matrix.shape[0])])
        for index in range(len(output_modes))
    ]


def materialize_mpo(
    cores: list[torch.Tensor],
    output_modes: list[int],
    input_modes: list[int],
) -> torch.Tensor:
    if len(cores) != len(output_modes) or len(cores) != len(input_modes):
        raise ValueError("MPO core count and modes disagree")
    experts = cores[0].shape[0]
    first = cores[0]
    if first.shape[1] != 1:
        raise ValueError("first MPO left bond must be one")
    state = first[:, 0].reshape(experts, output_modes[0] * input_modes[0], first.shape[-1])
    for index, core in enumerate(cores[1:], start=1):
        if state.shape[-1] != core.shape[1]:
            raise ValueError("adjacent MPO bonds disagree")
        flattened = core.reshape(
            experts,
            core.shape[1],
            output_modes[index] * input_modes[index],
            core.shape[-1],
        )
        state = torch.einsum("epr,erqs->epqs", state, flattened).reshape(
            experts, -1, core.shape[-1]
        )
    if state.shape[-1] != 1:
        raise ValueError("last MPO right bond must be one")
    physical = state[..., 0].reshape(
        experts, *(output_modes[index] * input_modes[index] for index in range(len(cores)))
    )
    return physical_to_matrix(physical, output_modes, input_modes)


def fit_functional_cores(
    frames: list[Any],
    target: torch.Tensor,
    initial_cores: list[torch.Tensor],
    output_modes: list[int],
    input_modes: list[int],
    *,
    steps: int,
    learning_rate: float,
    gradient_clip: float,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    parameters = [torch.nn.Parameter(core.detach().float().clone()) for core in initial_cores]
    optimizer = torch.optim.Adam(parameters, lr=float(learning_rate))
    denominator = target.float().square().sum().detach().clamp_min(1e-30)
    best_loss = math.inf
    best = [parameter.detach().clone() for parameter in parameters]
    initial_loss = math.nan
    maximum_gradient = 0.0
    for step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        delta = materialize_mpo(parameters, output_modes, input_modes)
        prediction = apply_delta(frames, delta, target.shape[0])
        loss = (prediction - target.float()).square().sum() / denominator
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite functional MPO objective")
        if step == 0:
            initial_loss = float(loss.detach())
        loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(parameters, float(gradient_clip)))
        maximum_gradient = max(maximum_gradient, gradient)
        optimizer.step()
        if float(loss.detach()) < best_loss:
            best_loss = float(loss.detach())
            best = [parameter.detach().clone() for parameter in parameters]
    with torch.no_grad():
        best_delta = materialize_mpo(best, output_modes, input_modes)
        best_prediction = apply_delta(frames, best_delta, target.shape[0])
        best_recomputed = float(
            (best_prediction - target.float()).square().sum() / denominator
        )
    return best, {
        "steps": int(steps),
        "initial_relative_error_squared": initial_loss,
        "best_relative_error_squared": best_recomputed,
        "maximum_preclip_gradient_norm": maximum_gradient,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--terminal-snapshot", required=True, type=Path)
    parser.add_argument("--parent-result", required=True, type=Path)
    parser.add_argument("--kronecker-seal", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    plan = json.loads(args.plan.read_text())
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("MPO oracle plan schema mismatch")
    source = plan["source"]
    for path, expected, label in (
        (args.terminal_snapshot, source["terminal_snapshot_sha256"], "terminal snapshot"),
        (args.parent_result, source["parent_result_sha256"], "parent result"),
        (args.kronecker_seal, source["kronecker_seal_sha256"], "Kronecker seal"),
        (args.data_dir / "manifest.json", source["dataset_manifest_sha256"], "dataset manifest"),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"{label} hash disagrees with frozen plan")

    payload = load_terminal_snapshot(args.terminal_snapshot)
    layers = [int(value) for value in source["layers"]]
    stepzero_model = model_from_exact_stepzero(payload, int(source["model_seed"]), args.device)
    stepzero_hashes = selected_stepzero_hashes(stepzero_model, layers)
    initial_mapping = dict(stepzero_model.named_parameters())
    initial = {layer: layer_state_from_mapping(initial_mapping, layer) for layer in layers}
    terminal = {layer: layer_state_from_mapping(payload["model"], layer) for layer in layers}
    del initial_mapping, stepzero_model
    torch.cuda.empty_cache()
    terminal_model = load_model(args.terminal_snapshot, args.device)
    inputs = fixed_inputs(terminal_model, plan, args.data_dir, layers, args.device)
    del terminal_model
    torch.cuda.empty_cache()

    mechanism = plan["mechanism"]
    output_modes = [int(value) for value in mechanism["output_modes"]]
    input_modes = [int(value) for value in mechanism["input_modes"]]
    rank = int(mechanism["bond_rank"])
    coordinates = coordinate_count(output_modes, input_modes, rank)
    compression = math.prod(output_modes) * math.prod(input_modes) / coordinates
    if coordinates != int(mechanism["coordinates_per_expert"]):
        raise ValueError("registered MPO coordinate count is incorrect")
    if abs(compression - float(mechanism["coordinate_compression_ratio"])) > 1e-9:
        raise ValueError("registered MPO compression is incorrect")
    parent = json.loads(args.parent_result.read_text())
    static_means = parent_static_means(parent)
    kronecker = json.loads(args.kronecker_seal.read_text())
    kronecker_means = {
        bank: float(kronecker["results"][bank]["heldout_recovery_mean"])
        for bank in ("discovery_a", "discovery_b")
    }
    protocol = plan["functional_protocol"]
    fit = plan["fit"]
    rows: list[dict[str, Any]] = []
    saved_cores: dict[str, torch.Tensor] = {}
    heldout_actions: dict[tuple[str, int], torch.Tensor] = {}

    for layer in layers:
        delta = terminal[layer].c_proj.to(args.device).float() - initial[layer].c_proj.to(args.device).float()
        initial_cores = truncated_mpo_svd(delta, output_modes, input_modes, rank)
        svd_delta = materialize_mpo(initial_cores, output_modes, input_modes)
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
            discovery_target = cproj_target_action(
                discovery_frames,
                initial[layer].c_proj,
                terminal[layer].c_proj,
                discovery_x.shape[0],
                discovery_x.shape[1],
                args.device,
            )
            cores, diagnostics = fit_functional_cores(
                discovery_frames,
                discovery_target,
                initial_cores,
                output_modes,
                input_modes,
                steps=int(fit["adam_steps"]),
                learning_rate=float(fit["learning_rate"]),
                gradient_clip=float(fit["gradient_clip"]),
            )
            fitted_delta = materialize_mpo(cores, output_modes, input_modes)
            discovery_action = apply_delta(discovery_frames, fitted_delta, discovery_x.shape[0])
            heldout_action = apply_delta(heldout_frames, fitted_delta, heldout_x.shape[0])
            heldout_actions[(bank, layer)] = heldout_action.detach().cpu()
            for index, core in enumerate(cores):
                saved_cores[f"{bank}:layer{layer}:core{index}"] = core.detach().cpu()
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
                    "svd_heldout_recovery": recovery_fraction(svd_heldout, heldout_target),
                    "discovery_recovery": recovery_fraction(discovery_action, discovery_target),
                    "heldout_recovery": recovery_fraction(heldout_action, heldout_target),
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
        mean = sum(float(row["heldout_recovery"]) for row in selected) / len(selected)
        summaries[bank] = {
            "heldout_recovery_mean": mean,
            "heldout_recovery_minimum": min(float(row["heldout_recovery"]) for row in selected),
            "heldout_recovery_by_layer": [float(row["heldout_recovery"]) for row in selected],
            "svd_heldout_recovery_mean": sum(float(row["svd_heldout_recovery"]) for row in selected) / len(selected),
            "fitted_parameter_recovery_mean": sum(float(row["fitted_parameter_recovery"]) for row in selected) / len(selected),
            "heldout_bank_action_cosine_mean": sum(float(row["heldout_bank_action_cosine"]) for row in selected) / len(selected),
            "minimum_discovery_assignments": min(int(row["minimum_discovery_assignments"]) for row in selected),
            "improvement_over_static_256x": mean - static_means[bank],
            "improvement_over_rank2_kronecker": mean - kronecker_means[bank],
            "all_fits_non_degrading": all(
                float(row["best_relative_error_squared"]) <= float(row["initial_relative_error_squared"]) + 1e-8
                for row in selected
            ),
        }
    gates = plan["frozen_gates"]
    gate_results: dict[str, Any] = {}
    for bank in banks:
        summary = summaries[bank]
        gate_results[bank] = {
            "mean_recovery": summary["heldout_recovery_mean"] >= float(gates["heldout_recovery_mean_min"]),
            "minimum_layer_recovery": summary["heldout_recovery_minimum"] >= float(gates["heldout_recovery_every_layer_min"]),
            "improvement_over_kronecker": summary["improvement_over_rank2_kronecker"] >= float(gates["improvement_over_kronecker_min"]),
            "bank_action_cosine": summary["heldout_bank_action_cosine_mean"] >= float(gates["heldout_bank_action_cosine_mean_min"]),
            "occupancy": summary["minimum_discovery_assignments"] >= int(gates["minimum_expert_assignments"]),
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
    decision = "PASS_MPO_REPRESENTABILITY_UPPER_BOUND" if passed else "REJECT_RANK9_MPO_CPROJ_AT_EQUAL_BUDGET"

    args.output.mkdir(parents=True, exist_ok=False)
    rows_path = args.output / "mpo_oracle_rows.csv"
    with rows_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=union_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)
    cores_path = args.output / "mpo_cores.pt"
    torch.save(saved_cores, cores_path)
    result = {
        "schema_version": "nanogpt_sparse_moe_cproj_mpo_oracle_result_v1",
        "decision": decision,
        "all_values_finite": finite,
        "summary": summaries,
        "gates": gate_results,
        "mechanism": {**mechanism, "coordinates_per_expert": coordinates, "coordinate_compression_ratio": compression},
        "stepzero_selected_tensor_sha256": stepzero_hashes,
        "source": {
            "terminal_snapshot_sha256": file_sha256(args.terminal_snapshot),
            "parent_result_sha256": file_sha256(args.parent_result),
            "kronecker_seal_sha256": file_sha256(args.kronecker_seal),
            "plan_sha256": file_sha256(args.plan),
            "dataset_manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
        },
        "authorization": {"training": False, "mfu_preflight": False, "generated_experts": False, "larger_rung": False, "causal_output_frame_oracle": bool(passed)},
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
    result_path = args.output / "mpo_oracle_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    status = {
        "state": "finished",
        "exit_code": 0,
        "decision": decision,
        "result_sha256": file_sha256(result_path),
        "rows_sha256": file_sha256(rows_path),
        "cores_sha256": file_sha256(cores_path),
        "wall_seconds": time.time() - started,
    }
    status_path = args.output / "mpo_oracle_status.json"
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": status, "summary": summaries}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
