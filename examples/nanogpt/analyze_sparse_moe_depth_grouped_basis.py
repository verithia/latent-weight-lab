#!/usr/bin/env python3
"""Gate three depth-grouped residual-output bases at the same 200x budget."""
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
    cproj_target_action,
    routed_hidden_frames,
)
from examples.nanogpt.analyze_sparse_moe_cproj_kronecker_oracle import fixed_inputs
from examples.nanogpt.analyze_sparse_moe_paired_alignment import file_sha256
from examples.nanogpt.analyze_sparse_moe_paired_atom_oracle import union_fieldnames
from examples.nanogpt.analyze_sparse_moe_shared_residual_basis import (
    basis_from_grams,
    local_oracle_recovery,
    projection_recovery,
)
from examples.nanogpt.analyze_sparse_moe_cproj_functional_state_budget import (
    subspace_overlap,
)
from examples.nanogpt.analyze_sparse_moe_stepzero_task_gradient_oracle import (
    layer_state_from_mapping,
    load_terminal_snapshot,
    model_from_exact_stepzero,
    selected_stepzero_hashes,
)


PLAN_SCHEMA = "nanogpt_sparse_moe_depth_grouped_basis_plan_v1"
RESULT_SCHEMA = "nanogpt_sparse_moe_depth_grouped_basis_result_v1"


def allocate_group_ranks(
    group_eigenvalues: list[torch.Tensor],
    *,
    total_rank: int,
    minimum_rank: int,
) -> list[int]:
    groups = len(group_eigenvalues)
    if groups <= 0 or total_rank < groups * minimum_rank:
        raise ValueError("rank budget cannot satisfy the group minimum")
    allocations = [int(minimum_rank)] * groups
    candidates: list[tuple[float, int]] = []
    for group, values in enumerate(group_eigenvalues):
        normalized = values.double().clamp_min(0)
        normalized = normalized / normalized.sum().clamp_min(1e-30)
        for index in range(int(minimum_rank), normalized.numel()):
            candidates.append((float(normalized[index]), group))
    candidates.sort(key=lambda pair: (-pair[0], pair[1]))
    remaining = int(total_rank) - sum(allocations)
    if remaining > len(candidates):
        raise ValueError("rank budget exceeds available basis dimensions")
    for _value, group in candidates[:remaining]:
        allocations[group] += 1
    return allocations


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("depth-grouped basis plan schema mismatch")
    analysis = plan.get("analysis", {})
    if analysis.get("parameter_updates") != 0:
        raise ValueError("depth-grouped basis oracle must have zero updates")
    if analysis.get("layer_groups") != [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
    ]:
        raise ValueError("registered contiguous depth groups changed")
    if analysis.get("experts") != 8 or analysis.get("top_k") != 2:
        raise ValueError("registered sparse-MoE topology changed")
    if analysis.get("matrix_shape") != [768, 1536]:
        raise ValueError("registered c_proj shape changed")
    if analysis.get("total_shared_basis_rank") != 545:
        raise ValueError("registered total basis-rank budget changed")
    if analysis.get("minimum_rank_per_group") != 64:
        raise ValueError("registered minimum group rank changed")
    if analysis.get("per_matrix_coordinates") != 1536:
        raise ValueError("registered per-matrix state changed")
    if analysis.get("total_coordinates_used") != 566016:
        raise ValueError("registered total state changed")
    if analysis.get("realized_global_compression") != 200.07598371777476:
        raise ValueError("registered realized compression changed")
    if analysis.get("calibration_banks") != ["calibration_a", "calibration_b"]:
        raise ValueError("registered calibration banks changed")
    if analysis.get("fresh_evaluation_banks") != ["evaluation_c", "heldout"]:
        raise ValueError("registered evaluation banks changed")
    if plan.get("decision_rule", {}).get("gates") != {
        "fresh_recovery_mean_minimum": 0.90,
        "fresh_recovery_every_layer_minimum": 0.80,
        "grouped_minus_local_oracle_minimum": -0.05,
        "independent_group_basis_overlap_minimum": 0.80,
        "minimum_expert_assignments": 128,
    }:
        raise ValueError("registered depth-grouped gates changed")
    authorization = plan.get("authorization", {})
    if authorization.get("run_zero_update_oracle") is not True:
        raise ValueError("depth-grouped oracle is not authorized")
    for key in (
        "implement_candidate_structure",
        "run_exact_config_mfu",
        "run_language_model_training",
        "larger_rung",
    ):
        if authorization.get(key) is not False:
            raise ValueError(f"plan must keep {key} false")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--terminal-snapshot", required=True, type=Path)
    parser.add_argument("--shared-basis-seal", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    if args.output.exists():
        raise FileExistsError("depth-grouped basis output already exists")
    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    source = plan["source"]
    for path, key, label in (
        (args.terminal_snapshot, "terminal_snapshot_sha256", "terminal snapshot"),
        (args.shared_basis_seal, "shared_basis_seal_sha256", "shared-basis seal"),
        (args.data_dir / "manifest.json", "dataset_manifest_sha256", "dataset manifest"),
        (Path(__file__), "analyzer_sha256", "analyzer"),
    ):
        if file_sha256(path) != source[key]:
            raise ValueError(f"{label} hash disagrees with frozen plan")

    analysis = plan["analysis"]
    groups = [[int(layer) for layer in value] for value in analysis["layer_groups"]]
    layers = [layer for group in groups for layer in group]
    payload = load_terminal_snapshot(args.terminal_snapshot)
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

    banks = [spec["name"] for spec in plan["functional_protocol"]["discovery_banks"]] + ["heldout"]
    joint_grams: dict[tuple[str, int], torch.Tensor] = {}
    expert_grams: dict[tuple[str, int, int], torch.Tensor] = {}
    occupancies: dict[tuple[str, int], list[int]] = {}
    for bank in banks:
        for layer in layers:
            x = inputs[bank][layer].to(args.device)
            frames, counts = routed_hidden_frames(
                terminal[layer], x, int(analysis["top_k"]), args.device
            )
            occupancies[(bank, layer)] = counts
            joint = cproj_target_action(
                frames,
                initial[layer].c_proj,
                terminal[layer].c_proj,
                x.shape[0],
                x.shape[1],
                args.device,
            )
            joint_grams[(bank, layer)] = (joint.T @ joint).detach().cpu()
            delta = (
                terminal[layer].c_proj.to(args.device).float()
                - initial[layer].c_proj.to(args.device).float()
            )
            for expert, frame in enumerate(frames):
                local = (
                    frame.hidden.float() @ delta[expert].T
                ) * frame.probabilities[:, None]
                expert_grams[(bank, layer, expert)] = (local.T @ local).detach().cpu()
            del x, frames, joint, delta
            torch.cuda.empty_cache()

    calibration_banks = [str(value) for value in analysis["calibration_banks"]]
    evaluation_banks = [str(value) for value in analysis["fresh_evaluation_banks"]]
    bases: dict[tuple[str, int], torch.Tensor] = {}
    spectra: dict[tuple[str, int], torch.Tensor] = {}
    allocations: dict[str, list[int]] = {}
    for bank in calibration_banks:
        for group_index, group in enumerate(groups):
            values, basis = basis_from_grams(
                [joint_grams[(bank, layer)] for layer in group]
            )
            spectra[(bank, group_index)] = values
            bases[(bank, group_index)] = basis
        allocations[bank] = allocate_group_ranks(
            [spectra[(bank, index)] for index in range(len(groups))],
            total_rank=int(analysis["total_shared_basis_rank"]),
            minimum_rank=int(analysis["minimum_rank_per_group"]),
        )

    group_overlaps: dict[int, float] = {}
    for group_index in range(len(groups)):
        rank = min(
            allocations[calibration_banks[0]][group_index],
            allocations[calibration_banks[1]][group_index],
        )
        group_overlaps[group_index] = subspace_overlap(
            bases[(calibration_banks[0], group_index)],
            bases[(calibration_banks[1], group_index)],
            rank,
        )

    rows: list[dict[str, Any]] = []
    for calibration_bank in calibration_banks:
        for evaluation_bank in evaluation_banks:
            for group_index, group in enumerate(groups):
                rank = allocations[calibration_bank][group_index]
                basis = bases[(calibration_bank, group_index)]
                for layer in group:
                    gram = joint_grams[(evaluation_bank, layer)]
                    grouped = projection_recovery(gram, basis, rank)
                    local = local_oracle_recovery(gram, rank)
                    expert_values = [
                        projection_recovery(
                            expert_grams[(evaluation_bank, layer, expert)],
                            basis,
                            rank,
                        )
                        for expert in range(int(analysis["experts"]))
                    ]
                    rows.append(
                        {
                            "calibration_bank": calibration_bank,
                            "evaluation_bank": evaluation_bank,
                            "group": group_index,
                            "layer": layer,
                            "allocated_rank": rank,
                            "grouped_recovery": grouped,
                            "local_oracle_recovery": local,
                            "grouped_minus_local_oracle": grouped - local,
                            "minimum_expert_recovery": min(expert_values),
                            "mean_expert_recovery": sum(expert_values) / len(expert_values),
                            "minimum_expert_assignments": min(occupancies[(evaluation_bank, layer)]),
                            "independent_group_basis_overlap": group_overlaps[group_index],
                        }
                    )

    mean_recovery = sum(float(row["grouped_recovery"]) for row in rows) / len(rows)
    minimum_recovery = min(float(row["grouped_recovery"]) for row in rows)
    minimum_gap = min(float(row["grouped_minus_local_oracle"]) for row in rows)
    minimum_overlap = min(group_overlaps.values())
    minimum_occupancy = min(int(row["minimum_expert_assignments"]) for row in rows)
    gates = plan["decision_rule"]["gates"]
    gate_results = {
        "fresh_recovery_mean": mean_recovery >= float(gates["fresh_recovery_mean_minimum"]),
        "fresh_recovery_every_layer": minimum_recovery >= float(gates["fresh_recovery_every_layer_minimum"]),
        "grouped_minus_local_oracle": minimum_gap >= float(gates["grouped_minus_local_oracle_minimum"]),
        "independent_group_basis_overlap": minimum_overlap >= float(gates["independent_group_basis_overlap_minimum"]),
        "minimum_expert_assignments": minimum_occupancy >= int(gates["minimum_expert_assignments"]),
    }
    gate_results["all_pass"] = all(gate_results.values())
    decision = (
        "PASS_DEPTH_GROUPED_RESIDUAL_OUTPUT_FRAMES"
        if gate_results["all_pass"]
        else "REJECT_DEPTH_GROUPED_RESIDUAL_OUTPUT_FRAMES"
    )
    allocation_l1 = sum(
        abs(left - right)
        for left, right in zip(
            allocations[calibration_banks[0]], allocations[calibration_banks[1]]
        )
    )
    finite = all(
        math.isfinite(float(value))
        for row in rows
        for key, value in row.items()
        if key not in {"calibration_bank", "evaluation_bank"}
        and isinstance(value, (int, float))
    )
    if not finite:
        raise RuntimeError("nonfinite depth-grouped basis result")

    args.output.mkdir(parents=True, exist_ok=False)
    rows_path = args.output / "depth_grouped_basis_rows.csv"
    with rows_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=union_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)
    bases_path = args.output / "depth_grouped_bases.pt"
    torch.save(
        {
            **{f"basis:{bank}:group{group}": value for (bank, group), value in bases.items()},
            **{f"spectrum:{bank}:group{group}": value for (bank, group), value in spectra.items()},
        },
        bases_path,
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "decision": decision,
        "all_values_finite": finite,
        "allocations": allocations,
        "allocation_l1_difference": allocation_l1,
        "group_overlaps": {str(key): value for key, value in group_overlaps.items()},
        "summary": {
            "fresh_recovery_mean": mean_recovery,
            "fresh_recovery_minimum": minimum_recovery,
            "grouped_minus_local_oracle_minimum": minimum_gap,
            "minimum_expert_recovery": min(float(row["minimum_expert_recovery"]) for row in rows),
            "minimum_expert_assignments": minimum_occupancy,
            "independent_group_basis_overlap_minimum": minimum_overlap,
        },
        "gates": gate_results,
        "stepzero_selected_tensor_sha256": stepzero_hashes,
        "source": {
            "terminal_snapshot_sha256": file_sha256(args.terminal_snapshot),
            "shared_basis_seal_sha256": file_sha256(args.shared_basis_seal),
            "dataset_manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
            "plan_sha256": file_sha256(args.plan),
        },
        "scope_caveat": plan["scope_caveat"],
        "authorization": {
            "coefficient_mechanism_oracle": bool(gate_results["all_pass"]),
            "candidate_implementation": False,
            "mfu_preflight": False,
            "training": False,
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
    result_path = args.output / "depth_grouped_basis_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    status = {
        "state": "finished",
        "exit_code": 0,
        "decision": decision,
        "result_sha256": file_sha256(result_path),
        "rows_sha256": file_sha256(rows_path),
        "bases_sha256": file_sha256(bases_path),
        "wall_seconds": time.time() - started,
    }
    status_path = args.output / "depth_grouped_basis_status.json"
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": status, "summary": result["summary"], "allocations": allocations}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
