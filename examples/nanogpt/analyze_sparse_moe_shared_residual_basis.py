#!/usr/bin/env python3
"""Gate a model-wide residual-output basis for sparse-MoE c_proj writes.

The oracle learns one output-space basis from all layers on one terminal token
bank and evaluates the unchanged basis on independent banks.  Projection
coefficients are oracle values, so a pass establishes only output-frame
shareability; it does not establish that a compact causal coefficient
generator exists.
"""
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
from examples.nanogpt.analyze_sparse_moe_cproj_functional_state_budget import (
    action_spectrum,
    subspace_overlap,
)
from examples.nanogpt.analyze_sparse_moe_cproj_kronecker_oracle import fixed_inputs
from examples.nanogpt.analyze_sparse_moe_paired_alignment import file_sha256
from examples.nanogpt.analyze_sparse_moe_paired_atom_oracle import union_fieldnames
from examples.nanogpt.analyze_sparse_moe_stepzero_task_gradient_oracle import (
    layer_state_from_mapping,
    load_terminal_snapshot,
    model_from_exact_stepzero,
    selected_stepzero_hashes,
)


PLAN_SCHEMA = "nanogpt_sparse_moe_shared_residual_basis_plan_v1"
RESULT_SCHEMA = "nanogpt_sparse_moe_shared_residual_basis_result_v1"


def layout_state(
    *,
    dense_values: int,
    matrices: int,
    output_width: int,
    global_compression: float,
    per_matrix_coordinates: int,
) -> dict[str, float | int]:
    total_budget = int(math.floor(dense_values / float(global_compression)))
    local_state = int(matrices) * int(per_matrix_coordinates)
    remaining = total_budget - local_state
    rank = min(int(output_width), max(0, remaining // int(output_width)))
    used = local_state + rank * int(output_width)
    return {
        "global_compression_target": float(global_compression),
        "per_matrix_coordinates": int(per_matrix_coordinates),
        "total_coordinate_budget": total_budget,
        "per_matrix_coordinate_total": local_state,
        "shared_basis_rank": rank,
        "shared_basis_scalars": rank * int(output_width),
        "total_coordinates_used": used,
        "realized_global_compression": dense_values / max(used, 1),
    }


def basis_from_grams(grams: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    if not grams:
        raise ValueError("at least one Gram matrix is required")
    total = torch.stack([gram.float() for gram in grams]).sum(dim=0)
    eigenvalues, eigenvectors = torch.linalg.eigh(total)
    return eigenvalues.clamp_min(0).flip(0), eigenvectors.flip(1)


def projection_recovery(gram: torch.Tensor, basis: torch.Tensor, rank: int) -> float:
    width = max(0, min(int(rank), basis.shape[1]))
    denominator = torch.trace(gram.float()).clamp_min(1e-30)
    if width == 0:
        return 0.0
    selected = basis[:, :width].float()
    captured = torch.trace(selected.T @ gram.float() @ selected)
    return float(captured / denominator)


def local_oracle_recovery(gram: torch.Tensor, rank: int) -> float:
    values = torch.linalg.eigvalsh(gram.float()).clamp_min(0).flip(0)
    denominator = values.sum().clamp_min(1e-30)
    return float(values[: max(0, min(int(rank), values.numel()))].sum() / denominator)


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("shared residual-basis plan schema mismatch")
    analysis = plan.get("analysis", {})
    if analysis.get("parameter_updates") != 0:
        raise ValueError("shared-basis oracle must have zero parameter updates")
    if analysis.get("layers") != list(range(12)):
        raise ValueError("shared-basis oracle must cover all 12 layers")
    if analysis.get("experts") != 8 or analysis.get("top_k") != 2:
        raise ValueError("registered sparse-MoE topology changed")
    if analysis.get("matrix_shape") != [768, 1536]:
        raise ValueError("registered c_proj matrix shape changed")
    if analysis.get("fit_banks") != ["discovery_a", "discovery_b"]:
        raise ValueError("registered fit banks changed")
    if analysis.get("evaluation_banks") != [
        "discovery_a",
        "discovery_b",
        "heldout",
    ]:
        raise ValueError("registered evaluation banks changed")
    expected_layouts = []
    dense = 12 * 8 * 768 * 1536
    for compression, coordinates in (
        (500.0, 1536),
        (281.27038626609444, 1536),
        (281.27038626609444, 3072),
        (200.0, 1536),
        (200.0, 3072),
        (200.0, 4194),
    ):
        state = layout_state(
            dense_values=dense,
            matrices=96,
            output_width=768,
            global_compression=compression,
            per_matrix_coordinates=coordinates,
        )
        expected_layouts.append(state)
    if analysis.get("layouts") != expected_layouts:
        raise ValueError("registered shared-basis state layouts changed")
    if plan.get("decision_rule", {}).get("gates") != {
        "cross_bank_recovery_mean_minimum": 0.90,
        "cross_bank_recovery_every_layer_minimum": 0.80,
        "shared_minus_local_oracle_minimum": -0.05,
        "independent_basis_overlap_minimum": 0.80,
        "minimum_expert_assignments": 128,
    }:
        raise ValueError("registered shared-basis gates changed")
    authorization = plan.get("authorization", {})
    if authorization.get("run_zero_update_oracle") is not True:
        raise ValueError("shared-basis oracle is not authorized")
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
    parser.add_argument("--state-budget-seal", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    if args.output.exists():
        raise FileExistsError("shared residual-basis output already exists")
    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    source = plan["source"]
    for path, key, label in (
        (args.terminal_snapshot, "terminal_snapshot_sha256", "terminal snapshot"),
        (args.state_budget_seal, "state_budget_seal_sha256", "state-budget seal"),
        (args.data_dir / "manifest.json", "dataset_manifest_sha256", "dataset manifest"),
        (Path(__file__), "analyzer_sha256", "analyzer"),
    ):
        if file_sha256(path) != source[key]:
            raise ValueError(f"{label} hash disagrees with frozen plan")

    analysis = plan["analysis"]
    layers = [int(value) for value in analysis["layers"]]
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

    banks = ["discovery_a", "discovery_b", "heldout"]
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

    fit_banks = [str(value) for value in analysis["fit_banks"]]
    bases: dict[str, torch.Tensor] = {}
    spectra: dict[str, torch.Tensor] = {}
    for bank in fit_banks:
        values, basis = basis_from_grams(
            [joint_grams[(bank, layer)] for layer in layers]
        )
        bases[bank] = basis
        spectra[bank] = values

    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    gates = plan["decision_rule"]["gates"]
    for layout_index, layout in enumerate(analysis["layouts"]):
        rank = int(layout["shared_basis_rank"])
        layout_name = (
            f"g{str(layout['global_compression_target']).replace('.', 'p')}"
            f"_q{layout['per_matrix_coordinates']}_r{rank}"
        )
        layout_rows: list[dict[str, Any]] = []
        overlap = subspace_overlap(
            bases[fit_banks[0]], bases[fit_banks[1]], rank
        )
        for fit_bank in fit_banks:
            evaluation_banks = [
                bank
                for bank in banks
                if bank != fit_bank
            ]
            for evaluation_bank in evaluation_banks:
                for layer in layers:
                    gram = joint_grams[(evaluation_bank, layer)]
                    shared = projection_recovery(gram, bases[fit_bank], rank)
                    local = local_oracle_recovery(gram, rank)
                    expert_values = [
                        projection_recovery(
                            expert_grams[(evaluation_bank, layer, expert)],
                            bases[fit_bank],
                            rank,
                        )
                        for expert in range(int(analysis["experts"]))
                    ]
                    row = {
                        "layout_index": layout_index,
                        "layout": layout_name,
                        "global_compression_target": layout["global_compression_target"],
                        "realized_global_compression": layout["realized_global_compression"],
                        "per_matrix_coordinates": layout["per_matrix_coordinates"],
                        "shared_basis_rank": rank,
                        "fit_bank": fit_bank,
                        "evaluation_bank": evaluation_bank,
                        "layer": layer,
                        "shared_recovery": shared,
                        "local_oracle_recovery": local,
                        "shared_minus_local_oracle": shared - local,
                        "minimum_expert_recovery": min(expert_values),
                        "mean_expert_recovery": sum(expert_values) / len(expert_values),
                        "minimum_expert_assignments": min(
                            occupancies[(evaluation_bank, layer)]
                        ),
                        "independent_basis_overlap": overlap,
                    }
                    rows.append(row)
                    layout_rows.append(row)
        mean_recovery = sum(float(row["shared_recovery"]) for row in layout_rows) / len(layout_rows)
        minimum_recovery = min(float(row["shared_recovery"]) for row in layout_rows)
        minimum_gap = min(float(row["shared_minus_local_oracle"]) for row in layout_rows)
        minimum_occupancy = min(int(row["minimum_expert_assignments"]) for row in layout_rows)
        gate_results = {
            "cross_bank_recovery_mean": mean_recovery
            >= float(gates["cross_bank_recovery_mean_minimum"]),
            "cross_bank_recovery_every_layer": minimum_recovery
            >= float(gates["cross_bank_recovery_every_layer_minimum"]),
            "shared_minus_local_oracle": minimum_gap
            >= float(gates["shared_minus_local_oracle_minimum"]),
            "independent_basis_overlap": overlap
            >= float(gates["independent_basis_overlap_minimum"]),
            "minimum_expert_assignments": minimum_occupancy
            >= int(gates["minimum_expert_assignments"]),
        }
        gate_results["all_pass"] = all(gate_results.values())
        summaries[layout_name] = {
            **layout,
            "cross_bank_recovery_mean": mean_recovery,
            "cross_bank_recovery_minimum": minimum_recovery,
            "shared_minus_local_oracle_minimum": minimum_gap,
            "independent_basis_overlap": overlap,
            "minimum_expert_recovery": min(
                float(row["minimum_expert_recovery"]) for row in layout_rows
            ),
            "minimum_expert_assignments": minimum_occupancy,
            "gates": gate_results,
        }

    passing = [name for name, summary in summaries.items() if summary["gates"]["all_pass"]]
    if passing:
        passing.sort(
            key=lambda name: (
                -float(summaries[name]["realized_global_compression"]),
                int(summaries[name]["per_matrix_coordinates"]),
            )
        )
        selected = passing[0]
        decision = "PASS_SHARED_RESIDUAL_OUTPUT_FRAME"
    else:
        selected = None
        decision = "REJECT_SHARED_RESIDUAL_OUTPUT_FRAME_AT_REGISTERED_BUDGETS"
    finite = all(
        math.isfinite(float(value))
        for row in rows
        for key, value in row.items()
        if key not in {"layout", "fit_bank", "evaluation_bank"}
        and isinstance(value, (int, float))
    )
    if not finite:
        raise RuntimeError("nonfinite shared residual-basis result")

    args.output.mkdir(parents=True, exist_ok=False)
    rows_path = args.output / "shared_residual_basis_rows.csv"
    with rows_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=union_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)
    bases_path = args.output / "shared_residual_bases.pt"
    torch.save(
        {
            **{f"basis:{key}": value for key, value in bases.items()},
            **{f"spectrum:{key}": value for key, value in spectra.items()},
        },
        bases_path,
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "decision": decision,
        "selected_layout": selected,
        "all_values_finite": finite,
        "summaries": summaries,
        "stepzero_selected_tensor_sha256": stepzero_hashes,
        "source": {
            "terminal_snapshot_sha256": file_sha256(args.terminal_snapshot),
            "state_budget_seal_sha256": file_sha256(args.state_budget_seal),
            "dataset_manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
            "plan_sha256": file_sha256(args.plan),
        },
        "scope_caveat": plan["scope_caveat"],
        "authorization": {
            "coefficient_mechanism_oracle": bool(passing),
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
    result_path = args.output / "shared_residual_basis_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    status = {
        "state": "finished",
        "exit_code": 0,
        "decision": decision,
        "selected_layout": selected,
        "result_sha256": file_sha256(result_path),
        "rows_sha256": file_sha256(rows_path),
        "bases_sha256": file_sha256(bases_path),
        "wall_seconds": time.time() - started,
    }
    status_path = args.output / "shared_residual_basis_status.json"
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": status, "summaries": summaries}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
