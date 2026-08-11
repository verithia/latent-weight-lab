#!/usr/bin/env python3
"""Measure the routed functional-rank budget of sparse-MoE c_proj writes.

This is a zero-update descriptive oracle.  It applies the exact terminal
c_proj displacement to terminal same-run expert activations and measures the
singular spectrum of the resulting routed residual write.  The joint spectrum
gives an optimistic necessary condition for explicit per-expert low-rank
state: if each of E experts has ordinary matrix rank at most r, the summed
routed write has rank at most E*r, even when cross-expert cancellation is
perfect.  It is not a lower bound for arbitrary nonlinear procedural maps.
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
from examples.nanogpt.analyze_sparse_moe_cproj_kronecker_oracle import fixed_inputs
from examples.nanogpt.analyze_sparse_moe_paired_alignment import file_sha256
from examples.nanogpt.analyze_sparse_moe_paired_atom_oracle import union_fieldnames
from examples.nanogpt.analyze_sparse_moe_stepzero_task_gradient_oracle import (
    layer_state_from_mapping,
    load_terminal_snapshot,
    model_from_exact_stepzero,
    selected_stepzero_hashes,
)


PLAN_SCHEMA = "nanogpt_sparse_moe_cproj_functional_state_budget_plan_v1"
RESULT_SCHEMA = "nanogpt_sparse_moe_cproj_functional_state_budget_result_v1"


def intrinsic_rank_dimension(rank: int, rows: int, columns: int) -> int:
    """Dimension r(m+n-r) of the ordinary rank-r matrix manifold."""
    if rank < 0 or rank > min(rows, columns):
        raise ValueError("rank outside matrix dimensions")
    return rank * (rows + columns - rank)


def largest_rank_within_budget(budget: int, rows: int, columns: int) -> int:
    valid = [
        rank
        for rank in range(min(rows, columns) + 1)
        if intrinsic_rank_dimension(rank, rows, columns) <= int(budget)
    ]
    return max(valid)


def rank_for_energy(eigenvalues: torch.Tensor, fraction: float) -> int:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("energy fraction must lie in (0, 1]")
    values = eigenvalues.double().flatten().clamp_min(0)
    total = values.sum().clamp_min(1e-30)
    return int(torch.searchsorted(values.cumsum(0), fraction * total).item() + 1)


def recovery_at_rank(eigenvalues: torch.Tensor, rank: int) -> float:
    values = eigenvalues.double().flatten().clamp_min(0)
    total = values.sum().clamp_min(1e-30)
    return float(values[: max(0, min(int(rank), values.numel()))].sum() / total)


def compression_label(target: float) -> str:
    normalized = f"{float(target):.6f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"compression_{normalized}x"


def action_spectrum(action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return descending output-space energy and matching directions."""
    action = action.float()
    gram = action.T @ action
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    return eigenvalues.clamp_min(0).flip(0), eigenvectors.flip(1)


def subspace_overlap(left: torch.Tensor, right: torch.Tensor, rank: int) -> float:
    """Mean squared canonical cosine between two leading output subspaces."""
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("subspace bases must share an ambient dimension")
    width = min(int(rank), left.shape[1], right.shape[1])
    if width <= 0:
        raise ValueError("subspace rank must be positive")
    cross = left[:, :width].T @ right[:, :width]
    return float(cross.square().sum() / width)


def spectrum_metrics(
    eigenvalues: torch.Tensor,
    *,
    energy_thresholds: list[float],
    compression_targets: list[float],
    experts: int,
    output_width: int,
    input_width: int,
    joint: bool,
) -> dict[str, float | int]:
    values = eigenvalues.double().flatten().clamp_min(0)
    total = values.sum().clamp_min(1e-30)
    probabilities = values / total
    positive = probabilities > 0
    metrics: dict[str, float | int] = {
        "action_energy": float(total),
        "stable_rank": float(total / values.max().clamp_min(1e-30)),
        "entropy_effective_rank": float(
            torch.exp(-(probabilities[positive] * probabilities[positive].log()).sum())
        ),
    }
    for fraction in energy_thresholds:
        label = f"rank_{int(round(100 * fraction))}pct"
        rank = rank_for_energy(values, fraction)
        metrics[label] = rank
        if joint:
            # Sum_e rank(W_e) must be at least rank(joint routed action).  The
            # even allocation is descriptive; the budget gate below uses the
            # exact optimistic upper bound E*r_budget instead.
            per_expert = int(math.ceil(rank / experts))
            dof = intrinsic_rank_dimension(per_expert, output_width, input_width)
            metrics[f"{label}_even_per_expert_rank"] = per_expert
            metrics[f"{label}_even_per_expert_intrinsic_dof"] = dof
            metrics[f"{label}_even_per_expert_state_compression"] = (
                output_width * input_width / max(dof, 1)
            )
        else:
            dof = intrinsic_rank_dimension(rank, output_width, input_width)
            metrics[f"{label}_intrinsic_dof"] = dof
            metrics[f"{label}_state_compression"] = (
                output_width * input_width / max(dof, 1)
            )
    dense = output_width * input_width
    for target in compression_targets:
        budget = int(math.floor(dense / float(target)))
        per_expert_rank = largest_rank_within_budget(
            budget, output_width, input_width
        )
        action_rank = experts * per_expert_rank if joint else per_expert_rank
        label = compression_label(target)
        metrics[f"{label}_coordinate_budget_per_expert"] = budget
        metrics[f"{label}_ordinary_rank_per_expert"] = per_expert_rank
        metrics[f"{label}_optimistic_action_rank"] = action_rank
        metrics[f"{label}_best_recovery"] = recovery_at_rank(values, action_rank)
    return metrics


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("functional state-budget plan schema mismatch")
    analysis = plan.get("analysis", {})
    if analysis.get("parameter_updates") != 0:
        raise ValueError("state-budget analysis must have zero parameter updates")
    if analysis.get("experts") != 8 or analysis.get("top_k") != 2:
        raise ValueError("registered sparse-MoE topology changed")
    if analysis.get("matrix_shape") != [768, 1536]:
        raise ValueError("registered c_proj matrix shape changed")
    if analysis.get("layers") != [0, 5, 11]:
        raise ValueError("registered layer sample changed")
    if analysis.get("energy_thresholds") != [0.5, 0.8, 0.9, 0.95, 0.99]:
        raise ValueError("registered energy thresholds changed")
    if analysis.get("compression_targets") != [200.0, 281.27038626609444, 500.0]:
        raise ValueError("registered compression targets changed")
    if analysis.get("subspace_overlap_ranks") != [1, 2, 4, 8, 16, 32, 64]:
        raise ValueError("registered overlap ranks changed")
    gates = plan.get("decision_rule", {}).get("gates", {})
    if gates != {
        "compression_200x_joint_recovery_mean_minimum": 0.90,
        "compression_200x_joint_recovery_every_layer_bank_minimum": 0.80,
    }:
        raise ValueError("state-budget decision gates changed")
    authorization = plan.get("authorization", {})
    if authorization.get("run_zero_update_analysis") is not True:
        raise ValueError("zero-update analysis is not authorized")
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
    parser.add_argument("--mpo-seal", required=True, type=Path)
    parser.add_argument("--dense-budget-seal", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    if args.output.exists():
        raise FileExistsError("functional state-budget output already exists")
    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    source = plan["source"]
    for path, key, label in (
        (args.terminal_snapshot, "terminal_snapshot_sha256", "terminal snapshot"),
        (args.mpo_seal, "mpo_seal_sha256", "MPO seal"),
        (args.dense_budget_seal, "dense_budget_seal_sha256", "dense budget seal"),
        (args.data_dir / "manifest.json", "dataset_manifest_sha256", "dataset manifest"),
        (Path(__file__), "analyzer_sha256", "analyzer"),
    ):
        if file_sha256(path) != source[key]:
            raise ValueError(f"{label} hash disagrees with frozen plan")

    payload = load_terminal_snapshot(args.terminal_snapshot)
    analysis = plan["analysis"]
    layers = [int(value) for value in analysis["layers"]]
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

    protocol = plan["functional_protocol"]
    bank_names = [spec["name"] for spec in protocol["discovery_banks"]] + [
        "heldout"
    ]
    experts = int(analysis["experts"])
    output_width, input_width = [int(value) for value in analysis["matrix_shape"]]
    thresholds = [float(value) for value in analysis["energy_thresholds"]]
    compression_targets = [float(value) for value in analysis["compression_targets"]]
    overlap_ranks = [int(value) for value in analysis["subspace_overlap_ranks"]]
    rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    spectra: dict[str, torch.Tensor] = {}
    joint_bases: dict[tuple[str, int], torch.Tensor] = {}

    for layer in layers:
        delta = (
            terminal[layer].c_proj.to(args.device).float()
            - initial[layer].c_proj.to(args.device).float()
        )
        for bank in bank_names:
            x = inputs[bank][layer].to(args.device)
            frames, counts = routed_hidden_frames(
                terminal[layer], x, int(analysis["top_k"]), args.device
            )
            joint = cproj_target_action(
                frames,
                initial[layer].c_proj,
                terminal[layer].c_proj,
                x.shape[0],
                x.shape[1],
                args.device,
            )
            joint_values, joint_vectors = action_spectrum(joint)
            joint_bases[(bank, layer)] = joint_vectors.detach().cpu()
            spectra[f"joint:{bank}:layer{layer}"] = joint_values.detach().cpu()
            rows.append(
                {
                    "scope": "joint",
                    "bank": bank,
                    "layer": layer,
                    "expert": -1,
                    "assignments": sum(counts),
                    "minimum_expert_assignments": min(counts),
                    **spectrum_metrics(
                        joint_values,
                        energy_thresholds=thresholds,
                        compression_targets=compression_targets,
                        experts=experts,
                        output_width=output_width,
                        input_width=input_width,
                        joint=True,
                    ),
                }
            )
            for expert, frame in enumerate(frames):
                local = (
                    frame.hidden.float() @ delta[expert].T
                ) * frame.probabilities[:, None]
                values, _vectors = action_spectrum(local)
                spectra[f"expert{expert}:{bank}:layer{layer}"] = values.detach().cpu()
                rows.append(
                    {
                        "scope": "expert",
                        "bank": bank,
                        "layer": layer,
                        "expert": expert,
                        "assignments": counts[expert],
                        "minimum_expert_assignments": min(counts),
                        **spectrum_metrics(
                            values,
                            energy_thresholds=thresholds,
                            compression_targets=compression_targets,
                            experts=experts,
                            output_width=output_width,
                            input_width=input_width,
                            joint=False,
                        ),
                    }
                )

    pairs = [("discovery_a", "discovery_b"), ("discovery_a", "heldout"), ("discovery_b", "heldout")]
    for left, right in pairs:
        for layer in layers:
            for rank in overlap_ranks:
                overlap_rows.append(
                    {
                        "left_bank": left,
                        "right_bank": right,
                        "layer": layer,
                        "rank": rank,
                        "joint_output_subspace_overlap": subspace_overlap(
                            joint_bases[(left, layer)], joint_bases[(right, layer)], rank
                        ),
                    }
                )

    joint_rows = [row for row in rows if row["scope"] == "joint"]
    budget_key = "compression_200x_best_recovery"
    mean_recovery = sum(float(row[budget_key]) for row in joint_rows) / len(joint_rows)
    minimum_recovery = min(float(row[budget_key]) for row in joint_rows)
    gates = plan["decision_rule"]["gates"]
    passed = (
        mean_recovery >= float(gates["compression_200x_joint_recovery_mean_minimum"])
        and minimum_recovery >= float(gates["compression_200x_joint_recovery_every_layer_bank_minimum"])
    )
    decision = (
        "EXPLICIT_LOW_RANK_WRITE_STATE_COMPATIBLE_WITH_200X"
        if passed
        else "EXPLICIT_LOW_RANK_WRITE_STATE_INCOMPATIBLE_WITH_200X"
    )
    aggregate: dict[str, Any] = {
        "compression_200x_joint_recovery_mean": mean_recovery,
        "compression_200x_joint_recovery_minimum": minimum_recovery,
        "joint_rank80_mean": sum(float(row["rank_80pct"]) for row in joint_rows) / len(joint_rows),
        "joint_rank90_mean": sum(float(row["rank_90pct"]) for row in joint_rows) / len(joint_rows),
        "joint_rank90_minimum": min(int(row["rank_90pct"]) for row in joint_rows),
        "minimum_expert_assignments": min(int(row["minimum_expert_assignments"]) for row in joint_rows),
    }
    for rank in overlap_ranks:
        selected = [row for row in overlap_rows if int(row["rank"]) == rank]
        aggregate[f"joint_subspace_overlap_rank{rank}_mean"] = sum(
            float(row["joint_output_subspace_overlap"]) for row in selected
        ) / len(selected)
        aggregate[f"joint_subspace_overlap_rank{rank}_minimum"] = min(
            float(row["joint_output_subspace_overlap"]) for row in selected
        )
    finite = all(
        math.isfinite(float(value))
        for row in rows
        for key, value in row.items()
        if key not in {"scope", "bank"} and isinstance(value, (int, float))
    ) and all(
        math.isfinite(float(row["joint_output_subspace_overlap"]))
        for row in overlap_rows
    )
    if not finite:
        raise RuntimeError("nonfinite functional state-budget result")

    args.output.mkdir(parents=True, exist_ok=False)
    rows_path = args.output / "functional_state_budget_rows.csv"
    with rows_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=union_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)
    overlap_path = args.output / "functional_state_budget_overlaps.csv"
    with overlap_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=union_fieldnames(overlap_rows))
        writer.writeheader()
        writer.writerows(overlap_rows)
    spectra_path = args.output / "functional_state_budget_spectra.pt"
    torch.save(spectra, spectra_path)
    result = {
        "schema_version": RESULT_SCHEMA,
        "decision": decision,
        "all_values_finite": finite,
        "aggregate": aggregate,
        "gates": {
            "compression_200x_joint_recovery_mean": mean_recovery
            >= float(gates["compression_200x_joint_recovery_mean_minimum"]),
            "compression_200x_joint_recovery_every_layer_bank": minimum_recovery
            >= float(gates["compression_200x_joint_recovery_every_layer_bank_minimum"]),
        },
        "stepzero_selected_tensor_sha256": stepzero_hashes,
        "source": {
            "terminal_snapshot_sha256": file_sha256(args.terminal_snapshot),
            "mpo_seal_sha256": file_sha256(args.mpo_seal),
            "dense_budget_seal_sha256": file_sha256(args.dense_budget_seal),
            "dataset_manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
            "plan_sha256": file_sha256(args.plan),
        },
        "scope_caveat": plan["scope_caveat"],
        "authorization": {
            "training": False,
            "mfu_preflight": False,
            "generated_experts": False,
            "larger_rung": False,
            "choose_next_architecture_class": True,
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
    result_path = args.output / "functional_state_budget_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    status = {
        "state": "finished",
        "exit_code": 0,
        "decision": decision,
        "result_sha256": file_sha256(result_path),
        "rows_sha256": file_sha256(rows_path),
        "overlaps_sha256": file_sha256(overlap_path),
        "spectra_sha256": file_sha256(spectra_path),
        "wall_seconds": time.time() - started,
    }
    status_path = args.output / "functional_state_budget_status.json"
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": status, "aggregate": aggregate}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
