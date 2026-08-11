#!/usr/bin/env python3
"""Score causal per-neuron paired charts for dense complete-expert MoE states.

This is a zero-update representability upper bound.  Every hidden neuron keeps
its incoming ``c_fc`` row coupled to its outgoing ``c_proj`` column.  The
candidate image is built only from causally prior, same-gauge atom chords; the
target chord is used only to fit oracle coordinates, and exact nonlinear
residual-write recovery is evaluated on held-out activations.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_cproj_manifold import load_model
from examples.nanogpt.analyze_residual_compatibility import fixed_validation_batches
from examples.nanogpt.analyze_sparse_moe_paired_alignment import (
    collect_inputs,
    file_sha256,
    tensor_sha256,
)
from examples.nanogpt.analyze_sparse_moe_rolling_tangent_oracle import (
    LayerState,
    align_layer_sequence,
    load_layer_state,
    orthonormal_span,
    recovery_fraction,
    sparse_moe_output,
)
from latent_weight_lab.block_fht import _load_block_fht_ext, block_fht_slice


def _atom_views(state: LayerState) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return state.router.float(), state.c_fc.float(), state.c_proj.float().transpose(1, 2)


def _state_from_atom_views(
    router: torch.Tensor,
    c_fc: torch.Tensor,
    c_proj_atoms: torch.Tensor,
) -> LayerState:
    return LayerState(router, c_fc, c_proj_atoms.transpose(1, 2))


def _state_chord(right: LayerState, left: LayerState) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    right_router, right_fc, right_proj = _atom_views(right)
    left_router, left_fc, left_proj = _atom_views(left)
    return right_router - left_router, right_fc - left_fc, right_proj - left_proj


def energy_recovery(predicted: torch.Tensor, target: torch.Tensor) -> float:
    denominator = target.float().square().sum().clamp_min(1e-30)
    return float(1.0 - (predicted.float() - target.float()).square().sum() / denominator)


def project_rows(
    target: torch.Tensor,
    basis: torch.Tensor,
    ridge_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project independent row targets onto independent short row bases.

    ``target`` has shape ``[..., width]``. ``basis`` has shape
    ``[rank, ..., width]`` and may broadcast singleton atom dimensions.
    """
    if basis.ndim != target.ndim + 1 or basis.shape[-1] != target.shape[-1]:
        raise ValueError("basis and target shapes disagree")
    rank = basis.shape[0]
    expanded = basis.expand((rank,) + target.shape).movedim(0, -2).float()
    live_target = target.float()
    gram = torch.einsum("...kd,...ld->...kl", expanded, expanded)
    rhs = torch.einsum("...kd,...d->...k", expanded, live_target)
    ridge = float(ridge_ratio) * gram.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1e-30)
    identity = torch.eye(rank, device=gram.device, dtype=gram.dtype)
    coordinates = torch.linalg.solve(gram + ridge[..., None, None] * identity, rhs)
    projected = torch.einsum("...k,...kd->...d", coordinates, expanded)
    return projected, coordinates


def fixed_local_basis(width: int, rank: int, seed: int, device: str) -> torch.Tensor:
    latent_width = max(rank, 32)
    vectors = []
    for coordinate in range(rank):
        latent = torch.zeros(latent_width, device=device)
        latent[coordinate] = 1.0
        vectors.append(block_fht_slice(latent, width, 3, seed, 0, width).detach().cpu())
    return orthonormal_span(vectors, rank, scale=1.0)


def _history_indices(transition: int, rank: int, *, static: bool) -> list[int]:
    if transition < rank:
        raise ValueError("target transition has insufficient causal history")
    return list(range(rank)) if static else list(range(transition - rank, transition))


def _stack_history(
    chords: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    indices: list[int],
    item: int,
    device: str,
) -> torch.Tensor:
    return torch.stack([chords[index][item] for index in indices]).to(device=device)


def reconstruct_family(
    left: LayerState,
    target_chord: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    chords: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    transition: int,
    family: str,
    method: str,
    layer: int,
    ridge_ratio: float,
    device: str,
) -> tuple[LayerState, dict[str, float], dict[str, Any]]:
    if family == "coupled_four":
        rank = 4
    elif family == "separate_three_plus_three":
        rank = 3
    else:
        raise ValueError(f"unknown coordinate family: {family}")
    if method not in {"fixed_structured", "static", "moving"}:
        raise ValueError(f"unknown method: {method}")

    target_router, target_fc, target_proj = (value.to(device=device) for value in target_chord)
    static = method == "static"
    history = _history_indices(transition, rank, static=static)
    if method == "fixed_structured":
        history = []

    if family == "coupled_four":
        target_pair = torch.cat((target_fc, target_proj), dim=-1)
        if method == "fixed_structured":
            pair_basis = fixed_local_basis(
                target_pair.shape[-1], rank, 20260901 + 1009 * layer, device
            ).to(device)[:, None, None, :]
            router_basis = fixed_local_basis(
                target_router.shape[-1], rank, 20260903 + 1009 * layer, device
            ).to(device)[:, None, :]
        else:
            fc_basis = _stack_history(chords, history, 1, device)
            proj_basis = _stack_history(chords, history, 2, device)
            pair_basis = torch.cat((fc_basis, proj_basis), dim=-1)
            router_basis = _stack_history(chords, history, 0, device)
        projected_pair, pair_coordinates = project_rows(target_pair, pair_basis, ridge_ratio)
        projected_fc, projected_proj = projected_pair.split(target_fc.shape[-1], dim=-1)
        projected_router, router_coordinates = project_rows(target_router, router_basis, ridge_ratio)
        coordinate_stats = {
            "pair_coordinate_l2_mean": float(pair_coordinates.norm(dim=-1).mean()),
            "pair_coordinate_l2_max": float(pair_coordinates.norm(dim=-1).max()),
            "router_coordinate_l2_mean": float(router_coordinates.norm(dim=-1).mean()),
        }
    else:
        if method == "fixed_structured":
            fc_basis = fixed_local_basis(
                target_fc.shape[-1], rank, 20260905 + 1009 * layer, device
            ).to(device)[:, None, None, :]
            proj_basis = fixed_local_basis(
                target_proj.shape[-1], rank, 20260907 + 1009 * layer, device
            ).to(device)[:, None, None, :]
            router_basis = fixed_local_basis(
                target_router.shape[-1], rank, 20260909 + 1009 * layer, device
            ).to(device)[:, None, :]
        else:
            fc_basis = _stack_history(chords, history, 1, device)
            proj_basis = _stack_history(chords, history, 2, device)
            router_basis = _stack_history(chords, history, 0, device)
        projected_fc, fc_coordinates = project_rows(target_fc, fc_basis, ridge_ratio)
        projected_proj, proj_coordinates = project_rows(target_proj, proj_basis, ridge_ratio)
        projected_router, router_coordinates = project_rows(target_router, router_basis, ridge_ratio)
        coordinate_stats = {
            "fc_coordinate_l2_mean": float(fc_coordinates.norm(dim=-1).mean()),
            "fc_coordinate_l2_max": float(fc_coordinates.norm(dim=-1).max()),
            "proj_coordinate_l2_mean": float(proj_coordinates.norm(dim=-1).mean()),
            "proj_coordinate_l2_max": float(proj_coordinates.norm(dim=-1).max()),
            "router_coordinate_l2_mean": float(router_coordinates.norm(dim=-1).mean()),
        }

    left_router, left_fc, left_proj = (value.to(device=device) for value in _atom_views(left))
    reconstructed = _state_from_atom_views(
        left_router + projected_router,
        left_fc + projected_fc,
        left_proj + projected_proj,
    )
    recoveries = {
        "router_parameter_recovery": energy_recovery(projected_router, target_router),
        "c_fc_parameter_recovery": energy_recovery(projected_fc, target_fc),
        "c_proj_parameter_recovery": energy_recovery(projected_proj, target_proj),
        "paired_parameter_recovery": energy_recovery(
            torch.cat((projected_fc, projected_proj), dim=-1),
            torch.cat((target_fc, target_proj), dim=-1),
        ),
    }
    return reconstructed, recoveries, {"basis_transition_indices": history, **coordinate_stats}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    keys = sorted({(row["family"], row["method"]) for row in rows})
    for family, method in keys:
        selected = [row for row in rows if row["family"] == family and row["method"] == method]
        exact = [float(row["heldout_exact_recovery"]) for row in selected]
        summary[f"{family}:{method}"] = {
            "mean": sum(exact) / len(exact),
            "minimum": min(exact),
            "maximum": max(exact),
            "by_layer": {
                str(layer): sum(float(row["heldout_exact_recovery"]) for row in selected if row["layer"] == layer)
                / sum(row["layer"] == layer for row in selected)
                for layer in sorted({int(row["layer"]) for row in selected})
            },
            "paired_parameter_recovery_mean": sum(float(row["paired_parameter_recovery"]) for row in selected)
            / len(selected),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--terminal-checkpoint", required=True, type=Path)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ridge-ratio", type=float, default=1e-6)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text())
    causal = plan["causal_basis"]
    evaluation = plan["evaluation"]
    layers = [int(value) for value in causal["layers"]]
    steps = [int(value) for value in causal["trajectory_steps"]]
    discovery_transitions = int(causal["chronological_discovery_transitions"])
    if len(steps) - 1 != discovery_transitions + int(causal["chronological_heldout_transitions"]):
        raise ValueError("registered transition split does not cover the trajectory")
    if "cuda" in args.device and _load_block_fht_ext() is None:
        raise RuntimeError("native BlockFHT extension is required for the fixed control")

    discovery_batches = fixed_validation_batches(
        args.data_dir,
        int(evaluation["batch_size"]),
        int(evaluation["block_size"]),
        int(evaluation["batches"]),
        int(evaluation["activation_discovery_seed"]),
    )
    heldout_batches = fixed_validation_batches(
        args.data_dir,
        int(evaluation["batch_size"]),
        int(evaluation["block_size"]),
        int(evaluation["batches"]),
        int(evaluation["activation_heldout_seed"]),
    )
    model = load_model(args.terminal_checkpoint, args.device)
    try:
        discovery_inputs = collect_inputs(
            model, discovery_batches, layers,
            int(evaluation["activation_sample_cap_per_layer"]), args.device,
        )
        heldout_inputs = collect_inputs(
            model, heldout_batches, layers,
            int(evaluation["activation_sample_cap_per_layer"]), args.device,
        )
        top_k = int(model.config.moe_top_k)
    finally:
        del model
        if "cuda" in args.device:
            torch.cuda.empty_cache()

    rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    for layer in layers:
        states = [
            load_layer_state(args.snapshot_dir / f"step_{step:06d}.pt", layer, step)
            for step in steps
        ]
        states, aligned = align_layer_sequence(
            states,
            discovery_inputs[layer].to(args.device),
            heldout_inputs[layer].to(args.device),
            args.device,
        )
        for row in aligned:
            row["layer"] = layer
        alignment_rows.extend(aligned)
        chords = [_state_chord(right, left) for left, right in zip(states[:-1], states[1:])]
        heldout = heldout_inputs[layer].to(args.device)
        for transition in range(discovery_transitions, len(chords)):
            left = states[transition].to(args.device)
            right = states[transition + 1].to(args.device)
            base_output = sparse_moe_output(left, heldout, top_k)
            target_output = sparse_moe_output(right, heldout, top_k) - base_output
            for family in ("coupled_four", "separate_three_plus_three"):
                for method in ("fixed_structured", "static", "moving"):
                    reconstructed, parameter_recoveries, metadata = reconstruct_family(
                        states[transition], chords[transition], chords, transition,
                        family, method, layer, args.ridge_ratio, args.device,
                    )
                    prediction = sparse_moe_output(reconstructed, heldout, top_k) - base_output
                    rows.append(
                        {
                            "family": family,
                            "method": method,
                            "layer": layer,
                            "start_step": steps[transition],
                            "end_step": steps[transition + 1],
                            "heldout_exact_recovery": recovery_fraction(prediction, target_output),
                            "target_output_energy": float(target_output.square().sum()),
                            **parameter_recoveries,
                            **metadata,
                        }
                    )

    summary = summarize(rows)
    assignment_minimum = min(float(row["assignment_overlap"]) for row in alignment_rows)
    gates = plan["frozen_gates"]
    family_gates: dict[str, Any] = {}
    for family in ("coupled_four", "separate_three_plus_three"):
        moving = summary[f"{family}:moving"]
        static = summary[f"{family}:static"]
        fixed = summary[f"{family}:fixed_structured"]
        family_gates[family] = {
            "assignment_overlap_pass": assignment_minimum >= float(gates["assignment_overlap_min"]),
            "heldout_exact_mean_pass": moving["mean"] >= float(gates["heldout_exact_recovery_mean_min"]),
            "heldout_exact_every_layer_pass": min(moving["by_layer"].values())
            >= float(gates["heldout_exact_recovery_every_layer_min"]),
            "moving_minus_static_pass": moving["mean"] - static["mean"]
            >= float(gates["moving_minus_static_mean_min"]),
            "moving_minus_fixed_pass": moving["mean"] - fixed["mean"]
            >= float(gates["moving_minus_fixed_structured_mean_min"]),
        }
        family_gates[family]["all_pass"] = all(family_gates[family].values())
    finite = all(
        math.isfinite(float(row[key]))
        for row in rows
        for key in (
            "heldout_exact_recovery", "target_output_energy",
            "router_parameter_recovery", "c_fc_parameter_recovery",
            "c_proj_parameter_recovery", "paired_parameter_recovery",
        )
    )

    args.output.mkdir(parents=True, exist_ok=True)
    rows_path = args.output / "paired_atom_rows.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    alignment_path = args.output / "paired_atom_alignment_rows.csv"
    with alignment_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(alignment_rows[0]))
        writer.writeheader()
        writer.writerows(alignment_rows)
    result = {
        "schema_version": "nanogpt_sparse_moe_paired_atom_oracle_result_v1",
        "plan": {"path": str(args.plan), "sha256": file_sha256(args.plan)},
        "terminal_checkpoint": {
            "path": str(args.terminal_checkpoint),
            "sha256": file_sha256(args.terminal_checkpoint),
        },
        "snapshots": {
            str(step): file_sha256(args.snapshot_dir / f"step_{step:06d}.pt") for step in steps
        },
        "activation_frames": {
            "discovery_seed": int(evaluation["activation_discovery_seed"]),
            "heldout_seed": int(evaluation["activation_heldout_seed"]),
            "discovery_tokens_sha256": tensor_sha256(torch.cat(discovery_batches)),
            "heldout_tokens_sha256": tensor_sha256(torch.cat(heldout_batches)),
        },
        "assignment_overlap_minimum": assignment_minimum,
        "summary": summary,
        "gates": {"families": family_gates, "all_values_finite": finite},
        "rows": {"path": str(rows_path), "sha256": file_sha256(rows_path)},
        "alignment_rows": {
            "path": str(alignment_path), "sha256": file_sha256(alignment_path)
        },
        "source_sha256": file_sha256(Path(__file__)),
    }
    result_path = args.output / "paired_atom_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
