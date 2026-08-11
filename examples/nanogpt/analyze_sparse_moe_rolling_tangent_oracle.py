#!/usr/bin/env python3
"""Score causal moving sparse-MoE charts without performing an update.

The oracle separates image capacity from coordinate prediction.  It builds
four-coordinate layer directions from only causally available parameter
chords, fits coordinates on one fixed activation split, materializes the
corresponding expert/router update, and scores it on a disjoint split.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from examples.nanogpt.analyze_cproj_manifold import load_model
from examples.nanogpt.analyze_residual_compatibility import fixed_validation_batches
from examples.nanogpt.analyze_sparse_moe_paired_alignment import (
    collect_inputs,
    file_sha256,
    tensor_sha256,
)
from examples.nanogpt.moe_paired_geometry import (
    apply_neuron_permutation,
    functional_atom_similarity,
    maximum_weight_assignment,
)
from examples.nanogpt.parameter_trajectory import SCHEMA_VERSION
from latent_weight_lab.block_fht import _load_block_fht_ext, block_fht_slice


@dataclass
class LayerState:
    router: torch.Tensor
    c_fc: torch.Tensor
    c_proj: torch.Tensor

    def to(self, device: str, dtype: torch.dtype = torch.float32) -> "LayerState":
        return LayerState(
            self.router.to(device=device, dtype=dtype),
            self.c_fc.to(device=device, dtype=dtype),
            self.c_proj.to(device=device, dtype=dtype),
        )


def flatten_state(state: LayerState) -> torch.Tensor:
    return torch.cat(
        (state.router.reshape(-1), state.c_fc.reshape(-1), state.c_proj.reshape(-1))
    )


def unflatten_state(vector: torch.Tensor, template: LayerState) -> LayerState:
    offset = 0
    values = []
    for reference in (template.router, template.c_fc, template.c_proj):
        count = reference.numel()
        values.append(vector[offset : offset + count].reshape(reference.shape))
        offset += count
    if offset != vector.numel():
        raise ValueError("layer vector length does not match the state template")
    return LayerState(*values)


def add_direction(state: LayerState, direction: LayerState) -> LayerState:
    return LayerState(
        state.router + direction.router,
        state.c_fc + direction.c_fc,
        state.c_proj + direction.c_proj,
    )


def subtract_states(right: LayerState, left: LayerState) -> torch.Tensor:
    return flatten_state(right).float() - flatten_state(left).float()


def sparse_moe_output(
    state: LayerState,
    activations: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Evaluate the exact bias-free complete-expert MoE on a flat frame."""
    x = activations.float()
    router = state.router.float()
    c_fc = state.c_fc.float()
    c_proj = state.c_proj.float()
    logits = x @ router.T
    tie = torch.arange(logits.shape[-1], device=x.device, dtype=logits.dtype)
    selected = torch.topk(
        logits - tie * torch.finfo(logits.dtype).eps,
        top_k,
        dim=-1,
        largest=True,
        sorted=True,
    ).indices
    probabilities = F.softmax(logits.gather(-1, selected), dim=-1)
    hidden = F.gelu(torch.einsum("nd,ehd->enh", x, c_fc))
    expert_outputs = torch.einsum("enh,edh->end", hidden, c_proj).permute(1, 0, 2)
    gathered = expert_outputs.gather(
        1,
        selected.unsqueeze(-1).expand(-1, -1, expert_outputs.shape[-1]),
    )
    return (gathered * probabilities.unsqueeze(-1)).sum(dim=1)


def causal_history_indices(target_transition: int, budget: int) -> list[int]:
    if target_transition < 0 or budget < 1:
        raise ValueError("target transition must be non-negative and budget positive")
    return list(range(max(0, target_transition - budget), target_transition))


def _normalized_rows(vectors: Iterable[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    stacked = torch.stack([value.float().reshape(-1) for value in vectors])
    norms = stacked.norm(dim=1)
    if bool((norms <= 0).any()):
        raise ValueError("zero trajectory chord cannot define a tangent chart")
    return stacked / norms[:, None], norms


def orthonormal_span(
    vectors: Iterable[torch.Tensor],
    rank: int,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Return a stable origin-through PCA span without a dense P-by-P SVD."""
    rows, norms = _normalized_rows(vectors)
    gram = rows @ rows.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    keep = torch.nonzero(eigenvalues > 1e-10, as_tuple=False).flatten()
    if not len(keep):
        raise ValueError("trajectory chord Gram matrix is numerically empty")
    keep = keep[-min(int(rank), len(keep)) :].flip(0)
    values = eigenvalues.index_select(0, keep)
    coefficients = eigenvectors.index_select(1, keep).T
    basis = (coefficients @ rows) / values.sqrt()[:, None]
    target_scale = float(norms.median()) if scale is None else float(scale)
    return basis * target_scale


def fit_coordinates(
    basis_outputs: torch.Tensor,
    target_output: torch.Tensor,
    ridge_ratio: float,
) -> torch.Tensor:
    design = basis_outputs.reshape(basis_outputs.shape[0], -1).float()
    target = target_output.reshape(-1).float()
    gram = design @ design.T
    ridge = float(ridge_ratio) * float(gram.diag().mean().clamp_min(1e-30))
    return torch.linalg.solve(
        gram + torch.eye(gram.shape[0], device=gram.device) * ridge,
        design @ target,
    )


def recovery_fraction(predicted: torch.Tensor, target: torch.Tensor) -> float:
    denominator = target.float().square().sum().clamp_min(1e-30)
    return float(1.0 - (predicted.float() - target.float()).square().sum() / denominator)


def load_layer_state(path: Path, layer: int, expected_step: int) -> LayerState:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"trajectory schema mismatch: {path}")
    if int(payload.get("step", -1)) != expected_step:
        raise ValueError(f"trajectory step mismatch: {path}")
    if payload.get("layers") is None or layer not in payload["layers"]:
        raise ValueError(f"trajectory layer mismatch: {path}")
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError(f"trajectory payload has no parameters: {path}")
    prefix = f"transformer.h.{layer}.mlp."
    return LayerState(
        parameters[prefix + "router.weight"],
        parameters[prefix + "expert_c_fc"],
        parameters[prefix + "expert_c_proj"],
    )


def align_layer_sequence(
    states: list[LayerState],
    discovery_activations: torch.Tensor,
    heldout_activations: torch.Tensor,
    device: str,
) -> tuple[list[LayerState], list[dict[str, Any]]]:
    """Sequentially quotient only stable paired hidden-neuron permutations."""
    aligned = [states[0]]
    rows: list[dict[str, Any]] = []
    for transition, candidate in enumerate(states[1:]):
        left = aligned[-1].to(device)
        right = candidate.to(device)
        next_fc = right.c_fc.clone()
        next_proj = right.c_proj.clone()
        for expert in range(left.c_fc.shape[0]):
            fit_similarity = functional_atom_similarity(
                left.c_fc[expert], left.c_proj[expert],
                right.c_fc[expert], right.c_proj[expert], discovery_activations,
            )
            eval_similarity = functional_atom_similarity(
                left.c_fc[expert], left.c_proj[expert],
                right.c_fc[expert], right.c_proj[expert], heldout_activations,
            )
            fit_permutation = maximum_weight_assignment(fit_similarity)
            eval_permutation = maximum_weight_assignment(eval_similarity)
            next_fc[expert], next_proj[expert] = apply_neuron_permutation(
                right.c_fc[expert], right.c_proj[expert], fit_permutation
            )
            identity = torch.arange(fit_permutation.numel(), device=fit_permutation.device)
            rows.append(
                {
                    "transition": transition,
                    "expert": expert,
                    "assignment_overlap": float((fit_permutation == eval_permutation).float().mean()),
                    "identity_fraction": float((fit_permutation == identity).float().mean()),
                }
            )
        aligned.append(
            LayerState(
                right.router.detach().cpu().to(states[0].router.dtype),
                next_fc.detach().cpu().to(states[0].c_fc.dtype),
                next_proj.detach().cpu().to(states[0].c_proj.dtype),
            )
        )
    return aligned, rows


def fixed_blockfht_basis(
    template: LayerState,
    rank: int,
    scale: float,
    layer: int,
    device: str,
) -> torch.Tensor:
    pieces: list[list[torch.Tensor]] = [[] for _ in range(rank)]
    seeds = (20260823 + layer * 1009, 20260829 + layer * 1009, 20260831 + layer * 1009)
    for reference, seed in zip((template.router, template.c_fc, template.c_proj), seeds):
        size = reference.numel()
        for coordinate in range(rank):
            latent = torch.zeros(rank, device=device)
            latent[coordinate] = 1.0
            pieces[coordinate].append(
                block_fht_slice(latent, size, 3, seed, 0, size).detach().cpu()
            )
    vectors = [torch.cat(parts) for parts in pieces]
    return orthonormal_span(vectors, rank, scale=scale)


def basis_output_directions(
    left: LayerState,
    basis: torch.Tensor,
    activations: torch.Tensor,
    top_k: int,
    device: str,
) -> torch.Tensor:
    live = left.to(device)
    base = sparse_moe_output(live, activations, top_k)
    outputs = []
    for vector in basis:
        direction = unflatten_state(vector.to(device), live)
        outputs.append(sparse_moe_output(add_direction(live, direction), activations, top_k) - base)
    return torch.stack(outputs)


def score_basis(
    left: LayerState,
    right: LayerState,
    basis: torch.Tensor,
    discovery: torch.Tensor,
    heldout: torch.Tensor,
    top_k: int,
    ridge_ratio: float,
    device: str,
) -> dict[str, Any]:
    live_left = left.to(device)
    live_right = right.to(device)
    discovery_target = sparse_moe_output(live_right, discovery, top_k) - sparse_moe_output(live_left, discovery, top_k)
    heldout_target = sparse_moe_output(live_right, heldout, top_k) - sparse_moe_output(live_left, heldout, top_k)
    discovery_directions = basis_output_directions(left, basis, discovery, top_k, device)
    coordinates = fit_coordinates(discovery_directions, discovery_target, ridge_ratio)
    heldout_directions = basis_output_directions(left, basis, heldout, top_k, device)
    tangent_prediction = torch.einsum("k,knd->nd", coordinates, heldout_directions)
    combined = torch.einsum("k,kp->p", coordinates.cpu(), basis)
    materialized = add_direction(live_left, unflatten_state(combined.to(device), live_left))
    materialized_prediction = sparse_moe_output(materialized, heldout, top_k) - sparse_moe_output(live_left, heldout, top_k)
    return {
        "coordinates": [float(value) for value in coordinates],
        "heldout_tangent_recovery": recovery_fraction(tangent_prediction, heldout_target),
        "heldout_materialized_recovery": recovery_fraction(materialized_prediction, heldout_target),
        "target_energy": float(heldout_target.float().square().sum()),
    }


def summarize(rows: list[dict[str, Any]], methods: list[str], layers: list[int]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        recoveries = [float(row["heldout_materialized_recovery"]) for row in selected]
        result[method] = {
            "mean": sum(recoveries) / len(recoveries),
            "minimum": min(recoveries),
            "maximum": max(recoveries),
            "by_layer": {
                str(layer): sum(
                    float(row["heldout_materialized_recovery"])
                    for row in selected if row["layer"] == layer
                ) / sum(row["layer"] == layer for row in selected)
                for layer in layers
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--terminal-checkpoint", required=True, type=Path)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--sample-cap", type=int, default=2048)
    parser.add_argument("--ridge-ratio", type=float, default=1e-6)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text())
    source = plan["source_run"]
    anti_leakage = plan["anti_leakage"]
    layers = [int(value) for value in source["layers"]]
    steps = [int(value) for value in source["trajectory_steps"]]
    rank = int(plan["matched_coordinate_budget"]["coordinates_per_probed_layer"])
    if "cuda" in args.device and _load_block_fht_ext() is None:
        raise RuntimeError("native BlockFHT extension is required for the fixed control")
    discovery_transition_count = int(anti_leakage["trajectory_discovery_transitions"])
    if len(steps) - 1 != discovery_transition_count + int(anti_leakage["trajectory_heldout_transitions"]):
        raise ValueError("trajectory split does not cover every registered transition")
    if int(anti_leakage["chronological_boundary_step"]) != steps[discovery_transition_count]:
        raise ValueError("chronological boundary does not match the registered steps")

    discovery_batches = fixed_validation_batches(
        args.data_dir, args.batch_size, args.block_size, args.batches,
        int(anti_leakage["activation_discovery_seed"]),
    )
    heldout_batches = fixed_validation_batches(
        args.data_dir, args.batch_size, args.block_size, args.batches,
        int(anti_leakage["activation_heldout_seed"]),
    )
    model = load_model(args.terminal_checkpoint, args.device)
    try:
        discovery_inputs = collect_inputs(model, discovery_batches, layers, args.sample_cap, args.device)
        heldout_inputs = collect_inputs(model, heldout_batches, layers, args.sample_cap, args.device)
        top_k = int(model.config.moe_top_k)
    finally:
        del model
        if "cuda" in args.device:
            torch.cuda.empty_cache()

    rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    snapshot_hashes = {
        str(step): file_sha256(args.snapshot_dir / f"step_{step:06d}.pt")
        for step in steps
    }
    methods = ["fixed_blockfht", "static_paired_chart", "state_conditioned_paired_chart"]
    for layer in layers:
        states = [
            load_layer_state(args.snapshot_dir / f"step_{step:06d}.pt", layer, step)
            for step in steps
        ]
        discovery = discovery_inputs[layer].to(args.device)
        heldout = heldout_inputs[layer].to(args.device)
        states, layer_alignment = align_layer_sequence(states, discovery, heldout, args.device)
        for row in layer_alignment:
            row["layer"] = layer
        alignment_rows.extend(layer_alignment)
        chords = [subtract_states(right, left) for left, right in zip(states[:-1], states[1:])]
        discovery_chords = chords[:discovery_transition_count]
        chord_scale = float(torch.tensor([value.norm() for value in discovery_chords]).median())
        static_basis = orthonormal_span(discovery_chords, rank, scale=chord_scale)
        fixed_basis = fixed_blockfht_basis(states[0], rank, chord_scale, layer, args.device)
        for transition in range(discovery_transition_count, len(chords)):
            history = causal_history_indices(transition, rank)
            if len(history) != rank or max(history) >= transition:
                raise AssertionError("state-conditioned history leaked the target transition")
            moving_basis = orthonormal_span(
                [chords[index] for index in history], rank,
                scale=float(torch.tensor([chords[index].norm() for index in history]).median()),
            )
            for method, basis in (
                ("fixed_blockfht", fixed_basis),
                ("static_paired_chart", static_basis),
                ("state_conditioned_paired_chart", moving_basis),
            ):
                result = score_basis(
                    states[transition], states[transition + 1], basis,
                    discovery, heldout, top_k, args.ridge_ratio, args.device,
                )
                rows.append(
                    {
                        "method": method,
                        "layer": layer,
                        "start_step": steps[transition],
                        "end_step": steps[transition + 1],
                        "basis_transition_indices": history if method == "state_conditioned_paired_chart" else list(range(discovery_transition_count)) if method == "static_paired_chart" else [],
                        **result,
                    }
                )

    summary = summarize(rows, methods, layers)
    gates = plan["frozen_gates"]
    state_summary = summary["state_conditioned_paired_chart"]
    static_mean = summary["static_paired_chart"]["mean"]
    fixed_mean = summary["fixed_blockfht"]["mean"]
    assignment_minimum = min(float(row["assignment_overlap"]) for row in alignment_rows)
    finite = all(
        math.isfinite(float(row[key]))
        for row in rows
        for key in ("heldout_tangent_recovery", "heldout_materialized_recovery", "target_energy")
    )
    gate_results = {
        "assignment_overlap_pass": assignment_minimum >= float(gates["discovery_eval_assignment_overlap_min"]),
        "state_conditioned_mean_pass": state_summary["mean"] >= float(gates["heldout_residual_output_recovery_mean_min"]),
        "state_conditioned_every_layer_pass": min(state_summary["by_layer"].values()) >= float(gates["heldout_residual_output_recovery_every_probed_layer_min"]),
        "state_minus_static_pass": state_summary["mean"] - static_mean >= float(gates["state_conditioned_minus_static_recovery_min"]),
        "state_minus_fixed_blockfht_pass": state_summary["mean"] - fixed_mean >= float(gates["state_conditioned_minus_fixed_blockfht_recovery_min"]),
        "all_values_finite": finite,
    }
    gate_results["all_pass"] = all(gate_results.values())

    args.output.mkdir(parents=True, exist_ok=True)
    rows_path = args.output / "rolling_tangent_rows.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    alignment_path = args.output / "rolling_tangent_alignment_rows.csv"
    with alignment_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(alignment_rows[0]))
        writer.writeheader()
        writer.writerows(alignment_rows)
    result = {
        "schema_version": "nanogpt_sparse_moe_rolling_tangent_oracle_result_v1",
        "plan": {"path": str(args.plan), "sha256": file_sha256(args.plan)},
        "terminal_checkpoint": {"path": str(args.terminal_checkpoint), "sha256": file_sha256(args.terminal_checkpoint)},
        "snapshots": snapshot_hashes,
        "activation_frames": {
            "discovery_seed": int(anti_leakage["activation_discovery_seed"]),
            "heldout_seed": int(anti_leakage["activation_heldout_seed"]),
            "discovery_tokens_sha256": tensor_sha256(torch.cat(discovery_batches)),
            "heldout_tokens_sha256": tensor_sha256(torch.cat(heldout_batches)),
        },
        "summary": summary,
        "assignment_overlap_minimum": assignment_minimum,
        "gates": gate_results,
        "rows": {"path": str(rows_path), "sha256": file_sha256(rows_path)},
        "alignment_rows": {"path": str(alignment_path), "sha256": file_sha256(alignment_path)},
        "source_sha256": file_sha256(Path(__file__)),
    }
    result_path = args.output / "rolling_tangent_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
