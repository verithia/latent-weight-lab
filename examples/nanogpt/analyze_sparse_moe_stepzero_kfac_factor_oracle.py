#!/usr/bin/env python3
"""Score expert-conditioned activation/error KFAC factor directions."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from examples.nanogpt.analyze_mlp_activation_update_alignment import git_commit
from examples.nanogpt.analyze_residual_compatibility import fixed_validation_batches
from examples.nanogpt.analyze_sparse_moe_paired_alignment import (
    collect_inputs,
    file_sha256,
)
from examples.nanogpt.analyze_sparse_moe_paired_atom_oracle import (
    _state_chord,
    reconstruct_family,
    union_fieldnames,
)
from examples.nanogpt.analyze_sparse_moe_rolling_tangent_oracle import (
    LayerState,
    recovery_fraction,
    sparse_moe_output,
)
from examples.nanogpt.analyze_sparse_moe_stepzero_task_gradient_oracle import (
    FAMILIES,
    all_finite,
    family_overlap,
    layer_state_from_mapping,
    load_terminal_snapshot,
    model_from_exact_stepzero,
    reconstruct_gradient_family,
    selected_stepzero_hashes,
)


PLAN_SCHEMA = "nanogpt_sparse_moe_stepzero_kfac_factor_oracle_plan_v1"


class SparseMLPGeometryCollector:
    def __init__(self, model: torch.nn.Module, layers: list[int]) -> None:
        self.layers = set(layers)
        self.inputs: dict[int, list[torch.Tensor]] = defaultdict(list)
        self.errors: dict[int, list[torch.Tensor]] = defaultdict(list)
        self.handles = []
        for layer, block in enumerate(model.transformer.h):
            if layer in self.layers:
                self.handles.append(block.mlp.register_forward_hook(self._hook(layer)))

    def _hook(self, layer: int):
        def hook(_module, inputs, output):
            values = inputs[0]
            self.inputs[layer].append(
                values.detach().float().reshape(-1, values.shape[-1]).cpu()
            )

            def save_error(gradient: torch.Tensor) -> None:
                self.errors[layer].append(
                    gradient.detach().float().reshape(-1, gradient.shape[-1]).cpu()
                )

            output.register_hook(save_error)

        return hook

    def tensors(self) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        if set(self.inputs) != self.layers or set(self.errors) != self.layers:
            raise RuntimeError("sparse MLP geometry collection is incomplete")
        inputs = {layer: torch.cat(self.inputs[layer]) for layer in sorted(self.layers)}
        errors = {layer: torch.cat(self.errors[layer]) for layer in sorted(self.layers)}
        if any(inputs[layer].shape != errors[layer].shape for layer in self.layers):
            raise RuntimeError("sparse MLP input/error shapes disagree")
        return inputs, errors

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def collect_geometry(
    model: torch.nn.Module,
    batches: list[torch.Tensor],
    layers: list[int],
    device: str,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], float]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for layer in layers:
        mlp = model.transformer.h[layer].mlp
        mlp.router.weight.requires_grad_(True)
        mlp.expert_c_fc.requires_grad_(True)
        mlp.expert_c_proj.requires_grad_(True)
    model.train()
    model.zero_grad(set_to_none=True)
    collector = SparseMLPGeometryCollector(model, layers)
    losses = []
    try:
        for tokens in batches:
            tokens = tokens.to(device)
            inputs = tokens[:, :-1].contiguous()
            targets = tokens[:, 1:].contiguous()
            with torch.autocast(
                device_type="cuda" if device.startswith("cuda") else "cpu",
                dtype=torch.bfloat16,
                enabled=device.startswith("cuda"),
            ):
                _logits, loss = model(inputs, targets)
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("non-finite KFAC discovery loss")
            losses.append(float(loss.detach()))
            (loss / len(batches)).backward()
        inputs, errors = collector.tensors()
        return inputs, errors, sum(losses) / len(losses)
    finally:
        collector.close()
        model.zero_grad(set_to_none=True)
        model.eval()


def gelu_derivative(values: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(values / math.sqrt(2.0))) + (
        values * torch.exp(-0.5 * values.square()) / math.sqrt(2.0 * math.pi)
    )


def weighted_top_eigenbasis(
    rows: torch.Tensor,
    weights: torch.Tensor,
    rank: int,
    ridge_ratio: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if rows.ndim != 2 or weights.ndim != 1 or rows.shape[0] != weights.numel():
        raise ValueError("weighted covariance inputs disagree")
    positive = weights.float().clamp_min(0.0)
    denominator = positive.sum().clamp_min(1e-30)
    covariance = rows.float().T @ (positive[:, None] * rows.float()) / denominator
    covariance = 0.5 * (covariance + covariance.T)
    trace = covariance.diagonal().sum().clamp_min(1e-30)
    covariance = covariance + (
        float(ridge_ratio) * trace / covariance.shape[0]
    ) * torch.eye(covariance.shape[0], device=covariance.device)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    selected_values = eigenvalues[-rank:].flip(0)
    selected_vectors = eigenvectors[:, -rank:].flip(1).contiguous()
    return selected_vectors, {
        "weight_sum": float(denominator),
        "trace": float(trace),
        "top_rank_energy_fraction": float(selected_values.sum() / trace),
        "largest_eigenvalue": float(selected_values[0]),
        "smallest_selected_eigenvalue": float(selected_values[-1]),
    }


def route_assignments(
    state: LayerState, inputs: torch.Tensor, top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = inputs.float() @ state.router.float().T
    tie = torch.arange(logits.shape[-1], device=logits.device, dtype=logits.dtype)
    indices = torch.topk(
        logits - tie * torch.finfo(logits.dtype).eps,
        top_k,
        dim=-1,
        sorted=True,
    ).indices
    probabilities = F.softmax(logits.gather(-1, indices), dim=-1)
    return indices, probabilities


def build_kfac_basis(
    state: LayerState,
    inputs: torch.Tensor,
    errors: torch.Tensor,
    rank: int,
    ridge_ratio: float,
    minimum_assignments: int,
    device: str,
) -> tuple[list[LayerState], list[dict[str, Any]]]:
    state = state.to(device)
    inputs = inputs.to(device).float()
    errors = errors.to(device).float()
    indices, probabilities = route_assignments(state, inputs, top_k=2)
    experts, hidden, width = state.c_fc.shape
    router_basis = torch.zeros(rank, experts, width, device=device)
    fc_basis = torch.zeros(rank, experts, hidden, width, device=device)
    proj_basis = torch.zeros(rank, experts, width, hidden, device=device)
    rows: list[dict[str, Any]] = []
    for expert in range(experts):
        locations = (indices == expert).nonzero(as_tuple=False)
        if locations.shape[0] == 0:
            raise RuntimeError(
                f"expert {expert} has no assignments; covariance is undefined"
            )
        token = locations[:, 0]
        slot = locations[:, 1]
        x = inputs.index_select(0, token)
        p = probabilities[token, slot]
        output_error = errors.index_select(0, token)
        routed_error = p[:, None] * output_error
        pre = x @ state.c_fc[expert].T
        activation = F.gelu(pre)
        hidden_error = (routed_error @ state.c_proj[expert]) * gelu_derivative(pre)
        incoming_weight = hidden_error.square().mean(dim=-1)
        outgoing_weight = activation.square().mean(dim=-1)
        router_weight = p.square() * output_error.square().mean(dim=-1)
        incoming, incoming_stats = weighted_top_eigenbasis(
            x, incoming_weight, rank, ridge_ratio
        )
        outgoing, outgoing_stats = weighted_top_eigenbasis(
            routed_error, outgoing_weight, rank, ridge_ratio
        )
        router, router_stats = weighted_top_eigenbasis(
            x, router_weight, rank, ridge_ratio
        )
        for coordinate in range(rank):
            router_basis[coordinate, expert] = router[:, coordinate]
            fc_basis[coordinate, expert] = incoming[:, coordinate].expand(hidden, -1)
            proj_basis[coordinate, expert] = outgoing[:, coordinate, None].expand(-1, hidden)
        rows.append(
            {
                "expert": expert,
                "assignments": int(locations.shape[0]),
                **{f"incoming_{key}": value for key, value in incoming_stats.items()},
                **{f"outgoing_{key}": value for key, value in outgoing_stats.items()},
                **{f"router_{key}": value for key, value in router_stats.items()},
            }
        )
    bank = [
        LayerState(router_basis[k].cpu(), fc_basis[k].cpu(), proj_basis[k].cpu())
        for k in range(rank)
    ]
    return bank, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--terminal-snapshot", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ridge-ratio", type=float, default=1e-6)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("KFAC oracle plan schema mismatch")
    payload = load_terminal_snapshot(args.terminal_snapshot)
    causal = plan["causal_source"]
    if file_sha256(args.terminal_snapshot) != causal["terminal_manifold_snapshot_sha256"]:
        raise ValueError("terminal snapshot hash disagrees with the plan")
    layers = [int(value) for value in causal["layers"]]
    model = model_from_exact_stepzero(payload, int(causal["model_seed"]), args.device)
    stepzero_hashes = selected_stepzero_hashes(model, layers)
    stepzero_mapping = dict(model.named_parameters())
    initial = {
        layer: layer_state_from_mapping(stepzero_mapping, layer) for layer in layers
    }
    terminal = {
        layer: layer_state_from_mapping(payload["model"], layer) for layer in layers
    }
    minimum_assignments = int(plan["frozen_gates"]["minimum_expert_assignments"])
    banks: dict[str, dict[int, list[LayerState]]] = {}
    geometry_rows: list[dict[str, Any]] = []
    for spec in plan["geometry_banks"]:
        batches = fixed_validation_batches(
            args.data_dir,
            int(spec["batch_size"]),
            int(spec["block_size"]) + 1,
            int(spec["batches"]),
            int(spec["seed"]),
        )
        inputs, errors, loss = collect_geometry(model, batches, layers, args.device)
        bank_by_layer = {}
        for layer in layers:
            bank, rows = build_kfac_basis(
                initial[layer],
                inputs[layer],
                errors[layer],
                rank=4,
                ridge_ratio=args.ridge_ratio,
                minimum_assignments=minimum_assignments,
                device=args.device,
            )
            bank_by_layer[layer] = bank
            for row in rows:
                row.update({"bank": spec["name"], "layer": layer, "loss": loss})
            geometry_rows.extend(rows)
        banks[spec["name"]] = bank_by_layer

    evaluation = plan["evaluation"]
    heldout_batches = fixed_validation_batches(
        args.data_dir,
        int(evaluation["heldout_batch_size"]),
        int(evaluation["heldout_block_size"]),
        int(evaluation["heldout_batches"]),
        int(evaluation["heldout_seed"]),
    )
    model.eval()
    heldout_inputs = collect_inputs(
        model,
        heldout_batches,
        layers,
        int(evaluation["activation_sample_cap_per_layer"]),
        args.device,
    )
    del model
    torch.cuda.empty_cache()

    rows: list[dict[str, Any]] = []
    fixed_by_family_layer = {}
    for layer in layers:
        left, right = initial[layer], terminal[layer]
        chord = _state_chord(right, left)
        heldout = heldout_inputs[layer].to(args.device)
        base_output = sparse_moe_output(left.to(args.device), heldout, 2)
        target_output = sparse_moe_output(right.to(args.device), heldout, 2) - base_output
        for family in FAMILIES:
            rank = 4 if family == "coupled_four" else 3
            fixed, fixed_recoveries, _metadata = reconstruct_family(
                left, chord, [], rank, family, "fixed_structured", layer,
                args.ridge_ratio, args.device,
            )
            fixed_exact = recovery_fraction(
                sparse_moe_output(fixed, heldout, 2) - base_output, target_output
            )
            fixed_by_family_layer[(family, layer)] = fixed_exact
            rows.append(
                {"bank": "fixed_structured", "family": family, "layer": layer,
                 "heldout_exact_recovery": fixed_exact, **fixed_recoveries}
            )
            for bank_name, bank in banks.items():
                reconstructed, recoveries = reconstruct_gradient_family(
                    left, chord, bank[layer], family, args.ridge_ratio, args.device
                )
                exact = recovery_fraction(
                    sparse_moe_output(reconstructed, heldout, 2) - base_output,
                    target_output,
                )
                rows.append(
                    {"bank": bank_name, "family": family, "layer": layer,
                     "heldout_exact_recovery": exact, **recoveries}
                )

    overlaps = {
        family: {
            str(layer): family_overlap(
                banks["discovery_a"][layer], banks["discovery_b"][layer], family
            )
            for layer in layers
        }
        for family in FAMILIES
    }
    summary: dict[str, Any] = {}
    gate_results: dict[str, Any] = {}
    gates = plan["frozen_gates"]
    occupancy_by_bank = {
        bank_name: {
            "minimum": min(
                int(row["assignments"])
                for row in geometry_rows
                if row["bank"] == bank_name
            ),
            "by_layer": {
                str(layer): min(
                    int(row["assignments"])
                    for row in geometry_rows
                    if row["bank"] == bank_name and int(row["layer"]) == layer
                )
                for layer in layers
            },
        }
        for bank_name in banks
    }
    for family in FAMILIES:
        fixed_mean = sum(fixed_by_family_layer[(family, layer)] for layer in layers) / len(layers)
        overlap_mean = sum(overlaps[family].values()) / len(layers)
        summary[family] = {
            "fixed_structured_mean": fixed_mean,
            "discovery_bank_subspace_overlap": {
                "mean": overlap_mean, "by_layer": overlaps[family]
            },
        }
        gate_results[family] = {}
        for bank_name in banks:
            selected = [
                row for row in rows
                if row["family"] == family and row["bank"] == bank_name
            ]
            exact = [float(row["heldout_exact_recovery"]) for row in selected]
            raw = float(plan["controls"]["raw_gradient_exact_mean"][family][bank_name])
            bank_summary = {
                "mean": sum(exact) / len(exact),
                "minimum": min(exact),
                "by_layer": {str(row["layer"]): float(row["heldout_exact_recovery"]) for row in selected},
                "paired_parameter_recovery_mean": sum(float(row["paired_parameter_recovery"]) for row in selected) / len(selected),
                "raw_gradient_mean": raw,
                "minimum_expert_assignments": occupancy_by_bank[bank_name]["minimum"],
            }
            bank_summary["kfac_minus_raw_gradient"] = bank_summary["mean"] - raw
            summary[family][bank_name] = bank_summary
            bank_gates = {
                "mean_pass": bank_summary["mean"] >= float(gates["heldout_exact_recovery_mean_min_each_bank"]),
                "every_layer_pass": bank_summary["minimum"] >= float(gates["heldout_exact_recovery_every_layer_min_each_bank"]),
                "minus_raw_pass": bank_summary["kfac_minus_raw_gradient"] >= float(gates["kfac_minus_raw_gradient_mean_min_each_bank"]),
                "bank_overlap_pass": overlap_mean >= float(gates["discovery_bank_mean_subspace_overlap_min"]),
                "minimum_expert_assignments_pass": occupancy_by_bank[bank_name]["minimum"]
                >= minimum_assignments,
            }
            bank_gates["all_pass"] = all(bank_gates.values())
            gate_results[family][bank_name] = bank_gates

    finite = all_finite({"rows": rows, "geometry": geometry_rows, "summary": summary})
    passing = [
        family for family in FAMILIES
        if all(gate_results[family][bank]["all_pass"] for bank in banks)
    ]
    stable = [
        family for family in FAMILIES
        if summary[family]["discovery_bank_subspace_overlap"]["mean"]
        >= float(gates["discovery_bank_mean_subspace_overlap_min"])
        and all(
            occupancy_by_bank[bank_name]["minimum"] >= minimum_assignments
            for bank_name in banks
        )
    ]
    if finite and passing:
        decision = "PASS_BOTH_BANKS_AUTHORIZE_5TPP_ONLINE_KFAC_ACQUISITION_ONLY"
    elif finite and stable:
        decision = "STABLE_BUT_INSUFFICIENT_AUTHORIZE_PER_ATOM_FACTOR_PLAN_ONLY"
    else:
        decision = "REJECT_STATIC_STEPZERO_KFAC_IMAGE_REQUIRE_ONLINE_INPUT_CONDITIONING"

    args.output.mkdir(parents=True, exist_ok=True)
    rows_path = args.output / "kfac_factor_oracle_rows.csv"
    geometry_path = args.output / "kfac_factor_geometry.csv"
    for path, values in ((rows_path, rows), (geometry_path, geometry_rows)):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=union_fieldnames(values))
            writer.writeheader()
            writer.writerows(values)
    result = {
        "schema_version": "nanogpt_sparse_moe_stepzero_kfac_factor_oracle_result_v1",
        "decision": decision,
        "passing_families": passing,
        "stable_families": stable,
        "all_values_finite": finite,
        "summary": summary,
        "gates": gate_results,
        "expert_occupancy": occupancy_by_bank,
        "stepzero_selected_tensor_sha256": stepzero_hashes,
        "source": {
            "terminal_snapshot_sha256": file_sha256(args.terminal_snapshot),
            "plan_sha256": file_sha256(args.plan),
            "dataset_manifest_sha256": causal["dataset_manifest_sha256"],
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
    result_path = args.output / "kfac_factor_oracle_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    status = {
        "state": "finished", "exit_code": 0, "decision": decision,
        "result_sha256": file_sha256(result_path),
        "rows_sha256": file_sha256(rows_path),
        "geometry_sha256": file_sha256(geometry_path),
        "wall_seconds": time.time() - started,
    }
    status_path = args.output / "kfac_factor_oracle_status.json"
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "summary": summary}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
