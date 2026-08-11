#!/usr/bin/env python3
"""Test causal step-zero task-gradient bases for sparse-MoE expert atoms.

The endpoint is used only to fit oracle coordinates.  Candidate directions are
negative task gradients from independent validation microbatches at the exact
scratch initialization, so neither the endpoint nor held-out token frame can
orient the basis.
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
import torch.nn.functional as F

from examples.nanogpt.analyze_mlp_activation_update_alignment import git_commit
from examples.nanogpt.analyze_residual_compatibility import fixed_validation_batches
from examples.nanogpt.analyze_sparse_moe_paired_alignment import (
    collect_inputs,
    file_sha256,
    tensor_sha256,
)
from examples.nanogpt.analyze_sparse_moe_paired_atom_oracle import (
    _atom_views,
    _state_chord,
    _state_from_atom_views,
    energy_recovery,
    project_rows,
    reconstruct_family,
    union_fieldnames,
)
from examples.nanogpt.analyze_sparse_moe_rolling_tangent_oracle import (
    LayerState,
    recovery_fraction,
    sparse_moe_output,
)
from examples.nanogpt.model import GPT, GPTConfig
from examples.nanogpt.muon import muon_update_batched


PLAN_SCHEMA = "nanogpt_sparse_moe_stepzero_task_gradient_oracle_plan_v1"
MUON_PLAN_SCHEMA = "nanogpt_sparse_moe_stepzero_muon_action_oracle_plan_v1"
JACOBIAN_PLAN_SCHEMA = "nanogpt_sparse_moe_stepzero_token_jacobian_oracle_plan_v1"
SNAPSHOT_SCHEMA = "nanogpt_manifold_snapshot_v1"
FAMILIES = ("coupled_four", "separate_three_plus_three")


def load_terminal_snapshot(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != SNAPSHOT_SCHEMA:
        raise ValueError(f"not a manifold snapshot: {path}")
    if not isinstance(payload.get("model"), dict) or not payload["model"]:
        raise ValueError("manifold snapshot has no model state")
    if not isinstance(payload.get("model_config"), dict):
        raise ValueError("manifold snapshot has no model config")
    return payload


def model_from_exact_stepzero(payload: dict[str, Any], model_seed: int, device: str) -> GPT:
    resolved = payload["run_identity"]["resolved_config"]
    if resolved.get("init_from", "scratch") != "scratch":
        raise ValueError("step-zero reconstruction requires scratch initialization")
    if int(resolved["model_seed"]) != int(model_seed):
        raise ValueError("model seed disagrees with the frozen plan")
    torch.manual_seed(int(model_seed))
    model = GPT(GPTConfig(**payload["model_config"]))
    if set(model.state_dict()) != set(payload["model"]):
        raise ValueError("current model inventory disagrees with the source snapshot")
    model.to(device)
    return model


def layer_state_from_mapping(mapping: dict[str, torch.Tensor], layer: int) -> LayerState:
    prefix = f"transformer.h.{layer}.mlp"
    return LayerState(
        mapping[f"{prefix}.router.weight"].detach().float().cpu(),
        mapping[f"{prefix}.expert_c_fc"].detach().float().cpu(),
        mapping[f"{prefix}.expert_c_proj"].detach().float().cpu(),
    )


def selected_stepzero_hashes(model: GPT, layers: list[int]) -> dict[str, str]:
    mapping = dict(model.named_parameters())
    result: dict[str, str] = {}
    for layer in layers:
        prefix = f"transformer.h.{layer}.mlp"
        for target in ("router.weight", "expert_c_fc", "expert_c_proj"):
            name = f"{prefix}.{target}"
            result[name] = tensor_sha256(mapping[name].detach().cpu())
    return result


def collect_gradient_bank(
    model: GPT,
    batches: list[torch.Tensor],
    layers: list[int],
    device: str,
    direction_transform: str,
    muon_ns_steps: int,
    weight_decay: float,
    adam_epsilon: float,
    coordinates: int | None = None,
) -> tuple[dict[int, list[LayerState]], list[dict[str, Any]]]:
    selected: dict[int, tuple[torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter]] = {}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for layer in layers:
        mlp = model.transformer.h[layer].mlp
        mlp.router.weight.requires_grad_(True)
        mlp.expert_c_fc.requires_grad_(True)
        mlp.expert_c_proj.requires_grad_(True)
        selected[layer] = (mlp.router.weight, mlp.expert_c_fc, mlp.expert_c_proj)

    model.train()
    bank = {layer: [] for layer in layers}
    rows: list[dict[str, Any]] = []
    autocast_enabled = device.startswith("cuda")
    if direction_transform == "jacobian_sketch":
        if len(batches) != 1 or coordinates is None or coordinates <= 0:
            raise ValueError("Jacobian sketch requires one batch and positive coordinates")
        work = [(coordinate, batches[0]) for coordinate in range(coordinates)]
    else:
        work = list(enumerate(batches))
    for microbatch, tokens in work:
        model.zero_grad(set_to_none=True)
        tokens = tokens.to(device)
        inputs = tokens[:, :-1].contiguous()
        targets = tokens[:, 1:].contiguous()
        with torch.autocast(
            device_type="cuda" if autocast_enabled else "cpu",
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            if direction_transform == "jacobian_sketch":
                logits, _unused_loss = model(inputs, None)
                token_loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                    reduction="none",
                )
                mask = walsh_token_mask(
                    token_loss.numel(), microbatch, token_loss.device
                ).to(dtype=token_loss.dtype)
                loss = (token_loss * mask).mean()
                if microbatch == 0:
                    load_balance, router_z = model.moe_router_losses()
                    loss = (
                        loss
                        + float(model.config.moe_load_balance_aux_coefficient)
                        * load_balance
                        + float(model.config.moe_router_z_loss_coefficient)
                        * router_z
                    )
            else:
                _logits, loss = model(inputs, targets)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("non-finite task loss while collecting gradient bank")
        loss.backward()
        for layer, parameters in selected.items():
            if any(parameter.grad is None for parameter in parameters):
                raise RuntimeError(f"missing selected gradient at layer {layer}")
            gradient = LayerState(
                *(parameter.grad.detach().float() for parameter in parameters)
            )
            parameter_state = LayerState(
                *(parameter.detach().float() for parameter in parameters)
            )
            state = stepzero_optimizer_action(
                gradient,
                parameter_state,
                direction_transform=direction_transform,
                muon_ns_steps=muon_ns_steps,
                weight_decay=weight_decay,
                adam_epsilon=adam_epsilon,
            )
            router, c_fc, c_proj = (
                state.router.cpu().clone(),
                state.c_fc.cpu().clone(),
                state.c_proj.cpu().clone(),
            )
            state = LayerState(router, c_fc, c_proj)
            bank[layer].append(state)
            rows.append(
                {
                    "microbatch": microbatch,
                    "layer": layer,
                    "direction_transform": direction_transform,
                    "loss": float(loss.detach()),
                    "router_sha256": tensor_sha256(router),
                    "c_fc_sha256": tensor_sha256(c_fc),
                    "c_proj_sha256": tensor_sha256(c_proj),
                    "router_fro": float(router.norm()),
                    "c_fc_fro": float(c_fc.norm()),
                    "c_proj_fro": float(c_proj.norm()),
                }
            )
    model.zero_grad(set_to_none=True)
    model.eval()
    return bank, rows


def walsh_token_mask(length: int, coordinate: int, device: torch.device | str) -> torch.Tensor:
    """Return DC or a deterministic orthogonal Walsh sign row."""
    if length <= 0 or coordinate < 0:
        raise ValueError("Walsh length and coordinate must be non-negative")
    indices = torch.arange(length, device=device, dtype=torch.long)
    if coordinate == 0:
        return torch.ones(length, device=device)
    parity = torch.zeros(length, device=device, dtype=torch.long)
    bit = 0
    value = int(coordinate)
    while value:
        if value & 1:
            parity = parity ^ ((indices >> bit) & 1)
        value >>= 1
        bit += 1
    return 1.0 - 2.0 * parity.float()


def stepzero_optimizer_action(
    gradient: LayerState,
    parameter: LayerState,
    *,
    direction_transform: str,
    muon_ns_steps: int = 5,
    weight_decay: float = 0.1,
    adam_epsilon: float = 1e-8,
) -> LayerState:
    """Return raw descent or the exact production first-step action."""
    if direction_transform in {"raw", "jacobian_sketch"}:
        return LayerState(-gradient.router, -gradient.c_fc, -gradient.c_proj)
    if direction_transform != "muon_action":
        raise ValueError(f"unknown direction transform: {direction_transform}")
    router = -gradient.router / (gradient.router.abs() + float(adam_epsilon))
    c_fc = -muon_update_batched(gradient.c_fc, steps=muon_ns_steps)
    c_proj = -muon_update_batched(gradient.c_proj, steps=muon_ns_steps)
    if weight_decay != 0.0:
        c_fc = c_fc - float(weight_decay) * parameter.c_fc
        c_proj = c_proj - float(weight_decay) * parameter.c_proj
    return LayerState(router, c_fc, c_proj)


def reconstruct_gradient_family(
    left: LayerState,
    target_chord: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    directions: list[LayerState],
    family: str,
    ridge_ratio: float,
    device: str,
) -> tuple[LayerState, dict[str, float]]:
    rank = 4 if family == "coupled_four" else 3
    if family not in FAMILIES or len(directions) < rank:
        raise ValueError("invalid family or insufficient gradient directions")
    target_router, target_fc, target_proj = (
        value.to(device=device) for value in target_chord
    )
    views = [_atom_views(direction) for direction in directions[:rank]]
    router_basis = torch.stack([view[0] for view in views]).to(device)
    fc_basis = torch.stack([view[1] for view in views]).to(device)
    proj_basis = torch.stack([view[2] for view in views]).to(device)
    projected_router, _router_coordinates = project_rows(
        target_router, router_basis, ridge_ratio
    )
    if family == "coupled_four":
        target_pair = torch.cat((target_fc, target_proj), dim=-1)
        pair_basis = torch.cat((fc_basis, proj_basis), dim=-1)
        projected_pair, _pair_coordinates = project_rows(
            target_pair, pair_basis, ridge_ratio
        )
        projected_fc, projected_proj = projected_pair.split(
            target_fc.shape[-1], dim=-1
        )
    else:
        projected_fc, _fc_coordinates = project_rows(
            target_fc, fc_basis, ridge_ratio
        )
        projected_proj, _proj_coordinates = project_rows(
            target_proj, proj_basis, ridge_ratio
        )

    left_router, left_fc, left_proj = (
        value.to(device=device) for value in _atom_views(left)
    )
    reconstructed = _state_from_atom_views(
        left_router + projected_router,
        left_fc + projected_fc,
        left_proj + projected_proj,
    )
    return reconstructed, {
        "router_parameter_recovery": energy_recovery(projected_router, target_router),
        "c_fc_parameter_recovery": energy_recovery(projected_fc, target_fc),
        "c_proj_parameter_recovery": energy_recovery(projected_proj, target_proj),
        "paired_parameter_recovery": energy_recovery(
            torch.cat((projected_fc, projected_proj), dim=-1),
            torch.cat((target_fc, target_proj), dim=-1),
        ),
    }


def row_span_overlap(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape or left.ndim < 2:
        raise ValueError("basis banks must have identical [rank, ..., width] shape")
    rank = left.shape[0]
    left_matrix = left.movedim(0, -2).float()
    right_matrix = right.movedim(0, -2).float()
    left_q = torch.linalg.qr(left_matrix.transpose(-2, -1), mode="reduced").Q
    right_q = torch.linalg.qr(right_matrix.transpose(-2, -1), mode="reduced").Q
    overlap = (left_q.transpose(-2, -1) @ right_q).square().sum(dim=(-2, -1)) / rank
    return float(overlap.mean())


def family_overlap(
    left: list[LayerState], right: list[LayerState], family: str
) -> float:
    rank = 4 if family == "coupled_four" else 3
    left_views = [_atom_views(value) for value in left[:rank]]
    right_views = [_atom_views(value) for value in right[:rank]]
    if family == "coupled_four":
        left_basis = torch.cat(
            (
                torch.stack([value[1] for value in left_views]),
                torch.stack([value[2] for value in left_views]),
            ),
            dim=-1,
        )
        right_basis = torch.cat(
            (
                torch.stack([value[1] for value in right_views]),
                torch.stack([value[2] for value in right_views]),
            ),
            dim=-1,
        )
        return row_span_overlap(left_basis, right_basis)
    fc_overlap = row_span_overlap(
        torch.stack([value[1] for value in left_views]),
        torch.stack([value[1] for value in right_views]),
    )
    proj_overlap = row_span_overlap(
        torch.stack([value[2] for value in left_views]),
        torch.stack([value[2] for value in right_views]),
    )
    return 0.5 * (fc_overlap + proj_overlap)


def all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


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
    plan_schema = plan.get("schema_version")
    if plan_schema not in {PLAN_SCHEMA, MUON_PLAN_SCHEMA, JACOBIAN_PLAN_SCHEMA}:
        raise ValueError("task-gradient oracle plan schema mismatch")
    direction_transform = {
        PLAN_SCHEMA: "raw",
        MUON_PLAN_SCHEMA: "muon_action",
        JACOBIAN_PLAN_SCHEMA: "jacobian_sketch",
    }[plan_schema]
    payload = load_terminal_snapshot(args.terminal_snapshot)
    causal = plan["causal_source"]
    if file_sha256(args.terminal_snapshot) != causal["terminal_manifold_snapshot_sha256"]:
        raise ValueError("terminal snapshot hash disagrees with the plan")
    if payload["run_identity"]["config_sha256"] != causal["run_config_sha256"]:
        raise ValueError("run config identity disagrees with the plan")
    layers = [int(value) for value in causal["layers"]]
    model = model_from_exact_stepzero(
        payload, int(causal["model_seed"]), args.device
    )
    stepzero_hashes = selected_stepzero_hashes(model, layers)
    stepzero_mapping = dict(model.named_parameters())
    initial = {
        layer: layer_state_from_mapping(stepzero_mapping, layer) for layer in layers
    }
    terminal = {
        layer: layer_state_from_mapping(payload["model"], layer) for layer in layers
    }

    banks: dict[str, dict[int, list[LayerState]]] = {}
    gradient_rows: list[dict[str, Any]] = []
    for bank_spec in plan["gradient_banks"]:
        batch_count = int(
            bank_spec.get("batches", bank_spec.get("independent_microbatches", 0))
        )
        batches = fixed_validation_batches(
            args.data_dir,
            int(bank_spec["batch_size"]),
            int(bank_spec["block_size"]) + 1,
            batch_count,
            int(bank_spec["seed"]),
        )
        bank, rows = collect_gradient_bank(
            model,
            batches,
            layers,
            args.device,
            direction_transform,
            muon_ns_steps=5,
            weight_decay=0.1,
            adam_epsilon=1e-8,
            coordinates=(
                int(bank_spec["coordinates"])
                if "coordinates" in bank_spec
                else None
            ),
        )
        banks[bank_spec["name"]] = bank
        for row in rows:
            row["bank"] = bank_spec["name"]
        gradient_rows.extend(rows)

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
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    rows: list[dict[str, Any]] = []
    fixed_by_family_layer: dict[tuple[str, int], float] = {}
    for layer in layers:
        left = initial[layer]
        right = terminal[layer]
        target_chord = _state_chord(right, left)
        heldout = heldout_inputs[layer].to(args.device)
        base_output = sparse_moe_output(left.to(args.device), heldout, 2)
        target_output = sparse_moe_output(right.to(args.device), heldout, 2) - base_output
        for family in FAMILIES:
            rank = 4 if family == "coupled_four" else 3
            fixed, fixed_recoveries, _metadata = reconstruct_family(
                left,
                target_chord,
                [],
                rank,
                family,
                "fixed_structured",
                layer,
                args.ridge_ratio,
                args.device,
            )
            fixed_prediction = (
                sparse_moe_output(fixed, heldout, 2) - base_output
            )
            fixed_exact = recovery_fraction(fixed_prediction, target_output)
            fixed_by_family_layer[(family, layer)] = fixed_exact
            rows.append(
                {
                    "bank": "fixed_structured",
                    "family": family,
                    "layer": layer,
                    "heldout_exact_recovery": fixed_exact,
                    **fixed_recoveries,
                }
            )
            for bank_name, bank in banks.items():
                reconstructed, recoveries = reconstruct_gradient_family(
                    left,
                    target_chord,
                    bank[layer],
                    family,
                    args.ridge_ratio,
                    args.device,
                )
                prediction = (
                    sparse_moe_output(reconstructed, heldout, 2) - base_output
                )
                rows.append(
                    {
                        "bank": bank_name,
                        "family": family,
                        "layer": layer,
                        "heldout_exact_recovery": recovery_fraction(
                            prediction, target_output
                        ),
                        **recoveries,
                    }
                )

    overlaps = {
        family: {
            str(layer): family_overlap(
                banks["discovery_a"][layer],
                banks["discovery_b"][layer],
                family,
            )
            for layer in layers
        }
        for family in FAMILIES
    }
    temporal = plan["controls"].get("temporal_reference")
    temporal_best = {
        "coupled_four": 0.044682,
        "separate_three_plus_three": 0.052700,
    }
    gates = plan["frozen_gates"]
    summary: dict[str, Any] = {}
    decisions: dict[str, Any] = {}
    for family in FAMILIES:
        summary[family] = {}
        fixed_mean = sum(
            fixed_by_family_layer[(family, layer)] for layer in layers
        ) / len(layers)
        summary[family]["fixed_structured_mean"] = fixed_mean
        overlap_mean = sum(overlaps[family].values()) / len(layers)
        summary[family]["discovery_bank_subspace_overlap"] = {
            "mean": overlap_mean,
            "by_layer": overlaps[family],
        }
        decisions[family] = {}
        for bank_name in banks:
            selected = [
                row
                for row in rows
                if row["family"] == family and row["bank"] == bank_name
            ]
            exact = [float(row["heldout_exact_recovery"]) for row in selected]
            bank_summary = {
                "mean": sum(exact) / len(exact),
                "minimum": min(exact),
                "by_layer": {
                    str(row["layer"]): float(row["heldout_exact_recovery"])
                    for row in selected
                },
                "paired_parameter_recovery_mean": sum(
                    float(row["paired_parameter_recovery"]) for row in selected
                ) / len(selected),
            }
            summary[family][bank_name] = bank_summary
            bank_gates = {
                "mean_pass": bank_summary["mean"]
                >= float(gates["heldout_exact_recovery_mean_min_each_bank"]),
                "every_layer_pass": bank_summary["minimum"]
                >= float(gates["heldout_exact_recovery_every_layer_min_each_bank"]),
                "bank_overlap_pass": overlap_mean
                >= float(gates["discovery_bank_mean_subspace_overlap_min"]),
            }
            if direction_transform == "raw":
                bank_gates.update(
                    {
                        "minus_fixed_pass": bank_summary["mean"] - fixed_mean
                        >= float(
                            gates[
                                "task_gradient_minus_fixed_structured_mean_min_each_bank"
                            ]
                        ),
                        "minus_temporal_pass": bank_summary["mean"]
                        - temporal_best[family]
                        >= float(
                            gates[
                                "task_gradient_minus_best_temporal_reference_min_each_bank"
                            ]
                        ),
                    }
                )
            elif direction_transform == "muon_action":
                raw_mean = float(
                    plan["controls"]["raw_gradient_exact_mean"][family][bank_name]
                )
                bank_summary["raw_gradient_mean"] = raw_mean
                bank_summary["optimizer_action_minus_raw_gradient"] = (
                    bank_summary["mean"] - raw_mean
                )
                bank_gates["minus_raw_gradient_pass"] = (
                    bank_summary["mean"] - raw_mean
                    >= float(
                        gates[
                            "optimizer_action_minus_raw_gradient_mean_min_each_bank"
                        ]
                    )
                )
            else:
                raw_mean = float(
                    plan["controls"]["raw_gradient_exact_mean"][family][bank_name]
                )
                bank_summary["raw_gradient_mean"] = raw_mean
                bank_summary["jacobian_sketch_minus_raw_gradient"] = (
                    bank_summary["mean"] - raw_mean
                )
                bank_gates["minus_raw_gradient_pass"] = (
                    bank_summary["mean"] - raw_mean
                    >= float(
                        gates[
                            "jacobian_sketch_minus_raw_gradient_mean_min_each_bank"
                        ]
                    )
                )
            bank_gates["all_pass"] = all(bank_gates.values())
            decisions[family][bank_name] = bank_gates

    finite = all_finite({"rows": rows, "summary": summary, "overlaps": overlaps})
    both_pass = [
        family
        for family in FAMILIES
        if all(decisions[family][bank]["all_pass"] for bank in banks)
    ]
    one_pass = [
        family
        for family in FAMILIES
        if sum(decisions[family][bank]["all_pass"] for bank in banks) == 1
    ]
    if finite and both_pass:
        decision = (
            "PASS_BOTH_DISCOVERY_BANKS_AUTHORIZE_5TPP_OPTIMIZER_ACTION_SKETCH_ACQUISITION_ONLY"
            if direction_transform == "muon_action"
            else (
                "PASS_BOTH_DISCOVERY_BANKS_AUTHORIZE_5TPP_TOKEN_JACOBIAN_SKETCH_ACQUISITION_ONLY"
                if direction_transform == "jacobian_sketch"
                else "PASS_BOTH_DISCOVERY_BANKS_AUTHORIZE_5TPP_GRADIENT_SKETCH_ACQUISITION_ONLY"
            )
        )
    elif finite and one_pass:
        decision = "PASS_ONE_BANK_ONLY_REJECT_BATCH_FRAGILE_BASIS"
    elif direction_transform == "muon_action" and finite and all(
        summary[family][bank]["optimizer_action_minus_raw_gradient"] > 0.0
        for family in FAMILIES
        for bank in banks
    ):
        decision = "FAIL_STRICT_GATES_BUT_IMPROVE_RAW_AUTHORIZE_FUNCTIONAL_JACOBIAN_PLAN_ONLY"
    elif direction_transform == "muon_action":
        decision = "FAIL_STRICT_GATES_NO_RAW_IMPROVEMENT_REJECT_OPTIMIZER_ACTION_IMAGE"
    elif direction_transform == "jacobian_sketch" and finite and all(
        summary[family][bank]["jacobian_sketch_minus_raw_gradient"] > 0.0
        for family in FAMILIES
        for bank in banks
    ):
        decision = "FAIL_STRICT_GATES_BUT_IMPROVE_RAW_AUTHORIZE_KFAC_PLAN_ONLY"
    elif direction_transform == "jacobian_sketch":
        decision = "FAIL_STRICT_GATES_NO_RAW_IMPROVEMENT_REJECT_TOKEN_JACOBIAN_IMAGE"
    else:
        decision = "FAIL_BOTH_BANKS_REJECT_RAW_TASK_GRADIENT_LATENT_IMAGE"

    args.output.mkdir(parents=True, exist_ok=True)
    rows_path = args.output / "task_gradient_oracle_rows.csv"
    gradient_path = args.output / "task_gradient_hashes.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=union_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)
    with gradient_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=union_fieldnames(gradient_rows))
        writer.writeheader()
        writer.writerows(gradient_rows)
    result = {
        "schema_version": {
            "raw": "nanogpt_sparse_moe_stepzero_task_gradient_oracle_result_v1",
            "muon_action": "nanogpt_sparse_moe_stepzero_muon_action_oracle_result_v1",
            "jacobian_sketch": "nanogpt_sparse_moe_stepzero_token_jacobian_oracle_result_v1",
        }[direction_transform],
        "direction_transform": direction_transform,
        "decision": decision,
        "passing_families": both_pass,
        "one_bank_only_families": one_pass,
        "all_values_finite": finite,
        "summary": summary,
        "gates": decisions,
        "stepzero_selected_tensor_sha256": stepzero_hashes,
        "source": {
            "terminal_snapshot_sha256": file_sha256(args.terminal_snapshot),
            "plan_sha256": file_sha256(args.plan),
            "dataset_manifest_sha256": causal["dataset_manifest_sha256"],
            "temporal_reference": temporal,
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
    result_path = args.output / "task_gradient_oracle_result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    status = {
        "state": "finished",
        "exit_code": 0,
        "decision": decision,
        "result_sha256": file_sha256(result_path),
        "rows_sha256": file_sha256(rows_path),
        "gradient_hashes_sha256": file_sha256(gradient_path),
        "wall_seconds": time.time() - started,
    }
    status_path = args.output / "task_gradient_oracle_status.json"
    status_path.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": status, "summary": summary}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
