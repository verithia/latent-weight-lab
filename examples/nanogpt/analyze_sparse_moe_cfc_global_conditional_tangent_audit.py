#!/usr/bin/env python3
"""Audit whether <=7 global conditional directions capture expert conflict."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_cproj_manifold import load_model
from examples.nanogpt.analyze_mlp_activation_update_alignment import git_commit
from examples.nanogpt.analyze_sparse_moe_cfc_learned_butterfly_frame_oracle import (
    ButterflyCFCState,
    LearnedButterflyCFC,
)
from examples.nanogpt.analyze_sparse_moe_cfc_spectral_feature_oracle import (
    collect_protocol_inputs,
    dense_targets,
    normalized_fit_loss,
    route_and_sample,
)
from examples.nanogpt.analyze_sparse_moe_paired_alignment import file_sha256
from examples.nanogpt.analyze_sparse_moe_stepzero_task_gradient_oracle import (
    layer_state_from_mapping,
    load_terminal_snapshot,
)


PLAN_SCHEMA = "nanogpt_sparse_moe_cfc_global_conditional_tangent_audit_plan_v1"


def expert_angle_gradients(
    operator: LearnedButterflyCFC,
    shared: ButterflyCFCState,
    inputs: torch.Tensor,
    c_fc: torch.Tensor,
    c_proj: torch.Tensor,
) -> torch.Tensor:
    """Return one shared-angle functional gradient per expert."""
    gradients: list[torch.Tensor] = []
    for expert in range(operator.experts):
        input_angles = shared.input_angles.to(operator.device).detach().clone()
        hidden_angles = shared.hidden_angles.to(operator.device).detach().clone()
        input_angles.requires_grad_(True)
        hidden_angles.requires_grad_(True)
        expert_state = ButterflyCFCState(
            input_angles,
            hidden_angles,
            shared.spectrum[expert : expert + 1].to(operator.device),
            shared.bias[expert : expert + 1].to(operator.device),
        )
        expert_inputs = inputs[expert : expert + 1]
        target_pre, target_output = dense_targets(
            expert_inputs,
            c_fc[expert : expert + 1],
            c_proj[expert : expert + 1],
            operator.device,
        )
        predicted_pre, predicted_output = operator.expert_output(
            expert_inputs,
            c_proj[expert : expert + 1],
            expert_state,
        )
        loss = normalized_fit_loss(
            predicted_pre,
            predicted_output,
            target_pre,
            target_output,
        )
        input_gradient, hidden_gradient = torch.autograd.grad(
            loss, (input_angles, hidden_angles)
        )
        gradient = torch.cat(
            (input_gradient.reshape(-1), hidden_gradient.reshape(-1))
        ).detach()
        if not torch.isfinite(gradient).all() or float(gradient.norm()) <= 1e-12:
            raise RuntimeError("non-finite or zero expert angle gradient")
        gradients.append(gradient.cpu())
    return torch.stack(gradients)


def centered_normalized(gradients: torch.Tensor) -> torch.Tensor:
    """Remove each layer's shared update and normalize every expert residual."""
    if gradients.ndim != 3:
        raise ValueError("gradients must be [layer, expert, coordinate]")
    centered = gradients - gradients.mean(dim=1, keepdim=True)
    norms = centered.norm(dim=-1, keepdim=True)
    if bool((norms <= 1e-12).any()):
        raise RuntimeError("centering produced a zero expert gradient")
    return centered / norms


def right_basis(rows: torch.Tensor, rank: int) -> torch.Tensor:
    if rows.ndim != 2:
        raise ValueError("basis input must be a matrix")
    if not 1 <= int(rank) <= min(rows.shape):
        raise ValueError("invalid tangent rank")
    _u, _s, vh = torch.linalg.svd(rows, full_matrices=False)
    return vh[: int(rank)].T.contiguous()


def projection_scores(
    train: torch.Tensor,
    test: torch.Tensor,
    rank: int,
) -> dict[str, Any]:
    """Fit a global row-space basis and score heldout normalized rows."""
    if train.shape != test.shape or train.ndim != 3:
        raise ValueError("train/test gradient banks must agree")
    layers, experts, coordinates = train.shape
    basis = right_basis(train.reshape(layers * experts, coordinates), int(rank))
    train_coefficients = train.reshape(layers * experts, coordinates) @ basis
    test_coefficients = test.reshape(layers * experts, coordinates) @ basis
    explained = test_coefficients.square().sum(dim=-1).reshape(layers, experts)
    coefficient_cosines = torch.nn.functional.cosine_similarity(
        train_coefficients,
        test_coefficients,
        dim=-1,
        eps=1e-12,
    ).reshape(layers, experts)
    per_layer_mean = explained.mean(dim=1)
    return {
        "rank": int(rank),
        "explained_energy_mean": float(explained.mean()),
        "explained_energy_minimum_layer": float(per_layer_mean.min()),
        "explained_energy_minimum_row": float(explained.min()),
        "explained_energy_by_layer_mean": [float(value) for value in per_layer_mean],
        "coefficient_cosine_mean": float(coefficient_cosines.mean()),
        "coefficient_cosine_minimum_layer": float(
            coefficient_cosines.mean(dim=1).min()
        ),
    }


def per_layer_projection_scores(
    train: torch.Tensor,
    test: torch.Tensor,
    rank: int,
) -> dict[str, Any]:
    if train.shape != test.shape or train.ndim != 3:
        raise ValueError("train/test gradient banks must agree")
    rows: list[dict[str, Any]] = []
    for layer in range(train.shape[0]):
        basis = right_basis(train[layer], int(rank))
        coefficients = test[layer] @ basis
        explained = coefficients.square().sum(dim=-1)
        rows.append(
            {
                "explained_energy_mean": float(explained.mean()),
                "explained_energy_minimum_row": float(explained.min()),
            }
        )
    return {
        "rank": int(rank),
        "by_layer": rows,
        "explained_energy_mean": sum(
            row["explained_energy_mean"] for row in rows
        )
        / len(rows),
        "explained_energy_minimum_layer_mean": min(
            row["explained_energy_mean"] for row in rows
        ),
        "explained_energy_minimum_row": min(
            row["explained_energy_minimum_row"] for row in rows
        ),
    }


def corresponding_cosines(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    if left.shape != right.shape or left.ndim != 3:
        raise ValueError("gradient banks must agree")
    values = (left * right).sum(dim=-1)
    return {
        "mean": float(values.mean()),
        "minimum_layer_mean": float(values.mean(dim=1).min()),
        "minimum_row": float(values.min()),
        "by_layer_mean": [float(value) for value in values.mean(dim=1)],
    }


def classify(
    global_pass: bool,
    per_layer_pass: bool,
    stability_pass: bool,
) -> str:
    if global_pass and stability_pass:
        return "GLOBAL_RANK7_CONDITIONAL_TANGENT_SUPPORTED"
    if per_layer_pass:
        return "CROSS_LAYER_TEMPLATE_SHARING_LIMIT"
    if not stability_pass or not per_layer_pass:
        return "CROSS_BANK_TANGENT_INSTABILITY"
    return "BUDGET_COMPATIBLE_CONDITIONAL_TANGENT_REJECTED"


def validate_plan(plan: dict[str, Any], plan_path: Path) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("conditional tangent plan schema mismatch")
    identity = plan["identity"]
    if identity.get("entrypoint_sha256") != file_sha256(Path(__file__)):
        raise ValueError("entrypoint hash is not sealed in the plan")
    root = Path(__file__).resolve().parents[2]
    for relative, expected in identity["helper_sha256"].items():
        if file_sha256(root / relative) != expected:
            raise ValueError(f"helper hash drift: {relative}")
    budget = plan["coordinate_budget"]
    if not (
        float(budget["rank7_compression_ratio"]) >= 200.0
        and float(budget["rank8_compression_ratio"]) < 200.0
    ):
        raise ValueError("registered rank boundary is inconsistent")
    if not file_sha256(plan_path):
        raise AssertionError("unreachable empty plan hash")


def all_tensor_values_finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(all_tensor_values_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_tensor_values_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--parent-plan", required=True, type=Path)
    parser.add_argument("--parent-coordinates", required=True, type=Path)
    parser.add_argument("--conflict-result", required=True, type=Path)
    parser.add_argument("--terminal-snapshot", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    validate_plan(plan, args.plan)
    parent = plan["causal_parent"]
    source = plan["source"]
    for path, expected, label in (
        (args.parent_plan, source["shared_endpoint_plan_sha256"], "parent plan"),
        (
            args.parent_coordinates,
            source["shared_endpoint_coordinates_sha256"],
            "parent coordinates",
        ),
        (
            args.conflict_result,
            parent["conflict_remote_result_sha256"],
            "conflict result",
        ),
        (
            args.terminal_snapshot,
            source["terminal_manifold_snapshot_sha256"],
            "terminal snapshot",
        ),
        (
            args.data_dir / "manifest.json",
            source["dataset_manifest_sha256"],
            "dataset manifest",
        ),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"{label} hash disagrees with frozen plan")
    conflict_result = json.loads(args.conflict_result.read_text(encoding="utf-8"))
    if conflict_result.get("classification") != "ONE_SWEEP_BUTTERFLY_TOPOLOGY_INSUFFICIENT":
        raise ValueError("causal parent classification drift")
    parent_plan = json.loads(args.parent_plan.read_text(encoding="utf-8"))
    coordinates = torch.load(
        args.parent_coordinates, map_location="cpu", weights_only=False
    )
    payload = load_terminal_snapshot(args.terminal_snapshot)
    if int(payload["next_iter"]) != int(source["next_iter"]):
        raise ValueError("snapshot step disagrees with frozen plan")
    model = load_model(args.terminal_snapshot, args.device)
    model.eval()
    inputs = collect_protocol_inputs(model, parent_plan, args.data_dir, args.device)
    terminal_mapping = dict(model.named_parameters())
    layers = [int(value) for value in source["layers"]]
    states = {
        layer: layer_state_from_mapping(terminal_mapping, layer) for layer in layers
    }
    del terminal_mapping, model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    constants = plan["replay_constants"]
    banks = [str(value) for value in source["discovery_banks"]]
    raw_gradients: dict[str, torch.Tensor] = {}
    occupancy: dict[str, dict[str, list[int]]] = {}
    for bank_index, bank in enumerate(banks):
        layer_gradients: list[torch.Tensor] = []
        occupancy[bank] = {}
        for layer in layers:
            state = states[layer]
            sampled, counts = route_and_sample(
                state,
                inputs[bank][layer],
                top_k=int(constants["top_k"]),
                samples_per_expert=int(constants["fit_samples_per_expert"]),
                seed=(
                    int(constants["sampling_seed_base"])
                    + int(constants["sampling_seed_bank_stride"]) * bank_index
                    + int(constants["sampling_seed_layer_stride"]) * layer
                ),
            )
            occupancy[bank][str(layer)] = counts
            operator = LearnedButterflyCFC(
                experts=int(source["num_experts"]),
                input_width=int(parent_plan["source"]["input_width"]),
                hidden_width=int(parent_plan["source"]["expert_hidden_width"]),
                input_padded_width=int(parent_plan["candidate"]["input_padded_width"]),
                hidden_padded_width=int(parent_plan["candidate"]["hidden_padded_width"]),
                seed=int(constants["operator_seed"]),
                layer=layer,
                device=args.device,
            )
            layer_gradients.append(
                expert_angle_gradients(
                    operator,
                    coordinates["candidate"][bank][str(layer)],
                    sampled,
                    state.c_fc,
                    state.c_proj,
                )
            )
        raw_gradients[bank] = torch.stack(layer_gradients)

    normalized = {bank: centered_normalized(raw_gradients[bank]) for bank in banks}
    ranks = [int(value) for value in constants["ranks"]]
    directions = ((banks[0], banks[1]), (banks[1], banks[0]))
    global_curves: dict[str, list[dict[str, Any]]] = {}
    per_layer_curves: dict[str, list[dict[str, Any]]] = {}
    for train_bank, test_bank in directions:
        label = f"{train_bank}_to_{test_bank}"
        global_curves[label] = [
            projection_scores(normalized[train_bank], normalized[test_bank], rank)
            for rank in ranks
        ]
        per_layer_curves[label] = [
            per_layer_projection_scores(
                normalized[train_bank], normalized[test_bank], rank
            )
            for rank in ranks
        ]
    stability = corresponding_cosines(normalized[banks[0]], normalized[banks[1]])
    thresholds = plan["frozen_rank7_gates"]
    rank7_global = {label: rows[-1] for label, rows in global_curves.items()}
    rank7_per_layer = {
        label: rows[-1] for label, rows in per_layer_curves.items()
    }
    global_gates: dict[str, dict[str, bool]] = {}
    per_layer_gates: dict[str, dict[str, bool]] = {}
    for label, row in rank7_global.items():
        global_gates[label] = {
            "mean_pass": float(row["explained_energy_mean"])
            >= float(thresholds["global_cross_bank_explained_energy_mean_min_each_direction"]),
            "minimum_layer_pass": float(row["explained_energy_minimum_layer"])
            >= float(thresholds["global_cross_bank_explained_energy_minimum_layer_min_each_direction"]),
            "minimum_row_pass": float(row["explained_energy_minimum_row"])
            >= float(thresholds["global_cross_bank_explained_energy_minimum_row_min_each_direction"]),
        }
        global_gates[label]["all_pass"] = all(global_gates[label].values())
    for label, row in rank7_per_layer.items():
        per_layer_gates[label] = {
            "mean_pass": float(row["explained_energy_mean"])
            >= float(thresholds["per_layer_localization_mean_min_each_direction"]),
            "minimum_row_pass": float(row["explained_energy_minimum_row"])
            >= float(thresholds["per_layer_localization_minimum_row_min_each_direction"]),
        }
        per_layer_gates[label]["all_pass"] = all(per_layer_gates[label].values())
    stability_gates = {
        "mean_pass": float(stability["mean"])
        >= float(thresholds["corresponding_gradient_cosine_mean_min"]),
        "minimum_layer_pass": float(stability["minimum_layer_mean"])
        >= float(thresholds["corresponding_gradient_cosine_minimum_layer_min"]),
    }
    stability_gates["all_pass"] = all(stability_gates.values())
    finite = all_tensor_values_finite(
        {
            "raw": raw_gradients,
            "normalized": normalized,
            "global": global_curves,
            "per_layer": per_layer_curves,
            "stability": stability,
        }
    )
    global_pass = finite and all(row["all_pass"] for row in global_gates.values())
    per_layer_pass = finite and all(
        row["all_pass"] for row in per_layer_gates.values()
    )
    classification = classify(global_pass, per_layer_pass, stability_gates["all_pass"])

    args.output.mkdir(parents=True, exist_ok=False)
    gradients_path = args.output / "expert_angle_gradients.pt"
    torch.save(
        {
            "schema_version": "nanogpt_sparse_moe_cfc_expert_angle_gradients_v1",
            "layers": layers,
            "banks": banks,
            "raw_gradients": raw_gradients,
            "centered_normalized_gradients": normalized,
        },
        gradients_path,
    )
    result = {
        "schema_version": "nanogpt_sparse_moe_cfc_global_conditional_tangent_audit_result_v1",
        "classification": classification,
        "identity": {
            "git_commit": git_commit(Path(__file__).resolve().parents[2]),
            "plan_sha256": file_sha256(args.plan),
            "entrypoint_sha256": file_sha256(Path(__file__)),
            "parent_plan_sha256": file_sha256(args.parent_plan),
            "parent_coordinates_sha256": file_sha256(args.parent_coordinates),
            "conflict_result_sha256": file_sha256(args.conflict_result),
            "terminal_snapshot_sha256": file_sha256(args.terminal_snapshot),
            "dataset_manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
        },
        "execution": {
            "device": args.device,
            "wall_seconds": time.time() - started,
            "checkpoint_updates": 0,
            "maximum_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated())
                if args.device.startswith("cuda")
                else 0
            ),
            "gradients_path": str(gradients_path),
            "gradients_sha256": file_sha256(gradients_path),
        },
        "coordinate_budget": plan["coordinate_budget"],
        "occupancy": occupancy,
        "direct_corresponding_gradient_cosine": stability,
        "global_cross_bank_rank_curves": global_curves,
        "per_layer_cross_bank_rank_curves": per_layer_curves,
        "rank7_global_gates": global_gates,
        "rank7_per_layer_gates": per_layer_gates,
        "stability_gates": stability_gates,
        "global_rank7_passed": global_pass,
        "per_layer_rank7_passed": per_layer_pass,
        "all_values_and_gradients_finite": finite,
        "authorization": {
            "all_layer_acquisition": classification
            == "GLOBAL_RANK7_CONDITIONAL_TANGENT_SUPPORTED",
            "functional_oracle": False,
            "implementation": False,
            "initialization_fit_shadow": False,
            "mfu_preflight": False,
            "language_model_training": False,
            "larger_rung": False,
            "generated_cproj": False,
        },
    }
    result_path = args.output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
