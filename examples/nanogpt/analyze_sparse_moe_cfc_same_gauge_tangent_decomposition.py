#!/usr/bin/env python3
"""Separate activation-bank effects from mapping-endpoint tangent drift."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_cproj_manifold import load_model
from examples.nanogpt.analyze_mlp_activation_update_alignment import git_commit
from examples.nanogpt.analyze_sparse_moe_cfc_global_conditional_tangent_audit import (
    all_tensor_values_finite,
    centered_normalized,
    corresponding_cosines,
    expert_angle_gradients,
    per_layer_projection_scores,
    projection_scores,
)
from examples.nanogpt.analyze_sparse_moe_cfc_learned_butterfly_frame_oracle import (
    ButterflyCFCState,  # noqa: F401 - required for __main__ pickle compatibility
    LearnedButterflyCFC,
)
from examples.nanogpt.analyze_sparse_moe_cfc_spectral_feature_oracle import (
    collect_protocol_inputs,
    route_and_sample,
)
from examples.nanogpt.analyze_sparse_moe_paired_alignment import file_sha256
from examples.nanogpt.analyze_sparse_moe_stepzero_task_gradient_oracle import (
    layer_state_from_mapping,
    load_terminal_snapshot,
)


PLAN_SCHEMA = "nanogpt_sparse_moe_cfc_same_gauge_tangent_decomposition_plan_v1"


def comparison_metrics(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    rank: int,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    direct = corresponding_cosines(left, right)
    left_to_right = projection_scores(left, right, rank)
    right_to_left = projection_scores(right, left, rank)
    per_layer_left_to_right = per_layer_projection_scores(left, right, rank)
    per_layer_right_to_left = per_layer_projection_scores(right, left, rank)
    directional = (left_to_right, right_to_left)
    gates = {
        "cross_projection_mean_pass": all(
            float(row["explained_energy_mean"])
            >= float(
                thresholds[
                    "global_rank7_cross_projection_mean_min_both_directions"
                ]
            )
            for row in directional
        ),
        "cross_projection_minimum_layer_pass": all(
            float(row["explained_energy_minimum_layer"])
            >= float(
                thresholds[
                    "global_rank7_cross_projection_minimum_layer_min_both_directions"
                ]
            )
            for row in directional
        ),
        "cross_projection_minimum_row_pass": all(
            float(row["explained_energy_minimum_row"])
            >= float(
                thresholds[
                    "global_rank7_cross_projection_minimum_row_min_both_directions"
                ]
            )
            for row in directional
        ),
        "direct_cosine_mean_pass": float(direct["mean"])
        >= float(thresholds["direct_corresponding_gradient_cosine_mean_min"]),
        "direct_cosine_minimum_layer_pass": float(direct["minimum_layer_mean"])
        >= float(
            thresholds[
                "direct_corresponding_gradient_cosine_minimum_layer_min"
            ]
        ),
    }
    gates["all_pass"] = all(gates.values())
    return {
        "direct_corresponding_cosine": direct,
        "global_rank7_left_to_right": left_to_right,
        "global_rank7_right_to_left": right_to_left,
        "per_layer_rank7_left_to_right": per_layer_left_to_right,
        "per_layer_rank7_right_to_left": per_layer_right_to_left,
        "gates": gates,
    }


def classify(data_pass: bool, endpoint_pass: bool) -> str:
    if not data_pass:
        return "ACTIVATION_CONDITIONED_TANGENT_INSTABILITY"
    if not endpoint_pass:
        return "NONLINEAR_ENDPOINT_CHART_DRIFT"
    if data_pass and endpoint_pass:
        return "SAME_GAUGE_TANGENT_STABLE"
    return "MIXED_NONSEPARABLE_TANGENT_INSTABILITY"


def validate_plan(plan: dict[str, Any], plan_path: Path) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("same-gauge tangent plan schema mismatch")
    identity = plan["identity"]
    if identity.get("entrypoint_sha256") != file_sha256(Path(__file__)):
        raise ValueError("entrypoint hash is not sealed in the plan")
    root = Path(__file__).resolve().parents[2]
    for relative, expected in identity["helper_sha256"].items():
        if file_sha256(root / relative) != expected:
            raise ValueError(f"helper hash drift: {relative}")
    if not file_sha256(plan_path):
        raise AssertionError("unreachable empty plan hash")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--parent-plan", required=True, type=Path)
    parser.add_argument("--parent-coordinates", required=True, type=Path)
    parser.add_argument("--parent-gradient-artifact", required=True, type=Path)
    parser.add_argument("--parent-result", required=True, type=Path)
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
            args.parent_gradient_artifact,
            parent["gradient_artifact_sha256"],
            "parent gradient artifact",
        ),
        (args.parent_result, parent["remote_result_sha256"], "parent result"),
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
    parent_result = json.loads(args.parent_result.read_text(encoding="utf-8"))
    if parent_result.get("classification") != "CROSS_BANK_TANGENT_INSTABILITY":
        raise ValueError("causal parent classification drift")
    parent_plan = json.loads(args.parent_plan.read_text(encoding="utf-8"))
    coordinates = torch.load(
        args.parent_coordinates, map_location="cpu", weights_only=False
    )
    parent_gradients = torch.load(
        args.parent_gradient_artifact, map_location="cpu", weights_only=False
    )
    if parent_gradients.get("schema_version") != (
        "nanogpt_sparse_moe_cfc_expert_angle_gradients_v1"
    ):
        raise ValueError("parent gradient artifact schema drift")
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
    endpoints = [str(value) for value in source["endpoints"]]
    banks = [str(value) for value in source["data_banks"]]
    sampled: dict[str, dict[int, torch.Tensor]] = {bank: {} for bank in banks}
    occupancy: dict[str, dict[str, list[int]]] = {bank: {} for bank in banks}
    for bank_index, bank in enumerate(banks):
        for layer in layers:
            values, counts = route_and_sample(
                states[layer],
                inputs[bank][layer],
                top_k=int(constants["top_k"]),
                samples_per_expert=int(constants["fit_samples_per_expert"]),
                seed=(
                    int(constants["sampling_seed_base"])
                    + int(constants["sampling_seed_bank_stride"]) * bank_index
                    + int(constants["sampling_seed_layer_stride"]) * layer
                ),
            )
            sampled[bank][layer] = values
            occupancy[bank][str(layer)] = counts

    cells: dict[str, dict[str, torch.Tensor]] = {
        endpoints[0]: {
            banks[0]: parent_gradients["raw_gradients"][banks[0]],
        },
        endpoints[1]: {
            banks[1]: parent_gradients["raw_gradients"][banks[1]],
        },
    }
    for endpoint, bank in ((endpoints[0], banks[1]), (endpoints[1], banks[0])):
        layer_gradients: list[torch.Tensor] = []
        for layer in layers:
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
                    coordinates["candidate"][endpoint][str(layer)],
                    sampled[bank][layer],
                    states[layer].c_fc,
                    states[layer].c_proj,
                )
            )
        cells.setdefault(endpoint, {})[bank] = torch.stack(layer_gradients)

    normalized = {
        endpoint: {
            bank: centered_normalized(cells[endpoint][bank]) for bank in banks
        }
        for endpoint in endpoints
    }
    rank = int(constants["diagnostic_rank"])
    thresholds = plan["frozen_gates"]
    comparisons = {
        "fixed_endpoint_a_data_a_vs_b": comparison_metrics(
            normalized[endpoints[0]][banks[0]],
            normalized[endpoints[0]][banks[1]],
            rank=rank,
            thresholds=thresholds,
        ),
        "fixed_endpoint_b_data_a_vs_b": comparison_metrics(
            normalized[endpoints[1]][banks[0]],
            normalized[endpoints[1]][banks[1]],
            rank=rank,
            thresholds=thresholds,
        ),
        "fixed_data_a_endpoint_a_vs_b": comparison_metrics(
            normalized[endpoints[0]][banks[0]],
            normalized[endpoints[1]][banks[0]],
            rank=rank,
            thresholds=thresholds,
        ),
        "fixed_data_b_endpoint_a_vs_b": comparison_metrics(
            normalized[endpoints[0]][banks[1]],
            normalized[endpoints[1]][banks[1]],
            rank=rank,
            thresholds=thresholds,
        ),
    }
    data_labels = (
        "fixed_endpoint_a_data_a_vs_b",
        "fixed_endpoint_b_data_a_vs_b",
    )
    endpoint_labels = (
        "fixed_data_a_endpoint_a_vs_b",
        "fixed_data_b_endpoint_a_vs_b",
    )
    finite = all_tensor_values_finite(
        {"cells": cells, "normalized": normalized, "comparisons": comparisons}
    )
    data_pass = finite and all(
        comparisons[label]["gates"]["all_pass"] for label in data_labels
    )
    endpoint_pass = finite and all(
        comparisons[label]["gates"]["all_pass"] for label in endpoint_labels
    )
    classification = classify(data_pass, endpoint_pass)

    args.output.mkdir(parents=True, exist_ok=False)
    crossed_path = args.output / "crossed_expert_angle_gradients.pt"
    torch.save(
        {
            "schema_version": "nanogpt_sparse_moe_cfc_same_gauge_crossed_gradients_v1",
            "layers": layers,
            "endpoints": endpoints,
            "data_banks": banks,
            "cells": cells,
            "centered_normalized_cells": normalized,
        },
        crossed_path,
    )
    result = {
        "schema_version": "nanogpt_sparse_moe_cfc_same_gauge_tangent_decomposition_result_v1",
        "classification": classification,
        "identity": {
            "git_commit": git_commit(Path(__file__).resolve().parents[2]),
            "plan_sha256": file_sha256(args.plan),
            "entrypoint_sha256": file_sha256(Path(__file__)),
            "parent_plan_sha256": file_sha256(args.parent_plan),
            "parent_coordinates_sha256": file_sha256(args.parent_coordinates),
            "parent_gradient_artifact_sha256": file_sha256(
                args.parent_gradient_artifact
            ),
            "parent_result_sha256": file_sha256(args.parent_result),
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
            "crossed_gradients_path": str(crossed_path),
            "crossed_gradients_sha256": file_sha256(crossed_path),
        },
        "occupancy": occupancy,
        "comparisons": comparisons,
        "fixed_endpoint_data_effect_passed": data_pass,
        "fixed_data_endpoint_effect_passed": endpoint_pass,
        "all_values_and_gradients_finite": finite,
        "authorization": {
            "conditional_tangent_functional_oracle_preregistration": (
                classification == "SAME_GAUGE_TANGENT_STABLE"
            ),
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
