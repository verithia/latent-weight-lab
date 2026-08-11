#!/usr/bin/env python3
"""Measure whether the frozen c_fc context action complements its static residual."""
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
from examples.nanogpt.analyze_sparse_moe_cfc_spectral_feature_oracle import (
    CompactCFCState,
    SpectralCFC,
    action_cosine,
    collect_protocol_inputs,
    routed_outputs,
)
from examples.nanogpt.analyze_sparse_moe_paired_alignment import file_sha256
from examples.nanogpt.analyze_sparse_moe_stepzero_task_gradient_oracle import (
    all_finite,
    layer_state_from_mapping,
    load_terminal_snapshot,
)


PLAN_SCHEMA = "nanogpt_sparse_moe_cfc_context_residual_action_audit_plan_v1"


def residual_direction_metrics(
    static: torch.Tensor,
    context_gated: torch.Tensor,
    target: torch.Tensor,
    *,
    transferred_alpha: float | None = None,
) -> dict[str, float]:
    residual = (target.float() - static.float()).reshape(-1).double()
    direction = (context_gated.float() - static.float()).reshape(-1).double()
    residual_energy = float(residual.square().sum())
    direction_energy = float(direction.square().sum())
    dot = float((residual * direction).sum())
    signed_cosine = dot / max(math.sqrt(residual_energy * direction_energy), 1e-30)
    optimal_alpha = max(0.0, dot / max(direction_energy, 1e-30))

    def score(alpha: float) -> tuple[float, float]:
        error = residual - float(alpha) * direction
        residual_recovery = 1.0 - float(error.square().sum()) / max(residual_energy, 1e-30)
        prediction = static.float().double() + float(alpha) * direction.reshape_as(static)
        target_energy = float(target.float().double().square().sum())
        total_recovery = 1.0 - float(
            (prediction - target.float().double()).square().sum()
        ) / max(target_energy, 1e-30)
        return residual_recovery, total_recovery

    optimal_residual_recovery, optimal_total_recovery = score(optimal_alpha)
    selected_alpha = optimal_alpha if transferred_alpha is None else float(transferred_alpha)
    transferred_residual_recovery, transferred_total_recovery = score(selected_alpha)
    return {
        "signed_cosine": signed_cosine,
        "optimal_positive_alpha": optimal_alpha,
        "optimal_residual_recovery": optimal_residual_recovery,
        "optimal_total_recovery": optimal_total_recovery,
        "transferred_alpha": selected_alpha,
        "transferred_alpha_residual_recovery": transferred_residual_recovery,
        "transferred_alpha_total_recovery": transferred_total_recovery,
        "residual_energy": residual_energy,
        "direction_energy": direction_energy,
    }


def result_authorization(passed: bool) -> dict[str, bool]:
    return {
        "direct_sum_preregistration": bool(passed),
        "new_coordinate_fit": False,
        "production_implementation": False,
        "mfu_preflight": False,
        "language_model_training": False,
        "larger_rung": False,
    }


def validate_plan(plan: dict[str, Any], plan_path: Path) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("c_fc context residual action plan schema mismatch")
    identity = plan["identity"]
    if identity.get("entrypoint_sha256") != file_sha256(Path(__file__)):
        raise ValueError("entrypoint hash is not sealed in the frozen plan")
    root = Path(__file__).resolve().parents[2]
    for relative, expected in identity["helper_sha256"].items():
        if file_sha256(root / relative) != expected:
            raise ValueError(f"helper hash drift: {relative}")
    if float(plan["frozen_operator"]["static_beta"]) != 0.0:
        raise ValueError("static beta must remain zero")
    if float(plan["frozen_operator"]["context_beta"]) != 1.0:
        raise ValueError("context beta must remain one")
    if file_sha256(plan_path) == "":
        raise AssertionError("unreachable empty plan hash")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--terminal-snapshot", required=True, type=Path)
    parser.add_argument("--parent-coordinates", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    validate_plan(plan, args.plan)
    source = plan["source"]
    if file_sha256(args.terminal_snapshot) != source["terminal_manifold_snapshot_sha256"]:
        raise ValueError("terminal snapshot hash disagrees with frozen plan")
    if file_sha256(args.parent_coordinates) != source["parent_coordinates_sha256"]:
        raise ValueError("parent coordinate hash disagrees with frozen plan")
    payload = load_terminal_snapshot(args.terminal_snapshot)
    if int(payload["next_iter"]) != int(source["next_iter"]):
        raise ValueError("terminal snapshot step disagrees with frozen plan")
    parent = torch.load(args.parent_coordinates, map_location="cpu", weights_only=False)
    if parent.get("schema_version") != "nanogpt_sparse_moe_cfc_context_modulated_spectral_coordinates_v1":
        raise ValueError("parent coordinate schema mismatch")

    model = load_model(args.terminal_snapshot, args.device)
    model.eval()
    inputs = collect_protocol_inputs(model, plan, args.data_dir, args.device)
    mapping = dict(model.named_parameters())
    layers = [int(layer) for layer in source["layers"]]
    states = {layer: layer_state_from_mapping(mapping, layer) for layer in layers}
    del mapping, model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    spec = plan["frozen_operator"]
    banks = [str(bank) for bank in source["banks"]]
    scores: dict[str, dict[str, Any]] = {}
    heldout_context_actions: dict[tuple[str, int], torch.Tensor] = {}
    for bank in banks:
        scores[bank] = {}
        for layer in layers:
            compact = parent["fitted"][bank]["control"][str(layer)]
            if not isinstance(compact, CompactCFCState):
                raise TypeError("parent control state type mismatch")
            common = {
                "experts": int(spec["num_experts"]),
                "input_width": int(spec["input_width"]),
                "hidden_width": int(spec["expert_hidden_width"]),
                "padded_width": int(spec["padded_width"]),
                "seed": int(spec["fixed_operator_seed"]),
                "layer": layer,
                "device": args.device,
                "context_seed_offset": int(spec["fixed_context_seed_offset"]),
            }
            static_operator = SpectralCFC(
                **common, context_beta=float(spec["static_beta"])
            )
            context_operator = SpectralCFC(
                **common, context_beta=float(spec["context_beta"])
            )
            torch.testing.assert_close(
                static_operator.signs, context_operator.signs, atol=0, rtol=0
            )

            actions: dict[str, dict[str, torch.Tensor]] = {}
            for split in (bank, "heldout"):
                static_action, target, _, _ = routed_outputs(
                    states[layer],
                    inputs[split][layer],
                    static_operator,
                    compact,
                    spectral=True,
                    top_k=int(spec["top_k"]),
                )
                gated_action, gated_target, _, _ = routed_outputs(
                    states[layer],
                    inputs[split][layer],
                    context_operator,
                    compact,
                    spectral=True,
                    top_k=int(spec["top_k"]),
                )
                if not torch.equal(target, gated_target):
                    raise RuntimeError("static and context targets drift")
                actions[split] = {
                    "static": static_action,
                    "gated": gated_action,
                    "target": target,
                }
            discovery_metrics = residual_direction_metrics(
                actions[bank]["static"],
                actions[bank]["gated"],
                actions[bank]["target"],
            )
            heldout_metrics = residual_direction_metrics(
                actions["heldout"]["static"],
                actions["heldout"]["gated"],
                actions["heldout"]["target"],
                transferred_alpha=discovery_metrics["optimal_positive_alpha"],
            )
            scores[bank][str(layer)] = {
                "discovery": discovery_metrics,
                "heldout": heldout_metrics,
            }
            heldout_context_actions[(bank, layer)] = (
                actions["heldout"]["gated"] - actions["heldout"]["static"]
            )

    agreement = {
        str(layer): action_cosine(
            heldout_context_actions[(banks[0], layer)],
            heldout_context_actions[(banks[1], layer)],
        )
        for layer in layers
    }
    agreement_mean = sum(agreement.values()) / len(agreement)
    frozen = plan["frozen_gates"]
    summaries: dict[str, dict[str, float | bool]] = {}
    gates: dict[str, dict[str, bool]] = {}
    for bank in banks:
        heldout_optimal = [
            float(scores[bank][str(layer)]["heldout"]["optimal_residual_recovery"])
            for layer in layers
        ]
        heldout_transferred = [
            float(scores[bank][str(layer)]["heldout"]["transferred_alpha_residual_recovery"])
            for layer in layers
        ]
        discovery_alphas = [
            float(scores[bank][str(layer)]["discovery"]["optimal_positive_alpha"])
            for layer in layers
        ]
        summaries[bank] = {
            "heldout_optimal_residual_recovery_mean": sum(heldout_optimal) / len(heldout_optimal),
            "heldout_optimal_residual_recovery_minimum_layer": min(heldout_optimal),
            "heldout_transferred_alpha_residual_recovery_mean": sum(heldout_transferred) / len(heldout_transferred),
            "heldout_transferred_alpha_residual_recovery_minimum_layer": min(heldout_transferred),
            "minimum_discovery_optimal_alpha": min(discovery_alphas),
        }
        gates[bank] = {
            "optimal_mean_pass": summaries[bank]["heldout_optimal_residual_recovery_mean"] >= float(frozen["heldout_optimal_residual_recovery_mean_min_each_bank"]),
            "optimal_every_layer_pass": summaries[bank]["heldout_optimal_residual_recovery_minimum_layer"] >= float(frozen["heldout_optimal_residual_recovery_every_layer_min_each_bank"]),
            "transferred_mean_pass": summaries[bank]["heldout_transferred_alpha_residual_recovery_mean"] >= float(frozen["heldout_transferred_alpha_residual_recovery_mean_min_each_bank"]),
            "discovery_alpha_positive_pass": summaries[bank]["minimum_discovery_optimal_alpha"] > 0.0,
            "context_action_agreement_pass": agreement_mean >= float(frozen["heldout_context_action_cross_bank_cosine_mean_min"]),
        }

    finite = all_finite(
        {"scores": scores, "summaries": summaries, "agreement": agreement}
    )
    for bank in banks:
        gates[bank]["finite_pass"] = finite
        gates[bank]["all_pass"] = all(gates[bank].values())
    passed = finite and all(gates[bank]["all_pass"] for bank in banks)

    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema_version": "nanogpt_sparse_moe_cfc_context_residual_action_audit_result_v1",
        "classification": (
            "CONTEXT_RESIDUAL_ACTION_COMPLEMENTARY"
            if passed
            else "CONTEXT_RESIDUAL_ACTION_REJECTED"
        ),
        "passed": passed,
        "identity": {
            "git_commit": git_commit(Path(__file__).resolve().parents[2]),
            "plan_sha256": file_sha256(args.plan),
            "entrypoint_sha256": file_sha256(Path(__file__)),
            "terminal_snapshot_sha256": file_sha256(args.terminal_snapshot),
            "parent_coordinates_sha256": file_sha256(args.parent_coordinates),
            "dataset_manifest_sha256": source["dataset_manifest_sha256"],
        },
        "execution": {
            "device": args.device,
            "wall_seconds": time.time() - started,
            "checkpoint_updates": 0,
            "coordinate_updates": 0,
            "maximum_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated())
                if args.device.startswith("cuda")
                else 0
            ),
        },
        "scores": scores,
        "summaries": summaries,
        "heldout_context_action_cross_bank_cosine": {
            "mean": agreement_mean,
            "by_layer": agreement,
        },
        "gates": gates,
        "all_values_finite": finite,
        "authorization": result_authorization(passed),
    }
    result_path = args.output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
