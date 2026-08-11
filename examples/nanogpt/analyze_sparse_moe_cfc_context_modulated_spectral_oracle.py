#!/usr/bin/env python3
"""Gate routed-input modulation of the compact spectral sparse-MoE c_fc map."""
from __future__ import annotations

import argparse
import json
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
    fit_compact_state,
    route_and_sample,
    routed_outputs,
)
from examples.nanogpt.analyze_sparse_moe_paired_alignment import file_sha256
from examples.nanogpt.analyze_sparse_moe_rolling_tangent_oracle import (
    recovery_fraction,
)
from examples.nanogpt.analyze_sparse_moe_stepzero_task_gradient_oracle import (
    all_finite,
    layer_state_from_mapping,
    load_terminal_snapshot,
)


PLAN_SCHEMA = "nanogpt_sparse_moe_cfc_context_modulated_spectral_oracle_plan_v1"


def result_authorization(passed: bool) -> dict[str, bool]:
    return {
        "production_implementation": bool(passed),
        "initialization_fit_shadow": bool(passed),
        "mfu_preflight": False,
        "language_model_training": False,
        "larger_rung": False,
        "generated_cproj": False,
    }


def validate_plan(plan: dict[str, Any], plan_path: Path) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("context-modulated spectral c_fc plan schema mismatch")
    identity = plan["identity"]
    if identity.get("entrypoint_sha256") != file_sha256(Path(__file__)):
        raise ValueError("entrypoint hash is not sealed in the frozen plan")
    root = Path(__file__).resolve().parents[2]
    for relative, expected in identity["helper_sha256"].items():
        if file_sha256(root / relative) != expected:
            raise ValueError(f"helper hash drift: {relative}")
    candidate = plan["candidate"]
    expected = (
        2 * int(candidate["padded_width"])
        + int(candidate["expert_hidden_width"])
    )
    if expected != int(candidate["trainable_coordinates_per_expert"]):
        raise ValueError("compact c_fc coordinate accounting drift")
    if float(candidate["context_beta"]) != 1.0:
        raise ValueError("the sole registered treatment requires beta=1")
    if float(plan["control"]["context_beta"]) != 0.0:
        raise ValueError("the matched static control requires beta=0")
    if file_sha256(plan_path) == "":
        raise AssertionError("unreachable empty plan hash")


def fit_is_healthy(diagnostics: dict[str, float | int]) -> bool:
    initial = float(diagnostics["initial_loss"])
    final = float(diagnostics["final_loss"])
    minimum = float(diagnostics["minimum_loss"])
    return all(torch.isfinite(torch.tensor(value)) for value in (initial, final, minimum)) and final < initial


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--terminal-snapshot", required=True, type=Path)
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
    payload = load_terminal_snapshot(args.terminal_snapshot)
    if int(payload["next_iter"]) != int(source["next_iter"]):
        raise ValueError("terminal snapshot step disagrees with frozen plan")

    model = load_model(args.terminal_snapshot, args.device)
    model.eval()
    inputs = collect_protocol_inputs(model, plan, args.data_dir, args.device)
    terminal_mapping = dict(model.named_parameters())
    layers = [int(layer) for layer in source["layers"]]
    states = {
        layer: layer_state_from_mapping(terminal_mapping, layer)
        for layer in layers
    }
    del terminal_mapping, model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    candidate = plan["candidate"]
    control = plan["control"]
    fit_spec = plan["fit_protocol"]
    samples_per_expert = int(plan["data_protocol"]["fit_samples_per_expert"])
    banks = [spec["name"] for spec in plan["data_protocol"]["discovery_banks"]]
    fitted: dict[str, dict[str, dict[str, CompactCFCState]]] = {}
    fit_diagnostics: dict[str, dict[str, Any]] = {}
    occupancy: dict[str, dict[str, list[int]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    heldout_actions: dict[tuple[str, int], torch.Tensor] = {}

    common_operator = {
        "experts": int(candidate["num_experts"]),
        "input_width": int(candidate["input_width"]),
        "hidden_width": int(candidate["expert_hidden_width"]),
        "padded_width": int(candidate["padded_width"]),
        "seed": int(candidate["fixed_operator_seed"]),
        "device": args.device,
        "context_seed_offset": int(candidate["fixed_context_seed_offset"]),
    }

    for bank_index, bank in enumerate(banks):
        fitted[bank] = {"candidate": {}, "control": {}}
        fit_diagnostics[bank] = {}
        occupancy[bank] = {}
        summaries[bank] = {}
        for layer in layers:
            state = states[layer]
            sampled, counts = route_and_sample(
                state,
                inputs[bank][layer],
                top_k=int(candidate["top_k"]),
                samples_per_expert=samples_per_expert,
                seed=20260932 + 1009 * bank_index + 17 * layer,
            )
            occupancy[bank][str(layer)] = counts
            treatment_operator = SpectralCFC(
                **common_operator,
                layer=layer,
                context_beta=float(candidate["context_beta"]),
            )
            control_operator = SpectralCFC(
                **common_operator,
                layer=layer,
                context_beta=float(control["context_beta"]),
            )
            torch.testing.assert_close(
                treatment_operator.signs,
                control_operator.signs,
                atol=0,
                rtol=0,
            )
            treatment_state, treatment_diag = fit_compact_state(
                treatment_operator,
                sampled,
                state.c_fc,
                state.c_proj,
                spectral=True,
                steps=int(fit_spec["steps"]),
                learning_rate=float(fit_spec["learning_rate"]),
                weight_decay=float(fit_spec["weight_decay"]),
            )
            control_state, control_diag = fit_compact_state(
                control_operator,
                sampled,
                state.c_fc,
                state.c_proj,
                spectral=True,
                steps=int(fit_spec["steps"]),
                learning_rate=float(fit_spec["learning_rate"]),
                weight_decay=float(fit_spec["weight_decay"]),
            )
            fitted[bank]["candidate"][str(layer)] = treatment_state
            fitted[bank]["control"][str(layer)] = control_state
            fit_diagnostics[bank][str(layer)] = {
                "candidate": treatment_diag,
                "control": control_diag,
                "candidate_healthy": fit_is_healthy(treatment_diag),
                "control_healthy": fit_is_healthy(control_diag),
            }

            predicted, target, expert_recovery, pre_recovery = routed_outputs(
                state,
                inputs["heldout"][layer],
                treatment_operator,
                treatment_state,
                spectral=True,
                top_k=int(candidate["top_k"]),
            )
            control_predicted, control_target, _, _ = routed_outputs(
                state,
                inputs["heldout"][layer],
                control_operator,
                control_state,
                spectral=True,
                top_k=int(candidate["top_k"]),
            )
            if not torch.equal(target, control_target):
                raise RuntimeError("candidate and control targets drift")
            treatment_recovery = recovery_fraction(predicted, target)
            control_recovery = recovery_fraction(control_predicted, target)
            summaries[bank][str(layer)] = {
                "mixture_recovery": treatment_recovery,
                "control_mixture_recovery": control_recovery,
                "candidate_minus_control_recovery": treatment_recovery - control_recovery,
                "expert_recovery": expert_recovery,
                "minimum_expert_recovery": min(expert_recovery),
                "pregelu_recovery": pre_recovery,
                "minimum_pregelu_recovery": min(pre_recovery),
            }
            heldout_actions[(bank, layer)] = predicted

    agreement_by_layer = {
        str(layer): action_cosine(
            heldout_actions[(banks[0], layer)],
            heldout_actions[(banks[1], layer)],
        )
        for layer in layers
    }
    agreement_mean = sum(agreement_by_layer.values()) / len(agreement_by_layer)
    frozen = plan["frozen_gates"]
    bank_gates: dict[str, dict[str, bool]] = {}
    for bank in banks:
        rows = [summaries[bank][str(layer)] for layer in layers]
        mixture = [float(row["mixture_recovery"]) for row in rows]
        improvements = [float(row["candidate_minus_control_recovery"]) for row in rows]
        minimum_expert = min(float(row["minimum_expert_recovery"]) for row in rows)
        minimum_occupancy = min(min(occupancy[bank][str(layer)]) for layer in layers)
        all_fit_losses_decrease = all(
            bool(fit_diagnostics[bank][str(layer)][kind + "_healthy"])
            for layer in layers
            for kind in ("candidate", "control")
        )
        aggregate = {
            "mixture_recovery_mean": sum(mixture) / len(mixture),
            "mixture_recovery_minimum_layer": min(mixture),
            "minimum_expert_recovery": minimum_expert,
            "candidate_minus_control_recovery_mean": sum(improvements) / len(improvements),
            "minimum_discovery_assignments": minimum_occupancy,
            "all_fit_losses_decrease": all_fit_losses_decrease,
        }
        summaries[bank]["aggregate"] = aggregate
        bank_gates[bank] = {
            "mean_recovery_pass": aggregate["mixture_recovery_mean"] >= float(frozen["heldout_mixture_recovery_mean_min_each_bank"]),
            "every_layer_pass": aggregate["mixture_recovery_minimum_layer"] >= float(frozen["heldout_mixture_recovery_every_layer_min_each_bank"]),
            "every_expert_pass": minimum_expert >= float(frozen["heldout_expert_recovery_min_each_bank"]),
            "context_gain_pass": aggregate["candidate_minus_control_recovery_mean"] >= float(frozen["candidate_minus_same_run_static_recovery_mean_min_each_bank"]),
            "occupancy_pass": minimum_occupancy >= int(frozen["minimum_discovery_assignments_per_expert"]),
            "fit_health_pass": all_fit_losses_decrease,
            "action_agreement_pass": agreement_mean >= float(frozen["heldout_bank_action_cosine_mean_min"]),
        }

    finite = all_finite(
        {
            "summaries": summaries,
            "fit_diagnostics": fit_diagnostics,
            "agreement_by_layer": agreement_by_layer,
        }
    )
    for bank in banks:
        bank_gates[bank]["finite_pass"] = finite
        bank_gates[bank]["all_pass"] = all(bank_gates[bank].values())
    fit_healthy = all(
        bool(bank_gates[bank]["fit_health_pass"]) for bank in banks
    )
    passed = fit_healthy and all(bank_gates[bank]["all_pass"] for bank in banks)
    classification = (
        "CONTEXT_MODULATED_SPECTRAL_CFC_REPRESENTABILITY_PASSES"
        if passed
        else (
            "CONTEXT_MODULATED_SPECTRAL_CFC_OPTIMIZATION_INCONCLUSIVE"
            if not fit_healthy
            else "CONTEXT_MODULATED_SPECTRAL_CFC_REPRESENTABILITY_REJECTED"
        )
    )

    args.output.mkdir(parents=True, exist_ok=False)
    coordinates_path = args.output / "compact_coordinates.pt"
    torch.save(
        {
            "schema_version": "nanogpt_sparse_moe_cfc_context_modulated_spectral_coordinates_v1",
            "fitted": fitted,
        },
        coordinates_path,
    )
    result = {
        "schema_version": "nanogpt_sparse_moe_cfc_context_modulated_spectral_oracle_result_v1",
        "classification": classification,
        "passed": passed,
        "identity": {
            "git_commit": git_commit(Path(__file__).resolve().parents[2]),
            "plan_sha256": file_sha256(args.plan),
            "entrypoint_sha256": file_sha256(Path(__file__)),
            "terminal_snapshot_sha256": file_sha256(args.terminal_snapshot),
            "dataset_manifest_sha256": source["dataset_manifest_sha256"],
        },
        "execution": {
            "device": args.device,
            "wall_seconds": time.time() - started,
            "checkpoint_updates": 0,
            "coordinates_path": str(coordinates_path),
            "coordinates_sha256": file_sha256(coordinates_path),
            "maximum_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated())
                if args.device.startswith("cuda")
                else 0
            ),
        },
        "occupancy": occupancy,
        "fit_diagnostics": fit_diagnostics,
        "summaries": summaries,
        "heldout_bank_action_cosine": {
            "mean": agreement_mean,
            "by_layer": agreement_by_layer,
        },
        "gates": bank_gates,
        "all_values_finite": finite,
        "authorization": result_authorization(passed),
    }
    result_path = args.output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
