#!/usr/bin/env python3
"""Gate the maximum 200x-valid asymmetric KFAC full-MLP chart."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_mlp_activation_update_alignment import git_commit
from examples.nanogpt.analyze_residual_compatibility import fixed_validation_batches
from examples.nanogpt.analyze_sparse_moe_cfc_spectral_feature_oracle import (
    action_cosine,
)
from examples.nanogpt.analyze_sparse_moe_paired_alignment import (
    collect_inputs,
    file_sha256,
)
from examples.nanogpt.analyze_sparse_moe_paired_atom_oracle import (
    _atom_views,
    _state_chord,
    _state_from_atom_views,
    energy_recovery,
    project_rows,
)
from examples.nanogpt.analyze_sparse_moe_rolling_tangent_oracle import (
    LayerState,
    recovery_fraction,
    sparse_moe_output,
)
from examples.nanogpt.analyze_sparse_moe_stepzero_kfac_factor_oracle import (
    build_kfac_basis,
    collect_geometry,
)
from examples.nanogpt.analyze_sparse_moe_stepzero_task_gradient_oracle import (
    all_finite,
    layer_state_from_mapping,
    load_terminal_snapshot,
    model_from_exact_stepzero,
    row_span_overlap,
)


PLAN_SCHEMA = "nanogpt_sparse_moe_stepzero_kfac_maxbudget_oracle_plan_v1"


def coordinate_compression(incoming_rank: int, outgoing_rank: int) -> float:
    coordinates = int(incoming_rank) + int(outgoing_rank)
    if coordinates <= 0:
        raise ValueError("paired coordinate count must be positive")
    return 1536.0 / coordinates


def reconstruct_asymmetric_family(
    left: LayerState,
    target_chord: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    directions: list[LayerState],
    *,
    incoming_rank: int,
    outgoing_rank: int,
    router_rank: int,
    ridge_ratio: float,
    device: str,
) -> tuple[LayerState, dict[str, float]]:
    required = max(int(incoming_rank), int(outgoing_rank), int(router_rank))
    if min(incoming_rank, outgoing_rank, router_rank) <= 0:
        raise ValueError("all asymmetric ranks must be positive")
    if len(directions) < required:
        raise ValueError("insufficient KFAC directions for asymmetric family")

    target_router, target_fc, target_proj = (
        value.to(device=device) for value in target_chord
    )
    views = [_atom_views(direction) for direction in directions[:required]]
    router_basis = torch.stack(
        [views[index][0] for index in range(router_rank)]
    ).to(device)
    fc_basis = torch.stack(
        [views[index][1] for index in range(incoming_rank)]
    ).to(device)
    proj_basis = torch.stack(
        [views[index][2] for index in range(outgoing_rank)]
    ).to(device)
    projected_router, _ = project_rows(target_router, router_basis, ridge_ratio)
    projected_fc, _ = project_rows(target_fc, fc_basis, ridge_ratio)
    projected_proj, _ = project_rows(target_proj, proj_basis, ridge_ratio)

    left_router, left_fc, left_proj = (
        value.to(device=device) for value in _atom_views(left)
    )
    reconstructed = _state_from_atom_views(
        left_router + projected_router,
        left_fc + projected_fc,
        left_proj + projected_proj,
    )
    return reconstructed, {
        "router_parameter_recovery": energy_recovery(
            projected_router, target_router
        ),
        "c_fc_parameter_recovery": energy_recovery(projected_fc, target_fc),
        "c_proj_parameter_recovery": energy_recovery(
            projected_proj, target_proj
        ),
        "paired_parameter_recovery": energy_recovery(
            torch.cat((projected_fc, projected_proj), dim=-1),
            torch.cat((target_fc, target_proj), dim=-1),
        ),
    }


def asymmetric_overlap(
    left: list[LayerState],
    right: list[LayerState],
    *,
    incoming_rank: int,
    outgoing_rank: int,
) -> dict[str, float]:
    required = max(incoming_rank, outgoing_rank)
    if len(left) < required or len(right) < required:
        raise ValueError("insufficient directions for asymmetric overlap")
    left_views = [_atom_views(value) for value in left[:required]]
    right_views = [_atom_views(value) for value in right[:required]]
    incoming = row_span_overlap(
        torch.stack([left_views[index][1] for index in range(incoming_rank)]),
        torch.stack([right_views[index][1] for index in range(incoming_rank)]),
    )
    outgoing = row_span_overlap(
        torch.stack([left_views[index][2] for index in range(outgoing_rank)]),
        torch.stack([right_views[index][2] for index in range(outgoing_rank)]),
    )
    return {"incoming": incoming, "outgoing": outgoing}


def result_authorization(passed: bool) -> dict[str, bool]:
    return {
        "structured_basis_approximation_preregistration": bool(passed),
        "dense_or_lora_basis": False,
        "production_implementation": False,
        "mfu_preflight": False,
        "language_model_training": False,
        "generated_experts": False,
        "larger_rung": False,
    }


def validate_plan(plan: dict[str, Any], plan_path: Path) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("maximum-budget KFAC plan schema mismatch")
    identity = plan["identity"]
    if identity.get("entrypoint_sha256") != file_sha256(Path(__file__)):
        raise ValueError("entrypoint hash is not sealed in the frozen plan")
    root = Path(__file__).resolve().parents[2]
    for relative, expected in identity["helper_sha256"].items():
        if file_sha256(root / relative) != expected:
            raise ValueError(f"helper hash drift: {relative}")
    source = plan["source"]
    if file_sha256(root / source["parent_normalized_result"]) != source[
        "parent_normalized_result_sha256"
    ]:
        raise ValueError("parent normalized result hash drift")
    control = plan["families"]["control_separate_3_plus_3"]
    candidate = plan["families"]["candidate_separate_3_plus_4"]
    if (
        int(control["incoming_rank"]),
        int(control["outgoing_rank"]),
        int(control["router_rank"]),
    ) != (3, 3, 3):
        raise ValueError("control rank allocation drift")
    if (
        int(candidate["incoming_rank"]),
        int(candidate["outgoing_rank"]),
        int(candidate["router_rank"]),
    ) != (3, 4, 3):
        raise ValueError("candidate rank allocation drift")
    expected = coordinate_compression(3, 4)
    if abs(expected - float(candidate["paired_coordinate_compression_ratio"])) > 1e-12:
        raise ValueError("candidate compression accounting drift")
    if expected < 200.0:
        raise ValueError("candidate violates the 200x coordinate floor")
    if file_sha256(plan_path) == "":
        raise AssertionError("unreachable empty plan hash")


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
    validate_plan(plan, args.plan)
    source = plan["source"]
    if file_sha256(args.terminal_snapshot) != source[
        "terminal_manifold_snapshot_sha256"
    ]:
        raise ValueError("terminal snapshot hash disagrees with frozen plan")
    payload = load_terminal_snapshot(args.terminal_snapshot)
    if int(payload["next_iter"]) != int(source["next_iter"]):
        raise ValueError("terminal snapshot step disagrees with frozen plan")
    layers = [int(layer) for layer in source["layers"]]
    model = model_from_exact_stepzero(
        payload, int(source["model_seed"]), args.device
    )
    initial_mapping = dict(model.named_parameters())
    initial = {
        layer: layer_state_from_mapping(initial_mapping, layer) for layer in layers
    }
    terminal = {
        layer: layer_state_from_mapping(payload["model"], layer) for layer in layers
    }

    banks: dict[str, dict[int, list[LayerState]]] = {}
    geometry_rows: list[dict[str, Any]] = []
    minimum_assignments = int(
        plan["frozen_gates"]["minimum_expert_assignments_each_bank"]
    )
    for spec in plan["geometry_protocol"]["discovery_banks"]:
        batches = fixed_validation_batches(
            args.data_dir,
            int(spec["batch_size"]),
            int(spec["block_size"]) + 1,
            int(spec["batches"]),
            int(spec["seed"]),
        )
        inputs, errors, loss = collect_geometry(
            model, batches, layers, args.device
        )
        layer_banks: dict[int, list[LayerState]] = {}
        for layer in layers:
            directions, rows = build_kfac_basis(
                initial[layer],
                inputs[layer],
                errors[layer],
                rank=4,
                ridge_ratio=args.ridge_ratio,
                minimum_assignments=minimum_assignments,
                device=args.device,
            )
            layer_banks[layer] = directions
            for row in rows:
                row.update(
                    {"bank": spec["name"], "layer": layer, "loss": loss}
                )
            geometry_rows.extend(rows)
        banks[spec["name"]] = layer_banks

    heldout = plan["geometry_protocol"]["heldout"]
    heldout_batches = fixed_validation_batches(
        args.data_dir,
        int(heldout["batch_size"]),
        int(heldout["block_size"]),
        int(heldout["batches"]),
        int(heldout["seed"]),
    )
    model.eval()
    heldout_inputs = collect_inputs(
        model,
        heldout_batches,
        layers,
        int(heldout["activation_sample_cap_per_layer"]),
        args.device,
    )
    del model, initial_mapping
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    family_specs = plan["families"]
    scores: dict[str, dict[str, dict[str, Any]]] = {}
    candidate_actions: dict[tuple[str, int], torch.Tensor] = {}
    for bank_name, bank in banks.items():
        scores[bank_name] = {}
        for layer in layers:
            left, right = initial[layer], terminal[layer]
            chord = _state_chord(right, left)
            x = heldout_inputs[layer].to(args.device)
            base_output = sparse_moe_output(left.to(args.device), x, 2)
            target_action = (
                sparse_moe_output(right.to(args.device), x, 2) - base_output
            )
            scores[bank_name][str(layer)] = {}
            for family_name, family in family_specs.items():
                reconstructed, parameter_metrics = reconstruct_asymmetric_family(
                    left,
                    chord,
                    bank[layer],
                    incoming_rank=int(family["incoming_rank"]),
                    outgoing_rank=int(family["outgoing_rank"]),
                    router_rank=int(family["router_rank"]),
                    ridge_ratio=args.ridge_ratio,
                    device=args.device,
                )
                action = (
                    sparse_moe_output(reconstructed, x, 2) - base_output
                )
                scores[bank_name][str(layer)][family_name] = {
                    "heldout_exact_recovery": recovery_fraction(
                        action, target_action
                    ),
                    **parameter_metrics,
                }
                if family_name == "candidate_separate_3_plus_4":
                    candidate_actions[(bank_name, layer)] = action.detach().cpu()

    bank_names = list(banks)
    overlaps = {
        str(layer): asymmetric_overlap(
            banks[bank_names[0]][layer],
            banks[bank_names[1]][layer],
            incoming_rank=3,
            outgoing_rank=4,
        )
        for layer in layers
    }
    overlap_summary = {
        side: sum(overlaps[str(layer)][side] for layer in layers) / len(layers)
        for side in ("incoming", "outgoing")
    }
    action_agreement = {
        str(layer): action_cosine(
            candidate_actions[(bank_names[0], layer)],
            candidate_actions[(bank_names[1], layer)],
        )
        for layer in layers
    }
    action_agreement_mean = sum(action_agreement.values()) / len(layers)
    occupancy = {
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
                    if row["bank"] == bank_name
                    and int(row["layer"]) == layer
                )
                for layer in layers
            },
        }
        for bank_name in banks
    }

    summaries: dict[str, dict[str, float]] = {}
    gates: dict[str, dict[str, bool]] = {}
    frozen = plan["frozen_gates"]
    for bank_name in banks:
        control = [
            float(
                scores[bank_name][str(layer)]["control_separate_3_plus_3"][
                    "heldout_exact_recovery"
                ]
            )
            for layer in layers
        ]
        candidate = [
            float(
                scores[bank_name][str(layer)]["candidate_separate_3_plus_4"][
                    "heldout_exact_recovery"
                ]
            )
            for layer in layers
        ]
        gains = [right - left for left, right in zip(control, candidate)]
        summaries[bank_name] = {
            "control_mean": sum(control) / len(control),
            "candidate_mean": sum(candidate) / len(candidate),
            "candidate_minimum_layer": min(candidate),
            "candidate_minus_control_mean": sum(gains) / len(gains),
            "candidate_minus_control_minimum_layer": min(gains),
        }
        gates[bank_name] = {
            "candidate_mean_pass": summaries[bank_name]["candidate_mean"]
            >= float(
                frozen["candidate_heldout_exact_recovery_mean_min_each_bank"]
            ),
            "candidate_every_layer_pass": summaries[bank_name][
                "candidate_minimum_layer"
            ]
            >= float(
                frozen["candidate_heldout_exact_recovery_every_layer_min_each_bank"]
            ),
            "candidate_minus_control_pass": summaries[bank_name][
                "candidate_minus_control_mean"
            ]
            >= float(frozen["candidate_minus_control_mean_min_each_bank"]),
            "candidate_action_agreement_pass": action_agreement_mean
            >= float(frozen["candidate_cross_bank_action_cosine_mean_min"]),
            "incoming_overlap_pass": overlap_summary["incoming"]
            >= float(frozen["incoming_cross_bank_subspace_overlap_mean_min"]),
            "outgoing_overlap_pass": overlap_summary["outgoing"]
            >= float(frozen["outgoing_cross_bank_subspace_overlap_mean_min"]),
            "occupancy_pass": occupancy[bank_name]["minimum"]
            >= minimum_assignments,
        }

    finite = all_finite(
        {
            "scores": scores,
            "summaries": summaries,
            "overlaps": overlaps,
            "action_agreement": action_agreement,
            "geometry_rows": geometry_rows,
        }
    )
    for bank_name in banks:
        gates[bank_name]["finite_pass"] = finite
        gates[bank_name]["all_pass"] = all(gates[bank_name].values())
    passed = finite and all(gates[name]["all_pass"] for name in banks)

    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema_version": "nanogpt_sparse_moe_stepzero_kfac_maxbudget_oracle_result_v1",
        "classification": (
            "MAXBUDGET_KFAC_REPRESENTABILITY_PASSED"
            if passed
            else "MAXBUDGET_KFAC_REPRESENTABILITY_REJECTED"
        ),
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
            "coordinate_updates": 0,
            "maximum_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated())
                if args.device.startswith("cuda")
                else 0
            ),
        },
        "coordinate_accounting": {
            name: {
                "incoming_rank": int(spec["incoming_rank"]),
                "outgoing_rank": int(spec["outgoing_rank"]),
                "router_rank": int(spec["router_rank"]),
                "paired_coordinate_compression_ratio": coordinate_compression(
                    int(spec["incoming_rank"]), int(spec["outgoing_rank"])
                ),
            }
            for name, spec in family_specs.items()
        },
        "scores": scores,
        "summaries": summaries,
        "candidate_cross_bank_action_cosine": {
            "mean": action_agreement_mean,
            "by_layer": action_agreement,
        },
        "candidate_cross_bank_subspace_overlap": {
            "mean": overlap_summary,
            "by_layer": overlaps,
        },
        "expert_occupancy": occupancy,
        "gates": gates,
        "all_values_finite": finite,
        "authorization": result_authorization(passed),
    }
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
