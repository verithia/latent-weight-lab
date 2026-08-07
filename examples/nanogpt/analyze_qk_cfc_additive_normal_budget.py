#!/usr/bin/env python3
"""Measure the additive low-rank state needed beyond bilateral c_fc tangents."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_mlp_cfc_directed_product_terminal import (
    cfc_modules,
    directed_optimizer,
)
from examples.nanogpt.analyze_mlp_cfc_exact_current_matcher import (
    file_sha256,
    git_commit,
    load_model_and_optimizer,
)
from examples.nanogpt.analyze_qk_cfc_state_side_decomposition import (
    MATRIX_SHAPE,
    OUTPUT_SCHEDULE,
    STATE_NAMES,
    candidate_predictions,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_SCHEMA = "mai_124m_qk_cfc_additive_normal_budget_plan_v1"
RESULT_SCHEMA = "mai_124m_qk_cfc_additive_normal_budget_result_v1"
PARENT_SCHEMA = "mai_124m_qk_cfc_state_side_decomposition_sealed_result_v1"
BILATERAL_FAMILIES = ("output6_then_input", "input_then_output6")
LATE_LAYERS = (8, 9, 10, 11)
RANK_FRONTIER = (0, 16, 32, 64, 96, 128, 160, 192, 256, 384, 512, 768)


def recovery_from_spectrum(
    target_energy: float, residual_spectrum: torch.Tensor, rank: int
) -> float:
    values = residual_spectrum.double().flatten().clamp_min(0)
    rank = min(max(int(rank), 0), values.numel())
    remaining = values[rank:].sum().item()
    return 1.0 - remaining / max(float(target_energy), 1e-30)


def minimum_rank_for_recovery(
    target_energy: float, residual_spectrum: torch.Tensor, threshold: float
) -> int:
    for rank in range(residual_spectrum.numel() + 1):
        if recovery_from_spectrum(target_energy, residual_spectrum, rank) >= threshold:
            return rank
    raise RuntimeError("full-rank completion did not reach threshold")


def minimum_material_rank_allocation(
    layer_rows: list[dict[str, Any]],
    *,
    aggregate_threshold: float,
    late_layer_threshold: float,
) -> dict[str, Any]:
    """Allocate equal-cost singular components after enforcing each late gate."""
    ranks = [0] * len(layer_rows)
    for layer in LATE_LAYERS:
        row = layer_rows[layer]
        ranks[layer] = minimum_rank_for_recovery(
            row["target_energy"], row["residual_spectrum"], late_layer_threshold
        )

    target_total = sum(float(row["target_energy"]) for row in layer_rows)
    maximum_residual = (1.0 - aggregate_threshold) * target_total
    remaining = sum(
        float(row["residual_spectrum"][ranks[layer] :].double().sum())
        for layer, row in enumerate(layer_rows)
    )
    available: list[tuple[float, int, int]] = []
    for layer, row in enumerate(layer_rows):
        for index, energy in enumerate(row["residual_spectrum"].tolist()):
            if index >= ranks[layer]:
                available.append((float(energy), layer, index))
    available.sort(reverse=True)
    for energy, layer, index in available:
        if remaining <= maximum_residual:
            break
        if index != ranks[layer]:
            continue
        ranks[layer] += 1
        remaining -= energy

    all_recovery = 1.0 - remaining / max(target_total, 1e-30)
    late_recoveries = [
        recovery_from_spectrum(
            layer_rows[layer]["target_energy"],
            layer_rows[layer]["residual_spectrum"],
            ranks[layer],
        )
        for layer in LATE_LAYERS
    ]
    return {
        "ranks_by_layer": ranks,
        "total_rank": sum(ranks),
        "all_energy_weighted_recovery": all_recovery,
        "late_minimum_layer_recovery": min(late_recoveries),
        "late_layer_recoveries": late_recoveries,
    }


def fixed_rank_summary(layer_rows: list[dict[str, Any]], rank: int) -> dict[str, float]:
    target_total = sum(float(row["target_energy"]) for row in layer_rows)
    remaining_total = sum(
        float(row["residual_spectrum"][rank:].double().sum()) for row in layer_rows
    )
    return {
        "all_energy_weighted_recovery": 1.0
        - remaining_total / max(target_total, 1e-30),
        "late_minimum_layer_recovery": min(
            recovery_from_spectrum(
                layer_rows[layer]["target_energy"],
                layer_rows[layer]["residual_spectrum"],
                rank,
            )
            for layer in LATE_LAYERS
        ),
    }


def classify(
    family_accounting: dict[str, dict[str, Any]], rule: dict[str, Any]
) -> dict[str, Any]:
    selected = min(
        BILATERAL_FAMILIES,
        key=lambda family: float(family_accounting[family]["joint_byte_ratio_to_dense"]),
    )
    ratio = float(family_accounting[selected]["joint_byte_ratio_to_dense"])
    compact = float(rule["maximum_compact_joint_byte_ratio"])
    dense = float(rule["maximum_dense_equivalent_joint_byte_ratio"])
    if ratio <= compact:
        classification = "LOW_RANK_NORMAL_COMPLETION_PLAUSIBLE"
    elif ratio <= dense:
        classification = "DENSE_SCALE_NORMAL_COMPLETION"
    else:
        classification = "NORMAL_COMPLETION_EXCEEDS_DENSE_STATE"
    return {
        "classification": classification,
        "selected_oracle_family": selected,
        "selected_joint_byte_ratio_to_dense": ratio,
        "thresholds": rule,
        "parameter_updates_to_checkpoint": 0,
    }


def validate(args: argparse.Namespace, plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unexpected additive-normal plan schema")
    expected = {
        "entrypoint_sha256": file_sha256(Path(__file__)),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "config_sha256": file_sha256(args.config),
        "dataset_manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
        "parent_result_sha256": file_sha256(args.parent_result),
    }
    if plan.get("identity") != expected:
        raise ValueError(f"additive-normal identity mismatch: {expected}")
    parent = json.loads(args.parent_result.read_text())
    if (
        parent.get("schema_version") != PARENT_SCHEMA
        or parent.get("classification")
        != "WEIGHT_RELATIVE_TEMPORAL_STATE_INSUFFICIENT"
    ):
        raise ValueError("parent result does not authorize additive-normal analysis")
    expected_protocol = {
        "parameter_updates": 0,
        "checkpoint_next_iter": 9489,
        "state_names": list(STATE_NAMES),
        "matrix_shape": list(MATRIX_SHAPE),
        "late_layers": list(LATE_LAYERS),
        "bilateral_families": list(BILATERAL_FAMILIES),
        "output_schedule": list(OUTPUT_SCHEDULE),
        "rank_frontier": list(RANK_FRONTIER),
        "oracle_answer_conditioned": True,
        "rank_factor_storage": "fp32_U_and_V",
        "support_storage": "uint16_lower_bound",
    }
    if plan.get("protocol") != expected_protocol:
        raise ValueError("additive-normal protocol changed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--parent-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    plan = json.loads(args.plan.read_text())
    validate(args, plan)
    config = json.loads(args.config.read_text())
    started = time.time()
    model, optimizer, checkpoint = load_model_and_optimizer(args.checkpoint, config, "cpu")
    if int(checkpoint["next_iter"]) != int(plan["protocol"]["checkpoint_next_iter"]):
        raise ValueError("checkpoint next_iter changed")
    modules = cfc_modules(model)
    owner = directed_optimizer(optimizer)
    reference = modules[0]
    source = torch.stack([module.weight.float() for module in modules], dim=0).to(
        args.device
    ).contiguous()
    spectra: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for state_name in STATE_NAMES:
        target = torch.stack(
            [owner.state[module.weight][state_name].float() for module in modules], dim=0
        ).to(args.device).contiguous()
        candidates, _solver_rows = candidate_predictions(
            source,
            target,
            schedule=OUTPUT_SCHEDULE,
            ridge_ratio=reference.ridge_ratio,
            chunk_size=reference.chunk_size,
        )
        spectra[state_name] = {}
        for family in BILATERAL_FAMILIES:
            layer_rows = []
            for layer in range(len(modules)):
                residual = (target[layer] - candidates[family][layer]).float()
                gram = residual.transpose(0, 1) @ residual
                eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0).flip(0).cpu()
                layer_rows.append(
                    {
                        "layer": layer,
                        "target_energy": float(target[layer].double().square().sum()),
                        "base_residual_energy": float(residual.double().square().sum()),
                        "residual_spectrum": eigenvalues,
                    }
                )
            spectra[state_name][family] = layer_rows

    frontier: dict[str, Any] = {}
    allocations: dict[str, Any] = {}
    family_accounting: dict[str, Any] = {}
    rows, columns = MATRIX_SHAPE
    dense_bytes = rows * columns * 4
    output_coordinates = sum(OUTPUT_SCHEDULE) * rows
    bilateral_bytes = output_coordinates * 6 + columns * columns * 4
    for family in BILATERAL_FAMILIES:
        frontier[family] = {}
        allocations[family] = {}
        material_factor_bytes = 0
        intrinsic_lower_bound_bytes = 0
        for state_name in STATE_NAMES:
            layer_rows = spectra[state_name][family]
            frontier[family][state_name] = {
                str(rank): fixed_rank_summary(layer_rows, rank)
                for rank in RANK_FRONTIER
            }
            allocation = minimum_material_rank_allocation(
                layer_rows,
                aggregate_threshold=float(plan["decision_rule"]["minimum_aggregate_recovery"]),
                late_layer_threshold=float(plan["decision_rule"]["minimum_late_layer_recovery"]),
            )
            allocation["material_factor_bytes"] = (
                allocation["total_rank"] * (rows + columns) * 4
            )
            allocation["intrinsic_lower_bound_bytes"] = sum(
                rank * (rows + columns - rank) * 4
                for rank in allocation["ranks_by_layer"]
            )
            material_factor_bytes += allocation["material_factor_bytes"]
            intrinsic_lower_bound_bytes += allocation["intrinsic_lower_bound_bytes"]
            allocations[family][state_name] = allocation
        dense_joint_bytes = len(STATE_NAMES) * len(modules) * dense_bytes
        base_joint_bytes = len(STATE_NAMES) * len(modules) * bilateral_bytes
        family_accounting[family] = {
            "dense_joint_bytes": dense_joint_bytes,
            "bilateral_base_joint_bytes": base_joint_bytes,
            "material_factor_bytes": material_factor_bytes,
            "intrinsic_lower_bound_bytes": intrinsic_lower_bound_bytes,
            "joint_material_bytes": base_joint_bytes + material_factor_bytes,
            "joint_byte_ratio_to_dense": (
                base_joint_bytes + material_factor_bytes
            )
            / dense_joint_bytes,
            "joint_intrinsic_lower_bound_ratio_to_dense": (
                base_joint_bytes + intrinsic_lower_bound_bytes
            )
            / dense_joint_bytes,
        }
    decision = classify(family_accounting, plan["decision_rule"])
    serializable_spectra = {
        state_name: {
            family: [
                {
                    "layer": row["layer"],
                    "target_energy": row["target_energy"],
                    "base_residual_energy": row["base_residual_energy"],
                    "residual_participation_rank": float(
                        row["residual_spectrum"].double().sum().square()
                        / row["residual_spectrum"].double().square().sum().clamp_min(1e-30)
                    ),
                }
                for row in layer_rows
            ]
            for family, layer_rows in families.items()
        }
        for state_name, families in spectra.items()
    }
    result = {
        "schema_version": RESULT_SCHEMA,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "decision": decision,
        "frontier": frontier,
        "allocations": allocations,
        "family_accounting": family_accounting,
        "residual_summaries": serializable_spectra,
        "identity": {
            **plan["identity"],
            "plan_sha256": file_sha256(args.plan),
            "checkpoint_next_iter": int(checkpoint["next_iter"]),
        },
        "execution": {
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": str(Path(__file__).resolve()),
            "command": sys.argv,
            "device": args.device,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "direct_foreground_polling": True,
            "watchdog": False,
            "callback": False,
        },
        "authorization": {
            "candidate_implementation": False,
            "mfu_preflight": False,
            "language_model_training": False,
            "larger_rung": False,
            "latent_native_optimizer_theory": True,
        },
    }
    args.output.mkdir(parents=True)
    path = args.output / "result.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
