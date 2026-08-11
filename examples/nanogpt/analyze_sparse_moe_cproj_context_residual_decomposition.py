#!/usr/bin/env python3
"""Test pure context modulation as a direct-sum residual at fixed coordinates."""
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

from examples.nanogpt.analyze_mlp_activation_update_alignment import git_commit
from examples.nanogpt.analyze_sparse_moe_cproj_context_modulated_fht_oracle import (
    ContextModulatedCProjOperator,
    action_cosine,
    cgls_fit,
    cproj_target_action,
    routed_hidden_frames,
)
from examples.nanogpt.analyze_sparse_moe_paired_alignment import (
    collect_inputs,
    file_sha256,
)
from examples.nanogpt.analyze_sparse_moe_paired_atom_oracle import union_fieldnames
from examples.nanogpt.analyze_sparse_moe_rolling_tangent_oracle import (
    LayerState,
    recovery_fraction,
)
from examples.nanogpt.analyze_sparse_moe_stepzero_task_gradient_oracle import (
    all_finite,
    layer_state_from_mapping,
    load_terminal_snapshot,
    model_from_exact_stepzero,
    selected_stepzero_hashes,
)
from examples.nanogpt.analyze_residual_compatibility import fixed_validation_batches


PLAN_SCHEMA = "nanogpt_sparse_moe_cproj_context_residual_decomposition_plan_v1"


class DirectSumResidualOperator:
    """Use early static factors and late pure-context factors at fixed rank."""

    def __init__(
        self,
        static: ContextModulatedCProjOperator,
        coupled: ContextModulatedCProjOperator,
        static_factors: int,
    ) -> None:
        if static.coordinate_shape != coupled.coordinate_shape:
            raise ValueError("direct-sum operators must share coordinates")
        if not 0 < static_factors < static.factors:
            raise ValueError("direct sum requires both static and context factors")
        self.static = static
        self.coupled = coupled
        self.static_factors = int(static_factors)
        self.coordinate_shape = static.coordinate_shape
        self.token_count = static.token_count
        self.output_width = static.output_width
        self.device = static.device

    def _parts(self, coordinates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        static = coordinates.clone()
        context = coordinates.clone()
        static[:, self.static_factors :] = 0.0
        context[:, : self.static_factors] = 0.0
        return static, context

    def apply(self, coordinates: torch.Tensor) -> torch.Tensor:
        static, context = self._parts(coordinates)
        return (
            self.static.apply(static)
            + math.sqrt(2.0) * self.coupled.apply(context)
            - self.static.apply(context)
        )

    def adjoint(self, output_cotangent: torch.Tensor) -> torch.Tensor:
        static_gradient = self.static.adjoint(output_cotangent)
        context_gradient = (
            math.sqrt(2.0) * self.coupled.adjoint(output_cotangent)
            - static_gradient
        )
        result = static_gradient.clone()
        result[:, self.static_factors :] = context_gradient[:, self.static_factors :]
        return result


def collect_split_inputs(
    model: torch.nn.Module,
    plan: dict[str, Any],
    data_dir: Path,
    layers: list[int],
    device: str,
) -> dict[str, dict[int, torch.Tensor]]:
    frozen = plan["frozen_data_and_operator"]
    result = {}
    for spec in frozen["discovery_banks"]:
        batches = fixed_validation_batches(
            data_dir,
            int(spec["batch_size"]),
            int(spec["block_size"]),
            int(spec["batches"]),
            int(spec["seed"]),
        )
        result[spec["name"]] = collect_inputs(
            model, batches, layers, int(spec["tokens"]), device
        )
    heldout = frozen["heldout"]
    batches = fixed_validation_batches(
        data_dir,
        int(heldout["batch_size"]),
        int(heldout["block_size"]),
        int(heldout["batches"]),
        int(heldout["seed"]),
    )
    result["heldout"] = collect_inputs(
        model, batches, layers, int(heldout["tokens"]), device
    )
    return result


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
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("context residual decomposition plan schema mismatch")
    source = plan["source"]
    if file_sha256(args.terminal_snapshot) != source["terminal_manifold_snapshot_sha256"]:
        raise ValueError("terminal snapshot hash disagrees with frozen plan")
    layers = [int(value) for value in source["layers"]]
    payload = load_terminal_snapshot(args.terminal_snapshot)
    model = model_from_exact_stepzero(payload, int(source["model_seed"]), args.device)
    stepzero_hashes = selected_stepzero_hashes(model, layers)
    mapping = dict(model.named_parameters())
    initial = {layer: layer_state_from_mapping(mapping, layer) for layer in layers}
    terminal = {
        layer: layer_state_from_mapping(payload["model"], layer) for layer in layers
    }
    inputs = collect_split_inputs(model, plan, args.data_dir, layers, args.device)
    del model
    torch.cuda.empty_cache()

    frozen = plan["frozen_data_and_operator"]
    families = {
        "two_factor": {"factors": 2, "static_factors": 1},
        "three_factor": {"factors": 3, "static_factors": 2},
    }
    rows: list[dict[str, Any]] = []
    coordinates_payload: dict[str, torch.Tensor] = {}
    heldout_actions: dict[tuple[str, str, int], torch.Tensor] = {}
    bank_names = [spec["name"] for spec in frozen["discovery_banks"]]
    for bank in bank_names:
        for layer in layers:
            discovery_x = inputs[bank][layer].to(args.device)
            heldout_x = inputs["heldout"][layer].to(args.device)
            discovery_frames, discovery_counts = routed_hidden_frames(
                terminal[layer], discovery_x, 2, args.device
            )
            heldout_frames, _heldout_counts = routed_hidden_frames(
                terminal[layer], heldout_x, 2, args.device
            )
            discovery_target = cproj_target_action(
                discovery_frames,
                initial[layer].c_proj,
                terminal[layer].c_proj,
                discovery_x.shape[0],
                discovery_x.shape[1],
                args.device,
            )
            heldout_target = cproj_target_action(
                heldout_frames,
                initial[layer].c_proj,
                terminal[layer].c_proj,
                heldout_x.shape[0],
                heldout_x.shape[1],
                args.device,
            )
            for family, spec in families.items():
                factors = spec["factors"]
                for mode in ("direct_sum", "all_static"):
                    static_fit = ContextModulatedCProjOperator(
                        discovery_frames,
                        discovery_x.shape[0],
                        discovery_x.shape[1],
                        factors,
                        0.0,
                        int(frozen["fixed_operator_seed"]),
                        layer,
                        args.device,
                    )
                    if mode == "direct_sum":
                        coupled_fit = ContextModulatedCProjOperator(
                            discovery_frames,
                            discovery_x.shape[0],
                            discovery_x.shape[1],
                            factors,
                            1.0,
                            int(frozen["fixed_operator_seed"]),
                            layer,
                            args.device,
                        )
                        fit_operator = DirectSumResidualOperator(
                            static_fit, coupled_fit, spec["static_factors"]
                        )
                    else:
                        fit_operator = static_fit
                    coordinates, diagnostics = cgls_fit(
                        fit_operator,
                        discovery_target,
                        maximum_iterations=int(frozen["maximum_cgls_iterations"]),
                        tolerance=float(frozen["relative_normal_residual_tolerance"]),
                        ridge_ratio=float(frozen["ridge_ratio"]),
                        probe_seed=int(frozen["ridge_probe_seed"]) + 1009 * layer + factors,
                    )
                    discovery_action = fit_operator.apply(coordinates)
                    static_test = ContextModulatedCProjOperator(
                        heldout_frames,
                        heldout_x.shape[0],
                        heldout_x.shape[1],
                        factors,
                        0.0,
                        int(frozen["fixed_operator_seed"]),
                        layer,
                        args.device,
                    )
                    if mode == "direct_sum":
                        coupled_test = ContextModulatedCProjOperator(
                            heldout_frames,
                            heldout_x.shape[0],
                            heldout_x.shape[1],
                            factors,
                            1.0,
                            int(frozen["fixed_operator_seed"]),
                            layer,
                            args.device,
                        )
                        test_operator = DirectSumResidualOperator(
                            static_test, coupled_test, spec["static_factors"]
                        )
                    else:
                        test_operator = static_test
                    heldout_action = test_operator.apply(coordinates)
                    key = f"{bank}:{family}:{mode}:layer{layer}"
                    coordinates_payload[key] = coordinates.detach().cpu()
                    heldout_actions[(bank, family + ":" + mode, layer)] = heldout_action.detach().cpu()
                    rows.append(
                        {
                            "bank": bank,
                            "layer": layer,
                            "family": family,
                            "factors": factors,
                            "static_factors": spec["static_factors"],
                            "mode": mode,
                            "coordinates": coordinates.numel(),
                            "c_proj_compression_ratio": discovery_x.shape[1] / factors,
                            "minimum_expert_assignments": min(discovery_counts),
                            "discovery_recovery": recovery_fraction(
                                discovery_action, discovery_target
                            ),
                            "heldout_recovery": recovery_fraction(
                                heldout_action, heldout_target
                            ),
                            "coordinate_l2": float(coordinates.norm()),
                            **diagnostics,
                        }
                    )

    summary: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    frozen_gates = plan["frozen_gates"]
    for family in families:
        agreement = {
            str(layer): action_cosine(
                heldout_actions[(bank_names[0], family + ":direct_sum", layer)],
                heldout_actions[(bank_names[1], family + ":direct_sum", layer)],
            )
            for layer in layers
        }
        agreement_mean = sum(agreement.values()) / len(agreement)
        summary[family] = {
            "heldout_bank_action_cosine": {
                "mean": agreement_mean,
                "by_layer": agreement,
            }
        }
        gates[family] = {}
        for bank in bank_names:
            direct = [
                row for row in rows
                if row["family"] == family and row["bank"] == bank
                and row["mode"] == "direct_sum"
            ]
            static = {
                int(row["layer"]): row for row in rows
                if row["family"] == family and row["bank"] == bank
                and row["mode"] == "all_static"
            }
            direct_values = [float(row["heldout_recovery"]) for row in direct]
            gains = [
                float(row["heldout_recovery"])
                - float(static[int(row["layer"])]["heldout_recovery"])
                for row in direct
            ]
            bank_summary = {
                "heldout_mean": sum(direct_values) / len(direct_values),
                "heldout_minimum": min(direct_values),
                "heldout_by_layer": {
                    str(row["layer"]): float(row["heldout_recovery"])
                    for row in direct
                },
                "all_static_mean": sum(
                    float(row["heldout_recovery"]) for row in static.values()
                ) / len(static),
                "direct_sum_minus_all_static_mean": sum(gains) / len(gains),
                "direct_sum_minus_all_static_by_layer": {
                    str(row["layer"]): gain for row, gain in zip(direct, gains)
                },
                "minimum_gain": min(gains),
                "minimum_expert_assignments": min(
                    int(row["minimum_expert_assignments"]) for row in direct
                ),
                "maximum_relative_normal_residual": max(
                    float(row["relative_normal_residual"]) for row in direct
                ),
            }
            summary[family][bank] = bank_summary
            bank_gates = {
                "complementarity_pass": bank_summary["direct_sum_minus_all_static_mean"]
                >= float(frozen_gates["direct_sum_minus_all_static_mean_min_each_bank"]),
                "recovery_pass": bank_summary["heldout_mean"]
                >= float(frozen_gates["direct_sum_heldout_mean_min_each_bank"]),
                "every_layer_gain_pass": bank_summary["minimum_gain"]
                >= float(frozen_gates["direct_sum_minus_all_static_every_layer_min"]),
                "action_agreement_pass": agreement_mean
                >= float(frozen_gates["heldout_bank_action_cosine_mean_min"]),
                "occupancy_pass": bank_summary["minimum_expert_assignments"]
                >= int(frozen_gates["minimum_expert_assignments"]),
            }
            bank_gates["all_pass"] = all(bank_gates.values())
            gates[family][bank] = bank_gates

    finite = all_finite({"rows": rows, "summary": summary})
    passing = [
        family for family in families
        if finite and all(gates[family][bank]["all_pass"] for bank in bank_names)
    ]
    complementary_without_occupancy = [
        family for family in families
        if finite and all(
            all(
                value for key, value in gates[family][bank].items()
                if key not in {"occupancy_pass", "all_pass"}
            )
            for bank in bank_names
        )
    ]
    if passing:
        decision = "PASS_ALL_AUTHORIZE_MULTIPHASE_SHADOW_ONLY"
    elif complementary_without_occupancy:
        decision = "COMPLEMENTARY_BUT_ROUTER_OCCUPANCY_FAIL_REPAIR_ROUTER_FIRST"
    else:
        decision = "REJECT_NO_COMPLEMENTARITY_CLOSE_CONTEXT_MODULATION"

    args.output.mkdir(parents=True, exist_ok=True)
    rows_path = args.output / "context_residual_rows.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=union_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)
    coordinates_path = args.output / "context_residual_coordinates.pt"
    torch.save(coordinates_payload, coordinates_path)
    result = {
        "schema_version": "nanogpt_sparse_moe_cproj_context_residual_decomposition_result_v1",
        "decision": decision,
        "passing_families": passing,
        "complementary_without_occupancy": complementary_without_occupancy,
        "all_values_finite": finite,
        "summary": summary,
        "gates": gates,
        "stepzero_selected_tensor_sha256": stepzero_hashes,
        "source": {
            "terminal_snapshot_sha256": file_sha256(args.terminal_snapshot),
            "plan_sha256": file_sha256(args.plan),
            "dataset_manifest_sha256": source["dataset_manifest_sha256"],
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
    result_path = args.output / "context_residual_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    status = {
        "state": "finished",
        "exit_code": 0,
        "decision": decision,
        "result_sha256": file_sha256(result_path),
        "rows_sha256": file_sha256(rows_path),
        "coordinates_sha256": file_sha256(coordinates_path),
        "wall_seconds": time.time() - started,
    }
    status_path = args.output / "context_residual_status.json"
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "summary": summary}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
