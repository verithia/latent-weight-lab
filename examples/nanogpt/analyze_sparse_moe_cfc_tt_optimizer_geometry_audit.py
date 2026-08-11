#!/usr/bin/env python3
"""Audit functional-metric conditioning of the fitted sparse-MoE c_fc TT."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch.func import jvp, vjp

from examples.nanogpt.analyze_mlp_activation_update_alignment import git_commit
from examples.nanogpt.analyze_sparse_moe_cfc_functional_tt_oracle import (
    materialize_layer,
    prepare,
)
from examples.nanogpt.analyze_sparse_moe_cfc_spectral_feature_oracle import (
    dense_targets,
    route_and_sample,
)
from examples.nanogpt.analyze_sparse_moe_paired_alignment import file_sha256


PLAN_SCHEMA = "nanogpt_sparse_moe_cfc_tt_optimizer_geometry_audit_plan_v1"

TensorTuple = tuple[torch.Tensor, ...]
FeatureFunction = Callable[..., torch.Tensor]


def tuple_dot(left: TensorTuple, right: TensorTuple) -> torch.Tensor:
    return sum(
        (a.double() * b.double()).sum()
        for a, b in zip(left, right, strict=True)
    )


def tuple_all_finite(values: TensorTuple) -> bool:
    return all(bool(torch.isfinite(value).all()) for value in values)


def canonicalize_raw_cores(raw_cores: TensorTuple) -> TensorTuple:
    """Mirror the parent's differentiable QR chart at a saved endpoint."""
    canonical: list[torch.Tensor] = []
    for raw in raw_cores[:-1]:
        left_rank, mode, right_rank = raw.shape
        q, _r = torch.linalg.qr(
            raw.reshape(left_rank * mode, right_rank), mode="reduced"
        )
        canonical.append(q.reshape(left_rank, mode, right_rank))
    canonical.append(raw_cores[-1])
    return tuple(canonical)


def make_feature_function(
    bank_tasks: dict[int, torch.Tensor],
    states: dict[int, Any],
    parent_plan: dict[str, Any],
    *,
    device: str,
) -> tuple[FeatureFunction, torch.Tensor, dict[str, float]]:
    """Build a feature whose squared residual equals the parent objective."""
    layers = sorted(bank_tasks)
    mechanism = parent_plan["mechanism"]
    experts = int(parent_plan["source"]["num_experts"])
    hidden_modes = [int(value) for value in mechanism["hidden_modes"]]
    input_modes = [int(value) for value in mechanism["input_modes"]]
    contexts: list[
        tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ] = []
    target_parts: list[torch.Tensor] = []
    for layer in layers:
        inputs = bank_tasks[layer].to(device=device, dtype=torch.float32)
        target_pre, target_output = dense_targets(
            inputs, states[layer].c_fc, states[layer].c_proj, device
        )
        output_denominator = (
            float(experts * len(layers))
            * target_output.square().sum(dim=(1, 2))
        ).sqrt().clamp_min(1e-12)
        pre_denominator = (
            float(experts * len(layers))
            * target_pre.square().sum(dim=(1, 2))
        ).sqrt().clamp_min(1e-12)
        contexts.append(
            (
                layer,
                inputs,
                target_pre,
                target_output,
                pre_denominator,
                output_denominator,
            )
        )
        for expert in range(experts):
            target_parts.append(
                target_output[expert].reshape(-1)
                / output_denominator[expert]
            )
            target_parts.append(
                0.5
                * target_pre[expert].reshape(-1)
                / pre_denominator[expert]
            )
    target_feature = torch.cat(target_parts)

    def feature(*raw_cores: torch.Tensor) -> torch.Tensor:
        cores = canonicalize_raw_cores(tuple(raw_cores))
        parts: list[torch.Tensor] = []
        for (
            layer,
            inputs,
            _target_pre,
            _target_output,
            pre_denominator,
            output_denominator,
        ) in contexts:
            candidate_c_fc = materialize_layer(
                list(cores),
                layer,
                experts,
                hidden_modes,
                input_modes,
            )
            predicted_pre = torch.bmm(
                inputs, candidate_c_fc.transpose(1, 2)
            )
            predicted_output = torch.bmm(
                F.gelu(predicted_pre),
                states[layer]
                .c_proj.to(device=device, dtype=torch.float32)
                .transpose(1, 2),
            )
            for expert in range(experts):
                parts.append(
                    predicted_output[expert].reshape(-1)
                    / output_denominator[expert]
                )
                parts.append(
                    0.5
                    * predicted_pre[expert].reshape(-1)
                    / pre_denominator[expert]
                )
        return torch.cat(parts)

    return feature, target_feature, {
        "feature_coordinates": float(target_feature.numel()),
        "target_feature_norm": float(target_feature.double().norm()),
    }


def pcg(
    operator: Callable[[TensorTuple], TensorTuple],
    rhs: TensorTuple,
    inverse_diagonal: TensorTuple,
    *,
    iterations: int,
) -> tuple[TensorTuple, dict[str, float | int]]:
    """Fixed-iteration diagonal-preconditioned conjugate gradients."""
    solution = tuple(torch.zeros_like(value) for value in rhs)
    residual = tuple(value.clone() for value in rhs)
    preconditioned = tuple(
        value * diagonal
        for value, diagonal in zip(residual, inverse_diagonal, strict=True)
    )
    direction = tuple(value.clone() for value in preconditioned)
    initial_norm = tuple_dot(residual, residual).sqrt().clamp_min(1e-30)
    rz = tuple_dot(residual, preconditioned)
    completed = 0
    for step in range(int(iterations)):
        image = operator(direction)
        curvature = tuple_dot(direction, image)
        if not torch.isfinite(curvature) or float(curvature) <= 0.0:
            break
        alpha = rz / curvature
        solution = tuple(
            value + alpha.to(value.dtype) * delta
            for value, delta in zip(solution, direction, strict=True)
        )
        residual = tuple(
            value - alpha.to(value.dtype) * delta
            for value, delta in zip(residual, image, strict=True)
        )
        completed = step + 1
        preconditioned_next = tuple(
            value * diagonal
            for value, diagonal in zip(
                residual, inverse_diagonal, strict=True
            )
        )
        rz_next = tuple_dot(residual, preconditioned_next)
        if float(tuple_dot(residual, residual).sqrt() / initial_norm) <= 1e-8:
            preconditioned = preconditioned_next
            rz = rz_next
            break
        beta = rz_next / rz.clamp_min(1e-300)
        direction = tuple(
            value + beta.to(value.dtype) * old
            for value, old in zip(
                preconditioned_next, direction, strict=True
            )
        )
        preconditioned = preconditioned_next
        rz = rz_next
    relative = float(tuple_dot(residual, residual).sqrt() / initial_norm)
    return solution, {
        "iterations": completed,
        "relative_normal_residual": relative,
    }


def score_image(
    image: torch.Tensor,
    residual: torch.Tensor,
    *,
    transferred_alpha: float | None = None,
) -> dict[str, float]:
    target_energy = residual.double().square().sum().clamp_min(1e-30)
    image_energy = image.double().square().sum().clamp_min(1e-30)
    dot = (image.double() * residual.double()).sum()
    cosine = dot / (target_energy.sqrt() * image_energy.sqrt())
    optimal_alpha = max(0.0, float(dot / image_energy))

    def recovery(alpha: float) -> float:
        return float(
            (2.0 * alpha * dot - alpha * alpha * image_energy)
            / target_energy
        )

    result = {
        "directional_cosine": float(cosine),
        "optimal_positive_alpha": optimal_alpha,
        "optimal_positive_scalar_recovery": recovery(optimal_alpha),
        "image_norm": float(image_energy.sqrt()),
        "residual_norm": float(target_energy.sqrt()),
    }
    if transferred_alpha is not None:
        result["transferred_alpha"] = float(transferred_alpha)
        result["transferred_alpha_recovery"] = recovery(
            float(transferred_alpha)
        )
    return result


def solve_metric_directions(
    feature: FeatureFunction,
    target_feature: torch.Tensor,
    primals: TensorTuple,
    *,
    damping_ratio: float,
    hutchinson_samples: int,
    hutchinson_seed: int,
    pcg_iterations: int,
) -> tuple[dict[str, TensorTuple], dict[str, Any]]:
    base_feature, pullback = vjp(feature, *primals)
    residual = (target_feature - base_feature).detach()
    rhs = tuple(value.detach() for value in pullback(residual))
    coordinate_count = sum(value.numel() for value in primals)
    generator = torch.Generator(device=residual.device)
    generator.manual_seed(int(hutchinson_seed))
    diagonal = tuple(torch.zeros_like(value) for value in primals)
    for _ in range(int(hutchinson_samples)):
        probe = (
            torch.randint(
                0,
                2,
                residual.shape,
                device=residual.device,
                generator=generator,
            ).to(dtype=residual.dtype)
            * 2.0
            - 1.0
        )
        pulled = tuple(value.detach() for value in pullback(probe))
        diagonal = tuple(
            total + value.square()
            for total, value in zip(diagonal, pulled, strict=True)
        )
    diagonal = tuple(
        value / float(hutchinson_samples) for value in diagonal
    )
    mean_eigenvalue = sum(
        value.double().sum() for value in diagonal
    ) / float(coordinate_count)
    damping = float(damping_ratio) * mean_eigenvalue.clamp_min(1e-30)
    inverse_diagonal = tuple(
        (value + damping.to(value.dtype)).reciprocal()
        for value in diagonal
    )

    def system(direction: TensorTuple) -> TensorTuple:
        _, image = jvp(feature, primals, direction)
        pulled = pullback(image)
        return tuple(
            value.detach() + damping.to(delta.dtype) * delta
            for value, delta in zip(pulled, direction, strict=True)
        )

    natural, solver = pcg(
        system,
        rhs,
        inverse_diagonal,
        iterations=int(pcg_iterations),
    )
    diagonal_direction = tuple(
        value * scale
        for value, scale in zip(rhs, inverse_diagonal, strict=True)
    )
    directions = {
        "euclidean_pullback": rhs,
        "diagonal_metric": diagonal_direction,
        "gauss_newton": natural,
    }
    source_scores: dict[str, Any] = {}
    for name, direction in directions.items():
        _, image = jvp(feature, primals, direction)
        source_scores[name] = score_image(image.detach(), residual)
    return directions, {
        "base_objective": float(residual.double().square().sum()),
        "coordinate_count": coordinate_count,
        "mean_metric_eigenvalue": float(mean_eigenvalue),
        "damping": float(damping),
        "diagonal_minimum": min(float(value.amin()) for value in diagonal),
        "diagonal_maximum": max(float(value.amax()) for value in diagonal),
        "diagonal_dynamic_range": max(
            float(value.amax()) for value in diagonal
        )
        / max(min(float(value.amin()) for value in diagonal), 1e-30),
        "rhs_norm": float(tuple_dot(rhs, rhs).sqrt()),
        "solver": solver,
        "source_scores": source_scores,
        "finite": (
            bool(torch.isfinite(residual).all())
            and tuple_all_finite(rhs)
            and tuple_all_finite(diagonal)
            and all(tuple_all_finite(value) for value in directions.values())
        ),
    }


def tensor_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    numerator = (left.double() * right.double()).sum()
    denominator = left.double().norm() * right.double().norm()
    return float(numerator / denominator.clamp_min(1e-30))


def all_finite(value: Any) -> bool:
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite(item) for item in value)
    return True


def classify_gates(gates: dict[str, bool]) -> str:
    if not gates["all_values_and_gradients_finite"]:
        return "NONFINITE_OPTIMIZER_GEOMETRY_AUDIT"
    if not gates["pcg_converged"]:
        return "OPTIMIZER_GEOMETRY_NUMERICALLY_INCONCLUSIVE"
    if not gates["gauss_newton_source_recovery_pass"]:
        return "LOCAL_TT_TANGENT_INSUFFICIENT"
    structural = (
        "gauss_newton_heldout_recovery_pass",
        "gauss_newton_gain_pass",
        "gauss_newton_transferred_alpha_pass",
        "same_endpoint_stability_pass",
        "cross_endpoint_stability_pass",
    )
    if not all(gates[key] for key in structural):
        return "FUNCTIONAL_METRIC_NOT_STABLE"
    if gates["diagonal_simple_repair_pass"]:
        return "DIAGONAL_METRIC_REPAIR_AUTHORIZED"
    return "COUPLED_METRIC_REPAIR_AUTHORIZED"


def validate_plan(plan: dict[str, Any], plan_path: Path) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("optimizer-geometry audit plan schema mismatch")
    identity = plan["identity"]
    if identity.get("entrypoint_sha256") is not None and identity[
        "entrypoint_sha256"
    ] != file_sha256(Path(__file__)):
        raise ValueError("entrypoint hash is not sealed in the plan")
    root = Path(__file__).resolve().parents[2]
    for relative, expected in identity["helper_sha256"].items():
        if file_sha256(root / relative) != expected:
            raise ValueError(f"helper hash drift: {relative}")
    if not file_sha256(plan_path):
        raise AssertionError("unreachable empty plan hash")


def build_bank_tasks(
    discovery_tasks: list[tuple[str, int, torch.Tensor]],
    heldout_inputs: dict[str, dict[int, torch.Tensor]],
    states: dict[int, Any],
    plan: dict[str, Any],
) -> dict[str, dict[int, torch.Tensor]]:
    source = plan["source"]
    output: dict[str, dict[int, torch.Tensor]] = {}
    for bank, layer, inputs in discovery_tasks:
        output.setdefault(bank, {})[int(layer)] = inputs
    for bank in source["heldout_sampling_bank_indices"]:
        bank_index = int(source["heldout_sampling_bank_indices"][bank])
        output[bank] = {}
        for layer in [int(value) for value in source["layers"]]:
            sampled, _counts = route_and_sample(
                states[layer],
                heldout_inputs[bank][layer],
                top_k=int(source["top_k"]),
                samples_per_expert=int(
                    source["samples_per_expert_per_bank_layer"]
                ),
                seed=(
                    int(source["sampling_seed_base"])
                    + int(source["sampling_seed_bank_stride"]) * bank_index
                    + int(source["sampling_seed_layer_stride"]) * layer
                ),
            )
            output[bank][layer] = sampled
    return output


def run_audit(
    plan: dict[str, Any],
    parent_plan: dict[str, Any],
    coordinates: dict[str, Any],
    bank_tasks: dict[str, dict[int, torch.Tensor]],
    states: dict[int, Any],
    *,
    device: str,
    preflight: bool,
) -> dict[str, Any]:
    mechanism = plan["mechanism"]
    seeds = [str(value) for value in plan["source"]["fitted_seeds"]]
    source_banks = list(plan["source"]["discovery_banks"])
    evaluation_banks = source_banks + [
        row["name"] for row in plan["source"]["heldout_banks"]
    ]
    if preflight:
        seeds = seeds[:1]
        source_banks = source_banks[:1]
        evaluation_banks = [source_banks[0], evaluation_banks[2]]
    features: dict[str, FeatureFunction] = {}
    targets: dict[str, torch.Tensor] = {}
    feature_diagnostics: dict[str, Any] = {}
    for bank in evaluation_banks:
        feature, target, diagnostic = make_feature_function(
            bank_tasks[bank], states, parent_plan, device=device
        )
        features[bank] = feature
        targets[bank] = target
        feature_diagnostics[bank] = diagnostic

    diagnostics: dict[str, Any] = {}
    scores: dict[str, Any] = {}
    heldout_images: dict[tuple[str, str, str, str], torch.Tensor] = {}
    operations = {"metric_solves": 0, "score_jvps": 0}
    for seed_index, seed in enumerate(seeds):
        primals = tuple(
            value.to(device=device, dtype=torch.float32)
            for value in coordinates["fitted"][seed]
        )
        bank_residuals = {
            bank: (
                targets[bank] - features[bank](*primals).detach()
            ).detach()
            for bank in evaluation_banks
        }
        diagnostics[seed] = {}
        scores[seed] = {}
        for source_index, source_bank in enumerate(source_banks):
            directions, metric = solve_metric_directions(
                features[source_bank],
                targets[source_bank],
                primals,
                damping_ratio=float(
                    mechanism["damping_ratio_to_hutchinson_mean_eigenvalue"]
                ),
                hutchinson_samples=(
                    1 if preflight else int(mechanism["hutchinson_samples"])
                ),
                hutchinson_seed=(
                    int(mechanism["hutchinson_seed"])
                    + 1009 * seed_index
                    + 17 * source_index
                ),
                pcg_iterations=(
                    2 if preflight else int(mechanism["pcg_iterations"])
                ),
            )
            diagnostics[seed][source_bank] = metric
            scores[seed][source_bank] = {}
            operations["metric_solves"] += 1
            source_alphas = {
                name: float(row["optimal_positive_alpha"])
                for name, row in metric["source_scores"].items()
            }
            for direction_name, direction in directions.items():
                scores[seed][source_bank][direction_name] = {}
                for bank in evaluation_banks:
                    _, image = jvp(features[bank], primals, direction)
                    scores[seed][source_bank][direction_name][bank] = score_image(
                        image.detach(),
                        bank_residuals[bank],
                        transferred_alpha=source_alphas[direction_name],
                    )
                    operations["score_jvps"] += 1
                    if bank.startswith("heldout"):
                        heldout_images[
                            (seed, source_bank, direction_name, bank)
                        ] = image.detach().cpu()

    stability: dict[str, Any] = {
        "same_endpoint_cross_discovery": {},
        "cross_endpoint_corresponding": {},
    }
    if not preflight:
        heldout_banks = [row["name"] for row in plan["source"]["heldout_banks"]]
        for seed in seeds:
            stability["same_endpoint_cross_discovery"][seed] = {}
            for direction_name in (
                "euclidean_pullback",
                "diagonal_metric",
                "gauss_newton",
            ):
                stability["same_endpoint_cross_discovery"][seed][
                    direction_name
                ] = {
                    bank: tensor_cosine(
                        heldout_images[(seed, source_banks[0], direction_name, bank)],
                        heldout_images[(seed, source_banks[1], direction_name, bank)],
                    )
                    for bank in heldout_banks
                }
        first, second = seeds
        for source_bank in source_banks:
            stability["cross_endpoint_corresponding"][source_bank] = {}
            for direction_name in (
                "euclidean_pullback",
                "diagonal_metric",
                "gauss_newton",
            ):
                stability["cross_endpoint_corresponding"][source_bank][
                    direction_name
                ] = {
                    bank: tensor_cosine(
                        heldout_images[(first, source_bank, direction_name, bank)],
                        heldout_images[(second, source_bank, direction_name, bank)],
                    )
                    for bank in heldout_banks
                }
    return {
        "diagnostics": diagnostics,
        "scores": scores,
        "stability": stability,
        "feature_diagnostics": feature_diagnostics,
        "operations": operations,
    }


def build_gates(result: dict[str, Any], plan: dict[str, Any]) -> dict[str, bool]:
    thresholds = plan["frozen_gates"]
    source_banks = list(plan["source"]["discovery_banks"])
    heldout_banks = [row["name"] for row in plan["source"]["heldout_banks"]]
    seeds = [str(value) for value in plan["source"]["fitted_seeds"]]
    pcg_values: list[float] = []
    source_values: list[float] = []
    heldout_values: list[float] = []
    gain_values: list[float] = []
    transferred_values: list[float] = []
    diagonal_fractions: list[float] = []
    for seed in seeds:
        for source in source_banks:
            pcg_values.append(
                float(
                    result["diagnostics"][seed][source]["solver"][
                        "relative_normal_residual"
                    ]
                )
            )
            source_values.append(
                float(
                    result["scores"][seed][source]["gauss_newton"][source][
                        "optimal_positive_scalar_recovery"
                    ]
                )
            )
            for bank in heldout_banks:
                gn = float(
                    result["scores"][seed][source]["gauss_newton"][bank][
                        "optimal_positive_scalar_recovery"
                    ]
                )
                euclidean = float(
                    result["scores"][seed][source]["euclidean_pullback"][bank][
                        "optimal_positive_scalar_recovery"
                    ]
                )
                diagonal = float(
                    result["scores"][seed][source]["diagonal_metric"][bank][
                        "optimal_positive_scalar_recovery"
                    ]
                )
                heldout_values.append(gn)
                gain_values.append(gn - euclidean)
                transferred_values.append(
                    float(
                        result["scores"][seed][source]["gauss_newton"][bank][
                            "transferred_alpha_recovery"
                        ]
                    )
                )
                diagonal_fractions.append(diagonal / max(gn, 1e-30))
    same_endpoint = [
        float(value)
        for seed_rows in result["stability"][
            "same_endpoint_cross_discovery"
        ].values()
        for value in seed_rows["gauss_newton"].values()
    ]
    cross_endpoint = [
        float(value)
        for source_rows in result["stability"][
            "cross_endpoint_corresponding"
        ].values()
        for value in source_rows["gauss_newton"].values()
    ]
    gates = {
        "pcg_converged": max(pcg_values) <= float(
            thresholds["pcg_relative_normal_residual_max_each_seed_and_source_bank"]
        ),
        "gauss_newton_source_recovery_pass": min(source_values) >= float(
            thresholds["gauss_newton_source_optimal_linear_recovery_min"]
        ),
        "gauss_newton_heldout_recovery_pass": min(heldout_values) >= float(
            thresholds[
                "gauss_newton_heldout_optimal_linear_recovery_min_each_direction_and_bank"
            ]
        ),
        "gauss_newton_gain_pass": min(gain_values) >= float(
            thresholds["gauss_newton_minus_euclidean_heldout_recovery_min"]
        ),
        "gauss_newton_transferred_alpha_pass": min(transferred_values) >= float(
            thresholds["gauss_newton_transferred_alpha_heldout_recovery_min"]
        ),
        "same_endpoint_stability_pass": min(same_endpoint) >= float(
            thresholds[
                "same_endpoint_cross_discovery_direction_heldout_action_cosine_min"
            ]
        ),
        "cross_endpoint_stability_pass": min(cross_endpoint) >= float(
            thresholds[
                "cross_endpoint_corresponding_direction_heldout_action_cosine_min"
            ]
        ),
        "diagonal_simple_repair_pass": min(diagonal_fractions) >= float(
            thresholds["diagonal_within_fraction_of_gauss_newton_for_simple_repair"]
        ),
        "all_values_and_gradients_finite": all_finite(result),
    }
    gates["all_pass"] = all(gates.values())
    return gates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--terminal-snapshot", required=True, type=Path)
    parser.add_argument("--functional-tt-coordinates", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    started = time.time()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    validate_plan(plan, args.plan)
    root = Path(__file__).resolve().parents[2]
    source = plan["source"]
    parent_plan_path = root / source["functional_tt_plan"]
    fit_input_plan_path = root / source["fit_input_plan"]
    expected = {
        parent_plan_path: source["functional_tt_plan_sha256"],
        fit_input_plan_path: source["fit_input_plan_sha256"],
        args.terminal_snapshot: source["terminal_manifold_snapshot_sha256"],
        args.functional_tt_coordinates: source[
            "functional_tt_coordinates_sha256"
        ],
        args.data_dir / "manifest.json": source["dataset_manifest_sha256"],
    }
    for path, digest in expected.items():
        if file_sha256(path) != digest:
            raise ValueError(f"source hash mismatch: {path}")
    parent_plan = json.loads(parent_plan_path.read_text(encoding="utf-8"))
    fit_input_plan = json.loads(fit_input_plan_path.read_text(encoding="utf-8"))
    _dense, states, discovery_tasks, heldout_inputs = prepare(
        parent_plan,
        fit_input_plan,
        args.terminal_snapshot,
        args.data_dir,
        args.device,
    )
    coordinates = torch.load(
        args.functional_tt_coordinates, map_location="cpu", weights_only=False
    )
    if coordinates.get("schema_version") != (
        "nanogpt_sparse_moe_cfc_functional_tt_coordinates_v1"
    ):
        raise ValueError("functional TT coordinate schema mismatch")
    bank_tasks = build_bank_tasks(
        discovery_tasks, heldout_inputs, states, plan
    )
    result = run_audit(
        plan,
        parent_plan,
        coordinates,
        bank_tasks,
        states,
        device=args.device,
        preflight=bool(args.preflight_only),
    )
    wall_seconds = time.time() - started
    execution = {
        "device": args.device,
        "wall_seconds": wall_seconds,
        "maximum_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated())
            if args.device.startswith("cuda")
            else 0
        ),
    }
    if args.preflight_only:
        operations = result["operations"]
        measured_units = (
            1
            + 1
            + 2
            + int(operations["score_jvps"])
        )
        full_units_per_solve = (
            1
            + int(plan["mechanism"]["hutchinson_samples"])
            + int(plan["mechanism"]["pcg_iterations"])
            + 3
            * (
                len(plan["source"]["discovery_banks"])
                + len(plan["source"]["heldout_banks"])
            )
        )
        full_solves = len(plan["source"]["fitted_seeds"]) * len(
            plan["source"]["discovery_banks"]
        )
        payload = {
            "schema_version": (
                "nanogpt_sparse_moe_cfc_tt_optimizer_geometry_preflight_v1"
            ),
            "execution": execution,
            "measured_operation_units": measured_units,
            "projected_full_operation_units": full_units_per_solve
            * full_solves,
            "projected_full_wall_seconds": wall_seconds
            * float(full_units_per_solve * full_solves)
            / float(measured_units),
            "audit": result,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    gates = build_gates(result, plan)
    classification = classify_gates(gates)
    authorization = {
        "diagonal_functional_refit": classification
        == "DIAGONAL_METRIC_REPAIR_AUTHORIZED",
        "coupled_gauss_newton_functional_refit": classification
        == "COUPLED_METRIC_REPAIR_AUTHORIZED",
        "language_model_training": False,
        "larger_rung": False,
        "generated_cproj": False,
    }
    payload = {
        "schema_version": (
            "nanogpt_sparse_moe_cfc_tt_optimizer_geometry_audit_result_v1"
        ),
        "classification": classification,
        "passed": classification
        in {
            "DIAGONAL_METRIC_REPAIR_AUTHORIZED",
            "COUPLED_METRIC_REPAIR_AUTHORIZED",
        },
        "identity": {
            "git_commit": git_commit(root),
            "plan_sha256": file_sha256(args.plan),
            "entrypoint_sha256": file_sha256(Path(__file__)),
            "terminal_snapshot_sha256": file_sha256(args.terminal_snapshot),
            "functional_tt_coordinates_sha256": file_sha256(
                args.functional_tt_coordinates
            ),
            "dataset_manifest_sha256": file_sha256(
                args.data_dir / "manifest.json"
            ),
        },
        "execution": execution,
        "gates": gates,
        "authorization": authorization,
        **result,
    }
    if not all_finite(payload):
        raise RuntimeError("optimizer-geometry audit emitted nonfinite values")
    if args.output is None:
        parser.error("scientific audit requires --output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
