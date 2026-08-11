#!/usr/bin/env python3
"""Audit the gauge-quotiented horizontal tangent of the fitted c_fc TT."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.func import jvp

from examples.nanogpt.analyze_mlp_activation_update_alignment import git_commit
from examples.nanogpt.analyze_sparse_moe_cfc_functional_tt_oracle import prepare
from examples.nanogpt.analyze_sparse_moe_cfc_tt_optimizer_geometry_audit import (
    FeatureFunction,
    all_finite,
    build_bank_tasks,
    build_gates,
    classify_gates,
    make_feature_function,
    score_image,
    solve_metric_directions,
    tensor_cosine,
)
from examples.nanogpt.analyze_sparse_moe_paired_alignment import file_sha256


PLAN_SCHEMA = "nanogpt_sparse_moe_cfc_horizontal_tangent_audit_plan_v1"


def horizontal_chart(
    base_cores: list[torch.Tensor],
    ambient_feature: FeatureFunction,
) -> tuple[FeatureFunction, tuple[torch.Tensor, ...], dict[str, Any]]:
    """Construct intrinsic left-canonical TT tangent coordinates."""
    bases = [core.detach() for core in base_cores]
    specifications: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    coordinates: list[torch.Tensor] = []
    diagnostics: list[dict[str, Any]] = []
    for index, core in enumerate(bases[:-1]):
        left_rank, mode, right_rank = core.shape
        matrix = core.reshape(left_rank * mode, right_rank)
        full_q, _r = torch.linalg.qr(matrix, mode="complete")
        complement = full_q[:, right_rank:].contiguous()
        gram_error = float(
            (
                matrix.transpose(0, 1) @ matrix
                - torch.eye(
                    right_rank,
                    device=matrix.device,
                    dtype=matrix.dtype,
                )
            )
            .double()
            .norm()
        )
        cross_error = (
            float(
                (
                    matrix.transpose(0, 1) @ complement
                ).double().norm()
            )
            if complement.numel()
            else 0.0
        )
        count = int(complement.shape[1] * right_rank)
        diagnostics.append(
            {
                "core": index,
                "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
                "horizontal_coordinates": count,
                "base_orthogonality_error_fro": gram_error,
                "complement_cross_error_fro": cross_error,
            }
        )
        if count:
            coordinate = torch.zeros(
                complement.shape[1],
                right_rank,
                device=matrix.device,
                dtype=matrix.dtype,
            )
            specifications.append((index, matrix, complement))
            coordinates.append(coordinate)
    coordinates.append(torch.zeros_like(bases[-1]))

    def feature(*horizontal: torch.Tensor) -> torch.Tensor:
        if len(horizontal) != len(specifications) + 1:
            raise ValueError("horizontal coordinate tuple length mismatch")
        candidate = list(bases)
        for coordinate, (index, matrix, complement) in zip(
            horizontal[:-1], specifications, strict=True
        ):
            if coordinate.shape != (complement.shape[1], matrix.shape[1]):
                raise ValueError("horizontal core coordinate shape mismatch")
            candidate[index] = (
                matrix + complement @ coordinate
            ).reshape_as(bases[index])
        candidate[-1] = bases[-1] + horizontal[-1]
        return ambient_feature(*candidate)

    total = sum(value.numel() for value in coordinates)
    return feature, tuple(coordinates), {
        "horizontal_coordinates": total,
        "per_core": diagnostics
        + [
            {
                "core": len(bases) - 1,
                "matrix_shape": list(bases[-1].shape),
                "horizontal_coordinates": int(bases[-1].numel()),
                "base_orthogonality_error_fro": None,
                "complement_cross_error_fro": None,
            }
        ],
    }


def validate_plan(plan: dict[str, Any], plan_path: Path) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("horizontal TT tangent plan schema mismatch")
    identity = plan["identity"]
    if identity.get("entrypoint_sha256") is not None and identity[
        "entrypoint_sha256"
    ] != file_sha256(Path(__file__)):
        raise ValueError("entrypoint hash is not sealed in the plan")
    root = Path(__file__).resolve().parents[2]
    for relative, expected in identity["helper_sha256"].items():
        if file_sha256(root / relative) != expected:
            raise ValueError(f"helper hash drift: {relative}")
    horizontal = plan["horizontal_parameterization"]
    if sum(horizontal["per_core_horizontal_coordinates"]) != int(
        horizontal["horizontal_tangent_coordinates"]
    ):
        raise ValueError("horizontal coordinate accounting drift")
    if int(horizontal["ambient_forward_coordinates"]) - int(
        horizontal["horizontal_tangent_coordinates"]
    ) != int(horizontal["removed_redundant_coordinates"]):
        raise ValueError("removed horizontal coordinate accounting drift")
    if not file_sha256(plan_path):
        raise AssertionError("unreachable empty plan hash")


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
    protocol = plan["metric_protocol"]
    seeds = [str(value) for value in plan["source"]["fitted_seeds"]]
    source_banks = list(plan["source"]["discovery_banks"])
    evaluation_banks = source_banks + [
        row["name"] for row in plan["source"]["heldout_banks"]
    ]
    if preflight:
        seeds = seeds[:1]
        source_banks = source_banks[:1]
        evaluation_banks = [source_banks[0], evaluation_banks[2]]

    diagnostics: dict[str, Any] = {}
    scores: dict[str, Any] = {}
    chart_diagnostics: dict[str, Any] = {}
    feature_diagnostics: dict[str, Any] = {}
    heldout_images: dict[tuple[str, str, str, str], torch.Tensor] = {}
    operations = {"metric_solves": 0, "score_jvps": 0}
    for seed_index, seed in enumerate(seeds):
        base_cores = [
            value.to(device=device, dtype=torch.float32)
            for value in coordinates["fitted"][seed]
        ]
        features: dict[str, FeatureFunction] = {}
        targets: dict[str, torch.Tensor] = {}
        primals: tuple[torch.Tensor, ...] | None = None
        chart_diagnostics[seed] = {}
        feature_diagnostics[seed] = {}
        for bank in evaluation_banks:
            ambient, target, feature_row = make_feature_function(
                bank_tasks[bank], states, parent_plan, device=device
            )
            feature, bank_primals, chart_row = horizontal_chart(
                base_cores, ambient
            )
            if primals is None:
                primals = bank_primals
            elif [value.shape for value in primals] != [
                value.shape for value in bank_primals
            ]:
                raise RuntimeError("horizontal chart changed across banks")
            features[bank] = feature
            targets[bank] = target
            chart_diagnostics[seed][bank] = chart_row
            feature_diagnostics[seed][bank] = feature_row
        if primals is None:
            raise RuntimeError("no horizontal chart was built")
        if sum(value.numel() for value in primals) != int(
            plan["horizontal_parameterization"][
                "horizontal_tangent_coordinates"
            ]
        ):
            raise RuntimeError("horizontal coordinate count drift at runtime")
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
                    protocol["damping_ratio_to_hutchinson_mean_eigenvalue"]
                ),
                hutchinson_samples=(
                    1 if preflight else int(protocol["hutchinson_samples"])
                ),
                hutchinson_seed=(
                    int(protocol["hutchinson_seed"])
                    + 1009 * seed_index
                    + 17 * source_index
                ),
                pcg_iterations=(
                    2 if preflight else int(protocol["pcg_iterations"])
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
        "chart_diagnostics": chart_diagnostics,
        "feature_diagnostics": feature_diagnostics,
        "operations": operations,
    }


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
        measured_units = 1 + 1 + 2 + int(result["operations"]["score_jvps"])
        full_units_per_solve = (
            1
            + int(plan["metric_protocol"]["hutchinson_samples"])
            + int(plan["metric_protocol"]["pcg_iterations"])
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
                "nanogpt_sparse_moe_cfc_horizontal_tangent_preflight_v1"
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
        "horizontal_diagonal_functional_refit": classification
        == "DIAGONAL_METRIC_REPAIR_AUTHORIZED",
        "horizontal_gauss_newton_functional_refit": classification
        == "COUPLED_METRIC_REPAIR_AUTHORIZED",
        "language_model_training": False,
        "larger_rung": False,
        "generated_cproj": False,
    }
    payload = {
        "schema_version": (
            "nanogpt_sparse_moe_cfc_horizontal_tangent_audit_result_v1"
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
        raise RuntimeError("horizontal TT audit emitted nonfinite values")
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
