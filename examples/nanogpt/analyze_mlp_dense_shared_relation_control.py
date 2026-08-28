#!/usr/bin/env python3
"""H38 over-budget control for a dense shared left/right task-atom relation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Literal

import torch

from examples.nanogpt.analyze_mlp_shared_separable_conditioner_loo import (
    CANONICAL_SHAPE,
    canonicalize,
    normalized,
    optimal_scalar,
    squared_cosine,
)
from examples.nanogpt.analyze_mlp_synthetic_muon_program import (
    DENSE_MLP_SCALARS,
    build_dense_model,
    initialization_match,
    make_prompt,
    sha256,
)
from examples.nanogpt.analyze_mlp_synthetic_muon_program_joint import (
    FROZEN_PARAMETERS,
    joint_leading_pc,
)
from examples.nanogpt.analyze_mlp_virtual_lookahead_joint import (
    make_model_lookahead_program,
)
from examples.nanogpt.muon import zeropower_via_newtonschulz5


PROMPT_LENGTH = 375
PROMPT_WIDTH = 768
DEPLOYED_MLP_MATRICES = 24
ONE_SIDED_ALS_ITERATIONS = 8
TWO_SIDED_ALS_ITERATIONS = 8
RIDGE_RELATIVE = 1e-6
NORMAL_RESIDUAL_GATE = 1e-5
DENOMINATOR_FLOOR = 1e-30


def control_accounting() -> dict[str, int | float | str]:
    rows, columns = CANONICAL_SHAPE
    prompt_scalars = PROMPT_LENGTH * PROMPT_WIDTH
    left_scalars = rows * rows
    right_scalars = columns * columns
    coefficient_scalars = DEPLOYED_MLP_MATRICES
    total_scalars = prompt_scalars + left_scalars + right_scalars + coefficient_scalars
    return {
        "prompt_scalars": prompt_scalars,
        "shared_left_scalars": left_scalars,
        "shared_right_scalars": right_scalars,
        "node_coefficient_scalars": coefficient_scalars,
        "right_only_total_scalars": prompt_scalars + right_scalars + coefficient_scalars,
        "left_only_total_scalars": prompt_scalars + left_scalars + coefficient_scalars,
        "full_control_state_scalars": total_scalars,
        "dense_mlp_denominator_scalars": DENSE_MLP_SCALARS,
        "state_fraction": total_scalars / DENSE_MLP_SCALARS,
        "fp16_checkpoint_bytes": 2 * total_scalars,
        "classification": "over-budget premise-relaxation control",
    }


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().float().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _ridge(gram: torch.Tensor, relative: float) -> torch.Tensor:
    dimension = gram.shape[0]
    mean_diagonal = gram.diagonal().mean().clamp_min(torch.finfo(gram.dtype).tiny)
    return relative * mean_diagonal * torch.eye(
        dimension, device=gram.device, dtype=gram.dtype
    )


def _cholesky_solve_with_refinement(
    gram: torch.Tensor,
    right_hand_side: torch.Tensor,
    *,
    transpose_solution: bool,
    ridge_relative: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    regularized = gram + _ridge(gram, ridge_relative)
    factor, info = torch.linalg.cholesky_ex(regularized)
    if int(info.max()) != 0:
        raise RuntimeError(f"Cholesky failed at leading minor {int(info.max())}")
    rhs = right_hand_side.T if transpose_solution else right_hand_side
    solution = torch.cholesky_solve(rhs, factor)
    # One deterministic iterative-refinement step improves the FP32 normal residual.
    residual = rhs - regularized @ solution
    solution = solution + torch.cholesky_solve(residual, factor)
    final_residual = rhs - regularized @ solution
    relative_residual = float(
        final_residual.double().norm()
        / rhs.double().norm().clamp_min(DENOMINATOR_FLOOR)
    )
    returned = solution.T if transpose_solution else solution
    return returned, {
        "ridge_absolute": float((_ridge(gram, ridge_relative))[0, 0]),
        "relative_normal_equation_residual": relative_residual,
    }


def solve_right(
    atoms: tuple[torch.Tensor, ...],
    targets: tuple[torch.Tensor, ...],
    coefficients: torch.Tensor,
    indices: tuple[int, ...],
    *,
    ridge_relative: float = RIDGE_RELATIVE,
) -> tuple[torch.Tensor, dict[str, float]]:
    columns = atoms[0].shape[1]
    gram = torch.zeros(columns, columns, device=atoms[0].device, dtype=atoms[0].dtype)
    rhs = torch.zeros_like(gram)
    for index in indices:
        feature = coefficients[index] * atoms[index]
        gram.addmm_(feature.T, feature)
        rhs.addmm_(feature.T, targets[index])
    return _cholesky_solve_with_refinement(
        gram, rhs, transpose_solution=False, ridge_relative=ridge_relative
    )


def solve_left(
    atoms: tuple[torch.Tensor, ...],
    targets: tuple[torch.Tensor, ...],
    coefficients: torch.Tensor,
    indices: tuple[int, ...],
    *,
    ridge_relative: float = RIDGE_RELATIVE,
) -> tuple[torch.Tensor, dict[str, float]]:
    rows = atoms[0].shape[0]
    gram = torch.zeros(rows, rows, device=atoms[0].device, dtype=atoms[0].dtype)
    rhs = torch.zeros_like(gram)
    for index in indices:
        feature = coefficients[index] * atoms[index]
        gram.addmm_(feature, feature.T)
        rhs.addmm_(targets[index], feature.T)
    return _cholesky_solve_with_refinement(
        gram, rhs, transpose_solution=True, ridge_relative=ridge_relative
    )


def _metrics(
    bases: tuple[torch.Tensor, ...],
    targets: tuple[torch.Tensor, ...],
    coefficients: torch.Tensor,
    indices: tuple[int, ...],
) -> dict[str, Any]:
    captures = [squared_cosine(bases[index], targets[index]) for index in indices]
    residuals = [
        float((coefficients[index] * bases[index] - targets[index]).double().square().sum())
        for index in indices
    ]
    return {
        "captures": captures,
        "minimum_capture": min(captures),
        "median_capture": statistics.median(captures),
        "maximum_capture": max(captures),
        "mean_residual_energy": statistics.mean(residuals),
        "maximum_residual_energy": max(residuals),
        "finite": all(bool(torch.isfinite(bases[index]).all()) for index in indices),
    }


def _update_coefficients(
    bases: tuple[torch.Tensor, ...],
    targets: tuple[torch.Tensor, ...],
    coefficients: torch.Tensor,
    indices: tuple[int, ...],
) -> torch.Tensor:
    result = coefficients.clone()
    for index in indices:
        result[index] = optimal_scalar(bases[index], targets[index])
    return result


def fit_one_sided(
    atoms: tuple[torch.Tensor, ...],
    targets: tuple[torch.Tensor, ...],
    indices: tuple[int, ...],
    *,
    side: Literal["left", "right"],
    iterations: int = ONE_SIDED_ALS_ITERATIONS,
) -> dict[str, Any]:
    coefficients = torch.ones(len(atoms), device=atoms[0].device, dtype=atoms[0].dtype)
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for iteration in range(iterations):
        if side == "right":
            matrix, solve = solve_right(atoms, targets, coefficients, indices)
            bases = tuple(atom @ matrix for atom in atoms)
        else:
            matrix, solve = solve_left(atoms, targets, coefficients, indices)
            bases = tuple(matrix @ atom for atom in atoms)
        coefficients = _update_coefficients(bases, targets, coefficients, indices)
        coefficient_rms = coefficients[list(indices)].square().mean().sqrt().clamp_min(1e-12)
        coefficients[list(indices)] /= coefficient_rms
        matrix = matrix * coefficient_rms
        bases = tuple(base * coefficient_rms for base in bases)
        metrics = _metrics(bases, targets, coefficients, indices)
        row = {"iteration": iteration + 1, **solve, **metrics}
        history.append(row)
        if best is None or metrics["mean_residual_energy"] < best["metrics"]["mean_residual_energy"]:
            best = {
                "matrix": matrix.detach().clone(),
                "coefficients": coefficients.detach().clone(),
                "metrics": metrics,
                "solve": solve,
                "iteration": iteration + 1,
            }
    assert best is not None
    return {
        "side": side,
        "indices": list(indices),
        "iterations": iterations,
        "history": history,
        "best_iteration": best["iteration"],
        "metrics": best["metrics"],
        "solve": best["solve"],
        "matrix": best["matrix"],
        "coefficients": best["coefficients"],
        "matrix_sha256": tensor_sha256(best["matrix"]),
    }


def _two_sided_bases(
    atoms: tuple[torch.Tensor, ...], left: torch.Tensor, right: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    return tuple((left @ atom) @ right for atom in atoms)


def fit_two_sided(
    atoms: tuple[torch.Tensor, ...],
    targets: tuple[torch.Tensor, ...],
    indices: tuple[int, ...],
    *,
    initializations: tuple[str, ...] = ("identity", "right_only", "left_only"),
    iterations: int = TWO_SIDED_ALS_ITERATIONS,
    left_parent: dict[str, Any] | None = None,
    right_parent: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows, columns = atoms[0].shape
    device, dtype = atoms[0].device, atoms[0].dtype
    starts: list[dict[str, Any]] = []
    for initialization in initializations:
        left = torch.eye(rows, device=device, dtype=dtype)
        right = torch.eye(columns, device=device, dtype=dtype)
        coefficients = torch.ones(len(atoms), device=device, dtype=dtype)
        if initialization == "right_only":
            if right_parent is None:
                raise ValueError("right-only initialization requires its fitted parent")
            right = right_parent["matrix"].detach().clone()
            coefficients = right_parent["coefficients"].detach().clone()
        elif initialization == "left_only":
            if left_parent is None:
                raise ValueError("left-only initialization requires its fitted parent")
            left = left_parent["matrix"].detach().clone()
            coefficients = left_parent["coefficients"].detach().clone()
        elif initialization != "identity":
            raise ValueError(initialization)

        history: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None
        for iteration in range(iterations):
            right_atoms = tuple(atom @ right for atom in atoms)
            left, left_solve = solve_left(right_atoms, targets, coefficients, indices)
            left_atoms = tuple(left @ atom for atom in atoms)
            right, right_solve = solve_right(left_atoms, targets, coefficients, indices)
            bases = _two_sided_bases(atoms, left, right)
            coefficients = _update_coefficients(bases, targets, coefficients, indices)

            # Fix the bilinear and coefficient gauges without changing predictions.
            left_rms = left.square().mean().sqrt().clamp_min(1e-12)
            left = left / left_rms
            right = right * left_rms
            coefficient_rms = coefficients[list(indices)].square().mean().sqrt().clamp_min(1e-12)
            coefficients[list(indices)] /= coefficient_rms
            right = right * coefficient_rms
            bases = tuple(base * coefficient_rms for base in bases)

            metrics = _metrics(bases, targets, coefficients, indices)
            row = {
                "iteration": iteration + 1,
                "left_relative_normal_equation_residual": left_solve[
                    "relative_normal_equation_residual"
                ],
                "right_relative_normal_equation_residual": right_solve[
                    "relative_normal_equation_residual"
                ],
                **metrics,
            }
            history.append(row)
            if best is None or metrics["mean_residual_energy"] < best["metrics"]["mean_residual_energy"]:
                best = {
                    "left": left.detach().clone(),
                    "right": right.detach().clone(),
                    "coefficients": coefficients.detach().clone(),
                    "metrics": metrics,
                    "left_solve": left_solve,
                    "right_solve": right_solve,
                    "iteration": iteration + 1,
                }
        assert best is not None
        starts.append(
            {
                "initialization": initialization,
                "history": history,
                "best_iteration": best["iteration"],
                "metrics": best["metrics"],
                "left_solve": best["left_solve"],
                "right_solve": best["right_solve"],
                "left": best["left"],
                "right": best["right"],
                "coefficients": best["coefficients"],
            }
        )
    winner = min(starts, key=lambda row: row["metrics"]["mean_residual_energy"])
    return {
        "indices": list(indices),
        "iterations": iterations,
        "initializations": [row["initialization"] for row in starts],
        "start_summaries": [
            {
                key: value
                for key, value in row.items()
                if key not in {"left", "right", "coefficients"}
            }
            for row in starts
        ],
        "winning_initialization": winner["initialization"],
        "best_iteration": winner["best_iteration"],
        "metrics": winner["metrics"],
        "left_solve": winner["left_solve"],
        "right_solve": winner["right_solve"],
        "left": winner["left"],
        "right": winner["right"],
        "coefficients": winner["coefficients"],
        "left_sha256": tensor_sha256(winner["left"]),
        "right_sha256": tensor_sha256(winner["right"]),
    }


def _serializable_fit(fit: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in fit.items()
        if not isinstance(value, torch.Tensor)
    }


def audit_relation(
    atoms: tuple[torch.Tensor, ...],
    targets: tuple[torch.Tensor, ...],
    *,
    iterations: int,
    heldout_indices: tuple[int, ...],
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    atoms = tuple(normalized(value.detach().float()) for value in atoms)
    targets = tuple(normalized(value.detach().float()) for value in targets)
    all_indices = tuple(range(len(atoms)))
    right = fit_one_sided(atoms, targets, all_indices, side="right", iterations=iterations)
    left = fit_one_sided(atoms, targets, all_indices, side="left", iterations=iterations)
    two = fit_two_sided(
        atoms,
        targets,
        all_indices,
        iterations=iterations,
        left_parent=left,
        right_parent=right,
    )
    fit_all_metrics = two["metrics"]
    capacity_pass = (
        fit_all_metrics["finite"]
        and fit_all_metrics["minimum_capture"] >= 0.20
        and fit_all_metrics["median_capture"] >= 0.40
    )
    folds: list[dict[str, Any]] = []
    if capacity_pass:
        for heldout in heldout_indices:
            train = tuple(index for index in all_indices if index != heldout)
            fold_right = fit_one_sided(atoms, targets, train, side="right", iterations=iterations)
            fold_left = fit_one_sided(atoms, targets, train, side="left", iterations=iterations)
            fold_two = fit_two_sided(
                atoms,
                targets,
                train,
                iterations=iterations,
                left_parent=fold_left,
                right_parent=fold_right,
            )
            base = (fold_two["left"] @ atoms[heldout]) @ fold_two["right"]
            folds.append(
                {
                    "heldout_index": heldout,
                    "leave_one_out_capture": squared_cosine(base, targets[heldout]),
                    "raw_capture": squared_cosine(atoms[heldout], targets[heldout]),
                    "heldout_optimal_scalar": float(optimal_scalar(base, targets[heldout])),
                    "train_metrics": fold_two["metrics"],
                    "winning_initialization": fold_two["winning_initialization"],
                    "left_sha256": fold_two["left_sha256"],
                    "right_sha256": fold_two["right_sha256"],
                }
            )
    loo_captures = [row["leave_one_out_capture"] for row in folds]
    loo = {
        "performed": bool(folds),
        "folds": folds,
        "minimum_capture": min(loo_captures) if loo_captures else None,
        "median_capture": statistics.median(loo_captures) if loo_captures else None,
        "maximum_capture": max(loo_captures) if loo_captures else None,
    }
    transfer_pass = bool(folds) and (
        loo["minimum_capture"] >= 0.10 and loo["median_capture"] >= 0.20
    )
    maximum_solver_residual = max(
        right["solve"]["relative_normal_equation_residual"],
        left["solve"]["relative_normal_equation_residual"],
        two["left_solve"]["relative_normal_equation_residual"],
        two["right_solve"]["relative_normal_equation_residual"],
    )
    result = {
        "fit_all": {
            "right_only": _serializable_fit(right),
            "left_only": _serializable_fit(left),
            "two_sided": _serializable_fit(two),
        },
        "capacity_pass": capacity_pass,
        "leave_one_out": loo,
        "transfer_pass": transfer_pass,
        "maximum_fit_all_solver_residual": maximum_solver_residual,
        "solver_residual_pass": maximum_solver_residual <= NORMAL_RESIDUAL_GATE,
        "retained_for_structure_analysis": capacity_pass and transfer_pass,
    }
    tensors = {
        "left": two["left"].half().cpu(),
        "right": two["right"].half().cpu(),
        "coefficients": two["coefficients"].half().cpu(),
    }
    return result, tensors


def self_test(device_name: str) -> dict[str, Any]:
    device = torch.device(device_name)
    torch.manual_seed(383)
    rows, columns, count = 32, 16, 6
    atoms = tuple(normalized(torch.randn(rows, columns, device=device)) for _ in range(count))
    left_true = torch.eye(rows, device=device) + 0.08 * torch.randn(rows, rows, device=device)
    right_true = torch.eye(columns, device=device) + 0.08 * torch.randn(columns, columns, device=device)
    right_targets = tuple(normalized(atom @ right_true) for atom in atoms)
    left_targets = tuple(normalized(left_true @ atom) for atom in atoms)
    two_targets = tuple(normalized((left_true @ atom) @ right_true) for atom in atoms)
    indices = tuple(range(count))
    right = fit_one_sided(atoms, right_targets, indices, side="right")
    left = fit_one_sided(atoms, left_targets, indices, side="left")
    two = fit_two_sided(
        atoms,
        two_targets,
        indices,
        left_parent=left,
        right_parent=right,
    )
    minima = {
        "right": right["metrics"]["minimum_capture"],
        "left": left["metrics"]["minimum_capture"],
        "two_sided": two["metrics"]["minimum_capture"],
    }
    if min(minima.values()) < 0.99:
        raise AssertionError(minima)
    residuals = {
        "right": right["solve"]["relative_normal_equation_residual"],
        "left": left["solve"]["relative_normal_equation_residual"],
        "two_left": two["left_solve"]["relative_normal_equation_residual"],
        "two_right": two["right_solve"]["relative_normal_equation_residual"],
    }
    if max(residuals.values()) > NORMAL_RESIDUAL_GATE:
        raise AssertionError(residuals)
    accounting = control_accounting()
    if accounting["full_control_state_scalars"] != 10_315_032:
        raise AssertionError(accounting)
    return {"status": "passed", "minimum_captures": minima, "residuals": residuals}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--trajectory-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(args.device), sort_keys=True))
        return
    if any(value is None for value in (args.config, args.plan, args.trajectory_dir, args.output)):
        parser.error("config, plan, trajectory, and output are required")
    assert args.config is not None and args.plan is not None
    assert args.trajectory_dir is not None and args.output is not None

    accounting = control_accounting()
    if accounting["full_control_state_scalars"] != 10_315_032:
        raise ValueError("H38 accounting mismatch")
    plan = json.loads(args.plan.read_text())
    if plan["frozen_control"]["full_control_state_scalars"] != 10_315_032:
        raise ValueError("plan/accounting mismatch")
    config = json.loads(args.config.read_text())
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    model = build_dense_model(config, args.device)
    prompt, targets, prompt_manifest = make_prompt(
        model, config, prompt_length=PROMPT_LENGTH, device=args.device
    )
    joint_target, leading_fraction, target_manifest, w0_references = joint_leading_pc(
        args.trajectory_dir, parameters=FROZEN_PARAMETERS, device=args.device
    )
    model_parameters = dict(model.named_parameters())
    w0_matches = {
        parameter: initialization_match(model_parameters[parameter], w0_references[parameter])
        for parameter in FROZEN_PARAMETERS
    }
    if not all(bool(record["accepted"]) for record in w0_matches.values()):
        raise ValueError(f"model/trajectory W0 mismatch: {w0_matches}")
    _, loss_function, initial_weights, function_manifest = make_model_lookahead_program(
        model,
        parameters=FROZEN_PARAMETERS,
        targets=targets,
        ns_steps=5,
        momentum=0.0,
    )
    gradients = torch.func.grad(loss_function, argnums=0)(initial_weights, prompt)
    raw_atoms = tuple(
        zeropower_via_newtonschulz5(gradient, steps=5).detach() for gradient in gradients
    )
    split_targets = torch.split(joint_target, [weight.numel() for weight in initial_weights])
    atoms = tuple(
        canonicalize(parameter, atom)
        for parameter, atom in zip(FROZEN_PARAMETERS, raw_atoms, strict=True)
    )
    target_parts = tuple(
        canonicalize(parameter, part.reshape_as(weight))
        for parameter, part, weight in zip(
            FROZEN_PARAMETERS, split_targets, initial_weights, strict=True
        )
    )

    iterations = 1 if args.preflight else TWO_SIDED_ALS_ITERATIONS
    heldouts = (0,) if args.preflight else tuple(range(len(atoms)))
    audit, compact_tensors = audit_relation(
        atoms, target_parts, iterations=iterations, heldout_indices=heldouts
    )
    audit["leading_pc_energy_fraction"] = leading_fraction
    audit["preflight"] = args.preflight
    classification = (
        "PREFLIGHT"
        if args.preflight
        else ("RELATION_RETAINED" if audit["retained_for_structure_analysis"] else "RELATION_REJECTED")
    )

    checkpoint_path = output / "fit_all_dense_shared_relation_control.pt"
    torch.save(compact_tensors, checkpoint_path)
    accounting_path = output / "accounting.json"
    accounting_path.write_text(json.dumps(accounting, indent=2, sort_keys=True) + "\n")
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    runtime_seconds = time.time() - started
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_mlp_dense_shared_relation_control_v1",
        "classification": classification,
        "plan": plan,
        "accounting": accounting,
        "prompt_manifest": prompt_manifest,
        "target_manifest": target_manifest,
        "w0_storage_matches": w0_matches,
        "function_manifest": {
            **function_manifest,
            "task_atom": "NS5 first task gradient at W0",
            "canonical_shape": list(CANONICAL_SHAPE),
            "shared_relation": "unrestricted dense left/right premise control",
            "compact_candidate": False,
        },
        "solver_self_test": self_test(args.device),
        "audit": audit,
        "execution": {
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_status": subprocess.check_output(["git", "status", "--short"], text=True).splitlines(),
            "entrypoint": str(script),
            "entrypoint_sha256": sha256(script),
            "config": str(args.config),
            "config_sha256": sha256(args.config),
            "plan": str(args.plan),
            "plan_sha256": sha256(args.plan),
            "command": [str(script), *sys.argv[1:]],
            "runtime_seconds": runtime_seconds,
            "projected_binding_runtime_seconds": runtime_seconds * TWO_SIDED_ALS_ITERATIONS if args.preflight else runtime_seconds,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else 0,
            "device": args.device,
        },
        "outputs": {
            "accounting": {"path": str(accounting_path), "sha256": sha256(accounting_path)},
            "fit_all_maps": {"path": str(checkpoint_path), "sha256": sha256(checkpoint_path)},
        },
        "limitations": [
            "H38 violates the one-percent state ceiling and is never a deployable candidate.",
            "A negative result applies only to a shared bilinear relation around this task atom.",
            "A positive result authorizes structure analysis of the fitted maps, not CE or scale-up.",
        ],
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "metadata": str(metadata_path),
                "classification": classification,
                "capacity_pass": audit["capacity_pass"],
                "transfer_pass": audit["transfer_pass"],
                "fit_all_median_capture": audit["fit_all"]["two_sided"]["metrics"]["median_capture"],
                "loo_median_capture": audit["leave_one_out"]["median_capture"],
                "runtime_seconds": runtime_seconds,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
