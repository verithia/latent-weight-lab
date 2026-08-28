#!/usr/bin/env python3
"""H34a leave-one-node-out gate for shared row/column task conditioning."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_mlp_shared_sign_preconditioner_loo import (
    CANONICAL_SHAPE,
    canonicalize,
    normalized,
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


PROMPT_LENGTH = 731
PROMPT_WIDTH = 768
DEPLOYED_MLP_MATRICES = 24
ALS_ITERATIONS = 20
DENOMINATOR_FLOOR = 1e-12


def conditioner_accounting() -> dict[str, int | float]:
    prompt_scalars = PROMPT_LENGTH * PROMPT_WIDTH
    row_scalars, column_scalars = CANONICAL_SHAPE
    total_scalars = (
        prompt_scalars + row_scalars + column_scalars + DEPLOYED_MLP_MATRICES
    )
    return {
        "prompt_scalars": prompt_scalars,
        "shared_row_gain_scalars": row_scalars,
        "shared_column_gain_scalars": column_scalars,
        "node_coefficient_scalars": DEPLOYED_MLP_MATRICES,
        "total_state_scalars": total_scalars,
        "dense_mlp_denominator_scalars": DENSE_MLP_SCALARS,
        "state_fraction": total_scalars / DENSE_MLP_SCALARS,
        "fp16_checkpoint_bytes": 2 * total_scalars,
        "persistent_dense_basis_scalars": 0,
    }


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().float().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def optimal_scalar(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    denominator = prediction.square().sum().clamp_min(DENOMINATOR_FLOOR)
    return (prediction * target).sum() / denominator


def fit_separable_conditioner(
    atoms: tuple[torch.Tensor, ...],
    targets: tuple[torch.Tensor, ...],
    train_indices: tuple[int, ...],
    *,
    iterations: int = ALS_ITERATIONS,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if not train_indices:
        raise ValueError("at least one training node is required")
    atom_rows = tuple(normalized(value) for value in atoms)
    target_rows = tuple(normalized(value) for value in targets)
    shape = tuple(atom_rows[0].shape)
    if len(shape) != 2 or any(tuple(value.shape) != shape for value in (*atom_rows, *target_rows)):
        raise ValueError("separable fit requires matching matrices")
    device = atom_rows[0].device
    left = torch.ones(shape[0], device=device)
    right = torch.ones(shape[1], device=device)
    coefficient_rows: list[list[float]] = []
    for _ in range(iterations):
        coefficients = []
        for index in train_indices:
            base = left[:, None] * atom_rows[index] * right[None, :]
            coefficients.append(optimal_scalar(base, target_rows[index]))
        coefficient_rows.append([float(value) for value in coefficients])

        left_numerator = torch.zeros_like(left)
        left_denominator = torch.zeros_like(left)
        for coefficient, index in zip(coefficients, train_indices, strict=True):
            feature = coefficient * atom_rows[index] * right[None, :]
            left_numerator += (target_rows[index] * feature).sum(dim=1)
            left_denominator += feature.square().sum(dim=1)
        left = left_numerator / left_denominator.clamp_min(DENOMINATOR_FLOOR)

        right_numerator = torch.zeros_like(right)
        right_denominator = torch.zeros_like(right)
        for coefficient, index in zip(coefficients, train_indices, strict=True):
            feature = coefficient * left[:, None] * atom_rows[index]
            right_numerator += (target_rows[index] * feature).sum(dim=0)
            right_denominator += feature.square().sum(dim=0)
        right = right_numerator / right_denominator.clamp_min(DENOMINATOR_FLOOR)

        left_rms = left.square().mean().sqrt().clamp_min(DENOMINATOR_FLOOR)
        left = left / left_rms
        right = right * left_rms

    training_captures = []
    final_coefficients = []
    for index in train_indices:
        base = left[:, None] * atom_rows[index] * right[None, :]
        coefficient = optimal_scalar(base, target_rows[index])
        final_coefficients.append(float(coefficient))
        training_captures.append(squared_cosine(base, target_rows[index]))
    diagnostics = {
        "iterations": iterations,
        "train_indices": list(train_indices),
        "training_capture_minimum": min(training_captures),
        "training_capture_median": statistics.median(training_captures),
        "training_capture_maximum": max(training_captures),
        "final_training_coefficients": final_coefficients,
        "coefficient_history": coefficient_rows,
        "left_minimum": float(left.min()),
        "left_maximum": float(left.max()),
        "left_rms": float(left.square().mean().sqrt()),
        "right_minimum": float(right.min()),
        "right_maximum": float(right.max()),
        "right_rms": float(right.square().mean().sqrt()),
        "left_sha256": tensor_sha256(left),
        "right_sha256": tensor_sha256(right),
    }
    return left, right, diagnostics


def separable_transfer(
    atoms: tuple[torch.Tensor, ...],
    targets: tuple[torch.Tensor, ...],
    *,
    iterations: int = ALS_ITERATIONS,
) -> dict[str, Any]:
    if len(atoms) != len(targets) or len(atoms) < 2:
        raise ValueError("separable audit requires matching multiple nodes")
    atom_rows = tuple(normalized(value) for value in atoms)
    target_rows = tuple(normalized(value) for value in targets)
    all_indices = tuple(range(len(atoms)))
    all_left, all_right, all_fit = fit_separable_conditioner(
        atom_rows,
        target_rows,
        all_indices,
        iterations=iterations,
    )
    rows = []
    for heldout in all_indices:
        train_indices = tuple(index for index in all_indices if index != heldout)
        left, right, fit = fit_separable_conditioner(
            atom_rows,
            target_rows,
            train_indices,
            iterations=iterations,
        )
        heldout_base = left[:, None] * atom_rows[heldout] * right[None, :]
        fit_all_base = all_left[:, None] * atom_rows[heldout] * all_right[None, :]
        rows.append(
            {
                "heldout_index": heldout,
                "leave_one_out_capture": squared_cosine(
                    heldout_base, target_rows[heldout]
                ),
                "fit_all_capture": squared_cosine(
                    fit_all_base, target_rows[heldout]
                ),
                "raw_no_conditioner_capture": squared_cosine(
                    atom_rows[heldout], target_rows[heldout]
                ),
                "heldout_optimal_scalar": float(
                    optimal_scalar(heldout_base, target_rows[heldout])
                ),
                "fit": fit,
            }
        )
    loo = [row["leave_one_out_capture"] for row in rows]
    return {
        "rows": rows,
        "fit_all": all_fit,
        "minimum_leave_one_out_capture": min(loo),
        "median_leave_one_out_capture": statistics.median(loo),
        "maximum_leave_one_out_capture": max(loo),
    }


def self_test(device: str) -> dict[str, Any]:
    torch.manual_seed(211)
    shape = (97, 41)
    left = torch.exp(0.2 * torch.randn(shape[0], device=device))
    right = torch.exp(0.2 * torch.randn(shape[1], device=device))
    atoms = tuple(torch.randn(shape, device=device) for _ in range(6))
    targets = tuple(
        left[:, None] * atom * right[None, :] + 0.001 * torch.randn_like(atom)
        for atom in atoms
    )
    result = separable_transfer(atoms, targets, iterations=ALS_ITERATIONS)
    if result["minimum_leave_one_out_capture"] < 0.99:
        raise AssertionError(result)
    accounting = conditioner_accounting()
    if accounting["total_state_scalars"] != 565_272:
        raise AssertionError(accounting)
    return {
        "status": "passed",
        "minimum_leave_one_out_capture": result["minimum_leave_one_out_capture"],
        "accounting": accounting,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--trajectory-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(args.device), sort_keys=True))
        return
    if any(value is None for value in (args.config, args.plan, args.trajectory_dir, args.output)):
        parser.error("config, plan, trajectory, and output are required")
    assert args.config is not None and args.plan is not None
    assert args.trajectory_dir is not None and args.output is not None

    accounting = conditioner_accounting()
    if accounting["total_state_scalars"] != 565_272 or accounting["state_fraction"] > 0.01:
        raise ValueError("H34a accounting mismatch")
    config = json.loads(args.config.read_text())
    plan = json.loads(args.plan.read_text())
    if plan["frozen_decoder"]["total_state_scalars"] != 565_272:
        raise ValueError("plan/accounting mismatch")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()

    model = build_dense_model(config, args.device)
    prompt, targets, prompt_manifest = make_prompt(
        model,
        config,
        prompt_length=PROMPT_LENGTH,
        device=args.device,
    )
    joint_target, leading_fraction, target_manifest, w0_references = joint_leading_pc(
        args.trajectory_dir,
        parameters=FROZEN_PARAMETERS,
        device=args.device,
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
        zeropower_via_newtonschulz5(gradient, steps=5).detach()
        for gradient in gradients
    )
    split_targets = torch.split(
        joint_target,
        [weight.numel() for weight in initial_weights],
    )
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
    transfer = separable_transfer(atoms, target_parts, iterations=ALS_ITERATIONS)
    minimum_gate = transfer["minimum_leave_one_out_capture"] >= 0.05
    median_gate = transfer["median_leave_one_out_capture"] >= 0.10
    retained = minimum_gate and median_gate
    transfer.update(
        {
            "minimum_gate": minimum_gate,
            "median_gate": median_gate,
            "retained": retained,
            "leading_pc_energy_fraction": leading_fraction,
        }
    )

    torch.cuda.synchronize()
    accounting_path = output / "accounting.json"
    accounting_path.write_text(json.dumps(accounting, indent=2, sort_keys=True) + "\n")
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_mlp_shared_separable_conditioner_loo_v1",
        "classification": "RETAINED" if retained else "REJECTED",
        "retained": retained,
        "plan": plan,
        "accounting": accounting,
        "prompt_manifest": prompt_manifest,
        "target_manifest": target_manifest,
        "w0_storage_matches": w0_matches,
        "function_manifest": {
            **function_manifest,
            "task_atom": "NS5 first task gradient at W0",
            "canonical_shape": list(CANONICAL_SHAPE),
            "shared_conditioner": "row gain outer-product column gain",
            "persistent_dense_basis": False,
        },
        "transfer": transfer,
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
            "runtime_seconds": time.time() - started,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "device": args.device,
        },
        "outputs": {
            "accounting": {"path": str(accounting_path), "sha256": sha256(accounting_path)}
        },
        "limitations": [
            "The gain fields are selected on one registered joint PC, but every selection score is leave-one-node-out.",
            "A pass authorizes a tangent/transfer audit, never CE or scale-up.",
        ],
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "metadata": str(metadata_path),
                "classification": metadata["classification"],
                "minimum_leave_one_out_capture": transfer["minimum_leave_one_out_capture"],
                "median_leave_one_out_capture": transfer["median_leave_one_out_capture"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
