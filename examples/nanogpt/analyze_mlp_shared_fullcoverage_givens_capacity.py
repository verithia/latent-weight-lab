#!/usr/bin/env python3
"""H36a fit-all capacity gate for a shared full-coverage Givens transport."""
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
from typing import Any

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


PROMPT_LENGTH = 371
PROMPT_WIDTH = 768
DEPLOYED_MLP_MATRICES = 24
STAGES = 146
PERMUTATION_SEED = 367
BINDING_ITERATIONS = 64
PREFLIGHT_ITERATIONS = 4
LEARNING_RATE = 0.01
GRADIENT_CLIP_NORM = 10.0
SMOOTH_ABSOLUTE_EPSILON = 1e-12


def givens_accounting() -> dict[str, int | float]:
    rows, columns = CANONICAL_SHAPE
    prompt_scalars = PROMPT_LENGTH * PROMPT_WIDTH
    hidden_angle_scalars = STAGES * (rows // 2)
    residual_angle_scalars = STAGES * (columns // 2)
    total_scalars = (
        prompt_scalars
        + hidden_angle_scalars
        + residual_angle_scalars
        + DEPLOYED_MLP_MATRICES
    )
    return {
        "prompt_scalars": prompt_scalars,
        "shared_hidden_angle_scalars": hidden_angle_scalars,
        "shared_residual_angle_scalars": residual_angle_scalars,
        "node_coefficient_scalars": DEPLOYED_MLP_MATRICES,
        "stages": STAGES,
        "permutation_seed": PERMUTATION_SEED,
        "total_state_scalars": total_scalars,
        "dense_mlp_denominator_scalars": DENSE_MLP_SCALARS,
        "state_fraction": total_scalars / DENSE_MLP_SCALARS,
        "fp16_checkpoint_bytes": 2 * total_scalars,
        "persistent_dense_basis_scalars": 0,
        "persistent_permutation_scalars": 0,
    }


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().float().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def make_stage_permutations(
    size: int,
    stages: int,
    seed: int,
    device: torch.device,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    if size % 2:
        raise ValueError("pair rotations require an even channel count")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    rows = []
    for _ in range(stages):
        permutation = torch.randperm(size, generator=generator)
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(size)
        rows.append((permutation.to(device), inverse.to(device)))
    return tuple(rows)


def rotate_rows(
    value: torch.Tensor,
    angles: torch.Tensor,
    permutation: torch.Tensor,
    inverse: torch.Tensor,
) -> torch.Tensor:
    permuted = value.index_select(0, permutation)
    first = permuted[0::2]
    second = permuted[1::2]
    cosine = angles.cos()[:, None]
    sine = angles.sin()[:, None]
    rotated = torch.stack(
        (cosine * first - sine * second, sine * first + cosine * second),
        dim=1,
    ).reshape_as(permuted)
    return rotated.index_select(0, inverse)


def fullcoverage_transport(
    atom: torch.Tensor,
    hidden_angles: torch.Tensor,
    residual_angles: torch.Tensor,
    hidden_permutations: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    residual_permutations: tuple[tuple[torch.Tensor, torch.Tensor], ...],
) -> torch.Tensor:
    result = atom
    for angles, (permutation, inverse) in zip(
        hidden_angles, hidden_permutations, strict=True
    ):
        result = rotate_rows(result, angles, permutation, inverse)
    result = result.T
    for angles, (permutation, inverse) in zip(
        residual_angles, residual_permutations, strict=True
    ):
        result = rotate_rows(result, angles, permutation, inverse)
    return result.T


def fit_fullcoverage_capacity(
    atoms: tuple[torch.Tensor, ...],
    targets: tuple[torch.Tensor, ...],
    *,
    stages: int = STAGES,
    iterations: int = BINDING_ITERATIONS,
    learning_rate: float = LEARNING_RATE,
    permutation_seed: int = PERMUTATION_SEED,
) -> dict[str, Any]:
    if len(atoms) != len(targets) or not atoms:
        raise ValueError("Givens fit requires nonempty matching atoms and targets")
    atom_rows = tuple(normalized(value.detach().float()) for value in atoms)
    target_rows = tuple(normalized(value.detach().float()) for value in targets)
    shape = tuple(atom_rows[0].shape)
    if len(shape) != 2 or any(tuple(value.shape) != shape for value in (*atom_rows, *target_rows)):
        raise ValueError("Givens fit requires equal matrix shapes")
    rows, columns = shape
    device = atom_rows[0].device
    hidden_permutations = make_stage_permutations(
        rows, stages, permutation_seed, device
    )
    residual_permutations = make_stage_permutations(
        columns, stages, permutation_seed + 1, device
    )
    hidden_angles = torch.nn.Parameter(torch.zeros(stages, rows // 2, device=device))
    residual_angles = torch.nn.Parameter(
        torch.zeros(stages, columns // 2, device=device)
    )
    parameters = [hidden_angles, residual_angles]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=0.0)

    raw_captures = [
        squared_cosine(atom, target)
        for atom, target in zip(atom_rows, target_rows, strict=True)
    ]
    loss_history: list[dict[str, float | int]] = []
    capture_history: list[dict[str, float | int]] = []
    recorded_steps = {0, 1, 2, 3, 7, 15, 31, 63}
    for step in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        step_captures = []
        for atom, target in zip(atom_rows, target_rows, strict=True):
            prediction = fullcoverage_transport(
                atom,
                hidden_angles,
                residual_angles,
                hidden_permutations,
                residual_permutations,
            )
            inner = (prediction * target).sum()
            node_loss = -torch.sqrt(inner.square() + SMOOTH_ABSOLUTE_EPSILON) / len(atom_rows)
            node_loss.backward()
            loss_value += float(node_loss.detach())
            if step in recorded_steps or step == iterations - 1:
                step_captures.append(float(inner.detach().square()))
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP_NORM)
        )
        optimizer.step()
        if step in recorded_steps or step == iterations - 1:
            loss_history.append(
                {"iteration": step + 1, "loss": loss_value, "gradient_norm": gradient_norm}
            )
            capture_history.append(
                {
                    "iteration": step + 1,
                    "minimum_capture": min(step_captures),
                    "median_capture": statistics.median(step_captures),
                    "maximum_capture": max(step_captures),
                }
            )

    captures = []
    coefficients = []
    norm_errors = []
    with torch.no_grad():
        for atom, target in zip(atom_rows, target_rows, strict=True):
            prediction = fullcoverage_transport(
                atom,
                hidden_angles,
                residual_angles,
                hidden_permutations,
                residual_permutations,
            )
            captures.append(squared_cosine(prediction, target))
            coefficients.append(float(optimal_scalar(prediction, target)))
            norm_errors.append(abs(float(prediction.norm()) - 1.0))
    return {
        "iterations": iterations,
        "stages": stages,
        "learning_rate": learning_rate,
        "permutation_seed": permutation_seed,
        "loss_history": loss_history,
        "capture_history": capture_history,
        "raw_captures": raw_captures,
        "fit_captures": captures,
        "minimum_fit_capture": min(captures),
        "median_fit_capture": statistics.median(captures),
        "maximum_fit_capture": max(captures),
        "optimal_node_coefficients": coefficients,
        "maximum_norm_error": max(norm_errors),
        "angle_norms": {
            "hidden": float(hidden_angles.norm()),
            "residual": float(residual_angles.norm()),
        },
        "angle_sha256": {
            "hidden": tensor_sha256(hidden_angles),
            "residual": tensor_sha256(residual_angles),
        },
    }


def self_test(device_name: str) -> dict[str, Any]:
    device = torch.device(device_name)
    torch.manual_seed(31)
    shape = (16, 8)
    stages = 4
    atoms = tuple(normalized(torch.randn(shape, device=device)) for _ in range(3))
    hidden_permutations = make_stage_permutations(shape[0], stages, 17, device)
    residual_permutations = make_stage_permutations(shape[1], stages, 18, device)
    zero_hidden = torch.zeros(stages, shape[0] // 2, device=device)
    zero_residual = torch.zeros(stages, shape[1] // 2, device=device)
    identity = fullcoverage_transport(
        atoms[0], zero_hidden, zero_residual, hidden_permutations, residual_permutations
    )
    if not torch.equal(identity, atoms[0]):
        raise AssertionError("zero-angle transform is not exact identity")
    true_hidden = 0.2 * torch.randn_like(zero_hidden)
    true_residual = 0.2 * torch.randn_like(zero_residual)
    targets = tuple(
        fullcoverage_transport(
            atom,
            true_hidden,
            true_residual,
            hidden_permutations,
            residual_permutations,
        )
        for atom in atoms
    )
    fit = fit_fullcoverage_capacity(
        atoms,
        targets,
        stages=stages,
        iterations=96,
        learning_rate=0.01,
        permutation_seed=17,
    )
    if fit["median_fit_capture"] < 0.90 or fit["maximum_norm_error"] > 1e-5:
        raise AssertionError(fit)
    accounting = givens_accounting()
    if accounting["total_state_scalars"] != 565_272 or accounting["state_fraction"] > 0.01:
        raise AssertionError(accounting)
    return {"status": "passed", "fit": fit, "accounting": accounting}


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

    accounting = givens_accounting()
    if accounting["total_state_scalars"] != 565_272 or accounting["state_fraction"] > 0.01:
        raise ValueError("H36a accounting mismatch")
    config = json.loads(args.config.read_text())
    plan = json.loads(args.plan.read_text())
    if plan["frozen_decoder"]["total_state_scalars"] != 565_272:
        raise ValueError("plan/accounting mismatch")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
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
        zeropower_via_newtonschulz5(gradient, steps=5).detach()
        for gradient in gradients
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
    iterations = PREFLIGHT_ITERATIONS if args.preflight else BINDING_ITERATIONS
    fit = fit_fullcoverage_capacity(atoms, target_parts, iterations=iterations)
    minimum_gate = fit["minimum_fit_capture"] >= 0.05
    median_gate = fit["median_fit_capture"] >= 0.10
    retained = minimum_gate and median_gate and not args.preflight
    fit.update(
        {
            "minimum_gate": minimum_gate,
            "median_gate": median_gate,
            "retained": retained,
            "leading_pc_energy_fraction": leading_fraction,
        }
    )

    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    accounting_path = output / "accounting.json"
    accounting_path.write_text(json.dumps(accounting, indent=2, sort_keys=True) + "\n")
    script = Path(__file__).resolve()
    runtime_seconds = time.time() - started
    metadata = {
        "schema_version": "nanogpt_mlp_shared_fullcoverage_givens_capacity_v1",
        "classification": "PREFLIGHT" if args.preflight else ("RETAINED" if retained else "REJECTED"),
        "retained": retained,
        "preflight": args.preflight,
        "plan": plan,
        "accounting": accounting,
        "prompt_manifest": prompt_manifest,
        "target_manifest": target_manifest,
        "w0_storage_matches": w0_matches,
        "function_manifest": {
            **function_manifest,
            "task_atom": "NS5 first task gradient at W0",
            "canonical_shape": list(CANONICAL_SHAPE),
            "shared_conditioner": "146 procedurally permuted full-coverage Givens stages per axis",
            "singular_value_invariant": True,
            "persistent_dense_basis": False,
        },
        "fit": fit,
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
            "projected_binding_runtime_seconds": (
                runtime_seconds * BINDING_ITERATIONS / PREFLIGHT_ITERATIONS
                if args.preflight
                else runtime_seconds
            ),
            "peak_cuda_allocated_bytes": (
                torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else 0
            ),
            "device": args.device,
        },
        "outputs": {
            "accounting": {"path": str(accounting_path), "sha256": sha256(accounting_path)}
        },
        "limitations": [
            "H36a is fit-all capacity, not transfer.",
            "The orthogonal transport cannot change task-atom singular values except for one node scalar.",
            "A pass authorizes only H36b LOO, never CE or scale-up.",
        ],
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "metadata": str(metadata_path),
                "classification": metadata["classification"],
                "minimum_fit_capture": fit["minimum_fit_capture"],
                "median_fit_capture": fit["median_fit_capture"],
                "runtime_seconds": runtime_seconds,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
