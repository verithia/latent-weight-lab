#!/usr/bin/env python3
"""H37a fit-all capacity gate for a shared full-coverage non-orthogonal flow."""
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

from examples.nanogpt.analyze_mlp_shared_fullcoverage_givens_capacity import (
    make_stage_permutations,
)
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
STAGES = 36
PARAMETERS_PER_PAIR = 4
PERMUTATION_SEED = 379
BINDING_ITERATIONS = 64
PREFLIGHT_ITERATIONS = 4
LEARNING_RATE = 0.01
GRADIENT_CLIP_NORM = 10.0
SMOOTH_ABSOLUTE_EPSILON = 1e-12
LOG_GAIN_BOUND = 1.5
SHEAR_BOUND = 1.0


def gl2_accounting() -> dict[str, int | float]:
    rows, columns = CANONICAL_SHAPE
    prompt_scalars = PROMPT_LENGTH * PROMPT_WIDTH
    block_scalars = STAGES * PARAMETERS_PER_PAIR * (rows // 2 + columns // 2)
    total_scalars = prompt_scalars + block_scalars + DEPLOYED_MLP_MATRICES
    return {
        "prompt_scalars": prompt_scalars,
        "shared_gl2_block_scalars": block_scalars,
        "node_coefficient_scalars": DEPLOYED_MLP_MATRICES,
        "stages": STAGES,
        "parameters_per_pair": PARAMETERS_PER_PAIR,
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


def gl2_rows(
    value: torch.Tensor,
    raw_blocks: torch.Tensor,
    permutation: torch.Tensor,
    inverse: torch.Tensor,
) -> torch.Tensor:
    permuted = value.index_select(0, permutation)
    first = permuted[0::2]
    second = permuted[1::2]
    theta, raw_gain_one, raw_gain_two, raw_shear = raw_blocks.unbind(dim=1)
    cosine = theta.cos()[:, None]
    sine = theta.sin()[:, None]
    gain_one = (LOG_GAIN_BOUND * raw_gain_one.tanh()).exp()[:, None]
    gain_two = (LOG_GAIN_BOUND * raw_gain_two.tanh()).exp()[:, None]
    shear = (SHEAR_BOUND * raw_shear.tanh())[:, None]
    triangular_one = gain_one * first + shear * second
    triangular_two = gain_two * second
    rotated = torch.stack(
        (
            cosine * triangular_one - sine * triangular_two,
            sine * triangular_one + cosine * triangular_two,
        ),
        dim=1,
    ).reshape_as(permuted)
    return rotated.index_select(0, inverse)


def fullcoverage_gl2_transport(
    atom: torch.Tensor,
    hidden_blocks: torch.Tensor,
    residual_blocks: torch.Tensor,
    hidden_permutations: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    residual_permutations: tuple[tuple[torch.Tensor, torch.Tensor], ...],
) -> torch.Tensor:
    result = atom
    for blocks, (permutation, inverse) in zip(
        hidden_blocks, hidden_permutations, strict=True
    ):
        result = gl2_rows(result, blocks, permutation, inverse)
    result = result.T
    for blocks, (permutation, inverse) in zip(
        residual_blocks, residual_permutations, strict=True
    ):
        result = gl2_rows(result, blocks, permutation, inverse)
    return result.T


def fit_gl2_capacity(
    atoms: tuple[torch.Tensor, ...],
    targets: tuple[torch.Tensor, ...],
    *,
    stages: int = STAGES,
    iterations: int = BINDING_ITERATIONS,
    learning_rate: float = LEARNING_RATE,
    permutation_seed: int = PERMUTATION_SEED,
) -> dict[str, Any]:
    if len(atoms) != len(targets) or not atoms:
        raise ValueError("GL2 fit requires nonempty matching atoms and targets")
    atom_rows = tuple(normalized(value.detach().float()) for value in atoms)
    target_rows = tuple(normalized(value.detach().float()) for value in targets)
    shape = tuple(atom_rows[0].shape)
    if len(shape) != 2 or any(tuple(value.shape) != shape for value in (*atom_rows, *target_rows)):
        raise ValueError("GL2 fit requires equal matrix shapes")
    rows, columns = shape
    device = atom_rows[0].device
    hidden_permutations = make_stage_permutations(rows, stages, permutation_seed, device)
    residual_permutations = make_stage_permutations(
        columns, stages, permutation_seed + 1, device
    )
    hidden_blocks = torch.nn.Parameter(
        torch.zeros(stages, rows // 2, PARAMETERS_PER_PAIR, device=device)
    )
    residual_blocks = torch.nn.Parameter(
        torch.zeros(stages, columns // 2, PARAMETERS_PER_PAIR, device=device)
    )
    parameters = [hidden_blocks, residual_blocks]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=0.0)

    raw_captures = [
        squared_cosine(atom, target)
        for atom, target in zip(atom_rows, target_rows, strict=True)
    ]
    loss_history: list[dict[str, float | int]] = []
    capture_history: list[dict[str, float | int]] = []
    norm_history: list[dict[str, float | int]] = []
    recorded_steps = {0, 1, 2, 3, 7, 15, 31, 63}
    for step in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        step_captures = []
        step_norms = []
        for atom, target in zip(atom_rows, target_rows, strict=True):
            prediction = fullcoverage_gl2_transport(
                atom,
                hidden_blocks,
                residual_blocks,
                hidden_permutations,
                residual_permutations,
            )
            prediction_norm = prediction.norm().clamp_min(1e-12)
            cosine = (prediction * target).sum() / prediction_norm
            node_loss = -torch.sqrt(cosine.square() + SMOOTH_ABSOLUTE_EPSILON) / len(atom_rows)
            node_loss.backward()
            loss_value += float(node_loss.detach())
            if step in recorded_steps or step == iterations - 1:
                step_captures.append(float(cosine.detach().square()))
                step_norms.append(float(prediction_norm.detach()))
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
            norm_history.append(
                {
                    "iteration": step + 1,
                    "minimum_output_norm": min(step_norms),
                    "maximum_output_norm": max(step_norms),
                    "output_norm_ratio": max(step_norms) / min(step_norms),
                }
            )

    captures = []
    coefficients = []
    output_norms = []
    finite = True
    with torch.no_grad():
        for atom, target in zip(atom_rows, target_rows, strict=True):
            prediction = fullcoverage_gl2_transport(
                atom,
                hidden_blocks,
                residual_blocks,
                hidden_permutations,
                residual_permutations,
            )
            finite = finite and bool(torch.isfinite(prediction).all())
            captures.append(squared_cosine(prediction, target))
            coefficients.append(float(optimal_scalar(prediction, target)))
            output_norms.append(float(prediction.norm()))
    output_norm_ratio = max(output_norms) / max(min(output_norms), 1e-30)
    return {
        "iterations": iterations,
        "stages": stages,
        "learning_rate": learning_rate,
        "permutation_seed": permutation_seed,
        "loss_history": loss_history,
        "capture_history": capture_history,
        "norm_history": norm_history,
        "raw_captures": raw_captures,
        "fit_captures": captures,
        "minimum_fit_capture": min(captures),
        "median_fit_capture": statistics.median(captures),
        "maximum_fit_capture": max(captures),
        "optimal_node_coefficients": coefficients,
        "finite_outputs": finite,
        "output_norms": output_norms,
        "output_norm_ratio": output_norm_ratio,
        "block_norms": {
            "hidden": float(hidden_blocks.detach().norm()),
            "residual": float(residual_blocks.detach().norm()),
        },
        "block_sha256": {
            "hidden": tensor_sha256(hidden_blocks),
            "residual": tensor_sha256(residual_blocks),
        },
    }


def self_test(device_name: str) -> dict[str, Any]:
    device = torch.device(device_name)
    torch.manual_seed(37)
    shape = (16, 8)
    stages = 3
    atoms = tuple(normalized(torch.randn(shape, device=device)) for _ in range(3))
    hidden_permutations = make_stage_permutations(shape[0], stages, 23, device)
    residual_permutations = make_stage_permutations(shape[1], stages, 24, device)
    zero_hidden = torch.zeros(stages, shape[0] // 2, PARAMETERS_PER_PAIR, device=device)
    zero_residual = torch.zeros(stages, shape[1] // 2, PARAMETERS_PER_PAIR, device=device)
    identity = fullcoverage_gl2_transport(
        atoms[0], zero_hidden, zero_residual, hidden_permutations, residual_permutations
    )
    if not torch.equal(identity, atoms[0]):
        raise AssertionError("zero-state GL2 flow is not exact identity")
    true_hidden = 0.12 * torch.randn_like(zero_hidden)
    true_residual = 0.12 * torch.randn_like(zero_residual)
    targets = tuple(
        fullcoverage_gl2_transport(
            atom,
            true_hidden,
            true_residual,
            hidden_permutations,
            residual_permutations,
        )
        for atom in atoms
    )
    fit = fit_gl2_capacity(
        atoms,
        targets,
        stages=stages,
        iterations=96,
        learning_rate=0.01,
        permutation_seed=23,
    )
    if fit["median_fit_capture"] < 0.90 or not fit["finite_outputs"]:
        raise AssertionError(fit)
    accounting = gl2_accounting()
    if accounting["total_state_scalars"] != 564_504 or accounting["state_fraction"] > 0.01:
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

    accounting = gl2_accounting()
    if accounting["total_state_scalars"] != 564_504 or accounting["state_fraction"] > 0.01:
        raise ValueError("H37a accounting mismatch")
    config = json.loads(args.config.read_text())
    plan = json.loads(args.plan.read_text())
    if plan["frozen_decoder"]["total_state_scalars"] != 564_504:
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
    fit = fit_gl2_capacity(atoms, target_parts, iterations=iterations)
    finite_gate = fit["finite_outputs"] and fit["output_norm_ratio"] <= 10_000.0
    minimum_gate = fit["minimum_fit_capture"] >= 0.05
    median_gate = fit["median_fit_capture"] >= 0.10
    retained = finite_gate and minimum_gate and median_gate and not args.preflight
    fit.update(
        {
            "finite_gate": finite_gate,
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
        "schema_version": "nanogpt_mlp_shared_fullcoverage_gl2_capacity_v1",
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
            "shared_conditioner": "36 procedurally permuted full-coverage bounded GL2 QR stages per axis",
            "singular_value_invariant": False,
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
            "H37a is fit-all capacity, not transfer.",
            "The bounded GL2 flow is only one non-orthogonal full-rank factorization.",
            "A pass authorizes only H37b LOO, never CE or scale-up.",
        ],
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "metadata": str(metadata_path),
                "classification": metadata["classification"],
                "finite_gate": finite_gate,
                "minimum_fit_capture": fit["minimum_fit_capture"],
                "median_fit_capture": fit["median_fit_capture"],
                "runtime_seconds": runtime_seconds,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
