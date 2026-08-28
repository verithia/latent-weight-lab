#!/usr/bin/env python3
"""H35a fit-all capacity gate for shared bilateral task-atom transport."""
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


PROMPT_LENGTH = 375
PROMPT_WIDTH = 768
DEPLOYED_MLP_MATRICES = 24
TRANSPORT_RANK = 36
BINDING_ITERATIONS = 128
PREFLIGHT_ITERATIONS = 8
LEARNING_RATE = 0.01
GRADIENT_CLIP_NORM = 10.0
FIT_SEED = 353


def transport_accounting() -> dict[str, int | float]:
    rows, columns = CANONICAL_SHAPE
    prompt_scalars = PROMPT_LENGTH * PROMPT_WIDTH
    left_factor_scalars = 2 * rows * TRANSPORT_RANK
    right_factor_scalars = 2 * columns * TRANSPORT_RANK
    node_coefficient_scalars = 2 * DEPLOYED_MLP_MATRICES
    total_scalars = (
        prompt_scalars
        + left_factor_scalars
        + right_factor_scalars
        + node_coefficient_scalars
    )
    return {
        "prompt_scalars": prompt_scalars,
        "shared_left_factor_scalars": left_factor_scalars,
        "shared_right_factor_scalars": right_factor_scalars,
        "node_coefficient_scalars": node_coefficient_scalars,
        "transport_rank": TRANSPORT_RANK,
        "maximum_correction_rank": 2 * TRANSPORT_RANK,
        "total_state_scalars": total_scalars,
        "dense_mlp_denominator_scalars": DENSE_MLP_SCALARS,
        "state_fraction": total_scalars / DENSE_MLP_SCALARS,
        "fp16_checkpoint_bytes": 2 * total_scalars,
        "persistent_dense_basis_scalars": 0,
    }


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().float().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def bilateral_correction(
    atom: torch.Tensor,
    left_output: torch.Tensor,
    left_input: torch.Tensor,
    right_input: torch.Tensor,
    right_output: torch.Tensor,
) -> torch.Tensor:
    return left_output @ (left_input.T @ atom) + (atom @ right_input) @ right_output.T


def fit_bilateral_capacity(
    atoms: tuple[torch.Tensor, ...],
    targets: tuple[torch.Tensor, ...],
    *,
    rank: int = TRANSPORT_RANK,
    iterations: int = BINDING_ITERATIONS,
    learning_rate: float = LEARNING_RATE,
    seed: int = FIT_SEED,
) -> dict[str, Any]:
    if len(atoms) != len(targets) or not atoms:
        raise ValueError("bilateral fit requires nonempty matching atoms and targets")
    atom_rows = tuple(normalized(value.detach().float()) for value in atoms)
    target_rows = tuple(normalized(value.detach().float()) for value in targets)
    shape = tuple(atom_rows[0].shape)
    if len(shape) != 2 or any(tuple(value.shape) != shape for value in (*atom_rows, *target_rows)):
        raise ValueError("bilateral fit requires equal matrix shapes")
    rows, columns = shape
    if not 0 < rank <= min(rows, columns):
        raise ValueError("invalid transport rank")

    device = atom_rows[0].device
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    left_output = torch.nn.Parameter(
        torch.randn(rows, rank, device=device) / math.sqrt(rows)
    )
    left_input = torch.nn.Parameter(
        torch.randn(rows, rank, device=device) / math.sqrt(rows)
    )
    right_input = torch.nn.Parameter(
        torch.randn(columns, rank, device=device) / math.sqrt(columns)
    )
    right_output = torch.nn.Parameter(
        torch.randn(columns, rank, device=device) / math.sqrt(columns)
    )
    base_coefficients = torch.nn.Parameter(
        torch.stack(
            [optimal_scalar(atom, target) for atom, target in zip(atom_rows, target_rows, strict=True)]
        )
    )
    correction_coefficients = torch.nn.Parameter(
        torch.ones(len(atom_rows), device=device)
    )
    parameters = [
        left_output,
        left_input,
        right_input,
        right_output,
        base_coefficients,
        correction_coefficients,
    ]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=0.0)

    raw_captures = [
        squared_cosine(atom, target)
        for atom, target in zip(atom_rows, target_rows, strict=True)
    ]
    loss_history: list[dict[str, float | int]] = []
    capture_history: list[dict[str, float | int]] = []
    recorded_steps = {0, 1, 2, 3, 7, 15, 31, 63, 95, 127}
    initial_loss = None
    for step in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        step_captures = []
        for index, (atom, target) in enumerate(zip(atom_rows, target_rows, strict=True)):
            correction = bilateral_correction(
                atom, left_output, left_input, right_input, right_output
            )
            prediction = (
                base_coefficients[index] * atom
                + correction_coefficients[index] * correction
            )
            node_loss = (prediction - target).square().sum() / len(atom_rows)
            node_loss.backward()
            loss_value += float(node_loss.detach())
            if step in recorded_steps or step == iterations - 1:
                step_captures.append(squared_cosine(prediction.detach(), target))
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP_NORM)
        )
        optimizer.step()
        if initial_loss is None:
            initial_loss = loss_value
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
    correction_norms = []
    for index, (atom, target) in enumerate(zip(atom_rows, target_rows, strict=True)):
        correction = bilateral_correction(
            atom, left_output, left_input, right_input, right_output
        )
        prediction = (
            base_coefficients[index] * atom
            + correction_coefficients[index] * correction
        )
        captures.append(squared_cosine(prediction, target))
        correction_norms.append(float(correction.norm()))
    return {
        "iterations": iterations,
        "rank": rank,
        "learning_rate": learning_rate,
        "seed": seed,
        "initial_loss": initial_loss,
        "final_recorded_loss": loss_history[-1]["loss"],
        "loss_history": loss_history,
        "capture_history": capture_history,
        "raw_captures": raw_captures,
        "fit_captures": captures,
        "minimum_fit_capture": min(captures),
        "median_fit_capture": statistics.median(captures),
        "maximum_fit_capture": max(captures),
        "correction_norms": correction_norms,
        "base_coefficients": [float(value) for value in base_coefficients.detach()],
        "correction_coefficients": [float(value) for value in correction_coefficients.detach()],
        "factor_norms": {
            "left_output": float(left_output.norm()),
            "left_input": float(left_input.norm()),
            "right_input": float(right_input.norm()),
            "right_output": float(right_output.norm()),
        },
        "factor_sha256": {
            "left_output": tensor_sha256(left_output),
            "left_input": tensor_sha256(left_input),
            "right_input": tensor_sha256(right_input),
            "right_output": tensor_sha256(right_output),
        },
    }


def self_test(device_name: str) -> dict[str, Any]:
    device = torch.device(device_name)
    torch.manual_seed(29)
    shape = (24, 12)
    rank = 3
    atoms = tuple(torch.randn(shape, device=device) for _ in range(4))
    left_output = torch.randn(shape[0], rank, device=device) / math.sqrt(shape[0])
    left_input = torch.randn(shape[0], rank, device=device) / math.sqrt(shape[0])
    right_input = torch.randn(shape[1], rank, device=device) / math.sqrt(shape[1])
    right_output = torch.randn(shape[1], rank, device=device) / math.sqrt(shape[1])
    targets = tuple(
        0.4 * normalized(atom)
        + bilateral_correction(
            normalized(atom), left_output, left_input, right_input, right_output
        )
        for atom in atoms
    )
    result = fit_bilateral_capacity(
        atoms,
        targets,
        rank=rank,
        iterations=192,
        learning_rate=0.01,
        seed=353,
    )
    if result["median_fit_capture"] < 0.90:
        raise AssertionError(result)
    accounting = transport_accounting()
    if accounting["total_state_scalars"] != 564_528 or accounting["state_fraction"] > 0.01:
        raise AssertionError(accounting)
    return {"status": "passed", "fit": result, "accounting": accounting}


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

    accounting = transport_accounting()
    if accounting["total_state_scalars"] != 564_528 or accounting["state_fraction"] > 0.01:
        raise ValueError("H35a accounting mismatch")
    config = json.loads(args.config.read_text())
    plan = json.loads(args.plan.read_text())
    if plan["frozen_decoder"]["total_state_scalars"] != 564_528:
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
    fit = fit_bilateral_capacity(atoms, target_parts, iterations=iterations)
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
        "schema_version": "nanogpt_mlp_shared_bilateral_transport_capacity_v1",
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
            "shared_conditioner": "rank-36 additive bilateral left/right transport",
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
            "H35a is a fit-all capacity discriminator; it does not measure transfer.",
            "A pass authorizes only a separately preregistered leave-one-node-out audit, never CE or scale-up.",
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
