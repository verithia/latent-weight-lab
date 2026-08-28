#!/usr/bin/env python3
"""Joint-PC1 gates for raw and orthogonal-secanted virtual-lookahead atoms."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
from torch.func import functional_call

from examples.nanogpt.analyze_mlp_synthetic_muon_program import (
    DENSE_MLP_SCALARS,
    build_dense_model,
    initialization_match,
    make_prompt,
    math_sdpa_context,
    project_direction,
    projection_metrics,
    sha256,
)
from examples.nanogpt.analyze_mlp_synthetic_muon_program_joint import (
    FROZEN_PARAMETERS,
    joint_leading_pc,
)
from examples.nanogpt.analyze_mlp_synthetic_muon_program_twostep_joint import (
    dense_step_norms,
)
from examples.nanogpt.muon import zeropower_via_newtonschulz5


FROZEN_MOMENTUM = 0.95
DEPLOYED_MLP_MATRICES = 24


def lookahead_accounting(prompt_length: int, width: int) -> dict[str, int | float]:
    prompt_scalars = prompt_length * width
    lookahead_scalars = DEPLOYED_MLP_MATRICES
    output_coefficient_scalars = 2 * DEPLOYED_MLP_MATRICES
    total_scalars = prompt_scalars + lookahead_scalars + output_coefficient_scalars
    return {
        "prompt_length": prompt_length,
        "prompt_width": width,
        "prompt_scalars": prompt_scalars,
        "lookahead_scale_scalars": lookahead_scalars,
        "output_coefficient_scalars": output_coefficient_scalars,
        "total_scalars": total_scalars,
        "dense_mlp_denominator_scalars": DENSE_MLP_SCALARS,
        "deployable_scalar_fraction": total_scalars / DENSE_MLP_SCALARS,
        "deployable_checkpoint_fp16_bytes": 2 * total_scalars,
        "fp32_coordinate_master_bytes_during_fit": 4 * total_scalars,
        "fp32_coordinate_gradient_bytes_during_fit": 4 * total_scalars,
        "adam_fp32_moment_bytes_during_fit": 8 * total_scalars,
    }


def make_generic_lookahead_program(
    initial_weights: tuple[torch.Tensor, ...],
    loss_function: Callable[[tuple[torch.Tensor, ...], torch.Tensor], torch.Tensor],
    *,
    ns_steps: int,
    momentum: float,
    orthogonal_secant: bool = False,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    if not 0.0 <= momentum < 1.0:
        raise ValueError("momentum must be in [0, 1)")
    gradient_function = torch.func.grad(loss_function, argnums=0)

    def program(
        prompt: torch.Tensor,
        lookahead_scales: torch.Tensor,
        output_coefficients: torch.Tensor,
    ) -> torch.Tensor:
        count = len(initial_weights)
        if tuple(lookahead_scales.shape) != (count,):
            raise ValueError("lookahead scale shape mismatch")
        if tuple(output_coefficients.shape) != (2, count):
            raise ValueError("output coefficient shape mismatch")
        gradients1 = gradient_function(initial_weights, prompt)
        polar_inputs1 = tuple((1.0 + momentum) * gradient for gradient in gradients1)
        directions1 = tuple(
            zeropower_via_newtonschulz5(value, steps=ns_steps)
            for value in polar_inputs1
        )
        virtual_weights1 = tuple(
            weight - lookahead_scales[index] * direction
            for index, (weight, direction) in enumerate(
                zip(initial_weights, directions1, strict=True)
            )
        )
        gradients2 = gradient_function(virtual_weights1, prompt)
        polar_inputs2 = tuple(
            (1.0 + momentum) * gradient2 + momentum * momentum * gradient1
            for gradient1, gradient2 in zip(gradients1, gradients2, strict=True)
        )
        directions2 = tuple(
            zeropower_via_newtonschulz5(value, steps=ns_steps)
            for value in polar_inputs2
        )
        second_atoms = (
            tuple(
                normalized_orthogonal_secant(direction1, direction2)
                for direction1, direction2 in zip(
                    directions1, directions2, strict=True
                )
            )
            if orthogonal_secant
            else directions2
        )
        return torch.cat(
            [
                (
                    output_coefficients[0, index] * directions1[index]
                    + output_coefficients[1, index] * second_atoms[index]
                ).flatten()
                for index in range(count)
            ]
        )

    return program


def normalized_orthogonal_secant(
    first: torch.Tensor,
    second: torch.Tensor,
) -> torch.Tensor:
    """Remove the first-atom component and norm-match the residual."""
    if first.shape != second.shape:
        raise ValueError("lookahead atom shape mismatch")
    first_work = first.float()
    second_work = second.float()
    epsilon = torch.finfo(first_work.dtype).eps
    first_square_norm = first_work.square().sum()
    safe_square_norm = first_square_norm.clamp_min(epsilon)
    projection = (first_work * second_work).sum() / safe_square_norm
    secant = second_work - projection * first_work
    first_norm = first_square_norm.sqrt()
    safe_secant_norm = secant.square().sum().sqrt().clamp_min(
        epsilon * first_norm.detach().clamp_min(1.0)
    )
    return (secant * (first_norm / safe_secant_norm)).to(first.dtype)


def atom_geometry(
    first_atoms: tuple[torch.Tensor, ...],
    second_atoms: tuple[torch.Tensor, ...],
) -> dict[str, Any]:
    """Report whether H30c's raw atom pair is collinear or scale imbalanced."""
    if len(first_atoms) != len(second_atoms):
        raise ValueError("lookahead atom count mismatch")
    rows = []
    for index, (first, second) in enumerate(
        zip(first_atoms, second_atoms, strict=True)
    ):
        first_work = first.detach().float()
        second_work = second.detach().float()
        curvature = normalized_orthogonal_secant(first, second).detach().float()
        first_norm = first_work.norm()
        second_norm = second_work.norm()
        raw_projection = (first_work * second_work).sum() / first_work.square().sum()
        raw_secant = second_work - raw_projection * first_work
        rows.append(
            {
                "index": index,
                "raw_atom_cosine": float(
                    (first_work * second_work).sum() / (first_norm * second_norm)
                ),
                "raw_secant_to_first_norm_ratio": float(raw_secant.norm() / first_norm),
                "normalized_secant_to_first_cosine": float(
                    (first_work * curvature).sum()
                    / (first_norm * curvature.norm())
                ),
                "normalized_secant_to_first_norm_ratio": float(
                    curvature.norm() / first_norm
                ),
            }
        )
    return {
        "per_matrix": rows,
        "raw_atom_cosine_minimum": min(row["raw_atom_cosine"] for row in rows),
        "raw_atom_cosine_maximum": max(row["raw_atom_cosine"] for row in rows),
        "raw_secant_to_first_norm_ratio_minimum": min(
            row["raw_secant_to_first_norm_ratio"] for row in rows
        ),
        "raw_secant_to_first_norm_ratio_maximum": max(
            row["raw_secant_to_first_norm_ratio"] for row in rows
        ),
        "normalized_secant_absolute_cosine_maximum": max(
            abs(row["normalized_secant_to_first_cosine"]) for row in rows
        ),
        "normalized_secant_norm_ratio_maximum_error": max(
            abs(row["normalized_secant_to_first_norm_ratio"] - 1.0)
            for row in rows
        ),
    }


def raw_lookahead_atoms(
    initial_weights: tuple[torch.Tensor, ...],
    loss_function: Callable[[tuple[torch.Tensor, ...], torch.Tensor], torch.Tensor],
    prompt: torch.Tensor,
    lookahead_scales: torch.Tensor,
    *,
    ns_steps: int,
    momentum: float,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Regenerate the two raw H30c atoms once for sealed diagnostics."""
    gradient_function = torch.func.grad(loss_function, argnums=0)
    gradients1 = gradient_function(initial_weights, prompt)
    directions1 = tuple(
        zeropower_via_newtonschulz5((1.0 + momentum) * gradient, steps=ns_steps)
        for gradient in gradients1
    )
    virtual_weights1 = tuple(
        weight - lookahead_scales[index] * direction
        for index, (weight, direction) in enumerate(
            zip(initial_weights, directions1, strict=True)
        )
    )
    gradients2 = gradient_function(virtual_weights1, prompt)
    directions2 = tuple(
        zeropower_via_newtonschulz5(
            (1.0 + momentum) * gradient2 + momentum * momentum * gradient1,
            steps=ns_steps,
        )
        for gradient1, gradient2 in zip(gradients1, gradients2, strict=True)
    )
    return directions1, directions2


def calibrate_lookahead_scales(
    initial_weights: tuple[torch.Tensor, ...],
    loss_function: Callable[[tuple[torch.Tensor, ...], torch.Tensor], torch.Tensor],
    prompt: torch.Tensor,
    target_first_step_norms: torch.Tensor,
    *,
    ns_steps: int,
    momentum: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if tuple(target_first_step_norms.shape) != (len(initial_weights),):
        raise ValueError("target first-step norm shape mismatch")
    if not torch.isfinite(target_first_step_norms).all() or not (target_first_step_norms > 0).all():
        raise ValueError("target first-step norms must be finite and positive")
    gradient_function = torch.func.grad(loss_function, argnums=0)
    gradients1 = gradient_function(initial_weights, prompt)
    directions1 = tuple(
        zeropower_via_newtonschulz5((1.0 + momentum) * gradient, steps=ns_steps).detach()
        for gradient in gradients1
    )
    direction_norms = torch.stack([direction.float().norm() for direction in directions1])
    scales = (target_first_step_norms / direction_norms).detach()
    generated_norms = scales * direction_norms
    relative_error = (generated_norms - target_first_step_norms).abs() / target_first_step_norms
    manifest = {
        "momentum": momentum,
        "target_first_step_norms": target_first_step_norms.detach().cpu().tolist(),
        "raw_first_direction_norms": direction_norms.detach().cpu().tolist(),
        "lookahead_scales": scales.detach().cpu().tolist(),
        "maximum_norm_match_relative_error": float(relative_error.max()),
        "lookahead_is_not_output_scale": True,
        "stores_dense_direction_or_buffer": False,
    }
    return scales, manifest


def make_model_lookahead_program(
    model: torch.nn.Module,
    *,
    parameters: tuple[str, ...],
    targets: torch.Tensor,
    ns_steps: int,
    momentum: float,
    orthogonal_secant: bool = False,
) -> tuple[
    Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    Callable[[tuple[torch.Tensor, ...], torch.Tensor], torch.Tensor],
    tuple[torch.Tensor, ...],
    dict[str, Any],
]:
    model_parameters = {name: value.detach() for name, value in model.named_parameters()}
    buffers = {name: value.detach() for name, value in model.named_buffers()}
    missing = [name for name in parameters if name not in model_parameters]
    if missing:
        raise ValueError(f"missing selected parameters: {missing}")
    initial_weights = tuple(model_parameters[name] for name in parameters)
    selected_set = set(parameters)
    static_parameters = {
        name: value for name, value in model_parameters.items() if name not in selected_set
    }

    def task_loss(weights: tuple[torch.Tensor, ...], prompt: torch.Tensor) -> torch.Tensor:
        call_parameters = dict(static_parameters)
        call_parameters.update(zip(parameters, weights, strict=True))
        with math_sdpa_context():
            _, loss = functional_call(
                model,
                (call_parameters, buffers),
                (None, targets),
                {"input_embeddings": prompt},
                tie_weights=True,
                strict=False,
            )
        assert loss is not None
        return loss

    function = make_generic_lookahead_program(
        initial_weights,
        task_loss,
        ns_steps=ns_steps,
        momentum=momentum,
        orthogonal_secant=orthogonal_secant,
    )
    manifest = {
        "parameters": list(parameters),
        "selected_weight_shapes": {
            name: list(weight.shape)
            for name, weight in zip(parameters, initial_weights, strict=True)
        },
        "momentum": momentum,
        "lookahead_steps": 1,
        "polar_atoms": 2,
        "lookahead_decoupled_from_output": True,
        "orthogonal_secant": orthogonal_secant,
        "secant_normalization": "equal Frobenius norm to first atom",
        "output_scalars": sum(weight.numel() for weight in initial_weights),
    }
    return function, task_loss, initial_weights, manifest


def self_test(device: str, *, orthogonal_secant: bool = False) -> dict[str, Any]:
    torch.manual_seed(20260909)
    weights = (torch.randn(7, 5, device=device), torch.randn(4, 7, device=device))
    prompt = torch.randn(1, 6, 5, device=device)
    target = torch.randn(1, 6, 4, device=device)

    def loss_fn(active: tuple[torch.Tensor, ...], values: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.gelu(values @ active[0].T)
        prediction = hidden @ active[1].T
        return 0.5 * (prediction - target).square().mean()

    target_norms = torch.tensor([0.08, 0.06], device=device)
    lookahead_scales, calibration = calibrate_lookahead_scales(
        weights,
        loss_fn,
        prompt,
        target_norms,
        ns_steps=5,
        momentum=FROZEN_MOMENTUM,
    )
    function = make_generic_lookahead_program(
        weights,
        loss_fn,
        ns_steps=5,
        momentum=FROZEN_MOMENTUM,
        orthogonal_secant=orthogonal_secant,
    )
    coefficients = torch.ones(2, len(weights), device=device)
    primals = (prompt, lookahead_scales, coefficients)
    direction = (
        torch.randn_like(prompt),
        torch.randn_like(lookahead_scales),
        torch.randn_like(coefficients),
    )
    _, exact_target = torch.func.jvp(function, primals, direction)
    projected, diagnostics = project_direction(
        function,
        primals,
        exact_target,
        cg_iterations=80,
        cg_tolerance=1e-8,
        relative_ridge=1e-10,
    )
    metrics = projection_metrics(exact_target, projected)
    if metrics["path_energy_capture"] < 0.999:
        raise AssertionError((metrics, diagnostics))
    return {
        "status": "passed",
        "orthogonal_secant": orthogonal_secant,
        **metrics,
        **diagnostics,
        "calibration": calibration,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--trajectory-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt-length", type=int, default=737)
    parser.add_argument("--ns-steps", type=int, default=5)
    parser.add_argument("--momentum", type=float, default=FROZEN_MOMENTUM)
    parser.add_argument("--cg-iterations", type=int, default=20)
    parser.add_argument("--cg-tolerance", type=float, default=1e-5)
    parser.add_argument("--relative-ridge", type=float, default=1e-6)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--orthogonal-secant", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(
            json.dumps(
                self_test(args.device, orthogonal_secant=args.orthogonal_secant),
                sort_keys=True,
            )
        )
        return
    if any(value is None for value in (args.config, args.plan, args.trajectory_dir, args.output)):
        parser.error("config, plan, trajectory, and output are required")
    if args.prompt_length != 737 or args.ns_steps != 5 or args.momentum != FROZEN_MOMENTUM:
        raise ValueError("the frozen lookahead oracle requires length 737, beta=0.95, and NS5")
    expected_cg = 1 if args.preflight else 20
    if args.cg_iterations != expected_cg:
        raise ValueError(f"this lookahead mode requires {expected_cg} CG iterations")
    accounting = lookahead_accounting(args.prompt_length, 768)
    if int(accounting["total_scalars"]) != 566_088 or float(accounting["deployable_scalar_fraction"]) > 0.01:
        raise ValueError("lookahead accounting mismatch")

    assert args.output is not None and args.config is not None
    assert args.plan is not None and args.trajectory_dir is not None
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    config = json.loads(args.config.read_text())
    plan = json.loads(args.plan.read_text())
    if int(plan["frozen_decoder"]["state_scalars"]) != 566_088:
        raise ValueError("plan/accounting mismatch")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = build_dense_model(config, args.device)
    prompt, targets, prompt_manifest = make_prompt(
        model, config, prompt_length=args.prompt_length, device=args.device
    )
    torch.cuda.reset_peak_memory_stats()
    joint_target, fraction, target_manifest, w0_references = joint_leading_pc(
        args.trajectory_dir,
        parameters=FROZEN_PARAMETERS,
        device=args.device,
    )
    w0_matches = {
        parameter: initialization_match(
            dict(model.named_parameters())[parameter], w0_references[parameter]
        )
        for parameter in FROZEN_PARAMETERS
    }
    if not all(bool(record["accepted"]) for record in w0_matches.values()):
        raise ValueError(f"model/trajectory W0 mismatch: {w0_matches}")
    function, loss_function, initial_weights, function_manifest = make_model_lookahead_program(
        model,
        parameters=FROZEN_PARAMETERS,
        targets=targets,
        ns_steps=args.ns_steps,
        momentum=args.momentum,
        orthogonal_secant=args.orthogonal_secant,
    )
    registered_norms_cpu, norm_identities = dense_step_norms(
        args.trajectory_dir, FROZEN_PARAMETERS
    )
    if norm_identities != target_manifest["trajectory_identities"]:
        raise ValueError("trajectory identity changed during norm acquisition")
    target_first_norms = registered_norms_cpu[0].to(args.device)
    lookahead_scales, calibration_manifest = calibrate_lookahead_scales(
        initial_weights,
        loss_function,
        prompt,
        target_first_norms,
        ns_steps=args.ns_steps,
        momentum=args.momentum,
    )
    atom_geometry_manifest = None
    if args.orthogonal_secant:
        raw_first_atoms, raw_second_atoms = raw_lookahead_atoms(
            initial_weights,
            loss_function,
            prompt,
            lookahead_scales,
            ns_steps=args.ns_steps,
            momentum=args.momentum,
        )
        atom_geometry_manifest = atom_geometry(raw_first_atoms, raw_second_atoms)
    output_coefficients = torch.ones(2, len(FROZEN_PARAMETERS), device=args.device)
    primals = (prompt, lookahead_scales, output_coefficients)
    solve_started = time.time()
    projected, diagnostics = project_direction(
        function,
        primals,
        joint_target,
        cg_iterations=args.cg_iterations,
        cg_tolerance=args.cg_tolerance,
        relative_ridge=args.relative_ridge,
    )
    metrics = projection_metrics(joint_target, projected)
    best_possible = fraction * metrics["path_energy_capture"] + (1.0 - fraction)
    observed_gate = metrics["path_energy_capture"] >= 0.10
    ceiling_gate = best_possible >= 0.20
    retained = observed_gate and ceiling_gate
    row = {
        "path": "joint_trajectory_centered",
        "leading_pc_energy_fraction": fraction,
        "best_possible_total_capture": best_possible,
        "observed_capture_gate": observed_gate,
        "best_possible_gate": ceiling_gate,
        "retained": retained,
        "solve_seconds": time.time() - solve_started,
        **metrics,
        **diagnostics,
    }
    torch.cuda.synchronize()
    accounting_path = output / "accounting.json"
    accounting_path.write_text(json.dumps(accounting, indent=2, sort_keys=True) + "\n")
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": (
            "nanogpt_mlp_orthogonal_secant_joint_pc1_v1"
            if args.orthogonal_secant
            else "nanogpt_mlp_virtual_lookahead_joint_pc1_v1"
        ),
        "classification": "PREFLIGHT" if args.preflight else ("RETAINED" if retained else "REJECTED"),
        "preflight": args.preflight,
        "retained": retained,
        "plan": plan,
        "accounting": accounting,
        "prompt_manifest": prompt_manifest,
        "target_manifest": target_manifest,
        "w0_storage_matches": w0_matches,
        "function_manifest": function_manifest,
        "calibration_manifest": calibration_manifest,
        "atom_geometry_manifest": atom_geometry_manifest,
        "row": row,
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
            "This is a local tangent ceiling at one deterministic prompt, one norm-matched virtual state, and unit output coefficients.",
            "A pass authorizes one frozen remaining-PC/common/innovation/late audit, never CE.",
        ],
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"metadata": str(metadata_path), "classification": metadata["classification"], "row": row}, sort_keys=True))


if __name__ == "__main__":
    main()
