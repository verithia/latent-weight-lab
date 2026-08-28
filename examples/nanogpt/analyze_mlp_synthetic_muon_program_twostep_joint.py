#!/usr/bin/env python3
"""Joint-PC1 gate for the H30a two-step synthetic Muon program.

This is a representation oracle, not CE training.  A single compact prompt
is replayed twice from W0.  Each replay takes a task-gradient descent step
whose matrix direction is obtained by NS5.  The second gradient is evaluated
at the generated W1, so the decoder tangent can rotate with generated state.
"""
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
    load_trajectory_parameter,
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
from examples.nanogpt.muon import zeropower_via_newtonschulz5


DEPLOYED_MLP_MATRICES = 24
MACROSTEPS = 2


def two_step_latent_accounting(prompt_length: int, width: int) -> dict[str, int | float]:
    prompt_scalars = prompt_length * width
    amplitude_scalars = MACROSTEPS * DEPLOYED_MLP_MATRICES
    total_scalars = prompt_scalars + amplitude_scalars
    return {
        "prompt_length": prompt_length,
        "prompt_width": width,
        "prompt_scalars": prompt_scalars,
        "macrosteps": MACROSTEPS,
        "deployed_mlp_matrices": DEPLOYED_MLP_MATRICES,
        "amplitude_scalars": amplitude_scalars,
        "total_scalars": total_scalars,
        "dense_mlp_denominator_scalars": DENSE_MLP_SCALARS,
        "deployable_scalar_fraction": total_scalars / DENSE_MLP_SCALARS,
        "deployable_checkpoint_fp16_bytes": 2 * total_scalars,
        "fp32_coordinate_master_bytes_during_fit": 4 * total_scalars,
        "fp32_coordinate_gradient_bytes_during_fit": 4 * total_scalars,
        "adam_fp32_moment_bytes_during_fit": 8 * total_scalars,
    }


def make_generic_two_step_program(
    initial_weights: tuple[torch.Tensor, ...],
    loss_function: Callable[[tuple[torch.Tensor, ...], torch.Tensor], torch.Tensor],
    *,
    ns_steps: int,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return prompt/amplitude -> concatenated W2-W0 displacement."""
    gradient_function = torch.func.grad(loss_function, argnums=0)

    def program(prompt: torch.Tensor, amplitudes: torch.Tensor) -> torch.Tensor:
        expected = (MACROSTEPS, len(initial_weights))
        if tuple(amplitudes.shape) != expected:
            raise ValueError(f"two-step amplitude shape mismatch: {tuple(amplitudes.shape)} != {expected}")
        gradients1 = gradient_function(initial_weights, prompt)
        directions1 = tuple(
            zeropower_via_newtonschulz5(gradient, steps=ns_steps)
            for gradient in gradients1
        )
        weights1 = tuple(
            weight - amplitudes[0, index] * direction
            for index, (weight, direction) in enumerate(
                zip(initial_weights, directions1, strict=True)
            )
        )
        gradients2 = gradient_function(weights1, prompt)
        directions2 = tuple(
            zeropower_via_newtonschulz5(gradient, steps=ns_steps)
            for gradient in gradients2
        )
        return torch.cat(
            [
                (
                    -amplitudes[0, index] * directions1[index]
                    - amplitudes[1, index] * directions2[index]
                ).flatten()
                for index in range(len(initial_weights))
            ]
        )

    return program


def calibrate_two_step_amplitudes(
    initial_weights: tuple[torch.Tensor, ...],
    loss_function: Callable[[tuple[torch.Tensor, ...], torch.Tensor], torch.Tensor],
    prompt: torch.Tensor,
    target_step_norms: torch.Tensor,
    *,
    ns_steps: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Norm-match two generated descent steps without importing directions."""
    expected = (MACROSTEPS, len(initial_weights))
    if tuple(target_step_norms.shape) != expected:
        raise ValueError("target step norm shape mismatch")
    if not torch.isfinite(target_step_norms).all() or not (target_step_norms > 0).all():
        raise ValueError("target step norms must be finite and positive")
    gradient_function = torch.func.grad(loss_function, argnums=0)
    gradients1 = gradient_function(initial_weights, prompt)
    directions1 = tuple(
        zeropower_via_newtonschulz5(gradient, steps=ns_steps).detach()
        for gradient in gradients1
    )
    norms1 = torch.stack([direction.float().norm() for direction in directions1])
    amplitudes1 = target_step_norms[0] / norms1
    weights1 = tuple(
        (weight - amplitudes1[index] * direction).detach()
        for index, (weight, direction) in enumerate(
            zip(initial_weights, directions1, strict=True)
        )
    )
    gradients2 = gradient_function(weights1, prompt)
    directions2 = tuple(
        zeropower_via_newtonschulz5(gradient, steps=ns_steps).detach()
        for gradient in gradients2
    )
    norms2 = torch.stack([direction.float().norm() for direction in directions2])
    amplitudes2 = target_step_norms[1] / norms2
    amplitudes = torch.stack((amplitudes1, amplitudes2)).detach()
    generated = torch.stack((amplitudes1 * norms1, amplitudes2 * norms2))
    relative_error = (generated - target_step_norms).abs() / target_step_norms
    manifest = {
        "target_step_norms": target_step_norms.detach().cpu().tolist(),
        "raw_direction_norms": torch.stack((norms1, norms2)).detach().cpu().tolist(),
        "anchor_amplitudes": amplitudes.detach().cpu().tolist(),
        "maximum_norm_match_relative_error": float(relative_error.max()),
        "uses_only_dense_step_norms": True,
        "stores_dense_direction": False,
    }
    return amplitudes, manifest


def make_model_two_step_program(
    model: torch.nn.Module,
    *,
    parameters: tuple[str, ...],
    targets: torch.Tensor,
    ns_steps: int,
) -> tuple[
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
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

    function = make_generic_two_step_program(
        initial_weights,
        task_loss,
        ns_steps=ns_steps,
    )
    manifest = {
        "parameters": list(parameters),
        "selected_weight_shapes": {
            name: list(weight.shape)
            for name, weight in zip(parameters, initial_weights, strict=True)
        },
        "macrosteps": MACROSTEPS,
        "descent_sign": -1,
        "same_prompt_and_targets_each_step": True,
        "output_scalars": sum(weight.numel() for weight in initial_weights),
    }
    return function, task_loss, initial_weights, manifest


def dense_step_norms(
    trajectory_dir: Path,
    parameters: tuple[str, ...],
) -> tuple[torch.Tensor, dict[str, str]]:
    norms = torch.empty(MACROSTEPS, len(parameters), dtype=torch.float32)
    identities: dict[str, str] = {}
    for index, parameter in enumerate(parameters):
        states, identity = load_trajectory_parameter(trajectory_dir, parameter)
        if states.shape[0] < 3:
            raise ValueError("trajectory lacks the first two dense displacements")
        for step in range(MACROSTEPS):
            norms[step, index] = (states[step + 1].float() - states[step].float()).norm()
        identities[parameter] = identity
        del states
    if not torch.isfinite(norms).all() or not (norms > 0).all():
        raise ValueError("invalid dense step norms")
    return norms, identities


def self_test(device: str) -> dict[str, Any]:
    torch.manual_seed(20260906)
    weights = (torch.randn(7, 5, device=device), torch.randn(4, 7, device=device))
    prompt = torch.randn(1, 6, 5, device=device)
    target = torch.randn(1, 6, 4, device=device)

    def loss_fn(active: tuple[torch.Tensor, ...], values: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.gelu(values @ active[0].T)
        prediction = hidden @ active[1].T
        return 0.5 * (prediction - target).square().mean()

    target_norms = torch.tensor([[0.08, 0.06], [0.05, 0.04]], device=device)
    amplitudes, calibration = calibrate_two_step_amplitudes(
        weights, loss_fn, prompt, target_norms, ns_steps=5
    )
    function = make_generic_two_step_program(weights, loss_fn, ns_steps=5)
    primals = (prompt, amplitudes)
    direction = (torch.randn_like(prompt), torch.randn_like(amplitudes))
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
    if calibration["maximum_norm_match_relative_error"] > 1e-6:
        raise AssertionError(calibration)
    return {"status": "passed", **metrics, **diagnostics, "calibration": calibration}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--trajectory-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt-length", type=int, default=737)
    parser.add_argument("--ns-steps", type=int, default=5)
    parser.add_argument("--cg-iterations", type=int, default=20)
    parser.add_argument("--cg-tolerance", type=float, default=1e-5)
    parser.add_argument("--relative-ridge", type=float, default=1e-6)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(args.device), sort_keys=True))
        return
    if any(value is None for value in (args.config, args.plan, args.trajectory_dir, args.output)):
        parser.error("config, plan, trajectory, and output are required")
    if args.prompt_length != 737 or args.ns_steps != 5:
        raise ValueError("the frozen H30a oracle requires length 737 and NS5")
    expected_cg = 1 if args.preflight else 20
    if args.cg_iterations != expected_cg:
        raise ValueError(f"this H30a mode requires {expected_cg} CG iterations")
    accounting = two_step_latent_accounting(args.prompt_length, 768)
    if int(accounting["total_scalars"]) != 566_064 or float(accounting["deployable_scalar_fraction"]) > 0.01:
        raise ValueError("H30a accounting mismatch")

    assert args.output is not None and args.config is not None
    assert args.plan is not None and args.trajectory_dir is not None
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    config = json.loads(args.config.read_text())
    plan = json.loads(args.plan.read_text())
    if int(plan["frozen_decoder"]["state_scalars"]) != 566_064:
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
    function, loss_function, initial_weights, function_manifest = make_model_two_step_program(
        model,
        parameters=FROZEN_PARAMETERS,
        targets=targets,
        ns_steps=args.ns_steps,
    )
    registered_norms_cpu, norm_identities = dense_step_norms(
        args.trajectory_dir, FROZEN_PARAMETERS
    )
    if norm_identities != target_manifest["trajectory_identities"]:
        raise ValueError("trajectory identity changed during norm acquisition")
    registered_norms = registered_norms_cpu.to(args.device)
    anchor_amplitudes, calibration_manifest = calibrate_two_step_amplitudes(
        initial_weights,
        loss_function,
        prompt,
        registered_norms,
        ns_steps=args.ns_steps,
    )
    primals = (prompt, anchor_amplitudes)
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
        "schema_version": "nanogpt_mlp_synthetic_muon_program_twostep_joint_pc1_v1",
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
            "This is a local tangent ceiling at one deterministic prompt and two norm-matched state transitions.",
            "A pass authorizes one frozen remaining-PC/common/innovation/late audit, never CE.",
        ],
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"metadata": str(metadata_path), "classification": metadata["classification"], "row": row}, sort_keys=True))


if __name__ == "__main__":
    main()
