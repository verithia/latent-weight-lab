#!/usr/bin/env python3
"""Joint-PC1 gate for H30b exact two-step Muon momentum replay."""
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
    MACROSTEPS,
    dense_step_norms,
    make_generic_two_step_program,
    two_step_latent_accounting,
)
from examples.nanogpt.muon import zeropower_via_newtonschulz5


FROZEN_MOMENTUM = 0.95


def make_generic_momentum_program(
    initial_weights: tuple[torch.Tensor, ...],
    loss_function: Callable[[tuple[torch.Tensor, ...], torch.Tensor], torch.Tensor],
    *,
    ns_steps: int,
    momentum: float,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if not 0.0 <= momentum < 1.0:
        raise ValueError("momentum must be in [0, 1)")
    gradient_function = torch.func.grad(loss_function, argnums=0)

    def program(prompt: torch.Tensor, amplitudes: torch.Tensor) -> torch.Tensor:
        expected = (MACROSTEPS, len(initial_weights))
        if tuple(amplitudes.shape) != expected:
            raise ValueError(f"momentum amplitude shape mismatch: {tuple(amplitudes.shape)} != {expected}")
        gradients1 = gradient_function(initial_weights, prompt)
        buffers1 = gradients1
        polar_inputs1 = tuple(
            gradient + momentum * buffer
            for gradient, buffer in zip(gradients1, buffers1, strict=True)
        )
        directions1 = tuple(
            zeropower_via_newtonschulz5(value, steps=ns_steps)
            for value in polar_inputs1
        )
        weights1 = tuple(
            weight - amplitudes[0, index] * direction
            for index, (weight, direction) in enumerate(
                zip(initial_weights, directions1, strict=True)
            )
        )
        gradients2 = gradient_function(weights1, prompt)
        buffers2 = tuple(
            momentum * buffer + gradient
            for buffer, gradient in zip(buffers1, gradients2, strict=True)
        )
        polar_inputs2 = tuple(
            gradient + momentum * buffer
            for gradient, buffer in zip(gradients2, buffers2, strict=True)
        )
        directions2 = tuple(
            zeropower_via_newtonschulz5(value, steps=ns_steps)
            for value in polar_inputs2
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


def calibrate_momentum_amplitudes(
    initial_weights: tuple[torch.Tensor, ...],
    loss_function: Callable[[tuple[torch.Tensor, ...], torch.Tensor], torch.Tensor],
    prompt: torch.Tensor,
    target_step_norms: torch.Tensor,
    *,
    ns_steps: int,
    momentum: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    expected = (MACROSTEPS, len(initial_weights))
    if tuple(target_step_norms.shape) != expected:
        raise ValueError("target step norm shape mismatch")
    if not torch.isfinite(target_step_norms).all() or not (target_step_norms > 0).all():
        raise ValueError("target step norms must be finite and positive")
    gradient_function = torch.func.grad(loss_function, argnums=0)
    gradients1 = gradient_function(initial_weights, prompt)
    buffers1 = gradients1
    polar_inputs1 = tuple(
        gradient + momentum * buffer
        for gradient, buffer in zip(gradients1, buffers1, strict=True)
    )
    directions1 = tuple(
        zeropower_via_newtonschulz5(value, steps=ns_steps).detach()
        for value in polar_inputs1
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
    buffers2 = tuple(
        momentum * buffer.detach() + gradient
        for buffer, gradient in zip(buffers1, gradients2, strict=True)
    )
    polar_inputs2 = tuple(
        gradient + momentum * buffer
        for gradient, buffer in zip(gradients2, buffers2, strict=True)
    )
    directions2 = tuple(
        zeropower_via_newtonschulz5(value, steps=ns_steps).detach()
        for value in polar_inputs2
    )
    norms2 = torch.stack([direction.float().norm() for direction in directions2])
    amplitudes2 = target_step_norms[1] / norms2
    amplitudes = torch.stack((amplitudes1, amplitudes2)).detach()
    generated = torch.stack((amplitudes1 * norms1, amplitudes2 * norms2))
    relative_error = (generated - target_step_norms).abs() / target_step_norms
    manifest = {
        "momentum": momentum,
        "target_step_norms": target_step_norms.detach().cpu().tolist(),
        "raw_direction_norms": torch.stack((norms1, norms2)).detach().cpu().tolist(),
        "anchor_amplitudes": amplitudes.detach().cpu().tolist(),
        "maximum_norm_match_relative_error": float(relative_error.max()),
        "buffer_is_regenerated": True,
        "stores_dense_direction_or_buffer": False,
    }
    return amplitudes, manifest


def make_model_momentum_program(
    model: torch.nn.Module,
    *,
    parameters: tuple[str, ...],
    targets: torch.Tensor,
    ns_steps: int,
    momentum: float,
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

    function = make_generic_momentum_program(
        initial_weights,
        task_loss,
        ns_steps=ns_steps,
        momentum=momentum,
    )
    manifest = {
        "parameters": list(parameters),
        "selected_weight_shapes": {
            name: list(weight.shape)
            for name, weight in zip(parameters, initial_weights, strict=True)
        },
        "macrosteps": MACROSTEPS,
        "momentum": momentum,
        "descent_sign": -1,
        "same_prompt_and_targets_each_step": True,
        "buffers_regenerated_and_discarded": True,
        "output_scalars": sum(weight.numel() for weight in initial_weights),
    }
    return function, task_loss, initial_weights, manifest


def self_test(device: str) -> dict[str, Any]:
    torch.manual_seed(20260907)
    weights = (torch.randn(7, 5, device=device), torch.randn(4, 7, device=device))
    prompt = torch.randn(1, 6, 5, device=device)
    target = torch.randn(1, 6, 4, device=device)

    def loss_fn(active: tuple[torch.Tensor, ...], values: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.gelu(values @ active[0].T)
        prediction = hidden @ active[1].T
        return 0.5 * (prediction - target).square().mean()

    target_norms = torch.tensor([[0.08, 0.06], [0.05, 0.04]], device=device)
    amplitudes, calibration = calibrate_momentum_amplitudes(
        weights,
        loss_fn,
        prompt,
        target_norms,
        ns_steps=5,
        momentum=FROZEN_MOMENTUM,
    )
    function = make_generic_momentum_program(
        weights, loss_fn, ns_steps=5, momentum=FROZEN_MOMENTUM
    )
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
    parser.add_argument("--momentum", type=float, default=FROZEN_MOMENTUM)
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
    if args.prompt_length != 737 or args.ns_steps != 5 or args.momentum != FROZEN_MOMENTUM:
        raise ValueError("the frozen H30b oracle requires length 737, beta=0.95, and NS5")
    expected_cg = 1 if args.preflight else 20
    if args.cg_iterations != expected_cg:
        raise ValueError(f"this H30b mode requires {expected_cg} CG iterations")
    accounting = two_step_latent_accounting(args.prompt_length, 768)
    if int(accounting["total_scalars"]) != 566_064 or float(accounting["deployable_scalar_fraction"]) > 0.01:
        raise ValueError("H30b accounting mismatch")

    assert args.output is not None and args.config is not None
    assert args.plan is not None and args.trajectory_dir is not None
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    config = json.loads(args.config.read_text())
    plan = json.loads(args.plan.read_text())
    if float(plan["frozen_decoder"]["muon_momentum"]) != FROZEN_MOMENTUM:
        raise ValueError("plan/momentum mismatch")
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
    function, loss_function, initial_weights, function_manifest = make_model_momentum_program(
        model,
        parameters=FROZEN_PARAMETERS,
        targets=targets,
        ns_steps=args.ns_steps,
        momentum=args.momentum,
    )
    registered_norms_cpu, norm_identities = dense_step_norms(
        args.trajectory_dir, FROZEN_PARAMETERS
    )
    if norm_identities != target_manifest["trajectory_identities"]:
        raise ValueError("trajectory identity changed during norm acquisition")
    registered_norms = registered_norms_cpu.to(args.device)
    anchor_amplitudes, calibration_manifest = calibrate_momentum_amplitudes(
        initial_weights,
        loss_function,
        prompt,
        registered_norms,
        ns_steps=args.ns_steps,
        momentum=args.momentum,
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
        "schema_version": "nanogpt_mlp_synthetic_muon_momentum_twostep_joint_pc1_v1",
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
            "This is a local tangent ceiling at one deterministic prompt and exact beta=0.95 two-step recurrence.",
            "A pass authorizes one frozen remaining-PC/common/innovation/late audit, never CE.",
        ],
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"metadata": str(metadata_path), "classification": metadata["classification"], "row": row}, sort_keys=True))


if __name__ == "__main__":
    main()
