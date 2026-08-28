#!/usr/bin/env python3
"""Joint-PC1 gate for H29i dual-prompt parallel synthetic gradients."""
from __future__ import annotations

import argparse
import hashlib
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
from examples.nanogpt.muon import zeropower_via_newtonschulz5


BRANCHES = 2
BRANCH_LENGTH = 368
DEPLOYED_MLP_MATRICES = 24


def dual_prompt_accounting(width: int) -> dict[str, int | float]:
    prompt_scalars = BRANCHES * BRANCH_LENGTH * width
    amplitude_scalars = BRANCHES * DEPLOYED_MLP_MATRICES
    total_scalars = prompt_scalars + amplitude_scalars
    return {
        "parallel_branches": BRANCHES,
        "branch_prompt_length": BRANCH_LENGTH,
        "prompt_width": width,
        "prompt_scalars": prompt_scalars,
        "amplitude_scalars": amplitude_scalars,
        "total_scalars": total_scalars,
        "dense_mlp_denominator_scalars": DENSE_MLP_SCALARS,
        "deployable_scalar_fraction": total_scalars / DENSE_MLP_SCALARS,
        "deployable_checkpoint_fp16_bytes": 2 * total_scalars,
        "fp32_coordinate_master_bytes_during_fit": 4 * total_scalars,
        "fp32_coordinate_gradient_bytes_during_fit": 4 * total_scalars,
        "adam_fp32_moment_bytes_during_fit": 8 * total_scalars,
    }


def split_registered_prompt(
    prompt: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor], dict[str, Any]]:
    if prompt.ndim != 3 or targets.ndim != 2 or prompt.shape[1] != 737 or targets.shape[1] != 737:
        raise ValueError("dual-prompt source must have length 737")
    slices = ((0, 368), (369, 737))
    prompts = tuple(prompt[:, start:end].contiguous() for start, end in slices)
    target_parts = tuple(targets[:, start:end].contiguous() for start, end in slices)
    manifest = {
        "source_length": 737,
        "source_slices": [list(item) for item in slices],
        "omitted_source_positions": [368],
        "prompt_sha256": [
            hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
            for value in prompts
        ],
        "target_sha256": [
            hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
            for value in target_parts
        ],
        "positions_reset_per_branch": True,
    }
    return prompts, target_parts, manifest


def make_generic_parallel_program(
    initial_weights: tuple[torch.Tensor, ...],
    loss_functions: tuple[
        Callable[[tuple[torch.Tensor, ...], torch.Tensor], torch.Tensor],
        Callable[[tuple[torch.Tensor, ...], torch.Tensor], torch.Tensor],
    ],
    *,
    ns_steps: int,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    gradient_functions = tuple(torch.func.grad(loss, argnums=0) for loss in loss_functions)

    def program(
        prompt1: torch.Tensor,
        prompt2: torch.Tensor,
        amplitudes: torch.Tensor,
    ) -> torch.Tensor:
        expected = (BRANCHES, len(initial_weights))
        if tuple(amplitudes.shape) != expected:
            raise ValueError(f"parallel amplitude shape mismatch: {tuple(amplitudes.shape)} != {expected}")
        prompts = (prompt1, prompt2)
        branch_directions: list[tuple[torch.Tensor, ...]] = []
        for branch in range(BRANCHES):
            gradients = gradient_functions[branch](initial_weights, prompts[branch])
            branch_directions.append(
                tuple(
                    zeropower_via_newtonschulz5(gradient, steps=ns_steps)
                    for gradient in gradients
                )
            )
        return torch.cat(
            [
                sum(
                    amplitudes[branch, index] * branch_directions[branch][index]
                    for branch in range(BRANCHES)
                ).flatten()
                for index in range(len(initial_weights))
            ]
        )

    return program


def make_model_parallel_program(
    model: torch.nn.Module,
    *,
    parameters: tuple[str, ...],
    targets: tuple[torch.Tensor, torch.Tensor],
    ns_steps: int,
) -> tuple[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor], dict[str, Any]]:
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

    def make_loss(branch_targets: torch.Tensor) -> Callable[[tuple[torch.Tensor, ...], torch.Tensor], torch.Tensor]:
        def task_loss(weights: tuple[torch.Tensor, ...], prompt: torch.Tensor) -> torch.Tensor:
            call_parameters = dict(static_parameters)
            call_parameters.update(zip(parameters, weights, strict=True))
            with math_sdpa_context():
                _, loss = functional_call(
                    model,
                    (call_parameters, buffers),
                    (None, branch_targets),
                    {"input_embeddings": prompt},
                    tie_weights=True,
                    strict=False,
                )
            assert loss is not None
            return loss

        return task_loss

    loss_functions = tuple(make_loss(value) for value in targets)
    function = make_generic_parallel_program(
        initial_weights,
        loss_functions,  # type: ignore[arg-type]
        ns_steps=ns_steps,
    )
    manifest = {
        "parameters": list(parameters),
        "selected_weight_shapes": {
            name: list(weight.shape)
            for name, weight in zip(parameters, initial_weights, strict=True)
        },
        "parallel_branches": BRANCHES,
        "same_w0_across_branches": True,
        "output_combination": "sum",
        "output_scalars": sum(weight.numel() for weight in initial_weights),
    }
    return function, manifest


def self_test(device: str) -> dict[str, Any]:
    torch.manual_seed(20260908)
    weights = (torch.randn(7, 5, device=device), torch.randn(4, 7, device=device))
    prompt1 = torch.randn(1, 5, 5, device=device)
    prompt2 = torch.randn(1, 6, 5, device=device)
    target1 = torch.randn(1, 5, 4, device=device)
    target2 = torch.randn(1, 6, 4, device=device)

    def make_loss(target: torch.Tensor) -> Callable[[tuple[torch.Tensor, ...], torch.Tensor], torch.Tensor]:
        def loss_fn(active: tuple[torch.Tensor, ...], values: torch.Tensor) -> torch.Tensor:
            hidden = torch.nn.functional.gelu(values @ active[0].T)
            prediction = hidden @ active[1].T
            return 0.5 * (prediction - target).square().mean()

        return loss_fn

    function = make_generic_parallel_program(
        weights,
        (make_loss(target1), make_loss(target2)),
        ns_steps=5,
    )
    amplitudes = torch.ones(BRANCHES, len(weights), device=device)
    primals = (prompt1, prompt2, amplitudes)
    direction = (
        torch.randn_like(prompt1),
        torch.randn_like(prompt2),
        torch.randn_like(amplitudes),
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
    return {"status": "passed", **metrics, **diagnostics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--trajectory-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
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
    if args.ns_steps != 5:
        raise ValueError("the frozen H29i oracle requires NS5")
    expected_cg = 1 if args.preflight else 20
    if args.cg_iterations != expected_cg:
        raise ValueError(f"this H29i mode requires {expected_cg} CG iterations")
    accounting = dual_prompt_accounting(768)
    if int(accounting["total_scalars"]) != 565_296 or float(accounting["deployable_scalar_fraction"]) > 0.01:
        raise ValueError("H29i accounting mismatch")

    assert args.output is not None and args.config is not None
    assert args.plan is not None and args.trajectory_dir is not None
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    config = json.loads(args.config.read_text())
    plan = json.loads(args.plan.read_text())
    if int(plan["frozen_decoder"]["state_scalars"]) != 565_296:
        raise ValueError("plan/accounting mismatch")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = build_dense_model(config, args.device)
    full_prompt, full_targets, source_prompt_manifest = make_prompt(
        model, config, prompt_length=737, device=args.device
    )
    prompts, targets, split_manifest = split_registered_prompt(full_prompt, full_targets)
    del full_prompt, full_targets
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
    function, function_manifest = make_model_parallel_program(
        model,
        parameters=FROZEN_PARAMETERS,
        targets=targets,
        ns_steps=args.ns_steps,
    )
    amplitudes = torch.ones(BRANCHES, len(FROZEN_PARAMETERS), device=args.device)
    primals = (*prompts, amplitudes)
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
        "schema_version": "nanogpt_mlp_dual_prompt_parallel_joint_pc1_v1",
        "classification": "PREFLIGHT" if args.preflight else ("RETAINED" if retained else "REJECTED"),
        "preflight": args.preflight,
        "retained": retained,
        "plan": plan,
        "accounting": accounting,
        "source_prompt_manifest": source_prompt_manifest,
        "split_manifest": split_manifest,
        "target_manifest": target_manifest,
        "w0_storage_matches": w0_matches,
        "function_manifest": function_manifest,
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
            "This is a local tangent ceiling at two deterministic disjoint prompt slices and unit amplitude anchors.",
            "A pass authorizes one frozen remaining-PC/common/innovation/late audit, never CE.",
        ],
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"metadata": str(metadata_path), "classification": metadata["classification"], "row": row}, sort_keys=True))


if __name__ == "__main__":
    main()
