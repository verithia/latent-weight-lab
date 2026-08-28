#!/usr/bin/env python3
"""Joint six-matrix PC1 gate for the H29 synthetic Muon program."""
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
    latent_accounting,
    load_trajectory_parameter,
    make_prompt,
    math_sdpa_context,
    project_direction,
    projection_metrics,
    sha256,
)
from examples.nanogpt.muon import zeropower_via_newtonschulz5


FROZEN_PARAMETERS = tuple(
    f"transformer.h.{layer}.mlp.{suffix}.weight"
    for layer in (0, 6, 11)
    for suffix in ("c_fc", "c_proj")
)


def joint_leading_pc(
    trajectory_dir: Path,
    *,
    parameters: tuple[str, ...],
    device: str,
) -> tuple[torch.Tensor, float, dict[str, Any], dict[str, torch.Tensor]]:
    """Compute a concatenated PC via a streamed sum of temporal Gram matrices."""
    gram: torch.Tensor | None = None
    identities: dict[str, str] = {}
    w0_references: dict[str, torch.Tensor] = {}
    state_count: int | None = None
    for parameter in parameters:
        states_cpu, identity = load_trajectory_parameter(trajectory_dir, parameter)
        state_count = states_cpu.shape[0] if state_count is None else state_count
        if states_cpu.shape[0] != state_count:
            raise ValueError("joint parameter state counts differ")
        w0_references[parameter] = states_cpu[0].clone()
        states = states_cpu.to(device, dtype=torch.float32)
        centered = states - states.mean(dim=0, keepdim=True)
        flat = centered.flatten(1)
        contribution = flat @ flat.T
        gram = contribution if gram is None else gram + contribution
        identities[parameter] = identity
        del states_cpu, states, centered, flat, contribution
        torch.cuda.empty_cache()
    assert gram is not None and state_count is not None
    gram = (gram + gram.T) * 0.5
    eigenvalues, vectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0.0)
    vectors = vectors[:, order]
    leading = vectors[:, 0]
    scale = eigenvalues[0].sqrt().clamp_min(1e-20)
    parts: list[torch.Tensor] = []
    for parameter in parameters:
        states_cpu, identity = load_trajectory_parameter(trajectory_dir, parameter)
        if identity != identities[parameter]:
            raise ValueError("trajectory identity changed across PC passes")
        states = states_cpu.to(device, dtype=torch.float32)
        centered = states - states.mean(dim=0, keepdim=True)
        part = (leading @ centered.flatten(1)) / scale
        parts.append(part.contiguous())
        del states_cpu, states, centered
        torch.cuda.empty_cache()
    target = torch.cat(parts)
    total_energy = float(eigenvalues.sum().clamp_min(1e-30))
    fraction = float(eigenvalues[0]) / total_energy
    return target, fraction, {
        "parameters": list(parameters),
        "state_count": state_count,
        "leading_eigenvalue": float(eigenvalues[0]),
        "total_path_energy": total_energy,
        "trajectory_identities": identities,
        "part_scalar_counts": [int(part.numel()) for part in parts],
        "target_norm": float(target.double().norm()),
    }, w0_references


def make_joint_program_function(
    model: torch.nn.Module,
    *,
    parameters: tuple[str, ...],
    targets: torch.Tensor,
    ns_steps: int,
) -> tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], dict[str, Any]]:
    model_parameters = {name: value.detach() for name, value in model.named_parameters()}
    buffers = {name: value.detach() for name, value in model.named_buffers()}
    missing = [name for name in parameters if name not in model_parameters]
    if missing:
        raise ValueError(f"missing selected parameters: {missing}")
    selected_weights = tuple(model_parameters[name] for name in parameters)
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

    gradient_function = torch.func.grad(task_loss, argnums=0)

    def program(prompt: torch.Tensor, amplitudes: torch.Tensor) -> torch.Tensor:
        gradients = gradient_function(selected_weights, prompt)
        if amplitudes.shape != (len(parameters),):
            raise ValueError("joint amplitude shape mismatch")
        outputs = [
            amplitudes[index]
            * zeropower_via_newtonschulz5(gradient, steps=ns_steps).flatten()
            for index, gradient in enumerate(gradients)
        ]
        return torch.cat(outputs)

    return program, {
        "parameters": list(parameters),
        "selected_weight_shapes": {
            name: list(weight.shape) for name, weight in zip(parameters, selected_weights, strict=True)
        },
        "output_scalars": sum(weight.numel() for weight in selected_weights),
    }


def self_test(device: str) -> dict[str, Any]:
    torch.manual_seed(20260903)
    weights = (torch.randn(7, 5, device=device), torch.randn(4, 7, device=device))
    prompt = torch.randn(1, 6, 5, device=device)
    target = torch.randn(1, 6, 4, device=device)

    def loss_fn(active: tuple[torch.Tensor, ...], values: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.gelu(values @ active[0].T)
        prediction = hidden @ active[1].T
        return 0.5 * (prediction - target).square().mean()

    gradient_fn = torch.func.grad(loss_fn, argnums=0)

    def function(values: torch.Tensor, amplitudes: torch.Tensor) -> torch.Tensor:
        gradients = gradient_fn(weights, values)
        return torch.cat(
            [
                amplitudes[index]
                * zeropower_via_newtonschulz5(gradient, steps=5).flatten()
                for index, gradient in enumerate(gradients)
            ]
        )

    primals = (prompt, torch.ones(2, device=device))
    direction = (torch.randn_like(prompt), torch.randn(2, device=device))
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
    if any(value is None for value in (args.config, args.trajectory_dir, args.output)):
        parser.error("config, trajectory, and output are required")
    if args.prompt_length != 737 or args.ns_steps != 5:
        raise ValueError("the frozen H29c oracle requires length 737 and NS5")
    expected_cg = 1 if args.preflight else 20
    if args.cg_iterations != expected_cg:
        raise ValueError(f"this H29c mode requires {expected_cg} CG iterations")
    accounting = latent_accounting(args.prompt_length, 768)
    if int(accounting["total_scalars"]) != 566_040:
        raise ValueError("H29c accounting mismatch")

    output = args.output
    assert output is not None and args.config is not None and args.trajectory_dir is not None
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    config = json.loads(args.config.read_text())
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = build_dense_model(config, args.device)
    prompt, targets, prompt_manifest = make_prompt(
        model,
        config,
        prompt_length=args.prompt_length,
        device=args.device,
    )
    torch.cuda.reset_peak_memory_stats()
    joint_target, fraction, target_manifest, w0_references = joint_leading_pc(
        args.trajectory_dir,
        parameters=FROZEN_PARAMETERS,
        device=args.device,
    )
    w0_matches = {
        parameter: initialization_match(
            dict(model.named_parameters())[parameter],
            w0_references[parameter],
        )
        for parameter in FROZEN_PARAMETERS
    }
    if not all(bool(record["accepted"]) for record in w0_matches.values()):
        raise ValueError(f"model/trajectory W0 mismatch: {w0_matches}")
    function, function_manifest = make_joint_program_function(
        model,
        parameters=FROZEN_PARAMETERS,
        targets=targets,
        ns_steps=args.ns_steps,
    )
    primals = (prompt, torch.ones(len(FROZEN_PARAMETERS), device=args.device))
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
    retained = best_possible >= 0.20
    row = {
        "path": "joint_trajectory_centered",
        "leading_pc_energy_fraction": fraction,
        "best_possible_total_capture": best_possible,
        "retains_twenty_percent_possibility": retained,
        "solve_seconds": time.time() - solve_started,
        **metrics,
        **diagnostics,
    }
    torch.cuda.synchronize()
    accounting_path = output / "accounting.json"
    accounting_path.write_text(json.dumps(accounting, indent=2, sort_keys=True) + "\n")
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_mlp_synthetic_muon_program_joint_pc1_v1",
        "classification": "PREFLIGHT" if args.preflight else ("RETAINED" if retained else "REJECTED"),
        "preflight": args.preflight,
        "retained": retained,
        "accounting": accounting,
        "prompt_manifest": prompt_manifest,
        "target_manifest": target_manifest,
        "w0_storage_matches": w0_matches,
        "function_manifest": function_manifest,
        "row": row,
        "execution": {
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "entrypoint": str(script),
            "entrypoint_sha256": sha256(script),
            "config": str(args.config),
            "config_sha256": sha256(args.config),
            "command": [str(script), *sys.argv[1:]],
            "runtime_seconds": time.time() - started,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "device": args.device,
        },
        "outputs": {
            "accounting": {"path": str(accounting_path), "sha256": sha256(accounting_path)}
        },
        "limitations": [
            "This is a local tangent ceiling at one deterministic soft-token anchor.",
            "A pass authorizes a joint remaining-PC/common/innovation/late audit, never CE.",
        ],
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"metadata": str(metadata_path), "classification": metadata["classification"], "row": row}, sort_keys=True))


if __name__ == "__main__":
    main()
