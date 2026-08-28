#!/usr/bin/env python3
"""H30e span gate for a 64-step basis-free synthetic Muon orbit."""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

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
from examples.nanogpt.analyze_mlp_synthetic_muon_program_twostep_joint import (
    dense_step_norms,
)
from examples.nanogpt.analyze_mlp_virtual_lookahead_joint import (
    make_model_lookahead_program,
)
from examples.nanogpt.muon import zeropower_via_newtonschulz5


FROZEN_STEPS = 64
PREFLIGHT_STEPS = 4
FROZEN_MOMENTUM = 0.95
PROMPT_LENGTH = 736
PROMPT_WIDTH = 768
DEPLOYED_MLP_MATRICES = 24
GRAM_RELATIVE_THRESHOLD = 1e-6
CUMULATIVE_STEPS = (1, 2, 4, 8, 16, 32, 64)


def orbit_accounting() -> dict[str, int | float]:
    prompt_scalars = PROMPT_LENGTH * PROMPT_WIDTH
    total_scalars = prompt_scalars + DEPLOYED_MLP_MATRICES + FROZEN_STEPS
    return {
        "prompt_length": PROMPT_LENGTH,
        "prompt_width": PROMPT_WIDTH,
        "prompt_scalars": prompt_scalars,
        "matrix_base_scale_scalars": DEPLOYED_MLP_MATRICES,
        "shared_schedule_ratio_scalars": FROZEN_STEPS,
        "total_state_scalars": total_scalars,
        "dense_mlp_denominator_scalars": DENSE_MLP_SCALARS,
        "state_fraction": total_scalars / DENSE_MLP_SCALARS,
        "fp16_checkpoint_bytes": 2 * total_scalars,
        "persistent_dense_basis_scalars": 0,
        "persistent_optimizer_state_scalars": 0,
    }


def schedule_from_registered_norms(
    registered_norms: torch.Tensor,
    *,
    steps: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if registered_norms.ndim != 2 or registered_norms.shape[0] < steps:
        raise ValueError("insufficient registered step norms")
    selected = registered_norms[:steps].double()
    if not torch.isfinite(selected).all() or not (selected > 0).all():
        raise ValueError("registered step norms must be finite and positive")
    ratios_by_node = selected / selected[0]
    ratios = ratios_by_node.median(dim=1).values
    if not torch.allclose(ratios[0], torch.tensor(1.0, dtype=ratios.dtype)):
        raise ValueError("step-zero schedule ratio must equal one")
    return ratios.float(), {
        "steps": steps,
        "schedule_ratios": ratios.tolist(),
        "minimum_ratio": float(ratios.min()),
        "maximum_ratio": float(ratios.max()),
        "node_ratio_minimum": float(ratios_by_node.min()),
        "node_ratio_maximum": float(ratios_by_node.max()),
        "rule": "median across nodes of registered step norm divided by node step-zero norm",
    }


def first_step_scales(
    initial_weights: tuple[torch.Tensor, ...],
    loss_function: Callable[[tuple[torch.Tensor, ...], torch.Tensor], torch.Tensor],
    prompt: torch.Tensor,
    target_norms: torch.Tensor,
    *,
    ns_steps: int,
    momentum: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    gradient_function = torch.func.grad(loss_function, argnums=0)
    gradients = gradient_function(initial_weights, prompt)
    directions = tuple(
        zeropower_via_newtonschulz5((1.0 + momentum) * gradient, steps=ns_steps)
        for gradient in gradients
    )
    direction_norms = torch.stack([direction.detach().float().norm() for direction in directions])
    scales = target_norms.to(direction_norms) / direction_norms
    relative_error = (scales * direction_norms - target_norms.to(direction_norms)).abs() / target_norms.to(direction_norms)
    return scales.detach(), {
        "target_first_step_norms": target_norms.detach().cpu().tolist(),
        "raw_first_direction_norms": direction_norms.cpu().tolist(),
        "base_scales": scales.detach().cpu().tolist(),
        "maximum_norm_match_relative_error": float(relative_error.max()),
    }


def generate_orbit_atoms(
    initial_weights: tuple[torch.Tensor, ...],
    loss_function: Callable[[tuple[torch.Tensor, ...], torch.Tensor], torch.Tensor],
    prompt: torch.Tensor,
    base_scales: torch.Tensor,
    schedule_ratios: torch.Tensor,
    *,
    ns_steps: int,
    momentum: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    steps = int(schedule_ratios.numel())
    if tuple(base_scales.shape) != (len(initial_weights),):
        raise ValueError("base scale shape mismatch")
    total_output_scalars = sum(weight.numel() for weight in initial_weights)
    atom_rows = torch.empty(
        (steps, total_output_scalars),
        dtype=torch.float32,
        device=prompt.device,
    )
    active_weights = tuple(weight.detach() for weight in initial_weights)
    momenta = tuple(torch.zeros_like(weight) for weight in initial_weights)
    gradient_and_value = torch.func.grad_and_value(loss_function, argnums=0)
    step_rows = []
    for step in range(steps):
        gradients, loss = gradient_and_value(active_weights, prompt)
        momenta = tuple(
            momentum * buffer + gradient
            for buffer, gradient in zip(momenta, gradients, strict=True)
        )
        polar_inputs = tuple(
            gradient + momentum * buffer
            for gradient, buffer in zip(gradients, momenta, strict=True)
        )
        directions = tuple(
            zeropower_via_newtonschulz5(value, steps=ns_steps)
            for value in polar_inputs
        )
        updates = tuple(
            base_scales[index] * schedule_ratios[step] * direction
            for index, direction in enumerate(directions)
        )
        joint_update = torch.cat([update.detach().float().flatten() for update in updates])
        joint_norm = joint_update.norm()
        if not torch.isfinite(joint_norm) or float(joint_norm) == 0.0:
            raise ValueError(f"invalid joint update at step {step}")
        atom_rows[step].copy_(joint_update / joint_norm)
        active_weights = tuple(
            (weight - update).detach()
            for weight, update in zip(active_weights, updates, strict=True)
        )
        momenta = tuple(buffer.detach() for buffer in momenta)
        step_rows.append(
            {
                "step": step,
                "loss": float(loss.detach()),
                "schedule_ratio": float(schedule_ratios[step]),
                "joint_update_norm": float(joint_norm),
            }
        )
    return atom_rows, {
        "steps": step_rows,
        "initial_loss": step_rows[0]["loss"],
        "terminal_loss": step_rows[-1]["loss"],
        "minimum_loss": min(row["loss"] for row in step_rows),
        "maximum_loss": max(row["loss"] for row in step_rows),
    }


def gram_projection_capture(
    gram: torch.Tensor,
    target_products: torch.Tensor,
    *,
    relative_threshold: float,
) -> tuple[float, int, float, torch.Tensor]:
    eigenvalues, eigenvectors = torch.linalg.eigh(gram.double())
    maximum = eigenvalues[-1].clamp_min(torch.finfo(eigenvalues.dtype).eps)
    keep = eigenvalues > relative_threshold * maximum
    coordinates = eigenvectors.T @ target_products.double()
    capture = (coordinates[keep].square() / eigenvalues[keep]).sum()
    stable_rank = eigenvalues.clamp_min(0).sum() / maximum
    return float(capture.clamp(0.0, 1.0)), int(keep.sum()), float(stable_rank), eigenvalues


def orbit_span_metrics(
    atom_rows: torch.Tensor,
    target: torch.Tensor,
    *,
    relative_threshold: float = GRAM_RELATIVE_THRESHOLD,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    if atom_rows.ndim != 2 or target.ndim != 1 or atom_rows.shape[1] != target.numel():
        raise ValueError("orbit span shape mismatch")
    target_unit = target.detach().float().flatten()
    target_unit = target_unit / target_unit.norm()
    gram = atom_rows @ atom_rows.T
    products = atom_rows @ target_unit
    cumulative = []
    for count in CUMULATIVE_STEPS:
        if count > atom_rows.shape[0]:
            continue
        capture, numerical_rank, stable_rank, _ = gram_projection_capture(
            gram[:count, :count],
            products[:count],
            relative_threshold=relative_threshold,
        )
        cumulative.append(
            {
                "steps": count,
                "pc1_span_capture": capture,
                "gram_numerical_rank": numerical_rank,
                "gram_stable_rank": stable_rank,
            }
        )
    final_capture, numerical_rank, stable_rank, eigenvalues = gram_projection_capture(
        gram,
        products,
        relative_threshold=relative_threshold,
    )
    consecutive = torch.diagonal(gram, offset=1)
    half = atom_rows.shape[0] // 2
    first_gram = gram[:half, :half]
    late_innovations = []
    for index in range(half, atom_rows.shape[0]):
        cross = gram[:half, index]
        capture, _, _, _ = gram_projection_capture(
            first_gram,
            cross,
            relative_threshold=relative_threshold,
        )
        late_innovations.append(max(0.0, 1.0 - capture))
    metrics = {
        "relative_eigenvalue_threshold": relative_threshold,
        "pc1_span_capture": final_capture,
        "gram_numerical_rank": numerical_rank,
        "gram_stable_rank": stable_rank,
        "gram_eigenvalues": eigenvalues.cpu().tolist(),
        "cumulative": cumulative,
        "consecutive_cosine_minimum": float(consecutive.min()) if consecutive.numel() else None,
        "consecutive_cosine_median": (
            statistics.median(consecutive.cpu().tolist()) if consecutive.numel() else None
        ),
        "consecutive_cosine_maximum": float(consecutive.max()) if consecutive.numel() else None,
        "late_half_innovation_minimum": min(late_innovations) if late_innovations else None,
        "late_half_innovation_median": statistics.median(late_innovations) if late_innovations else None,
        "late_half_innovation_maximum": max(late_innovations) if late_innovations else None,
    }
    return metrics, gram, products


def self_test(device: str) -> dict[str, Any]:
    torch.manual_seed(53)
    target = torch.randn(97, device=device)
    target = target / target.norm()
    orthogonal_noise = torch.randn(7, 97, device=device)
    orthogonal_noise -= (orthogonal_noise @ target).unsqueeze(1) * target
    orthogonal_noise = orthogonal_noise / orthogonal_noise.norm(dim=1, keepdim=True)
    atoms = torch.cat((target.unsqueeze(0), orthogonal_noise), dim=0)
    metrics, _, _ = orbit_span_metrics(atoms, target)
    if metrics["pc1_span_capture"] < 0.9999 or metrics["gram_numerical_rank"] != 8:
        raise AssertionError(metrics)
    accounting = orbit_accounting()
    if accounting["total_state_scalars"] != 565_336:
        raise AssertionError(accounting)
    return {"status": "passed", "metrics": metrics, "accounting": accounting}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--trajectory-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=FROZEN_STEPS)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(args.device), sort_keys=True))
        return
    if any(value is None for value in (args.config, args.plan, args.trajectory_dir, args.output)):
        parser.error("config, plan, trajectory, and output are required")
    expected_steps = PREFLIGHT_STEPS if args.preflight else FROZEN_STEPS
    if args.steps != expected_steps:
        raise ValueError(f"this mode requires exactly {expected_steps} steps")
    assert args.config is not None and args.plan is not None
    assert args.trajectory_dir is not None and args.output is not None

    accounting = orbit_accounting()
    if accounting["total_state_scalars"] != 565_336 or accounting["state_fraction"] > 0.01:
        raise ValueError("H30e accounting mismatch")
    config = json.loads(args.config.read_text())
    plan = json.loads(args.plan.read_text())
    if plan["frozen_decoder"]["total_state_scalars"] != 565_336:
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
        momentum=FROZEN_MOMENTUM,
    )
    registered_norms, trajectory_identities = dense_step_norms(
        args.trajectory_dir,
        FROZEN_PARAMETERS,
    )
    if trajectory_identities != target_manifest["trajectory_identities"]:
        raise ValueError("trajectory identity changed during schedule acquisition")
    full_schedule, schedule_manifest = schedule_from_registered_norms(
        registered_norms,
        steps=FROZEN_STEPS,
    )
    first_norms = registered_norms[0].to(args.device)
    base_scales, scale_manifest = first_step_scales(
        initial_weights,
        loss_function,
        prompt,
        first_norms,
        ns_steps=5,
        momentum=FROZEN_MOMENTUM,
    )
    atom_rows, orbit_manifest = generate_orbit_atoms(
        initial_weights,
        loss_function,
        prompt,
        base_scales,
        full_schedule[: args.steps].to(args.device),
        ns_steps=5,
        momentum=FROZEN_MOMENTUM,
    )
    span, gram, products = orbit_span_metrics(atom_rows, joint_target)
    capture_gate = span["pc1_span_capture"] >= 0.10
    stable_rank_gate = span["gram_stable_rank"] >= 8.0
    numerical_rank_gate = span["gram_numerical_rank"] >= 16
    retained = (
        not args.preflight and capture_gate and stable_rank_gate and numerical_rank_gate
    )
    gates = {
        "minimum_pc1_span_capture": 0.10,
        "minimum_stable_rank": 8.0,
        "minimum_numerical_rank": 16,
        "capture_gate": capture_gate,
        "stable_rank_gate": stable_rank_gate,
        "numerical_rank_gate": numerical_rank_gate,
        "retained": retained,
    }

    torch.cuda.synchronize()
    accounting_path = output / "accounting.json"
    accounting_path.write_text(json.dumps(accounting, indent=2, sort_keys=True) + "\n")
    gram_path = output / "atom_gram.npy"
    products_path = output / "target_products.npy"
    np.save(gram_path, gram.detach().cpu().numpy())
    np.save(products_path, products.detach().cpu().numpy())
    script = Path(__file__).resolve()
    runtime = time.time() - started
    metadata = {
        "schema_version": "nanogpt_mlp_long_horizon_synthetic_orbit_v1",
        "classification": "PREFLIGHT" if args.preflight else ("RETAINED" if retained else "REJECTED"),
        "preflight": args.preflight,
        "retained": retained,
        "plan": plan,
        "accounting": accounting,
        "prompt_manifest": prompt_manifest,
        "target_manifest": target_manifest,
        "w0_storage_matches": w0_matches,
        "function_manifest": {
            **function_manifest,
            "orbit_steps": args.steps,
            "endpoint_is_decoder_output": True,
            "persistent_generated_atoms": False,
            "persistent_optimizer_buffer": False,
        },
        "scale_manifest": scale_manifest,
        "schedule_manifest": schedule_manifest,
        "orbit_manifest": orbit_manifest,
        "span": {**span, "leading_pc_energy_fraction": leading_fraction},
        "gates": gates,
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
            "runtime_seconds": runtime,
            "projected_64_step_runtime_seconds": runtime * FROZEN_STEPS / args.steps,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "device": args.device,
        },
        "outputs": {
            "accounting": {"path": str(accounting_path), "sha256": sha256(accounting_path)},
            "atom_gram": {"path": str(gram_path), "sha256": sha256(gram_path)},
            "target_products": {"path": str(products_path), "sha256": sha256(products_path)},
        },
        "limitations": [
            "The atom span is an optimistic upper bound on the endpoint prompt Jacobian, not a representation pass.",
            "A pass authorizes one checkpointed endpoint-Jacobian audit, never CE or scale-up.",
        ],
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "metadata": str(metadata_path),
                "classification": metadata["classification"],
                "pc1_span_capture": span["pc1_span_capture"],
                "gram_stable_rank": span["gram_stable_rank"],
                "gram_numerical_rank": span["gram_numerical_rank"],
                "runtime_seconds": runtime,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
