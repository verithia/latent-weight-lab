#!/usr/bin/env python3
"""Causal tangent replay for a compact nonlinear bilateral MLP chart.

This is a zero-language-model-update oracle.  The only evolving objects are
rank-r row/column coordinates of a smooth full-rank matrix decoder.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.func import functional_call, jvp, vjp

from examples.nanogpt.analyze_mlp_highcadence_basis import file_sha256
from examples.nanogpt.analyze_mlp_optimizer_path_rate_distortion import (
    load_probe_learning_rates,
)
from examples.nanogpt.analyze_mlp_optimizer_probe_span import load_probe_inventory
from examples.nanogpt.analyze_mlp_product_fht_tangent_anchor import git_commit, summarize
from examples.nanogpt.analyze_parameter_trajectory import write_csv


class NonlinearBilateralKernel(torch.nn.Module):
    """``scale * (sin(U V^T) - sin(U0 V0^T))`` with compact U/V state."""

    def __init__(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        *,
        output_scale: float,
    ) -> None:
        super().__init__()
        if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
            raise ValueError("left/right must be rank-matched matrices")
        self.left = torch.nn.Parameter(left.detach().clone())
        self.right = torch.nn.Parameter(right.detach().clone())
        self.register_buffer("initial_left", left.detach().clone())
        self.register_buffer("initial_right", right.detach().clone())
        self.output_scale = float(output_scale)

    def forward(self) -> torch.Tensor:
        current = self.left @ self.right.transpose(0, 1)
        initial = self.initial_left @ self.initial_right.transpose(0, 1)
        return self.output_scale * (current.sin() - initial.sin())

    @property
    def coordinate_count(self) -> int:
        return self.left.numel() + self.right.numel()


class MultiatomNonlinearBilateralKernel(torch.nn.Module):
    """A sum of independently gated rank-one sine-kernel atoms."""

    def __init__(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        *,
        output_scale: float,
    ) -> None:
        super().__init__()
        if left.ndim != 3 or right.ndim != 3:
            raise ValueError("multiatom left/right must be three-dimensional")
        if left.shape[0] != right.shape[0] or left.shape[2] != right.shape[2]:
            raise ValueError("multiatom left/right must have matched atom/rank axes")
        self.left = torch.nn.Parameter(left.detach().clone())
        self.right = torch.nn.Parameter(right.detach().clone())
        self.register_buffer("initial_left", left.detach().clone())
        self.register_buffer("initial_right", right.detach().clone())
        self.output_scale = float(output_scale)

    @staticmethod
    def products(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.einsum("hmr,hnr->hmn", left, right)

    def forward(self) -> torch.Tensor:
        current = self.products(self.left, self.right)
        initial = self.products(self.initial_left, self.initial_right)
        return self.output_scale * (current.sin() - initial.sin()).sum(dim=0)

    @property
    def coordinate_count(self) -> int:
        return self.left.numel() + self.right.numel()


def gradient_seeded_factors(
    gradient: torch.Tensor,
    *,
    rank: int,
    product_rms: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if gradient.ndim != 2:
        raise ValueError("gradient must be a matrix")
    if rank <= 0 or rank > min(gradient.shape):
        raise ValueError("rank is outside the matrix dimensions")
    u, s, vh = torch.linalg.svd(gradient.float(), full_matrices=False)
    root = s[:rank].clamp_min(1e-30).sqrt()
    left = u[:, :rank] * root.unsqueeze(0)
    right = vh[:rank].transpose(0, 1) * root.unsqueeze(0)
    product = left @ right.transpose(0, 1)
    factor_scale = math.sqrt(
        float(product_rms) / float(product.square().mean().sqrt().clamp_min(1e-30))
    )
    return left * factor_scale, right * factor_scale


def gradient_seeded_multiatom_factors(
    gradient: torch.Tensor,
    *,
    atoms: int,
    rank_per_atom: int,
    product_rms: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    total_rank = atoms * rank_per_atom
    left, right = gradient_seeded_factors(
        gradient, rank=total_rank, product_rms=1.0
    )
    left_atoms = []
    right_atoms = []
    for index in range(atoms):
        start = index * rank_per_atom
        stop = start + rank_per_atom
        left_value = left[:, start:stop]
        right_value = right[:, start:stop]
        product = left_value @ right_value.transpose(0, 1)
        factor_scale = math.sqrt(
            float(product_rms)
            / float(product.square().mean().sqrt().clamp_min(1e-30))
        )
        left_atoms.append(left_value * factor_scale)
        right_atoms.append(right_value * factor_scale)
    return torch.stack(left_atoms), torch.stack(right_atoms)


def _tuple_dot(
    left: tuple[torch.Tensor, ...], right: tuple[torch.Tensor, ...]
) -> torch.Tensor:
    return sum(
        (a * b).sum() for a, b in zip(left, right, strict=True)
    )


def project_target(
    module: NonlinearBilateralKernel | MultiatomNonlinearBilateralKernel,
    target: torch.Tensor,
    *,
    cg_steps: int,
    damping_ratio: float,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, dict[str, float]]:
    """Return damped natural coordinates and their exact decoder JVP."""
    if cg_steps <= 0 or damping_ratio <= 0.0:
        raise ValueError("cg_steps and damping_ratio must be positive")
    named = dict(module.named_parameters())
    names = sorted(named)
    primals = tuple(named[name].detach() for name in names)

    def materialize(*coordinates: torch.Tensor) -> torch.Tensor:
        replacements = dict(zip(names, coordinates, strict=True))
        return functional_call(module, replacements, (), strict=False)

    _, pullback = vjp(materialize, *primals)
    rhs = tuple(value.detach() for value in pullback(target))
    coordinate_count = sum(value.numel() for value in primals)
    probe = tuple(torch.ones_like(value) for value in primals)
    _, probe_image = jvp(materialize, primals, probe)
    mean_eigenvalue = probe_image.square().sum() / float(coordinate_count)
    damping = float(damping_ratio) * mean_eigenvalue.clamp_min(1e-30)

    def normal(direction: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        _, image = jvp(materialize, primals, direction)
        pulled = pullback(image)
        return tuple(
            value.detach() + damping * coordinate
            for value, coordinate in zip(pulled, direction, strict=True)
        )

    solution = tuple(torch.zeros_like(value) for value in rhs)
    residual = tuple(value.clone() for value in rhs)
    conjugate = tuple(value.clone() for value in residual)
    norm_squared = _tuple_dot(residual, residual).clamp_min(1e-30)
    initial_norm_squared = norm_squared
    completed = 0
    for step in range(cg_steps):
        image = normal(conjugate)
        denominator = _tuple_dot(conjugate, image)
        if not torch.isfinite(denominator) or float(denominator) <= 0.0:
            break
        alpha = norm_squared / denominator
        solution = tuple(
            value + alpha * direction
            for value, direction in zip(solution, conjugate, strict=True)
        )
        residual = tuple(
            value - alpha * image_value
            for value, image_value in zip(residual, image, strict=True)
        )
        next_norm_squared = _tuple_dot(residual, residual)
        completed = step + 1
        if float(next_norm_squared / initial_norm_squared) <= 1e-10:
            norm_squared = next_norm_squared
            break
        beta = next_norm_squared / norm_squared.clamp_min(1e-30)
        conjugate = tuple(
            value + beta * direction
            for value, direction in zip(residual, conjugate, strict=True)
        )
        norm_squared = next_norm_squared
    _, action = jvp(materialize, primals, solution)
    target_norm = target.float().norm().clamp_min(1e-30)
    action_norm = action.float().norm().clamp_min(1e-30)
    cosine = (action.float() * target.float()).sum() / (action_norm * target_norm)
    return solution, action.detach(), {
        "action_cosine": float(cosine),
        "action_capture": float(cosine.square()),
        "action_to_target_norm_ratio": float(action_norm / target_norm),
        "cg_steps_completed": completed,
        "cg_relative_residual": float(
            (norm_squared / initial_norm_squared).clamp_min(0).sqrt()
        ),
    }


def apply_normalized_step(
    module: NonlinearBilateralKernel | MultiatomNonlinearBilateralKernel,
    coordinates: tuple[torch.Tensor, ...],
    action: torch.Tensor,
    *,
    norm_reference: torch.Tensor,
    learning_rate: float,
    coordinate_cap: float,
) -> dict[str, float]:
    scale = float(
        norm_reference.float().norm().clamp_min(1e-30)
        / action.float().norm().clamp_min(1e-30)
    )
    directions = tuple(value * scale for value in coordinates)
    uncapped = float(learning_rate) * max(float(value.abs().max()) for value in directions)
    cap_scale = min(1.0, float(coordinate_cap) / max(uncapped, 1e-30))
    with torch.no_grad():
        module.left.add_(directions[0], alpha=float(learning_rate) * cap_scale)
        module.right.add_(directions[1], alpha=float(learning_rate) * cap_scale)
    return {
        "normalization_scale": scale,
        "uncapped_maximum_coordinate_update": uncapped,
        "cap_scale": cap_scale,
        "applied_maximum_coordinate_update": uncapped * cap_scale,
    }


def coordinate_statistics(
    module: NonlinearBilateralKernel | MultiatomNonlinearBilateralKernel,
) -> dict[str, float]:
    current = torch.cat((module.left.detach().flatten(), module.right.detach().flatten())).float()
    initial = torch.cat((module.initial_left.flatten(), module.initial_right.flatten())).float()
    movement = current - initial
    return {
        "coordinate_rms": float(current.square().mean().sqrt()),
        "coordinate_max_abs": float(current.abs().max()),
        "coordinate_movement_rms": float(movement.square().mean().sqrt()),
        "coordinate_movement_max_abs": float(movement.abs().max()),
    }


def split_name(step: int, discovery_stop: int, validation_stop: int) -> str:
    if step < discovery_stop:
        return "discovery"
    if step < validation_stop:
        return "validation"
    return "test"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--targets", default="mlp.c_fc,mlp.c_proj")
    parser.add_argument("--rank", type=int, default=6)
    parser.add_argument("--atoms", type=int, default=1)
    parser.add_argument("--product-rms", type=float, default=0.5)
    parser.add_argument("--output-scale", type=float, default=0.02)
    parser.add_argument("--cg-steps", type=int, default=12)
    parser.add_argument("--damping-ratio", type=float, default=1e-4)
    parser.add_argument("--coordinate-cap", type=float, default=0.02)
    parser.add_argument("--discovery-stop", type=int, default=119)
    parser.add_argument("--validation-stop", type=int, default=179)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    targets = {value for value in args.targets.split(",") if value}
    if targets != {"mlp.c_fc", "mlp.c_proj"}:
        raise ValueError("the frozen oracle requires both MLP matrices")
    paths = sorted(args.probe_dir.glob("step_*.pt"))
    steps, values, input_metadata = load_probe_inventory(paths, layers={args.layer}, targets=targets)
    learning_rates = load_probe_learning_rates(paths, set(values))
    args.output.mkdir(parents=True, exist_ok=True)
    scores: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    updates: list[dict[str, Any]] = []
    final_state: dict[str, Any] = {}
    accounting: dict[str, Any] = {}
    final_delta_ranks: dict[str, int] = {}
    cumulative_recovery: dict[str, Any] = {}

    for parameter in sorted(values):
        gradients = torch.stack(values[parameter]["raw_gradient_descent"]).to(args.device, torch.float32)
        norm_references = torch.stack(values[parameter]["exact_applied_direction"]).to(args.device, torch.float32)
        if args.atoms <= 0 or args.rank % args.atoms:
            raise ValueError("atoms must be positive and divide rank exactly")
        if args.atoms == 1:
            left, right = gradient_seeded_factors(
                gradients[0], rank=args.rank, product_rms=args.product_rms
            )
        else:
            left, right = gradient_seeded_multiatom_factors(
                gradients[0],
                atoms=args.atoms,
                rank_per_atom=args.rank // args.atoms,
                product_rms=args.product_rms,
            )

        def new_module() -> NonlinearBilateralKernel | MultiatomNonlinearBilateralKernel:
            cls = (
                NonlinearBilateralKernel
                if args.atoms == 1
                else MultiatomNonlinearBilateralKernel
            )
            return cls(left, right, output_scale=args.output_scale).to(args.device)

        initial_module = new_module()
        for index, (step, gradient) in enumerate(zip(steps, gradients, strict=True)):
            _, _, metrics = project_target(
                initial_module,
                gradient,
                cg_steps=args.cg_steps,
                damping_ratio=args.damping_ratio,
            )
            scores.append(
                {
                    "parameter": parameter,
                    # ``summarize`` uses the literal identity label for the
                    # frozen control.  Here identity means the immutable
                    # gradient-seeded chart state, before any causal motion.
                    "anchor": "identity",
                    "probe_index": index,
                    "step": step,
                    "split": split_name(step, args.discovery_stop, args.validation_stop),
                    **metrics,
                    **coordinate_statistics(initial_module),
                }
            )
        del initial_module

        module = new_module()
        target_path = torch.zeros_like(gradients[0])
        capped = 0
        update_count = 0
        for index, (step, gradient) in enumerate(zip(steps, gradients, strict=True)):
            coordinates, action, metrics = project_target(
                module,
                gradient,
                cg_steps=args.cg_steps,
                damping_ratio=args.damping_ratio,
            )
            scores.append(
                {
                    "parameter": parameter,
                    "anchor": "rolling_kernel",
                    "probe_index": index,
                    "step": step,
                    "split": split_name(step, args.discovery_stop, args.validation_stop),
                    **metrics,
                    **coordinate_statistics(module),
                }
            )
            if index + 1 == len(steps):
                continue
            interval = steps[index + 1] - step
            learning_rate = learning_rates[parameter][index]
            interval_rows: list[dict[str, float]] = []
            for _ in range(interval):
                coordinates, action, _ = project_target(
                    module,
                    gradient,
                    cg_steps=args.cg_steps,
                    damping_ratio=args.damping_ratio,
                )
                diagnostics = apply_normalized_step(
                    module,
                    coordinates,
                    action,
                    norm_reference=norm_references[index],
                    learning_rate=learning_rate,
                    coordinate_cap=args.coordinate_cap,
                )
                interval_rows.append(diagnostics)
                capped += int(diagnostics["cap_scale"] < 1.0)
                update_count += 1
                normalized_target = gradient * (
                    norm_references[index].norm()
                    / gradient.norm().clamp_min(1e-30)
                )
                target_path.add_(normalized_target, alpha=learning_rate)
            updates.append(
                {
                    "parameter": parameter,
                    "probe_index": index,
                    "step": step,
                    "interval_updates": interval,
                    "learning_rate": learning_rate,
                    "mean_normalization_scale": sum(x["normalization_scale"] for x in interval_rows) / interval,
                    "minimum_cap_scale": min(x["cap_scale"] for x in interval_rows),
                    "maximum_applied_coordinate_update": max(x["applied_maximum_coordinate_update"] for x in interval_rows),
                    **coordinate_statistics(module),
                }
            )
        parameter_rows = [row for row in scores if row["parameter"] == parameter]
        summaries.extend(summarize(parameter_rows, parameter=parameter))
        with torch.no_grad():
            delta = module().float()
            final_delta_ranks[parameter] = int(torch.linalg.matrix_rank(delta))
            target_norm = target_path.norm().clamp_min(1e-30)
            delta_norm = delta.norm().clamp_min(1e-30)
            cosine = (delta * target_path).sum() / (delta_norm * target_norm)
            cumulative_recovery[parameter] = {
                "target_norm": float(target_norm),
                "decoded_delta_norm": float(delta_norm),
                "cosine": float(cosine),
                "energy_recovery": float(1.0 - (delta - target_path).square().sum() / target_path.square().sum().clamp_min(1e-30)),
            }
        final_state[parameter] = {
            "left": module.left.detach().cpu(),
            "right": module.right.detach().cpu(),
            "initial_left": module.initial_left.detach().cpu(),
            "initial_right": module.initial_right.detach().cpu(),
            "update_count": update_count,
            "capped_update_count": capped,
        }
        dense = gradients.shape[1] * gradients.shape[2]
        accounting[parameter] = {
            "dense_scalars": dense,
            "coordinate_scalars": module.coordinate_count,
            "coordinate_fraction": module.coordinate_count / dense,
        }

    scores_path = args.output / "probe_scores.csv"
    summary_path = args.output / "summary.csv"
    updates_path = args.output / "replay_updates.csv"
    state_path = args.output / "final_state.pt"
    write_csv(scores_path, scores)
    write_csv(summary_path, summaries)
    write_csv(updates_path, updates)
    torch.save(final_state, state_path)
    metadata = {
        "schema_version": "nanogpt_mlp_nonlinear_bilateral_kernel_v1",
        "method": "gradient-seeded sine-kernel bilateral chart causal replay",
        "source_commit": git_commit(Path(__file__).resolve().parents[2]),
        "input": input_metadata,
        "layer": args.layer,
        "rank": args.rank,
        "atoms": args.atoms,
        "product_rms": args.product_rms,
        "output_scale": args.output_scale,
        "cg_steps": args.cg_steps,
        "damping_ratio": args.damping_ratio,
        "coordinate_cap": args.coordinate_cap,
        "steps": steps,
        "accounting": accounting,
        "final_delta_ranks": final_delta_ranks,
        "cumulative_recovery": cumulative_recovery,
        "promotion_gate": {
            "validation_and_test_mean_action_capture_each_target": 0.40,
            "test_minimum_action_capture_each_target": 0.20,
            "test_enrichment_over_initial_each_target": 4.0,
            "final_delta_rank_each_target": 768,
        },
        "limitations": [
            "Dense-path gradients are replayed without updating a language model.",
            "Missing steps use zero-order-held gradients and registered learning rates.",
            "The norm reference is the paired dense-Muon applied direction.",
        ],
        "runtime_seconds": time.time() - started,
        "probe_scores_sha256": file_sha256(scores_path),
        "summary_sha256": file_sha256(summary_path),
        "replay_updates_sha256": file_sha256(updates_path),
        "final_state_sha256": file_sha256(state_path),
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
