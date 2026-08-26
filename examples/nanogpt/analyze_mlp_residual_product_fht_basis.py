#!/usr/bin/env python3
"""Fit a basis-free ProductFHT chart to dense MLP residual PCs.

This is a noncausal representation ceiling.  Dense residual PCA supplies only
the fitting/evaluation targets.  Candidate state consists solely of learned
ProductFHT diagonals/output gains and procedural seeds; no dense target basis,
ambient atom, or dense shadow is retained.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_mlp_highcadence_basis import file_sha256
from examples.nanogpt.analyze_mlp_product_fht_tangent_anchor import (
    natural_pullback_action,
)
from examples.nanogpt.analyze_mlp_residual_qtt_basis import (
    residual_temporal_basis,
)
from examples.nanogpt.analyze_parameter_trajectory import (
    PARAMETER_PATTERN,
    load_snapshots,
    parse_int_list,
    write_csv,
)
from latent_weight_lab import ProductFHTLinear


TARGET_SEED_OFFSETS = {"mlp.c_fc": 2, "mlp.c_proj": 3}


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def flatten_coordinates(
    diagonal: torch.Tensor, output: torch.Tensor
) -> torch.Tensor:
    return torch.cat((diagonal.reshape(-1), output.reshape(-1)))


def split_coordinates(
    module: ProductFHTLinear, vector: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    diagonal_count = module.product_log_diagonals.numel()
    return (
        vector[:diagonal_count].reshape_as(module.product_log_diagonals),
        vector[diagonal_count:].reshape_as(module.product_output_log_gain),
    )


def coordinate_vjp(module: ProductFHTLinear, target: torch.Tensor) -> torch.Tensor:
    weight = module._weight_from_factors(
        module.product_log_diagonals,
        module.product_output_log_gain,
    )
    diagonal, output = torch.autograd.grad(
        weight,
        (module.product_log_diagonals, module.product_output_log_gain),
        grad_outputs=target.to(weight.dtype),
        create_graph=False,
        retain_graph=False,
    )
    return flatten_coordinates(diagonal, output).detach()


def coordinate_jvp(module: ProductFHTLinear, vector: torch.Tensor) -> torch.Tensor:
    diagonal, output = split_coordinates(module, vector)
    return module._weight_jvp_from_factors(diagonal, output).detach()


def coordinate_metric(module: ProductFHTLinear) -> torch.Tensor:
    diagonal_metric = (
        module.weight_std
        * module.weight_std
        * module.out_features
        * module.in_features
        / module.padded_features
    )
    diagonal = torch.full_like(
        module.product_log_diagonals,
        max(diagonal_metric, 1e-12),
    )
    with torch.no_grad():
        weight = module._weight_from_factors(
            module.product_log_diagonals,
            module.product_output_log_gain,
        )
        output = weight.float().square().sum(dim=1).clamp_min(1e-12)
    return flatten_coordinates(diagonal, output)


def cg_project(
    module: ProductFHTLinear,
    target: torch.Tensor,
    *,
    maximum_iterations: int,
    relative_tolerance: float,
    damping_ratio: float,
) -> dict[str, Any]:
    """Approximately solve the exact damped Jacobian least-squares problem."""
    target = target.float()
    right = coordinate_vjp(module, target)
    metric = coordinate_metric(module)
    damping = damping_ratio * float(metric.mean())

    def normal(vector: torch.Tensor) -> torch.Tensor:
        return coordinate_vjp(module, coordinate_jvp(module, vector)) + damping * vector

    estimate = torch.zeros_like(right)
    residual = right.clone()
    preconditioned = residual / (metric + damping).clamp_min(1e-12)
    direction = preconditioned.clone()
    rz = torch.dot(residual.double(), preconditioned.double())
    initial_norm = residual.double().norm().clamp_min(1e-30)
    iterations = 0
    for iteration in range(maximum_iterations):
        action = normal(direction)
        denominator = torch.dot(direction.double(), action.double()).clamp_min(1e-30)
        step = rz / denominator
        estimate.add_(direction, alpha=float(step))
        residual.add_(action, alpha=-float(step))
        iterations = iteration + 1
        if float(residual.double().norm() / initial_norm) <= relative_tolerance:
            break
        next_preconditioned = residual / (metric + damping).clamp_min(1e-12)
        next_rz = torch.dot(residual.double(), next_preconditioned.double())
        direction.mul_(float(next_rz / rz)).add_(next_preconditioned)
        preconditioned = next_preconditioned
        rz = next_rz
    projected = coordinate_jvp(module, estimate)
    target_energy = target.double().square().sum().clamp_min(1e-30)
    error_energy = (target - projected).double().square().sum()
    capture = 1.0 - error_energy / target_energy
    return {
        "cg_projection_capture": float(capture),
        "cg_iterations": iterations,
        "cg_final_normal_relative_residual": float(
            residual.double().norm() / initial_norm
        ),
        "projected_to_target_norm_ratio": float(
            projected.double().norm() / target.double().norm().clamp_min(1e-30)
        ),
    }


def deterministic_weighted_mixture(
    basis: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    update: int,
    width: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed + 104729 * update)
    indices = torch.multinomial(
        probabilities.detach().cpu(),
        width,
        replacement=True,
        generator=generator,
    ).tolist()
    signs = (
        torch.randint(0, 2, (width,), generator=generator) * 2 - 1
    ).tolist()
    mixture = torch.zeros_like(basis[0])
    for index, sign in zip(indices, signs, strict=True):
        mixture.add_(basis[index], alpha=float(sign))
    return mixture / mixture.norm().clamp_min(1e-30)


def fit_anchor(
    module: ProductFHTLinear,
    basis: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    updates: int,
    learning_rate: float,
    mixture_width: int,
    bound: float,
    seed: int,
) -> list[dict[str, Any]]:
    optimizer = torch.optim.Adam(
        [module.product_log_diagonals, module.product_output_log_gain],
        lr=learning_rate,
    )
    history: list[dict[str, Any]] = []
    for update in range(updates):
        target = deterministic_weighted_mixture(
            basis,
            probabilities,
            update=update,
            width=mixture_width,
            seed=seed,
        )
        optimizer.zero_grad(set_to_none=True)
        _action, cosine, score = natural_pullback_action(
            module, target, differentiable_anchor=True
        )
        regularizer = 1e-4 * (
            module.product_log_diagonals.square().mean()
            + module.product_output_log_gain.square().mean()
        )
        (-score + regularizer).backward()
        optimizer.step()
        with torch.no_grad():
            module.product_log_diagonals.clamp_(-bound, bound)
            module.product_output_log_gain.clamp_(-bound, bound)
        if update == 0 or (update + 1) % 16 == 0 or update + 1 == updates:
            history.append(
                {
                    "fit_update": update + 1,
                    "mixture_action_capture": float(score.detach()),
                    "mixture_action_cosine": float(cosine.detach()),
                    "anchor_rms": float(
                        torch.cat(
                            (
                                module.product_log_diagonals.detach().flatten(),
                                module.product_output_log_gain.detach().flatten(),
                            )
                        ).square().mean().sqrt()
                    ),
                    "anchor_max_abs": max(
                        float(module.product_log_diagonals.detach().abs().max()),
                        float(module.product_output_log_gain.detach().abs().max()),
                    ),
                }
            )
    return history


def evaluate_basis(
    module: ProductFHTLinear,
    basis: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    anchor: str,
    cg_iterations: int,
    cg_tolerance: float,
    cg_damping_ratio: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, target in enumerate(basis):
        _action, cosine, score = natural_pullback_action(
            module, target, differentiable_anchor=False
        )
        projection = cg_project(
            module,
            target,
            maximum_iterations=cg_iterations,
            relative_tolerance=cg_tolerance,
            damping_ratio=cg_damping_ratio,
        )
        rows.append(
            {
                "anchor": anchor,
                "pc": index + 1,
                "variance_weight": float(probabilities[index]),
                "natural_action_cosine": float(cosine),
                "natural_action_capture": float(score),
                **projection,
            }
        )
    weights = probabilities.double()
    captures = torch.tensor(
        [row["cg_projection_capture"] for row in rows],
        dtype=torch.float64,
        device=weights.device,
    )
    natural = torch.tensor(
        [row["natural_action_capture"] for row in rows],
        dtype=torch.float64,
        device=weights.device,
    )
    summary = {
        "anchor": anchor,
        "weighted_cg_projection_capture": float((weights * captures).sum()),
        "minimum_cg_projection_capture": float(captures.min()),
        "maximum_cg_projection_capture": float(captures.max()),
        "weighted_natural_action_capture": float((weights * natural).sum()),
        "maximum_cg_normal_relative_residual": max(
            row["cg_final_normal_relative_residual"] for row in rows
        ),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--targets", default="mlp.c_fc,mlp.c_proj")
    parser.add_argument("--basis-rank", type=int, default=16)
    parser.add_argument("--factors", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--fit-updates", type=int, default=128)
    parser.add_argument("--fit-lr", type=float, default=0.02)
    parser.add_argument("--mixture-width", type=int, default=4)
    parser.add_argument("--anchor-bound", type=float, default=0.5)
    parser.add_argument("--fit-seed", type=int, default=20260826)
    parser.add_argument("--cg-iterations", type=int, default=24)
    parser.add_argument("--cg-tolerance", type=float, default=1e-5)
    parser.add_argument("--cg-damping-ratio", type=float, default=1e-6)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    targets = {item for item in args.targets.split(",") if item}
    paths = sorted(args.snapshot_dir.glob("step_*.pt"))
    steps, values, snapshot_metadata = load_snapshots(
        paths, layers={args.layer}, targets=targets
    )
    all_rows: list[dict[str, Any]] = []
    all_summary: list[dict[str, Any]] = []
    all_history: list[dict[str, Any]] = []
    anchors: dict[str, Any] = {}
    accounting: dict[str, Any] = {}
    retained_fractions: dict[str, float] = {}
    for target_index, (parameter, tensors) in enumerate(sorted(values.items())):
        match = PARAMETER_PATTERN.match(parameter)
        if match is None:
            raise ValueError(f"unsupported parameter {parameter}")
        target_name = match.group("target")
        positions = torch.stack(tensors).to(args.device, dtype=torch.float32)
        _residuals, eigenvalues, basis = residual_temporal_basis(
            positions, maximum_rank=args.basis_rank
        )
        retained = eigenvalues[: basis.shape[1]]
        total = eigenvalues.sum().clamp_min(1e-30)
        probabilities = retained / retained.sum().clamp_min(1e-30)
        basis_matrices = basis.T.reshape(basis.shape[1], *positions.shape[1:])
        out_features, in_features = positions.shape[1:]
        weight_std = (
            0.02
            if target_name == "mlp.c_fc"
            else 0.02 / math.sqrt(2 * args.n_layer)
        )
        module = ProductFHTLinear(
            in_features,
            out_features,
            factors=args.factors,
            seed=args.base_seed + args.layer * 4 + TARGET_SEED_OFFSETS[target_name],
            weight_std=weight_std,
            weight_space_muon=False,
            natural_gradient=True,
        ).to(args.device)
        dense_scalars = out_features * in_features
        fraction = module.trainable_scalar_count / dense_scalars
        if fraction > 0.01:
            raise ValueError(f"ProductFHT state exceeds one percent: {fraction}")
        for anchor_name in ("identity",):
            rows, summary = evaluate_basis(
                module,
                basis_matrices,
                probabilities,
                anchor=anchor_name,
                cg_iterations=args.cg_iterations,
                cg_tolerance=args.cg_tolerance,
                cg_damping_ratio=args.cg_damping_ratio,
            )
            all_rows.extend({"parameter": parameter, **row} for row in rows)
            all_summary.append({"parameter": parameter, **summary})
        history = fit_anchor(
            module,
            basis_matrices,
            probabilities,
            updates=args.fit_updates,
            learning_rate=args.fit_lr,
            mixture_width=args.mixture_width,
            bound=args.anchor_bound,
            seed=args.fit_seed + target_index,
        )
        rows, summary = evaluate_basis(
            module,
            basis_matrices,
            probabilities,
            anchor="fitted",
            cg_iterations=args.cg_iterations,
            cg_tolerance=args.cg_tolerance,
            cg_damping_ratio=args.cg_damping_ratio,
        )
        all_rows.extend({"parameter": parameter, **row} for row in rows)
        all_summary.append({"parameter": parameter, **summary})
        all_history.extend({"parameter": parameter, **row} for row in history)
        anchors[parameter] = {
            "product_log_diagonals": module.product_log_diagonals.detach().cpu(),
            "product_output_log_gain": module.product_output_log_gain.detach().cpu(),
            "seed": module.seed,
            "factors": module.factors,
        }
        accounting[parameter] = {
            "dense_scalars": dense_scalars,
            "stored_scalars": module.trainable_scalar_count,
            "stored_scalar_fraction": fraction,
            "largest_stored_vector": max(
                module.padded_features, module.out_features
            ),
            "padded_features": module.padded_features,
        }
        retained_fractions[parameter] = float(retained.sum() / total)
        del positions, basis, basis_matrices, module
        torch.cuda.empty_cache()

    args.output.mkdir(parents=True, exist_ok=True)
    rows_path = args.output / "pc_projection.csv"
    summary_path = args.output / "summary.csv"
    history_path = args.output / "fit_history.csv"
    anchors_path = args.output / "product_fht_anchors.pt"
    write_csv(rows_path, all_rows)
    write_csv(summary_path, all_summary)
    write_csv(history_path, all_history)
    torch.save(anchors, anchors_path)
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_mlp_residual_product_fht_basis_v1",
        "steps": steps,
        "snapshot_metadata": snapshot_metadata,
        "layer": args.layer,
        "targets": sorted(targets),
        "basis_rank": args.basis_rank,
        "retained_residual_energy_fraction": retained_fractions,
        "factors": args.factors,
        "fit_updates": args.fit_updates,
        "fit_lr": args.fit_lr,
        "mixture_width": args.mixture_width,
        "anchor_bound": args.anchor_bound,
        "cg": {
            "iterations": args.cg_iterations,
            "relative_tolerance": args.cg_tolerance,
            "damping_ratio": args.cg_damping_ratio,
        },
        "accounting": accounting,
        "state_contract": {
            "stored": "ProductFHT log diagonals and output gains",
            "not_stored": "no residual PC, ambient atom, dense shadow, or index table",
            "procedural": "Hadamard stages and sign patterns regenerated from seeds",
        },
        "analysis_execution": {
            "git_commit": git_commit(script.parents[2]),
            "entrypoint": str(script),
            "entrypoint_sha256": file_sha256(script),
            "command": sys.argv,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "device": args.device,
        },
        "outputs": {
            rows_path.name: file_sha256(rows_path),
            summary_path.name: file_sha256(summary_path),
            history_path.name: file_sha256(history_path),
            anchors_path.name: file_sha256(anchors_path),
        },
        "limitations": [
            "The full residual horizon is used to fit an optimistic noncausal chart.",
            "Jacobian basis recovery is necessary but not value or CE recovery.",
            "The damped matrix-free CG score may slightly underestimate exact projection.",
            "Only the preregistered five-stage approximately-one-percent chart is tested.",
        ],
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": all_summary, "metadata": str(metadata_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
