#!/usr/bin/env python3
"""Audit equal-budget multi-branch ProductFHT tangents on MLP residual PCs.

All candidates contain the same number of learned FHT diagonals and one
shared output gain.  Only the serial/parallel partition of the stages changes.
Branch signs are regenerated from seeds; no dense PCA vector or ambient basis
is candidate state.
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
from torch import nn

from examples.nanogpt.analyze_mlp_highcadence_basis import file_sha256
from examples.nanogpt.analyze_mlp_residual_product_fht_basis import (
    deterministic_weighted_mixture,
)
from examples.nanogpt.analyze_mlp_residual_qtt_basis import residual_temporal_basis
from examples.nanogpt.analyze_parameter_trajectory import (
    PARAMETER_PATTERN,
    load_snapshots,
    write_csv,
)
from latent_weight_lab.block_fht import (
    next_power_of_two,
    normalized_fht_last_dim,
    signs_for,
)


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


def parse_topologies(value: str) -> list[tuple[int, ...]]:
    topologies: list[tuple[int, ...]] = []
    for item in value.split(";"):
        depths = tuple(int(depth) for depth in item.split("+") if depth)
        if not depths or min(depths) < 1:
            raise ValueError(f"invalid topology {item!r}")
        topologies.append(depths)
    if not topologies:
        raise ValueError("at least one topology is required")
    total = sum(topologies[0])
    if any(sum(topology) != total for topology in topologies):
        raise ValueError("all topologies must have the same total FHT depth")
    return topologies


class MultiBranchProductFHT(nn.Module):
    """Sum of independently seeded ProductFHT paths with shared output gain."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        branch_depths: tuple[int, ...],
        seed: int,
        weight_std: float,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.branch_depths = tuple(int(depth) for depth in branch_depths)
        self.seed = int(seed)
        self.weight_std = float(weight_std)
        self.padded_features = next_power_of_two(
            max(self.in_features, self.out_features)
        )
        self.branch_log_diagonals = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(depth, self.padded_features))
                for depth in self.branch_depths
            ]
        )
        self.shared_output_log_gain = nn.Parameter(torch.zeros(self.out_features))
        reference = torch.empty(1)
        for branch, depth in enumerate(self.branch_depths):
            branch_signs = torch.stack(
                [
                    signs_for(
                        reference,
                        block=stage,
                        layer=stage + 1,
                        seed=self.seed + 104729 * branch,
                        block_size=self.padded_features,
                    )
                    for stage in range(depth)
                ]
            )
            self.register_buffer(
                f"branch_signs_{branch}", branch_signs, persistent=False
            )

    @property
    def coordinate_tensors(self) -> tuple[torch.Tensor, ...]:
        return tuple(self.branch_log_diagonals) + (self.shared_output_log_gain,)

    @property
    def trainable_scalar_count(self) -> int:
        return sum(parameter.numel() for parameter in self.coordinate_tensors)

    @property
    def total_factors(self) -> int:
        return sum(self.branch_depths)

    def _branch_weight(
        self, branch: int, log_diagonals: torch.Tensor
    ) -> torch.Tensor:
        matrix = torch.eye(
            self.out_features,
            self.padded_features,
            dtype=log_diagonals.dtype,
            device=log_diagonals.device,
        )
        signs = getattr(self, f"branch_signs_{branch}").to(
            device=matrix.device, dtype=matrix.dtype
        )
        for stage in range(log_diagonals.shape[0]):
            matrix = normalized_fht_last_dim(matrix * signs[stage])
            matrix = matrix * torch.exp(log_diagonals[stage].clamp(-6.0, 6.0))
        return matrix[:, : self.in_features]

    def _branch_jvp(
        self,
        branch: int,
        log_diagonals: torch.Tensor,
        direction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        matrix = torch.eye(
            self.out_features,
            self.padded_features,
            dtype=log_diagonals.dtype,
            device=log_diagonals.device,
        )
        tangent = torch.zeros_like(matrix)
        signs = getattr(self, f"branch_signs_{branch}").to(
            device=matrix.device, dtype=matrix.dtype
        )
        for stage in range(log_diagonals.shape[0]):
            matrix = normalized_fht_last_dim(matrix * signs[stage])
            tangent = normalized_fht_last_dim(tangent * signs[stage])
            coordinate = log_diagonals[stage]
            clamped = coordinate.clamp(-6.0, 6.0)
            diagonal = torch.exp(clamped)
            active = ((coordinate > -6.0) & (coordinate < 6.0)).to(matrix.dtype)
            diagonal_tangent = diagonal * direction[stage] * active
            tangent = tangent * diagonal + matrix * diagonal_tangent
            matrix = matrix * diagonal
        return matrix[:, : self.in_features], tangent[:, : self.in_features]

    def weight(self) -> torch.Tensor:
        scale = self.weight_std / math.sqrt(len(self.branch_depths))
        inner = sum(
            self._branch_weight(branch, diagonals)
            for branch, diagonals in enumerate(self.branch_log_diagonals)
        )
        output_gain = torch.exp(self.shared_output_log_gain.clamp(-6.0, 6.0))
        return scale * output_gain.view(-1, 1) * inner

    def split_coordinates(
        self, vector: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        offset = 0
        branches: list[torch.Tensor] = []
        for parameter in self.branch_log_diagonals:
            count = parameter.numel()
            branches.append(vector[offset : offset + count].reshape_as(parameter))
            offset += count
        output = vector[offset:].reshape_as(self.shared_output_log_gain)
        return branches, output

    def jvp(
        self, vector: torch.Tensor, *, differentiable_anchor: bool = False
    ) -> torch.Tensor:
        branch_directions, output_direction = self.split_coordinates(vector)
        anchors = (
            list(self.branch_log_diagonals)
            if differentiable_anchor
            else [parameter.detach() for parameter in self.branch_log_diagonals]
        )
        output_anchor = (
            self.shared_output_log_gain
            if differentiable_anchor
            else self.shared_output_log_gain.detach()
        )
        matrices: list[torch.Tensor] = []
        tangents: list[torch.Tensor] = []
        for branch, (anchor, direction) in enumerate(
            zip(anchors, branch_directions, strict=True)
        ):
            matrix, tangent = self._branch_jvp(branch, anchor, direction)
            matrices.append(matrix)
            tangents.append(tangent)
        inner = sum(matrices)
        inner_tangent = sum(tangents)
        clamped = output_anchor.clamp(-6.0, 6.0)
        output_gain = torch.exp(clamped)
        output_active = (
            (output_anchor > -6.0) & (output_anchor < 6.0)
        ).to(output_anchor.dtype)
        output_tangent = output_gain * output_direction * output_active
        scale = self.weight_std / math.sqrt(len(self.branch_depths))
        return scale * (
            output_gain.view(-1, 1) * inner_tangent
            + output_tangent.view(-1, 1) * inner
        )

    def ideal_forward_scalar_ops(self) -> int:
        log_width = int(math.log2(self.padded_features))
        fht_and_diagonal = self.total_factors * self.padded_features * (
            log_width + 1
        )
        branch_combination = max(len(self.branch_depths) - 1, 0) * self.padded_features
        return fht_and_diagonal + branch_combination + self.out_features


def flatten_tensors(tensors: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


def coordinate_vjp(
    module: MultiBranchProductFHT, target: torch.Tensor
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        module.weight(),
        module.coordinate_tensors,
        grad_outputs=target.to(dtype=module.shared_output_log_gain.dtype),
        create_graph=False,
        retain_graph=False,
    )
    return flatten_tensors(gradients).detach()


def coordinate_metric(module: MultiBranchProductFHT) -> torch.Tensor:
    branch_scale = module.weight_std / math.sqrt(len(module.branch_depths))
    diagonal_metric = (
        branch_scale
        * branch_scale
        * module.out_features
        * module.in_features
        / module.padded_features
    )
    pieces = [
        torch.full_like(parameter, max(diagonal_metric, 1e-12)).reshape(-1)
        for parameter in module.branch_log_diagonals
    ]
    with torch.no_grad():
        row_metric = module.weight().float().square().sum(dim=1).clamp_min(1e-12)
    pieces.append(row_metric.to(module.shared_output_log_gain.dtype))
    return torch.cat(pieces)


def natural_action(
    module: MultiBranchProductFHT,
    target: torch.Tensor,
    *,
    differentiable_anchor: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    direction = (coordinate_vjp(module, target) / coordinate_metric(module)).detach()
    action = module.jvp(direction, differentiable_anchor=differentiable_anchor)
    cosine = torch.sum(action.float() * target.float()) / (
        action.float().norm() * target.float().norm()
    ).clamp_min(1e-30)
    return action, cosine, cosine.square()


def cg_project(
    module: MultiBranchProductFHT,
    target: torch.Tensor,
    *,
    maximum_iterations: int,
    relative_tolerance: float,
    damping_ratio: float,
) -> dict[str, Any]:
    target = target.float()
    right = coordinate_vjp(module, target)
    metric = coordinate_metric(module)
    damping = damping_ratio * float(metric.mean())

    def normal(vector: torch.Tensor) -> torch.Tensor:
        return coordinate_vjp(module, module.jvp(vector)) + damping * vector

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
        rz = next_rz
    projected = module.jvp(estimate)
    target_energy = target.double().square().sum().clamp_min(1e-30)
    error_energy = (target - projected).double().square().sum()
    return {
        "cg_projection_capture": float(1.0 - error_energy / target_energy),
        "cg_iterations": iterations,
        "cg_final_normal_relative_residual": float(
            residual.double().norm() / initial_norm
        ),
    }


def fit_anchor(
    module: MultiBranchProductFHT,
    basis: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    updates: int,
    learning_rate: float,
    mixture_width: int,
    bound: float,
    seed: int,
) -> list[dict[str, Any]]:
    optimizer = torch.optim.Adam(module.coordinate_tensors, lr=learning_rate)
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
        _action, cosine, score = natural_action(
            module, target, differentiable_anchor=True
        )
        regularizer = 1e-4 * torch.stack(
            [coordinate.square().mean() for coordinate in module.coordinate_tensors]
        ).mean()
        (-score + regularizer).backward()
        optimizer.step()
        with torch.no_grad():
            for coordinate in module.coordinate_tensors:
                coordinate.clamp_(-bound, bound)
        if update == 0 or (update + 1) % 16 == 0 or update + 1 == updates:
            flat = flatten_tensors(tuple(t.detach() for t in module.coordinate_tensors))
            history.append(
                {
                    "fit_update": update + 1,
                    "mixture_action_capture": float(score.detach()),
                    "mixture_action_cosine": float(cosine.detach()),
                    "anchor_rms": float(flat.square().mean().sqrt()),
                    "anchor_max_abs": float(flat.abs().max()),
                }
            )
    return history


def evaluate_basis(
    module: MultiBranchProductFHT,
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
        _action, cosine, score = natural_action(
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
    return rows, {
        "anchor": anchor,
        "weighted_cg_projection_capture": float((weights * captures).sum()),
        "minimum_cg_projection_capture": float(captures.min()),
        "maximum_cg_projection_capture": float(captures.max()),
        "maximum_cg_normal_relative_residual": max(
            row["cg_final_normal_relative_residual"] for row in rows
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--targets", default="mlp.c_fc,mlp.c_proj")
    parser.add_argument("--basis-rank", type=int, default=16)
    parser.add_argument("--topologies", default="5;3+2;2+2+1;1+1+1+1+1")
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
    topologies = parse_topologies(args.topologies)
    targets = {item for item in args.targets.split(",") if item}
    paths = sorted(args.snapshot_dir.glob("step_*.pt"))
    steps, values, snapshot_metadata = load_snapshots(
        paths, layers={args.layer}, targets=targets
    )
    all_rows: list[dict[str, Any]] = []
    all_summary: list[dict[str, Any]] = []
    all_history: list[dict[str, Any]] = []
    anchors: dict[str, Any] = {}
    retained_fractions: dict[str, float] = {}
    accounting: dict[str, Any] = {}
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
        probabilities = retained / retained.sum().clamp_min(1e-30)
        basis_matrices = basis.T.reshape(basis.shape[1], *positions.shape[1:])
        out_features, in_features = positions.shape[1:]
        dense_scalars = out_features * in_features
        weight_std = (
            0.02 if target_name == "mlp.c_fc" else 0.02 / math.sqrt(2 * args.n_layer)
        )
        parameter_anchors: dict[str, Any] = {}
        for topology_index, topology in enumerate(topologies):
            topology_name = "+".join(str(depth) for depth in topology)
            module = MultiBranchProductFHT(
                in_features,
                out_features,
                branch_depths=topology,
                seed=(
                    args.base_seed
                    + args.layer * 4
                    + TARGET_SEED_OFFSETS[target_name]
                    + topology_index * 1_000_003
                ),
                weight_std=weight_std,
            ).to(args.device)
            fraction = module.trainable_scalar_count / dense_scalars
            if fraction > 0.01:
                raise ValueError(f"topology {topology_name} exceeds one percent")
            history = fit_anchor(
                module,
                basis_matrices,
                probabilities,
                updates=args.fit_updates,
                learning_rate=args.fit_lr,
                mixture_width=args.mixture_width,
                bound=args.anchor_bound,
                seed=args.fit_seed + target_index * 100 + topology_index,
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
            prefix = {
                "parameter": parameter,
                "topology": topology_name,
                "branch_count": len(topology),
                "total_factors": sum(topology),
            }
            all_rows.extend({**prefix, **row} for row in rows)
            all_summary.append({**prefix, **summary})
            all_history.extend({**prefix, **row} for row in history)
            parameter_anchors[topology_name] = {
                "branch_log_diagonals": [
                    value.detach().cpu() for value in module.branch_log_diagonals
                ],
                "shared_output_log_gain": module.shared_output_log_gain.detach().cpu(),
                "seed": module.seed,
            }
            accounting[f"{parameter}:{topology_name}"] = {
                "dense_scalars": dense_scalars,
                "stored_scalars": module.trainable_scalar_count,
                "stored_scalar_fraction": fraction,
                "ideal_forward_scalar_ops": module.ideal_forward_scalar_ops(),
                "ideal_forward_ops_to_dense_madds": (
                    module.ideal_forward_scalar_ops() / dense_scalars
                ),
            }
            del module
            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()
        anchors[parameter] = parameter_anchors
        retained_fractions[parameter] = float(
            retained.sum() / eigenvalues.sum().clamp_min(1e-30)
        )
        del positions, basis, basis_matrices

    args.output.mkdir(parents=True, exist_ok=True)
    rows_path = args.output / "pc_projection.csv"
    summary_path = args.output / "summary.csv"
    history_path = args.output / "fit_history.csv"
    anchors_path = args.output / "multibranch_fht_anchors.pt"
    write_csv(rows_path, all_rows)
    write_csv(summary_path, all_summary)
    write_csv(history_path, all_history)
    torch.save(anchors, anchors_path)
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_mlp_residual_multibranch_fht_basis_v1",
        "steps": steps,
        "snapshot_metadata": snapshot_metadata,
        "layer": args.layer,
        "targets": sorted(targets),
        "basis_rank": args.basis_rank,
        "retained_residual_energy_fraction": retained_fractions,
        "topologies": [list(topology) for topology in topologies],
        "fit": {
            "updates": args.fit_updates,
            "learning_rate": args.fit_lr,
            "mixture_width": args.mixture_width,
            "anchor_bound": args.anchor_bound,
        },
        "cg": {
            "iterations": args.cg_iterations,
            "relative_tolerance": args.cg_tolerance,
            "damping_ratio": args.cg_damping_ratio,
        },
        "accounting": accounting,
        "state_contract": {
            "stored": "five ProductFHT diagonal vectors and one shared output gain",
            "not_stored": "no residual PC, ambient atom, dense shadow, sign table, or tangent coefficient",
            "procedural": "branch connectivity and signs regenerated from integer seeds",
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
            "The same 239-state horizon is used for optimistic noncausal anchor fitting and evaluation.",
            "Jacobian recovery is necessary but not sufficient for online optimization or CE closure.",
            "Ideal forward arithmetic assumes a fused matrix-free implementation not yet performance-gated.",
        ],
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": all_summary, "metadata": str(metadata_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
