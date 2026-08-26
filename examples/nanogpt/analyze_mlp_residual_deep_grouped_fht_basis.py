#!/usr/bin/env python3
"""Audit equal-state deeper ProductFHT tangents with grouped diagonals.

Each depth/group pair stores exactly five full-diagonal equivalents plus one
output gain.  Group assignments, signs, and FHT signs are regenerated from
integer seeds; no residual PC or dense basis is candidate state.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from examples.nanogpt.analyze_mlp_highcadence_basis import file_sha256
from examples.nanogpt.analyze_mlp_residual_multibranch_fht_basis import (
    TARGET_FIT_OFFSETS,
    TARGET_SEED_OFFSETS,
    evaluate_basis,
    fit_anchor,
    git_commit,
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


def parse_depth_groups(value: str) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for item in value.split(","):
        depth, group = (int(part) for part in item.split(":"))
        if min(depth, group) < 1:
            raise ValueError(f"invalid depth/group {item!r}")
        result.append((depth, group))
    if not result:
        raise ValueError("at least one depth/group pair is required")
    equivalents = {depth / group for depth, group in result}
    if len(equivalents) != 1:
        raise ValueError("all candidates must have equal diagonal state")
    return result


class DeepGroupedProductFHT(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        depth: int,
        group_size: int,
        seed: int,
        weight_std: float,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.depth = int(depth)
        self.group_size = int(group_size)
        self.seed = int(seed)
        self.weight_std = float(weight_std)
        self.padded_features = next_power_of_two(
            max(self.in_features, self.out_features)
        )
        if self.padded_features % self.group_size:
            raise ValueError("group size must divide padded width")
        self.grouped_log_diagonals = nn.Parameter(
            torch.zeros(
                self.depth, self.padded_features // self.group_size
            )
        )
        self.shared_output_log_gain = nn.Parameter(torch.zeros(self.out_features))
        reference = torch.empty(1)
        factor_signs = torch.stack(
            [
                signs_for(
                    reference,
                    block=stage,
                    layer=stage + 1,
                    seed=self.seed,
                    block_size=self.padded_features,
                )
                for stage in range(self.depth)
            ]
        )
        self.register_buffer("factor_signs", factor_signs, persistent=False)
        if self.group_size == 1:
            permutations = torch.arange(self.padded_features).repeat(self.depth, 1)
            expansion_signs = torch.ones(self.depth, self.padded_features)
        else:
            permutations = []
            expansion_signs = []
            for stage in range(self.depth):
                generator = torch.Generator(device="cpu").manual_seed(
                    self.seed + 1_000_003 * (stage + 1) + self.group_size
                )
                permutations.append(
                    torch.randperm(self.padded_features, generator=generator)
                )
                expansion_signs.append(
                    torch.randint(
                        0,
                        2,
                        (self.padded_features,),
                        generator=generator,
                    ).float().mul(2.0).sub(1.0)
                )
            permutations = torch.stack(permutations)
            expansion_signs = torch.stack(expansion_signs)
        self.register_buffer(
            "expansion_permutations", permutations, persistent=False
        )
        self.register_buffer(
            "expansion_signs", expansion_signs, persistent=False
        )

    @property
    def coordinate_tensors(self) -> tuple[torch.Tensor, ...]:
        return (self.grouped_log_diagonals, self.shared_output_log_gain)

    @property
    def trainable_scalar_count(self) -> int:
        return sum(parameter.numel() for parameter in self.coordinate_tensors)

    def split_coordinates(
        self, vector: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count = self.grouped_log_diagonals.numel()
        return (
            vector[:count].reshape_as(self.grouped_log_diagonals),
            vector[count:].reshape_as(self.shared_output_log_gain),
        )

    def _expand(self, grouped: torch.Tensor, stage: int) -> torch.Tensor:
        expanded = grouped.repeat_interleave(self.group_size)
        if self.group_size == 1:
            return expanded
        return (
            expanded[self.expansion_permutations[stage]]
            * self.expansion_signs[stage].to(expanded.dtype)
        )

    def _weight_and_jvp(
        self,
        grouped_anchor: torch.Tensor,
        output_anchor: torch.Tensor,
        grouped_direction: torch.Tensor | None,
        output_direction: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        matrix = torch.eye(
            self.out_features,
            self.padded_features,
            dtype=grouped_anchor.dtype,
            device=grouped_anchor.device,
        )
        tangent = torch.zeros_like(matrix) if grouped_direction is not None else None
        factor_signs = self.factor_signs.to(matrix.device, matrix.dtype)
        for stage in range(self.depth):
            matrix = normalized_fht_last_dim(matrix * factor_signs[stage])
            if tangent is not None:
                tangent = normalized_fht_last_dim(tangent * factor_signs[stage])
            full_coordinate = self._expand(grouped_anchor[stage], stage)
            clamped = full_coordinate.clamp(-6.0, 6.0)
            diagonal = torch.exp(clamped)
            if tangent is not None and grouped_direction is not None:
                full_direction = self._expand(grouped_direction[stage], stage)
                active = (
                    (full_coordinate > -6.0) & (full_coordinate < 6.0)
                ).to(matrix.dtype)
                diagonal_tangent = diagonal * full_direction * active
                tangent = tangent * diagonal + matrix * diagonal_tangent
            matrix = matrix * diagonal
        output_clamped = output_anchor.clamp(-6.0, 6.0)
        output_gain = torch.exp(output_clamped)
        scale = self.weight_std * math.sqrt(self.padded_features)
        weight = (
            scale
            * output_gain.view(-1, 1)
            * matrix[:, : self.in_features]
        )
        if tangent is None or output_direction is None:
            return weight, None
        output_active = (
            (output_anchor > -6.0) & (output_anchor < 6.0)
        ).to(output_anchor.dtype)
        output_tangent = output_gain * output_direction * output_active
        tangent = scale * (
            output_gain.view(-1, 1) * tangent[:, : self.in_features]
            + output_tangent.view(-1, 1) * matrix[:, : self.in_features]
        )
        return weight, tangent

    def weight(self) -> torch.Tensor:
        weight, _ = self._weight_and_jvp(
            self.grouped_log_diagonals,
            self.shared_output_log_gain,
            None,
            None,
        )
        return weight

    def jvp(
        self, vector: torch.Tensor, *, differentiable_anchor: bool = False
    ) -> torch.Tensor:
        grouped_direction, output_direction = self.split_coordinates(vector)
        grouped_anchor = (
            self.grouped_log_diagonals
            if differentiable_anchor
            else self.grouped_log_diagonals.detach()
        )
        output_anchor = (
            self.shared_output_log_gain
            if differentiable_anchor
            else self.shared_output_log_gain.detach()
        )
        _, tangent = self._weight_and_jvp(
            grouped_anchor,
            output_anchor,
            grouped_direction,
            output_direction,
        )
        if tangent is None:
            raise AssertionError("JVP unexpectedly absent")
        return tangent

    def coordinate_metric(self) -> torch.Tensor:
        per_position = (
            self.weight_std
            * self.weight_std
            * self.out_features
            * self.in_features
            / self.padded_features
        )
        grouped_metric = torch.full_like(
            self.grouped_log_diagonals,
            max(per_position * self.group_size, 1e-12),
        ).reshape(-1)
        with torch.no_grad():
            row_metric = self.weight().float().square().sum(dim=1).clamp_min(1e-12)
        return torch.cat(
            (grouped_metric, row_metric.to(self.shared_output_log_gain.dtype))
        )

    def ideal_forward_scalar_ops(self) -> int:
        log_width = int(math.log2(self.padded_features))
        return (
            self.depth * self.padded_features * (log_width + 2)
            + self.out_features
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--targets", default="mlp.c_fc,mlp.c_proj")
    parser.add_argument("--basis-rank", type=int, default=16)
    parser.add_argument("--depth-groups", default="5:1,10:2,20:4,40:8")
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
    depth_groups = parse_depth_groups(args.depth_groups)
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
    for parameter, tensors in sorted(values.items()):
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
        expected_state: int | None = None
        for candidate_index, (depth, group_size) in enumerate(depth_groups):
            name = f"depth{depth}_group{group_size}"
            module = DeepGroupedProductFHT(
                in_features,
                out_features,
                depth=depth,
                group_size=group_size,
                seed=args.base_seed + args.layer * 4 + TARGET_SEED_OFFSETS[target_name],
                weight_std=weight_std,
            ).to(args.device)
            if expected_state is None:
                expected_state = module.trainable_scalar_count
            if module.trainable_scalar_count != expected_state:
                raise ValueError("depth/group candidates do not have equal state")
            fraction = module.trainable_scalar_count / dense_scalars
            if fraction > 0.01:
                raise ValueError(f"candidate {name} exceeds one percent")
            history = fit_anchor(
                module,
                basis_matrices,
                probabilities,
                updates=args.fit_updates,
                learning_rate=args.fit_lr,
                mixture_width=args.mixture_width,
                bound=args.anchor_bound,
                seed=(
                    args.fit_seed
                    + TARGET_FIT_OFFSETS[target_name]
                    + candidate_index * 100
                ),
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
                "candidate": name,
                "depth": depth,
                "group_size": group_size,
            }
            all_rows.extend({**prefix, **row} for row in rows)
            all_summary.append({**prefix, **summary})
            all_history.extend({**prefix, **row} for row in history)
            parameter_anchors[name] = {
                "grouped_log_diagonals": module.grouped_log_diagonals.detach().cpu(),
                "shared_output_log_gain": module.shared_output_log_gain.detach().cpu(),
                "seed": module.seed,
            }
            accounting[f"{parameter}:{name}"] = {
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
    anchors_path = args.output / "deep_grouped_fht_anchors.pt"
    write_csv(rows_path, all_rows)
    write_csv(summary_path, all_summary)
    write_csv(history_path, all_history)
    torch.save(anchors, anchors_path)
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_mlp_residual_deep_grouped_fht_basis_v1",
        "steps": steps,
        "snapshot_metadata": snapshot_metadata,
        "layer": args.layer,
        "targets": sorted(targets),
        "basis_rank": args.basis_rank,
        "retained_residual_energy_fraction": retained_fractions,
        "depth_groups": [list(item) for item in depth_groups],
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
            "stored": "grouped ProductFHT diagonals and one output gain",
            "not_stored": "no PCA vector, ambient atom, dense shadow, grouping index, sign table, or tangent coefficient",
            "procedural": "group assignments and all signs regenerated from integer seeds",
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
