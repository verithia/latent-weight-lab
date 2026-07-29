#!/usr/bin/env python3
"""Project dense c_proj phase chords onto a natural-axis tensor chart.

The production bilateral chart uses fixed random FHT-conjugated rotations.
This diagnostic replaces those random bases with the model's deterministic
tensor axes: contiguous within-channel rotations for each MLP expansion/head
group plus one shared rotation across groups.  It adds no learned dense basis
and asks only whether this structured identity tangent is better aligned with
the dense Muon trajectory than its coordinate fraction predicts.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.nanogpt.analyze_mlp_bilateral_phase_capture import (
    aggregate_rows,
    file_sha256,
    git_commit,
    project_target,
    singular_frame_components,
)
from examples.nanogpt.analyze_parameter_trajectory import (
    load_snapshots,
    parse_int_list,
    write_csv,
)


class GroupedCayleyRotation(torch.nn.Module):
    """Orthogonal row-vector chart on deterministic group/channel axes."""

    def __init__(
        self,
        features: int,
        groups: int,
        *,
        coordinate_scale: float,
    ) -> None:
        super().__init__()
        self.features = int(features)
        self.groups = int(groups)
        self.coordinate_scale = float(coordinate_scale)
        if self.features <= 0 or self.groups <= 1:
            raise ValueError("features must be positive and groups must exceed one")
        if self.features % self.groups:
            raise ValueError("groups must divide features")
        if (
            not math.isfinite(self.coordinate_scale)
            or self.coordinate_scale <= 0.0
        ):
            raise ValueError("coordinate_scale must be positive and finite")
        self.channels = self.features // self.groups
        if self.channels <= 1:
            raise ValueError("each group must contain at least two channels")
        channel_rows, channel_columns = torch.triu_indices(
            self.channels,
            self.channels,
            offset=1,
        )
        group_rows, group_columns = torch.triu_indices(
            self.groups,
            self.groups,
            offset=1,
        )
        self.register_buffer(
            "channel_rows", channel_rows, persistent=False
        )
        self.register_buffer(
            "channel_columns", channel_columns, persistent=False
        )
        self.register_buffer("group_rows", group_rows, persistent=False)
        self.register_buffer(
            "group_columns", group_columns, persistent=False
        )
        self.local_coordinates = torch.nn.Parameter(
            torch.zeros(
                self.groups,
                self.channels * (self.channels - 1) // 2,
            )
        )
        self.group_coordinates = torch.nn.Parameter(
            torch.zeros(self.groups * (self.groups - 1) // 2)
        )

    def _cayley(
        self,
        coordinates: torch.Tensor,
        *,
        size: int,
        rows: torch.Tensor,
        columns: torch.Tensor,
    ) -> torch.Tensor:
        scaled = self.coordinate_scale * coordinates
        skew = scaled.new_zeros(*scaled.shape[:-1], size, size)
        rows = rows.to(device=scaled.device)
        columns = columns.to(device=scaled.device)
        skew[..., rows, columns] = scaled
        skew[..., columns, rows] = -scaled
        identity = torch.eye(
            size,
            device=scaled.device,
            dtype=scaled.dtype,
        ).expand(*scaled.shape[:-1], size, size)
        return torch.linalg.solve(identity - skew, identity + skew)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != self.features:
            raise ValueError(
                f"expected last dimension {self.features}, "
                f"got {values.shape[-1]}"
            )
        grouped = values.reshape(
            *values.shape[:-1],
            self.groups,
            self.channels,
        )
        local_rotation = self._cayley(
            self.local_coordinates,
            size=self.channels,
            rows=self.channel_rows,
            columns=self.channel_columns,
        )
        grouped = torch.einsum(
            "...gc,gcd->...gd",
            grouped,
            local_rotation,
        )
        group_rotation = self._cayley(
            self.group_coordinates,
            size=self.groups,
            rows=self.group_rows,
            columns=self.group_columns,
        )
        grouped = torch.einsum(
            "...gc,gh->...hc",
            grouped,
            group_rotation,
        )
        return grouped.reshape_as(values)

    def matrix(self, reference: torch.Tensor) -> torch.Tensor:
        identity = torch.eye(
            self.features,
            device=reference.device,
            dtype=reference.dtype,
        )
        return self(identity)


class NaturalAxisBilateralWeightChart(torch.nn.Module):
    """Bilateral c_proj chart using only deterministic tensor axes."""

    def __init__(
        self,
        hidden_features: int,
        output_features: int,
        *,
        hidden_groups: int,
        output_groups: int,
        coordinate_scale: float,
        gain_scale: float,
    ) -> None:
        super().__init__()
        self.hidden_rotation = GroupedCayleyRotation(
            hidden_features,
            hidden_groups,
            coordinate_scale=coordinate_scale,
        )
        self.hidden_log_gain = torch.nn.Parameter(
            torch.zeros(hidden_features)
        )
        self.output_rotation = GroupedCayleyRotation(
            output_features,
            output_groups,
            coordinate_scale=coordinate_scale,
        )
        self.output_log_gain = torch.nn.Parameter(
            torch.zeros(output_features)
        )
        self.gain_scale = float(gain_scale)
        if not math.isfinite(self.gain_scale) or self.gain_scale <= 0.0:
            raise ValueError("gain_scale must be positive and finite")

    def forward(self, base_weight: torch.Tensor) -> torch.Tensor:
        hidden_rotation = self.hidden_rotation.matrix(base_weight)
        charted = base_weight @ hidden_rotation.transpose(0, 1)
        hidden_gain = (
            self.gain_scale * self.hidden_log_gain
        ).exp().to(device=base_weight.device, dtype=base_weight.dtype)
        charted = charted * hidden_gain
        transposed = charted.transpose(0, 1)
        output_gain = (
            self.gain_scale * self.output_log_gain
        ).exp().to(device=base_weight.device, dtype=base_weight.dtype)
        transposed = transposed * output_gain
        output_rotation = self.output_rotation.matrix(transposed)
        transposed = transposed @ output_rotation
        return transposed.transpose(0, 1).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", default="0,3,6,9,11")
    parser.add_argument(
        "--phase-boundaries", default="0,60,120,180,238"
    )
    parser.add_argument(
        "--components",
        default=(
            "total,singular_value,in_frame_mixing,subspace_rotation"
        ),
    )
    parser.add_argument("--hidden-groups", type=int, default=48)
    parser.add_argument("--output-groups", type=int, default=12)
    parser.add_argument("--coordinate-scale", type=float, default=4.0)
    parser.add_argument("--gain-scale", type=float, default=4.0)
    parser.add_argument("--damping-ratio", type=float, default=1e-6)
    parser.add_argument("--cg-steps", type=int, default=24)
    parser.add_argument("--trace-seed", type=int, default=20260729)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    layers = parse_int_list(args.layers)
    boundaries = parse_int_list(args.phase_boundaries)
    components = [item for item in args.components.split(",") if item]
    allowed_components = {
        "total",
        "singular_value",
        "in_frame_mixing",
        "subspace_rotation",
    }
    if (
        not layers
        or len(boundaries) < 2
        or boundaries != sorted(set(boundaries))
        or not components
        or not set(components) <= allowed_components
    ):
        raise ValueError("invalid layers, phase boundaries, or components")
    paths = [
        args.snapshot_dir / f"step_{step:06d}.pt"
        for step in boundaries
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"phase-boundary snapshots are absent: {missing}")
    steps, values, snapshot_metadata = load_snapshots(
        paths,
        layers=set(layers),
        targets={"mlp.c_proj"},
    )
    if steps != boundaries:
        raise ValueError("loaded snapshot steps do not match phase boundaries")

    rows: list[dict[str, Any]] = []
    coordinate_count: int | None = None
    for name, tensors in sorted(values.items()):
        layer = int(name.split(".")[2])
        for phase_index, (start, end) in enumerate(
            zip(steps[:-1], steps[1:], strict=True)
        ):
            base = tensors[phase_index].to(
                device=args.device,
                dtype=torch.float64,
            )
            terminal = tensors[phase_index + 1].to(
                device=args.device,
                dtype=torch.float64,
            )
            component_targets = singular_frame_components(
                base,
                terminal - base,
            )
            for component_index, component in enumerate(components):
                chart = NaturalAxisBilateralWeightChart(
                    hidden_features=base.shape[1],
                    output_features=base.shape[0],
                    hidden_groups=args.hidden_groups,
                    output_groups=args.output_groups,
                    coordinate_scale=args.coordinate_scale,
                    gain_scale=args.gain_scale,
                ).to(device=args.device, dtype=torch.float64)
                actual_coordinate_count = sum(
                    parameter.numel() for parameter in chart.parameters()
                )
                if coordinate_count is None:
                    coordinate_count = actual_coordinate_count
                elif coordinate_count != actual_coordinate_count:
                    raise RuntimeError("coordinate count changed across cells")
                metrics = project_target(
                    chart,
                    base,
                    component_targets[component],
                    damping_ratio=args.damping_ratio,
                    cg_steps=args.cg_steps,
                    trace_seed=(
                        args.trace_seed
                        + layer * 1009
                        + phase_index * 101
                        + component_index
                    ),
                )
                row = {
                    "parameter": name,
                    "layer": layer,
                    "phase_start": start,
                    "phase_end": end,
                    "component": component,
                    **metrics,
                }
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
                del chart
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            del base, terminal, component_targets
    if coordinate_count is None:
        raise RuntimeError("analysis produced no rows")
    aggregates = aggregate_rows(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    detail_path = args.output / "natural_axis_phase_capture.csv"
    aggregate_path = (
        args.output / "natural_axis_phase_capture_aggregate.csv"
    )
    write_csv(detail_path, rows)
    write_csv(aggregate_path, aggregates)

    script = Path(__file__).resolve()
    repo = script.parents[2]
    metadata = {
        "schema_version": "nanogpt_mlp_natural_axis_phase_capture_v1",
        "snapshot_metadata": snapshot_metadata,
        "layers": layers,
        "phase_boundaries": boundaries,
        "components": components,
        "chart": {
            "formula": "R_out^T D_out W_base R_hidden^T D_hidden",
            "hidden_layout": (
                "48 contiguous groups x 64 channels = "
                "4 expansion groups x 12 model heads x 64 channels"
            ),
            "output_layout": (
                "12 contiguous model-head groups x 64 channels"
            ),
            "hidden_groups": args.hidden_groups,
            "hidden_channels_per_group": (
                rows[0]["coordinate_count"]
                and values[next(iter(values))][0].shape[1]
                // args.hidden_groups
            ),
            "output_groups": args.output_groups,
            "output_channels_per_group": (
                values[next(iter(values))][0].shape[0]
                // args.output_groups
            ),
            "coordinate_scale": args.coordinate_scale,
            "gain_scale": args.gain_scale,
            "coordinate_count": coordinate_count,
            "ambient_matrix_dimension": int(
                values[next(iter(values))][0].numel()
            ),
            "initialization": "exact identity",
            "fixed_random_basis": False,
            "learned_dense_basis": False,
            "lora_adapter": False,
        },
        "solver": {
            "system": "(J^T J + damping I) x = J^T target",
            "damping_ratio_to_hutchinson_mean_eigenvalue": (
                args.damping_ratio
            ),
            "cg_steps": args.cg_steps,
            "trace_seed": args.trace_seed,
            "analysis_dtype": "float64",
        },
        "analysis_execution": {
            "git_commit": git_commit(repo),
            "entrypoint": str(script),
            "entrypoint_sha256": file_sha256(script),
            "command": sys.argv,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "device": args.device,
        },
        "snapshot_files": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in paths
        ],
        "output": {
            "detail": {
                "path": str(detail_path),
                "sha256": file_sha256(detail_path),
            },
            "aggregate": {
                "path": str(aggregate_path),
                "sha256": file_sha256(aggregate_path),
            },
        },
        "limitations": [
            "This is local identity-Jacobian coverage, not nonlinear reachability.",
            "Natural contiguous axes are an architectural hypothesis, not learned trajectory bases.",
            "Only preregistered representative layers 0,3,6,9,11 are analyzed.",
            "Weight-space capture does not prove task-loss usefulness.",
        ],
    }
    metadata_path = (
        args.output / "natural_axis_phase_capture_metadata.json"
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "rows": len(rows),
                "aggregates": aggregates,
                "metadata": str(metadata_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
