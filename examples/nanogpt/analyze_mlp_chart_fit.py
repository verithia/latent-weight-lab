#!/usr/bin/env python3
"""Test whether compact MLP charts align with a measured dense trajectory.

This complements trajectory PCA.  PCA measures the dimension of one optimizer
path after training; it does not say that a randomly oriented chart of the same
dimension intersects that path.  Here we measure two concrete candidate
tangents on dense checkpoint-to-checkpoint displacements:

* the exact orthogonal projection onto the configured linear BlockFHT tangent;
* a shared hidden-channel radial tangent for paired ``c_fc`` rows and
  ``c_proj`` columns.

The BlockFHT calculation is exact when the generated matrix size is an integer
multiple of the padded latent block size.  That is true for the registered
124M MLP ratios used by the MAI ladder.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.parameter_trajectory import SCHEMA_VERSION
from latent_weight_lab.block_fht import (
    block_fht_grad_latent,
    next_power_of_two,
)


PARAMETER_PATTERN = re.compile(
    r"^transformer\.h\.(?P<layer>\d+)\.(?P<target>mlp\.(?:c_fc|c_proj))\.weight$"
)
TARGET_SEED_OFFSETS = {"mlp.c_fc": 2, "mlp.c_proj": 3}


def parse_int_list(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item]
    if any(item < 0 for item in values):
        raise ValueError("steps and layers must be non-negative")
    return values


def parse_float_list(value: str) -> list[float]:
    values = [float(item) for item in value.split(",") if item]
    if any(not math.isfinite(item) or not 0.0 < item <= 1.0 for item in values):
        raise ValueError("latent ratios must be finite and in (0, 1]")
    return values


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def resolved_latent_dim(size: int, ratio: float, latent_rows: int) -> tuple[int, tuple[int, int] | None]:
    latent_dim = max(1, round(int(size) * float(ratio)))
    if latent_rows <= 0:
        return latent_dim, None
    rows = min(int(latent_rows), latent_dim)
    columns = math.ceil(latent_dim / rows)
    return rows * columns, (rows, columns)


def exact_block_fht_projection(
    delta: torch.Tensor,
    *,
    latent_dim: int,
    latent_shape: tuple[int, int] | None,
    layers: int,
    seed: int,
) -> dict[str, float | int]:
    """Return exact projection energy for a repeated orthogonal BlockFHT map."""
    flat = delta.reshape(-1).contiguous().float()
    block_size = next_power_of_two(int(latent_dim))
    if flat.numel() % block_size:
        raise ValueError(
            f"exact projection requires size={flat.numel()} divisible by block_size={block_size}"
        )
    shape = latent_shape if latent_shape is not None else (latent_dim,)
    latent = torch.zeros(shape, device=flat.device, dtype=flat.dtype)
    adjoint = block_fht_grad_latent(
        latent,
        flat,
        flat.numel(),
        int(layers),
        int(seed),
    )
    repeated_blocks = flat.numel() // block_size
    delta_energy = flat.square().sum()
    projection_energy = adjoint.square().sum() / repeated_blocks
    fraction = projection_energy / delta_energy.clamp_min(1e-30)
    return {
        "latent_dim": int(latent_dim),
        "latent_block_size": int(block_size),
        "repeated_blocks": int(repeated_blocks),
        "delta_energy": float(delta_energy),
        "projection_energy": float(projection_energy),
        "projection_energy_fraction": float(fraction),
        "projection_norm_fraction": float(fraction.clamp_min(0.0).sqrt()),
    }


def shared_hidden_radial_projection(
    fc_base: torch.Tensor,
    fc_delta: torch.Tensor,
    proj_base: torch.Tensor,
    proj_delta: torch.Tensor,
) -> dict[str, float]:
    """Fit one gain per hidden channel to paired ``c_fc``/``c_proj`` deltas."""
    if fc_base.shape != fc_delta.shape or proj_base.shape != proj_delta.shape:
        raise ValueError("base and displacement shapes must match")
    if fc_base.shape[0] != proj_base.shape[1] or fc_base.shape[1] != proj_base.shape[0]:
        raise ValueError("c_fc and c_proj must be transposed rectangular partners")
    fc_base = fc_base.float()
    fc_delta = fc_delta.float()
    proj_base = proj_base.float()
    proj_delta = proj_delta.float()
    numerator = (fc_base * fc_delta).sum(dim=1) + (proj_base * proj_delta).sum(dim=0)
    denominator = fc_base.square().sum(dim=1) + proj_base.square().sum(dim=0)
    gains = numerator / denominator.clamp_min(1e-30)
    projection_energy = (numerator.square() / denominator.clamp_min(1e-30)).sum()
    delta_energy = fc_delta.square().sum() + proj_delta.square().sum()
    fc_norm = fc_delta.norm(dim=1)
    proj_norm = proj_delta.norm(dim=0)
    centered_fc = fc_norm - fc_norm.mean()
    centered_proj = proj_norm - proj_norm.mean()
    pearson = (centered_fc * centered_proj).sum() / (
        centered_fc.norm() * centered_proj.norm()
    ).clamp_min(1e-30)
    return {
        "delta_energy": float(delta_energy),
        "projection_energy": float(projection_energy),
        "projection_energy_fraction": float(projection_energy / delta_energy.clamp_min(1e-30)),
        "gain_rms": float(gains.square().mean().sqrt()),
        "gain_abs_mean": float(gains.abs().mean()),
        "paired_delta_norm_pearson": float(pearson),
    }


def load_requested_snapshots(
    snapshot_dir: Path,
    requested_steps: list[int],
) -> tuple[dict[int, dict[str, torch.Tensor]], dict[str, Any]]:
    result: dict[int, dict[str, torch.Tensor]] = {}
    metadata: dict[str, Any] | None = None
    for step in requested_steps:
        path = snapshot_dir / f"step_{step:06d}.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != SCHEMA_VERSION or payload.get("step") != step:
            raise ValueError(f"trajectory snapshot identity mismatch: {path}")
        if metadata is None:
            metadata = {
                "run_identity": payload.get("run_identity"),
                "run_identity_sha256": payload.get("run_identity_sha256"),
                "execution_provenance": payload.get("execution_provenance"),
                "model_config": payload.get("model_config"),
            }
        result[step] = payload["parameters"]
    assert metadata is not None
    return result, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", default="0,60,120,180,238")
    parser.add_argument("--layers", default="0,3,6,9,11")
    parser.add_argument("--targets", default="mlp.c_fc,mlp.c_proj")
    parser.add_argument("--latent-ratios", default="0.01,0.02,0.04,0.08")
    parser.add_argument("--latent-rows", type=int, default=154)
    parser.add_argument("--block-fht-layers", type=int, default=2)
    parser.add_argument("--block-fht-seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    steps = parse_int_list(args.steps)
    if len(steps) < 2 or steps != sorted(set(steps)):
        raise ValueError("steps must contain at least two unique sorted values")
    layers = set(parse_int_list(args.layers))
    targets = {item for item in args.targets.split(",") if item}
    if not targets or not targets <= TARGET_SEED_OFFSETS.keys():
        raise ValueError(f"targets must be a non-empty subset of {sorted(TARGET_SEED_OFFSETS)}")
    ratios = parse_float_list(args.latent_ratios)
    snapshots, metadata = load_requested_snapshots(args.snapshot_dir, steps)

    tangent_rows: list[dict[str, Any]] = []
    radial_rows: list[dict[str, Any]] = []
    for start, stop in zip(steps[:-1], steps[1:], strict=True):
        start_parameters = snapshots[start]
        stop_parameters = snapshots[stop]
        for name, base_cpu in sorted(start_parameters.items()):
            match = PARAMETER_PATTERN.match(name)
            if match is None:
                continue
            layer = int(match.group("layer"))
            target = match.group("target")
            if layer not in layers or target not in targets:
                continue
            delta = (stop_parameters[name] - base_cpu).to(args.device)
            for ratio in ratios:
                latent_dim, latent_shape = resolved_latent_dim(
                    delta.numel(), ratio, args.latent_rows
                )
                seed = args.block_fht_seed + layer * 4 + TARGET_SEED_OFFSETS[target]
                tangent_rows.append(
                    {
                        "parameter": name,
                        "layer": layer,
                        "target": target,
                        "start_step": start,
                        "stop_step": stop,
                        "latent_ratio_requested": ratio,
                        "latent_ratio_resolved": latent_dim / delta.numel(),
                        "seed": seed,
                        **exact_block_fht_projection(
                            delta,
                            latent_dim=latent_dim,
                            latent_shape=latent_shape,
                            layers=args.block_fht_layers,
                            seed=seed,
                        ),
                    }
                )
            del delta

        for layer in sorted(layers):
            fc_name = f"transformer.h.{layer}.mlp.c_fc.weight"
            proj_name = f"transformer.h.{layer}.mlp.c_proj.weight"
            if not all(
                name in start_parameters and name in stop_parameters
                for name in (fc_name, proj_name)
            ):
                continue
            fc_base = start_parameters[fc_name].to(args.device)
            proj_base = start_parameters[proj_name].to(args.device)
            radial_rows.append(
                {
                    "layer": layer,
                    "start_step": start,
                    "stop_step": stop,
                    **shared_hidden_radial_projection(
                        fc_base,
                        stop_parameters[fc_name].to(args.device) - fc_base,
                        proj_base,
                        stop_parameters[proj_name].to(args.device) - proj_base,
                    ),
                }
            )
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "block_fht_tangent_overlap.csv", tangent_rows)
    write_csv(args.output / "shared_hidden_radial_overlap.csv", radial_rows)
    analysis_metadata = {
        **metadata,
        "snapshot_dir": str(args.snapshot_dir),
        "steps": steps,
        "layers": sorted(layers),
        "targets": sorted(targets),
        "latent_ratios": ratios,
        "latent_rows": args.latent_rows,
        "block_fht_layers": args.block_fht_layers,
        "block_fht_seed": args.block_fht_seed,
        "method": {
            "block_fht_tangent": (
                "exact Euclidean projection onto the repeated orthogonal linear "
                "BlockFHT image; requires matrix size divisible by padded latent block"
            ),
            "shared_hidden_radial": (
                "least-squares paired radial tangent diag(g)@c_fc and c_proj@diag(g)"
            ),
        },
        "limitations": [
            "Overlap with one dense optimizer displacement is not a solution-manifold dimension.",
            "A low tangent overlap can still train well if the constrained optimum follows another path.",
            "A high tangent overlap is not sufficient for good loss or activation compatibility.",
        ],
    }
    (args.output / "analysis_metadata.json").write_text(
        json.dumps(analysis_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "tangent_rows": len(tangent_rows),
                "radial_rows": len(radial_rows),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
