#!/usr/bin/env python3
"""Compare terminal parameter geometry for the matched pre-GELU causal pair."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Iterable

import torch


GROUP_SUFFIXES = {
    "c_fc_weight": (".mlp.c_fc.weight",),
    "cproj_latent": (".mlp.c_proj.generator.latent",),
    "postgelu_hidden_coordinates": (
        ".mlp.hidden_block_rotation.coordinates",
    ),
    "postgelu_output_coordinates": (
        ".mlp.output_block_rotation.coordinates",
    ),
}
PREGELU_SUFFIX = ".mlp.pregelu_block_rotation.coordinates"


def matches(name: str, suffixes: Iterable[str]) -> bool:
    return any(name.endswith(suffix) for suffix in suffixes)


def paired_stats(
    parent: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    suffixes: tuple[str, ...],
) -> dict[str, float | int]:
    names = sorted(
        name
        for name in parent
        if matches(name, suffixes) and name in candidate
    )
    if not names:
        raise ValueError(f"no shared tensors matched {suffixes}")
    count = 0
    parent_square = 0.0
    candidate_square = 0.0
    difference_square = 0.0
    dot = 0.0
    max_abs_difference = 0.0
    for name in names:
        left = parent[name].detach().float()
        right = candidate[name].detach().float()
        if left.shape != right.shape:
            raise ValueError(f"shape mismatch for {name}")
        difference = right - left
        count += left.numel()
        parent_square += float(left.square().sum().item())
        candidate_square += float(right.square().sum().item())
        difference_square += float(difference.square().sum().item())
        dot += float((left * right).sum().item())
        max_abs_difference = max(
            max_abs_difference,
            float(difference.abs().max().item()),
        )
    denominator = math.sqrt(parent_square * candidate_square)
    return {
        "tensor_count": len(names),
        "coordinate_count": count,
        "parent_rms": math.sqrt(parent_square / count),
        "candidate_rms": math.sqrt(candidate_square / count),
        "difference_rms": math.sqrt(difference_square / count),
        "difference_over_parent_norm": (
            math.sqrt(difference_square / parent_square)
            if parent_square
            else math.inf
        ),
        "cosine": dot / denominator if denominator else math.nan,
        "max_abs_difference": max_abs_difference,
    }


def pregelu_stats(
    candidate: dict[str, torch.Tensor],
) -> dict[str, object]:
    names = sorted(
        name for name in candidate if name.endswith(PREGELU_SUFFIX)
    )
    if not names:
        raise ValueError("candidate has no pre-GELU coordinates")
    per_layer = []
    total_count = 0
    total_square = 0.0
    total_maximum = 0.0
    for name in names:
        values = candidate[name].detach().float()
        count = values.numel()
        square = float(values.square().sum().item())
        maximum = float(values.abs().max().item())
        total_count += count
        total_square += square
        total_maximum = max(total_maximum, maximum)
        per_layer.append(
            {
                "name": name,
                "coordinate_count": count,
                "rms": math.sqrt(square / count),
                "max_abs": maximum,
            }
        )
    return {
        "tensor_count": len(names),
        "coordinate_count": total_count,
        "rms": math.sqrt(total_square / total_count),
        "max_abs": total_maximum,
        "per_layer": per_layer,
    }


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".part",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    parent_checkpoint = torch.load(
        args.parent, map_location="cpu", weights_only=False
    )
    candidate_checkpoint = torch.load(
        args.candidate, map_location="cpu", weights_only=False
    )
    parent = parent_checkpoint["model"]
    candidate = candidate_checkpoint["model"]
    result: dict[str, object] = {
        "schema_version": "mai_124m_pregelu_causal_pair_geometry_v1",
        "parent_checkpoint": str(args.parent.resolve()),
        "candidate_checkpoint": str(args.candidate.resolve()),
        "parent_next_iter": int(parent_checkpoint["next_iter"]),
        "candidate_next_iter": int(candidate_checkpoint["next_iter"]),
        "paired_groups": {
            group: paired_stats(parent, candidate, suffixes)
            for group, suffixes in GROUP_SUFFIXES.items()
        },
        "candidate_pregelu_coordinates": pregelu_stats(candidate),
    }
    atomic_write_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
