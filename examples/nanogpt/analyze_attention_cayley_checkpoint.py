from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


KEY_PATTERN = re.compile(
    r"^transformer\.h\.(?P<layer>\d+)\.attn\."
    r"(?P<target>qk_input_cayley|v_input_cayley|cproj_input_cayley)\.right$"
)
TARGET_SEED_OFFSET = {
    "qk_input_cayley": 0,
    "v_input_cayley": 1,
    "cproj_input_cayley": 2,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seeded_right_frame(
    features: int,
    rank: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    frame = torch.randn(
        features,
        rank,
        generator=generator,
        dtype=torch.float32,
    )
    return F.normalize(frame, dim=0)


def frame_motion_metrics(
    initial_right: torch.Tensor,
    current_right: torch.Tensor,
    current_left: torch.Tensor,
) -> dict[str, float]:
    current_right = F.normalize(current_right.float(), dim=0)
    initial_right = F.normalize(initial_right.float(), dim=0)
    current_left = current_left.float()
    initial_basis = torch.linalg.qr(initial_right, mode="reduced").Q
    current_basis = torch.linalg.qr(current_right, mode="reduced").Q
    principal_cosines = torch.linalg.svdvals(
        initial_basis.transpose(0, 1) @ current_basis
    ).clamp(0.0, 1.0)
    left_projection = initial_basis @ (
        initial_basis.transpose(0, 1) @ current_left
    )
    left_norm_sq = current_left.square().sum()
    outside_fraction = (
        (current_left - left_projection).square().sum()
        / left_norm_sq.clamp_min(torch.finfo(torch.float32).tiny)
    )
    skew = (
        current_left @ current_right.transpose(0, 1)
        - current_right @ current_left.transpose(0, 1)
    )
    return {
        "right_principal_cosine_mean": float(principal_cosines.mean()),
        "right_principal_cosine_min": float(principal_cosines.min()),
        "right_chordal_distance": float(
            (len(principal_cosines) - principal_cosines.square().sum())
            .clamp_min(0.0)
            .sqrt()
        ),
        "left_norm": float(current_left.norm()),
        "left_energy_outside_initial_right": float(outside_fraction),
        "skew_frobenius_norm": float(skew.norm()),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["target"])].append(row)
        grouped["all"].append(row)
    result: dict[str, dict[str, float]] = {}
    metric_names = (
        "right_principal_cosine_mean",
        "right_principal_cosine_min",
        "right_chordal_distance",
        "left_norm",
        "left_energy_outside_initial_right",
        "skew_frobenius_norm",
    )
    for target, target_rows in grouped.items():
        result[target] = {
            metric: sum(float(row[metric]) for row in target_rows)
            / len(target_rows)
            for metric in metric_names
        }
    return result


def analyze(checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state = checkpoint["model"]
    config = checkpoint["model_config"]
    features = int(config["n_embd"])
    rank = int(config["block_fht_attn_cayley_rank"])
    base_seed = int(config["block_fht_attn_cayley_seed"])
    coordinate_scale = float(config["block_fht_attn_cayley_scale"])
    rows: list[dict[str, Any]] = []
    for right_key, right_flat in state.items():
        match = KEY_PATTERN.match(right_key)
        if match is None:
            continue
        layer = int(match.group("layer"))
        target = match.group("target")
        left_key = right_key.removesuffix(".right") + ".left"
        left = (
            coordinate_scale
            * state[left_key].reshape(features, rank).float()
        )
        right = right_flat.reshape(features, rank).float()
        seed = base_seed + layer * 64 + TARGET_SEED_OFFSET[target]
        metrics = frame_motion_metrics(
            seeded_right_frame(features, rank, seed),
            right,
            left,
        )
        rows.append(
            {
                "layer": layer,
                "target": target,
                "seed": seed,
                **metrics,
            }
        )
    if not rows:
        raise ValueError("checkpoint contains no attention Cayley factors")
    return {
        "schema_version": "attention_cayley_checkpoint_analysis_v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "next_iter": checkpoint.get("next_iter"),
        "features": features,
        "rank": rank,
        "coordinate_scale": coordinate_scale,
        "cells": len(rows),
        "summary": summarize(rows),
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze(args.checkpoint)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
