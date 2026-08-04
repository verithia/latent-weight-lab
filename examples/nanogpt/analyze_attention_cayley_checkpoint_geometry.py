#!/usr/bin/env python3
"""Measure learned-basis drift in attention Cayley checkpoints.

The attention Cayley chart is initialized with a zero left factor and a
seeded right frame.  Both factors are currently trainable.  This tool
reconstructs the exact seeded right frame without instantiating the GPT,
then measures whether training preserved that frame or converted the chart
into a moving low-rank basis.  It accepts both flattened AdamW factors and
matrix-shaped Muon factors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


MODULE_PATTERN = re.compile(
    r"^(?:_orig_mod\.)?transformer\.h\.(?P<layer>\d+)\.attn\."
    r"(?P<module>qk_input_cayley|qk_output_cayley|v_input_cayley|"
    r"v_output_cayley|cproj_input_cayley|cproj_output_cayley)\.left$"
)

MODULE_METADATA = {
    "qk_input_cayley": ("qk_shared", "attn.c_attn.qk_headwise", 0),
    "qk_output_cayley": ("qk_shared", "attn.c_attn.qk_headwise", 3),
    "v_input_cayley": ("v", "attn.c_attn.v", 1),
    "v_output_cayley": ("v", "attn.c_attn.v", 4),
    "cproj_input_cayley": ("cproj", "attn.c_proj", 2),
    "cproj_output_cayley": ("cproj", "attn.c_proj", 5),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def seeded_right_frame(features: int, rank: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return F.normalize(
        torch.randn(features, rank, generator=generator, dtype=torch.float32),
        dim=0,
    )


def orthonormal_frame(matrix: torch.Tensor) -> torch.Tensor:
    matrix = matrix.double()
    if matrix.ndim != 2 or matrix.shape[0] < matrix.shape[1]:
        raise ValueError("frame must be a tall rank-2 tensor")
    if int(torch.linalg.matrix_rank(matrix)) < matrix.shape[1]:
        raise ValueError("frame is rank deficient")
    return torch.linalg.qr(matrix, mode="reduced").Q


def cayley_generator(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.double()
    right = F.normalize(right.double(), dim=0)
    return left @ right.T - right @ left.T


def low_rank_skew_singular_values(
    left: torch.Tensor, right: torch.Tensor
) -> torch.Tensor:
    """Return the nonzero singular values of ``L R^T - R L^T``.

    The calculation stays in a ``2r`` coordinate system, so checkpoint
    analysis remains cheap even for the 1536-channel QK output chart.
    """
    left = left.double()
    right = F.normalize(right.double(), dim=0)
    factors = torch.cat((left, right), dim=1)
    reduced = torch.linalg.qr(factors, mode="reduced").R
    rank = left.shape[1]
    identity = torch.eye(rank, dtype=left.dtype)
    zero = torch.zeros_like(identity)
    symplectic = torch.cat(
        (torch.cat((zero, identity), dim=1), torch.cat((-identity, zero), dim=1)),
        dim=0,
    )
    return torch.linalg.svdvals(reduced @ symplectic @ reduced.T)


def low_rank_skew_difference_fro(
    *,
    left: torch.Tensor,
    first_right: torch.Tensor,
    second_right: torch.Tensor,
) -> float:
    """Return ``||K(L,R1)-K(L,R2)||_F`` without dense materialization."""
    left = left.double()
    first = F.normalize(first_right.double(), dim=0)
    second = F.normalize(second_right.double(), dim=0)
    factors = torch.cat((left, first, left, second), dim=1)
    reduced = torch.linalg.qr(factors, mode="reduced").R
    rank = left.shape[1]
    identity = torch.eye(rank, dtype=left.dtype)
    zero = torch.zeros_like(identity)
    symplectic = torch.cat(
        (torch.cat((zero, identity), dim=1), torch.cat((-identity, zero), dim=1)),
        dim=0,
    )
    block = torch.block_diag(symplectic, -symplectic)
    return float((reduced @ block @ reduced.T).norm())


def frame_drift_metrics(
    *,
    initial_right: torch.Tensor,
    final_right: torch.Tensor,
    final_left: torch.Tensor,
) -> dict[str, float]:
    initial = initial_right.double()
    final = final_right.double()
    left = final_left.double()
    q_initial = orthonormal_frame(initial)
    q_final = orthonormal_frame(final)
    canonical_cosines = torch.linalg.svdvals(q_initial.T @ q_final).clamp(0.0, 1.0)
    principal_angles = torch.rad2deg(torch.acos(canonical_cosines))
    mean_squared_cosine = canonical_cosines.square().mean()
    normalized_initial = F.normalize(initial, dim=0)
    normalized_final = F.normalize(final, dim=0)
    generator_singular_values = low_rank_skew_singular_values(left, final)
    generator_norm = generator_singular_values.norm().clamp_min(1e-30)
    projected_left = left - q_final @ (q_final.T @ left)
    left_norm = left.norm().clamp_min(1e-30)
    generator_singular_max = generator_singular_values.max()
    return {
        "right_raw_relative_drift": float((final - initial).norm() / initial.norm()),
        "right_normalized_relative_drift": float(
            (normalized_final - normalized_initial).norm()
            / normalized_initial.norm()
        ),
        "right_subspace_mean_squared_cosine": float(mean_squared_cosine),
        "right_subspace_projector_distance": float(
            torch.sqrt((1.0 - mean_squared_cosine).clamp_min(0.0))
        ),
        "right_subspace_mean_angle_degrees": float(principal_angles.mean()),
        "right_subspace_max_angle_degrees": float(principal_angles.max()),
        "right_column_norm_mean": float(final.norm(dim=0).mean()),
        "right_column_norm_std": float(final.norm(dim=0).std(unbiased=False)),
        "left_fro": float(left.norm()),
        "left_effective_fraction": float(projected_left.norm() / left_norm),
        "generator_fro": float(generator_norm),
        "generator_spectral": float(generator_singular_max),
        "maximum_cayley_angle_degrees": float(
            2.0 * torch.rad2deg(torch.atan(generator_singular_max))
        ),
        "fixed_right_generator_relative_error": float(
            low_rank_skew_difference_fro(
                left=left,
                first_right=final,
                second_right=initial,
            )
            / generator_norm
        ),
    }


def _rank_for_module(model_config: dict[str, Any], rank_key: str) -> int:
    ranks = model_config.get("block_fht_attn_cayley_ranks") or {}
    return int(ranks.get(rank_key, model_config["block_fht_attn_cayley_rank"]))


def analyze_checkpoint(path: Path, label: str) -> dict[str, Any]:
    started = time.time()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model")
    model_config = checkpoint.get("model_config")
    if not isinstance(state, dict) or not isinstance(model_config, dict):
        raise ValueError("checkpoint lacks model/model_config dictionaries")
    base_seed = int(model_config["block_fht_attn_cayley_seed"])
    rows: list[dict[str, Any]] = []
    for left_key, left_tensor in state.items():
        match = MODULE_PATTERN.match(left_key)
        if match is None:
            continue
        module = match.group("module")
        family, rank_key, seed_offset = MODULE_METADATA[module]
        rank = _rank_for_module(model_config, rank_key)
        layer = int(match.group("layer"))
        right_key = left_key[:-4] + "right"
        if right_key not in state:
            raise ValueError(f"missing right factor for {left_key}")
        left = left_tensor.detach().float().reshape(-1, rank)
        right = state[right_key].detach().float().reshape(-1, rank)
        if left.shape != right.shape:
            raise ValueError(f"factor shape mismatch for {left_key}")
        initial = seeded_right_frame(
            left.shape[0], rank, base_seed + layer * 64 + seed_offset
        )
        rows.append(
            {
                "parameter": left_key.removeprefix("_orig_mod."),
                "layer": layer,
                "module": module,
                "target_family": family,
                "features": int(left.shape[0]),
                "rank": rank,
                **frame_drift_metrics(
                    initial_right=initial,
                    final_right=right,
                    final_left=left,
                ),
            }
        )
    if not rows:
        raise ValueError("checkpoint contains no attention Cayley factors")

    metric_names = [
        name
        for name in rows[0]
        if name
        not in {"parameter", "layer", "module", "target_family", "features", "rank"}
    ]

    def aggregate(selected: list[dict[str, Any]]) -> dict[str, float]:
        return {
            name: sum(float(row[name]) for row in selected) / len(selected)
            for name in metric_names
        }

    families = sorted({str(row["target_family"]) for row in rows})
    return {
        "label": label,
        "checkpoint": str(path),
        "checkpoint_sha256": file_sha256(path),
        "next_iter": int(checkpoint.get("next_iter", -1)),
        "factor_optimizer": model_config.get(
            "block_fht_attn_cayley_factor_optimizer", "adamw"
        ),
        "module_count": len(rows),
        "aggregate": aggregate(rows),
        "by_target_family": {
            family: aggregate(
                [row for row in rows if row["target_family"] == family]
            )
            for family in families
        },
        "modules": rows,
        "elapsed_seconds": time.time() - started,
    }


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be LABEL=CHECKPOINT")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("--run must be LABEL=CHECKPOINT")
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=parse_run,
        help="LABEL=CHECKPOINT; repeat to compare runs",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    runs = [analyze_checkpoint(path, label) for label, path in args.run]
    result = {
        "schema_version": "attention_cayley_checkpoint_geometry_v1",
        "created_at_unix": time.time(),
        "source_commit": git_commit(Path(__file__).resolve().parents[2]),
        "runs": runs,
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
