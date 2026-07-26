from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from examples.nanogpt.analyze_cproj_manifold import (
    effective_c_proj_weight,
    load_model,
    spectral_residual_weight,
)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.reshape(-1).float()
    b_flat = b.reshape(-1).float()
    denominator = a_flat.norm() * b_flat.norm()
    if denominator <= 0:
        return float("nan")
    return float(torch.dot(a_flat, b_flat) / denominator)


def paired_basis_projection(
    target: torch.Tensor,
    in_basis: torch.Tensor,
    out_basis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Least-squares projection onto span{u_k v_k^T}.

    The fixed FHT-derived columns are not exactly orthonormal after the
    non-power-of-two crop.  The Hadamard product of their two Gram matrices is
    therefore the exact Gram matrix of the paired rank-one basis atoms.
    """

    target = target.float()
    in_basis = in_basis.float()
    out_basis = out_basis.float()
    if target.shape != (out_basis.shape[0], in_basis.shape[0]):
        raise ValueError(
            "target shape must be [out_features, in_features], got "
            f"{tuple(target.shape)} for bases {tuple(out_basis.shape)} and "
            f"{tuple(in_basis.shape)}"
        )
    if in_basis.shape[1] != out_basis.shape[1]:
        raise ValueError("input and output bases must have the same rank")

    in_gram = in_basis.transpose(0, 1) @ in_basis
    out_gram = out_basis.transpose(0, 1) @ out_basis
    atom_gram = in_gram * out_gram
    rhs = torch.diagonal(out_basis.transpose(0, 1) @ target @ in_basis)
    coefficients = torch.linalg.pinv(atom_gram, rtol=1e-7, atol=1e-10) @ rhs
    projection = (out_basis * coefficients.unsqueeze(0)) @ in_basis.transpose(0, 1)
    return coefficients, projection, atom_gram


def correction_alignment_metrics(
    target: torch.Tensor,
    in_basis: torch.Tensor,
    out_basis: torch.Tensor,
    learned: torch.Tensor,
) -> dict[str, float]:
    coefficients, optimum, atom_gram = paired_basis_projection(target, in_basis, out_basis)
    learned = learned.float()
    target = target.float()

    target_energy = target.square().sum().clamp_min(1e-30)
    optimum_error = (target - optimum).square().sum()
    learned_error = (target - learned).square().sum()
    singular = torch.linalg.svdvals(atom_gram)
    positive = singular[singular > singular.max().clamp_min(1e-30) * 1e-7]
    condition = (
        float(positive.max() / positive.min())
        if positive.numel() > 0
        else float("inf")
    )

    learned_coefficients = torch.diagonal(
        out_basis.transpose(0, 1) @ learned @ in_basis
    )
    learned_coefficients = torch.linalg.pinv(
        atom_gram, rtol=1e-7, atol=1e-10
    ) @ learned_coefficients

    return {
        "target_fro_norm": float(target.norm()),
        "optimal_fro_norm": float(optimum.norm()),
        "optimal_coeff_l2": float(coefficients.norm()),
        "explainable_energy_fraction": float(
            1.0 - optimum_error / target_energy
        ),
        "unexplained_residual_fraction": float(optimum_error / target_energy),
        "learned_fro_norm": float(learned.norm()),
        "learned_to_target_norm": float(learned.norm() / target.norm().clamp_min(1e-30)),
        "learned_to_optimal_norm": float(
            learned.norm() / optimum.norm().clamp_min(1e-30)
        ),
        "learned_target_cosine": cosine(learned, target),
        "learned_optimal_cosine": cosine(learned, optimum),
        "learned_coefficient_cosine": cosine(learned_coefficients, coefficients),
        "learned_error_reduction_fraction": float(
            1.0 - learned_error / target_energy
        ),
        "atom_gram_condition": condition,
        "atom_gram_rank": float(torch.linalg.matrix_rank(atom_gram)),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for candidate in sorted({str(row["candidate"]) for row in rows}):
        selected = [row for row in rows if row["candidate"] == candidate]
        metrics: dict[str, float] = {}
        for key, value in selected[0].items():
            if key in {"candidate", "layer"} or not isinstance(value, (int, float)):
                continue
            values = np.asarray([float(row[key]) for row in selected], dtype=np.float64)
            metrics[f"{key}_mean"] = float(np.mean(values))
        summary[candidate] = metrics
    return summary


def candidate_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must be NAME=CHECKPOINT")
    name, checkpoint = value.split("=", 1)
    if not name or not checkpoint:
        raise argparse.ArgumentTypeError("candidate must be NAME=CHECKPOINT")
    return name, Path(checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention-only", required=True, type=Path)
    parser.add_argument("--plain-cproj", required=True, type=Path)
    parser.add_argument("--candidate", action="append", required=True, type=candidate_arg)
    parser.add_argument("--layers", default="3,6,9")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    layers = [int(part) for part in args.layers.split(",")]
    attention = load_model(args.attention_only, args.device)
    plain = load_model(args.plain_cproj, args.device)
    target_by_layer = {
        layer: effective_c_proj_weight(attention, layer)
        - effective_c_proj_weight(plain, layer)
        for layer in layers
    }
    del attention, plain
    if "cuda" in args.device:
        torch.cuda.empty_cache()

    rows: list[dict[str, object]] = []
    for candidate, checkpoint in args.candidate:
        model = load_model(checkpoint, args.device)
        for layer in layers:
            mlp = model.transformer.h[layer].mlp
            learned = spectral_residual_weight(model, layer)
            if learned is None:
                raise ValueError(f"{candidate} layer {layer} has no spectral residual")
            metrics = correction_alignment_metrics(
                target_by_layer[layer],
                mlp.cproj_spectral_resid_in_basis,
                mlp.cproj_spectral_resid_out_basis,
                learned,
            )
            rows.append({"candidate": candidate, "layer": layer, **metrics})
        del model
        if "cuda" in args.device:
            torch.cuda.empty_cache()

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "cproj_correction_alignment.csv", rows)
    summary = summarize(rows)
    (args.output / "cproj_correction_alignment_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
