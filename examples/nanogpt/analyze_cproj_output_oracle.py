"""Measure whether output-side geometry can repair a generated MLP c_proj.

The diagnostic fits maps on the residual-width output of a generated c_proj
and evaluates them on held-out fixed validation windows.  It separates three
questions:

* can a norm-preserving output rotation align the generated update?
* is per-residual-channel scale/shear also required?
* how much is left even for an unconstrained residual-width linear map?

These are oracle fits, not trainable-model results.  They identify a useful
structure family before another smallest-rung training screen is registered.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from examples.nanogpt.analyze_cproj_functional_span import collect_layer_io
from examples.nanogpt.analyze_cproj_manifold import (
    effective_c_proj_weight,
    load_model,
)
from examples.nanogpt.analyze_residual_compatibility import fixed_validation_batches


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def left_orthogonal_procrustes(
    source: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Return Q minimizing ``||target - Q @ source||_F``."""

    cross = target.float() @ source.float().transpose(0, 1)
    left, _, right_h = torch.linalg.svd(cross, full_matrices=False)
    return left @ right_h


def functional_orthogonal_procrustes(
    source: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Return Q for row samples minimizing ``||target - source @ Q.T||_F``."""

    return left_orthogonal_procrustes(
        source.float().transpose(0, 1),
        target.float().transpose(0, 1),
    )


def optimal_scalar(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    denominator = source.float().square().sum().clamp_min(1e-30)
    return (source.float() * target.float()).sum() / denominator


def optimal_output_diagonal(
    source: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Fit target ~= source * diagonal for row-sample matrices."""

    denominator = source.float().square().sum(dim=0).clamp_min(1e-30)
    return (source.float() * target.float()).sum(dim=0) / denominator


def fit_diagonal_then_orthogonal(
    source: torch.Tensor,
    target: torch.Tensor,
    iterations: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Alternating fit for ``target ~= (source * d) @ Q.T``."""

    diagonal = torch.ones(
        source.shape[-1], device=source.device, dtype=torch.float32
    )
    rotation = torch.eye(
        source.shape[-1], device=source.device, dtype=torch.float32
    )
    source = source.float()
    target = target.float()
    for _ in range(int(iterations)):
        rotation = functional_orthogonal_procrustes(
            source * diagonal, target
        )
        rotated_target = target @ rotation
        diagonal = optimal_output_diagonal(source, rotated_target)
    return diagonal, rotation


def fit_full_linear(
    source: torch.Tensor,
    target: torch.Tensor,
    ridge_fraction: float = 1e-6,
) -> torch.Tensor:
    """Fit target ~= source @ A.T with a scale-relative ridge."""

    source = source.float()
    target = target.float()
    gram = source.transpose(0, 1) @ source
    ridge = (
        torch.trace(gram) / max(gram.shape[0], 1) * float(ridge_fraction)
    )
    rhs = source.transpose(0, 1) @ target
    # solve returns A.T in the row-sample convention.
    return torch.linalg.solve(
        gram + ridge * torch.eye(gram.shape[0], device=gram.device),
        rhs,
    ).transpose(0, 1)


def metrics(target: torch.Tensor, prediction: torch.Tensor) -> dict[str, float]:
    target = target.float()
    prediction = prediction.float()
    residual = target - prediction
    target_energy = target.square().sum().clamp_min(1e-30)
    denominator = target.norm() * prediction.norm()
    return {
        "explained_target_energy": float(
            1.0 - residual.square().sum() / target_energy
        ),
        "cosine": float(
            (target * prediction).sum() / denominator.clamp_min(1e-30)
        ),
        "target_rms": float(target.square().mean().sqrt()),
        "prediction_rms": float(prediction.square().mean().sqrt()),
        "residual_rms": float(residual.square().mean().sqrt()),
    }


def fit_oracles(
    source_train: torch.Tensor,
    target_train: torch.Tensor,
    source_holdout: torch.Tensor,
    target_holdout: torch.Tensor,
) -> list[dict[str, float | str]]:
    source_train = source_train.float()
    target_train = target_train.float()
    source_holdout = source_holdout.float()
    target_holdout = target_holdout.float()

    rotation = functional_orthogonal_procrustes(source_train, target_train)
    rotated_train = source_train @ rotation.transpose(0, 1)
    rotated_holdout = source_holdout @ rotation.transpose(0, 1)
    scalar = optimal_scalar(rotated_train, target_train)
    diagonal = optimal_output_diagonal(source_train, target_train)
    alt_diagonal, alt_rotation = fit_diagonal_then_orthogonal(
        source_train, target_train
    )
    full = fit_full_linear(source_train, target_train)

    predictions = {
        "identity": (source_train, source_holdout),
        "orthogonal": (rotated_train, rotated_holdout),
        "scalar_orthogonal": (
            scalar * rotated_train,
            scalar * rotated_holdout,
        ),
        "output_diagonal": (
            source_train * diagonal,
            source_holdout * diagonal,
        ),
        "diagonal_then_orthogonal": (
            (source_train * alt_diagonal) @ alt_rotation.transpose(0, 1),
            (source_holdout * alt_diagonal)
            @ alt_rotation.transpose(0, 1),
        ),
        "full_linear": (
            source_train @ full.transpose(0, 1),
            source_holdout @ full.transpose(0, 1),
        ),
    }
    rows: list[dict[str, float | str]] = []
    for family, (train_prediction, holdout_prediction) in predictions.items():
        rows.append(
            {
                "family": family,
                **{f"train_{key}": value for key, value in metrics(
                    target_train, train_prediction
                ).items()},
                **{f"holdout_{key}": value for key, value in metrics(
                    target_holdout, holdout_prediction
                ).items()},
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    rows: list[dict[str, object]]
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for family in sorted({str(row["family"]) for row in rows}):
        selected = [row for row in rows if row["family"] == family]
        output[family] = {
            key: float(np.mean([float(row[key]) for row in selected]))
            for key in selected[0]
            if key not in {"layer", "family"}
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention-only", required=True, type=Path)
    parser.add_argument("--plain-cproj", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", default="0,3,6,9,11")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--sample-cap", type=int, default=4096)
    parser.add_argument("--sample-seed", type=int, default=20260716)
    parser.add_argument("--holdout-sample-seed", type=int, default=20260717)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    layers = [int(part) for part in args.layers.split(",") if part]
    train_batches = fixed_validation_batches(
        args.data_dir,
        args.batch_size,
        args.block_size,
        args.batches,
        args.sample_seed,
    )
    holdout_batches = fixed_validation_batches(
        args.data_dir,
        args.batch_size,
        args.block_size,
        args.batches,
        args.holdout_sample_seed,
    )
    print("collecting fit-window activations", flush=True)
    attention_train = collect_layer_io(
        args.attention_only, train_batches, layers, args.sample_cap, args.device
    )
    plain_train = collect_layer_io(
        args.plain_cproj, train_batches, layers, args.sample_cap, args.device
    )
    print("collecting held-out activations", flush=True)
    attention_holdout = collect_layer_io(
        args.attention_only,
        holdout_batches,
        layers,
        args.sample_cap,
        args.device,
    )
    plain_holdout = collect_layer_io(
        args.plain_cproj,
        holdout_batches,
        layers,
        args.sample_cap,
        args.device,
    )

    rows: list[dict[str, object]] = []
    for layer in layers:
        layer_rows = fit_oracles(
            plain_train[(layer, "mlp_out")].to(args.device),
            attention_train[(layer, "mlp_out")].to(args.device),
            plain_holdout[(layer, "mlp_out")].to(args.device),
            attention_holdout[(layer, "mlp_out")].to(args.device),
        )
        rows.extend({"layer": layer, **row} for row in layer_rows)

    print("fitting endpoint weight oracles", flush=True)
    attention_model = load_model(args.attention_only, args.device)
    plain_model = load_model(args.plain_cproj, args.device)
    weight_rows: list[dict[str, object]] = []
    try:
        for layer in layers:
            source = effective_c_proj_weight(plain_model, layer)
            target = effective_c_proj_weight(attention_model, layer)
            # Treat matrix columns as samples so the same output-side fit
            # operates on c_proj residual coordinates.
            fitted = fit_oracles(
                source.transpose(0, 1),
                target.transpose(0, 1),
                source.transpose(0, 1),
                target.transpose(0, 1),
            )
            weight_rows.extend({"layer": layer, **row} for row in fitted)
    finally:
        del attention_model, plain_model
        if "cuda" in args.device:
            torch.cuda.empty_cache()

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "functional_output_oracles.csv", rows)
    write_csv(args.output / "weight_output_oracles.csv", weight_rows)
    metadata = {
        "schema_version": "cproj_output_oracle_v1",
        "attention_only": {
            "path": str(args.attention_only),
            "sha256": sha256(args.attention_only),
        },
        "plain_cproj": {
            "path": str(args.plain_cproj),
            "sha256": sha256(args.plain_cproj),
        },
        "data_dir": str(args.data_dir),
        "layers": layers,
        "sample_cap": args.sample_cap,
        "sample_seed": args.sample_seed,
        "holdout_sample_seed": args.holdout_sample_seed,
        "functional_summary": summarize(rows),
        "weight_summary": summarize(weight_rows),
    }
    (args.output / "cproj_output_oracle_summary.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
