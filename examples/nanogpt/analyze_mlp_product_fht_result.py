"""Diagnose a completed product-FHT c_proj screen against dense endpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from latent_weight_lab import ProductFHTLinear


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def spectral_metrics(weight: torch.Tensor) -> dict[str, float]:
    weight = weight.float()
    gram = weight @ weight.T
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0).flip(0)
    total = eigenvalues.sum().clamp_min(1e-30)
    probability = eigenvalues / total
    positive = probability > 0
    effective_rank = torch.exp(
        -(probability[positive] * probability[positive].log()).sum()
    )
    diagonal = torch.diag_embed(torch.diagonal(gram))
    row_norm = weight.square().sum(dim=1).sqrt()
    column_norm = weight.square().sum(dim=0).sqrt()
    return {
        "frobenius_norm": float(weight.norm()),
        "top1_energy_fraction": float(eigenvalues[:1].sum() / total),
        "top8_energy_fraction": float(eigenvalues[:8].sum() / total),
        "top64_energy_fraction": float(eigenvalues[:64].sum() / total),
        "stable_rank": float(total / eigenvalues[0].clamp_min(1e-30)),
        "effective_rank": float(effective_rank),
        "condition_number": float(
            torch.sqrt(
                eigenvalues[0]
                / eigenvalues[-1].clamp_min(1e-30)
            )
        ),
        "row_gram_offdiagonal_fraction": float(
            (gram - diagonal).norm() / gram.norm().clamp_min(1e-30)
        ),
        "row_norm_cv": float(
            row_norm.std() / row_norm.mean().clamp_min(1e-30)
        ),
        "column_norm_cv": float(
            column_norm.std() / column_norm.mean().clamp_min(1e-30)
        ),
    }


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        torch.sum(left * right)
        / (left.norm() * right.norm()).clamp_min(1e-30)
    )


def average(records: list[dict[str, Any]], key: str) -> float:
    return sum(float(record[key]) for record in records) / len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product-checkpoint", required=True, type=Path)
    parser.add_argument("--dense-start", required=True, type=Path)
    parser.add_argument("--dense-end", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoint = torch.load(
        args.product_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    dense_start_payload = torch.load(
        args.dense_start,
        map_location="cpu",
        weights_only=False,
    )
    dense_end_payload = torch.load(
        args.dense_end,
        map_location="cpu",
        weights_only=False,
    )
    state = checkpoint["model"]
    dense_start = dense_start_payload["parameters"]
    dense_end = dense_end_payload["parameters"]
    config = checkpoint["model_config"]
    factors = int(config["block_fht_cproj_product_fht_factors"])
    n_embd = int(config["n_embd"])
    target_std = 0.02 / (2 * int(config["n_layer"])) ** 0.5

    rows: list[dict[str, Any]] = []
    for layer in range(int(config["n_layer"])):
        prefix = f"transformer.h.{layer}.mlp.c_proj."
        module = ProductFHTLinear(
            4 * n_embd,
            n_embd,
            factors=factors,
            seed=int(config["block_fht_seed"]) + layer * 4 + 3,
            weight_std=target_std,
            diagonal_scale=float(
                config[
                    "block_fht_cproj_product_fht_diagonal_scale"
                ]
            ),
            weight_space_muon=bool(
                config[
                    "block_fht_cproj_product_fht_weight_space_muon"
                ]
            ),
            muon_momentum=float(
                config["block_fht_cproj_product_fht_muon_momentum"]
            ),
            muon_ns_steps=int(
                config["block_fht_cproj_product_fht_muon_ns_steps"]
            ),
            natural_gradient=bool(
                config[
                    "block_fht_cproj_product_fht_natural_gradient"
                ]
            ),
        ).to(args.device)
        with torch.no_grad():
            module.product_log_diagonals.copy_(
                state[prefix + "product_log_diagonals"].to(args.device)
            )
            module.product_output_log_gain.copy_(
                state[prefix + "product_output_log_gain"].to(args.device)
            )
            module.weight_space_momentum_buffer.copy_(
                state[
                    prefix + "weight_space_momentum_buffer"
                ].to(args.device)
            )
            product_final = module.weight.detach().float()
            final_diagonals = module.product_log_diagonals.detach().float()
            final_output_gain = (
                module.product_output_log_gain.detach().float()
            )
            module.product_log_diagonals.zero_()
            module.product_output_log_gain.zero_()
            product_initial = module.weight.detach().float()

        dense_key = f"transformer.h.{layer}.mlp.c_proj.weight"
        dense_initial = dense_start[dense_key].to(
            args.device, dtype=torch.float32
        )
        dense_final = dense_end[dense_key].to(
            args.device, dtype=torch.float32
        )
        product_delta = product_final - product_initial
        dense_delta = dense_final - dense_initial
        row = {
            "layer": layer,
            "product_relative_displacement": float(
                product_delta.norm()
                / product_initial.norm().clamp_min(1e-30)
            ),
            "dense_relative_displacement": float(
                dense_delta.norm()
                / dense_initial.norm().clamp_min(1e-30)
            ),
            "product_dense_delta_cosine": cosine(
                product_delta, dense_delta
            ),
            "product_final_dense_final_cosine": cosine(
                product_final, dense_final
            ),
            "product_final_dense_final_relative_error": float(
                (product_final - dense_final).norm()
                / dense_final.norm().clamp_min(1e-30)
            ),
            "factor_log_diagonal_mean": float(
                final_diagonals.mean()
            ),
            "factor_log_diagonal_std": float(
                final_diagonals.std()
            ),
            "factor_log_diagonal_max_abs": float(
                final_diagonals.abs().max()
            ),
            "output_log_gain_mean": float(final_output_gain.mean()),
            "output_log_gain_std": float(final_output_gain.std()),
            "weight_space_momentum_norm": float(
                state[
                    prefix + "weight_space_momentum_buffer"
                ].float().norm()
            ),
            "product_initial_spectrum": spectral_metrics(
                product_initial
            ),
            "product_final_spectrum": spectral_metrics(product_final),
            "dense_initial_spectrum": spectral_metrics(dense_initial),
            "dense_final_spectrum": spectral_metrics(dense_final),
        }
        rows.append(row)

    aggregate: dict[str, Any] = {
        "layers": len(rows),
        "product_relative_displacement_mean": average(
            rows, "product_relative_displacement"
        ),
        "dense_relative_displacement_mean": average(
            rows, "dense_relative_displacement"
        ),
        "product_dense_delta_cosine_mean": average(
            rows, "product_dense_delta_cosine"
        ),
        "product_final_dense_final_cosine_mean": average(
            rows, "product_final_dense_final_cosine"
        ),
        "product_final_dense_final_relative_error_mean": average(
            rows, "product_final_dense_final_relative_error"
        ),
        "factor_log_diagonal_std_mean": average(
            rows, "factor_log_diagonal_std"
        ),
        "factor_log_diagonal_max_abs_max": max(
            float(row["factor_log_diagonal_max_abs"]) for row in rows
        ),
    }
    for family in (
        "product_initial_spectrum",
        "product_final_spectrum",
        "dense_initial_spectrum",
        "dense_final_spectrum",
    ):
        aggregate[family] = {
            key: sum(
                float(row[family][key]) for row in rows
            )
            / len(rows)
            for key in rows[0][family]
        }

    payload = {
        "schema_version": "mai_124m_mlp_product_fht_diagnosis_v1",
        "product_checkpoint": {
            "path": str(args.product_checkpoint.resolve()),
            "sha256": sha256(args.product_checkpoint),
            "next_iter": checkpoint["next_iter"],
            "execution_provenance": checkpoint.get(
                "execution_provenance"
            ),
        },
        "dense_start": {
            "path": str(args.dense_start.resolve()),
            "sha256": sha256(args.dense_start),
            "step": dense_start_payload["step"],
        },
        "dense_end": {
            "path": str(args.dense_end.resolve()),
            "sha256": sha256(args.dense_end),
            "step": dense_end_payload["step"],
        },
        "aggregate": aggregate,
        "layers": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    temporary.replace(args.output)
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
