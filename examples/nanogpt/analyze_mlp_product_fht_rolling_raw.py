#!/usr/bin/env python3
"""Replay a moving ProductFHT atlas through sealed raw MLP gradients.

The replay is causal and zero-update with respect to the language model. It
uses the exact registered LR schedule, normalized materialized action, and
per-coordinate trust cap while updating only one compact ProductFHT state.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_mlp_highcadence_basis import file_sha256
from examples.nanogpt.analyze_mlp_optimizer_path_rate_distortion import (
    load_probe_learning_rates,
)
from examples.nanogpt.analyze_mlp_optimizer_probe_span import load_probe_inventory
from examples.nanogpt.analyze_mlp_product_fht_tangent_anchor import (
    TARGET_SEED_OFFSETS,
    evaluate_anchor,
    git_commit,
    natural_pullback_action,
    natural_pullback_coordinates,
    summarize,
)
from examples.nanogpt.analyze_parameter_trajectory import write_csv
from latent_weight_lab import ProductFHTLinear


def normalized_coordinate_step(
    module: ProductFHTLinear,
    target_descent: torch.Tensor,
    *,
    learning_rate: float,
    coordinate_cap: float,
    norm_reference: torch.Tensor | None = None,
) -> dict[str, float]:
    """Apply one normalized raw-gradient descent step to compact coordinates."""
    diagonal_direction, output_direction = natural_pullback_coordinates(
        module, target_descent
    )
    action = module._weight_jvp_from_factors(
        diagonal_direction, output_direction
    )
    target_norm = (
        target_descent if norm_reference is None else norm_reference
    ).float().norm().clamp_min(1e-30)
    action_norm = action.float().norm().clamp_min(1e-30)
    normalization_scale = float(target_norm / action_norm)
    diagonal_direction = diagonal_direction * normalization_scale
    output_direction = output_direction * normalization_scale
    largest_coordinate_update = float(learning_rate) * max(
        float(diagonal_direction.abs().max()),
        float(output_direction.abs().max()),
    )
    cap_scale = min(
        1.0,
        float(coordinate_cap) / max(largest_coordinate_update, 1e-30),
    )
    with torch.no_grad():
        module.product_log_diagonals.add_(
            diagonal_direction, alpha=float(learning_rate) * cap_scale
        )
        module.product_output_log_gain.add_(
            output_direction, alpha=float(learning_rate) * cap_scale
        )
    return {
        "normalization_scale": normalization_scale,
        "uncapped_maximum_coordinate_update": largest_coordinate_update,
        "cap_scale": cap_scale,
        "applied_maximum_coordinate_update": largest_coordinate_update * cap_scale,
    }


def coordinate_statistics(module: ProductFHTLinear) -> dict[str, float]:
    values = torch.cat(
        (
            module.product_log_diagonals.detach().float().flatten(),
            module.product_output_log_gain.detach().float().flatten(),
        )
    )
    return {
        "coordinate_rms": float(values.square().mean().sqrt()),
        "coordinate_max_abs": float(values.abs().max()),
        "coordinate_clamp_fraction": float((values.abs() >= 6.0).float().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--targets", default="mlp.c_fc,mlp.c_proj")
    parser.add_argument("--factors", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--discovery-stop", type=int, default=119)
    parser.add_argument("--validation-stop", type=int, default=179)
    parser.add_argument("--coordinate-cap", type=float, default=0.02)
    parser.add_argument(
        "--norm-reference-field",
        choices=("raw_gradient_descent", "exact_applied_direction"),
        default="exact_applied_direction",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    targets = {item for item in args.targets.split(",") if item}
    if targets != set(TARGET_SEED_OFFSETS):
        raise ValueError("the preregistered replay requires both MLP matrices")
    paths = sorted(args.probe_dir.glob("step_*.pt"))
    steps, values, input_metadata = load_probe_inventory(
        paths, layers={args.layer}, targets=targets
    )
    learning_rates = load_probe_learning_rates(paths, set(values))
    args.output.mkdir(parents=True, exist_ok=True)
    all_scores: list[dict[str, Any]] = []
    all_summary: list[dict[str, Any]] = []
    all_updates: list[dict[str, Any]] = []
    coordinates: dict[str, Any] = {}
    accounting: dict[str, Any] = {}
    final_ranks: dict[str, int] = {}
    for parameter in sorted(values):
        target_name = ".".join(parameter.split(".")[-3:-1])
        gradients = torch.stack(values[parameter]["raw_gradient_descent"]).to(
            args.device, dtype=torch.float32
        )
        norm_references = torch.stack(
            values[parameter][args.norm_reference_field]
        ).to(args.device, dtype=torch.float32)
        out_features, in_features = gradients.shape[1:]
        weight_std = (
            0.02
            if target_name == "mlp.c_fc"
            else 0.02 / math.sqrt(2 * args.n_layer)
        )

        def new_module() -> ProductFHTLinear:
            return ProductFHTLinear(
                in_features,
                out_features,
                factors=args.factors,
                seed=(
                    args.base_seed
                    + args.layer * 4
                    + TARGET_SEED_OFFSETS[target_name]
                ),
                weight_std=weight_std,
                weight_space_muon=False,
                natural_gradient=True,
            ).to(args.device)

        identity_module = new_module()
        identity_rows = evaluate_anchor(
            identity_module,
            gradients,
            steps=steps,
            anchor="identity",
            discovery_stop=args.discovery_stop,
            validation_stop=args.validation_stop,
        )
        del identity_module
        module = new_module()
        rolling_rows: list[dict[str, Any]] = []
        update_count = 0
        capped_update_count = 0
        for probe_index, (probe_step, gradient) in enumerate(
            zip(steps, gradients, strict=True)
        ):
            action, cosine, capture = natural_pullback_action(
                module, gradient, differentiable_anchor=False
            )
            rolling_rows.append(
                {
                    "anchor": "rolling_raw",
                    "probe_index": probe_index,
                    "step": probe_step,
                    "split": (
                        "discovery"
                        if probe_step < args.discovery_stop
                        else "validation"
                        if probe_step < args.validation_stop
                        else "test"
                    ),
                    "action_cosine": float(cosine),
                    "action_capture": float(capture),
                    "action_to_target_norm_ratio": float(
                        action.float().norm()
                        / gradient.float().norm().clamp_min(1e-30)
                    ),
                    **coordinate_statistics(module),
                }
            )
            if probe_index + 1 == len(steps):
                continue
            interval = steps[probe_index + 1] - probe_step
            learning_rate = learning_rates[parameter][probe_index]
            interval_rows = []
            for _ in range(interval):
                diagnostics = normalized_coordinate_step(
                    module,
                    gradient,
                    learning_rate=learning_rate,
                    coordinate_cap=args.coordinate_cap,
                    norm_reference=norm_references[probe_index],
                )
                interval_rows.append(diagnostics)
                update_count += 1
                capped_update_count += int(diagnostics["cap_scale"] < 1.0)
            all_updates.append(
                {
                    "parameter": parameter,
                    "probe_index": probe_index,
                    "step": probe_step,
                    "interval_updates": interval,
                    "learning_rate": learning_rate,
                    "mean_normalization_scale": sum(
                        row["normalization_scale"] for row in interval_rows
                    )
                    / interval,
                    "minimum_cap_scale": min(
                        row["cap_scale"] for row in interval_rows
                    ),
                    "maximum_applied_coordinate_update": max(
                        row["applied_maximum_coordinate_update"]
                        for row in interval_rows
                    ),
                    **coordinate_statistics(module),
                }
            )
        rows = [
            {"parameter": parameter, **row}
            for row in identity_rows + rolling_rows
        ]
        all_scores.extend(rows)
        all_summary.extend(summarize(rows, parameter=parameter))
        with torch.no_grad():
            final_ranks[parameter] = int(
                torch.linalg.matrix_rank(module.weight.detach().float())
            )
        coordinates[parameter] = {
            "product_log_diagonals": module.product_log_diagonals.detach().cpu(),
            "product_output_log_gain": module.product_output_log_gain.detach().cpu(),
            "seed": module.seed,
            "factors": module.factors,
            "update_count": update_count,
            "capped_update_count": capped_update_count,
        }
        accounting[parameter] = {
            "dense_scalars": out_features * in_features,
            "coordinate_scalars": module.trainable_scalar_count,
            "coordinate_fraction": module.trainable_scalar_count
            / (out_features * in_features),
        }
        del gradients, norm_references, module
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    scores_path = args.output / "probe_scores.csv"
    summary_path = args.output / "summary.csv"
    updates_path = args.output / "replay_updates.csv"
    coordinates_path = args.output / "final_coordinates.pt"
    write_csv(scores_path, all_scores)
    write_csv(summary_path, all_summary)
    write_csv(updates_path, all_updates)
    torch.save(coordinates, coordinates_path)
    metadata = {
        "schema_version": "nanogpt_mlp_product_fht_rolling_raw_v2",
        "method": "causal zero-order-hold direct-raw ProductFHT coordinate replay",
        "layer": args.layer,
        "targets": sorted(targets),
        "factors": args.factors,
        "coordinate_cap": args.coordinate_cap,
        "norm_reference_field": args.norm_reference_field,
        "steps": steps,
        "accounting": accounting,
        "final_decoded_matrix_ranks": final_ranks,
        "input": input_metadata,
        "runtime_seconds": time.time() - started,
        "source_commit": git_commit(Path(__file__).resolve().parents[2]),
        "probe_scores_sha256": file_sha256(scores_path),
        "summary_sha256": file_sha256(summary_path),
        "replay_updates_sha256": file_sha256(updates_path),
        "final_coordinates_sha256": file_sha256(coordinates_path),
        "promotion_gate": {
            "validation_and_test_mean_action_capture_each_target": 0.40,
            "test_minimum_action_capture_each_target": 0.20,
            "validation_and_test_enrichment_over_identity_each_target": 4.0,
        },
        "limitations": [
            "The replay uses gradients sampled on the dense Muon path, not gradients from a native compact-model trajectory.",
            "Missing steps use zero-order-held gradients and the LR recorded at the preceding probe.",
            "The exact-applied norm reference preserves dense-Muon step scale while retaining raw-gradient orientation.",
            "No language-model parameter, checkpoint, or optimizer state is updated.",
        ],
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"metadata_sha256={file_sha256(metadata_path)}")


if __name__ == "__main__":
    main()
