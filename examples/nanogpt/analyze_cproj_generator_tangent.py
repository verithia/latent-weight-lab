from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from examples.nanogpt.analyze_cproj_functional_span import collect_layer_io
from examples.nanogpt.analyze_cproj_manifold import load_model
from examples.nanogpt.analyze_residual_compatibility import fixed_validation_batches
from latent_weight_lab.block_fht import (
    BlockFHTLinear,
    block_fht_grad_latent,
    block_fht_slice,
)


def _generator_scale(module: BlockFHTLinear) -> float:
    scale = float(module.weight_scale)
    if module.residual_base_weight is not None:
        scale *= float(module.residual_base_scale)
    return scale


def validate_linear_tangent(module: BlockFHTLinear) -> None:
    unsupported = []
    if module.modulation_alpha != 0.0:
        unsupported.append("modulation_alpha")
    if module.output_gain is not None:
        unsupported.append("output_gain")
    if module.input_gain is not None:
        unsupported.append("input_gain")
    if module.spectral_core is not None:
        unsupported.append("spectral_core")
    if unsupported:
        raise ValueError(
            "generator tangent diagnostic requires a plain linear BlockFHT map; "
            "unsupported: " + ", ".join(unsupported)
        )


def tangent_apply(
    module: BlockFHTLinear,
    hidden: torch.Tensor,
    delta_latent: torch.Tensor,
) -> torch.Tensor:
    validate_linear_tangent(module)
    weight = block_fht_slice(
        delta_latent,
        module.generator.size,
        module.generator.layers,
        module.generator.seed,
        0,
        module.generator.size,
    )
    weight = (
        weight
        * _generator_scale(module)
    ).view(module.out_features, module.in_features)
    return F.linear(hidden, weight)


def tangent_adjoint(
    module: BlockFHTLinear,
    hidden: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    validate_linear_tangent(module)
    grad_weight = output.transpose(0, 1).matmul(hidden)
    return block_fht_grad_latent(
        module.generator.latent.detach(),
        (grad_weight * _generator_scale(module)).reshape(-1),
        module.generator.size,
        module.generator.layers,
        module.generator.seed,
        0,
        module.generator.size,
    )


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.reshape(-1).float()
    right = right.reshape(-1).float()
    denominator = left.norm() * right.norm()
    if denominator <= 0:
        return float("nan")
    return float(torch.dot(left, right) / denominator)


def cgls_generator_tangent(
    module: BlockFHTLinear,
    hidden: torch.Tensor,
    target: torch.Tensor,
    iterations: int,
    record_iterations: set[int],
    relative_tolerance: float = 1e-8,
    holdout_hidden: torch.Tensor | None = None,
    holdout_target: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    hidden = hidden.float()
    target = target.float()
    if (holdout_hidden is None) != (holdout_target is None):
        raise ValueError(
            "holdout_hidden and holdout_target must be provided together"
        )
    if holdout_hidden is not None:
        holdout_hidden = holdout_hidden.float()
        holdout_target = holdout_target.float()
    delta = torch.zeros_like(module.generator.latent.detach()).float()
    residual = target.clone()
    gradient = tangent_adjoint(module, hidden, residual)
    direction = gradient.clone()
    gamma = torch.dot(gradient, gradient)
    initial_gamma = gamma.detach().clone()
    target_energy = target.square().sum().clamp_min(1e-30)
    rows: list[dict[str, float]] = []

    for iteration in range(1, iterations + 1):
        projected = tangent_apply(module, hidden, direction)
        denominator = projected.square().sum()
        if denominator <= 0:
            break
        step = gamma / denominator
        delta.add_(direction, alpha=float(step))
        residual.sub_(projected, alpha=float(step))
        new_gradient = tangent_adjoint(module, hidden, residual)
        new_gamma = torch.dot(new_gradient, new_gradient)
        if iteration in record_iterations or iteration == iterations:
            prediction = target - residual
            row = {
                "iteration": float(iteration),
                "explained_energy": float(
                    1.0 - residual.square().sum() / target_energy
                ),
                "target_cosine": cosine(prediction, target),
                "target_rms": float(target.square().mean().sqrt()),
                "prediction_rms": float(prediction.square().mean().sqrt()),
                "residual_rms": float(residual.square().mean().sqrt()),
                "delta_latent_l2": float(delta.norm()),
                "relative_normal_residual": float(
                    (new_gamma / initial_gamma.clamp_min(1e-30)).sqrt()
                ),
            }
            if holdout_hidden is not None and holdout_target is not None:
                holdout_prediction = tangent_apply(
                    module, holdout_hidden, delta
                )
                holdout_residual = holdout_target - holdout_prediction
                row.update(
                    {
                        "holdout_explained_energy": float(
                            1.0
                            - holdout_residual.square().sum()
                            / holdout_target.square().sum().clamp_min(1e-30)
                        ),
                        "holdout_target_cosine": cosine(
                            holdout_prediction, holdout_target
                        ),
                        "holdout_target_rms": float(
                            holdout_target.square().mean().sqrt()
                        ),
                        "holdout_prediction_rms": float(
                            holdout_prediction.square().mean().sqrt()
                        ),
                        "holdout_residual_rms": float(
                            holdout_residual.square().mean().sqrt()
                        ),
                    }
                )
            rows.append(row)
        if new_gamma <= initial_gamma * float(relative_tolerance) ** 2:
            gamma = new_gamma
            break
        beta = new_gamma / gamma.clamp_min(1e-30)
        direction.mul_(float(beta)).add_(new_gradient)
        gradient = new_gradient
        gamma = new_gamma
    return delta, rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention-only", required=True, type=Path)
    parser.add_argument("--plain-cproj", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--layers", default="3,6,9")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--sample-cap", type=int, default=4096)
    parser.add_argument("--sample-seed", type=int, default=20260716)
    parser.add_argument("--holdout-sample-seed", type=int, default=20260717)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--record-iterations", default="1,2,4,8,16,32,64")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    layers = [int(part) for part in args.layers.split(",")]
    record_iterations = {
        int(part) for part in args.record_iterations.split(",") if part
    }
    record_iterations.add(int(args.iterations))
    batches = fixed_validation_batches(
        args.data_dir,
        args.batch_size,
        args.block_size,
        args.batches,
        args.sample_seed,
    )
    print("collecting matched attention-only activations", flush=True)
    attention = collect_layer_io(
        args.attention_only, batches, layers, args.sample_cap, args.device
    )
    print("collecting matched plain-cproj activations", flush=True)
    plain = collect_layer_io(
        args.plain_cproj, batches, layers, args.sample_cap, args.device
    )
    holdout_batches = fixed_validation_batches(
        args.data_dir,
        args.batch_size,
        args.block_size,
        args.batches,
        args.holdout_sample_seed,
    )
    print("collecting held-out attention-only activations", flush=True)
    attention_holdout = collect_layer_io(
        args.attention_only,
        holdout_batches,
        layers,
        args.sample_cap,
        args.device,
    )
    print("collecting held-out plain-cproj activations", flush=True)
    plain_holdout = collect_layer_io(
        args.plain_cproj,
        holdout_batches,
        layers,
        args.sample_cap,
        args.device,
    )
    model = load_model(args.plain_cproj, args.device)

    rows: list[dict[str, object]] = []
    summary: dict[str, dict[str, float]] = {}
    try:
        for layer in layers:
            module = model.transformer.h[layer].mlp.c_proj
            if not isinstance(module, BlockFHTLinear):
                raise TypeError(
                    f"layer {layer} c_proj is {type(module).__name__}, not BlockFHTLinear"
                )
            hidden = plain[(layer, "post_gelu")].to(args.device)
            target = (
                attention[(layer, "mlp_out")] - plain[(layer, "mlp_out")]
            ).to(args.device)
            holdout_hidden = plain_holdout[(layer, "post_gelu")].to(args.device)
            holdout_target = (
                attention_holdout[(layer, "mlp_out")]
                - plain_holdout[(layer, "mlp_out")]
            ).to(args.device)
            print(
                f"solving layer {layer}: samples={hidden.shape[0]} "
                f"latent={module.generator.latent.numel()} iterations={args.iterations}",
                flush=True,
            )
            delta, layer_rows = cgls_generator_tangent(
                module,
                hidden,
                target,
                args.iterations,
                record_iterations,
                holdout_hidden=holdout_hidden,
                holdout_target=holdout_target,
            )
            latent = module.generator.latent.detach().float()
            for row in layer_rows:
                rows.append(
                    {
                        "layer": layer,
                        "latent_dim": latent.numel(),
                        "weight_size": module.generator.size,
                        "latent_ratio": latent.numel() / module.generator.size,
                        **row,
                    }
                )
            final = layer_rows[-1]
            summary[str(layer)] = {
                **final,
                "latent_dim": float(latent.numel()),
                "weight_size": float(module.generator.size),
                "latent_ratio": float(latent.numel() / module.generator.size),
                "trained_latent_l2": float(latent.norm()),
                "fitted_delta_to_trained_latent_l2": float(
                    delta.norm() / latent.norm().clamp_min(1e-30)
                ),
            }
    finally:
        del model
        if "cuda" in args.device:
            torch.cuda.empty_cache()

    summary["mean"] = {
        key: float(np.mean([summary[str(layer)][key] for layer in layers]))
        for key in summary[str(layers[0])]
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "cproj_generator_tangent_cgls.csv", rows)
    (args.output / "cproj_generator_tangent_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
