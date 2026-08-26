#!/usr/bin/env python3
"""Causal zero-update replay of a compact low-displacement-rank MLP chart."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_mlp_highcadence_basis import file_sha256
from examples.nanogpt.analyze_mlp_nonlinear_bilateral_kernel import (
    apply_normalized_step,
    coordinate_statistics,
    project_target,
    split_name,
)
from examples.nanogpt.analyze_mlp_optimizer_path_rate_distortion import (
    load_probe_learning_rates,
)
from examples.nanogpt.analyze_mlp_optimizer_probe_span import load_probe_inventory
from examples.nanogpt.analyze_mlp_product_fht_tangent_anchor import git_commit, summarize
from examples.nanogpt.analyze_parameter_trajectory import write_csv


def cyclic_displacement(matrix: torch.Tensor, *, rho: float) -> torch.Tensor:
    return matrix - float(rho) * torch.roll(matrix, shifts=(1, 1), dims=(0, 1))


def inverse_cyclic_displacement(source: torch.Tensor, *, rho: float) -> torch.Tensor:
    if source.ndim != 2:
        raise ValueError("source must be a matrix")
    if not 0.0 < rho < 1.0:
        raise ValueError("rho must be in (0, 1)")
    rows, columns = source.shape
    row_frequency = torch.fft.fftfreq(rows, device=source.device)
    column_frequency = torch.fft.fftfreq(columns, device=source.device)
    phase = torch.exp(
        -2j
        * math.pi
        * (row_frequency[:, None] + column_frequency[None, :])
    )
    denominator = 1.0 - float(rho) * phase
    transformed = torch.fft.fft2(source)
    return torch.fft.ifft2(transformed / denominator).real


class LowDisplacementRankChart(torch.nn.Module):
    def __init__(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        *,
        rho: float,
        output_scale: float,
    ) -> None:
        super().__init__()
        if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
            raise ValueError("left/right must be rank-matched matrices")
        self.left = torch.nn.Parameter(left.detach().clone())
        self.right = torch.nn.Parameter(right.detach().clone())
        self.register_buffer("initial_left", left.detach().clone())
        self.register_buffer("initial_right", right.detach().clone())
        self.rho = float(rho)
        self.output_scale = float(output_scale)

    def decode(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return inverse_cyclic_displacement(
            left @ right.transpose(0, 1), rho=self.rho
        )

    def forward(self) -> torch.Tensor:
        return self.output_scale * (
            self.decode(self.left, self.right)
            - self.decode(self.initial_left, self.initial_right)
        )

    @property
    def coordinate_count(self) -> int:
        return self.left.numel() + self.right.numel()


def gradient_seeded_displacement_factors(
    gradient: torch.Tensor,
    *,
    rank: int,
    rho: float,
    decoded_rms: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    displaced = cyclic_displacement(gradient.float(), rho=rho)
    u, s, vh = torch.linalg.svd(displaced, full_matrices=False)
    root = s[:rank].clamp_min(1e-30).sqrt()
    left = u[:, :rank] * root.unsqueeze(0)
    right = vh[:rank].transpose(0, 1) * root.unsqueeze(0)
    decoded = inverse_cyclic_displacement(left @ right.transpose(0, 1), rho=rho)
    factor_scale = math.sqrt(
        float(decoded_rms)
        / float(decoded.square().mean().sqrt().clamp_min(1e-30))
    )
    return left * factor_scale, right * factor_scale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--targets", default="mlp.c_fc,mlp.c_proj")
    parser.add_argument("--rank", type=int, default=6)
    parser.add_argument("--rho", type=float, default=0.9)
    parser.add_argument("--decoded-seed-rms", type=float, default=0.5)
    parser.add_argument("--output-scale", type=float, default=0.02)
    parser.add_argument("--cg-steps", type=int, default=12)
    parser.add_argument("--damping-ratio", type=float, default=1e-4)
    parser.add_argument("--coordinate-cap", type=float, default=0.02)
    parser.add_argument("--discovery-stop", type=int, default=119)
    parser.add_argument("--validation-stop", type=int, default=179)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    targets = {value for value in args.targets.split(",") if value}
    if targets != {"mlp.c_fc", "mlp.c_proj"}:
        raise ValueError("the frozen oracle requires both MLP matrices")
    paths = sorted(args.probe_dir.glob("step_*.pt"))
    steps, values, input_metadata = load_probe_inventory(
        paths, layers={args.layer}, targets=targets
    )
    learning_rates = load_probe_learning_rates(paths, set(values))
    args.output.mkdir(parents=True, exist_ok=True)
    scores: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    updates: list[dict[str, Any]] = []
    final_state: dict[str, Any] = {}
    accounting: dict[str, Any] = {}
    final_delta_ranks: dict[str, int] = {}
    cumulative_recovery: dict[str, Any] = {}

    for parameter in sorted(values):
        gradients = torch.stack(values[parameter]["raw_gradient_descent"]).to(
            args.device, torch.float32
        )
        norm_references = torch.stack(
            values[parameter]["exact_applied_direction"]
        ).to(args.device, torch.float32)
        left, right = gradient_seeded_displacement_factors(
            gradients[0],
            rank=args.rank,
            rho=args.rho,
            decoded_rms=args.decoded_seed_rms,
        )

        def new_module() -> LowDisplacementRankChart:
            return LowDisplacementRankChart(
                left,
                right,
                rho=args.rho,
                output_scale=args.output_scale,
            ).to(args.device)

        initial_module = new_module()
        for index, (step, gradient) in enumerate(zip(steps, gradients, strict=True)):
            _, _, metrics = project_target(
                initial_module,
                gradient,
                cg_steps=args.cg_steps,
                damping_ratio=args.damping_ratio,
            )
            scores.append(
                {
                    "parameter": parameter,
                    "anchor": "identity",
                    "probe_index": index,
                    "step": step,
                    "split": split_name(step, args.discovery_stop, args.validation_stop),
                    **metrics,
                    **coordinate_statistics(initial_module),
                }
            )
        del initial_module

        module = new_module()
        target_path = torch.zeros_like(gradients[0])
        capped = 0
        update_count = 0
        for index, (step, gradient) in enumerate(zip(steps, gradients, strict=True)):
            coordinates, action, metrics = project_target(
                module,
                gradient,
                cg_steps=args.cg_steps,
                damping_ratio=args.damping_ratio,
            )
            scores.append(
                {
                    "parameter": parameter,
                    "anchor": "rolling_ldr",
                    "probe_index": index,
                    "step": step,
                    "split": split_name(step, args.discovery_stop, args.validation_stop),
                    **metrics,
                    **coordinate_statistics(module),
                }
            )
            if index + 1 == len(steps):
                continue
            interval = steps[index + 1] - step
            learning_rate = learning_rates[parameter][index]
            interval_rows: list[dict[str, float]] = []
            for _ in range(interval):
                coordinates, action, _ = project_target(
                    module,
                    gradient,
                    cg_steps=args.cg_steps,
                    damping_ratio=args.damping_ratio,
                )
                diagnostics = apply_normalized_step(
                    module,
                    coordinates,
                    action,
                    norm_reference=norm_references[index],
                    learning_rate=learning_rate,
                    coordinate_cap=args.coordinate_cap,
                )
                interval_rows.append(diagnostics)
                capped += int(diagnostics["cap_scale"] < 1.0)
                update_count += 1
                normalized_target = gradient * (
                    norm_references[index].norm()
                    / gradient.norm().clamp_min(1e-30)
                )
                target_path.add_(normalized_target, alpha=learning_rate)
            updates.append(
                {
                    "parameter": parameter,
                    "probe_index": index,
                    "step": step,
                    "interval_updates": interval,
                    "learning_rate": learning_rate,
                    "mean_normalization_scale": sum(x["normalization_scale"] for x in interval_rows) / interval,
                    "minimum_cap_scale": min(x["cap_scale"] for x in interval_rows),
                    "maximum_applied_coordinate_update": max(x["applied_maximum_coordinate_update"] for x in interval_rows),
                    **coordinate_statistics(module),
                }
            )
        parameter_rows = [row for row in scores if row["parameter"] == parameter]
        summaries.extend(summarize(parameter_rows, parameter=parameter))
        with torch.no_grad():
            delta = module().float()
            final_delta_ranks[parameter] = int(torch.linalg.matrix_rank(delta))
            target_norm = target_path.norm().clamp_min(1e-30)
            delta_norm = delta.norm().clamp_min(1e-30)
            cosine = (delta * target_path).sum() / (delta_norm * target_norm)
            cumulative_recovery[parameter] = {
                "target_norm": float(target_norm),
                "decoded_delta_norm": float(delta_norm),
                "cosine": float(cosine),
                "energy_recovery": float(
                    1.0
                    - (delta - target_path).square().sum()
                    / target_path.square().sum().clamp_min(1e-30)
                ),
            }
        final_state[parameter] = {
            "left": module.left.detach().cpu(),
            "right": module.right.detach().cpu(),
            "initial_left": module.initial_left.detach().cpu(),
            "initial_right": module.initial_right.detach().cpu(),
            "update_count": update_count,
            "capped_update_count": capped,
        }
        dense = gradients.shape[1] * gradients.shape[2]
        accounting[parameter] = {
            "dense_scalars": dense,
            "coordinate_scalars": module.coordinate_count,
            "coordinate_fraction": module.coordinate_count / dense,
        }

    scores_path = args.output / "probe_scores.csv"
    summary_path = args.output / "summary.csv"
    updates_path = args.output / "replay_updates.csv"
    state_path = args.output / "final_state.pt"
    write_csv(scores_path, scores)
    write_csv(summary_path, summaries)
    write_csv(updates_path, updates)
    torch.save(final_state, state_path)
    metadata = {
        "schema_version": "nanogpt_mlp_low_displacement_rank_v1",
        "method": "gradient-seeded cyclic low-displacement-rank causal replay",
        "source_commit": git_commit(Path(__file__).resolve().parents[2]),
        "input": input_metadata,
        "layer": args.layer,
        "rank": args.rank,
        "rho": args.rho,
        "decoded_seed_rms": args.decoded_seed_rms,
        "output_scale": args.output_scale,
        "cg_steps": args.cg_steps,
        "damping_ratio": args.damping_ratio,
        "coordinate_cap": args.coordinate_cap,
        "steps": steps,
        "accounting": accounting,
        "final_delta_ranks": final_delta_ranks,
        "cumulative_recovery": cumulative_recovery,
        "promotion_gate": {
            "validation_and_test_mean_action_capture_each_target": 0.40,
            "test_minimum_action_capture_each_target": 0.20,
            "test_enrichment_over_initial_each_target": 4.0,
            "final_delta_rank_each_target": 768,
        },
        "limitations": [
            "Dense-path gradients are replayed without updating a language model.",
            "Missing steps use zero-order-held gradients and registered learning rates.",
            "The norm reference is the paired dense-Muon applied direction.",
        ],
        "runtime_seconds": time.time() - started,
        "probe_scores_sha256": file_sha256(scores_path),
        "summary_sha256": file_sha256(summary_path),
        "replay_updates_sha256": file_sha256(updates_path),
        "final_state_sha256": file_sha256(state_path),
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
