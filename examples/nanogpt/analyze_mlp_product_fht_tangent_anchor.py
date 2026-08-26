#!/usr/bin/env python3
"""Fit and audit a task-oriented ProductFHT tangent on sealed MLP gradients.

This is a zero-update representability oracle. It fits only the coordinates
that orient a five-stage ProductFHT Jacobian, then freezes them and measures
the approximately natural pullback/JVP action on chronological held-out raw
gradients. It neither updates a language model nor treats the task-selected
anchor as free procedural state.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_mlp_highcadence_basis import file_sha256
from examples.nanogpt.analyze_mlp_optimizer_probe_span import load_probe_inventory
from examples.nanogpt.analyze_parameter_trajectory import write_csv
from latent_weight_lab import ProductFHTLinear


TARGET_SEED_OFFSETS = {"mlp.c_fc": 2, "mlp.c_proj": 3}


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def chronological_split(step: int, discovery_stop: int, validation_stop: int) -> str:
    if step < discovery_stop:
        return "discovery"
    if step < validation_stop:
        return "validation"
    return "test"


def natural_pullback_action(
    module: ProductFHTLinear,
    target: torch.Tensor,
    *,
    differentiable_anchor: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return natural-coordinate tangent action, cosine, and squared cosine.

    The pullback coordinate is detached before the JVP. Optimizing the score
    therefore alternates between choosing the best local coordinate action
    under the registered diagonal metric and rotating the chart so that this
    fixed action better matches the target.
    """
    diagonal_direction, output_direction = natural_pullback_coordinates(
        module, target
    )
    log_diagonals = module.product_log_diagonals
    output_log_gain = module.product_output_log_gain
    if differentiable_anchor:
        action = module._weight_jvp_at_factors(
            log_diagonals,
            output_log_gain,
            diagonal_direction,
            output_direction,
        )
    else:
        action = module._weight_jvp_from_factors(
            diagonal_direction,
            output_direction,
        )
    target_float = target.float()
    action_float = action.float()
    cosine = torch.sum(target_float * action_float) / (
        target_float.norm() * action_float.norm()
    ).clamp_min(1e-30)
    return action, cosine, cosine.square()


def natural_pullback_coordinates(
    module: ProductFHTLinear,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the detached approximate-natural coordinate pullback."""
    log_diagonals = module.product_log_diagonals
    output_log_gain = module.product_output_log_gain
    weight = module._weight_from_factors(log_diagonals, output_log_gain)
    diagonal_pullback, output_pullback = torch.autograd.grad(
        weight,
        (log_diagonals, output_log_gain),
        grad_outputs=target.to(dtype=weight.dtype),
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )
    diagonal_metric = (
        module.weight_std
        * module.weight_std
        * module.out_features
        * module.in_features
        / module.padded_features
    )
    diagonal_direction = (
        diagonal_pullback / max(diagonal_metric, 1e-12)
    ).detach()
    row_metric = weight.detach().float().square().sum(dim=1).clamp_min(1e-12)
    output_direction = (output_pullback / row_metric).detach()
    return diagonal_direction, output_direction


def deterministic_mixture(
    gradients: torch.Tensor,
    *,
    update: int,
    width: int,
    seed: int,
) -> torch.Tensor:
    """Return a deterministic equal-weight signed mixture of discovery rows."""
    if gradients.ndim != 3 or gradients.shape[0] < 1:
        raise ValueError("gradient inventory must be [time, out, in]")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 104729 * int(update))
    indices = torch.randint(
        gradients.shape[0], (int(width),), generator=generator
    ).tolist()
    signs = (
        torch.randint(0, 2, (int(width),), generator=generator) * 2 - 1
    ).tolist()
    mixture = torch.zeros_like(gradients[0])
    for index, sign in zip(indices, signs, strict=True):
        mixture.add_(gradients[index], alpha=float(sign))
    return mixture / mixture.norm().clamp_min(1e-30)


def fit_anchor(
    module: ProductFHTLinear,
    discovery: torch.Tensor,
    *,
    updates: int,
    learning_rate: float,
    mixture_width: int,
    bound: float,
    seed: int,
) -> list[dict[str, Any]]:
    optimizer = torch.optim.Adam(
        [module.product_log_diagonals, module.product_output_log_gain],
        lr=float(learning_rate),
    )
    history: list[dict[str, Any]] = []
    for update in range(int(updates)):
        target = deterministic_mixture(
            discovery,
            update=update,
            width=mixture_width,
            seed=seed,
        )
        optimizer.zero_grad(set_to_none=True)
        _action, cosine, score = natural_pullback_action(
            module, target, differentiable_anchor=True
        )
        regularizer = 1e-4 * (
            module.product_log_diagonals.square().mean()
            + module.product_output_log_gain.square().mean()
        )
        (-score + regularizer).backward()
        optimizer.step()
        with torch.no_grad():
            module.product_log_diagonals.clamp_(-bound, bound)
            module.product_output_log_gain.clamp_(-bound, bound)
        if update == 0 or (update + 1) % 8 == 0 or update + 1 == updates:
            squared_sum = (
                module.product_log_diagonals.detach().float().square().sum()
                + module.product_output_log_gain.detach().float().square().sum()
            )
            history.append(
                {
                    "fit_update": update + 1,
                    "mixture_action_cosine": float(cosine.detach()),
                    "mixture_action_capture": float(score.detach()),
                    "anchor_rms": float(
                        torch.sqrt(squared_sum / module.trainable_scalar_count)
                    ),
                    "anchor_max_abs": max(
                        float(module.product_log_diagonals.detach().abs().max()),
                        float(module.product_output_log_gain.detach().abs().max()),
                    ),
                }
            )
    return history


def evaluate_anchor(
    module: ProductFHTLinear,
    gradients: torch.Tensor,
    *,
    steps: list[int],
    anchor: str,
    discovery_stop: int,
    validation_stop: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (step, target) in enumerate(zip(steps, gradients, strict=True)):
        action, cosine, score = natural_pullback_action(
            module, target, differentiable_anchor=False
        )
        rows.append(
            {
                "anchor": anchor,
                "probe_index": index,
                "step": step,
                "split": chronological_split(step, discovery_stop, validation_stop),
                "action_cosine": float(cosine.detach()),
                "action_capture": float(score.detach()),
                "action_to_target_norm_ratio": float(
                    action.detach().float().norm()
                    / target.float().norm().clamp_min(1e-30)
                ),
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]], *, parameter: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    anchors = sorted(
        {str(row["anchor"]) for row in rows},
        key=lambda value: (value != "identity", value),
    )
    for anchor in anchors:
        for split in ("discovery", "validation", "test"):
            selected = [
                row
                for row in rows
                if row["anchor"] == anchor and row["split"] == split
            ]
            captures = [float(row["action_capture"]) for row in selected]
            cosines = [float(row["action_cosine"]) for row in selected]
            output.append(
                {
                    "parameter": parameter,
                    "anchor": anchor,
                    "split": split,
                    "count": len(selected),
                    "mean_action_capture": sum(captures) / len(captures),
                    "minimum_action_capture": min(captures),
                    "mean_action_cosine": sum(cosines) / len(cosines),
                }
            )
    identity = {
        row["split"]: row for row in output if row["anchor"] == "identity"
    }
    for row in output:
        baseline = float(identity[row["split"]]["mean_action_capture"])
        row["enrichment_over_identity"] = (
            float(row["mean_action_capture"]) / max(baseline, 1e-30)
        )
    return output


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
    parser.add_argument("--fit-updates", type=int, default=128)
    parser.add_argument("--fit-lr", type=float, default=0.02)
    parser.add_argument("--mixture-width", type=int, default=4)
    parser.add_argument("--anchor-bound", type=float, default=0.5)
    parser.add_argument("--fit-seed", type=int, default=20260826)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    targets = {item for item in args.targets.split(",") if item}
    if targets != set(TARGET_SEED_OFFSETS):
        raise ValueError("the preregistered oracle requires both MLP matrices")
    paths = sorted(args.probe_dir.glob("step_*.pt"))
    steps, values, input_metadata = load_probe_inventory(
        paths, layers={args.layer}, targets=targets
    )
    if not (
        steps[0] < args.discovery_stop < args.validation_stop <= steps[-1]
    ):
        raise ValueError("invalid chronological split")
    args.output.mkdir(parents=True, exist_ok=True)
    all_scores: list[dict[str, Any]] = []
    all_summary: list[dict[str, Any]] = []
    all_history: list[dict[str, Any]] = []
    anchors: dict[str, Any] = {}
    accounting: dict[str, Any] = {}
    for target_index, parameter in enumerate(sorted(values)):
        target = ".".join(parameter.split(".")[-3:-1])
        gradients = torch.stack(values[parameter]["raw_gradient_descent"]).to(
            args.device, dtype=torch.float32
        )
        gradients = gradients / gradients.flatten(1).norm(dim=1).clamp_min(
            1e-30
        ).view(-1, 1, 1)
        out_features, in_features = gradients.shape[1:]
        weight_std = (
            0.02
            if target == "mlp.c_fc"
            else 0.02 / math.sqrt(2 * args.n_layer)
        )
        module = ProductFHTLinear(
            in_features,
            out_features,
            factors=args.factors,
            seed=args.base_seed + args.layer * 4 + TARGET_SEED_OFFSETS[target],
            weight_std=weight_std,
            weight_space_muon=False,
            natural_gradient=True,
        ).to(args.device)
        identity_rows = evaluate_anchor(
            module,
            gradients,
            steps=steps,
            anchor="identity",
            discovery_stop=args.discovery_stop,
            validation_stop=args.validation_stop,
        )
        discovery_indices = [
            index for index, step in enumerate(steps) if step < args.discovery_stop
        ]
        history = fit_anchor(
            module,
            gradients[discovery_indices],
            updates=args.fit_updates,
            learning_rate=args.fit_lr,
            mixture_width=args.mixture_width,
            bound=args.anchor_bound,
            seed=args.fit_seed + target_index,
        )
        fitted_rows = evaluate_anchor(
            module,
            gradients,
            steps=steps,
            anchor="fitted",
            discovery_stop=args.discovery_stop,
            validation_stop=args.validation_stop,
        )
        rows = [
            {"parameter": parameter, **row}
            for row in identity_rows + fitted_rows
        ]
        all_scores.extend(rows)
        all_summary.extend(summarize(rows, parameter=parameter))
        all_history.extend({"parameter": parameter, **row} for row in history)
        anchors[parameter] = {
            "product_log_diagonals": module.product_log_diagonals.detach().cpu(),
            "product_output_log_gain": module.product_output_log_gain.detach().cpu(),
            "seed": module.seed,
            "factors": module.factors,
        }
        accounting[parameter] = {
            "dense_scalars": out_features * in_features,
            "anchor_scalars": module.trainable_scalar_count,
            "anchor_fraction": module.trainable_scalar_count
            / (out_features * in_features),
            "padded_features": module.padded_features,
        }
        del gradients, module
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    score_path = args.output / "probe_scores.csv"
    summary_path = args.output / "summary.csv"
    history_path = args.output / "fit_history.csv"
    anchor_path = args.output / "anchors.pt"
    write_csv(score_path, all_scores)
    write_csv(summary_path, all_summary)
    write_csv(history_path, all_history)
    torch.save(anchors, anchor_path)
    metadata = {
        "schema_version": "nanogpt_mlp_product_fht_tangent_anchor_v1",
        "method": "alternating exact-VJP and differentiable exact-JVP tangent orientation",
        "layer": args.layer,
        "targets": sorted(targets),
        "factors": args.factors,
        "base_seed": args.base_seed,
        "discovery_stop": args.discovery_stop,
        "validation_stop": args.validation_stop,
        "fit_updates": args.fit_updates,
        "fit_lr": args.fit_lr,
        "mixture_width": args.mixture_width,
        "anchor_bound": args.anchor_bound,
        "fit_seed": args.fit_seed,
        "accounting": accounting,
        "input": input_metadata,
        "runtime_seconds": time.time() - started,
        "source_commit": git_commit(Path(__file__).resolve().parents[2]),
        "probe_scores_sha256": file_sha256(score_path),
        "summary_sha256": file_sha256(summary_path),
        "fit_history_sha256": file_sha256(history_path),
        "anchors_sha256": file_sha256(anchor_path),
        "promotion_gate": {
            "validation_and_test_mean_action_capture_each_target": 0.40,
            "test_minimum_action_capture_each_target": 0.20,
            "validation_and_test_enrichment_over_identity_each_target": 4.0,
        },
        "limitations": [
            "The task-selected anchor is counted state and is not treated as a free procedural seed.",
            "This optimistic oracle uses the full discovery phase and is not a deployable online bootstrap.",
            "The diagonal natural metric is approximate; the action is its exact ProductFHT JVP.",
            "No language-model parameter or optimizer state is updated.",
        ],
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"metadata_sha256={file_sha256(metadata_path)}")


if __name__ == "__main__":
    main()
