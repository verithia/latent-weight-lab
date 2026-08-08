#!/usr/bin/env python3
"""Test an exact-base unbounded residual-cubic attention decoder.

This is a zero-update optimistic representability oracle.  The decoder is

    W(z) = W0 + h_s(Az),       h_s(x) = x + x^3 / (3 s^2),

where ``A`` is the unchanged production-seeded 1% two-stage BlockFHT chart.
The warp is the lowest-order odd smooth nonlinearity that preserves the exact
base and identity tangent without the range/saturation failure of tanh.  Its
Jacobian is ``diag(1 + (Az/s)^2) A``.  State coordinates and tangent
coordinates are oracle fits; tangent coordinates are fitted on one frozen
attention metric and evaluated unchanged on disjoint batches.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_attention_paper_activation_oracle import (
    AttentionFunctionalMetric,
    all_finite,
    file_sha256,
    git_commit,
    terminal_attention_metrics,
    weighted,
    write_csv,
)
from examples.nanogpt.analyze_mlp_cproj_paper_activation_oracle import (
    cgls,
    explained_energy,
)
from examples.nanogpt.analyze_residual_compatibility import fixed_validation_batches
from examples.nanogpt.train import require_block_fht_native_extension
from latent_weight_lab.block_fht import block_fht_grad_latent, block_fht_slice


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_SCHEMA = "mai_124m_attention_residual_cubic_oracle_plan_v1"
RESULT_SCHEMA = "mai_124m_attention_residual_cubic_oracle_result_v1"


def residual_cubic(value: torch.Tensor, scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``h_s(value)`` and its elementwise derivative."""
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("residual-cubic scale must be finite and positive")
    normalized = value / scale
    return value + value * normalized.square() / 3.0, 1.0 + normalized.square()


def inverse_residual_cubic(
    target: torch.Tensor, scale: float, iterations: int = 16
) -> torch.Tensor:
    """Invert the strictly increasing residual cubic with Newton iterations."""
    value = target.clone()
    for _ in range(iterations):
        mapped, derivative = residual_cubic(value, scale)
        value = value - (mapped - target) / derivative
    return value


def classify_target(
    summary: dict[str, float | bool], thresholds: dict[str, float]
) -> tuple[str, dict[str, bool]]:
    checks = {
        "condition": float(summary["maximum_jacobian_diagonal"])
        <= float(thresholds["maximum_jacobian_diagonal"]),
        "image": float(summary["eval_functional_image_recovery"])
        >= float(thresholds["eval_functional_image_recovery_minimum"]),
        "tangent": float(summary["eval_cubic_tangent_recovery"])
        >= float(thresholds["eval_cubic_tangent_recovery_minimum"]),
        "cubic_gain": float(summary["eval_cubic_gain_over_identity"])
        >= float(thresholds["eval_cubic_gain_over_identity_minimum"]),
    }
    return (
        "ATTENTION_RESIDUAL_CUBIC_ORACLE_PASS"
        if all(checks.values())
        else "ATTENTION_RESIDUAL_CUBIC_ORACLE_REJECT",
        checks,
    )


def validate_plan(plan: dict[str, Any], args: argparse.Namespace) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unexpected residual-cubic oracle plan schema")
    identity = plan["identity"]
    pinned = {
        Path(__file__): identity["entrypoint_sha256"],
        REPO_ROOT / identity["dense_config"]: identity["dense_config_sha256"],
        REPO_ROOT / identity["parent_result"]: identity["parent_result_sha256"],
        args.terminal_checkpoint: identity["terminal_checkpoint_sha256"],
        args.data_dir / "manifest.json": identity["dataset_manifest_sha256"],
    }
    for path, expected in pinned.items():
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"pinned identity mismatch: {path}")
    for name, expected in identity["probe_sha256"].items():
        path = args.probe_dir / name
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"optimizer probe mismatch: {path}")
    protocol = plan["protocol"]
    frozen = {
        "parameter_updates": 0,
        "latent_ratio": 0.01,
        "block_fht_layers": 2,
        "block_fht_seed": 1000,
        "decoder": "exact_base_residual_cubic",
        "scale_calibration": "per_target_layer_max_abs_dense_delta_over_discovery_steps",
        "minimum_scale": 1e-8,
        "inverse_iterations": 16,
        "cgls_iterations": 32,
        "fit_metric_seed": 20260809,
        "eval_metric_seed": 20260810,
    }
    for field, expected in frozen.items():
        if protocol.get(field) != expected:
            raise ValueError(f"frozen protocol changed: {field}")
    if plan["decision_rule"]["thresholds"] != {
        "maximum_jacobian_diagonal": 10.0,
        "eval_functional_image_recovery_minimum": 0.80,
        "eval_cubic_tangent_recovery_minimum": 0.80,
        "eval_cubic_gain_over_identity_minimum": 0.05,
    }:
        raise ValueError("residual-cubic oracle thresholds changed")
    if plan["authorization"] != {
        "model_implementation": False,
        "mfu_preflight": False,
        "language_model_training": False,
        "larger_rung": False,
    }:
        raise ValueError("zero-update oracle authorization changed")


def _target_tensor(payload: dict[str, Any], layer: int, spec: dict[str, Any], field: str) -> torch.Tensor:
    tensor = payload["parameters"][f"transformer.h.{layer}.{spec['parameter']}"][field].float()
    if spec["slice"] == "final n_embd rows":
        n_embd = int(payload["model_config"]["n_embd"])
        tensor = tensor[2 * n_embd :]
    return tensor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--probe-dir", required=True, type=Path)
    parser.add_argument("--terminal-checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    validate_plan(plan, args)
    if args.output_dir.exists():
        raise FileExistsError(f"output already exists: {args.output_dir}")
    require_block_fht_native_extension(True)
    started = time.time()
    protocol = plan["protocol"]
    layers = [int(value) for value in protocol["layers"]]
    steps = [int(value) for value in protocol["steps"]]
    discovery_steps = [int(value) for value in protocol["discovery_steps"]]
    heldout_steps = {int(value) for value in protocol["heldout_steps"]}

    def metric_batches(seed: int) -> list[torch.Tensor]:
        return fixed_validation_batches(
            args.data_dir,
            int(protocol["metric_batch_size"]),
            int(protocol["metric_block_size"]),
            int(protocol["metric_batches"]),
            seed,
        )

    print("collecting disjoint frozen terminal-dense attention metrics", flush=True)
    fit_inputs = terminal_attention_metrics(
        args.terminal_checkpoint,
        metric_batches(int(protocol["fit_metric_seed"])),
        layers,
        args.device,
    )
    eval_inputs = terminal_attention_metrics(
        args.terminal_checkpoint,
        metric_batches(int(protocol["eval_metric_seed"])),
        layers,
        args.device,
    )
    probes: dict[int, dict[str, Any]] = {}
    run_identity = None
    for step in steps:
        payload = torch.load(
            args.probe_dir / f"step_{step:06d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        if run_identity is None:
            run_identity = payload["run_identity_sha256"]
        elif payload["run_identity_sha256"] != run_identity:
            raise ValueError("optimizer probes do not share one run identity")
        probes[step] = payload
    if run_identity != plan["identity"]["probe_run_identity_sha256"]:
        raise ValueError("optimizer probe run identity changed")

    config = json.loads((REPO_ROOT / plan["identity"]["dense_config"]).read_text())
    latent_init_std = float(config.get("block_fht_latent_init_std", 0.02))
    rows: list[dict[str, Any]] = []
    for layer in layers:
        print(f"analyzing layer {layer}", flush=True)
        for target, spec in protocol["targets"].items():
            initial = _target_tensor(probes[steps[0]], layer, spec, "weight_before_step").to(args.device)
            discovery_deltas = [
                _target_tensor(probes[step], layer, spec, "weight_before_step") - initial.cpu()
                for step in discovery_steps
            ]
            scale = max(float(delta.abs().amax()) for delta in discovery_deltas)
            scale = max(scale, float(protocol["minimum_scale"]))
            size = initial.numel()
            latent_dim = max(1, round(size * float(protocol["latent_ratio"])))
            seed = int(protocol["block_fht_seed"]) + int(spec["seed_stride"]) * layer + int(spec["seed_offset"])
            weight_scale = float(spec["target_std"]) / latent_init_std
            template = torch.zeros(latent_dim, device=args.device)

            def apply_a(coordinate: torch.Tensor) -> torch.Tensor:
                return (
                    block_fht_slice(
                        coordinate,
                        size,
                        int(protocol["block_fht_layers"]),
                        seed,
                        0,
                        size,
                    )
                    * weight_scale
                ).view_as(initial)

            def adjoint_a(weight: torch.Tensor) -> torch.Tensor:
                return block_fht_grad_latent(
                    template,
                    (weight.reshape(-1) * weight_scale).contiguous(),
                    size,
                    int(protocol["block_fht_layers"]),
                    seed,
                    0,
                    size,
                )

            fit_metric = AttentionFunctionalMetric(target=target, **fit_inputs[layer])
            eval_metric = AttentionFunctionalMetric(target=target, **eval_inputs[layer])
            for step in steps:
                current = _target_tensor(probes[step], layer, spec, "weight_before_step").to(args.device)
                direction = _target_tensor(probes[step], layer, spec, "applied_direction_per_lr").to(args.device)
                delta = current - initial
                inverse = inverse_residual_cubic(
                    delta, scale, int(protocol["inverse_iterations"])
                )
                latent, _fit, coordinate_iterations = cgls(
                    apply_a,
                    adjoint_a,
                    inverse,
                    template,
                    int(protocol["cgls_iterations"]),
                )
                preactivation = apply_a(latent)
                mapped_delta, derivative = residual_cubic(preactivation, scale)
                fit_image_target = fit_metric.apply(delta)
                eval_image_target = eval_metric.apply(delta)
                fit_image_prediction = fit_metric.apply(mapped_delta)
                eval_image_prediction = eval_metric.apply(mapped_delta)
                fit_image_recovery, _ = explained_energy(fit_image_target, fit_image_prediction)
                eval_image_recovery, image_energy = explained_energy(eval_image_target, eval_image_prediction)
                fit_tangent_target = fit_metric.apply(direction)
                eval_tangent_target = eval_metric.apply(direction)

                def solve_tangent(diagonal: torch.Tensor) -> tuple[float, float, int]:
                    def apply(coordinate: torch.Tensor) -> torch.Tensor:
                        return fit_metric.apply(diagonal * apply_a(coordinate))

                    def adjoint(output: torch.Tensor) -> torch.Tensor:
                        return adjoint_a(diagonal * fit_metric.adjoint(output))

                    coordinate, fit_prediction, iterations = cgls(
                        apply,
                        adjoint,
                        fit_tangent_target,
                        template,
                        int(protocol["cgls_iterations"]),
                    )
                    fit_recovery, _ = explained_energy(fit_tangent_target, fit_prediction)
                    eval_prediction = eval_metric.apply(diagonal * apply_a(coordinate))
                    eval_recovery, _ = explained_energy(eval_tangent_target, eval_prediction)
                    return fit_recovery, eval_recovery, iterations

                cubic_fit, cubic_eval, cubic_iterations = solve_tangent(derivative)
                identity_fit, identity_eval, identity_iterations = solve_tangent(
                    torch.ones_like(derivative)
                )
                rows.append(
                    {
                        "target": target,
                        "layer": layer,
                        "step": step,
                        "heldout": step in heldout_steps,
                        "seed": seed,
                        "latent_dim": latent_dim,
                        "latent_ratio": latent_dim / size,
                        "scale": scale,
                        "maximum_abs_dense_delta_to_scale": float(delta.abs().amax()) / scale,
                        "maximum_abs_preactivation_to_scale": float(preactivation.abs().amax()) / scale,
                        "minimum_jacobian_diagonal": float(derivative.amin()),
                        "mean_jacobian_diagonal": float(derivative.mean()),
                        "maximum_jacobian_diagonal": float(derivative.amax()),
                        "fit_functional_image_recovery": fit_image_recovery,
                        "eval_functional_image_recovery": eval_image_recovery,
                        "eval_functional_image_energy": image_energy,
                        "fit_cubic_tangent_recovery": cubic_fit,
                        "eval_cubic_tangent_recovery": cubic_eval,
                        "fit_identity_tangent_recovery": identity_fit,
                        "eval_identity_tangent_recovery": identity_eval,
                        "eval_cubic_gain_over_identity": cubic_eval - identity_eval,
                        "eval_functional_tangent_energy": float(eval_tangent_target.double().square().sum()),
                        "coordinate_fit_iterations": coordinate_iterations,
                        "cubic_tangent_iterations": cubic_iterations,
                        "identity_tangent_iterations": identity_iterations,
                    }
                )

    summaries: dict[str, Any] = {}
    thresholds = plan["decision_rule"]["thresholds"]
    for target in protocol["targets"]:
        selected = [row for row in rows if row["target"] == target and row["heldout"]]
        summary: dict[str, float | bool] = {
            "maximum_jacobian_diagonal": max(
                float(row["maximum_jacobian_diagonal"])
                for row in rows
                if row["target"] == target
            ),
            "eval_functional_image_recovery": weighted(
                selected,
                "eval_functional_image_recovery",
                "eval_functional_image_energy",
            ),
            "eval_cubic_tangent_recovery": weighted(
                selected,
                "eval_cubic_tangent_recovery",
                "eval_functional_tangent_energy",
            ),
            "eval_identity_tangent_recovery": weighted(
                selected,
                "eval_identity_tangent_recovery",
                "eval_functional_tangent_energy",
            ),
        }
        summary["eval_cubic_gain_over_identity"] = float(
            summary["eval_cubic_tangent_recovery"]
        ) - float(summary["eval_identity_tangent_recovery"])
        classification, checks = classify_target(summary, thresholds)
        summaries[target] = {
            **summary,
            "classification": classification,
            "checks": checks,
            "passed": all(checks.values()),
        }

    args.output_dir.mkdir(parents=True)
    cells_path = args.output_dir / "attention_residual_cubic_oracle_cells.csv"
    write_csv(cells_path, rows)
    passed = [target for target, value in summaries.items() if value["passed"]]
    result = {
        "schema_version": RESULT_SCHEMA,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "classification": (
            "ATTENTION_RESIDUAL_CUBIC_ORACLE_HAS_PASSING_TARGET"
            if passed
            else "ATTENTION_RESIDUAL_CUBIC_ORACLE_REJECT_ALL"
        ),
        "execution": {
            "host": "PRO6",
            "device": args.device,
            "git_commit": git_commit(),
            "entrypoint": "examples.nanogpt.analyze_attention_residual_cubic_oracle",
            "parameter_updates": 0,
            "elapsed_seconds": time.time() - started,
        },
        "identity": {
            "plan_path": str(args.plan),
            "plan_sha256": file_sha256(args.plan),
            "probe_run_identity_sha256": run_identity,
            "terminal_checkpoint_sha256": file_sha256(args.terminal_checkpoint),
            "dataset_manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
        },
        "protocol": protocol,
        "summaries": summaries,
        "decision": {
            "passed_targets": passed,
            "causal_coordinate_transport_authorized": bool(passed),
            "model_implementation_authorized": False,
            "mfu_preflight_authorized": False,
            "language_model_training_authorized": False,
            "larger_rung_authorized": False,
        },
        "cells_csv": {"path": str(cells_path), "sha256": file_sha256(cells_path)},
        "all_reported_values_finite": all_finite(summaries),
        "limitations": [
            "State coordinates and fit-metric tangent coefficients are optimistic oracle fits.",
            "Tangent coordinates are evaluated unchanged on disjoint batches, but not transported across training steps.",
            "A pass authorizes only a separate coordinate-transport gate, never implementation or training.",
        ],
    }
    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
