#!/usr/bin/env python3
"""Necessary terminal image gate for an exact-base quadratic attention chart."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import torch

from examples.nanogpt.analyze_attention_paper_activation_oracle import (
    AttentionFunctionalMetric,
    all_finite,
    file_sha256,
    git_commit,
    terminal_attention_metrics,
    write_csv,
)
from examples.nanogpt.analyze_mlp_cproj_paper_activation_oracle import cgls, explained_energy
from examples.nanogpt.analyze_residual_compatibility import fixed_validation_batches
from examples.nanogpt.train import require_block_fht_native_extension
from latent_weight_lab.block_fht import block_fht_grad_latent, block_fht_slice


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_SCHEMA = "mai_124m_attention_quadratic_image_oracle_plan_v1"
RESULT_SCHEMA = "mai_124m_attention_quadratic_image_oracle_result_v1"


def centered(value: torch.Tensor) -> torch.Tensor:
    return value - value.mean()


def quadratic_map(
    linear: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
    scale: float,
    latent_std: float,
    factor: float,
) -> torch.Tensor:
    return factor * (linear + scale * centered(first * second) / latent_std)


def quadratic_jvp(
    first: torch.Tensor,
    second: torch.Tensor,
    dlinear: torch.Tensor,
    dfirst: torch.Tensor,
    dsecond: torch.Tensor,
    scale: float,
    latent_std: float,
    factor: float,
) -> torch.Tensor:
    return factor * (
        dlinear
        + scale * centered(dfirst * second + first * dsecond) / latent_std
    )


def quadratic_component_cotangents(
    cotangent: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
    scale: float,
    latent_std: float,
    factor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    centered_cotangent = centered(cotangent)
    return (
        factor * cotangent,
        factor * scale * centered_cotangent * second / latent_std,
        factor * scale * centered_cotangent * first / latent_std,
    )


def _target_tensor(payload: dict[str, Any], layer: int, spec: dict[str, Any]) -> torch.Tensor:
    tensor = payload["parameters"][f"transformer.h.{layer}.{spec['parameter']}"]["weight_before_step"].float()
    if spec["slice"] == "final n_embd rows":
        n_embd = int(payload["model_config"]["n_embd"])
        tensor = tensor[2 * n_embd :]
    return tensor


def validate_plan(plan: dict[str, Any], args: argparse.Namespace) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unexpected quadratic image plan schema")
    identity = plan["identity"]
    for path, expected in {
        Path(__file__): identity["entrypoint_sha256"],
        REPO_ROOT / identity["dense_config"]: identity["dense_config_sha256"],
        REPO_ROOT / identity["parent_result"]: identity["parent_result_sha256"],
        args.terminal_checkpoint: identity["terminal_checkpoint_sha256"],
        args.data_dir / "manifest.json": identity["dataset_manifest_sha256"],
        args.initial_probe: identity["initial_probe_sha256"],
        args.terminal_probe: identity["terminal_probe_sha256"],
    }.items():
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"pinned identity mismatch: {path}")
    protocol = plan["protocol"]
    frozen = {
        "parameter_updates": 0,
        "latent_ratio": 0.01,
        "block_fht_layers": 2,
        "block_fht_seed": 1000,
        "quadratic_scale": 1.0,
        "quadratic_seed_offset": 104729,
        "outer_gauss_newton_iterations": 8,
        "inner_cgls_iterations": 12,
        "fit_metric_seed": 20260809,
        "eval_metric_seed": 20260810,
        "terminal_step": 2372,
    }
    for field, expected in frozen.items():
        if protocol.get(field) != expected:
            raise ValueError(f"frozen protocol changed: {field}")
    if plan["decision_rule"]["thresholds"] != {
        "aggregate_eval_image_recovery_minimum": 0.80,
        "minimum_layer_eval_image_recovery": 0.60,
    }:
        raise ValueError("quadratic image thresholds changed")
    if plan["authorization"] != {
        "tangent_oracle": False,
        "model_implementation": False,
        "mfu_preflight": False,
        "language_model_training": False,
        "larger_rung": False,
    }:
        raise ValueError("quadratic image authorization changed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--initial-probe", required=True, type=Path)
    parser.add_argument("--terminal-probe", required=True, type=Path)
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
    initial_payload = torch.load(args.initial_probe, map_location="cpu", weights_only=False)
    terminal_payload = torch.load(args.terminal_probe, map_location="cpu", weights_only=False)
    run_identities = {
        initial_payload["run_identity_sha256"],
        terminal_payload["run_identity_sha256"],
    }
    if run_identities != {plan["identity"]["probe_run_identity_sha256"]}:
        raise ValueError("probe run identity changed")

    def batches(seed: int) -> list[torch.Tensor]:
        return fixed_validation_batches(
            args.data_dir,
            int(protocol["metric_batch_size"]),
            int(protocol["metric_block_size"]),
            int(protocol["metric_batches"]),
            seed,
        )

    print("collecting disjoint terminal attention metrics", flush=True)
    fit_inputs = terminal_attention_metrics(
        args.terminal_checkpoint, batches(int(protocol["fit_metric_seed"])), layers, args.device
    )
    eval_inputs = terminal_attention_metrics(
        args.terminal_checkpoint, batches(int(protocol["eval_metric_seed"])), layers, args.device
    )
    config = json.loads((REPO_ROOT / plan["identity"]["dense_config"]).read_text())
    latent_std = float(config.get("block_fht_latent_init_std", 0.02))
    q = float(protocol["quadratic_scale"])
    seed_offset = int(protocol["quadratic_seed_offset"])
    rows: list[dict[str, Any]] = []
    for layer in layers:
        print(f"analyzing layer {layer}", flush=True)
        for target, spec in protocol["targets"].items():
            initial = _target_tensor(initial_payload, layer, spec).to(args.device)
            current = _target_tensor(terminal_payload, layer, spec).to(args.device)
            target_delta = current - initial
            size = initial.numel()
            latent_dim = max(1, round(size * float(protocol["latent_ratio"])))
            block_size = 1 << (latent_dim - 1).bit_length()
            normalization = 1.0 / math.sqrt(1.0 + q * q * latent_dim / block_size)
            factor = normalization * float(spec["target_std"]) / latent_std
            seed = int(protocol["block_fht_seed"]) + int(spec["seed_stride"]) * layer + int(spec["seed_offset"])
            template = torch.zeros(latent_dim, device=args.device)

            def transform(coordinate: torch.Tensor, which_seed: int) -> torch.Tensor:
                return block_fht_slice(
                    coordinate, size, int(protocol["block_fht_layers"]), which_seed, 0, size
                ).view_as(initial)

            def adjoint(weight: torch.Tensor, which_seed: int) -> torch.Tensor:
                return block_fht_grad_latent(
                    template,
                    weight.reshape(-1).contiguous(),
                    size,
                    int(protocol["block_fht_layers"]),
                    which_seed,
                    0,
                    size,
                )

            seeds = (seed, seed + seed_offset, seed + 2 * seed_offset)
            fit_metric = AttentionFunctionalMetric(target=target, **fit_inputs[layer])
            eval_metric = AttentionFunctionalMetric(target=target, **eval_inputs[layer])
            fit_target = fit_metric.apply(target_delta)
            eval_target = eval_metric.apply(target_delta)
            coordinate = template.clone()
            fit_losses: list[float] = []
            accepted_steps = 0
            for _outer in range(int(protocol["outer_gauss_newton_iterations"])):
                a, b, c = (transform(coordinate, value) for value in seeds)
                mapped = quadratic_map(a, b, c, q, latent_std, factor)
                residual = fit_target - fit_metric.apply(mapped)
                fit_losses.append(float(residual.double().square().sum()))

                def apply_j(delta_coordinate: torch.Tensor) -> torch.Tensor:
                    da, db, dc = (transform(delta_coordinate, value) for value in seeds)
                    return fit_metric.apply(
                        quadratic_jvp(b, c, da, db, dc, q, latent_std, factor)
                    )

                def adjoint_j(output: torch.Tensor) -> torch.Tensor:
                    wa, wb, wc = quadratic_component_cotangents(
                        fit_metric.adjoint(output), b, c, q, latent_std, factor
                    )
                    return sum(adjoint(weight, value) for weight, value in zip((wa, wb, wc), seeds))

                step, _prediction, _iterations = cgls(
                    apply_j,
                    adjoint_j,
                    residual,
                    template,
                    int(protocol["inner_cgls_iterations"]),
                )
                best_coordinate = coordinate
                best_loss = fit_losses[-1]
                for multiplier in protocol["line_search_multipliers"]:
                    candidate = coordinate + float(multiplier) * step
                    ca, cb, cc = (transform(candidate, value) for value in seeds)
                    candidate_map = quadratic_map(ca, cb, cc, q, latent_std, factor)
                    candidate_loss = float(
                        (fit_target - fit_metric.apply(candidate_map)).double().square().sum()
                    )
                    if candidate_loss < best_loss:
                        best_loss = candidate_loss
                        best_coordinate = candidate
                if best_coordinate.data_ptr() == coordinate.data_ptr():
                    break
                coordinate = best_coordinate
                accepted_steps += 1

            a, b, c = (transform(coordinate, value) for value in seeds)
            mapped = quadratic_map(a, b, c, q, latent_std, factor)
            fit_prediction = fit_metric.apply(mapped)
            eval_prediction = eval_metric.apply(mapped)
            fit_recovery, fit_energy = explained_energy(fit_target, fit_prediction)
            eval_recovery, eval_energy = explained_energy(eval_target, eval_prediction)
            euclidean_recovery, euclidean_energy = explained_energy(target_delta, mapped)
            rows.append(
                {
                    "target": target,
                    "layer": layer,
                    "latent_dim": latent_dim,
                    "latent_ratio": latent_dim / size,
                    "seed": seed,
                    "accepted_gauss_newton_steps": accepted_steps,
                    "initial_fit_loss": fit_losses[0],
                    "final_fit_loss": float((fit_target - fit_prediction).double().square().sum()),
                    "fit_functional_image_recovery": fit_recovery,
                    "eval_functional_image_recovery": eval_recovery,
                    "eval_functional_image_energy": eval_energy,
                    "euclidean_image_recovery": euclidean_recovery,
                    "euclidean_image_energy": euclidean_energy,
                    "coordinate_rms": float(coordinate.square().mean().sqrt()),
                    "maximum_abs_mapped_delta": float(mapped.abs().amax()),
                    "finite": bool(torch.isfinite(mapped).all() and torch.isfinite(coordinate).all()),
                }
            )

    summaries: dict[str, Any] = {}
    thresholds = plan["decision_rule"]["thresholds"]
    for target in protocol["targets"]:
        selected = [row for row in rows if row["target"] == target]
        energy = sum(float(row["eval_functional_image_energy"]) for row in selected)
        aggregate = sum(
            float(row["eval_functional_image_recovery"])
            * float(row["eval_functional_image_energy"])
            for row in selected
        ) / max(energy, 1e-30)
        minimum = min(float(row["eval_functional_image_recovery"]) for row in selected)
        checks = {
            "finite": all(bool(row["finite"]) for row in selected),
            "aggregate_image": aggregate >= float(thresholds["aggregate_eval_image_recovery_minimum"]),
            "every_layer_image": minimum >= float(thresholds["minimum_layer_eval_image_recovery"]),
        }
        summaries[target] = {
            "aggregate_eval_functional_image_recovery": aggregate,
            "minimum_layer_eval_functional_image_recovery": minimum,
            "checks": checks,
            "passed": all(checks.values()),
        }

    args.output_dir.mkdir(parents=True)
    cells_path = args.output_dir / "attention_quadratic_image_oracle_cells.csv"
    write_csv(cells_path, rows)
    passed = [target for target, value in summaries.items() if value["passed"]]
    result = {
        "schema_version": RESULT_SCHEMA,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "classification": (
            "ATTENTION_QUADRATIC_IMAGE_ORACLE_HAS_PASSING_TARGET"
            if passed
            else "ATTENTION_QUADRATIC_IMAGE_ORACLE_REJECT_ALL"
        ),
        "execution": {
            "host": "PRO6",
            "device": args.device,
            "git_commit": git_commit(),
            "parameter_updates": 0,
            "elapsed_seconds": time.time() - started,
        },
        "identity": {
            "plan_sha256": file_sha256(args.plan),
            "initial_probe_sha256": file_sha256(args.initial_probe),
            "terminal_probe_sha256": file_sha256(args.terminal_probe),
            "probe_run_identity_sha256": next(iter(run_identities)),
            "terminal_checkpoint_sha256": file_sha256(args.terminal_checkpoint),
            "dataset_manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
        },
        "protocol": protocol,
        "summaries": summaries,
        "decision": {
            "passed_targets": passed,
            "tangent_oracle_authorized": bool(passed),
            "model_implementation_authorized": False,
            "mfu_preflight_authorized": False,
            "language_model_training_authorized": False,
            "larger_rung_authorized": False,
        },
        "cells_csv": {"path": str(cells_path), "sha256": file_sha256(cells_path)},
        "all_reported_values_finite": all_finite(summaries),
    }
    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
