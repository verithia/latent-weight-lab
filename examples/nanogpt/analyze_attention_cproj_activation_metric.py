#!/usr/bin/env python3
"""Gate the attention c_proj BlockFHT chart in activation-output geometry.

This is a zero-update endpoint diagnostic.  For fixed validation batches it
captures each selected attention output projection's input ``H`` and output
gradient ``R``, forms the transient dense gradient ``R.T @ H`` and its Muon
polar target, then solves two matrix-free least-squares problems over the
actual fixed BlockFHT tangent:

* ordinary Euclidean weight-space projection;
* activation-weighted projection minimizing ``||H (J dz - U).T||_F``.

An independent-seed BlockFHT tight frame with the same coordinate count is the
random-orientation control.  No model parameter is updated and no dense state
is retained after a cell is scored.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from examples.nanogpt.analyze_residual_compatibility import (
    fixed_validation_batches,
)
from examples.nanogpt.model import GPT, GPTConfig
from examples.nanogpt.muon import zeropower_via_newtonschulz5
from latent_weight_lab.block_fht import (
    BlockFHTLinear,
    block_fht_grad_latent,
    block_fht_slice,
)


SCHEMA_VERSION = "mai_124m_attention_cproj_activation_metric_v1"
PLAN_SCHEMA = "mai_124m_attention_cproj_activation_metric_fallback_plan_v1"
DECISION_SCHEMA = "mai_124m_attention_cproj_ratio10_promotion_result_v1"
RANDOM_SEED_OFFSET = 15485863


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite(item) for item in value)
    return True


def validate_activation(
    plan: dict[str, Any],
    decision: dict[str, Any],
    run_result: dict[str, Any],
    *,
    config_sha256: str,
    dataset_manifest_sha256: str,
    checkpoint_sha256: str,
) -> None:
    """Fail closed on the conditional plan and endpoint identity."""
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unexpected activation-metric plan schema")
    if decision.get("schema_version") != DECISION_SCHEMA:
        raise ValueError("unexpected capacity-decision schema")
    if decision.get("decision", {}).get("classification") != (
        "REJECT_CPROJ_RATIO10_TRANSFER"
    ):
        raise ValueError("capacity decision does not activate the fallback")
    source = plan["source_state"]
    run = run_result.get("run", {})
    expected = {
        "config_sha256": source["required_config_sha256"],
        "dataset_manifest_sha256": source[
            "required_dataset_manifest_sha256"
        ],
        "fixed_eval_indices_sha256": source[
            "required_fixed_eval_indices_sha256"
        ],
    }
    for field, value in expected.items():
        if run.get(field) != value:
            raise ValueError(f"run endpoint identity mismatch: {field}")
    if run.get("exit_code") != 0 or run.get("classification") != "clean":
        raise ValueError("run endpoint is not a clean terminal result")
    if config_sha256 != source["required_config_sha256"]:
        raise ValueError("production config SHA-256 mismatch")
    if dataset_manifest_sha256 != source["required_dataset_manifest_sha256"]:
        raise ValueError("dataset manifest SHA-256 mismatch")
    if checkpoint_sha256 != run.get("checkpoint_sha256"):
        raise ValueError("checkpoint SHA-256 mismatch")


def load_endpoint_model(checkpoint_path: Path, device: str) -> GPT:
    """Construct fixed CPU-seeded buffers before moving the endpoint to CUDA."""
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model = GPT(GPTConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()
    return model


@dataclass(frozen=True)
class TangentChart:
    size: int
    out_features: int
    in_features: int
    latent_dim: int
    layers: int
    seed: int
    weight_scale: float
    output_gain: torch.Tensor

    def jvp(self, coordinates: torch.Tensor) -> torch.Tensor:
        if coordinates.numel() != self.latent_dim:
            raise ValueError("coordinate dimension mismatch")
        generated = block_fht_slice(
            coordinates.reshape(-1),
            self.size,
            self.layers,
            self.seed,
            0,
            self.size,
        ).view(self.out_features, self.in_features)
        return (
            self.weight_scale
            * self.output_gain.to(
                device=generated.device, dtype=generated.dtype
            ).view(-1, 1)
            * generated
        )

    def vjp(self, weight_cotangent: torch.Tensor) -> torch.Tensor:
        if weight_cotangent.shape != (
            self.out_features,
            self.in_features,
        ):
            raise ValueError("weight cotangent shape mismatch")
        weighted = (
            self.weight_scale
            * self.output_gain.to(
                device=weight_cotangent.device,
                dtype=weight_cotangent.dtype,
            ).view(-1, 1)
            * weight_cotangent
        )
        template = torch.zeros(
            self.latent_dim,
            device=weight_cotangent.device,
            dtype=weight_cotangent.dtype,
        )
        return block_fht_grad_latent(
            template,
            weighted.reshape(-1),
            self.size,
            self.layers,
            self.seed,
        ).reshape(-1)


def chart_from_module(
    module: BlockFHTLinear,
    *,
    ratio: float,
    seed: int,
) -> TangentChart:
    size = int(module.in_features * module.out_features)
    latent_dim = max(1, round(size * float(ratio)))
    if module.output_gain is None:
        output_gain = torch.ones(
            module.out_features,
            device=module.generator.latent.device,
            dtype=torch.float32,
        )
    else:
        output_gain = module.output_gain.detach().float()
    return TangentChart(
        size=size,
        out_features=int(module.out_features),
        in_features=int(module.in_features),
        latent_dim=latent_dim,
        layers=int(module.generator.layers),
        seed=int(seed),
        weight_scale=float(module.weight_scale),
        output_gain=output_gain,
    )


def conjugate_gradient(
    operator: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    *,
    tolerance: float,
    max_iterations: int,
) -> tuple[torch.Tensor, dict[str, float | int | bool]]:
    """Solve a positive-semidefinite normal equation from the zero point."""
    x = torch.zeros_like(rhs)
    residual = rhs.clone()
    direction = residual.clone()
    initial = residual.double().norm().clamp_min(1e-30)
    squared = torch.dot(residual.double(), residual.double())
    relative = float(squared.sqrt() / initial)
    converged = relative <= tolerance
    iterations = 0
    for iteration in range(1, int(max_iterations) + 1):
        if converged:
            break
        product = operator(direction)
        curvature = torch.dot(direction.double(), product.double())
        if not torch.isfinite(curvature) or curvature <= 0.0:
            break
        alpha = squared / curvature
        x.add_(direction, alpha=float(alpha))
        residual.add_(product, alpha=-float(alpha))
        new_squared = torch.dot(residual.double(), residual.double())
        relative = float(new_squared.sqrt() / initial)
        iterations = iteration
        converged = relative <= tolerance
        if converged:
            squared = new_squared
            break
        beta = new_squared / squared.clamp_min(1e-300)
        direction.mul_(float(beta)).add_(residual)
        squared = new_squared
    return x, {
        "iterations": iterations,
        "relative_normal_residual": relative,
        "converged": converged,
    }


def solve_projection(
    chart: TangentChart,
    target: torch.Tensor,
    activations: torch.Tensor | None,
    *,
    tolerance: float,
    max_iterations: int,
) -> tuple[torch.Tensor, dict[str, float | int | bool]]:
    """Return ``J dz`` under Euclidean or activation-output geometry."""
    if activations is None:
        rhs = chart.vjp(target)

        def normal(coordinates: torch.Tensor) -> torch.Tensor:
            return chart.vjp(chart.jvp(coordinates))

    else:
        activations = activations.float()
        target_output = activations @ target.float().T
        rhs = chart.vjp(target_output.T @ activations)

        def normal(coordinates: torch.Tensor) -> torch.Tensor:
            weight = chart.jvp(coordinates)
            output = activations @ weight.T
            return chart.vjp(output.T @ activations)

    coordinates, diagnostics = conjugate_gradient(
        normal,
        rhs,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    return chart.jvp(coordinates), diagnostics


def output_metrics(
    activations: torch.Tensor,
    dense_gradient: torch.Tensor,
    target: torch.Tensor,
    projected: torch.Tensor,
) -> dict[str, float]:
    target_output = activations.float() @ target.float().T
    projected_output = activations.float() @ projected.float().T
    target_energy = target_output.double().square().sum().clamp_min(1e-30)
    projected_energy = projected_output.double().square().sum()
    dot = (target_output.double() * projected_output.double()).sum()
    projected_norm = projected_energy.sqrt()
    gradient_energy = dense_gradient.double().square().sum().clamp_min(1e-30)
    weight_energy = projected.double().square().sum().clamp_min(1e-30)
    gradient_dot = (dense_gradient.double() * projected.double()).sum()
    return {
        "target_output_energy": float(target_energy),
        "projected_output_energy": float(projected_energy),
        "activation_output_recovery": float(projected_energy / target_energy),
        "activation_output_cosine": float(
            dot / (target_energy.sqrt() * projected_norm).clamp_min(1e-30)
        ),
        "activation_projection_identity_residual": float(
            abs(dot - projected_energy) / target_energy
        ),
        "task_gradient_cosine": float(
            gradient_dot
            / (gradient_energy.sqrt() * weight_energy.sqrt()).clamp_min(1e-30)
        ),
        "task_gradient_alignment_per_target_output_energy": float(
            gradient_dot / target_energy
        ),
    }


class AttentionCProjCollector:
    def __init__(self, model: torch.nn.Module, layers: list[int]) -> None:
        self.layers = set(layers)
        self.activations: dict[int, torch.Tensor] = {}
        self.output_gradients: dict[int, torch.Tensor] = {}
        self.handles = []
        for layer, block in enumerate(model.transformer.h):
            if layer in self.layers:
                self.handles.append(
                    block.attn.c_proj.register_forward_hook(self._hook(layer))
                )

    def _hook(self, layer: int):
        def hook(_module, inputs, output):
            self.activations[layer] = (
                inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
            )

            def capture(gradient: torch.Tensor) -> None:
                self.output_gradients[layer] = (
                    gradient.detach()
                    .float()
                    .reshape(-1, gradient.shape[-1])
                )

            output.register_hook(capture)

        return hook

    def clear(self) -> None:
        self.activations.clear()
        self.output_gradients.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def aggregate(rows: list[dict[str, Any]], plan: dict[str, Any]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate zero cells")

    def summed(key: str, arm: str | None = None) -> float:
        return sum(
            float(row[key])
            for row in rows
            if arm is None or row["arm"] == arm
        )

    arms: dict[str, Any] = {}
    for arm in sorted({str(row["arm"]) for row in rows}):
        selected = [row for row in rows if row["arm"] == arm]
        target_energy = sum(float(row["target_output_energy"]) for row in selected)
        projected_energy = sum(
            float(row["projected_output_energy"]) for row in selected
        )
        arms[arm] = {
            "cells": len(selected),
            "activation_output_recovery": projected_energy
            / max(target_energy, 1e-30),
            "minimum_activation_output_cosine": min(
                float(row["activation_output_cosine"]) for row in selected
            ),
            "maximum_relative_normal_residual": max(
                float(row["relative_normal_residual"]) for row in selected
            ),
            "maximum_projection_identity_residual": max(
                float(row["activation_projection_identity_residual"])
                for row in selected
            ),
            "mean_task_gradient_cosine": sum(
                float(row["task_gradient_cosine"]) for row in selected
            )
            / len(selected),
        }
    candidate = arms["ratio10_activation"]
    euclidean = arms["ratio10_euclidean"]
    random = arms["ratio10_random_activation"]
    by_layer = {}
    for layer in sorted({int(row["layer"]) for row in rows}):
        selected = [
            row
            for row in rows
            if int(row["layer"]) == layer
            and row["arm"] == "ratio10_activation"
        ]
        by_layer[str(layer)] = {
            "cells": len(selected),
            "minimum_activation_output_cosine": min(
                float(row["activation_output_cosine"]) for row in selected
            ),
            "positive_cosine": all(
                float(row["activation_output_cosine"]) > 0.0
                for row in selected
            ),
        }
    numerical = plan["offline_diagnostic"]["numerical_guards"]
    gate_spec = plan["preregistered_gate"]
    valid_cells = len(
        {
            (int(row["batch"]), int(row["layer"]))
            for row in rows
            if row["arm"] == "ratio10_activation"
        }
    )
    multipliers = {
        "activation_over_euclidean": float(
            candidate["activation_output_recovery"]
            / max(float(euclidean["activation_output_recovery"]), 1e-30)
        ),
        "activation_over_equal_coordinate_random": float(
            candidate["activation_output_recovery"]
            / max(float(random["activation_output_recovery"]), 1e-30)
        ),
    }
    gate = {
        "all_metrics_finite": all_finite(rows) and all_finite(arms),
        "minimum_valid_cells": valid_cells
        >= int(numerical["minimum_valid_cells"]),
        "normal_residual": max(
            float(row["relative_normal_residual"]) for row in rows
        )
        <= float(numerical["maximum_relative_normal_residual"]),
        "ratio10_activation_output_recovery": float(
            candidate["activation_output_recovery"]
        )
        >= float(gate_spec["minimum_ratio10_activation_output_recovery"]),
        "multiplier_over_ratio10_euclidean": multipliers[
            "activation_over_euclidean"
        ]
        >= float(
            gate_spec["minimum_multiplier_over_ratio10_euclidean_projection"]
        ),
        "multiplier_over_equal_coordinate_random": multipliers[
            "activation_over_equal_coordinate_random"
        ]
        >= float(
            gate_spec["minimum_multiplier_over_equal_coordinate_random_control"]
        ),
        "positive_alignment_in_each_sampled_layer": all(
            bool(value["positive_cosine"]) for value in by_layer.values()
        ),
    }
    passed = all(gate.values())
    return {
        "valid_cells": valid_cells,
        "arms": arms,
        "multipliers": multipliers,
        "by_layer": by_layer,
        "gate": gate,
        "passed": passed,
        "decision": (
            "AUTHORIZE_ACTIVATION_METRIC_TRUST_REGION_IMPLEMENTATION"
            if passed
            else "REJECT_FIXED_RANDOM_CPROJ_CHART"
        ),
        "next_action": (
            gate_spec["pass_action"] if passed else gate_spec["fail_action"]
        ),
        "language_model_training_authorized": False,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--capacity-decision", required=True, type=Path)
    parser.add_argument("--run-result", required=True, type=Path)
    parser.add_argument("--production-config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--sample-seed", type=int, default=20260805)
    parser.add_argument("--cg-tolerance", type=float, default=1e-6)
    parser.add_argument("--cg-max-iterations", type=int, default=96)
    parser.add_argument("--cg-double-retry-iterations", type=int, default=512)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    args = parser.parse_args()

    started = time.time()
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    plan = json.loads(args.plan.read_text())
    decision = json.loads(args.capacity_decision.read_text())
    run_result = json.loads(args.run_result.read_text())
    config = json.loads(args.production_config.read_text())
    manifest = args.data_dir / "manifest.json"
    validate_activation(
        plan,
        decision,
        run_result,
        config_sha256=file_sha256(args.production_config),
        dataset_manifest_sha256=file_sha256(manifest),
        checkpoint_sha256=file_sha256(args.checkpoint),
    )
    layers = [int(value) for value in plan["source_state"]["layers"]]
    if int(args.batches) * len(layers) < int(
        plan["offline_diagnostic"]["numerical_guards"]["minimum_valid_cells"]
    ):
        raise ValueError("requested batches cannot produce enough valid cells")
    expected_targets = {"attn.c_attn.qk_headwise", "attn.c_proj"}
    if set(config["block_fht_targets"]) != expected_targets:
        raise ValueError("production config is not the registered QK+c_proj arm")
    if float(config["block_fht_latent_ratios"]["attn.c_proj"]) != 0.10:
        raise ValueError("production config is not the ratio-0.10 endpoint")

    batches = fixed_validation_batches(
        args.data_dir,
        int(args.batch_size),
        int(args.block_size) + 1,
        int(args.batches),
        int(args.sample_seed),
    )
    model = load_endpoint_model(args.checkpoint, args.device)
    model.prepare_block_fht_cache(dtype=torch.float32)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    collector = AttentionCProjCollector(model, layers)
    rows: list[dict[str, Any]] = []
    try:
        for batch_index, tokens in enumerate(batches):
            collector.clear()
            model.zero_grad(set_to_none=True)
            tokens = tokens.to(args.device)
            inputs = tokens[:, :-1].contiguous()
            targets = tokens[:, 1:].contiguous()
            _logits, loss = model(inputs, targets)
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("non-finite diagnostic task loss")
            loss.backward()
            if set(collector.activations) != set(layers) or set(
                collector.output_gradients
            ) != set(layers):
                raise RuntimeError("attention c_proj capture is incomplete")
            print(
                f"batch {batch_index + 1}/{len(batches)} loss={float(loss):.6f}",
                flush=True,
            )
            for layer in layers:
                module = model.transformer.h[layer].attn.c_proj
                if not isinstance(module, BlockFHTLinear):
                    raise TypeError("selected c_proj is not BlockFHTLinear")
                activations = collector.activations[layer]
                output_gradient = collector.output_gradients[layer]
                dense_gradient = output_gradient.T @ activations
                target = zeropower_via_newtonschulz5(
                    dense_gradient, steps=int(args.muon_ns_steps)
                )
                base_seed = int(module.generator.seed)
                charts = {
                    "ratio01": chart_from_module(
                        module, ratio=0.01, seed=base_seed
                    ),
                    "ratio10": chart_from_module(
                        module, ratio=0.10, seed=base_seed
                    ),
                    "ratio10_random": chart_from_module(
                        module,
                        ratio=0.10,
                        seed=base_seed + RANDOM_SEED_OFFSET,
                    ),
                }
                arms = {
                    "ratio01_euclidean": (
                        charts["ratio01"],
                        None,
                    ),
                    "ratio01_activation": (
                        charts["ratio01"],
                        activations,
                    ),
                    "ratio10_euclidean": (
                        charts["ratio10"],
                        None,
                    ),
                    "ratio10_activation": (
                        charts["ratio10"],
                        activations,
                    ),
                    "ratio10_random_activation": (
                        charts["ratio10_random"],
                        activations,
                    ),
                }
                for arm, (chart, metric_activations) in arms.items():
                    projected, diagnostics = solve_projection(
                        chart,
                        target,
                        metric_activations,
                        tolerance=float(args.cg_tolerance),
                        max_iterations=int(args.cg_max_iterations),
                    )
                    diagnostics["solver_dtype"] = "float32"
                    if (
                        not bool(diagnostics["converged"])
                        and metric_activations is not None
                    ):
                        # Near-null activation directions can reach the FP32
                        # residual floor before satisfying the preregistered
                        # normal-equation guard.  Retry only those solves in
                        # FP64 through the same exact Jacobian/adjoint; this
                        # changes numerical precision, not the objective.
                        projected, diagnostics = solve_projection(
                            chart,
                            target.double(),
                            metric_activations.double(),
                            tolerance=float(args.cg_tolerance),
                            max_iterations=int(
                                args.cg_double_retry_iterations
                            ),
                        )
                        diagnostics["solver_dtype"] = "float64_retry"
                    metrics = output_metrics(
                        activations,
                        dense_gradient,
                        target,
                        projected,
                    )
                    rows.append(
                        {
                            "batch": batch_index,
                            "layer": layer,
                            "arm": arm,
                            "task_loss": float(loss),
                            "activation_rows": int(activations.shape[0]),
                            "latent_coordinates": chart.latent_dim,
                            **diagnostics,
                            **metrics,
                        }
                    )
                print(
                    f"  layer {layer}: ratio10 activation recovery="
                    f"{rows[-2]['activation_output_recovery']:.6f} "
                    f"random={rows[-1]['activation_output_recovery']:.6f}",
                    flush=True,
                )
                del activations, output_gradient, dense_gradient, target
            for module in model.modules():
                if isinstance(module, BlockFHTLinear) and module._cached_weight is not None:
                    module._cached_weight.grad = None
    finally:
        collector.close()

    summary = aggregate(rows, plan)
    args.output.mkdir(parents=True, exist_ok=True)
    cells_path = args.output / "attention_cproj_activation_metric_cells.csv"
    write_csv(cells_path, rows)
    repo = Path(__file__).resolve().parents[2]
    result = {
        "schema_version": SCHEMA_VERSION,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "started_at": started_at,
        "source_commit": git_commit(repo),
        "source_sha256": file_sha256(Path(__file__)),
        "plan": {"path": str(args.plan), "sha256": file_sha256(args.plan)},
        "capacity_decision": {
            "path": str(args.capacity_decision),
            "sha256": file_sha256(args.capacity_decision),
        },
        "run_result": {
            "path": str(args.run_result),
            "sha256": file_sha256(args.run_result),
        },
        "production_config": {
            "path": str(args.production_config),
            "sha256": file_sha256(args.production_config),
        },
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": file_sha256(args.checkpoint),
        },
        "dataset_manifest": {
            "path": str(manifest),
            "sha256": file_sha256(manifest),
        },
        "protocol": {
            "layers": layers,
            "batches": int(args.batches),
            "batch_size": int(args.batch_size),
            "block_size": int(args.block_size),
            "sample_seed": int(args.sample_seed),
            "cg_tolerance": float(args.cg_tolerance),
            "cg_max_iterations": int(args.cg_max_iterations),
            "cg_double_retry_iterations": int(
                args.cg_double_retry_iterations
            ),
            "muon_ns_steps": int(args.muon_ns_steps),
            "random_control_seed_offset": RANDOM_SEED_OFFSET,
            "parameter_updates": 0,
            "dense_optimizer_state_retained": False,
        },
        "summary": summary,
        "cells_csv": {"path": str(cells_path), "sha256": file_sha256(cells_path)},
        "elapsed_seconds": time.time() - started,
    }
    result_path = args.output / "attention_cproj_activation_metric_result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
