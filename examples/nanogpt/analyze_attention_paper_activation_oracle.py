#!/usr/bin/env python3
"""Upper-bound the paper's elementwise decoder on dense attention motion.

This is a zero-update representability oracle.  A fixed two-stage BlockFHT
chart is wrapped by the paper-faithful, signed, condition-bounded tanh map and
initialized exactly at the dense step-zero V or attention-c_proj weight.  Each
captured state and exact Muon direction receives an oracle least-squares fit.
Consequently, passing is necessary but not sufficient for a causal compact
decoder; failure cannot be repaired by a different latent optimizer.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from examples.nanogpt.analyze_mlp_cproj_paper_activation_oracle import (
    activated_weight_and_derivative,
    activation_bias,
    activation_scale,
    cgls,
    explained_energy,
)
from examples.nanogpt.analyze_residual_compatibility import (
    fixed_validation_batches,
    load_model,
)
from examples.nanogpt.train import require_block_fht_native_extension
from latent_weight_lab.block_fht import block_fht_grad_latent, block_fht_slice


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_SCHEMA = "mai_124m_attention_paper_activation_oracle_plan_v1"
RESULT_SCHEMA = "mai_124m_attention_paper_activation_oracle_result_v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
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


@dataclass
class AttentionFunctionalMetric:
    """Frozen terminal-dense attention-output metric for one layer/target."""

    target: str
    cproj_inputs: torch.Tensor
    value_sources: torch.Tensor
    output_weight: torch.Tensor

    def apply(self, weight: torch.Tensor) -> torch.Tensor:
        if self.target == "cproj":
            return F.linear(self.cproj_inputs, weight)
        if self.target != "v":
            raise ValueError(f"unsupported attention metric target: {self.target}")
        heads = self.value_sources.shape[1]
        if weight.shape[0] % heads:
            raise ValueError("V output dimension must be divisible by head count")
        head_dim = weight.shape[0] // heads
        states = [
            F.linear(
                self.value_sources[:, head],
                weight[head * head_dim : (head + 1) * head_dim],
            )
            for head in range(heads)
        ]
        return F.linear(torch.cat(states, dim=-1), self.output_weight)

    def adjoint(self, output: torch.Tensor) -> torch.Tensor:
        if self.target == "cproj":
            return output.flatten(0, -2).T @ self.cproj_inputs.flatten(0, -2)
        if self.target != "v":
            raise ValueError(f"unsupported attention metric target: {self.target}")
        hidden_cotangent = F.linear(output, self.output_weight.T)
        heads = self.value_sources.shape[1]
        head_dim = hidden_cotangent.shape[-1] // heads
        gradients = []
        for head in range(heads):
            cotangent = hidden_cotangent[
                ..., head * head_dim : (head + 1) * head_dim
            ].flatten(0, -2)
            source = self.value_sources[:, head].flatten(0, -2)
            gradients.append(cotangent.T @ source)
        return torch.cat(gradients, dim=0)


class AttentionMetricCollector:
    def __init__(self, model: torch.nn.Module, layers: list[int]) -> None:
        self.ln1: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
        self.cproj_inputs: dict[int, list[torch.Tensor]] = {
            layer: [] for layer in layers
        }
        self.handles = []
        for layer, block in enumerate(model.transformer.h):
            if layer not in self.ln1:
                continue
            self.handles.append(block.ln_1.register_forward_hook(self._ln1(layer)))
            self.handles.append(
                block.attn.c_proj.register_forward_hook(self._cproj(layer))
            )

    def _ln1(self, layer: int):
        def hook(_module, _inputs, output):
            self.ln1[layer].append(output.detach().float().cpu())

        return hook

    def _cproj(self, layer: int):
        def hook(_module, inputs, _output):
            self.cproj_inputs[layer].append(inputs[0].detach().float().cpu())

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def terminal_attention_metrics(
    checkpoint: Path,
    batches: list[torch.Tensor],
    layers: list[int],
    device: str,
) -> dict[int, dict[str, torch.Tensor]]:
    model = load_model(checkpoint, device)
    collector = AttentionMetricCollector(model, layers)
    try:
        with torch.no_grad():
            for tokens in batches:
                model(tokens.to(device), None)
        result: dict[int, dict[str, torch.Tensor]] = {}
        for layer in layers:
            inputs = torch.cat(collector.ln1[layer], dim=0).to(device)
            cproj_inputs = torch.cat(collector.cproj_inputs[layer], dim=0).to(device)
            attn = model.transformer.h[layer].attn
            packed = attn.c_attn.weight.detach().float()
            n_embd = int(packed.shape[1])
            n_head = int(attn.n_head)
            head_dim = n_embd // n_head
            q, key, _value = packed.split(n_embd, dim=0)
            q = F.linear(inputs, q).view(
                inputs.shape[0], inputs.shape[1], n_head, head_dim
            ).transpose(1, 2)
            key = F.linear(inputs, key).view(
                inputs.shape[0], inputs.shape[1], n_head, head_dim
            ).transpose(1, 2)
            scores = q @ key.transpose(-2, -1) / math.sqrt(head_dim)
            length = inputs.shape[1]
            mask = torch.ones(
                (length, length), dtype=torch.bool, device=device
            ).tril()
            probabilities = torch.softmax(
                scores.masked_fill(~mask, -torch.inf), dim=-1
            )
            result[layer] = {
                "cproj_inputs": cproj_inputs,
                "value_sources": probabilities @ inputs.unsqueeze(1),
                "output_weight": attn.c_proj.weight.detach().float(),
            }
        return result
    finally:
        collector.close()
        del model


def weighted(rows: list[dict[str, Any]], field: str, energy: str) -> float:
    denominator = sum(float(row[energy]) for row in rows)
    return sum(float(row[field]) * float(row[energy]) for row in rows) / max(
        denominator, 1e-30
    )


def classify_target(
    summary: dict[str, float | bool], thresholds: dict[str, float]
) -> tuple[str, dict[str, bool]]:
    checks = {
        "range_valid": bool(summary["range_valid"]),
        "image": float(summary["functional_image_recovery"])
        >= float(thresholds["functional_image_recovery_minimum"]),
        "tangent": float(summary["activated_tangent_recovery"])
        >= float(thresholds["activated_tangent_recovery_minimum"]),
        "activation_gain": float(summary["activation_gain_over_identity"])
        >= float(thresholds["activation_gain_over_identity_minimum"]),
    }
    classification = (
        "ATTENTION_PAPER_ACTIVATION_ORACLE_PASS"
        if all(checks.values())
        else "ATTENTION_PAPER_ACTIVATION_ORACLE_REJECT"
    )
    return classification, checks


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_plan(plan: dict[str, Any], args: argparse.Namespace) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unexpected attention paper-activation plan schema")
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
    expected_probes = identity["probe_sha256"]
    for name, expected in expected_probes.items():
        path = args.probe_dir / name
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"optimizer probe mismatch: {path}")
    protocol = plan["protocol"]
    frozen = {
        "parameter_updates": 0,
        "latent_ratio": 0.01,
        "block_fht_layers": 2,
        "activation": "signed_condition_bounded_tanh",
        "activation_scale_multiplier": math.sqrt(10.0 / 9.0),
        "minimum_step0_activation_derivative": 0.1,
        "maximum_step0_jacobian_condition": 10.0,
        "coordinate_fit": "oracle_cgls_in_inverse_activation_preactivation",
        "tangent_fit": "oracle_cgls_in_frozen_terminal_attention_output_metric",
        "cgls_iterations": 32,
    }
    for field, expected in frozen.items():
        if protocol.get(field) != expected:
            raise ValueError(f"frozen protocol changed: {field}")
    if plan["authorization"] != {
        "model_implementation": False,
        "mfu_preflight": False,
        "language_model_training": False,
        "larger_rung": False,
    }:
        raise ValueError("zero-update oracle authorization changed")


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
    heldout_steps = {int(value) for value in protocol["heldout_steps"]}
    batches = fixed_validation_batches(
        args.data_dir,
        int(protocol["metric_batch_size"]),
        int(protocol["metric_block_size"]),
        int(protocol["metric_batches"]),
        int(protocol["metric_seed"]),
    )
    print("collecting frozen terminal-dense attention metric", flush=True)
    metric_inputs = terminal_attention_metrics(
        args.terminal_checkpoint, batches, layers, args.device
    )
    probes = {}
    run_identity = None
    for step in steps:
        path = args.probe_dir / f"step_{step:06d}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if run_identity is None:
            run_identity = payload["run_identity_sha256"]
        elif payload["run_identity_sha256"] != run_identity:
            raise ValueError("optimizer probes do not share one run identity")
        probes[step] = payload
    if run_identity != plan["identity"]["probe_run_identity_sha256"]:
        raise ValueError("optimizer probe run identity changed")

    config = json.loads((REPO_ROOT / plan["identity"]["dense_config"]).read_text())
    base_seed = int(config["block_fht_seed"])
    latent_init_std = float(config.get("block_fht_latent_init_std", 0.02))
    scale_multiplier = float(protocol["activation_scale_multiplier"])
    rows: list[dict[str, Any]] = []
    for layer in layers:
        print(f"analyzing layer {layer}", flush=True)
        inputs = metric_inputs[layer]
        for target, spec in protocol["targets"].items():
            initial_state = probes[steps[0]]["parameters"][
                f"transformer.h.{layer}.{spec['parameter']}"
            ]
            initial = initial_state["weight_before_step"].float()
            if target == "v":
                n_embd = int(probes[steps[0]]["model_config"]["n_embd"])
                initial = initial[2 * n_embd :]
            initial = initial.to(args.device)
            target_std = float(spec["target_std"])
            weight_scale = target_std / latent_init_std
            size = initial.numel()
            latent_dim = max(1, round(size * float(protocol["latent_ratio"])))
            seed = base_seed + int(spec["seed_stride"]) * layer + int(
                spec["seed_offset"]
            )
            template = torch.zeros(latent_dim, device=args.device)
            scale = activation_scale(initial, scale_multiplier)
            bias = activation_bias(initial, scale)

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

            metric = AttentionFunctionalMetric(
                target=target,
                cproj_inputs=inputs["cproj_inputs"],
                value_sources=inputs["value_sources"],
                output_weight=inputs["output_weight"],
            )
            for step in steps:
                parameter = probes[step]["parameters"][
                    f"transformer.h.{layer}.{spec['parameter']}"
                ]
                current = parameter["weight_before_step"].float()
                direction = parameter["applied_direction_per_lr"].float()
                if target == "v":
                    n_embd = int(probes[step]["model_config"]["n_embd"])
                    current = current[2 * n_embd :]
                    direction = direction[2 * n_embd :]
                current = current.to(args.device)
                direction = direction.to(args.device)
                maximum_ratio = float(current.abs().amax() / scale)
                range_valid = maximum_ratio < 1.0
                inverse = scale * torch.atanh(
                    (current / scale).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
                ) - bias
                latent, _fit, fit_iterations = cgls(
                    apply_a,
                    adjoint_a,
                    inverse,
                    template,
                    int(protocol["cgls_iterations"]),
                )
                mapped, derivative = activated_weight_and_derivative(
                    bias + apply_a(latent), scale
                )
                image_target = metric.apply(current - initial)
                image_prediction = metric.apply(mapped - initial)
                image_recovery, image_energy = explained_energy(
                    image_target, image_prediction
                )
                tangent_target = metric.apply(direction)

                def solve_tangent(diagonal: torch.Tensor) -> tuple[float, int]:
                    def apply(coordinate: torch.Tensor) -> torch.Tensor:
                        return metric.apply(diagonal * apply_a(coordinate))

                    def adjoint(output: torch.Tensor) -> torch.Tensor:
                        return adjoint_a(diagonal * metric.adjoint(output))

                    _coordinate, prediction, iterations = cgls(
                        apply,
                        adjoint,
                        tangent_target,
                        template,
                        int(protocol["cgls_iterations"]),
                    )
                    recovery, _energy = explained_energy(
                        tangent_target, prediction
                    )
                    return recovery, iterations

                activated_recovery, activated_iterations = solve_tangent(derivative)
                identity_recovery, identity_iterations = solve_tangent(
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
                        "activation_scale": float(scale),
                        "maximum_current_to_scale_ratio": maximum_ratio,
                        "range_valid": range_valid,
                        "functional_image_recovery": image_recovery,
                        "functional_image_energy": image_energy,
                        "activated_tangent_recovery": activated_recovery,
                        "identity_tangent_recovery": identity_recovery,
                        "activation_gain_over_identity": (
                            activated_recovery - identity_recovery
                        ),
                        "functional_tangent_energy": float(
                            tangent_target.double().square().sum()
                        ),
                        "mean_activation_derivative": float(derivative.mean()),
                        "minimum_activation_derivative": float(derivative.amin()),
                        "maximum_activation_derivative": float(derivative.amax()),
                        "coordinate_fit_iterations": fit_iterations,
                        "activated_tangent_iterations": activated_iterations,
                        "identity_tangent_iterations": identity_iterations,
                    }
                )

    summaries: dict[str, Any] = {}
    thresholds = plan["decision_rule"]["thresholds"]
    for target in protocol["targets"]:
        selected = [
            row for row in rows if row["target"] == target and row["heldout"]
        ]
        summary: dict[str, float | bool] = {
            "range_valid": all(bool(row["range_valid"]) for row in rows if row["target"] == target),
            "functional_image_recovery": weighted(
                selected, "functional_image_recovery", "functional_image_energy"
            ),
            "activated_tangent_recovery": weighted(
                selected, "activated_tangent_recovery", "functional_tangent_energy"
            ),
            "identity_tangent_recovery": weighted(
                selected, "identity_tangent_recovery", "functional_tangent_energy"
            ),
        }
        summary["activation_gain_over_identity"] = float(
            summary["activated_tangent_recovery"]
        ) - float(summary["identity_tangent_recovery"])
        classification, checks = classify_target(summary, thresholds)
        summaries[target] = {
            **summary,
            "classification": classification,
            "checks": checks,
            "passed": all(checks.values()),
        }

    args.output_dir.mkdir(parents=True)
    cells_path = args.output_dir / "attention_paper_activation_oracle_cells.csv"
    write_csv(cells_path, rows)
    passed = [target for target, value in summaries.items() if value["passed"]]
    result = {
        "schema_version": RESULT_SCHEMA,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "classification": (
            "ATTENTION_PAPER_ACTIVATION_ORACLE_HAS_PASSING_TARGET"
            if passed
            else "ATTENTION_PAPER_ACTIVATION_ORACLE_REJECT_ALL"
        ),
        "execution": {
            "host": "PRO6",
            "device": args.device,
            "git_commit": git_commit(),
            "entrypoint": "examples.nanogpt.analyze_attention_paper_activation_oracle",
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
            "causal_coordinate_analysis_authorized": bool(passed),
            "model_implementation_authorized": False,
            "mfu_preflight_authorized": False,
            "language_model_training_authorized": False,
            "larger_rung_authorized": False,
        },
        "cells_csv": {
            "path": str(cells_path),
            "sha256": file_sha256(cells_path),
        },
        "all_reported_values_finite": all_finite(summaries),
        "limitations": [
            "State coordinates and tangent coefficients are oracle fits that see the dense answers.",
            "The functional metric freezes terminal-dense activations to isolate parameter geometry from residual-stream coevolution.",
            "A pass is only a necessary representability result and cannot authorize a model implementation or training run.",
        ],
    }
    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
