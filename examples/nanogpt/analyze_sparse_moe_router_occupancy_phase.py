#!/usr/bin/env python3
"""Audit sparse-MoE router occupancy and terminal auxiliary gradient scale."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from examples.nanogpt.analyze_cproj_manifold import load_model
from examples.nanogpt.analyze_mlp_activation_update_alignment import git_commit
from examples.nanogpt.analyze_residual_compatibility import fixed_validation_batches
from examples.nanogpt.analyze_sparse_moe_paired_alignment import (
    collect_inputs,
    file_sha256,
)


PLAN_SCHEMA = "nanogpt_sparse_moe_router_occupancy_phase_audit_plan_v1"


def inventory_sha256(paths: list[Path]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update((path.name + "\0" + file_sha256(path) + "\n").encode())
    return digest.hexdigest()


def trajectory_paths(source: dict[str, Any]) -> list[Path]:
    directory = Path(source["trajectory_directory"])
    schema = source["trajectory_schema"]
    result = []
    for step in source["trajectory_steps"]:
        if schema == "nanogpt_moe_paired_snapshot_v1":
            name = f"step_{int(step):06d}_moe_paired_l0_l5_l11.pt"
        elif schema == "nanogpt_parameter_trajectory_v1":
            name = f"step_{int(step):06d}.pt"
        else:
            raise ValueError(f"unsupported trajectory schema: {schema}")
        result.append(directory / name)
    if any(not path.is_file() for path in result):
        missing = [str(path) for path in result if not path.is_file()]
        raise FileNotFoundError("missing trajectory files: " + ", ".join(missing))
    if inventory_sha256(result) != source["trajectory_inventory_sha256"]:
        raise ValueError(f"trajectory inventory mismatch for {source['name']}")
    return result


def load_router_mapping(
    path: Path,
    expected_schema: str,
    expected_step: int,
) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != expected_schema:
        raise ValueError(f"trajectory schema mismatch: {path}")
    if int(payload.get("step", -1)) != int(expected_step):
        raise ValueError(f"trajectory step mismatch: {path}")
    field = "model" if expected_schema == "nanogpt_moe_paired_snapshot_v1" else "parameters"
    mapping = payload.get(field)
    if not isinstance(mapping, dict):
        raise ValueError(f"trajectory state has no {field}: {path}")
    return mapping


def normalized_entropy(probability: torch.Tensor) -> float:
    probability = probability.float()
    entropy = -(probability * probability.clamp_min(1e-30).log()).sum()
    return float(entropy / math.log(probability.numel()))


def effective_count(probability: torch.Tensor) -> float:
    probability = probability.float()
    return float(torch.exp(-(probability * probability.clamp_min(1e-30).log()).sum()))


def centered_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float() - left.float().mean()
    right = right.float() - right.float().mean()
    denominator = left.norm() * right.norm()
    return float((left * right).sum() / denominator.clamp_min(1e-30))


def occupancy_statistics(
    router: torch.Tensor,
    activations: torch.Tensor,
    top_k: int,
    load_balance_coefficient: float,
    z_loss_coefficient: float,
) -> dict[str, Any]:
    logits = activations.float() @ router.float().T
    experts = int(router.shape[0])
    tie = torch.arange(experts, device=logits.device, dtype=logits.dtype)
    selected = torch.topk(
        logits - tie * torch.finfo(logits.dtype).eps,
        int(top_k),
        dim=-1,
        largest=True,
        sorted=True,
    ).indices
    counts = torch.bincount(selected.flatten(), minlength=experts).float()
    total_assignments = int(selected.numel())
    load = counts / float(total_assignments)
    importance = F.softmax(logits, dim=-1).mean(dim=0)
    load_balance = experts * torch.sum(importance * load)
    z_loss = torch.logsumexp(logits, dim=-1).square().mean()
    mean_load = load.mean()
    cv = load.std(unbiased=False) / mean_load.clamp_min(1e-30)
    minimum = float(load.min())
    maximum = float(load.max())
    min_count = int(counts.min())
    max_count = int(counts.max())
    return {
        "tokens": int(activations.shape[0]),
        "total_assignments": total_assignments,
        "expert_counts": [int(value) for value in counts.tolist()],
        "expert_fractions": [float(value) for value in load.tolist()],
        "importance": [float(value) for value in importance.tolist()],
        "minimum_count": min_count,
        "maximum_count": max_count,
        "minimum_fraction": minimum,
        "maximum_fraction": maximum,
        "maximum_to_minimum_ratio": maximum / minimum if minimum > 0 else None,
        "hard_normalized_entropy": normalized_entropy(load),
        "hard_effective_expert_count": effective_count(load),
        "hard_coefficient_of_variation": float(cv),
        "soft_normalized_entropy": normalized_entropy(importance),
        "soft_effective_expert_count": effective_count(importance),
        "importance_load_correlation": centered_cosine(importance, load),
        "experts_below_one_percent": int((load < 0.01).sum()),
        "experts_below_128_assignments": int((counts < 128).sum()),
        "load_balance_loss": float(load_balance),
        "router_z_loss": float(z_loss),
        "weighted_load_balance_loss": float(load_balance_coefficient * load_balance),
        "weighted_router_z_loss": float(z_loss_coefficient * z_loss),
    }


def _mean_gradients(
    model: torch.nn.Module,
    batches: list[torch.Tensor],
    layers: list[int],
    device: str,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    routers = [model.transformer.h[layer].mlp.router.weight for layer in layers]
    for router in routers:
        router.requires_grad_(True)
    coefficients = {
        "load_balance": float(model.config.moe_load_balance_aux_coefficient),
        "router_z": float(model.config.moe_router_z_loss_coefficient),
    }
    sums = {
        name: [torch.zeros_like(router, dtype=torch.float32) for router in routers]
        for name in ("task", "load_balance", "router_z")
    }
    scalar_sums = {name: 0.0 for name in sums}
    model.eval()
    for batch in batches:
        batch = batch.to(device)
        inputs = batch[:, :-1].contiguous()
        targets = batch[:, 1:].contiguous()
        logits, _unused = model(inputs, None)
        task = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
        )
        load_balance, router_z = model.moe_router_losses()
        components = {
            "task": task,
            "load_balance": coefficients["load_balance"] * load_balance,
            "router_z": coefficients["router_z"] * router_z,
        }
        for index, (name, component) in enumerate(components.items()):
            gradients = torch.autograd.grad(
                component,
                routers,
                retain_graph=index < len(components) - 1,
                allow_unused=False,
            )
            scalar_sums[name] += float(component.detach())
            for target, gradient in zip(sums[name], gradients, strict=True):
                target.add_(gradient.detach().float())
    count = float(len(batches))
    rows = []
    for index, layer in enumerate(layers):
        task = sums["task"][index] / count
        load_balance = sums["load_balance"][index] / count
        router_z = sums["router_z"][index] / count
        auxiliary = load_balance + router_z
        task_norm = task.norm()
        auxiliary_norm = auxiliary.norm()
        rows.append(
            {
                "layer": int(layer),
                "task_gradient_norm": float(task_norm),
                "weighted_load_balance_gradient_norm": float(load_balance.norm()),
                "weighted_router_z_gradient_norm": float(router_z.norm()),
                "combined_weighted_auxiliary_gradient_norm": float(auxiliary_norm),
                "auxiliary_to_task_gradient_norm_ratio": float(
                    auxiliary_norm / task_norm.clamp_min(1e-30)
                ),
                "load_balance_to_task_gradient_norm_ratio": float(
                    load_balance.norm() / task_norm.clamp_min(1e-30)
                ),
                "router_z_to_task_gradient_norm_ratio": float(
                    router_z.norm() / task_norm.clamp_min(1e-30)
                ),
                "auxiliary_task_gradient_cosine": float(
                    (auxiliary * task).sum()
                    / (auxiliary_norm * task_norm).clamp_min(1e-30)
                ),
            }
        )
    scalars = {name: value / count for name, value in scalar_sums.items()}
    return rows, scalars


def persistent_collapse_onset(rows: list[dict[str, Any]], threshold: float = 0.01) -> int | None:
    ordered = sorted(rows, key=lambda row: int(row["step"]))
    for index, row in enumerate(ordered):
        if all(float(item["minimum_fraction"]) < threshold for item in ordered[index:]):
            return int(row["step"])
    return None


def coefficient_from_ratio(ratio: float) -> float:
    multiplier = 16.0 if ratio <= 0 else min(16.0, max(2.0, 0.50 / ratio))
    requested = 0.01 * multiplier
    return next(value for value in (0.02, 0.04, 0.08, 0.16) if value >= requested - 1e-15)


def finite_tree(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    return True


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    plan = json.loads(args.plan.read_text())
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("router audit plan schema mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    status_path = args.output / "status.json"
    atomic_json(status_path, {"status": "running", "started_at_unix": started})
    layers = [int(value) for value in plan["sources"]["layers"]]
    top_k = int(plan["sources"]["top_k"])
    load_coefficient = float(plan["sources"]["load_balance_aux_coefficient"])
    z_coefficient = float(plan["sources"]["router_z_loss_coefficient"])
    activation = plan["activation_protocol"]
    gradient = plan["gradient_protocol"]
    occupancy_batches = fixed_validation_batches(
        args.data_dir,
        int(activation["batch_size"]),
        int(activation["block_size"]),
        int(activation["batches"]),
        int(activation["validation_seed"]),
    )
    gradient_batches = fixed_validation_batches(
        args.data_dir,
        int(gradient["batch_size"]),
        int(gradient["block_size"]),
        int(gradient["batches"]),
        int(gradient["validation_seed"]),
    )
    occupancy_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    seed_summaries: dict[str, Any] = {}
    source_receipts: dict[str, Any] = {}
    for seed_name in ("seed1", "seed2"):
        source = plan["sources"][seed_name]
        terminal = Path(source["terminal_model"])
        terminal_sha = file_sha256(terminal)
        if terminal_sha != source["terminal_model_sha256"]:
            raise ValueError(f"terminal model hash mismatch for {seed_name}")
        paths = trajectory_paths(source)
        model = load_model(terminal, args.device)
        resolved = torch.load(terminal, map_location="cpu", weights_only=False)["run_identity"]
        if resolved["config_sha256"] != source["config_sha256"]:
            raise ValueError(f"terminal config identity mismatch for {seed_name}")
        if resolved["data_manifest"]["sha256"] != plan["sources"]["dataset_manifest_sha256"]:
            raise ValueError(f"dataset manifest identity mismatch for {seed_name}")
        inputs = collect_inputs(
            model,
            occupancy_batches,
            layers,
            int(activation["sample_cap_per_layer"]),
            args.device,
        )
        for path, step in zip(paths, source["trajectory_steps"], strict=True):
            mapping = load_router_mapping(path, source["trajectory_schema"], int(step))
            for layer in layers:
                router = mapping[f"transformer.h.{layer}.mlp.router.weight"].to(args.device)
                metrics = occupancy_statistics(
                    router,
                    inputs[layer].to(args.device),
                    top_k,
                    load_coefficient,
                    z_coefficient,
                )
                occupancy_rows.append(
                    {
                        "seed": seed_name,
                        "model_seed": int(source["model_seed"]),
                        "step": int(step),
                        "phase": float(int(step) / 9495.0),
                        "layer": int(layer),
                        **metrics,
                    }
                )
        seed_gradient_rows, gradient_scalars = _mean_gradients(
            model,
            gradient_batches,
            layers,
            args.device,
        )
        for row in seed_gradient_rows:
            gradient_rows.append({"seed": seed_name, **row, **gradient_scalars})
        del model, inputs
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        terminal_rows = [
            row
            for row in occupancy_rows
            if row["seed"] == seed_name and int(row["step"]) == 9495
        ]
        collapsed_layers = [
            int(row["layer"])
            for row in terminal_rows
            if float(row["minimum_fraction"]) < 0.01 or int(row["minimum_count"]) < 128
        ]
        onset = {}
        for layer in collapsed_layers:
            onset[str(layer)] = persistent_collapse_onset(
                [
                    row
                    for row in occupancy_rows
                    if row["seed"] == seed_name and int(row["layer"]) == layer
                ]
            )
        seed_summaries[seed_name] = {
            "collapsed_terminal_layers": collapsed_layers,
            "replicated_terminal_collapse": bool(collapsed_layers),
            "persistent_collapse_onset_step": onset,
            "terminal_minimum_count": min(int(row["minimum_count"]) for row in terminal_rows),
            "terminal_minimum_fraction": min(float(row["minimum_fraction"]) for row in terminal_rows),
            "terminal_maximum_load_balance_loss": max(float(row["load_balance_loss"]) for row in terminal_rows),
            "terminal_objective_blind_rows": sum(
                float(row["minimum_fraction"]) < 0.01
                and float(row["load_balance_loss"]) <= 1.10
                for row in terminal_rows
            ),
        }
        source_receipts[seed_name] = {
            "terminal_model_sha256": terminal_sha,
            "trajectory_inventory_sha256": inventory_sha256(paths),
            "trajectory_file_count": len(paths),
        }
    ratios = [float(row["auxiliary_to_task_gradient_norm_ratio"]) for row in gradient_rows]
    median_ratio = statistics.median(ratios)
    collapse_replicated = all(
        seed_summaries[name]["replicated_terminal_collapse"]
        for name in ("seed1", "seed2")
    )
    under_scaled = collapse_replicated and median_ratio < 0.25
    if not collapse_replicated:
        decision = "NO_REPLICATED_COLLAPSE_NO_ROUTER_INTERVENTION"
        proposed_coefficient = None
    elif under_scaled:
        decision = "REPLICATED_COLLAPSE_UNDERSCALED_AUX_PREREGISTER_COEFFICIENT_CONTROL"
        proposed_coefficient = coefficient_from_ratio(median_ratio)
    else:
        decision = "REPLICATED_COLLAPSE_AUX_NOT_UNDERSCALED_REQUIRE_OBJECTIVE_CHANGE"
        proposed_coefficient = None
    result = {
        "schema_version": "nanogpt_sparse_moe_router_occupancy_phase_audit_result_v1",
        "status": "complete",
        "decision": decision,
        "collapse_replicated": collapse_replicated,
        "under_scaled_auxiliary": under_scaled,
        "median_terminal_auxiliary_to_task_gradient_norm_ratio": median_ratio,
        "formula_selected_load_balance_coefficient": proposed_coefficient,
        "seed_summaries": seed_summaries,
        "source_receipts": source_receipts,
        "protocol": {
            "plan": str(args.plan),
            "plan_sha256": file_sha256(args.plan),
            "data_dir": str(args.data_dir),
            "layers": layers,
            "activation_seed": int(activation["validation_seed"]),
            "gradient_seed": int(gradient["validation_seed"]),
        },
        "artifacts": {
            "occupancy_rows": str(args.output / "router_occupancy_rows.csv"),
            "gradient_rows": str(args.output / "router_gradient_rows.csv"),
        },
        "provenance": {
            "git_commit": git_commit(),
            "entrypoint": str(Path(__file__).resolve()),
            "entrypoint_sha256": file_sha256(Path(__file__).resolve()),
            "command": sys.argv,
        },
        "elapsed_seconds": time.time() - started,
    }
    result["all_values_finite"] = finite_tree(result) and finite_tree(occupancy_rows) and finite_tree(gradient_rows)
    write_csv(args.output / "router_occupancy_rows.csv", occupancy_rows)
    write_csv(args.output / "router_gradient_rows.csv", gradient_rows)
    atomic_json(args.output / "result.json", result)
    atomic_json(
        status_path,
        {
            "status": "complete",
            "decision": decision,
            "finished_at_unix": time.time(),
            "result_sha256": file_sha256(args.output / "result.json"),
        },
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
