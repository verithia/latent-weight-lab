#!/usr/bin/env python3
"""Test whether sparse-MoE c_proj writes are predictable from c_fc features.

This is a zero-update, deliberately optimistic upper bound.  A linear map is
fit from terminal c_fc feature directions (and, for the strongest family, the
same-run c_fc displacement) to the terminal-minus-step-zero c_proj write
directions.  The map is fit on hidden neurons that are never scored.  Scored
neurons may fit only one scalar radial coordinate after their direction has
been predicted.  Two neuron partitions, two independent runs, and two fixed
token frames distinguish a reusable feature-to-write relation from endpoint
interpolation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_cproj_manifold import load_model
from examples.nanogpt.analyze_mlp_activation_update_alignment import git_commit
from examples.nanogpt.analyze_residual_compatibility import fixed_validation_batches
from examples.nanogpt.analyze_sparse_moe_cproj_context_modulated_fht_oracle import (
    ExpertFrame,
    routed_hidden_frames,
)
from examples.nanogpt.analyze_sparse_moe_paired_alignment import (
    collect_inputs,
    file_sha256,
)
from examples.nanogpt.analyze_sparse_moe_paired_atom_oracle import union_fieldnames
from examples.nanogpt.analyze_sparse_moe_stepzero_task_gradient_oracle import (
    layer_state_from_mapping,
)


PLAN_SCHEMA = "nanogpt_sparse_moe_feature_write_predictability_plan_v1"
PAIRED_SCHEMA = "nanogpt_moe_paired_snapshot_v1"
TRAJECTORY_SCHEMA = "nanogpt_parameter_trajectory_v1"
FAMILIES = (
    "tied_current_ray",
    "tied_current_motion_plane",
    "dense_current_ridge",
    "dense_current_motion_ridge",
)


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(memoryview(array)).hexdigest()


def load_state(path: Path, layers: list[int]) -> tuple[dict[str, Any], dict[int, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") not in {
        PAIRED_SCHEMA,
        TRAJECTORY_SCHEMA,
    }:
        raise ValueError(f"unsupported sparse-MoE state: {path}")
    if payload.get("layers") != layers:
        raise ValueError(f"layer identity mismatch: {path}")
    mapping = payload.get("model") if payload.get("schema_version") == PAIRED_SCHEMA else payload.get("parameters")
    if not isinstance(mapping, dict):
        raise ValueError(f"state has no parameter mapping: {path}")
    return payload, {layer: layer_state_from_mapping(mapping, layer) for layer in layers}


def normalized_rows(value: torch.Tensor) -> torch.Tensor:
    return value.float() / value.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)


def partition_indices(hidden: int, fit_count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if fit_count <= 0 or fit_count >= hidden:
        raise ValueError("fit count must leave nonempty fit and score sets")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    permutation = torch.randperm(hidden, generator=generator)
    return permutation[:fit_count], permutation[fit_count:]


def ridge_predict(
    fit_features: torch.Tensor,
    fit_target: torch.Tensor,
    score_features: torch.Tensor,
    ridge_relative: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Batched ridge prediction, using the smaller primal or dual system."""
    if fit_features.ndim != 3 or fit_target.ndim != 3 or score_features.ndim != 3:
        raise ValueError("ridge tensors must be [expert, samples, width]")
    experts, samples, features = fit_features.shape
    if fit_target.shape[:2] != (experts, samples) or score_features.shape[0] != experts:
        raise ValueError("ridge batch/sample dimensions disagree")
    scale = fit_features.square().sum(dim=(1, 2)) / float(max(1, samples))
    regularizer = float(ridge_relative) * scale.clamp_min(1e-12)
    if features <= samples:
        gram = fit_features.transpose(1, 2) @ fit_features
        eye = torch.eye(features, device=gram.device, dtype=gram.dtype)[None]
        rhs = fit_features.transpose(1, 2) @ fit_target
        coefficients = torch.linalg.solve(gram + regularizer[:, None, None] * eye, rhs)
        prediction = score_features @ coefficients
        fit_prediction = fit_features @ coefficients
        system = gram
    else:
        gram = fit_features @ fit_features.transpose(1, 2)
        eye = torch.eye(samples, device=gram.device, dtype=gram.dtype)[None]
        dual = torch.linalg.solve(gram + regularizer[:, None, None] * eye, fit_target)
        prediction = (score_features @ fit_features.transpose(1, 2)) @ dual
        fit_prediction = gram @ dual
        system = gram
    eigenvalues = torch.linalg.eigvalsh(system).clamp_min(0)
    effective_condition = (
        (eigenvalues[:, -1] + regularizer)
        / (eigenvalues[:, 0] + regularizer).clamp_min(1e-30)
    )
    return prediction, {
        "ridge_relative": float(ridge_relative),
        "regularizer_minimum": float(regularizer.min()),
        "regularizer_maximum": float(regularizer.max()),
        "effective_condition_maximum": float(effective_condition.max()),
        "fit_recovery": float(
            1.0
            - (fit_prediction - fit_target).square().sum()
            / fit_target.square().sum().clamp_min(1e-30)
        ),
    }


def optimal_radial(predicted_direction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = (predicted_direction.float() * target.float()).sum(dim=-1, keepdim=True)
    denominator = predicted_direction.float().square().sum(dim=-1, keepdim=True).clamp_min(1e-30)
    return predicted_direction.float() * (numerator / denominator)


def optimal_plane(
    current: torch.Tensor,
    motion: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    basis = torch.stack((current.float(), motion.float()), dim=-2)
    gram = basis @ basis.transpose(-1, -2)
    trace = gram.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    eye = torch.eye(2, device=gram.device, dtype=gram.dtype)
    gram = gram + (1e-8 * trace.clamp_min(1e-12))[..., None, None] * eye
    rhs = (basis * target.float().unsqueeze(-2)).sum(dim=-1)
    coefficients = torch.linalg.solve(gram, rhs.unsqueeze(-1)).squeeze(-1)
    return (basis * coefficients.unsqueeze(-1)).sum(dim=-2)


def recovery(predicted: torch.Tensor, target: torch.Tensor) -> float:
    denominator = target.float().square().sum().clamp_min(1e-30)
    return float(1.0 - (predicted.float() - target.float()).square().sum() / denominator)


def direction_cosine_squared(predicted: torch.Tensor, target: torch.Tensor) -> float:
    numerator = (predicted.float() * target.float()).sum(dim=-1).square()
    denominator = (
        predicted.float().square().sum(dim=-1)
        * target.float().square().sum(dim=-1)
    ).clamp_min(1e-30)
    weights = target.float().square().sum(dim=-1)
    return float((weights * numerator / denominator).sum() / weights.sum().clamp_min(1e-30))


def subset_action(
    frames: list[ExpertFrame],
    score_indices: torch.Tensor,
    delta_rows: torch.Tensor,
    token_count: int,
) -> torch.Tensor:
    output = torch.zeros(
        token_count,
        delta_rows.shape[-1],
        device=delta_rows.device,
        dtype=torch.float32,
    )
    indices = score_indices.to(delta_rows.device)
    for expert, frame in enumerate(frames):
        hidden = frame.hidden.index_select(1, indices)
        action = hidden.float() @ delta_rows[expert].float()
        output.index_add_(
            0,
            frame.tokens,
            action * frame.probabilities.float()[:, None],
        )
    return output


def family_prediction(
    family: str,
    current: torch.Tensor,
    motion: torch.Tensor,
    target: torch.Tensor,
    fit_indices: torch.Tensor,
    score_indices: torch.Tensor,
    ridge_relative: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    current = normalized_rows(current)
    motion = normalized_rows(motion)
    score_target = target.index_select(1, score_indices)
    score_current = current.index_select(1, score_indices)
    score_motion = motion.index_select(1, score_indices)
    diagnostics: dict[str, float] = {}
    if family == "tied_current_ray":
        prediction = optimal_radial(score_current, score_target)
    elif family == "tied_current_motion_plane":
        prediction = optimal_plane(score_current, score_motion, score_target)
    elif family in {"dense_current_ridge", "dense_current_motion_ridge"}:
        fit_current = current.index_select(1, fit_indices)
        fit_motion = motion.index_select(1, fit_indices)
        fit_target = target.index_select(1, fit_indices)
        if family == "dense_current_ridge":
            fit_features = fit_current
            score_features = score_current
        else:
            fit_features = torch.cat((fit_current, fit_motion), dim=-1)
            score_features = torch.cat((score_current, score_motion), dim=-1)
        direction, diagnostics = ridge_predict(
            fit_features,
            fit_target,
            score_features,
            ridge_relative,
        )
        prediction = optimal_radial(direction, score_target)
    else:
        raise ValueError(f"unknown family: {family}")
    return prediction, diagnostics


def fixed_input_frames(
    model: torch.nn.Module,
    data_dir: Path,
    layers: list[int],
    frame_specs: list[dict[str, Any]],
    device: str,
) -> dict[str, dict[int, torch.Tensor]]:
    result: dict[str, dict[int, torch.Tensor]] = {}
    for spec in frame_specs:
        batches = fixed_validation_batches(
            data_dir,
            int(spec["batch_size"]),
            int(spec["block_size"]),
            int(spec["batches"]),
            int(spec["seed"]),
        )
        result[spec["name"]] = collect_inputs(
            model,
            batches,
            layers,
            int(spec["tokens"]),
            device,
        )
    return result


def aggregate(rows: list[dict[str, Any]], plan: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for family in FAMILIES:
        selected = [row for row in rows if row["family"] == family]
        weight_values = [float(row["weight_recovery"]) for row in selected]
        cosine_values = [float(row["direction_cosine_squared"]) for row in selected]
        functional_values = [float(row["functional_recovery"]) for row in selected]
        by_layer = {
            str(layer): sum(
                float(row["functional_recovery"])
                for row in selected
                if int(row["layer"]) == int(layer)
            )
            / sum(1 for row in selected if int(row["layer"]) == int(layer))
            for layer in plan["source"]["layers"]
        }
        by_seed = {
            seed["name"]: sum(
                float(row["functional_recovery"])
                for row in selected
                if row["seed"] == seed["name"]
            )
            / sum(1 for row in selected if row["seed"] == seed["name"])
            for seed in plan["source"]["runs"]
        }
        result[family] = {
            "weight_recovery_mean": sum(weight_values) / len(weight_values),
            "direction_cosine_squared_mean": sum(cosine_values) / len(cosine_values),
            "functional_recovery_mean": sum(functional_values) / len(functional_values),
            "functional_recovery_minimum": min(functional_values),
            "functional_recovery_by_layer": by_layer,
            "functional_recovery_by_seed": by_seed,
        }
    upper = result["dense_current_motion_ridge"]
    gates = plan["frozen_gates"]
    result["gate"] = {
        "upper_bound_mean_pass": upper["functional_recovery_mean"]
        >= float(gates["dense_upper_bound_functional_mean_min"]),
        "upper_bound_every_layer_pass": min(upper["functional_recovery_by_layer"].values())
        >= float(gates["dense_upper_bound_functional_every_layer_min"]),
        "upper_bound_each_seed_pass": min(upper["functional_recovery_by_seed"].values())
        >= float(gates["dense_upper_bound_functional_each_seed_min"]),
    }
    result["gate"]["pass"] = all(result["gate"].values())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    plan = json.loads(args.plan.read_text())
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("plan schema mismatch")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    source_hashes: dict[str, Any] = {}
    all_finite = True
    try:
        layers = [int(layer) for layer in plan["source"]["layers"]]
        data_dir = Path(plan["source"]["data_dir"])
        for seed_spec in plan["source"]["runs"]:
            stepzero_path = Path(seed_spec["stepzero_state"])
            terminal_path = Path(seed_spec["terminal_state"])
            model_path = Path(seed_spec["terminal_model"])
            _zero_payload, zero_states = load_state(stepzero_path, layers)
            _terminal_payload, terminal_states = load_state(terminal_path, layers)
            source_hashes[seed_spec["name"]] = {
                "stepzero_state_sha256": file_sha256(stepzero_path),
                "terminal_state_sha256": file_sha256(terminal_path),
                "terminal_model_sha256": file_sha256(model_path),
            }
            model = load_model(model_path, args.device)
            inputs = fixed_input_frames(
                model,
                data_dir,
                layers,
                plan["functional_protocol"]["token_frames"],
                args.device,
            )
            del model
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
            frame_cache: dict[tuple[str, int], list[ExpertFrame]] = {}
            for frame_spec in plan["functional_protocol"]["token_frames"]:
                frame_name = frame_spec["name"]
                for layer in layers:
                    frames, counts = routed_hidden_frames(
                        terminal_states[layer],
                        inputs[frame_name][layer],
                        int(plan["functional_protocol"]["top_k"]),
                        args.device,
                    )
                    frame_cache[(frame_name, layer)] = frames
                    source_hashes[seed_spec["name"]].setdefault("minimum_assignments", {})[
                        f"{frame_name}_layer{layer}"
                    ] = min(counts)
            for layer in layers:
                zero = zero_states[layer].to(args.device)
                terminal = terminal_states[layer].to(args.device)
                current = terminal.c_fc
                motion = terminal.c_fc - zero.c_fc
                target = (terminal.c_proj - zero.c_proj).transpose(1, 2)
                source_hashes[seed_spec["name"]].setdefault("tensor_sha256", {})[
                    f"layer{layer}_current_c_fc"
                ] = tensor_sha256(current)
                source_hashes[seed_spec["name"]]["tensor_sha256"][
                    f"layer{layer}_c_fc_motion"
                ] = tensor_sha256(motion)
                source_hashes[seed_spec["name"]]["tensor_sha256"][
                    f"layer{layer}_c_proj_motion"
                ] = tensor_sha256(target)
                for partition in plan["neuron_protocol"]["partitions"]:
                    fit_indices, score_indices = partition_indices(
                        current.shape[1],
                        int(plan["neuron_protocol"]["fit_neurons"]),
                        int(partition["seed"]) + 1009 * layer,
                    )
                    fit_indices = fit_indices.to(args.device)
                    score_indices = score_indices.to(args.device)
                    score_target = target.index_select(1, score_indices)
                    for family in FAMILIES:
                        prediction, diagnostics = family_prediction(
                            family,
                            current,
                            motion,
                            target,
                            fit_indices,
                            score_indices,
                            float(plan["fit"]["ridge_relative"]),
                        )
                        weight_recovery = recovery(prediction, score_target)
                        cosine_squared = direction_cosine_squared(prediction, score_target)
                        for frame_spec in plan["functional_protocol"]["token_frames"]:
                            frame_name = frame_spec["name"]
                            frames = frame_cache[(frame_name, layer)]
                            actual_action = subset_action(
                                frames,
                                score_indices,
                                score_target,
                                int(frame_spec["tokens"]),
                            )
                            predicted_action = subset_action(
                                frames,
                                score_indices,
                                prediction,
                                int(frame_spec["tokens"]),
                            )
                            functional_recovery = recovery(predicted_action, actual_action)
                            values = [weight_recovery, cosine_squared, functional_recovery]
                            all_finite = all_finite and all(math.isfinite(value) for value in values)
                            rows.append(
                                {
                                    "seed": seed_spec["name"],
                                    "layer": layer,
                                    "partition": partition["name"],
                                    "frame": frame_name,
                                    "family": family,
                                    "fit_neurons": int(fit_indices.numel()),
                                    "score_neurons": int(score_indices.numel()),
                                    "weight_recovery": weight_recovery,
                                    "direction_cosine_squared": cosine_squared,
                                    "functional_recovery": functional_recovery,
                                    **diagnostics,
                                }
                            )
                del zero, terminal, current, motion, target
            del inputs, frame_cache, zero_states, terminal_states
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        summary = aggregate(rows, plan)
        decision = (
            "FEATURE_FRAME_PREDICTIVE_COMPACT_FOLLOWUP_AUTHORIZED"
            if summary["gate"]["pass"]
            else "REJECT_FEATURE_ONLY_WRITE_DIRECTION_REQUIRE_INDEPENDENT_OUTPUT_FRAME"
        )
        result = {
            "schema_version": "nanogpt_sparse_moe_feature_write_predictability_result_v1",
            "decision": decision,
            "all_values_finite": all_finite,
            "git_commit": git_commit(Path.cwd()),
            "plan_sha256": file_sha256(args.plan),
            "source_hashes": source_hashes,
            "summary": summary,
            "wall_seconds": time.time() - started,
        }
        if not all_finite:
            raise RuntimeError("nonfinite feature-to-write result")
        with (output_dir / "feature_write_rows.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=union_fieldnames(rows))
            writer.writeheader()
            writer.writerows(rows)
        (output_dir / "feature_write_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        status = {
            "status": "finished",
            "exit_code": 0,
            "decision": decision,
            "result_sha256": file_sha256(output_dir / "feature_write_result.json"),
            "rows_sha256": file_sha256(output_dir / "feature_write_rows.csv"),
            "finished_at_unix": time.time(),
        }
        (output_dir / "feature_write_status.json").write_text(
            json.dumps(status, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    except Exception as error:
        status = {
            "status": "failed",
            "exit_code": 1,
            "error": repr(error),
            "finished_at_unix": time.time(),
        }
        (output_dir / "feature_write_status.json").write_text(
            json.dumps(status, indent=2, sort_keys=True) + "\n"
        )
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc!r}", file=sys.stderr)
        raise
