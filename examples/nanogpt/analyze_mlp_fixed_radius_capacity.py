#!/usr/bin/env python3
"""Bracket task-matched MLP chart capacity at the deployed update radius.

The exact functionmix midpoint supplies one common gradient and persisted
Muon state.  The deployed 64+24 c_fc and c_proj charts are reconstructed
before expanding only their residual coordinate counts to 40 or 64 stages.
Every expanded family is globally normalized back to the deployed family
Frobenius norm before paired held-out CE scoring.  No checkpoint state is
changed or written.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_mlp_cfc_exact_current_matcher import (
    exact_muon_update,
    file_sha256,
    fixed_batches,
    git_commit,
    load_model_and_optimizer,
)
from examples.nanogpt.analyze_mlp_dense_oracle_gap import (
    ExactVariantApplier,
    aggregate_direction_metrics,
    evaluate_candidates,
    family_fro,
    merge_updates,
    scale_family,
)
from examples.nanogpt.analyze_mlp_joint_prospective_step import (
    _autocast,
    _gradient_norm,
    assert_joint_matches_singletons,
    extract_production_updates,
    family_weights,
    historical_double_decay_update,
    optimizer_parameters,
)
from examples.nanogpt.analyze_mlp_joint_step_response_surface import (
    paired_comparison,
)
from examples.nanogpt.model import MultiOptimizer
from examples.nanogpt.muon_matched_givens import (
    MuonFunctionalShear,
    MuonMatchedGivens,
    apply_givens_flow,
    diagonal_metric_angles,
    fast_muon_matched_permutations,
    functional_coordinate_mix_update,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "nanogpt_mlp_fixed_radius_capacity_v1"


def quantized_update(
    base: torch.Tensor, update: torch.Tensor
) -> torch.Tensor:
    """Return the materialized delta after the model tensor's quantization."""
    endpoint = (base.float() + update.float()).to(dtype=base.dtype)
    return endpoint.float() - base.float()


def normalize_family_to_radius(
    bases: dict[int, torch.Tensor],
    raw_updates: dict[int, torch.Tensor],
    target_fro: float,
    *,
    iterations: int = 8,
) -> tuple[dict[int, torch.Tensor], dict[str, float]]:
    """Globally scale a family and account for BF16 endpoint quantization."""
    if set(bases) != set(raw_updates):
        raise ValueError("base and update layers differ")
    raw_fro = family_fro(raw_updates)
    if not math.isfinite(raw_fro) or raw_fro <= 0.0:
        raise ValueError("raw family update norm must be positive and finite")
    scale = float(target_fro) / raw_fro
    best: tuple[float, dict[int, torch.Tensor], float, float] | None = None
    for _ in range(int(iterations)):
        candidate = {
            layer: quantized_update(bases[layer], raw_updates[layer] * scale)
            for layer in sorted(raw_updates)
        }
        actual_fro = family_fro(candidate)
        error = abs(actual_fro - float(target_fro))
        if best is None or error < best[0]:
            best = (error, candidate, scale, actual_fro)
        if actual_fro <= 0.0:
            raise FloatingPointError("quantized family update vanished")
        scale *= float(target_fro) / actual_fro
    if best is None:
        raise RuntimeError("family normalization produced no candidate")
    error, candidate, selected_scale, actual_fro = best
    return candidate, {
        "raw_fro": raw_fro,
        "target_fro": float(target_fro),
        "actual_fro": actual_fro,
        "scale": selected_scale,
        "relative_radius_error": error / max(float(target_fro), 1e-30),
    }


@torch.no_grad()
def reconstruct_cproj_update(
    weight: torch.Tensor,
    requested_update: torch.Tensor,
    selection_direction: torch.Tensor,
    *,
    parent_stages: int,
    residual_stages: int,
    neighbors: int,
    seed: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Reproduce/expand the historical matched-Givens c_proj endpoint."""
    source = weight.float()
    requested = requested_update.float()
    direction = selection_direction.float()
    parent_permutations, parent_selection = fast_muon_matched_permutations(
        source,
        direction,
        stages=int(parent_stages),
        neighbors=int(neighbors),
        seed=int(seed),
    )
    parent_permutations = parent_permutations.to(source.device)
    parent_angles = diagonal_metric_angles(
        source, requested, parent_permutations
    )
    after_parent = apply_givens_flow(
        source, parent_angles, parent_permutations
    )
    residual = requested - (after_parent - source)
    residual_permutations, residual_selection = (
        fast_muon_matched_permutations(
            after_parent,
            residual,
            stages=int(residual_stages),
            neighbors=int(neighbors),
            seed=int(seed) + 1,
        )
    )
    residual_permutations = residual_permutations.to(source.device)
    residual_angles = diagonal_metric_angles(
        after_parent, residual, residual_permutations
    )
    endpoint = apply_givens_flow(
        after_parent, residual_angles, residual_permutations
    )
    endpoint.mul_(1.0 - float(learning_rate) * float(weight_decay))
    raw_update = endpoint - source
    requested_energy = requested.square().sum().clamp_min(1e-30)
    return raw_update, {
        "coordinates": int(
            (int(parent_stages) + int(residual_stages))
            * source.shape[1]
            // 2
        ),
        "parent_stages": int(parent_stages),
        "residual_stages": int(residual_stages),
        "requested_update_recovery": float(
            1.0 - (requested - raw_update).square().sum() / requested_energy
        ),
        "parent_matching": parent_selection,
        "residual_matching": residual_selection,
        "angle_rms": float(
            torch.cat((parent_angles, residual_angles)).square().mean().sqrt()
        ),
    }


def _owner_and_group(
    optimizer: MultiOptimizer, owner_type: type
) -> tuple[Any, dict[str, Any]]:
    owner = next(
        child for child in optimizer.optimizers if isinstance(child, owner_type)
    )
    if len(owner.param_groups) != 1:
        raise ValueError(f"{owner_type.__name__} must have one parameter group")
    return owner, owner.param_groups[0]


def extract_reconstructed_capacity_updates(
    checkpoint: Path,
    config: dict[str, Any],
    train_batches: list[torch.Tensor],
    residual_stages: list[int],
    *,
    device: str,
    dtype: torch.dtype,
) -> tuple[
    dict[str, dict[int, torch.Tensor]],
    dict[str, dict[int, torch.Tensor]],
    dict[str, Any],
]:
    """Extract dense targets and reconstruct every registered chart width."""
    model, optimizer, checkpoint_payload = load_model_and_optimizer(
        checkpoint, config, device
    )
    model.train()
    weights = family_weights(model)
    optimizer.zero_grad(set_to_none=True)
    model.prepare_block_fht_cache(dtype=dtype)
    losses: list[float] = []
    try:
        for tokens in train_batches:
            tokens = tokens.to(device)
            with _autocast(device, dtype):
                _logits, loss = model(
                    tokens[:, :-1].contiguous(),
                    tokens[:, 1:].contiguous(),
                )
            if loss is None:
                raise RuntimeError("model did not return a training loss")
            losses.append(float(loss.detach()))
            (loss / len(train_batches)).backward()
    finally:
        model.flush_block_fht_cache()
    all_parameters = optimizer_parameters(optimizer)
    gradient_norm_before_clip = _gradient_norm(all_parameters)
    reported_norm = torch.nn.utils.clip_grad_norm_(
        model.product_fht_clip_parameters(), float(config["grad_clip"])
    )
    gradient_norm_after_clip = _gradient_norm(all_parameters)
    cfc_owner, cfc_group = _owner_and_group(optimizer, MuonFunctionalShear)
    cproj_owner, cproj_group = _owner_and_group(optimizer, MuonMatchedGivens)
    owners = {"c_fc": (cfc_owner, cfc_group), "c_proj": (cproj_owner, cproj_group)}
    bases = {
        family: {
            layer: weight.detach().cpu().clone()
            for layer, weight in by_layer.items()
        }
        for family, by_layer in weights.items()
    }
    dense_historical = {"c_fc": {}, "c_proj": {}}
    raw_by_level: dict[str, dict[str, dict[int, torch.Tensor]]] = {
        str(stages): {"c_fc": {}, "c_proj": {}} for stages in residual_stages
    }
    diagnostics: dict[str, Any] = {str(stages): {"c_fc": {}, "c_proj": {}} for stages in residual_stages}
    for family in ("c_fc", "c_proj"):
        owner, group = owners[family]
        learning_rate = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        for layer, weight in weights[family].items():
            if weight.grad is None:
                raise RuntimeError(f"missing {family} gradient for layer {layer}")
            momentum = owner.state[weight].get("momentum_buffer")
            if momentum is None:
                raise RuntimeError(f"missing {family} momentum for layer {layer}")
            canonical, descent, _dense_row = exact_muon_update(
                weight,
                weight.grad,
                momentum,
                learning_rate=learning_rate,
                momentum=float(group["momentum"]),
                weight_decay=weight_decay,
                ns_steps=int(group["ns_steps"]),
            )
            direction = descent + weight_decay * weight.float()
            dense_historical[family][layer] = historical_double_decay_update(
                weight,
                canonical,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            ).cpu()
            if family == "c_fc":
                module = model.transformer.h[layer].mlp.c_fc
                inputs = module._functional_inputs
                pre_gelu = module._functional_pre_gelu
                if inputs is None or pre_gelu is None:
                    raise RuntimeError("functional c_fc context is missing")
                cproj_weight = model.transformer.h[layer].mlp.c_proj.weight
                for stages in residual_stages:
                    raw, row = functional_coordinate_mix_update(
                        weight,
                        canonical,
                        direction,
                        inputs,
                        pre_gelu,
                        cproj_weight,
                        parent_stages=int(module.parent_stages),
                        shear_stages=int(stages),
                        neighbors=int(module.neighbors),
                        seed=int(module.matching_seed) + int(module.optimizer_step),
                        beta=float(module.coordinate_mix_beta),
                        project_to_weight_norm=bool(module.project_to_weight_norm),
                        max_condition_number=module.max_condition_number,
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                    )
                    raw_by_level[str(stages)][family][layer] = raw.cpu()
                    diagnostics[str(stages)][family][layer] = row
            else:
                module = model.transformer.h[layer].mlp.c_proj
                for stages in residual_stages:
                    raw, row = reconstruct_cproj_update(
                        weight,
                        canonical,
                        direction,
                        parent_stages=int(module.stages),
                        residual_stages=int(stages),
                        neighbors=int(module.neighbors),
                        seed=int(module.matching_seed) + int(module.optimizer_step),
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                    )
                    raw_by_level[str(stages)][family][layer] = raw.cpu()
                    diagnostics[str(stages)][family][layer] = row
    metadata = {
        "checkpoint_next_iter": int(checkpoint_payload["next_iter"]),
        "mean_training_ce": sum(losses) / len(losses),
        "gradient_norm_before_clip": gradient_norm_before_clip,
        "clip_norm_reported": float(reported_norm),
        "gradient_norm_after_clip": gradient_norm_after_clip,
        "diagnostics": diagnostics,
    }
    del model, optimizer
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return bases, dense_historical, {"raw": raw_by_level, "metadata": metadata}


def candidate_names(levels: list[int]) -> list[str]:
    names = [
        "baseline",
        "production_cfc",
        "production_cproj",
        "production_joint",
        "dense_norm_cfc",
        "dense_norm_cproj",
        "dense_norm_joint",
        "hybrid_norm_cfc",
        "hybrid_norm_cproj",
    ]
    for level in levels:
        names.extend(
            [
                f"cfc{level}_only",
                f"cproj{level}_only",
                f"hybrid_cfc{level}",
                f"hybrid_cproj{level}",
                f"joint{level}",
            ]
        )
    return names


def fraction_recovered(candidate: float, production: float, oracle: float) -> float:
    denominator = production - oracle
    return (production - candidate) / denominator if denominator > 0.0 else float("nan")


def classify_capacity(
    rows: list[dict[str, Any]],
    levels: list[int],
    *,
    confidence_z: float,
    minimum_fraction: float,
    mean_fraction: float,
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    means: dict[str, float] = {}
    for row in rows:
        means.setdefault(row["point_id"], 0.0)
    for point_id in means:
        values = [float(row["ce"]) for row in rows if row["point_id"] == point_id]
        means[point_id] = sum(values) / len(values)
    oracle_refs = {
        "cfc_single": ("dense_norm_cfc", "production_cfc"),
        "cproj_single": ("dense_norm_cproj", "production_cproj"),
        "cfc_hybrid": ("hybrid_norm_cfc", "production_joint"),
        "cproj_hybrid": ("hybrid_norm_cproj", "production_joint"),
        "joint": ("dense_norm_joint", "production_joint"),
    }
    results = []
    for level in levels:
        point_pairs = {
            "cfc_single": (f"cfc{level}_only", "production_cfc"),
            "cproj_single": (f"cproj{level}_only", "production_cproj"),
            "cfc_hybrid": (f"hybrid_cfc{level}", "production_joint"),
            "cproj_hybrid": (f"hybrid_cproj{level}", "production_joint"),
            "joint": (f"joint{level}", "production_joint"),
        }
        fractions = {}
        reliable = {}
        for name, (candidate, reference) in point_pairs.items():
            comparison = paired_comparison(
                rows, candidate, reference, float(confidence_z)
            )
            comparisons[f"{name}_{level}"] = comparison
            reliable[name] = bool(comparison["candidate_reliably_better"])
            oracle, production = oracle_refs[name]
            fractions[name] = fraction_recovered(
                means[candidate], means[production], means[oracle]
            )
        cfc_pass = (
            reliable["cfc_single"]
            and reliable["cfc_hybrid"]
            and min(fractions["cfc_single"], fractions["cfc_hybrid"])
            >= float(minimum_fraction)
            and (fractions["cfc_single"] + fractions["cfc_hybrid"]) / 2.0
            >= float(mean_fraction)
        )
        cproj_pass = (
            reliable["cproj_single"]
            and reliable["cproj_hybrid"]
            and min(fractions["cproj_single"], fractions["cproj_hybrid"])
            >= float(minimum_fraction)
            and (fractions["cproj_single"] + fractions["cproj_hybrid"]) / 2.0
            >= float(mean_fraction)
        )
        joint_pass = (
            reliable["joint"]
            and fractions["joint"] >= float(mean_fraction)
        )
        results.append(
            {
                "residual_stages": int(level),
                "reliable": reliable,
                "oracle_gap_fraction_recovered": fractions,
                "cfc_pass": cfc_pass,
                "cproj_pass": cproj_pass,
                "joint_pass": joint_pass,
                "full_pass": cfc_pass and cproj_pass and joint_pass,
            }
        )
    selected = next((row for row in results if row["full_pass"]), None)
    if selected is not None:
        classification = "SAME_TOPOLOGY_CAPACITY_PASSES_FIXED_RADIUS"
        next_action = "IMPLEMENT_AND_PREFLIGHT_SMALLEST_PASSING_CAPACITY"
    elif any(row["cfc_pass"] or row["cproj_pass"] for row in results):
        classification = "CAPACITY_HELPS_INDIVIDUALS_BUT_JOINT_TOPOLOGY_FAILS"
        next_action = "DESIGN_COUPLED_CHART_WITHOUT_INCREASING_STEP_RADIUS"
    else:
        classification = "SAME_TOPOLOGY_CAPACITY_REJECTED"
        next_action = "CHANGE_TASK_MATCHED_CHART_TOPOLOGY_NOT_RADIUS"
    return {
        "classification": classification,
        "selected_residual_stages": (
            None if selected is None else selected["residual_stages"]
        ),
        "next_action": next_action,
        "candidate_means": means,
        "comparisons": comparisons,
        "levels": results,
    }


def validate_plan(
    plan_path: Path, checkpoint: Path, config_path: Path, data_dir: Path
) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    actual = {
        "checkpoint_sha256": file_sha256(checkpoint),
        "config_sha256": file_sha256(config_path),
        "dataset_manifest_sha256": file_sha256(data_dir / "manifest.json"),
        "entrypoint_sha256": file_sha256(Path(__file__).resolve()),
    }
    for key, value in actual.items():
        if value != plan["identity"][key]:
            raise ValueError(f"registered identity mismatch: {key}")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    plan = validate_plan(args.plan, args.checkpoint, args.config, args.data_dir)
    protocol = plan["protocol"]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[str(config["dtype"])]
    train_batches = fixed_batches(
        args.data_dir,
        "train",
        batch_size=int(config["batch_size"]),
        block_size=int(config["block_size"]) + 1,
        batches=int(protocol["gradient_accumulation_steps"]),
        seed=int(protocol["train_seed"]),
    )
    production, _dense, extracted = extract_production_updates(
        args.checkpoint,
        config,
        train_batches,
        device=args.device,
        dtype=dtype,
        return_dense_oracle=True,
    )
    exactness = assert_joint_matches_singletons(
        production["cfc_only"]["c_fc"],
        production["cproj_only"]["c_proj"],
        production["joint"],
    )
    levels = [int(value) for value in protocol["expanded_residual_stages"]]
    control_level = int(protocol["control_residual_stages"])
    all_levels = [control_level, *levels]
    bases, dense_historical, reconstructed = (
        extract_reconstructed_capacity_updates(
            args.checkpoint,
            config,
            train_batches,
            all_levels,
            device=args.device,
            dtype=dtype,
        )
    )
    raw = reconstructed["raw"]
    prod_cfc = production["cfc_only"]["c_fc"]
    prod_cproj = production["cproj_only"]["c_proj"]
    reconstructed_control = {
        "c_fc": {
            layer: quantized_update(bases["c_fc"][layer], update)
            for layer, update in raw[str(control_level)]["c_fc"].items()
        },
        "c_proj": {
            layer: quantized_update(bases["c_proj"][layer], update)
            for layer, update in raw[str(control_level)]["c_proj"].items()
        },
    }
    reconstruction_max_abs = {}
    for family, expected in (("c_fc", prod_cfc), ("c_proj", prod_cproj)):
        reconstruction_max_abs[family] = max(
            float((expected[layer] - reconstructed_control[family][layer]).abs().max())
            for layer in expected
        )
        if reconstruction_max_abs[family] > float(
            protocol["control_max_abs_tolerance"]
        ):
            raise RuntimeError(
                f"{family} production reconstruction failed: "
                f"{reconstruction_max_abs[family]}"
            )
    target_norms = {"c_fc": family_fro(prod_cfc), "c_proj": family_fro(prod_cproj)}
    normalized: dict[str, dict[str, dict[int, torch.Tensor]]] = {}
    normalization = {}
    for level in levels:
        normalized[str(level)] = {}
        normalization[str(level)] = {}
        for family in ("c_fc", "c_proj"):
            updates, row = normalize_family_to_radius(
                bases[family], raw[str(level)][family], target_norms[family]
            )
            if row["relative_radius_error"] > float(
                protocol["maximum_relative_radius_error"]
            ):
                raise RuntimeError(f"{family}/{level} radius normalization failed")
            normalized[str(level)][family] = updates
            normalization[str(level)][family] = row
    norm_dense = {
        family: scale_family(
            dense_historical[family],
            target_norms[family] / family_fro(dense_historical[family]),
        )
        for family in ("c_fc", "c_proj")
    }
    candidates: dict[str, dict[str, dict[int, torch.Tensor]]] = {
        "baseline": {},
        "production_cfc": {"c_fc": prod_cfc},
        "production_cproj": {"c_proj": prod_cproj},
        "production_joint": merge_updates(prod_cfc, prod_cproj),
        "dense_norm_cfc": {"c_fc": norm_dense["c_fc"]},
        "dense_norm_cproj": {"c_proj": norm_dense["c_proj"]},
        "dense_norm_joint": merge_updates(norm_dense["c_fc"], norm_dense["c_proj"]),
        "hybrid_norm_cfc": merge_updates(norm_dense["c_fc"], prod_cproj),
        "hybrid_norm_cproj": merge_updates(prod_cfc, norm_dense["c_proj"]),
    }
    for level in levels:
        cfc = normalized[str(level)]["c_fc"]
        cproj = normalized[str(level)]["c_proj"]
        candidates.update(
            {
                f"cfc{level}_only": {"c_fc": cfc},
                f"cproj{level}_only": {"c_proj": cproj},
                f"hybrid_cfc{level}": merge_updates(cfc, prod_cproj),
                f"hybrid_cproj{level}": merge_updates(prod_cfc, cproj),
                f"joint{level}": merge_updates(cfc, cproj),
            }
        )
    expected_names = candidate_names(levels)
    if list(candidates) != expected_names:
        raise RuntimeError("candidate set differs from registered order")
    model, _optimizer, checkpoint_payload = load_model_and_optimizer(
        args.checkpoint, config, args.device
    )
    applier = ExactVariantApplier(model)
    validation_windows = {
        f"window_{index + 1}": fixed_batches(
            args.data_dir,
            "val",
            batch_size=int(protocol["evaluation_batch_size"]),
            block_size=int(protocol["evaluation_block_size"]) + 1,
            batches=int(protocol["validation_batches_per_window"]),
            seed=int(seed),
        )
        for index, seed in enumerate(protocol["validation_seeds"])
    }
    ce_rows = evaluate_candidates(
        model,
        applier,
        validation_windows,
        candidates,
        device=args.device,
        dtype=dtype,
    )
    decision_rule = plan["decision_rule"]
    decision = classify_capacity(
        ce_rows,
        levels,
        confidence_z=float(decision_rule["confidence_z"]),
        minimum_fraction=float(decision_rule["minimum_oracle_gap_fraction"]),
        mean_fraction=float(decision_rule["mean_oracle_gap_fraction"]),
    )
    direction_recovery = {
        str(level): {
            family: aggregate_direction_metrics(
                norm_dense[family], normalized[str(level)][family]
            )
            for family in ("c_fc", "c_proj")
        }
        for level in levels
    }
    args.output.mkdir(parents=True, exist_ok=False)
    paths = {
        "ce": args.output / "heldout_ce.json",
        "reconstruction": args.output / "reconstruction_and_capacity.json",
        "prospective": args.output / "prospective_step_metadata.json",
    }
    paths["ce"].write_text(json.dumps(ce_rows, indent=2, sort_keys=True) + "\n")
    paths["reconstruction"].write_text(
        json.dumps(
            {
                "production_reconstruction_max_abs_error": reconstruction_max_abs,
                "normalization": normalization,
                "direction_recovery_against_norm_dense": direction_recovery,
                "chart_diagnostics": reconstructed["metadata"]["diagnostics"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    paths["prospective"].write_text(
        json.dumps(extracted, indent=2, sort_keys=True) + "\n"
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "production_reconstruction_max_abs_error": reconstruction_max_abs,
        "joint_singleton_update_max_abs_error": exactness,
        "normalization": normalization,
        "direction_recovery_against_norm_dense": direction_recovery,
        "parameter_updates_to_checkpoint": 0,
        "disposable_optimizer_steps": 3,
        "checkpoint_next_iter": int(checkpoint_payload["next_iter"]),
        "identity": {
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "config_sha256": file_sha256(args.config),
            "dataset_manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
            "plan_sha256": file_sha256(args.plan),
        },
        "outputs": {f"{name}_sha256": file_sha256(path) for name, path in paths.items()},
        "execution": {
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": str(Path(__file__).resolve()),
            "entrypoint_sha256": file_sha256(Path(__file__).resolve()),
            "command": sys.argv,
            "device": args.device,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "direct_foreground_polling": True,
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
