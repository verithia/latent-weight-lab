#!/usr/bin/env python3
"""Interpolate weight-fit and function-fit c_fc shear coordinates.

Both endpoint recipes use the same task/weight-selected pair topology after
the qualified fresh64 rotational parent.  The exact stage coordinates are
interpolated before replaying determinant-one maps.  Exact nonlinear GELU and
MLP-output effects are scored on fit and a new independent train holdout.
This is a zero-update Pareto diagnostic, not finite CE or training.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.nanogpt.analyze_mlp_cfc_exact_current_matcher import (
    _optimizer_and_group_for_parameter,
    _weight_decay_after_rotation,
    activation_effect_metrics,
    collect_window,
    direction_metrics,
    exact_muon_update,
    file_sha256,
    fixed_batches,
    load_model_and_optimizer,
)
from examples.nanogpt.analyze_mlp_cfc_functional_shear_fit import (
    CONTROL,
    WEIGHT_SHEAR,
    _fit_rotational_parent,
    pair_matched_permutations,
    sample_aligned,
)
from examples.nanogpt.analyze_mlp_cfc_functional_shear_radius import (
    aggregate_radius,
    scale_name,
)
from examples.nanogpt.analyze_mlp_cfc_residual_structure import (
    validate_identity,
    write_csv,
)
from examples.nanogpt.analyze_mlp_cfc_task_shear_fit import (
    apply_pair_stage,
    fit_pair_recipe,
)
from examples.nanogpt.analyze_mlp_muon_matched_givens import (
    diagonal_metric_causal_givens_update,
)
from examples.nanogpt.fast_task_matching import fast_muon_matched_permutations
from examples.nanogpt.muon_matched_givens import (
    _fit_functional_shear_recipe,
    mix_shear_recipes,
)


SCHEMA_VERSION = "nanogpt_mlp_cfc_functional_shear_pareto_v1"
WINDOWS = ("fit", "holdout")
PREFIX = "functional_mix"


def git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


@torch.no_grad()
def replay_blended_recipes(
    source: torch.Tensor,
    weight_recipe: list[tuple[torch.Tensor, torch.Tensor]],
    functional_recipe: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    beta: float,
    project_to_weight_norm: bool = False,
    max_condition_number: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Replay stage coordinates `(1-beta)*weight + beta*functional`."""
    if not 0.0 <= float(beta) <= 1.0:
        raise ValueError("beta must be in [0, 1]")
    if len(weight_recipe) != len(functional_recipe):
        raise ValueError("recipe lengths differ")
    mixed_recipe, projection_diagnostics = mix_shear_recipes(
        weight_recipe,
        functional_recipe,
        beta=beta,
        project_to_weight_norm=project_to_weight_norm,
        max_condition_number=max_condition_number,
    )
    current = source.float().clone()
    minimum_determinant = float("inf")
    maximum_determinant_error = 0.0
    maximum_condition_number = 0.0
    for weight_pairs, coordinates in mixed_recipe:
        current, finite = apply_pair_stage(
            current,
            weight_pairs.to(device=source.device),
            coordinates.to(device=source.device),
        )
        minimum_determinant = min(
            minimum_determinant, finite["minimum_determinant"]
        )
        maximum_determinant_error = max(
            maximum_determinant_error, finite["maximum_determinant_error"]
        )
        maximum_condition_number = max(
            maximum_condition_number, finite["maximum_condition_number"]
        )
    return current - source.float(), {
        "beta": float(beta),
        **projection_diagnostics,
        "minimum_determinant": minimum_determinant,
        "maximum_determinant_error": maximum_determinant_error,
        "maximum_condition_number": maximum_condition_number,
    }


@torch.no_grad()
def build_pareto_candidates(
    weight: torch.Tensor,
    dense_update: torch.Tensor,
    polar_descent_per_lr: torch.Tensor,
    inputs: torch.Tensor,
    pre_gelu: torch.Tensor,
    cproj_weight: torch.Tensor,
    *,
    betas: list[float],
    neighbors: int,
    seed: int,
    learning_rate: float,
    weight_decay: float,
    native_cache: Path | None,
    project_to_weight_norm: bool = False,
    max_condition_number: float | None = None,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    source = weight.float().T.contiguous()
    target = dense_update.float().T.contiguous()
    selection_direction = polar_descent_per_lr.float().T.contiguous()
    after_parent, parent, parent_diagnostics = _fit_rotational_parent(
        source,
        target,
        selection_direction,
        stages=64,
        neighbors=neighbors,
        seed=seed,
        native_cache=native_cache,
    )
    residual = target - parent
    control_permutations, control_selection = fast_muon_matched_permutations(
        after_parent,
        residual,
        stages=24,
        neighbors=neighbors,
        seed=seed + 1,
        cache_dir=native_cache,
    )
    control_residual, control_fit = diagonal_metric_causal_givens_update(
        after_parent,
        residual,
        stages=24,
        seed=seed + 1,
        permutations=control_permutations,
    )
    weight_permutations, weight_selection = pair_matched_permutations(
        after_parent,
        residual,
        stages=24,
        neighbors=neighbors,
        seed=seed + 2,
        family="shear",
        native_cache=native_cache,
    )
    weight_residual, weight_fit, weight_recipe = fit_pair_recipe(
        after_parent,
        residual,
        weight_permutations,
        stages=24,
        family="shear",
    )
    functional_fit_diagnostics: dict[str, float | bool] = {}
    functional_recipe = _fit_functional_shear_recipe(
        after_parent,
        residual,
        inputs,
        pre_gelu,
        cproj_weight,
        weight_permutations,
        max_condition_number=max_condition_number,
        fit_diagnostics=functional_fit_diagnostics,
    )
    functional_current = after_parent.float()
    maximum_determinant_error = 0.0
    maximum_pair_condition = 0.0
    for pairs, coordinates in functional_recipe:
        functional_current, finite = apply_pair_stage(
            functional_current,
            pairs.to(device=after_parent.device),
            coordinates.to(device=after_parent.device),
        )
        maximum_determinant_error = max(
            maximum_determinant_error,
            float(finite["maximum_determinant_error"]),
        )
        maximum_pair_condition = max(
            maximum_pair_condition,
            float(finite["maximum_condition_number"]),
        )
    functional_fit = {
        "family": "production_functional_shear",
        "coordinates": sum(
            int(coordinates.numel())
            for _pairs, coordinates in functional_recipe
        ),
        "stages": len(functional_recipe),
        "maximum_determinant_error": maximum_determinant_error,
        "maximum_condition_number": maximum_pair_condition,
        **functional_fit_diagnostics,
    }
    rotations = {
        CONTROL: parent + control_residual,
        WEIGHT_SHEAR: parent + weight_residual,
    }
    blend_diagnostics: list[dict[str, Any]] = []
    for beta in betas:
        blended, finite = replay_blended_recipes(
            after_parent,
            weight_recipe,
            functional_recipe,
            beta=beta,
            project_to_weight_norm=project_to_weight_norm,
            max_condition_number=max_condition_number,
        )
        name = scale_name(beta, prefix=PREFIX)
        rotations[name] = parent + blended
        blend_diagnostics.append(
            {"candidate": name, "beta": beta, "finite": finite}
        )
    candidates = {
        name: _weight_decay_after_rotation(
            source,
            rotation,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        .T.contiguous()
        for name, rotation in rotations.items()
    }
    diagnostics = [
        {"candidate": "fresh64_parent", **parent_diagnostics},
        {"candidate": CONTROL, "selection": control_selection, "fit": control_fit},
        {
            "candidate": WEIGHT_SHEAR,
            "selection": weight_selection,
            "fit": weight_fit,
        },
        {
            "candidate": "functional_coordinate_endpoint",
            "selection": weight_selection,
            "fit": functional_fit,
        },
        *blend_diagnostics,
    ]
    return candidates, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--native-cache", type=Path)
    args = parser.parse_args()
    started = time.time()
    plan = validate_identity(args.checkpoint, args.config, args.data_dir, args.plan)
    protocol = plan["fixed_protocol"]
    rule = plan["decision_rule"]
    layers = [int(layer) for layer in protocol["layers"]]
    betas = [float(beta) for beta in protocol["coordinate_mix_betas"]]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    batches = {
        "fit": fixed_batches(
            args.data_dir,
            "train",
            batch_size=int(protocol["batch_size"]),
            block_size=int(protocol["block_size"]),
            batches=int(protocol["batches_per_window"]),
            seed=int(protocol["fit_train_seed"]),
        ),
        "holdout": fixed_batches(
            args.data_dir,
            "train",
            batch_size=int(protocol["batch_size"]),
            block_size=int(protocol["block_size"]),
            batches=int(protocol["batches_per_window"]),
            seed=int(protocol["holdout_train_seed"]),
        ),
    }
    model, optimizer, checkpoint = load_model_and_optimizer(
        args.checkpoint, config, args.device
    )
    collected: dict[str, dict[str, Any]] = {}
    for window in WINDOWS:
        loss, gradients, inputs, pre_gelu = collect_window(
            model,
            batches[window],
            layers,
            device=args.device,
            dtype=torch.bfloat16,
        )
        collected[window] = {
            "loss": loss,
            "gradients": gradients,
            "inputs": inputs,
            "pre_gelu": pre_gelu,
        }
    sampled: dict[str, dict[str, dict[int, torch.Tensor]]] = {}
    sample_sha: dict[str, str] = {}
    for window_index, window in enumerate(WINDOWS):
        sampled[window] = {"inputs": {}, "pre_gelu": {}}
        for layer in layers:
            inputs, pre_gelu, sha = sample_aligned(
                collected[window]["inputs"][layer],
                collected[window]["pre_gelu"][layer],
                sample_cap=int(protocol["functional_sample_cap"]),
                seed=int(protocol["functional_sample_seed"]) + window_index,
            )
            sampled[window]["inputs"][layer] = inputs
            sampled[window]["pre_gelu"][layer] = pre_gelu
            sample_sha[f"{window}_layer{layer}"] = sha
    candidate_names = (
        CONTROL,
        WEIGHT_SHEAR,
        *(scale_name(beta, prefix=PREFIX) for beta in betas),
    )
    updates: dict[str, dict[int, torch.Tensor]] = {
        name: {} for name in candidate_names
    }
    dense_updates: dict[int, torch.Tensor] = {}
    fit_rows: list[dict[str, Any]] = []
    optimizer_rows: list[dict[str, Any]] = []
    for layer in layers:
        weight = model.transformer.h[layer].mlp.c_fc.weight
        owner, group = _optimizer_and_group_for_parameter(optimizer, weight)
        buffer = owner.state[weight].get("momentum_buffer")
        if buffer is None:
            raise RuntimeError(f"missing c_fc momentum at layer {layer}")
        dense_update, descent, optimizer_diag = exact_muon_update(
            weight.detach(),
            collected["fit"]["gradients"][layer].to(weight.device),
            buffer,
            learning_rate=float(group["lr"]),
            momentum=float(group["momentum"]),
            weight_decay=float(group["weight_decay"]),
            ns_steps=int(group["ns_steps"]),
        )
        polar_descent = descent + float(group["weight_decay"]) * weight.detach().float()
        fitted, diagnostics = build_pareto_candidates(
            weight.detach(),
            dense_update,
            polar_descent,
            sampled["fit"]["inputs"][layer],
            sampled["fit"]["pre_gelu"][layer],
            model.transformer.h[layer].mlp.c_proj.weight.detach(),
            betas=betas,
            neighbors=int(protocol["matching_neighbors"]),
            seed=int(protocol["matching_seed"]) + layer * 1009,
            learning_rate=float(group["lr"]),
            weight_decay=float(group["weight_decay"]),
            native_cache=args.native_cache,
        )
        dense_updates[layer] = dense_update.cpu()
        for name, update in fitted.items():
            updates[name][layer] = update.cpu()
        fit_rows.extend({"layer": layer, **row} for row in diagnostics)
        optimizer_rows.append({"layer": layer, **optimizer_diag})
        print(json.dumps({"layer_complete": layer, "layers_total": len(layers)}), flush=True)
    metric_rows: list[dict[str, Any]] = []
    for window in WINDOWS:
        for layer in layers:
            target = dense_updates[layer]
            cproj = model.transformer.h[layer].mlp.c_proj.weight.detach().cpu()
            for name in candidate_names:
                update = updates[name][layer]
                weight_metrics = direction_metrics(target, update)
                functional = activation_effect_metrics(
                    sampled[window]["inputs"][layer],
                    sampled[window]["pre_gelu"][layer],
                    cproj,
                    target,
                    update,
                    device=args.device,
                )
                metric_rows.append(
                    {
                        "window": window,
                        "layer": layer,
                        "candidate": name,
                        "weight_target_energy": weight_metrics["target_energy"],
                        "weight_fixed_scale_recovery": weight_metrics["fixed_scale_recovery"],
                        "post_gelu_target_energy": functional["post_gelu"]["target_energy"],
                        "post_gelu_fixed_scale_recovery": functional["post_gelu"]["fixed_scale_recovery"],
                        "mlp_output_target_energy": functional["mlp_output"]["target_energy"],
                        "mlp_output_fixed_scale_recovery": functional["mlp_output"]["fixed_scale_recovery"],
                        "predicted_ce_decrease": float(
                            -(collected[window]["gradients"][layer].double() * update.double()).sum()
                        ),
                    }
                )
    result = aggregate_radius(
        metric_rows,
        fit_rows,
        scales=betas,
        minimum_mlp_output_ratio=float(rule["minimum_mlp_output_ratio"]),
        minimum_post_gelu_ratio=float(rule["minimum_post_gelu_ratio"]),
        minimum_ce_descent_ratio=float(rule["minimum_ce_descent_ratio"]),
        minimum_weight_ratio=float(rule["minimum_weight_ratio"]),
        maximum_determinant_error=float(rule["maximum_determinant_error"]),
        maximum_condition_number=float(rule["maximum_condition_number"]),
        candidate_prefix=PREFIX,
    )
    result["decision"] = (
        "PROMOTE_FUNCTIONAL_COORDINATE_MIX_TO_FINITE_CE"
        if result["selected_scale"] is not None
        else "REJECT_FUNCTIONAL_COORDINATE_PARETO_MIX"
    )
    result["selected_beta"] = result.pop("selected_scale")
    result["passing_betas"] = result.pop("passing_scales")
    result["fit_loss_bfloat16"] = float(collected["fit"]["loss"])
    result["holdout_loss_bfloat16"] = float(collected["holdout"]["loss"])
    args.output.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": args.output / "cfc_functional_shear_pareto_metrics.csv",
        "fits": args.output / "cfc_functional_shear_pareto_fits.json",
        "optimizer": args.output / "cfc_functional_shear_pareto_optimizer.csv",
        "aggregate": args.output / "cfc_functional_shear_pareto_aggregate.json",
    }
    write_csv(paths["metrics"], metric_rows)
    paths["fits"].write_text(json.dumps(fit_rows, indent=2, sort_keys=True) + "\n")
    write_csv(paths["optimizer"], optimizer_rows)
    paths["aggregate"].write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "decision": result["decision"],
        "parameter_updates": 0,
        "checkpoint_next_iter": int(checkpoint["next_iter"]),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "config_sha256": file_sha256(args.config),
        "dataset_manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
        "plan_sha256": file_sha256(args.plan),
        "sample_indices_sha256": sample_sha,
        "analysis_execution": {
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": str(Path(__file__).resolve()),
            "entrypoint_sha256": file_sha256(Path(__file__).resolve()),
            "command": sys.argv,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "device": args.device,
            "direct_foreground_polling": True,
            "watchdog": False,
            "callback": False,
        },
        "protocol": protocol,
        "outputs": {f"{name}_sha256": file_sha256(path) for name, path in paths.items()},
        "limitations": plan["limitations"],
    }
    metadata_path = args.output / "cfc_functional_shear_pareto_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": result["decision"], "aggregate": str(paths["aggregate"]), "metadata": str(metadata_path)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
