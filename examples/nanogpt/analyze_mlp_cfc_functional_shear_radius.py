#!/usr/bin/env python3
"""Screen trust radii for function-fitted c_fc shear coordinates.

The sparse pair topology is fixed by the task/weight residual.  Functional
coordinates are fitted once on a registered train window, then only those
coordinates are scaled and replayed through the exact determinant-one maps.
Exact nonlinear post-GELU and MLP-output effects are scored on fit and a new
independent train holdout.  This is a zero-update diagnostic, not CE training.
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
    fit_functional_shear_recipe,
    pair_matched_permutations,
    replay_functional_shear_recipe,
    sample_aligned,
    safe_ratio,
    weighted,
)
from examples.nanogpt.analyze_mlp_cfc_residual_structure import (
    validate_identity,
    write_csv,
)
from examples.nanogpt.analyze_mlp_cfc_task_shear_fit import fit_pair_flow
from examples.nanogpt.analyze_mlp_muon_matched_givens import (
    diagonal_metric_causal_givens_update,
)
from examples.nanogpt.fast_task_matching import fast_muon_matched_permutations


SCHEMA_VERSION = "nanogpt_mlp_cfc_functional_shear_radius_v1"
WINDOWS = ("fit", "holdout")


def git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def scale_name(scale: float) -> str:
    return f"functional_radius_{scale:.6f}".replace(".", "p")


@torch.no_grad()
def build_radius_candidates(
    weight: torch.Tensor,
    dense_update: torch.Tensor,
    polar_descent_per_lr: torch.Tensor,
    inputs: torch.Tensor,
    pre_gelu: torch.Tensor,
    cproj_weight: torch.Tensor,
    *,
    scales: list[float],
    neighbors: int,
    seed: int,
    learning_rate: float,
    weight_decay: float,
    native_cache: Path | None,
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
    weight_residual, weight_fit = fit_pair_flow(
        after_parent,
        residual,
        weight_permutations,
        stages=24,
        family="shear",
    )
    _full, functional_fit, recipe = fit_functional_shear_recipe(
        after_parent,
        residual,
        inputs,
        pre_gelu,
        cproj_weight,
        weight_permutations,
        stages=24,
    )
    rotations = {
        CONTROL: parent + control_residual,
        WEIGHT_SHEAR: parent + weight_residual,
    }
    replay_diagnostics: list[dict[str, Any]] = []
    for scale in scales:
        fitted, finite = replay_functional_shear_recipe(
            after_parent, recipe, coordinate_scale=scale
        )
        name = scale_name(scale)
        rotations[name] = parent + fitted
        replay_diagnostics.append(
            {"candidate": name, "scale": scale, "finite": finite}
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
        {
            "candidate": CONTROL,
            "selection": control_selection,
            "fit": control_fit,
        },
        {
            "candidate": WEIGHT_SHEAR,
            "selection": weight_selection,
            "fit": weight_fit,
        },
        {
            "candidate": "functional_coordinate_fit",
            "selection": weight_selection,
            "fit": functional_fit,
        },
        *replay_diagnostics,
    ]
    return candidates, diagnostics


def aggregate_radius(
    metric_rows: list[dict[str, Any]],
    fit_rows: list[dict[str, Any]],
    *,
    scales: list[float],
    minimum_mlp_output_ratio: float,
    minimum_post_gelu_ratio: float,
    minimum_ce_descent_ratio: float,
    minimum_weight_ratio: float,
    maximum_determinant_error: float,
    maximum_condition_number: float,
) -> dict[str, Any]:
    names = (CONTROL, WEIGHT_SHEAR, *(scale_name(scale) for scale in scales))
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    for name in names:
        metrics[name] = {}
        for window in WINDOWS:
            rows = [
                row
                for row in metric_rows
                if row["candidate"] == name and row["window"] == window
            ]
            metrics[name][window] = {
                "weight_fixed_scale_recovery": weighted(
                    rows, "weight_fixed_scale_recovery", "weight_target_energy"
                ),
                "post_gelu_fixed_scale_recovery": weighted(
                    rows, "post_gelu_fixed_scale_recovery", "post_gelu_target_energy"
                ),
                "mlp_output_fixed_scale_recovery": weighted(
                    rows, "mlp_output_fixed_scale_recovery", "mlp_output_target_energy"
                ),
                "predicted_ce_decrease": sum(
                    float(row["predicted_ce_decrease"]) for row in rows
                ),
            }
    control = metrics[WEIGHT_SHEAR]
    ratios: dict[str, dict[str, dict[str, float]]] = {}
    passing: list[float] = []
    stable_by_name: dict[str, bool] = {}
    for scale in scales:
        name = scale_name(scale)
        finite_rows = [row["finite"] for row in fit_rows if row["candidate"] == name]
        stable = bool(
            finite_rows
            and max(float(row["maximum_determinant_error"]) for row in finite_rows)
            <= maximum_determinant_error
            and max(float(row["maximum_condition_number"]) for row in finite_rows)
            <= maximum_condition_number
        )
        stable_by_name[name] = stable
        ratios[name] = {}
        for window in WINDOWS:
            ratios[name][window] = {
                "mlp_output_vs_weight_shear": safe_ratio(
                    metrics[name][window]["mlp_output_fixed_scale_recovery"],
                    control[window]["mlp_output_fixed_scale_recovery"],
                ),
                "post_gelu_vs_weight_shear": safe_ratio(
                    metrics[name][window]["post_gelu_fixed_scale_recovery"],
                    control[window]["post_gelu_fixed_scale_recovery"],
                ),
                "ce_descent_vs_weight_shear": safe_ratio(
                    metrics[name][window]["predicted_ce_decrease"],
                    control[window]["predicted_ce_decrease"],
                ),
                "weight_vs_weight_shear": safe_ratio(
                    metrics[name][window]["weight_fixed_scale_recovery"],
                    control[window]["weight_fixed_scale_recovery"],
                ),
            }
        passes = bool(
            stable
            and min(
                ratios[name][window]["mlp_output_vs_weight_shear"]
                for window in WINDOWS
            )
            >= minimum_mlp_output_ratio
            and min(
                ratios[name][window]["post_gelu_vs_weight_shear"]
                for window in WINDOWS
            )
            >= minimum_post_gelu_ratio
            and min(
                ratios[name][window]["ce_descent_vs_weight_shear"]
                for window in WINDOWS
            )
            >= minimum_ce_descent_ratio
            and min(
                ratios[name][window]["weight_vs_weight_shear"]
                for window in WINDOWS
            )
            >= minimum_weight_ratio
        )
        if passes:
            passing.append(scale)
    selected = min(passing) if passing else None
    return {
        "decision": (
            "PROMOTE_FUNCTIONAL_COORDINATE_RADIUS_TO_FINITE_CE"
            if selected is not None
            else "REJECT_FUNCTIONAL_COORDINATE_TRUST_RADIUS"
        ),
        "parameter_updates": 0,
        "selected_scale": selected,
        "passing_scales": passing,
        "metrics": metrics,
        "ratios": ratios,
        "stable_by_candidate": stable_by_name,
        "thresholds": {
            "minimum_mlp_output_ratio": minimum_mlp_output_ratio,
            "minimum_post_gelu_ratio": minimum_post_gelu_ratio,
            "minimum_ce_descent_ratio": minimum_ce_descent_ratio,
            "minimum_weight_ratio": minimum_weight_ratio,
            "maximum_determinant_error": maximum_determinant_error,
            "maximum_condition_number": maximum_condition_number,
        },
    }


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
    scales = [float(scale) for scale in protocol["coordinate_scales"]]
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
    candidate_names = (CONTROL, WEIGHT_SHEAR, *(scale_name(scale) for scale in scales))
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
        fitted, diagnostics = build_radius_candidates(
            weight.detach(),
            dense_update,
            polar_descent,
            sampled["fit"]["inputs"][layer],
            sampled["fit"]["pre_gelu"][layer],
            model.transformer.h[layer].mlp.c_proj.weight.detach(),
            scales=scales,
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
        scales=scales,
        minimum_mlp_output_ratio=float(rule["minimum_mlp_output_ratio"]),
        minimum_post_gelu_ratio=float(rule["minimum_post_gelu_ratio"]),
        minimum_ce_descent_ratio=float(rule["minimum_ce_descent_ratio"]),
        minimum_weight_ratio=float(rule["minimum_weight_ratio"]),
        maximum_determinant_error=float(rule["maximum_determinant_error"]),
        maximum_condition_number=float(rule["maximum_condition_number"]),
    )
    result["fit_loss_bfloat16"] = float(collected["fit"]["loss"])
    result["holdout_loss_bfloat16"] = float(collected["holdout"]["loss"])
    args.output.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": args.output / "cfc_functional_shear_radius_metrics.csv",
        "fits": args.output / "cfc_functional_shear_radius_fits.json",
        "optimizer": args.output / "cfc_functional_shear_radius_optimizer.csv",
        "aggregate": args.output / "cfc_functional_shear_radius_aggregate.json",
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
    metadata_path = args.output / "cfc_functional_shear_radius_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": result["decision"], "aggregate": str(paths["aggregate"]), "metadata": str(metadata_path)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
