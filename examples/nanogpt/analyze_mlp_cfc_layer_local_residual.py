#!/usr/bin/env python3
"""Localize the dense-minus-fresh88 c_fc tangent without training."""

from __future__ import annotations

import argparse
import csv
import json
import math
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
    build_candidates,
    exact_muon_update,
    file_sha256,
    fixed_batches,
    load_model_and_optimizer,
)
from examples.nanogpt.analyze_mlp_cfc_residual_structure import (
    residual_metrics,
    validate_identity,
    write_csv,
)
from examples.nanogpt.analyze_mlp_cfc_trust_radius import (
    collect_gradient_window,
    repeated_losses,
    summarize,
)


SCHEMA_VERSION = "nanogpt_mlp_cfc_layer_local_residual_v1"
SUBSPACE_FAMILIES = ("left", "right", "joint")
FAMILY_PRIORITY = {"joint": 0, "left": 1, "right": 2}


def git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def existing_weight_subspace_projections(
    weight: torch.Tensor,
    residual: torch.Tensor,
    *,
    rank: int,
) -> tuple[dict[str, torch.Tensor], list[float]]:
    """Project a residual through the current dense weight's singular frame."""
    if rank < 1 or rank > min(weight.shape):
        raise ValueError("invalid existing-weight subspace rank")
    u, singular_values, vh = torch.linalg.svd(
        weight.float(), full_matrices=False
    )
    left_basis = u[:, :rank]
    right_basis = vh[:rank]
    residual_f = residual.float()
    left = left_basis @ (left_basis.T @ residual_f)
    right = (residual_f @ right_basis.T) @ right_basis
    joint = left_basis @ (
        (left_basis.T @ residual_f @ right_basis.T) @ right_basis
    )
    return {
        "left": left,
        "right": right,
        "joint": joint,
    }, [float(value) for value in singular_values[:rank]]


def make_subset_update(
    fresh: dict[int, torch.Tensor],
    replacements: dict[int, torch.Tensor],
    selected_layers: list[int],
) -> dict[int, torch.Tensor]:
    result = dict(fresh)
    for layer in selected_layers:
        result[layer] = replacements[layer]
    return result


def recovery(
    fresh: dict[str, float],
    dense: dict[str, float],
    candidate: dict[str, float],
) -> float:
    gap = fresh["mean"] - dense["mean"]
    if gap <= 0.0:
        return float("nan")
    return (fresh["mean"] - candidate["mean"]) / gap


def median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def aggregate_results(
    rows: list[dict[str, Any]],
    *,
    windows: list[str],
    candidate_specs: dict[str, dict[str, Any]],
    fit_ranking: list[dict[str, Any]],
    numerical_range_tolerance: float,
    minimum_recovery_every_window: float,
    minimum_median_recovery: float,
    top3_concentration_threshold: float,
    top6_concentration_threshold: float,
) -> dict[str, Any]:
    summaries: dict[str, dict[str, dict[str, float]]] = {}
    for window in windows:
        summaries[window] = {}
        for candidate in ("baseline", *candidate_specs):
            values = [
                float(row["loss"])
                for row in rows
                if row["window"] == window
                and row["candidate"] == candidate
            ]
            summaries[window][candidate] = summarize(values)
    stable = all(
        value["range"] <= numerical_range_tolerance
        for window in summaries.values()
        for value in window.values()
    )
    dense_positive = all(
        values["dense_exact"]["maximum"]
        < values["fresh88"]["minimum"]
        for values in summaries.values()
    )
    candidate_results: dict[str, dict[str, Any]] = {}
    for candidate, spec in candidate_specs.items():
        if candidate in {"fresh88", "dense_exact"}:
            continue
        by_window = {
            window: recovery(
                values["fresh88"],
                values["dense_exact"],
                values[candidate],
            )
            for window, values in summaries.items()
        }
        finite = all(math.isfinite(value) for value in by_window.values())
        beats_fresh = all(
            values[candidate]["maximum"] < values["fresh88"]["minimum"]
            for values in summaries.values()
        )
        minimum = min(by_window.values()) if finite else float("nan")
        med = median(list(by_window.values())) if finite else float("nan")
        structural = spec["family"] in SUBSPACE_FAMILIES and spec["scope"] in {
            "top3",
            "all",
        }
        qualifies = all(
            (
                dense_positive,
                stable,
                finite,
                beats_fresh,
                structural,
                minimum >= minimum_recovery_every_window,
                med >= minimum_median_recovery,
            )
        )
        candidate_results[candidate] = {
            **spec,
            "recovery_by_window": by_window,
            "minimum_recovery": minimum,
            "median_recovery": med,
            "beats_fresh_every_window": beats_fresh,
            "qualifies": qualifies,
        }

    qualified = [
        (name, value)
        for name, value in candidate_results.items()
        if value["qualifies"]
    ]
    if not dense_positive:
        decision = "DENSE_RESIDUAL_NOT_POSITIVE_CONTROL"
        selected_name = None
    elif qualified:
        selected_name, selected = min(
            qualified,
            key=lambda item: (
                int(item[1]["coordinates_total"]),
                FAMILY_PRIORITY[str(item[1]["family"])],
                item[0],
            ),
        )
        decision = "SELECT_LAYER_LOCAL_SUBSPACE_FOR_CHART_DESIGN"
    else:
        top3 = candidate_results["exact_top3"]
        top6 = candidate_results["exact_top6"]
        if (
            top3["minimum_recovery"] >= top3_concentration_threshold
            and top3["beats_fresh_every_window"]
        ):
            decision = "LOCALIZE_DENSE_DEFICIT_TO_TOP3_LAYERS"
            selected_name = "exact_top3"
        elif (
            top6["minimum_recovery"] >= top6_concentration_threshold
            and top6["beats_fresh_every_window"]
        ):
            decision = "LOCALIZE_DENSE_DEFICIT_TO_TOP6_LAYERS"
            selected_name = "exact_top6"
        else:
            decision = "DISTRIBUTED_CFC_TANGENT_REQUIRES_NEW_CHART"
            selected_name = max(
                (
                    item
                    for item in candidate_results.items()
                    if item[1]["scope"] in {"top3", "all"}
                    and math.isfinite(item[1]["minimum_recovery"])
                ),
                key=lambda item: (
                    float(item[1]["minimum_recovery"]),
                    float(item[1]["median_recovery"]),
                ),
            )[0]
    return {
        "decision": decision,
        "selected_candidate": selected_name,
        "fit_layer_ranking": fit_ranking,
        "summaries": summaries,
        "candidate_results": candidate_results,
        "gates": {
            "numerically_stable": stable,
            "dense_beats_fresh_every_window": dense_positive,
        },
        "thresholds": {
            "numerical_range_tolerance": numerical_range_tolerance,
            "minimum_recovery_every_window": minimum_recovery_every_window,
            "minimum_median_recovery": minimum_median_recovery,
            "top3_concentration_threshold": top3_concentration_threshold,
            "top6_concentration_threshold": top6_concentration_threshold,
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
    plan = validate_identity(
        args.checkpoint, args.config, args.data_dir, args.plan
    )
    protocol = plan["fixed_protocol"]
    decision_rule = plan["decision_rule"]
    layers = [int(value) for value in protocol["layers"]]
    rank = int(protocol["existing_weight_subspace_rank"])
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fit_batches = fixed_batches(
        args.data_dir,
        "train",
        batch_size=int(protocol["batch_size"]),
        block_size=int(protocol["block_size"]),
        batches=int(protocol["fit_batches"]),
        seed=int(protocol["fit_train_seed"]),
    )
    windows = [
        f"validation_{index + 1}"
        for index in range(len(protocol["validation_seeds"]))
    ]
    validation_batches = {
        window: fixed_batches(
            args.data_dir,
            "val",
            batch_size=int(protocol["batch_size"]),
            block_size=int(protocol["block_size"]),
            batches=int(protocol["validation_batches_per_window"]),
            seed=int(seed),
        )
        for window, seed in zip(
            windows, protocol["validation_seeds"], strict=True
        )
    }
    model, optimizer, checkpoint = load_model_and_optimizer(
        args.checkpoint, config, args.device
    )
    fit_gradient_loss, gradients = collect_gradient_window(
        model,
        fit_batches,
        layers,
        device=args.device,
        dtype=torch.bfloat16,
    )
    fresh: dict[int, torch.Tensor] = {}
    dense: dict[int, torch.Tensor] = {}
    projected: dict[str, dict[int, torch.Tensor]] = {
        family: {} for family in SUBSPACE_FAMILIES
    }
    weight_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    weight_spectra: dict[str, list[float]] = {}
    for layer in layers:
        weight = model.transformer.h[layer].mlp.c_fc.weight
        owner, group = _optimizer_and_group_for_parameter(optimizer, weight)
        buffer = owner.state[weight].get("momentum_buffer")
        if buffer is None:
            raise RuntimeError(f"missing c_fc momentum at layer {layer}")
        dense_update, descent, _diagnostics = exact_muon_update(
            weight.detach(),
            gradients[layer].to(weight.device),
            buffer,
            learning_rate=float(group["lr"]),
            momentum=float(group["momentum"]),
            weight_decay=float(group["weight_decay"]),
            ns_steps=int(group["ns_steps"]),
        )
        polar_descent = (
            descent + float(group["weight_decay"]) * weight.detach().float()
        )
        matched, selections = build_candidates(
            weight.detach(),
            dense_update,
            polar_descent,
            parent_stages=64,
            residual_stages=24,
            neighbors=int(protocol["matching_neighbors"]),
            seed=int(protocol["matching_seed"]) + layer * 1009,
            learning_rate=float(group["lr"]),
            weight_decay=float(group["weight_decay"]),
            native_cache=args.native_cache,
        )
        fresh_update = matched["fresh_expansion88"].float()
        residual = dense_update.float() - fresh_update
        projections, singular_values = existing_weight_subspace_projections(
            weight.detach(), residual, rank=rank
        )
        fresh[layer] = fresh_update.cpu()
        dense[layer] = dense_update.float().cpu()
        weight_spectra[str(layer)] = singular_values
        for family, approximation in projections.items():
            projected[family][layer] = (fresh_update + approximation).cpu()
            weight_rows.append(
                {
                    "layer": layer,
                    "family": family,
                    "rank": rank,
                    **residual_metrics(residual, approximation),
                }
            )
        selection_rows.extend(
            {"layer": layer, **selection} for selection in selections
        )

    repeats = int(protocol["evaluation_repeats"])
    fit_fresh_values = repeated_losses(
        model,
        fit_batches,
        fresh,
        repeats=repeats,
        device=args.device,
        dtype=torch.float32,
    )
    fit_fresh_mean = sum(fit_fresh_values) / len(fit_fresh_values)
    fit_ranking: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    for repeat, loss in enumerate(fit_fresh_values):
        fit_rows.append(
            {"candidate": "fresh88", "layer": -1, "repeat": repeat, "loss": loss}
        )
    for layer in layers:
        candidate = make_subset_update(fresh, dense, [layer])
        values = repeated_losses(
            model,
            fit_batches,
            candidate,
            repeats=repeats,
            device=args.device,
            dtype=torch.float32,
        )
        for repeat, loss in enumerate(values):
            fit_rows.append(
                {
                    "candidate": f"exact_layer{layer}",
                    "layer": layer,
                    "repeat": repeat,
                    "loss": loss,
                }
            )
        fit_ranking.append(
            {
                "layer": layer,
                "fit_ce": sum(values) / len(values),
                "fit_improvement_over_fresh": fit_fresh_mean
                - sum(values) / len(values),
            }
        )
    fit_ranking.sort(
        key=lambda row: (-float(row["fit_improvement_over_fresh"]), int(row["layer"]))
    )
    selected = [int(row["layer"]) for row in fit_ranking]
    top_sets = {1: selected[:1], 3: selected[:3], 6: selected[:6]}

    dense_coordinates = 3072 * 768
    subspace_coordinates = {
        "left": rank * 768,
        "right": 3072 * rank,
        "joint": rank * rank,
    }
    candidate_specs: dict[str, dict[str, Any]] = {
        "fresh88": {"family": "fresh88", "scope": "all", "coordinates_total": 12 * 135168},
        "dense_exact": {"family": "dense_exact", "scope": "all", "coordinates_total": 12 * dense_coordinates},
    }
    candidate_updates: dict[str, dict[int, torch.Tensor]] = {
        "fresh88": fresh,
        "dense_exact": dense,
    }
    for count, chosen_layers in top_sets.items():
        name = f"exact_top{count}"
        candidate_specs[name] = {
            "family": "dense_exact_subset",
            "scope": f"top{count}",
            "layers": chosen_layers,
            "coordinates_total": count * dense_coordinates,
        }
        candidate_updates[name] = make_subset_update(
            fresh, dense, chosen_layers
        )
    for family in SUBSPACE_FAMILIES:
        for scope, chosen_layers in (("top3", top_sets[3]), ("all", layers)):
            name = f"{family}_{scope}_rank{rank}"
            candidate_specs[name] = {
                "family": family,
                "scope": scope,
                "layers": chosen_layers,
                "coordinates_total": len(chosen_layers)
                * subspace_coordinates[family],
            }
            candidate_updates[name] = make_subset_update(
                fresh, projected[family], chosen_layers
            )
    for layer in layers:
        exact_name = f"exact_layer{layer}"
        candidate_specs[exact_name] = {
            "family": "dense_exact_subset",
            "scope": "single_layer",
            "layers": [layer],
            "coordinates_total": dense_coordinates,
        }
        candidate_updates[exact_name] = make_subset_update(
            fresh, dense, [layer]
        )
        for family in SUBSPACE_FAMILIES:
            name = f"{family}_layer{layer}_rank{rank}"
            candidate_specs[name] = {
                "family": family,
                "scope": "single_layer",
                "layers": [layer],
                "coordinates_total": subspace_coordinates[family],
            }
            candidate_updates[name] = make_subset_update(
                fresh, projected[family], [layer]
            )

    loss_rows: list[dict[str, Any]] = []
    for window, batches in validation_batches.items():
        baseline_values = repeated_losses(
            model,
            batches,
            None,
            repeats=repeats,
            device=args.device,
            dtype=torch.float32,
        )
        for repeat, loss in enumerate(baseline_values):
            loss_rows.append(
                {"window": window, "candidate": "baseline", "repeat": repeat, "loss": loss}
            )
        for candidate, updates in candidate_updates.items():
            values = repeated_losses(
                model,
                batches,
                updates,
                repeats=repeats,
                device=args.device,
                dtype=torch.float32,
            )
            for repeat, loss in enumerate(values):
                loss_rows.append(
                    {"window": window, "candidate": candidate, "repeat": repeat, "loss": loss}
                )
    aggregate = aggregate_results(
        loss_rows,
        windows=windows,
        candidate_specs=candidate_specs,
        fit_ranking=fit_ranking,
        numerical_range_tolerance=float(
            decision_rule["maximum_replicate_range"]
        ),
        minimum_recovery_every_window=float(
            decision_rule["minimum_subspace_recovery_every_window"]
        ),
        minimum_median_recovery=float(
            decision_rule["minimum_subspace_median_recovery"]
        ),
        top3_concentration_threshold=float(
            decision_rule["top3_minimum_recovery_for_concentration"]
        ),
        top6_concentration_threshold=float(
            decision_rule["top6_minimum_recovery_for_concentration"]
        ),
    )
    aggregate["fit_gradient_loss_bfloat16"] = fit_gradient_loss
    aggregate["parameter_updates"] = 0

    args.output.mkdir(parents=True, exist_ok=True)
    fit_path = args.output / "cfc_layer_local_fit_ranking.csv"
    losses_path = args.output / "cfc_layer_local_losses.csv"
    metrics_path = args.output / "cfc_layer_local_weight_metrics.csv"
    spectra_path = args.output / "cfc_layer_local_weight_spectra.json"
    selections_path = args.output / "cfc_layer_local_selections.json"
    aggregate_path = args.output / "cfc_layer_local_aggregate.json"
    write_csv(fit_path, fit_rows)
    write_csv(losses_path, loss_rows)
    write_csv(metrics_path, weight_rows)
    spectra_path.write_text(
        json.dumps(weight_spectra, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    selections_path.write_text(
        json.dumps(selection_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "decision": aggregate["decision"],
        "selected_candidate": aggregate["selected_candidate"],
        "parameter_updates": 0,
        "checkpoint_next_iter": int(checkpoint["next_iter"]),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "config_sha256": file_sha256(args.config),
        "dataset_manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
        "plan_sha256": file_sha256(args.plan),
        "analysis_execution": {
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": str(Path(__file__).resolve()),
            "entrypoint_sha256": file_sha256(Path(__file__).resolve()),
            "command": sys.argv,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "device": args.device,
        },
        "protocol": protocol,
        "outputs": {
            "fit_sha256": file_sha256(fit_path),
            "losses_sha256": file_sha256(losses_path),
            "metrics_sha256": file_sha256(metrics_path),
            "spectra_sha256": file_sha256(spectra_path),
            "selections_sha256": file_sha256(selections_path),
            "aggregate_sha256": file_sha256(aggregate_path),
        },
        "limitations": plan["limitations"],
    }
    metadata_path = args.output / "cfc_layer_local_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": aggregate["decision"],
                "selected_candidate": aggregate["selected_candidate"],
                "fit_layer_order": selected,
                "aggregate": str(aggregate_path),
                "metadata": str(metadata_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
