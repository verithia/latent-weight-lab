#!/usr/bin/env python3
"""Attribute the useful dense-minus-fresh88 c_fc residual without training."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from examples.nanogpt.analyze_mlp_cfc_trust_radius import (
    collect_gradient_window,
    repeated_losses,
    summarize,
)


SCHEMA_VERSION = "nanogpt_mlp_cfc_residual_structure_v1"
FAMILY_PRIORITY = {
    "input_diagonal": 0,
    "expansion_diagonal": 1,
    "bilateral_diagonal": 2,
    "low_rank_spectral": 3,
}


def git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fieldnames = list(rows[0])
    known = set(fieldnames)
    for row in rows[1:]:
        for field in row:
            if field not in known:
                fieldnames.append(field)
                known.add(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fit_input_diagonal(
    weight: torch.Tensor, residual: torch.Tensor
) -> torch.Tensor:
    """Fit W diag(a): one scale per 768-dimensional input channel."""
    numerator = (weight.double() * residual.double()).sum(dim=0)
    denominator = weight.double().square().sum(dim=0).clamp_min(1e-30)
    return (weight.double() * (numerator / denominator)).float()


def fit_expansion_diagonal(
    weight: torch.Tensor, residual: torch.Tensor
) -> torch.Tensor:
    """Fit diag(b) W: one scale per 3,072 expansion channel."""
    numerator = (weight.double() * residual.double()).sum(dim=1)
    denominator = weight.double().square().sum(dim=1).clamp_min(1e-30)
    return (weight.double() * (numerator / denominator)[:, None]).float()


def fit_bilateral_diagonal(
    weight: torch.Tensor,
    residual: torch.Tensor,
    *,
    iterations: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Fit diag(b)W + Wdiag(a) by deterministic alternating least squares."""
    if iterations < 2:
        raise ValueError("bilateral fit needs at least two iterations")
    w = weight.double()
    r = residual.double()
    a = torch.zeros(w.shape[1], dtype=torch.float64, device=w.device)
    b = torch.zeros(w.shape[0], dtype=torch.float64, device=w.device)
    col_den = w.square().sum(dim=0).clamp_min(1e-30)
    row_den = w.square().sum(dim=1).clamp_min(1e-30)
    for _ in range(int(iterations)):
        a = (w * (r - b[:, None] * w)).sum(dim=0) / col_den
        b = (w * (r - w * a[None, :])).sum(dim=1) / row_den
        # Fix the additive gauge a+=c, b-=c without changing the fit.
        gauge = a.mean()
        a = a - gauge
        b = b + gauge
    fitted = w * (a[None, :] + b[:, None])
    return fitted.float(), {
        "input_scale_rms": float(a.square().mean().sqrt()),
        "expansion_scale_rms": float(b.square().mean().sqrt()),
        "input_scale_max_abs": float(a.abs().max()),
        "expansion_scale_max_abs": float(b.abs().max()),
    }


def fit_low_rank(
    residual: torch.Tensor, ranks: list[int]
) -> tuple[dict[int, torch.Tensor], list[float]]:
    if not ranks or min(ranks) < 1 or max(ranks) > min(residual.shape):
        raise ValueError("invalid low-rank bracket")
    u, s, vh = torch.linalg.svd(
        residual.float(), full_matrices=False
    )
    fitted = {
        rank: (u[:, :rank] * s[:rank]) @ vh[:rank]
        for rank in ranks
    }
    return fitted, [float(value) for value in s[: max(ranks)]]


def residual_metrics(
    residual: torch.Tensor, approximation: torch.Tensor
) -> dict[str, float]:
    target = residual.double().reshape(-1)
    estimate = approximation.double().reshape(-1)
    energy = target.square().sum().clamp_min(1e-30)
    estimate_energy = estimate.square().sum().clamp_min(1e-30)
    dot = (target * estimate).sum()
    return {
        "residual_energy": float(energy),
        "approximation_energy": float(estimate_energy),
        "energy_recovery": float(
            1.0 - (target - estimate).square().sum() / energy
        ),
        "cosine": float(dot / (energy * estimate_energy).sqrt()),
        "positive_line_recovery": float(
            dot.clamp_min(0.0).square() / (energy * estimate_energy)
        ),
    }


def aggregate_losses(
    rows: list[dict[str, Any]],
    *,
    windows: list[str],
    candidates: dict[str, dict[str, Any]],
    numerical_range_tolerance: float,
    minimum_gap_recovery: float,
    median_gap_recovery: float,
) -> dict[str, Any]:
    summaries: dict[str, dict[str, dict[str, float]]] = {}
    for window in windows:
        summaries[window] = {}
        for candidate in ("baseline", *candidates):
            values = [
                float(row["loss"])
                for row in rows
                if row["window"] == window
                and row["candidate"] == candidate
            ]
            summaries[window][candidate] = summarize(values)
    stable = all(
        value["range"] <= float(numerical_range_tolerance)
        for window in summaries.values()
        for value in window.values()
    )
    dense_positive_control = all(
        values["dense_exact"]["maximum"]
        < values["fresh88"]["minimum"]
        for values in summaries.values()
    )
    recoveries: dict[str, dict[str, float]] = {}
    qualifications: dict[str, Any] = {}
    for candidate in candidates:
        if candidate in {"fresh88", "dense_exact"}:
            continue
        recoveries[candidate] = {}
        beats_fresh = True
        beats_baseline = True
        for window, values in summaries.items():
            baseline = values["baseline"]["mean"]
            fresh = values["fresh88"]["mean"]
            dense = values["dense_exact"]["mean"]
            loss = values[candidate]["mean"]
            dense_gap = max(fresh - dense, 1e-30)
            recoveries[candidate][window] = (fresh - loss) / dense_gap
            beats_fresh = beats_fresh and (
                values[candidate]["maximum"]
                < values["fresh88"]["minimum"]
            )
            beats_baseline = beats_baseline and (
                values[candidate]["maximum"]
                < values["baseline"]["minimum"]
            )
        ordered = sorted(recoveries[candidate].values())
        median = 0.5 * (ordered[1] + ordered[2])
        minimum = min(ordered)
        qualifies = all(
            (
                beats_fresh,
                beats_baseline,
                dense_positive_control,
                minimum >= float(minimum_gap_recovery),
                median >= float(median_gap_recovery),
            )
        )
        qualifications[candidate] = {
            "family": candidates[candidate]["family"],
            "coordinates_per_layer": candidates[candidate][
                "coordinates_per_layer"
            ],
            "gap_recovery_by_window": recoveries[candidate],
            "minimum_gap_recovery": minimum,
            "median_gap_recovery": median,
            "beats_fresh_every_window": beats_fresh,
            "beats_baseline_every_window": beats_baseline,
            "dense_beats_fresh_every_window": dense_positive_control,
            "qualifies": qualifies,
        }
    qualified = [
        (name, value)
        for name, value in qualifications.items()
        if value["qualifies"]
    ]
    if not dense_positive_control:
        selected_name, selected = max(
            qualifications.items(),
            key=lambda item: (
                float(item[1]["minimum_gap_recovery"]),
                float(item[1]["median_gap_recovery"]),
                -int(item[1]["coordinates_per_layer"]),
            ),
        )
        decision = "DENSE_RESIDUAL_NOT_POSITIVE_CONTROL"
    elif qualified:
        selected_name, selected = min(
            qualified,
            key=lambda item: (
                int(item[1]["coordinates_per_layer"]),
                FAMILY_PRIORITY[str(item[1]["family"])],
                item[0],
            ),
        )
        decision = "ATTRIBUTE_CFC_RESIDUAL_STRUCTURE"
    else:
        selected_name, selected = max(
            qualifications.items(),
            key=lambda item: (
                float(item[1]["minimum_gap_recovery"]),
                float(item[1]["median_gap_recovery"]),
                -int(item[1]["coordinates_per_layer"]),
            ),
        )
        decision = "NO_REGISTERED_RESIDUAL_FAMILY_SUFFICIENT"
    return {
        "decision": decision,
        "selected_candidate": selected_name,
        "selected_family": selected["family"],
        "selected_qualified": selected["qualifies"],
        "summaries": summaries,
        "qualifications": qualifications,
        "gates": {
            "numerically_stable": stable,
            "dense_beats_fresh_every_window": dense_positive_control,
        },
        "thresholds": {
            "numerical_range_tolerance": numerical_range_tolerance,
            "minimum_gap_recovery": minimum_gap_recovery,
            "median_gap_recovery": median_gap_recovery,
        },
    }


def validate_identity(
    checkpoint: Path,
    config: Path,
    data_dir: Path,
    plan: Path,
) -> dict[str, Any]:
    payload = json.loads(plan.read_text(encoding="utf-8"))
    expected = payload["identity"]
    actual = {
        "checkpoint_sha256": file_sha256(checkpoint),
        "config_sha256": file_sha256(config),
        "dataset_manifest_sha256": file_sha256(
            data_dir / "manifest.json"
        ),
    }
    for key, value in actual.items():
        if value != expected[key]:
            raise ValueError(f"registered identity mismatch: {key}")
    return payload


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
    layers = [int(value) for value in protocol["layers"]]
    ranks = [int(value) for value in protocol["low_rank_bracket"]]
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
    fit_loss, gradients = collect_gradient_window(
        model,
        fit_batches,
        layers,
        device=args.device,
        dtype=torch.bfloat16,
    )
    candidate_specs: dict[str, dict[str, Any]] = {
        "fresh88": {"family": "fresh88", "coordinates_per_layer": 135168},
        "dense_exact": {"family": "dense_exact", "coordinates_per_layer": 2359296},
        "fresh_plus_input_diag": {"family": "input_diagonal", "coordinates_per_layer": 768},
        "fresh_plus_expansion_diag": {"family": "expansion_diagonal", "coordinates_per_layer": 3072},
        "fresh_plus_bilateral_diag": {"family": "bilateral_diagonal", "coordinates_per_layer": 3840},
        **{
            f"fresh_plus_lowrank{rank}": {
                "family": "low_rank_spectral",
                "coordinates_per_layer": rank * (3072 + 768),
            }
            for rank in ranks
        },
    }
    updates: dict[str, dict[int, torch.Tensor]] = {
        candidate: {} for candidate in candidate_specs
    }
    structure_rows: list[dict[str, Any]] = []
    spectra: dict[str, list[float]] = {}
    selection_rows: list[dict[str, Any]] = []
    for layer in layers:
        weight = model.transformer.h[layer].mlp.c_fc.weight
        owner, group = _optimizer_and_group_for_parameter(optimizer, weight)
        buffer = owner.state[weight].get("momentum_buffer")
        if buffer is None:
            raise RuntimeError(f"missing c_fc momentum at layer {layer}")
        dense, descent, _diagnostics = exact_muon_update(
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
            dense,
            polar_descent,
            parent_stages=64,
            residual_stages=24,
            neighbors=int(protocol["matching_neighbors"]),
            seed=int(protocol["matching_seed"]) + layer * 1009,
            learning_rate=float(group["lr"]),
            weight_decay=float(group["weight_decay"]),
            native_cache=args.native_cache,
        )
        fresh = matched["fresh_expansion88"].float()
        residual = dense.float() - fresh
        input_diag = fit_input_diagonal(weight.detach().float(), residual)
        expansion_diag = fit_expansion_diagonal(
            weight.detach().float(), residual
        )
        bilateral, bilateral_stats = fit_bilateral_diagonal(
            weight.detach().float(),
            residual,
            iterations=int(protocol["bilateral_fit_iterations"]),
        )
        low_rank, singular_values = fit_low_rank(residual, ranks)
        spectra[str(layer)] = singular_values
        approximations = {
            "fresh_plus_input_diag": input_diag,
            "fresh_plus_expansion_diag": expansion_diag,
            "fresh_plus_bilateral_diag": bilateral,
            **{
                f"fresh_plus_lowrank{rank}": value
                for rank, value in low_rank.items()
            },
        }
        updates["fresh88"][layer] = fresh.cpu()
        updates["dense_exact"][layer] = dense.float().cpu()
        for candidate, approximation in approximations.items():
            updates[candidate][layer] = (fresh + approximation).cpu()
            metrics = residual_metrics(residual, approximation)
            structure_rows.append(
                {
                    "layer": layer,
                    "candidate": candidate,
                    "family": candidate_specs[candidate]["family"],
                    "coordinates_per_layer": candidate_specs[candidate][
                        "coordinates_per_layer"
                    ],
                    **metrics,
                }
            )
        structure_rows[-(len(ranks) + 1)]["input_scale_rms"] = bilateral_stats[
            "input_scale_rms"
        ]
        structure_rows[-(len(ranks) + 1)]["expansion_scale_rms"] = bilateral_stats[
            "expansion_scale_rms"
        ]
        selection_rows.extend(
            {"layer": layer, **selection} for selection in selections
        )

    repeats = int(protocol["evaluation_repeats"])
    loss_rows: list[dict[str, Any]] = []
    for window, batches in validation_batches.items():
        baseline = repeated_losses(
            model,
            batches,
            None,
            repeats=repeats,
            device=args.device,
            dtype=torch.float32,
        )
        for repeat, loss in enumerate(baseline):
            loss_rows.append(
                {"window": window, "candidate": "baseline", "repeat": repeat, "loss": loss}
            )
        for candidate, candidate_updates in updates.items():
            losses = repeated_losses(
                model,
                batches,
                candidate_updates,
                repeats=repeats,
                device=args.device,
                dtype=torch.float32,
            )
            for repeat, loss in enumerate(losses):
                loss_rows.append(
                    {"window": window, "candidate": candidate, "repeat": repeat, "loss": loss}
                )
    aggregate = aggregate_losses(
        loss_rows,
        windows=windows,
        candidates=candidate_specs,
        numerical_range_tolerance=float(
            plan["decision_rule"]["maximum_replicate_range"]
        ),
        minimum_gap_recovery=float(
            plan["decision_rule"]["minimum_gap_recovery_every_window"]
        ),
        median_gap_recovery=float(
            plan["decision_rule"]["minimum_median_gap_recovery"]
        ),
    )
    aggregate["fit_gradient_loss_bfloat16"] = fit_loss
    aggregate["parameter_updates"] = 0

    args.output.mkdir(parents=True, exist_ok=True)
    losses_path = args.output / "cfc_residual_structure_losses.csv"
    structure_path = args.output / "cfc_residual_structure_weight_metrics.csv"
    spectra_path = args.output / "cfc_residual_structure_spectra.json"
    selections_path = args.output / "cfc_residual_structure_selections.json"
    aggregate_path = args.output / "cfc_residual_structure_aggregate.json"
    write_csv(losses_path, loss_rows)
    write_csv(structure_path, structure_rows)
    spectra_path.write_text(
        json.dumps(spectra, indent=2, sort_keys=True) + "\n",
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
        "selected_family": aggregate["selected_family"],
        "parameter_updates": 0,
        "checkpoint_next_iter": int(checkpoint["next_iter"]),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "config_sha256": file_sha256(args.config),
        "dataset_manifest_sha256": file_sha256(
            args.data_dir / "manifest.json"
        ),
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
            "losses_sha256": file_sha256(losses_path),
            "structure_sha256": file_sha256(structure_path),
            "spectra_sha256": file_sha256(spectra_path),
            "selections_sha256": file_sha256(selections_path),
            "aggregate_sha256": file_sha256(aggregate_path),
        },
        "limitations": plan["limitations"],
    }
    metadata_path = args.output / "cfc_residual_structure_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": aggregate["decision"],
                "selected_candidate": aggregate["selected_candidate"],
                "selected_family": aggregate["selected_family"],
                "aggregate": str(aggregate_path),
                "metadata": str(metadata_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
