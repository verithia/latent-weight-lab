#!/usr/bin/env python3
"""Decompose the residual after a stateless-fresh hidden64 c_proj chart.

At each preregistered dense-Muon endpoint this diagnostic first constructs
the production-qualified, current-direction hidden-side stage-64 update.
Additional hidden-side rotations, output-side rotations, and two-sided
diagonal tangents are then fitted only to the remaining exact current Muon
update.  Equal-coordinate hidden-only controls distinguish genuinely new
structure from merely adding coordinates.

The diagnostic performs no model or optimizer update.  Future Muon probes
and fixed validation windows are scoring-only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.nanogpt.analyze_mlp_activation_update_alignment import (
    load_snapshot,
    model_from_snapshot,
)
from examples.nanogpt.analyze_mlp_muon_chart_staleness import (
    file_sha256,
    git_commit,
    write_csv,
)
from examples.nanogpt.analyze_mlp_muon_matched_functional_metric import (
    evaluate_and_collect,
    evaluate_with_updates,
    fixed_scale_recovery,
    output_space_metrics,
    task_descent_metrics,
)
from examples.nanogpt.analyze_mlp_muon_matched_givens import (
    diagonal_metric_causal_givens_update,
)
from examples.nanogpt.analyze_mlp_optimizer_state_direction import (
    reconstruct_directions,
)
from examples.nanogpt.analyze_mlp_task_gradient_direction import (
    collect_cproj_gradients,
    direction_metrics,
)
from examples.nanogpt.analyze_parameter_trajectory import parse_int_list
from examples.nanogpt.analyze_residual_compatibility import (
    fixed_validation_batches,
)
from examples.nanogpt.fast_task_matching import (
    build_task_edge_coloring,
    fast_muon_matched_permutations,
)
from examples.nanogpt.parameter_trajectory import (
    OPTIMIZER_PROBE_SCHEMA_VERSION,
)


WINDOWS = ("fit", "holdout")
CANDIDATES = (
    "dense_exact",
    "fresh_hidden64",
    "fresh_hidden72",
    "fresh_hidden80",
    "fresh_hidden64_plus_residual_output32",
    "fresh_hidden64_plus_residual_output64",
    "fresh_hidden64_plus_left_diagonal",
    "fresh_hidden64_plus_right_diagonal",
    "fresh_hidden64_plus_two_sided_diagonal",
)
COORDINATES = {
    "dense_exact": 768 * 3072,
    "fresh_hidden64": 64 * (3072 // 2),
    "fresh_hidden72": 72 * (3072 // 2),
    "fresh_hidden80": 80 * (3072 // 2),
    "fresh_hidden64_plus_residual_output32": (
        64 * (3072 // 2) + 32 * (768 // 2)
    ),
    "fresh_hidden64_plus_residual_output64": (
        64 * (3072 // 2) + 64 * (768 // 2)
    ),
    "fresh_hidden64_plus_left_diagonal": (
        64 * (3072 // 2) + 768
    ),
    "fresh_hidden64_plus_right_diagonal": (
        64 * (3072 // 2) + 3072
    ),
    "fresh_hidden64_plus_two_sided_diagonal": (
        64 * (3072 // 2) + 768 + 3072
    ),
}


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / max(denominator, 1e-30)


def weighted(
    rows: list[dict[str, Any]],
    value: str,
    energy: str,
) -> float:
    weights = torch.tensor(
        [float(row[energy]) for row in rows],
        dtype=torch.float64,
    )
    values = torch.tensor(
        [float(row[value]) for row in rows],
        dtype=torch.float64,
    )
    return float((weights * values).sum() / weights.sum().clamp_min(1e-30))


def fit_right_diagonal(
    source: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit ``source @ diag(scale)`` independently by hidden channel."""
    source = source.float()
    residual = residual.float()
    scale = (
        (source * residual).sum(dim=0)
        / source.square().sum(dim=0).clamp_min(1e-30)
    )
    return source * scale.unsqueeze(0), scale


def fit_left_diagonal(
    source: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit ``diag(scale) @ source`` independently by output channel."""
    source = source.float()
    residual = residual.float()
    scale = (
        (source * residual).sum(dim=1)
        / source.square().sum(dim=1).clamp_min(1e-30)
    )
    return scale.unsqueeze(1) * source, scale


def fit_two_sided_diagonal(
    source: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Fit one right-then-left diagonal Gauss-Seidel residual sweep."""
    right, right_scale = fit_right_diagonal(source, residual)
    after_right = source.float() + right
    left, left_scale = fit_left_diagonal(
        after_right, residual.float() - right
    )
    return right + left, {
        "right_scale_rms": float(right_scale.square().mean().sqrt()),
        "left_scale_rms": float(left_scale.square().mean().sqrt()),
    }


def load_probe(path: Path, expected_identity: str) -> dict[str, Any]:
    probe = torch.load(path, map_location="cpu", weights_only=False)
    if probe.get("schema_version") != OPTIMIZER_PROBE_SCHEMA_VERSION:
        raise ValueError(f"unexpected optimizer probe schema: {path}")
    if str(probe.get("run_identity_sha256")) != expected_identity:
        raise ValueError(f"optimizer probe identity mismatch: {path}")
    return probe


def requested_update(
    probe: dict[str, Any],
    parameter: str,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    state = {
        name: tensor.to(device)
        for name, tensor in probe["parameters"][parameter].items()
    }
    hyperparameters = probe["hyperparameters"][parameter]
    directions = reconstruct_directions(state, hyperparameters)
    update = (
        float(hyperparameters["lr"])
        * directions["exact_applied_direction"]
    )
    return update, directions["exact_applied_direction"], state


def build_candidates(
    source: torch.Tensor,
    dense_update: torch.Tensor,
    dense_direction: torch.Tensor,
    *,
    neighbors: int,
    matching_seed: int,
    native_cache: Path | None,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    """Build every preregistered residual-decomposition candidate."""
    hidden64_permutations, hidden64_diagnostics = (
        fast_muon_matched_permutations(
            source,
            dense_direction,
            stages=64,
            neighbors=neighbors,
            seed=matching_seed,
            cache_dir=native_cache,
        )
    )
    hidden64, _hidden64_fit = diagonal_metric_causal_givens_update(
        source,
        dense_update,
        stages=64,
        seed=matching_seed,
        permutations=hidden64_permutations,
    )
    after_hidden64 = source.float() + hidden64
    residual = dense_update.float() - hidden64

    hidden_extra_permutations, hidden_extra_diagnostics = (
        fast_muon_matched_permutations(
            after_hidden64,
            residual,
            stages=16,
            neighbors=neighbors,
            seed=matching_seed + 1,
            cache_dir=native_cache,
        )
    )
    hidden_extra8, _ = diagonal_metric_causal_givens_update(
        after_hidden64,
        residual,
        stages=8,
        seed=matching_seed + 1,
        permutations=hidden_extra_permutations[:8],
    )
    hidden_extra16, _ = diagonal_metric_causal_givens_update(
        after_hidden64,
        residual,
        stages=16,
        seed=matching_seed + 1,
        permutations=hidden_extra_permutations,
    )

    output_permutations, output_diagnostics = (
        fast_muon_matched_permutations(
            after_hidden64.T.contiguous(),
            residual.T.contiguous(),
            stages=64,
            neighbors=neighbors,
            seed=matching_seed + 2,
            cache_dir=native_cache,
        )
    )
    output32_t, _ = diagonal_metric_causal_givens_update(
        after_hidden64.T.contiguous(),
        residual.T.contiguous(),
        stages=32,
        seed=matching_seed + 2,
        permutations=output_permutations[:32],
    )
    output64_t, _ = diagonal_metric_causal_givens_update(
        after_hidden64.T.contiguous(),
        residual.T.contiguous(),
        stages=64,
        seed=matching_seed + 2,
        permutations=output_permutations,
    )

    left_diagonal, left_scale = fit_left_diagonal(
        after_hidden64, residual
    )
    right_diagonal, right_scale = fit_right_diagonal(
        after_hidden64, residual
    )
    two_sided_diagonal, two_sided_diagnostics = fit_two_sided_diagonal(
        after_hidden64, residual
    )
    candidates = {
        "dense_exact": dense_update.float(),
        "fresh_hidden64": hidden64,
        "fresh_hidden72": hidden64 + hidden_extra8,
        "fresh_hidden80": hidden64 + hidden_extra16,
        "fresh_hidden64_plus_residual_output32": (
            hidden64 + output32_t.T.contiguous()
        ),
        "fresh_hidden64_plus_residual_output64": (
            hidden64 + output64_t.T.contiguous()
        ),
        "fresh_hidden64_plus_left_diagonal": (
            hidden64 + left_diagonal
        ),
        "fresh_hidden64_plus_right_diagonal": (
            hidden64 + right_diagonal
        ),
        "fresh_hidden64_plus_two_sided_diagonal": (
            hidden64 + two_sided_diagonal
        ),
    }
    diagnostics = [
        {
            "selection": "hidden64",
            **hidden64_diagnostics,
        },
        {
            "selection": "hidden_residual16",
            **hidden_extra_diagnostics,
        },
        {
            "selection": "output_residual64",
            **output_diagnostics,
        },
        {
            "selection": "left_diagonal",
            "coordinate_rms": float(
                left_scale.square().mean().sqrt()
            ),
        },
        {
            "selection": "right_diagonal",
            "coordinate_rms": float(
                right_scale.square().mean().sqrt()
            ),
        },
        {
            "selection": "two_sided_diagonal",
            **two_sided_diagnostics,
        },
    ]
    return candidates, diagnostics


def finite_comparison(
    finite_rows: list[dict[str, Any]],
    candidate: str,
    control: str,
) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    for base_update in sorted(
        {int(row["base_update"]) for row in finite_rows}
    ):
        for window in WINDOWS:
            losses = {
                str(row["candidate"]): float(row["loss"])
                for row in finite_rows
                if int(row["base_update"]) == base_update
                and row["window"] == window
            }
            if candidate not in losses or control not in losses:
                raise ValueError("finite CE comparison is incomplete")
            delta = losses[control] - losses[candidate]
            comparisons.append(
                {
                    "base_update": base_update,
                    "window": window,
                    "control_minus_candidate_loss": delta,
                    "candidate_wins": delta > 0.0,
                }
            )
    return {
        "wins": sum(row["candidate_wins"] for row in comparisons),
        "comparisons": len(comparisons),
        "details": comparisons,
    }


def aggregate_results(
    rows: list[dict[str, Any]],
    finite_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for candidate in CANDIDATES:
        selected = [row for row in rows if row["candidate"] == candidate]
        fit = [row for row in selected if row["window"] == "fit"]
        metrics[candidate] = {
            "coordinates_per_layer": COORDINATES[candidate],
            "cells_times_windows": len(selected),
            "current_weight_recovery": weighted(
                fit,
                "current_weight_recovery",
                "current_weight_energy",
            ),
            "future_residual_positive_line_recovery": weighted(
                fit,
                "future_residual_positive_line_recovery",
                "future_residual_energy",
            ),
            "current_residual_fixed_scale_recovery": weighted(
                fit,
                "current_residual_fixed_scale_recovery",
                "current_residual_energy",
            ),
            "current_output_positive_line_recovery": weighted(
                selected,
                "current_output_positive_line_recovery",
                "current_output_energy",
            ),
            "current_output_fixed_scale_recovery": weighted(
                selected,
                "current_output_fixed_scale_recovery",
                "current_output_energy",
            ),
            "train_gradient_predicted_ce_decrease": sum(
                float(row["train_gradient_predicted_ce_decrease"])
                for row in fit
            ),
            "validation_gradient_predicted_ce_decrease": {
                window: sum(
                    float(
                        row[
                            "validation_gradient_predicted_ce_decrease"
                        ]
                    )
                    for row in selected
                    if row["window"] == window
                )
                for window in WINDOWS
            },
            "minimum_train_gradient_predicted_ce_decrease": min(
                float(row["train_gradient_predicted_ce_decrease"])
                for row in fit
            ),
        }

    comparisons = {
        "output32_vs_hidden72": (
            "fresh_hidden64_plus_residual_output32",
            "fresh_hidden72",
        ),
        "output64_vs_hidden80": (
            "fresh_hidden64_plus_residual_output64",
            "fresh_hidden80",
        ),
        "two_sided_diagonal_vs_hidden72": (
            "fresh_hidden64_plus_two_sided_diagonal",
            "fresh_hidden72",
        ),
        "hidden80_vs_hidden64": (
            "fresh_hidden80",
            "fresh_hidden64",
        ),
    }
    ratios: dict[str, Any] = {}
    for name, (candidate, control) in comparisons.items():
        candidate_metrics = metrics[candidate]
        control_metrics = metrics[control]
        ratios[name] = {
            "candidate": candidate,
            "control": control,
            "current_output_positive_line_ratio": safe_ratio(
                candidate_metrics[
                    "current_output_positive_line_recovery"
                ],
                control_metrics[
                    "current_output_positive_line_recovery"
                ],
            ),
            "future_residual_recovery_ratio": safe_ratio(
                candidate_metrics[
                    "future_residual_positive_line_recovery"
                ],
                control_metrics[
                    "future_residual_positive_line_recovery"
                ],
            ),
            "validation_gradient_ce_decrease_ratio": {
                window: safe_ratio(
                    candidate_metrics[
                        "validation_gradient_predicted_ce_decrease"
                    ][window],
                    control_metrics[
                        "validation_gradient_predicted_ce_decrease"
                    ][window],
                )
                for window in WINDOWS
            },
            "minimum_train_gradient_predicted_ce_decrease": (
                candidate_metrics[
                    "minimum_train_gradient_predicted_ce_decrease"
                ]
            ),
            "finite_ce": finite_comparison(
                finite_rows, candidate, control
            ),
        }

    def output_pass(record: dict[str, Any]) -> bool:
        return (
            record["current_output_positive_line_ratio"] >= 1.05
            and record["future_residual_recovery_ratio"] >= 1.10
            and min(
                record[
                    "validation_gradient_ce_decrease_ratio"
                ].values()
            )
            >= 1.05
            and record["finite_ce"]["wins"] >= 7
            and record[
                "minimum_train_gradient_predicted_ce_decrease"
            ]
            > 0.0
        )

    def diagonal_pass(record: dict[str, Any]) -> bool:
        return (
            record["current_output_positive_line_ratio"] >= 1.05
            and min(
                record[
                    "validation_gradient_ce_decrease_ratio"
                ].values()
            )
            >= 1.05
            and record["finite_ce"]["wins"] >= 7
            and record[
                "minimum_train_gradient_predicted_ce_decrease"
            ]
            > 0.0
        )

    output_passes = [
        name
        for name in ("output32_vs_hidden72", "output64_vs_hidden80")
        if output_pass(ratios[name])
    ]
    diagonal_passed = diagonal_pass(
        ratios["two_sided_diagonal_vs_hidden72"]
    )
    hidden = ratios["hidden80_vs_hidden64"]
    hidden_signal = (
        hidden["current_output_positive_line_ratio"] >= 1.15
        and min(
            hidden["validation_gradient_ce_decrease_ratio"].values()
        )
        >= 1.15
        and hidden["finite_ce"]["wins"] >= 7
        and hidden[
            "minimum_train_gradient_predicted_ce_decrease"
        ]
        > 0.0
    )

    passed_branches = list(output_passes)
    if diagonal_passed:
        passed_branches.append("two_sided_diagonal_vs_hidden72")
    if passed_branches:
        def conservative_score(name: str) -> float:
            record = ratios[name]
            values = [
                record["current_output_positive_line_ratio"],
                min(
                    record[
                        "validation_gradient_ce_decrease_ratio"
                    ].values()
                ),
                record["finite_ce"]["wins"]
                / record["finite_ce"]["comparisons"],
            ]
            if name.startswith("output"):
                values.append(record["future_residual_recovery_ratio"])
            return min(values)

        selected = min(
            passed_branches,
            key=lambda name: (
                -conservative_score(name),
                COORDINATES[ratios[name]["candidate"]],
            ),
        )
        decision = "SELECT_RESIDUAL_STRUCTURE_FOR_IMPLEMENTATION_PREFLIGHT"
    elif hidden_signal:
        selected = None
        decision = "HIDDEN_CAPACITY_SIGNAL_NO_AUTOMATIC_PROMOTION"
    else:
        selected = None
        decision = "REJECT_SPARSE_ORTHOGONAL_PLUS_DIAGONAL_EXPANSION"
    return {
        "candidate_metrics": metrics,
        "comparisons": ratios,
        "branch_passes": {
            "output": output_passes,
            "two_sided_diagonal": diagonal_passed,
            "hidden_capacity_signal": hidden_signal,
        },
        "selected_branch": selected,
        "decision": decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--probe-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument(
        "--expected-run-identity",
        default=(
            "a801a6e24a071abedaa120b4161065118ae8c85e24b22e810c8666f"
            "1631aecdf"
        ),
    )
    parser.add_argument("--layers", default="0,3,6,9,11")
    parser.add_argument("--base-updates", default="0,60,120,180")
    parser.add_argument("--future-updates", default="15,75,135,195")
    parser.add_argument("--neighbors", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--fit-seed", type=int, default=20260802)
    parser.add_argument("--holdout-seed", type=int, default=20260803)
    parser.add_argument("--matching-seed", type=int, default=161803)
    parser.add_argument("--native-cache", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    layers = parse_int_list(args.layers)
    base_updates = parse_int_list(args.base_updates)
    future_updates = parse_int_list(args.future_updates)
    if (
        not layers
        or len(base_updates) != 4
        or len(future_updates) != 4
        or any(
            future <= base
            for base, future in zip(
                base_updates, future_updates, strict=True
            )
        )
        or args.neighbors < 64
    ):
        raise ValueError("arguments violate the preregistered protocol")
    snapshot_paths = [
        args.snapshot_dir / f"step_{step:06d}.pt"
        for step in base_updates
    ]
    probe_paths = [
        args.probe_dir / f"step_{step:06d}.pt"
        for step in (*base_updates, *future_updates)
    ]
    missing = [
        str(path)
        for path in (*snapshot_paths, *probe_paths, args.plan)
        if not path.is_file()
    ]
    if missing:
        raise ValueError(f"required registered inputs are absent: {missing}")
    _library, native_library_path = build_task_edge_coloring(
        args.native_cache
    )
    batches_by_window = {
        "fit": fixed_validation_batches(
            args.data_dir,
            args.batch_size,
            args.block_size + 1,
            args.batches,
            args.fit_seed,
        ),
        "holdout": fixed_validation_batches(
            args.data_dir,
            args.batch_size,
            args.block_size + 1,
            args.batches,
            args.holdout_seed,
        ),
    }

    rows: list[dict[str, Any]] = []
    finite_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for base_update, future_update, snapshot_path in zip(
        base_updates, future_updates, snapshot_paths, strict=True
    ):
        payload = load_snapshot(snapshot_path)
        model = model_from_snapshot(payload, args.device)
        base_probe = load_probe(
            args.probe_dir / f"step_{base_update:06d}.pt",
            args.expected_run_identity,
        )
        future_probe = load_probe(
            args.probe_dir / f"step_{future_update:06d}.pt",
            args.expected_run_identity,
        )
        candidates_by_layer: dict[str, dict[int, torch.Tensor]] = {
            candidate: {} for candidate in CANDIDATES
        }
        future_by_layer: dict[int, torch.Tensor] = {}
        train_gradients: dict[int, torch.Tensor] = {}
        for layer in layers:
            parameter = f"transformer.h.{layer}.mlp.c_proj.weight"
            dense_update, dense_direction, base_state = requested_update(
                base_probe, parameter, args.device
            )
            source = base_state["weight_before_step"]
            model_source = (
                model.transformer.h[layer].mlp.c_proj.weight.detach()
            )
            torch.testing.assert_close(
                source,
                model_source,
                rtol=0.0,
                atol=0.0,
            )
            future_dense_update, _future_direction, _future_state = (
                requested_update(
                    future_probe, parameter, args.device
                )
            )
            candidates, diagnostics = build_candidates(
                source,
                dense_update,
                dense_direction,
                neighbors=args.neighbors,
                matching_seed=(
                    args.matching_seed + 1009 * layer + base_update
                ),
                native_cache=args.native_cache,
            )
            for candidate, update in candidates.items():
                candidates_by_layer[candidate][layer] = update
            future_by_layer[layer] = future_dense_update
            train_gradients[layer] = base_state["gradient_after_clip"]
            for diagnostic in diagnostics:
                selection_rows.append(
                    {
                        "layer": layer,
                        "base_update": base_update,
                        "selection": diagnostic["selection"],
                        "diagnostics_json": json.dumps(
                            diagnostic, sort_keys=True
                        ),
                    }
                )

        baseline_loss: dict[str, float] = {}
        activations: dict[str, dict[int, torch.Tensor]] = {}
        validation_gradients: dict[str, dict[int, torch.Tensor]] = {}
        for window in WINDOWS:
            baseline_loss[window], activations[window] = (
                evaluate_and_collect(
                    model,
                    batches_by_window[window],
                    layers,
                    args.device,
                )
            )
            validation_gradients[window], _gradient_loss = (
                collect_cproj_gradients(
                    model,
                    batches_by_window[window],
                    layers,
                    args.device,
                )
            )
            finite_rows.append(
                {
                    "base_update": base_update,
                    "future_update": future_update,
                    "window": window,
                    "candidate": "baseline",
                    "loss": baseline_loss[window],
                    "loss_change_from_baseline": 0.0,
                }
            )
            for candidate in CANDIDATES:
                loss = evaluate_with_updates(
                    model,
                    batches_by_window[window],
                    candidates_by_layer[candidate],
                    args.device,
                )
                finite_rows.append(
                    {
                        "base_update": base_update,
                        "future_update": future_update,
                        "window": window,
                        "candidate": candidate,
                        "loss": loss,
                        "loss_change_from_baseline": (
                            loss - baseline_loss[window]
                        ),
                    }
                )

        for window in WINDOWS:
            for layer in layers:
                hidden = activations[window][layer].to(args.device)
                dense = candidates_by_layer["dense_exact"][layer]
                parent = candidates_by_layer["fresh_hidden64"][layer]
                current_residual = dense - parent
                future = future_by_layer[layer]
                future_residual = future - parent
                for candidate in CANDIDATES:
                    update = candidates_by_layer[candidate][layer]
                    incremental = update - parent
                    current_weight = direction_metrics(dense, update)
                    future_weight = direction_metrics(future, update)
                    future_residual_metrics = direction_metrics(
                        future_residual, incremental
                    )
                    current_output = output_space_metrics(
                        hidden, dense, update
                    )
                    train = task_descent_metrics(
                        train_gradients[layer], update
                    )
                    validation = task_descent_metrics(
                        validation_gradients[window][layer],
                        update.cpu(),
                    )
                    row = {
                        "layer": layer,
                        "base_update": base_update,
                        "future_update": future_update,
                        "window": window,
                        "candidate": candidate,
                        "coordinates_per_layer": COORDINATES[candidate],
                        "current_weight_recovery": fixed_scale_recovery(
                            dense, update
                        ),
                        "current_weight_cosine": current_weight["cosine"],
                        "current_weight_energy": (
                            current_weight["target_chord_fro"] ** 2
                        ),
                        "current_residual_fixed_scale_recovery": (
                            fixed_scale_recovery(
                                current_residual, incremental
                            )
                        ),
                        "current_residual_energy": float(
                            current_residual.double().square().sum()
                        ),
                        "future_weight_positive_line_recovery": (
                            future_weight[
                                "positive_step_line_recovery"
                            ]
                        ),
                        "future_weight_cosine": future_weight["cosine"],
                        "future_weight_energy": (
                            future_weight["target_chord_fro"] ** 2
                        ),
                        "future_residual_positive_line_recovery": (
                            future_residual_metrics[
                                "positive_step_line_recovery"
                            ]
                        ),
                        "future_residual_cosine": (
                            future_residual_metrics["cosine"]
                        ),
                        "future_residual_energy": (
                            future_residual_metrics[
                                "target_chord_fro"
                            ]
                            ** 2
                        ),
                        "current_output_positive_line_recovery": (
                            current_output[
                                "positive_step_line_recovery"
                            ]
                        ),
                        "current_output_fixed_scale_recovery": (
                            current_output["fixed_scale_recovery"]
                        ),
                        "current_output_cosine": current_output["cosine"],
                        "current_output_energy": current_output[
                            "target_output_energy"
                        ],
                        "train_gradient_predicted_ce_decrease": train[
                            "predicted_ce_decrease"
                        ],
                        "validation_gradient_predicted_ce_decrease": (
                            validation["predicted_ce_decrease"]
                        ),
                        "update_fro": train["update_fro"],
                    }
                    rows.append(row)
                    print(json.dumps(row, sort_keys=True), flush=True)
        del model, payload, base_probe, future_probe
        if "cuda" in args.device:
            torch.cuda.empty_cache()

    aggregate = aggregate_results(rows, finite_rows)
    args.output.mkdir(parents=True, exist_ok=True)
    detail_path = args.output / "fast_fresh_residual_decomposition.csv"
    finite_path = (
        args.output / "fast_fresh_residual_decomposition_finite_ce.csv"
    )
    selection_path = (
        args.output / "fast_fresh_residual_decomposition_selections.csv"
    )
    aggregate_path = (
        args.output / "fast_fresh_residual_decomposition_aggregate.json"
    )
    write_csv(detail_path, rows)
    write_csv(finite_path, finite_rows)
    write_csv(selection_path, selection_rows)
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": (
            "nanogpt_mlp_fast_fresh_residual_decomposition_v1"
        ),
        "decision": aggregate["decision"],
        "parameter_updates": 0,
        "learned_dense_basis": False,
        "dense_residual_adapter": False,
        "lora_adapter": False,
        "expected_run_identity": args.expected_run_identity,
        "layers": layers,
        "base_updates": base_updates,
        "future_updates": future_updates,
        "neighbors": args.neighbors,
        "validation_windows": {
            "fit_seed": args.fit_seed,
            "holdout_seed": args.holdout_seed,
            "batch_size": args.batch_size,
            "block_size": args.block_size,
            "batches": args.batches,
        },
        "plan": {
            "path": str(args.plan),
            "sha256": file_sha256(args.plan),
        },
        "analysis_execution": {
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": str(script),
            "entrypoint_sha256": file_sha256(script),
            "command": sys.argv,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "device": args.device,
        },
        "native_matcher": {
            "library": str(native_library_path),
            "library_sha256": file_sha256(native_library_path),
        },
        "inputs": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in (*snapshot_paths, *probe_paths)
        ],
        "outputs": {
            "detail_sha256": file_sha256(detail_path),
            "finite_ce_sha256": file_sha256(finite_path),
            "selections_sha256": file_sha256(selection_path),
            "aggregate_sha256": file_sha256(aggregate_path),
        },
        "limitations": [
            "This is a zero-update causal tangent diagnostic, not training.",
            "Finite-step CE perturbs five representative c_proj layers only.",
            "The dense Muon replay is one optimizer path, not the global low-loss manifold.",
            "Hidden72/80 use a fresh residual correction after production hidden64; pair reuse across the two sequential sweeps is allowed.",
        ],
    }
    metadata_path = (
        args.output / "fast_fresh_residual_decomposition_metadata.json"
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": aggregate["decision"],
                "selected_branch": aggregate["selected_branch"],
                "metadata": str(metadata_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
