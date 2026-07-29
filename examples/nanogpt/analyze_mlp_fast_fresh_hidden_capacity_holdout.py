#!/usr/bin/env python3
"""Held-out gate for two-pass fresh hidden c_proj capacity.

This zero-update diagnostic reuses the common functional evaluator on base
updates, future updates, layers, and validation windows that were not scored
by the earlier fresh-chart decisions. It fits the qualified stateless
hidden64 chart, then selects a second current-residual topology with 24 or 48
stages to form hidden88 and hidden112. Future state is scoring-only.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt import (
    analyze_mlp_fast_fresh_residual_decomposition as runner,
)
from examples.nanogpt.analyze_mlp_muon_matched_givens import (
    diagonal_metric_causal_givens_update,
)
from examples.nanogpt.fast_task_matching import (
    fast_muon_matched_permutations,
)


WINDOWS = runner.WINDOWS
CANDIDATES = (
    "dense_exact",
    "fresh_hidden64",
    "fresh_hidden88",
    "fresh_hidden112",
)
COORDINATES = {
    "dense_exact": 768 * 3072,
    "fresh_hidden64": 64 * (3072 // 2),
    "fresh_hidden88": 88 * (3072 // 2),
    "fresh_hidden112": 112 * (3072 // 2),
}


def build_candidates(
    source: torch.Tensor,
    dense_update: torch.Tensor,
    dense_direction: torch.Tensor,
    *,
    neighbors: int,
    matching_seed: int,
    native_cache: Path | None,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    """Fit hidden64 and its fresh residual 24/48-stage continuations."""
    parent_permutations, parent_selection = (
        fast_muon_matched_permutations(
            source,
            dense_direction,
            stages=64,
            neighbors=neighbors,
            seed=matching_seed,
            cache_dir=native_cache,
        )
    )
    parent, parent_fit = diagonal_metric_causal_givens_update(
        source,
        dense_update,
        stages=64,
        seed=matching_seed,
        permutations=parent_permutations,
    )
    after_parent = source.float() + parent
    residual = dense_update.float() - parent
    residual_permutations, residual_selection = (
        fast_muon_matched_permutations(
            after_parent,
            residual,
            stages=48,
            neighbors=neighbors,
            seed=matching_seed + 1,
            cache_dir=native_cache,
        )
    )
    residual24, fit24 = diagonal_metric_causal_givens_update(
        after_parent,
        residual,
        stages=24,
        seed=matching_seed + 1,
        permutations=residual_permutations[:24],
    )
    residual48, fit48 = diagonal_metric_causal_givens_update(
        after_parent,
        residual,
        stages=48,
        seed=matching_seed + 1,
        permutations=residual_permutations,
    )
    return {
        "dense_exact": dense_update.float(),
        "fresh_hidden64": parent,
        "fresh_hidden88": parent + residual24,
        "fresh_hidden112": parent + residual48,
    }, [
        {
            "selection": "fresh_hidden64",
            **parent_selection,
            "fit": parent_fit,
        },
        {
            "selection": "fresh_hidden_residual48",
            **residual_selection,
            "fit24": fit24,
            "fit48": fit48,
        },
    ]


def _weighted_metrics(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    fit = [row for row in rows if row["window"] == "fit"]
    numeric = [
        value
        for row in rows
        for value in row.values()
        if isinstance(value, (int, float))
    ]
    return {
        "coordinates_per_layer": int(
            rows[0]["coordinates_per_layer"]
        ),
        "cells_times_windows": len(rows),
        "all_metrics_finite": all(
            math.isfinite(float(value)) for value in numeric
        ),
        "current_weight_recovery": runner.weighted(
            fit, "current_weight_recovery", "current_weight_energy"
        ),
        "current_residual_fixed_scale_recovery": runner.weighted(
            fit,
            "current_residual_fixed_scale_recovery",
            "current_residual_energy",
        ),
        "current_output_positive_line_recovery": runner.weighted(
            rows,
            "current_output_positive_line_recovery",
            "current_output_energy",
        ),
        "current_output_fixed_scale_recovery": runner.weighted(
            rows,
            "current_output_fixed_scale_recovery",
            "current_output_energy",
        ),
        "future_residual_positive_line_recovery": runner.weighted(
            fit,
            "future_residual_positive_line_recovery",
            "future_residual_energy",
        ),
        "positive_future_cells": sum(
            float(row["future_residual_cosine"]) > 0.0
            for row in fit
        ),
        "future_cells": len(fit),
        "validation_gradient_predicted_ce_decrease": {
            window: sum(
                float(
                    row[
                        "validation_gradient_predicted_ce_decrease"
                    ]
                )
                for row in rows
                if row["window"] == window
            )
            for window in WINDOWS
        },
        "minimum_train_gradient_predicted_ce_decrease": min(
            float(row["train_gradient_predicted_ce_decrease"])
            for row in fit
        ),
    }


def aggregate_results(
    rows: list[dict[str, Any]],
    finite_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply the immutable held-out generalization rule."""
    metrics = {
        candidate: _weighted_metrics(
            [row for row in rows if row["candidate"] == candidate]
        )
        for candidate in CANDIDATES
    }
    parent = metrics["fresh_hidden64"]
    comparisons: dict[str, Any] = {}
    passing: list[int] = []
    for depth in (88, 112):
        name = f"fresh_hidden{depth}"
        candidate = metrics[name]
        record = {
            "candidate": name,
            "control": "fresh_hidden64",
            "current_output_positive_line_ratio": runner.safe_ratio(
                candidate[
                    "current_output_positive_line_recovery"
                ],
                parent["current_output_positive_line_recovery"],
            ),
            "validation_gradient_ce_decrease_ratio": {
                window: runner.safe_ratio(
                    candidate[
                        "validation_gradient_predicted_ce_decrease"
                    ][window],
                    parent[
                        "validation_gradient_predicted_ce_decrease"
                    ][window],
                )
                for window in WINDOWS
            },
            "future_residual_positive_line_recovery": candidate[
                "future_residual_positive_line_recovery"
            ],
            "positive_future_cells": candidate[
                "positive_future_cells"
            ],
            "future_cells": candidate["future_cells"],
            "minimum_train_gradient_predicted_ce_decrease": (
                candidate[
                    "minimum_train_gradient_predicted_ce_decrease"
                ]
            ),
            "all_metrics_finite": candidate["all_metrics_finite"],
            "finite_ce": runner.finite_comparison(
                finite_rows, name, "fresh_hidden64"
            ),
            "output_gain_per_added_coordinate": (
                candidate[
                    "current_output_positive_line_recovery"
                ]
                - parent["current_output_positive_line_recovery"]
            )
            / (COORDINATES[name] - COORDINATES["fresh_hidden64"]),
        }
        record["passed"] = (
            record["current_output_positive_line_ratio"] >= 1.15
            and min(
                record[
                    "validation_gradient_ce_decrease_ratio"
                ].values()
            )
            >= 1.15
            and record[
                "future_residual_positive_line_recovery"
            ]
            >= 0.015
            and record["positive_future_cells"] >= 17
            and record["future_cells"] == 20
            and record["finite_ce"]["wins"] >= 7
            and record[
                "minimum_train_gradient_predicted_ce_decrease"
            ]
            > 0.0
            and record["all_metrics_finite"]
        )
        comparisons[f"hidden{depth}_vs_hidden64"] = record
        if record["passed"]:
            passing.append(depth)

    selected_depth = (
        88 if 88 in passing else (112 if 112 in passing else None)
    )
    selected_branch = (
        f"fresh_hidden{selected_depth}"
        if selected_depth is not None
        else None
    )
    decision = (
        "SELECT_TWO_PASS_FRESH_HIDDEN_FOR_IMPLEMENTATION_PREFLIGHT"
        if selected_branch is not None
        else "REJECT_SPARSE_FRESH_HIDDEN_DEPTH"
    )
    return {
        "candidate_metrics": metrics,
        "comparisons": comparisons,
        "passing_depths": passing,
        "selected_branch": selected_branch,
        "decision": decision,
    }


def main() -> None:
    """Install the held-out candidate family in the shared runner."""
    runner.CANDIDATES = CANDIDATES
    runner.COORDINATES = COORDINATES
    runner.build_candidates = build_candidates
    runner.aggregate_results = aggregate_results
    runner.OUTPUT_STEM = "fast_fresh_hidden_capacity_holdout"
    runner.METADATA_SCHEMA_VERSION = (
        "nanogpt_mlp_fast_fresh_hidden_capacity_holdout_v1"
    )
    runner.METADATA_LIMITATIONS = [
        "This is a zero-update held-out causal diagnostic, not training.",
        "Finite-step CE perturbs seven held-out c_proj layers only.",
        "The dense Muon replay is one optimizer path, not the global low-loss manifold.",
        "Passing authorizes production implementation and an MFU preflight only.",
    ]
    runner.__file__ = __file__
    runner.__doc__ = __doc__
    runner.main()


if __name__ == "__main__":
    main()
