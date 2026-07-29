#!/usr/bin/env python3
"""Gate a fresh task-selected trace-free pair chart for MLP c_proj.

The qualified stateless hidden64 Givens direction is fitted first.  This
diagnostic then fits determinant-one 2x2 hidden-channel blocks to the exact
remaining current Muon direction.  Each block has rotation, symmetric shear,
and paired anisotropic-scale coordinates.  Equal-coordinate fresh Givens
controls and same-topology ablations distinguish useful non-orthogonal
structure from merely adding parameters or selecting better edges.

The shared residual-diagnostic runner performs no parameter or optimizer
update. Future Muon probes and fixed validation windows are scoring-only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import torch

from examples.nanogpt import (
    analyze_mlp_fast_fresh_residual_decomposition as runner,
)
from examples.nanogpt.analyze_mlp_muon_matched_givens import (
    diagonal_metric_causal_givens_update,
)
from examples.nanogpt.fast_task_matching import (
    color_sorted_edges,
    fast_muon_matched_permutations,
)


Component = Literal["full", "skew", "nonorthogonal"]
WINDOWS = runner.WINDOWS
CANDIDATES = (
    "dense_exact",
    "fresh_hidden64",
    "fresh_hidden88",
    "fresh_hidden112",
    "fresh_hidden64_plus_sl2_8",
    "fresh_hidden64_plus_sl2_16",
    "fresh_hidden64_plus_sl2_8_skew_only",
    "fresh_hidden64_plus_sl2_16_skew_only",
    "fresh_hidden64_plus_sl2_8_nonorthogonal_only",
    "fresh_hidden64_plus_sl2_16_nonorthogonal_only",
)
COORDINATES = {
    "dense_exact": 768 * 3072,
    "fresh_hidden64": 64 * (3072 // 2),
    "fresh_hidden88": 88 * (3072 // 2),
    "fresh_hidden112": 112 * (3072 // 2),
    "fresh_hidden64_plus_sl2_8": (
        64 * (3072 // 2) + 8 * (3072 // 2) * 3
    ),
    "fresh_hidden64_plus_sl2_16": (
        64 * (3072 // 2) + 16 * (3072 // 2) * 3
    ),
    "fresh_hidden64_plus_sl2_8_skew_only": (
        64 * (3072 // 2) + 8 * (3072 // 2)
    ),
    "fresh_hidden64_plus_sl2_16_skew_only": (
        64 * (3072 // 2) + 16 * (3072 // 2)
    ),
    "fresh_hidden64_plus_sl2_8_nonorthogonal_only": (
        64 * (3072 // 2) + 8 * (3072 // 2) * 2
    ),
    "fresh_hidden64_plus_sl2_16_nonorthogonal_only": (
        64 * (3072 // 2) + 16 * (3072 // 2) * 2
    ),
}


def _pair_normal_equations(
    source: torch.Tensor,
    residual: torch.Tensor,
    pairs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact normal equations for D/S/A trace-free generators."""
    left, right = pairs.unbind(dim=1)
    u = source[:, left].double()
    v = source[:, right].double()
    residual_left = residual[:, left].double()
    residual_right = residual[:, right].double()
    norm_u = u.square().sum(dim=0)
    norm_v = v.square().sum(dim=0)
    cross = (u * v).sum(dim=0)
    total_norm = norm_u + norm_v
    q_d = (
        (u * residual_left).sum(dim=0)
        - (v * residual_right).sum(dim=0)
    )
    q_s = (
        (v * residual_left).sum(dim=0)
        + (u * residual_right).sum(dim=0)
    )
    q_a = (
        (v * residual_left).sum(dim=0)
        - (u * residual_right).sum(dim=0)
    )
    normal = torch.zeros(
        (pairs.shape[0], 3, 3),
        dtype=torch.float64,
        device=source.device,
    )
    normal[:, 0, 0] = total_norm
    normal[:, 1, 1] = total_norm
    normal[:, 2, 2] = total_norm
    normal[:, 0, 2] = 2.0 * cross
    normal[:, 2, 0] = 2.0 * cross
    normal[:, 1, 2] = norm_v - norm_u
    normal[:, 2, 1] = norm_v - norm_u
    rhs = torch.stack((q_d, q_s, q_a), dim=1)
    return normal, rhs


def fit_pair_coordinates(
    source: torch.Tensor,
    residual: torch.Tensor,
    pairs: torch.Tensor,
    component: Component,
) -> torch.Tensor:
    """Fit D/S/A coordinates for one disjoint pair stage."""
    normal, rhs = _pair_normal_equations(source, residual, pairs)
    scale = normal.diagonal(dim1=1, dim2=2).mean(dim=1)
    if component == "full":
        ridge = (
            scale.clamp_min(1e-30) * 1e-10
        ).reshape(-1, 1, 1)
        coordinates = torch.linalg.solve(
            normal
            + ridge
            * torch.eye(
                3, dtype=normal.dtype, device=normal.device
            ).unsqueeze(0),
            rhs.unsqueeze(-1),
        ).squeeze(-1)
    elif component == "skew":
        coordinates = torch.zeros_like(rhs)
        coordinates[:, 2] = (
            rhs[:, 2] / scale.clamp_min(1e-30)
        )
    elif component == "nonorthogonal":
        coordinates = torch.zeros_like(rhs)
        coordinates[:, :2] = (
            rhs[:, :2] / scale.clamp_min(1e-30).unsqueeze(1)
        )
    else:
        raise ValueError(f"unknown component family: {component}")
    return coordinates


def coordinates_to_generators(
    coordinates: torch.Tensor,
) -> torch.Tensor:
    """Map [d, s, a] coordinates to trace-free 2x2 generators."""
    d, s, a = coordinates.unbind(dim=1)
    generators = torch.empty(
        (coordinates.shape[0], 2, 2),
        dtype=coordinates.dtype,
        device=coordinates.device,
    )
    generators[:, 0, 0] = d
    generators[:, 0, 1] = s - a
    generators[:, 1, 0] = s + a
    generators[:, 1, 1] = -d
    return generators


def apply_pair_stage(
    source: torch.Tensor,
    pairs: torch.Tensor,
    coordinates: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply exact exp(B) to every disjoint pair in a stage."""
    left, right = pairs.unbind(dim=1)
    generators = coordinates_to_generators(coordinates)
    maps = torch.matrix_exp(generators).to(dtype=source.dtype)
    pair_values = torch.stack(
        (source[:, left], source[:, right]), dim=-1
    )
    transformed = torch.einsum(
        "rpk,pkj->rpj", pair_values, maps
    )
    result = source.clone()
    result[:, left] = transformed[:, :, 0]
    result[:, right] = transformed[:, :, 1]
    determinants = torch.linalg.det(maps.double())
    condition = torch.linalg.cond(maps.double())
    return result, {
        "minimum_determinant": float(determinants.min()),
        "maximum_determinant_error": float(
            (determinants - 1.0).abs().max()
        ),
        "maximum_condition_number": float(condition.max()),
    }


def fit_tracefree_flow(
    source: torch.Tensor,
    requested_update: torch.Tensor,
    permutations: torch.Tensor,
    *,
    stages: int,
    component: Component,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Stagewise-fit and exactly apply a task-selected pair flow."""
    if (
        source.ndim != 2
        or source.shape != requested_update.shape
        or permutations.ndim != 2
        or permutations.shape[1] != source.shape[1]
        or stages <= 0
        or stages > permutations.shape[0]
    ):
        raise ValueError("invalid trace-free flow inputs")
    current = source.float().clone()
    target = source.float() + requested_update.float()
    coordinate_rows: list[torch.Tensor] = []
    minimum_determinant = float("inf")
    maximum_determinant_error = 0.0
    maximum_condition_number = 0.0
    for stage in range(stages):
        pairs = (
            permutations[stage]
            .to(device=source.device, dtype=torch.long)
            .reshape(-1, 2)
        )
        coordinates = fit_pair_coordinates(
            current, target - current, pairs, component
        )
        current, finite = apply_pair_stage(
            current, pairs, coordinates
        )
        coordinate_rows.append(coordinates.detach())
        minimum_determinant = min(
            minimum_determinant, finite["minimum_determinant"]
        )
        maximum_determinant_error = max(
            maximum_determinant_error,
            finite["maximum_determinant_error"],
        )
        maximum_condition_number = max(
            maximum_condition_number,
            finite["maximum_condition_number"],
        )
    all_coordinates = torch.cat(coordinate_rows, dim=0)
    residual = target - current
    energy = requested_update.double().square().sum().clamp_min(1e-30)
    recovery = 1.0 - residual.double().square().sum() / energy
    return current - source.float(), {
        "stages": stages,
        "component": component,
        "coordinates": (
            stages
            * (source.shape[1] // 2)
            * {"full": 3, "skew": 1, "nonorthogonal": 2}[component]
        ),
        "requested_update_recovery": float(recovery),
        "rotation_rms": float(
            all_coordinates[:, 2].square().mean().sqrt()
        ),
        "rotation_max_abs": float(all_coordinates[:, 2].abs().max()),
        "symmetric_shear_rms": float(
            all_coordinates[:, 1].square().mean().sqrt()
        ),
        "symmetric_shear_max_abs": float(
            all_coordinates[:, 1].abs().max()
        ),
        "anisotropic_scale_rms": float(
            all_coordinates[:, 0].square().mean().sqrt()
        ),
        "anisotropic_scale_max_abs": float(
            all_coordinates[:, 0].abs().max()
        ),
        "minimum_determinant": minimum_determinant,
        "maximum_determinant_error": maximum_determinant_error,
        "maximum_condition_number": maximum_condition_number,
    }


def _exact_edge_scores(
    source: torch.Tensor,
    residual: torch.Tensor,
    edges: torch.Tensor,
) -> torch.Tensor:
    normal, rhs = _pair_normal_equations(source, residual, edges)
    scale = normal.diagonal(dim1=1, dim2=2).mean(dim=1)
    ridge = (
        scale.clamp_min(1e-30) * 1e-10
    ).reshape(-1, 1, 1)
    coordinates = torch.linalg.solve(
        normal
        + ridge
        * torch.eye(
            3, dtype=normal.dtype, device=normal.device
        ).unsqueeze(0),
        rhs.unsqueeze(-1),
    ).squeeze(-1)
    return (rhs * coordinates).sum(dim=1).clamp_min(0.0)


def tracefree_matched_permutations(
    source: torch.Tensor,
    residual: torch.Tensor,
    *,
    stages: int,
    neighbors: int,
    seed: int,
    native_cache: Path | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Shortlist and exactly rank edges by trace-free tangent recovery."""
    if (
        source.ndim != 2
        or source.shape != residual.shape
        or source.shape[1] <= 0
        or source.shape[1] % 2
        or stages <= 0
        or stages > 64
        or neighbors < stages
        or neighbors >= source.shape[1]
    ):
        raise ValueError("invalid trace-free matching inputs")
    source = source.float()
    residual = residual.float()
    width = source.shape[1]
    cross = source.T @ residual
    diagonal = cross.diagonal()
    norms = source.square().sum(dim=0)
    denominator = (
        norms.unsqueeze(1) + norms.unsqueeze(0)
    ).clamp_min(1e-30)
    preliminary = (
        (
            diagonal.unsqueeze(1) - diagonal.unsqueeze(0)
        ).square()
        + 2.0 * (cross.square() + cross.T.square())
    ) / denominator
    preliminary.fill_diagonal_(-1.0)
    _values, indices = torch.topk(
        preliminary, k=neighbors, dim=1
    )
    left = (
        torch.arange(width, device=source.device)
        .repeat_interleave(neighbors)
    )
    right = indices.reshape(-1)
    edges = torch.stack(
        (torch.minimum(left, right), torch.maximum(left, right)),
        dim=1,
    )
    edges = torch.unique(edges, dim=0)
    scores = _exact_edge_scores(source, residual, edges)
    order = torch.argsort(scores, descending=True)
    sorted_edges = (
        edges.index_select(0, order)
        .to(device="cpu", dtype=torch.int32)
    )
    permutations, diagnostics = color_sorted_edges(
        sorted_edges,
        width=width,
        stages=stages,
        seed=seed,
        cache_dir=native_cache,
    )
    diagnostics.update(
        {
            "preselection_neighbors": neighbors,
            "unique_shortlist_edges": int(edges.shape[0]),
            "maximum_exact_projection_score": float(scores.max()),
            "mean_exact_projection_score": float(scores.mean()),
            "score_family": "exact_tracefree_2x2_tangent",
        }
    )
    return permutations, diagnostics


def build_candidates(
    source: torch.Tensor,
    dense_update: torch.Tensor,
    dense_direction: torch.Tensor,
    *,
    neighbors: int,
    matching_seed: int,
    native_cache: Path | None,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    """Build all preregistered SL2 candidates and matched controls."""
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

    control_permutations, control_selection = (
        fast_muon_matched_permutations(
            after_parent,
            residual,
            stages=48,
            neighbors=neighbors,
            seed=matching_seed + 1,
            cache_dir=native_cache,
        )
    )
    control24, control24_fit = (
        diagonal_metric_causal_givens_update(
            after_parent,
            residual,
            stages=24,
            seed=matching_seed + 1,
            permutations=control_permutations[:24],
        )
    )
    control48, control48_fit = (
        diagonal_metric_causal_givens_update(
            after_parent,
            residual,
            stages=48,
            seed=matching_seed + 1,
            permutations=control_permutations,
        )
    )

    sl2_permutations, sl2_selection = (
        tracefree_matched_permutations(
            after_parent,
            residual,
            stages=16,
            neighbors=neighbors,
            seed=matching_seed + 2,
            native_cache=native_cache,
        )
    )
    fitted: dict[tuple[int, Component], tuple[torch.Tensor, dict]] = {}
    for stages in (8, 16):
        for component in ("full", "skew", "nonorthogonal"):
            fitted[(stages, component)] = fit_tracefree_flow(
                after_parent,
                residual,
                sl2_permutations,
                stages=stages,
                component=component,
            )

    candidates = {
        "dense_exact": dense_update.float(),
        "fresh_hidden64": parent,
        "fresh_hidden88": parent + control24,
        "fresh_hidden112": parent + control48,
        "fresh_hidden64_plus_sl2_8": (
            parent + fitted[(8, "full")][0]
        ),
        "fresh_hidden64_plus_sl2_16": (
            parent + fitted[(16, "full")][0]
        ),
        "fresh_hidden64_plus_sl2_8_skew_only": (
            parent + fitted[(8, "skew")][0]
        ),
        "fresh_hidden64_plus_sl2_16_skew_only": (
            parent + fitted[(16, "skew")][0]
        ),
        "fresh_hidden64_plus_sl2_8_nonorthogonal_only": (
            parent + fitted[(8, "nonorthogonal")][0]
        ),
        "fresh_hidden64_plus_sl2_16_nonorthogonal_only": (
            parent + fitted[(16, "nonorthogonal")][0]
        ),
    }
    diagnostics: list[dict[str, Any]] = [
        {
            "selection": "fresh_hidden64",
            **parent_selection,
            **{
                f"fit_{key}": value
                for key, value in parent_fit.items()
            },
        },
        {
            "selection": "fresh_hidden_residual48_control",
            **control_selection,
            "fit24": control24_fit,
            "fit48": control48_fit,
        },
        {
            "selection": "tracefree_residual16",
            **sl2_selection,
        },
    ]
    diagnostics.extend(
        {
            "selection": f"tracefree_{stages}_{component}",
            **fit_diagnostics,
        }
        for (stages, component), (_update, fit_diagnostics)
        in fitted.items()
    )
    return candidates, diagnostics


def _comparison(
    metrics: dict[str, Any],
    finite_rows: list[dict[str, Any]],
    candidate: str,
    control: str,
) -> dict[str, Any]:
    candidate_metrics = metrics[candidate]
    control_metrics = metrics[control]
    return {
        "candidate": candidate,
        "control": control,
        "current_output_positive_line_ratio": runner.safe_ratio(
            candidate_metrics[
                "current_output_positive_line_recovery"
            ],
            control_metrics[
                "current_output_positive_line_recovery"
            ],
        ),
        "future_residual_recovery_ratio": runner.safe_ratio(
            candidate_metrics[
                "future_residual_positive_line_recovery"
            ],
            control_metrics[
                "future_residual_positive_line_recovery"
            ],
        ),
        "validation_gradient_ce_decrease_ratio": {
            window: runner.safe_ratio(
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
        "finite_ce": runner.finite_comparison(
            finite_rows, candidate, control
        ),
    }


def aggregate_results(
    rows: list[dict[str, Any]],
    finite_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply the immutable equal-budget and attribution gates."""
    metrics: dict[str, Any] = {}
    for candidate in CANDIDATES:
        selected = [row for row in rows if row["candidate"] == candidate]
        fit = [row for row in selected if row["window"] == "fit"]
        metrics[candidate] = {
            "coordinates_per_layer": COORDINATES[candidate],
            "cells_times_windows": len(selected),
            "current_weight_recovery": runner.weighted(
                fit, "current_weight_recovery", "current_weight_energy"
            ),
            "future_residual_positive_line_recovery": runner.weighted(
                fit,
                "future_residual_positive_line_recovery",
                "future_residual_energy",
            ),
            "current_residual_fixed_scale_recovery": runner.weighted(
                fit,
                "current_residual_fixed_scale_recovery",
                "current_residual_energy",
            ),
            "current_output_positive_line_recovery": runner.weighted(
                selected,
                "current_output_positive_line_recovery",
                "current_output_energy",
            ),
            "current_output_fixed_scale_recovery": runner.weighted(
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

    comparison_specs = {
        "sl2_8_vs_equal_coordinate_hidden88": (
            "fresh_hidden64_plus_sl2_8",
            "fresh_hidden88",
        ),
        "sl2_16_vs_equal_coordinate_hidden112": (
            "fresh_hidden64_plus_sl2_16",
            "fresh_hidden112",
        ),
        "sl2_8_vs_same_topology_skew": (
            "fresh_hidden64_plus_sl2_8",
            "fresh_hidden64_plus_sl2_8_skew_only",
        ),
        "sl2_16_vs_same_topology_skew": (
            "fresh_hidden64_plus_sl2_16",
            "fresh_hidden64_plus_sl2_16_skew_only",
        ),
    }
    comparisons = {
        name: _comparison(
            metrics, finite_rows, candidate, control
        )
        for name, (candidate, control) in comparison_specs.items()
    }

    def functional_pass(
        record: dict[str, Any],
        *,
        finite_required: bool,
    ) -> bool:
        return (
            record["current_output_positive_line_ratio"] >= 1.05
            and record["future_residual_recovery_ratio"] >= 1.10
            and min(
                record[
                    "validation_gradient_ce_decrease_ratio"
                ].values()
            )
            >= 1.05
            and (
                not finite_required
                or record["finite_ce"]["wins"] >= 7
            )
            and record[
                "minimum_train_gradient_predicted_ce_decrease"
            ]
            > 0.0
        )

    passes: list[int] = []
    for stages in (8, 16):
        equal = comparisons[
            f"sl2_{stages}_vs_equal_coordinate_hidden"
            f"{88 if stages == 8 else 112}"
        ]
        attribution = comparisons[
            f"sl2_{stages}_vs_same_topology_skew"
        ]
        if functional_pass(equal, finite_required=True) and functional_pass(
            attribution, finite_required=False
        ):
            passes.append(stages)

    def conservative_ratio(stages: int) -> float:
        record = comparisons[
            f"sl2_{stages}_vs_equal_coordinate_hidden"
            f"{88 if stages == 8 else 112}"
        ]
        return min(
            record["current_output_positive_line_ratio"],
            record["future_residual_recovery_ratio"],
            *record[
                "validation_gradient_ce_decrease_ratio"
            ].values(),
        )

    selected_branch: str | None = None
    if 8 in passes:
        selected = 8
        if (
            16 in passes
            and conservative_ratio(16)
            >= 1.10 * conservative_ratio(8)
        ):
            selected = 16
        selected_branch = f"sl2_{selected}"
    elif 16 in passes:
        selected_branch = "sl2_16"
    decision = (
        "SELECT_TRACEFREE_PAIR_CHART_FOR_IMPLEMENTATION_PREFLIGHT"
        if selected_branch is not None
        else "REJECT_SPARSE_TRACEFREE_PAIR_BLOCKS"
    )
    return {
        "candidate_metrics": metrics,
        "comparisons": comparisons,
        "passing_depths": passes,
        "selected_branch": selected_branch,
        "decision": decision,
    }


def main() -> None:
    """Install the preregistered candidate family in the shared runner."""
    runner.CANDIDATES = CANDIDATES
    runner.COORDINATES = COORDINATES
    runner.build_candidates = build_candidates
    runner.aggregate_results = aggregate_results
    runner.OUTPUT_STEM = "fast_fresh_sl2_residual"
    runner.METADATA_SCHEMA_VERSION = (
        "nanogpt_mlp_fast_fresh_sl2_residual_v1"
    )
    runner.METADATA_LIMITATIONS = [
        "This is a zero-update causal tangent diagnostic, not training.",
        "Finite-step CE perturbs five representative c_proj layers only.",
        "The dense Muon replay is one optimizer path, not the global low-loss manifold.",
        "Trace-free topology uses a 128-neighbor approximate shortlist followed by exact 3x3 projected-recovery reranking.",
        "Passing authorizes production implementation and an MFU preflight only, not scientific training.",
    ]
    runner.__file__ = __file__
    runner.__doc__ = __doc__
    runner.main()


if __name__ == "__main__":
    main()
