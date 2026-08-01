from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_cfc_layer_local_residual import (
    aggregate_results,
    existing_weight_subspace_projections,
    make_subset_update,
)


def test_existing_weight_subspace_projections_match_projectors() -> None:
    generator = torch.Generator().manual_seed(17)
    weight = torch.randn(12, 5, generator=generator)
    residual = torch.randn(12, 5, generator=generator)
    projections, singular_values = existing_weight_subspace_projections(
        weight, residual, rank=3
    )
    u, _s, vh = torch.linalg.svd(weight.float(), full_matrices=False)
    left_expected = u[:, :3] @ (u[:, :3].T @ residual)
    right_expected = (residual @ vh[:3].T) @ vh[:3]
    joint_expected = u[:, :3] @ (
        (u[:, :3].T @ residual @ vh[:3].T) @ vh[:3]
    )
    torch.testing.assert_close(projections["left"], left_expected)
    torch.testing.assert_close(projections["right"], right_expected)
    torch.testing.assert_close(projections["joint"], joint_expected)
    assert len(singular_values) == 3


def test_make_subset_update_replaces_only_selected_layers() -> None:
    fresh = {0: torch.tensor([0.0]), 1: torch.tensor([1.0])}
    dense = {0: torch.tensor([2.0]), 1: torch.tensor([3.0])}
    result = make_subset_update(fresh, dense, [1])
    torch.testing.assert_close(result[0], fresh[0])
    torch.testing.assert_close(result[1], dense[1])


def test_aggregate_selects_qualifying_joint_subspace() -> None:
    specs = {
        "fresh88": {"family": "fresh88", "scope": "all", "coordinates_total": 100},
        "dense_exact": {"family": "dense_exact", "scope": "all", "coordinates_total": 1000},
        "exact_top3": {"family": "dense_exact_subset", "scope": "top3", "coordinates_total": 300},
        "exact_top6": {"family": "dense_exact_subset", "scope": "top6", "coordinates_total": 600},
        "joint_top3_rank2": {"family": "joint", "scope": "top3", "coordinates_total": 12},
        "left_all_rank2": {"family": "left", "scope": "all", "coordinates_total": 40},
    }
    losses = {
        "baseline": 5.2,
        "fresh88": 5.1,
        "dense_exact": 5.0,
        "exact_top3": 5.04,
        "exact_top6": 5.01,
        "joint_top3_rank2": 5.03,
        "left_all_rank2": 5.02,
    }
    rows = []
    windows = [f"validation_{index}" for index in range(1, 5)]
    for window in windows:
        for repeat in range(3):
            for candidate, loss in losses.items():
                rows.append(
                    {"window": window, "candidate": candidate, "repeat": repeat, "loss": loss}
                )
    result = aggregate_results(
        rows,
        windows=windows,
        candidate_specs=specs,
        fit_ranking=[{"layer": 0, "fit_improvement_over_fresh": 0.1}],
        numerical_range_tolerance=1e-7,
        minimum_recovery_every_window=0.5,
        minimum_median_recovery=0.65,
        top3_concentration_threshold=0.5,
        top6_concentration_threshold=0.75,
    )
    assert result["decision"] == "SELECT_LAYER_LOCAL_SUBSPACE_FOR_CHART_DESIGN"
    assert result["selected_candidate"] == "joint_top3_rank2"
    assert result["gates"]["dense_beats_fresh_every_window"]
