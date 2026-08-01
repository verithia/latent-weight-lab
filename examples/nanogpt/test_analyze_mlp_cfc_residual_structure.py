from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_cfc_residual_structure import (
    aggregate_losses,
    fit_bilateral_diagonal,
    fit_expansion_diagonal,
    fit_input_diagonal,
)


def test_diagonal_fits_recover_their_exact_families() -> None:
    generator = torch.Generator().manual_seed(7)
    weight = torch.randn(12, 5, generator=generator)
    input_scale = torch.randn(5, generator=generator)
    expansion_scale = torch.randn(12, generator=generator)
    input_residual = weight * input_scale
    expansion_residual = weight * expansion_scale[:, None]
    torch.testing.assert_close(
        fit_input_diagonal(weight, input_residual), input_residual
    )
    torch.testing.assert_close(
        fit_expansion_diagonal(weight, expansion_residual),
        expansion_residual,
    )
    bilateral = input_residual + expansion_residual
    fitted, _stats = fit_bilateral_diagonal(
        weight, bilateral, iterations=32
    )
    torch.testing.assert_close(fitted, bilateral, rtol=1e-5, atol=1e-5)


def test_aggregate_selects_smallest_qualifying_structure() -> None:
    candidates = {
        "fresh88": {"family": "fresh88", "coordinates_per_layer": 100},
        "dense_exact": {"family": "dense_exact", "coordinates_per_layer": 1000},
        "input": {"family": "input_diagonal", "coordinates_per_layer": 10},
        "lowrank": {"family": "low_rank_spectral", "coordinates_per_layer": 20},
    }
    rows = []
    for window in ("validation_1", "validation_2", "validation_3", "validation_4"):
        for repeat in range(3):
            for candidate, loss in (
                ("baseline", 5.2),
                ("fresh88", 5.1),
                ("dense_exact", 5.0),
                ("input", 5.04),
                ("lowrank", 5.02),
            ):
                rows.append({"window": window, "candidate": candidate, "loss": loss})
    result = aggregate_losses(
        rows,
        windows=[f"validation_{index}" for index in range(1, 5)],
        candidates=candidates,
        numerical_range_tolerance=1e-7,
        minimum_gap_recovery=0.5,
        median_gap_recovery=0.5,
    )
    assert result["selected_candidate"] == "input"
    assert result["selected_qualified"]
    assert result["gates"]["dense_beats_fresh_every_window"]


def test_aggregate_rejects_when_dense_is_not_a_positive_control() -> None:
    candidates = {
        "fresh88": {"family": "fresh88", "coordinates_per_layer": 100},
        "dense_exact": {"family": "dense_exact", "coordinates_per_layer": 1000},
        "input": {"family": "input_diagonal", "coordinates_per_layer": 10},
    }
    rows = []
    for window in ("validation_1", "validation_2", "validation_3", "validation_4"):
        for _repeat in range(3):
            for candidate, loss in (
                ("baseline", 5.2),
                ("fresh88", 5.1),
                ("dense_exact", 5.11),
                ("input", 5.0),
            ):
                rows.append({"window": window, "candidate": candidate, "loss": loss})
    result = aggregate_losses(
        rows,
        windows=[f"validation_{index}" for index in range(1, 5)],
        candidates=candidates,
        numerical_range_tolerance=1e-7,
        minimum_gap_recovery=0.5,
        median_gap_recovery=0.8,
    )
    assert result["decision"] == "DENSE_RESIDUAL_NOT_POSITIVE_CONTROL"
    assert not result["selected_qualified"]
    assert not result["gates"]["dense_beats_fresh_every_window"]
