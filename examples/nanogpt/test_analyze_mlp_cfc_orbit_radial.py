from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_cfc_orbit_radial import (
    aggregate,
    svd_orbit_radial_components,
)


def test_orbit_radial_components_reconstruct_residual() -> None:
    generator = torch.Generator().manual_seed(23)
    weight = torch.randn(12, 5, generator=generator)
    residual = torch.randn(12, 5, generator=generator)
    components, diagnostics = svd_orbit_radial_components(weight, residual)
    torch.testing.assert_close(
        components["bilateral_orbit"] + components["radial"],
        residual,
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        components["left_orbit_radial"],
        components["left_orbit"] + components["radial"],
    )
    assert diagnostics["bilateral_plus_radial_relative_error"] < 1e-6
    assert diagnostics["left_omega_skew_error"] < 1e-5
    assert diagnostics["right_omega_skew_error"] < 1e-5


def test_aggregate_selects_bilateral_orbit_when_left_is_insufficient() -> None:
    losses = {
        "baseline": 5.2,
        "fresh88": 5.1,
        "dense_exact": 5.0,
        "fresh_plus_left_orbit": 5.04,
        "fresh_plus_right_orbit": 5.08,
        "fresh_plus_bilateral_orbit": 5.01,
        "fresh_plus_radial": 5.09,
        "fresh_plus_left_orbit_radial": 5.03,
        "fresh_plus_right_orbit_radial": 5.07,
    }
    rows = []
    windows = [f"validation_{index}" for index in range(1, 5)]
    for window in windows:
        for repeat in range(3):
            for candidate, loss in losses.items():
                rows.append({"window": window, "candidate": candidate, "repeat": repeat, "loss": loss})
    result = aggregate(
        rows,
        windows=windows,
        numerical_range_tolerance=1e-7,
        sufficient_minimum_recovery=0.75,
        sufficient_median_recovery=0.85,
        radial_minimum_recovery=0.5,
        radial_median_recovery=0.65,
    )
    assert result["decision"] == "ADD_INPUT_SIDE_ROTATION_TO_CFC_CHART"
    assert result["gates"]["dense_beats_fresh_every_window"]
