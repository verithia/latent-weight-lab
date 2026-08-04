from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from examples.nanogpt.analyze_mlp_cproj_fht_block_affine_output import (
    aggregate_results,
    fit_fht_block_affine_pass,
    solve_block_affine,
    validate_plan,
)


PLAN = Path(__file__).parent / "configs/selection_artifacts/124m_mlp_cproj_fht_block_affine_output_plan.json"


def valid_plan() -> dict:
    return json.loads(PLAN.read_text())


def test_plan_validation_fails_closed() -> None:
    validate_plan(valid_plan())
    changed = copy.deepcopy(valid_plan())
    changed["analysis"]["fixed_block_basis"]["affine_block_size"] = 16
    with pytest.raises(ValueError):
        validate_plan(changed)


def test_block_affine_solver_recovers_general_linear_map() -> None:
    generator = torch.Generator().manual_seed(7)
    design = torch.randn(32, 4, generator=generator, dtype=torch.float64)
    target_map = torch.tensor(
        [[0.10, 0.04, 0.00, 0.00], [-0.03, -0.05, 0.02, 0.00], [0.00, 0.01, 0.08, -0.06], [0.02, 0.00, 0.04, -0.02]],
        dtype=torch.float64,
    )
    solved, diagnostics = solve_block_affine(
        design, design @ target_map, relative_ridge=0.0
    )
    torch.testing.assert_close(solved, target_map, rtol=1e-11, atol=1e-11)
    assert diagnostics["fit_residual_energy"] == pytest.approx(0.0, abs=1e-20)


def test_affine_pass_caps_energy_and_reports_nonorthogonal_parts() -> None:
    generator = torch.Generator().manual_seed(11)
    source = torch.randn(24, 8, generator=generator)
    residual = torch.randn(24, 8, generator=generator)
    updated, diagnostics = fit_fht_block_affine_pass(
        source, residual, activation=None, affine_block_size=4,
        basis_block_size=8, seed=13, trust_output_energy=1e-4,
    )
    assert torch.isfinite(updated).all()
    assert diagnostics["coordinates"] == 32
    assert diagnostics["trust_energy_obeyed"] is True
    assert diagnostics["bounded_output_delta_energy"] <= 1.0001e-4
    total_parts = (
        diagnostics["skew_coordinate_energy"]
        + diagnostics["symmetric_offdiag_coordinate_energy"]
        + diagnostics["diagonal_coordinate_energy"]
    )
    assert total_parts == pytest.approx(diagnostics["coordinate_energy"], rel=1e-6)


def synthetic_rows() -> tuple[list[dict], list[dict], list[dict]]:
    rows: list[dict] = []
    finite: list[dict] = []
    chart: list[dict] = []
    for phase in (0, 60, 120, 180):
        for layer in (0, 3, 6, 9, 11):
            for candidate in ("fht_block32_affine_fro", "fht_block32_affine_activation"):
                chart.append({
                    "phase_start": phase, "layer": layer, "arm": candidate,
                    "trust_energy_obeyed": True, "trust_scale": 0.8,
                    "minimum_singular_value_i_plus_b": 0.99,
                    "coordinate_energy": 1.0, "skew_coordinate_energy": 0.3,
                    "symmetric_offdiag_coordinate_energy": 0.5,
                    "diagonal_coordinate_energy": 0.2,
                })
            for window in ("fit", "holdout"):
                for arm, coordinates, residual, task in (
                    ("frobenius_output32", 147456, 1.05, 0.0035),
                    ("frobenius_output64", 159744, 1.00, 0.0038),
                    ("fht_block32_affine_fro", 159744, 0.90, 0.0046),
                    ("fht_block32_affine_activation", 159744, 0.88, 0.0048),
                ):
                    rows.append({
                        "phase_start": phase, "layer": layer, "window": window,
                        "arm": arm, "coordinates_per_layer": coordinates,
                        "activation_output_residual_energy": residual,
                        "validation_gradient_predicted_ce_decrease": task,
                        "update_energy": 1.0,
                    })
        for window in ("fit", "holdout"):
            finite.extend([
                {"phase_start": phase, "window": window, "arm": "frobenius_output32", "loss": 7.1818},
                {"phase_start": phase, "window": window, "arm": "frobenius_output64", "loss": 7.1815},
                {"phase_start": phase, "window": window, "arm": "fht_block32_affine_fro", "loss": 7.1805},
                {"phase_start": phase, "window": window, "arm": "fht_block32_affine_activation", "loss": 7.1800},
            ])
    return rows, finite, chart


def test_aggregate_uses_frozen_smallest_pass_order() -> None:
    rows, finite, chart = synthetic_rows()
    result = aggregate_results(rows, finite, chart)
    assert result["passed"] is True
    assert result["selected"] == "fht_block32_affine_fro"
    assert result["authorization"]["language_model_training_authorized"] is False
    for row in chart:
        if row["arm"] == "fht_block32_affine_fro":
            row["minimum_singular_value_i_plus_b"] = 0.90
    result = aggregate_results(rows, finite, chart)
    assert result["selected"] == "fht_block32_affine_activation"
