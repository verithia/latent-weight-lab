from __future__ import annotations

import copy

import pytest
import torch

from examples.nanogpt.analyze_mlp_cproj_fht_block_cayley_output import (
    aggregate_results,
    fit_fht_block_cayley_pass,
    solve_block_cayley_coordinates,
    validate_plan,
)


def valid_plan() -> dict:
    return {
        "schema_version": "mai_124m_mlp_cproj_fht_block_cayley_output_plan_v1",
        "authorization": {"implement_and_run_zero_update_analysis": True},
        "analysis": {
            "layers": [0, 3, 6, 9, 11],
            "phases": [[0, 60], [60, 120], [120, 180], [180, 238]],
            "fit_window": {"split": "validation", "seed": 20260804, "batch_size": 2, "block_size": 256, "batches": 4, "rows_per_layer": 2048},
            "holdout_window": {"split": "validation", "seed": 20260805, "batch_size": 2, "block_size": 256, "batches": 4, "rows_per_layer": 2048},
            "shared_hidden_chart": {"parent_stages": 64, "residual_stages": 24, "neighbors": 64, "matching_seed": 20260804, "coordinates_per_layer": 135168, "feedback": "zero for this one-step prospective diagnostic", "weight_decay_application": "identical production ordering in both arms"},
            "control": {"name": "frobenius_output32", "output_stages": 32, "output_coordinates_per_layer": 12288, "total_coordinates_per_layer": 147456, "definition": "Existing top-neighbor Frobenius pair selector with simultaneous Frobenius residual-fit angles."},
            "candidate": {"name": "fht_block32_cayley1", "basis": "one exact fixed signed/permuted normalized block-FHT basis and its inverse", "basis_block_size": 256, "basis_seed": 20260804, "rotation_block_size": 32, "stages": 1, "rotation_blocks": 24, "coordinates_per_block": 496, "output_coordinates_per_layer": 11904, "total_coordinates_per_layer": 147072, "fit": {"source": "X = basis(W_after_hidden^T)", "residual": "R = basis(remaining requested update^T)", "per_block_equation": "C B + B C = D - D^T, where C=X^T X and D=X^T R", "solver": "symmetric eigendecomposition of C in float64; denominator lambda_i+lambda_j+1e-6*mean(diag(C))", "cayley_coordinates": "A=B/2 because Cayley(A)=I+2A+O(A^2)", "trust_radius": "multiply all A blocks in a layer-phase cell by min(1, max_abs(control_output32_angle)/max_abs(A))", "application": "apply the exact block Cayley transform in the fixed basis, invert the basis, transpose, and use the same decoupled-weight-decay ordering as control"}},
            "parameter_updates": 0,
        },
    }


def test_plan_validation_fails_closed() -> None:
    validate_plan(valid_plan())
    changed = copy.deepcopy(valid_plan())
    changed["analysis"]["candidate"]["rotation_block_size"] = 16
    with pytest.raises(ValueError):
        validate_plan(changed)


def test_sylvester_solver_recovers_small_skew_tangent() -> None:
    source = torch.eye(4, dtype=torch.float64)
    target_tangent = torch.tensor(
        [[0.0, 0.1, 0.0, 0.0], [-0.1, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, -0.2], [0.0, 0.0, 0.2, 0.0]],
        dtype=torch.float64,
    )
    coordinates, diagnostics = solve_block_cayley_coordinates(
        source, source @ target_tangent, relative_ridge=0.0
    )
    torch.testing.assert_close(2.0 * coordinates, target_tangent, rtol=1e-12, atol=1e-12)
    assert diagnostics["linear_residual_energy"] == pytest.approx(0.0, abs=1e-20)


def test_exact_cayley_pass_obeys_frozen_trust_radius() -> None:
    generator = torch.Generator().manual_seed(7)
    source = torch.randn(16, 8, generator=generator)
    residual = torch.randn(16, 8, generator=generator)
    updated, diagnostics = fit_fht_block_cayley_pass(
        source,
        residual,
        rotation_block_size=4,
        basis_block_size=8,
        seed=11,
        trust_radius=1e-3,
    )
    assert torch.isfinite(updated).all()
    assert diagnostics["coordinates"] == 12
    assert diagnostics["trust_radius_obeyed"] is True
    assert diagnostics["maximum_abs_coordinate"] <= 1e-3 + 1e-9


def synthetic_rows() -> tuple[list[dict], list[dict], list[dict]]:
    rows: list[dict] = []
    finite: list[dict] = []
    chart: list[dict] = []
    for phase in (0, 60, 120, 180):
        for layer in (0, 3, 6, 9, 11):
            chart.append({"phase_start": phase, "layer": layer, "trust_radius_obeyed": True, "trust_scale": 0.9, "raw_maximum_abs_coordinate": 0.2, "maximum_abs_coordinate": 0.1})
            for window in ("fit", "holdout"):
                for arm, coordinates, residual, task, update in (
                    ("frobenius_output32", 147456, 1.0, 0.003, 1.0),
                    ("fht_block32_cayley1", 147072, 0.9, 0.005, 1.05),
                ):
                    rows.append({"phase_start": phase, "layer": layer, "window": window, "arm": arm, "coordinates_per_layer": coordinates, "activation_output_residual_energy": residual, "validation_gradient_predicted_ce_decrease": task, "update_energy": update, "weight_error_energy": residual})
        for window in ("fit", "holdout"):
            finite.extend(
                [
                    {"phase_start": phase, "window": window, "arm": "frobenius_output32", "loss": 7.1815},
                    {"phase_start": phase, "window": window, "arm": "fht_block32_cayley1", "loss": 7.1800},
                ]
            )
    return rows, finite, chart


def test_aggregate_requires_every_registered_gate() -> None:
    rows, finite, chart = synthetic_rows()
    result = aggregate_results(rows, finite, chart)
    assert result["passed"] is True
    assert result["authorization"]["language_model_training_authorized"] is False
    chart[0]["trust_radius_obeyed"] = False
    failed = aggregate_results(rows, finite, chart)
    assert failed["passed"] is False
