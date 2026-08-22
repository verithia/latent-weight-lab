from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from examples.nanogpt.muon import muon_update
from examples.nanogpt.pair_vq_polar_lowrank_momentum import (
    PairVQPolarLowRankMomentumOracle,
    deterministic_svd_lowrank,
    fit_lowrank_core,
    polar_sensitivity,
)


def test_registered_plan_identity_and_scope() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / (
        "examples/nanogpt/configs/selection_artifacts/"
        "124m_pair_vq_polar_sensitive_lowrank_momentum_plan.json"
    )
    plan = json.loads(path.read_text())
    assert plan["schema_version"].endswith("_v1")
    assert plan["frozen_protocol"]["candidate_parameter_updates"] == 0
    assert plan["decision_gate"]["automatic_endpoint"] is False
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "dad88a1a08daf04ae618f9b365c7a73317b38ef64b6610ca0675d9e50b3160c6"
    )


def test_randomized_svd_is_seeded_and_rng_isolated() -> None:
    torch.manual_seed(17)
    matrix = torch.randn(24, 16)
    state = torch.random.get_rng_state().clone()
    first = deterministic_svd_lowrank(matrix, q=8, niter=2, seed=99)
    assert torch.equal(torch.random.get_rng_state(), state)
    second = deterministic_svd_lowrank(matrix, q=8, niter=2, seed=99)
    for left, right in zip(first, second, strict=True):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)


def test_polar_sensitivity_is_finite_and_shape_preserving() -> None:
    generator = torch.Generator().manual_seed(21)
    gradient = torch.randn(12, 8, generator=generator)
    base = torch.randn(12, 8, generator=generator)
    reference_state = base + 0.01 * torch.randn(12, 8, generator=generator)
    reference_update = muon_update(
        gradient + 0.95 * reference_state, steps=3
    ).float()
    sensitivity = polar_sensitivity(
        reference_update=reference_update,
        gradient=gradient,
        base_state=base,
        momentum=0.95,
        ns_steps=3,
    )
    assert sensitivity.shape == base.shape
    assert torch.isfinite(sensitivity).all()
    assert float(sensitivity.square().sum()) > 0.0


def test_core_fit_is_monotonic_and_recovers_in_basis_residual() -> None:
    generator = torch.Generator().manual_seed(31)
    base = torch.randn(12, 8, generator=generator)
    left = torch.randn(12, 1, generator=generator)
    left = left / left.norm(dim=0, keepdim=True)
    right = torch.randn(8, 1, generator=generator)
    right = right / right.norm(dim=0, keepdim=True)
    reference_state = base + 0.025 * left @ right.T
    gradient = torch.randn(12, 8, generator=generator)
    reference_update = muon_update(
        gradient + 0.95 * reference_state, steps=3
    ).float()
    correction, fit = fit_lowrank_core(
        reference_state=reference_state,
        reference_update=reference_update,
        gradient=gradient,
        base_state=base,
        left=left,
        right=right,
        momentum=0.95,
        ns_steps=3,
        maximum_iterations=4,
        history_size=4,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        state_weight=0.01,
    )
    assert fit["objective_monotonic"] is True
    assert fit["selected_objective"] <= fit["initial_objective"] + 1e-10
    torch.testing.assert_close(
        base + correction, reference_state, rtol=1e-5, atol=1e-6
    )


def test_gate_selects_smallest_common_passing_candidate() -> None:
    oracle = PairVQPolarLowRankMomentumOracle.__new__(
        PairVQPolarLowRankMomentumOracle
    )
    oracle.update_indices = {0, 8}
    oracle.stop_on_gate = True
    oracle.plan = {
        "decision_gate": {
            "requirements_at_every_registered_probe": {
                "minimum_all_postpolar_cosine": 0.9999,
                "minimum_every_matrix_postpolar_cosine": 0.999,
                "minimum_all_postpolar_positive_line_energy_recovery": 0.999,
                "minimum_all_prepolar_cosine": 0.9999,
                "minimum_all_momentum_state_cosine": 0.9999,
                "all_metrics_finite": True,
                "objective_monotonic": True,
            }
        }
    }

    def metric(cosine: float) -> dict[str, float]:
        return {
            "cosine": cosine,
            "worst_matrix_cosine": cosine,
            "positive_line_energy_recovery": cosine,
        }

    def candidate(cosine: float, total: int | None) -> dict[str, object]:
        row = {
            "all": {
                "momentum_state": metric(cosine),
                "combined_prepolar": metric(cosine),
                "polar_update": metric(cosine),
            },
            "objective_monotonic": True,
            "storage": None,
        }
        if total is not None:
            row["storage"] = {"total_training_bytes": total}
        return row

    oracle.records = [
        {
            "optimizer_update_index": step,
            "stage_a_passed": True,
            "aggregate": {
                "fp16_full_residual_control": candidate(1.0, None),
                "e5m8_rank0": candidate(0.99, None),
                "e5m8_residual_svd_r8": candidate(0.99995, 200),
                "e5m8_polar_gradient_r4": candidate(0.99995, 100),
            },
        }
        for step in (0, 8)
    ]
    gate = oracle._gate()
    assert gate["classification"] == "PASS"
    assert gate["selected"] == "e5m8_polar_gradient_r4"
    assert oracle.probe_only is True
