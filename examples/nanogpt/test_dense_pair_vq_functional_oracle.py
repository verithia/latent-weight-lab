from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from examples.nanogpt.dense_pair_vq_functional_oracle import (
    PairVQFunctionalGradientOracle,
    _aggregate_directions,
    _direction_metrics,
    antithetic_average,
    gradient_comparison,
    gradient_cross_cosine,
)
from examples.nanogpt.muon import muon_update


def test_antithetic_average_is_exactly_centered() -> None:
    center = torch.tensor([[1.0, -2.0], [0.5, 4.0]])
    residual = torch.tensor([[0.25, -0.5], [1.0, -2.0]])
    minus = {"transformer.h.0.mlp.c_fc": center - residual}
    plus = {"transformer.h.0.mlp.c_fc": center + residual}
    averaged = antithetic_average(minus, plus)
    torch.testing.assert_close(
        averaged["transformer.h.0.mlp.c_fc"], center, rtol=0.0, atol=0.0
    )


def test_gradient_comparison_and_cross_cosine() -> None:
    reference = {
        "transformer.h.0.mlp.c_fc": torch.tensor([[1.0, 0.0]]),
        "transformer.h.0.mlp.c_proj": torch.tensor([[0.0, 2.0]]),
    }
    exact = {name: value.clone() for name, value in reference.items()}
    comparison = gradient_comparison(reference, exact)
    assert comparison["aggregate"]["relative_error"] == 0.0
    assert comparison["aggregate"]["cosine"] == 1.0
    assert gradient_cross_cosine(reference, exact) == 1.0


def test_polar_direction_aggregation_preserves_energy_accounting() -> None:
    reference = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    candidate = torch.tensor([[0.9, 0.1], [0.2, 1.8]])
    metrics = _direction_metrics(reference, candidate)
    row = {"prepolar": metrics}
    aggregate = _aggregate_directions([row, row], "prepolar")
    assert aggregate["reference_energy"] == 2.0 * metrics["reference_energy"]
    assert aggregate["candidate_energy"] == 2.0 * metrics["candidate_energy"]
    assert aggregate["error_energy"] == 2.0 * metrics["error_energy"]
    assert aggregate["relative_error"] == metrics["relative_error"]
    assert aggregate["cosine"] == metrics["cosine"]


def test_registered_functional_oracle_plan_is_immutable_and_causal() -> None:
    root = Path(__file__).resolve().parents[2]
    path = (
        root
        / "examples/nanogpt/configs/selection_artifacts/124m_pair_vq_antithetic_functional_gradient_oracle_plan.json"
    )
    plan = json.loads(path.read_text())
    assert plan["schema_version"].endswith("_v1")
    assert plan["frozen_protocol"]["primary_late_steps"] == [180, 238]
    assert (
        plan["frozen_gate"][
            "minimum_late_heldout_gradient_error_closure_vs_native"
        ]
        == 0.70
    )
    assert plan["decision_rule"]["automatic_endpoint"] is False
    assert plan["decision_rule"]["automatic_scale_up"] is False
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "c856d2d695a569673572004e457c275e632f6461e5e1682c4f224bc22e71ba4f"
    )


def test_same_momentum_polar_plan_is_immutable_and_nonintervening() -> None:
    root = Path(__file__).resolve().parents[2]
    path = (
        root
        / "examples/nanogpt/configs/selection_artifacts/124m_pair_vq_same_momentum_polar_amplification_oracle_plan.json"
    )
    plan = json.loads(path.read_text())
    assert plan["schema_version"].endswith("_v1")
    assert plan["frozen_protocol"]["primary_late_steps"] == [180, 238]
    assert plan["polar_gate"] == {
        "minimum_late_prepolar_cosine": 0.999,
        "maximum_late_polar_cosine": 0.9998,
        "minimum_late_polar_relative_error": 0.02,
        "minimum_late_relative_error_amplification": 2.0,
    }
    assert plan["decision_rule"]["automatic_endpoint"] is False
    assert plan["decision_rule"]["automatic_scale_up"] is False
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "d77f7536ec1ce10c5f966a4e17325ea04d8081db286b11b567c35c19d6ca046f"
    )


def test_early_stopped_polar_plan_is_immutable_and_causal() -> None:
    root = Path(__file__).resolve().parents[2]
    path = (
        root
        / "examples/nanogpt/configs/selection_artifacts/124m_pair_vq_early_stopped_muon_polar_stability_oracle_plan.json"
    )
    plan = json.loads(path.read_text())
    assert plan["schema_version"].endswith("_v1")
    assert plan["regularized_polar_frontier"] == {
        "native_ns_steps": 5,
        "candidate_ns_steps": [1, 2, 3, 4],
        "selection_rule": "select the largest candidate_ns_steps value that passes every frozen gate at both primary late steps",
    }
    assert plan["decision_rule"]["automatic_endpoint"] is False
    assert plan["decision_rule"]["automatic_scale_up"] is False
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "6a722065917c0bbd3a5852b913e883ec5e89b0925916409d0e5b9a31544a19a9"
    )


def test_codec_neighbor_stability_plan_is_immutable_and_nonintervening() -> None:
    root = Path(__file__).resolve().parents[2]
    path = (
        root
        / "examples/nanogpt/configs/selection_artifacts/124m_pair_vq_codec_neighbor_path_stability_oracle_plan.json"
    )
    plan = json.loads(path.read_text())
    assert plan["schema_version"].endswith("_v1")
    assert plan["state_and_compute_contract"]["candidate_parameter_updates"] == 0
    assert plan["decision_rule"]["automatic_training_endpoint"] is False
    assert plan["decision_rule"]["automatic_scale_up"] is False
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "541459da108d4c1b01481077827bcd898f00b2af0358513a168a70f1debaf8bf"
    )


def test_spectral_damping_plan_is_immutable_and_nonintervening() -> None:
    root = Path(__file__).resolve().parents[2]
    path = (
        root
        / "examples/nanogpt/configs/selection_artifacts/124m_pair_vq_norm_preserving_spectral_damping_oracle_plan.json"
    )
    plan = json.loads(path.read_text())
    assert plan["schema_version"].endswith("_v1")
    assert plan["spectral_damping_frontier"] == {
        "native_ns_steps": 5,
        "relative_rms_ridge": [0.03125, 0.0625, 0.125, 0.25],
        "one_global_rho_for_all_matrices": True,
        "selection_rule": "select the smallest relative_rms_ridge value that passes every frozen gate at both primary late steps",
    }
    assert plan["state_and_compute_contract"]["candidate_parameter_updates"] == 0
    assert plan["decision_rule"]["automatic_training_endpoint"] is False
    assert plan["decision_rule"]["automatic_scale_up"] is False
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "f890f4b7c560c1662bb1a0e6defc54726efff3ed042ed251a2ca389649916abd"
    )


def test_spectral_damping_preserves_norm_and_attenuates_weak_modes() -> None:
    request = torch.diag(torch.tensor([1.0, 0.2, 0.01], dtype=torch.float32))
    candidate = PairVQFunctionalGradientOracle._norm_preserving_spectral_damping_frontier(
        request,
        relative_rms_ridges=(0.125,),
        native_steps=5,
    )["0.125"]
    native = muon_update(request, steps=5).float()
    torch.testing.assert_close(candidate.norm(), native.norm(), rtol=2e-6, atol=1e-7)
    native_singular = torch.linalg.svdvals(native)
    candidate_singular = torch.linalg.svdvals(candidate)
    weak_to_strong_native = native_singular[-1] / native_singular[0]
    weak_to_strong_candidate = candidate_singular[-1] / candidate_singular[0]
    assert weak_to_strong_candidate < weak_to_strong_native


def test_spectral_damping_gate_selects_smallest_fully_passing_ridge() -> None:
    oracle = PairVQFunctionalGradientOracle.__new__(
        PairVQFunctionalGradientOracle
    )
    oracle.primary_late_steps = {180, 238}
    oracle.plan = {
        "spectral_damping_frontier": {
            "relative_rms_ridge": [0.03125, 0.0625, 0.125, 0.25]
        },
        "spectral_damping_gate": {
            "minimum_late_relative_error_closure_vs_native": 0.20,
            "maximum_late_relative_error_amplification": 3.0,
            "maximum_late_candidate_polar_relative_error": 0.02,
            "minimum_late_dense_native_cosine": 0.99,
            "minimum_late_matrix_dense_native_cosine": 0.98,
            "minimum_late_dense_native_norm_ratio": 0.9999,
            "maximum_late_dense_native_norm_ratio": 1.0001,
            "minimum_late_matrix_dense_task_alignment_retention": 0.98,
            "maximum_late_matrix_regression_fraction": 0.25,
            "minimum_virtual_weight_energy_recovery_weighted": 0.9999,
            "minimum_virtual_weight_energy_recovery_every_matrix": 0.999,
        },
    }

    def candidate(*, passing: bool) -> dict[str, object]:
        return {
            "relative_error_closure_vs_native": 0.25 if passing else 0.10,
            "relative_error_amplification": 2.5,
            "candidate_polar": {"relative_error": 0.018},
            "dense_native": {"cosine": 0.995},
            "minimum_matrix_dense_native_cosine": 0.985,
            "dense_native_norm_ratio": 1.0,
            "minimum_matrix_dense_task_alignment_retention": 0.99,
            "matrix_regression_fraction": 0.0,
        }

    oracle.records = [
        {
            "step": step,
            "virtual_weight": {
                "weighted_virtual_weight_energy_recovery": 0.99999,
                "worst_matrix_virtual_weight_energy_recovery": 0.9999,
            },
            "same_momentum_polar": {
                "spectral_damping": {
                    "0.03125": {"all": candidate(passing=False)},
                    "0.0625": {"all": candidate(passing=True)},
                    "0.125": {"all": candidate(passing=True)},
                    "0.25": {"all": candidate(passing=True)},
                }
            },
        }
        for step in (180, 238)
    ]
    gate = oracle._summarize_spectral_damping_gate()
    assert gate["passed"] is True
    assert gate["selected_relative_rms_ridge"] == 0.0625
    assert gate["classification"] == (
        "NORM_PRESERVING_SPECTRAL_DAMPING_CONFIRMED"
    )

    oracle.codec_stability_enabled = False
    oracle.spectral_damping_enabled = True
    combined = oracle._combined_gate()
    assert combined["selected_relative_rms_ridge"] == 0.0625


def test_codec_neighbor_isotropic_control_is_deterministic_and_energy_matched() -> None:
    oracle = PairVQFunctionalGradientOracle.__new__(
        PairVQFunctionalGradientOracle
    )
    oracle._isotropic_seed_base = 2026082300
    oracle._active_probe_step = 180
    residual = torch.linspace(-1.0, 1.0, 96).reshape(12, 8)
    first = oracle._isotropic_delta("transformer.h.3.mlp.c_fc", residual)
    second = oracle._isotropic_delta("transformer.h.3.mlp.c_fc", residual)
    assert torch.equal(first, second)
    torch.testing.assert_close(
        first.square().sum(),
        residual.square().sum(),
        rtol=1e-6,
        atol=1e-8,
    )


def test_codec_neighbor_stability_gate_requires_every_frozen_measurement() -> None:
    oracle = PairVQFunctionalGradientOracle.__new__(
        PairVQFunctionalGradientOracle
    )
    oracle.primary_late_steps = {180, 238}
    oracle.plan = {
        "frozen_gate": {
            "maximum_relative_neighbor_energy_mismatch": 1e-6,
            "minimum_late_actual_to_isotropic_logit_kl_ratio": 1.25,
            "minimum_late_actual_to_isotropic_gradient_error_ratio": 1.25,
            "minimum_late_actual_to_isotropic_postpolar_error_ratio": 1.25,
            "minimum_late_actual_worse_matrix_fraction": 0.75,
            "minimum_fit_to_heldout_excess_ratio_retention": 0.80,
            "minimum_virtual_weight_energy_recovery_weighted": 0.9999,
            "minimum_virtual_weight_energy_recovery_every_matrix": 0.999,
        }
    }

    def record(step: int, *, gradient_ratio: float = 1.5) -> dict:
        fit = {
            "ratios": {
                "logit_kl": 1.5,
                "gradient_error": gradient_ratio,
                "postpolar_error": 1.5,
            },
            "actual_worse_matrix_fraction": 0.875,
        }
        heldout = {
            "ratios": dict(fit["ratios"]),
            "actual_worse_matrix_fraction": 0.875,
        }
        return {
            "step": step,
            "virtual_weight": {
                "weighted_virtual_weight_energy_recovery": 0.99999,
                "worst_matrix_virtual_weight_energy_recovery": 0.9999,
            },
            "codec_neighbor_stability": {
                "neighbor_energy": {
                    "maximum_relative_energy_mismatch": 1e-8
                },
                "splits": {"fit": fit, "heldout": heldout},
            },
        }

    oracle.records = [record(180), record(238)]
    passed = oracle._summarize_codec_stability_gate()
    assert passed["passed"] is True
    assert passed["classification"] == "CODEC_NEIGHBOR_PATH_INSTABILITY_CONFIRMED"

    oracle.records = [record(180), record(238, gradient_ratio=1.1)]
    rejected = oracle._summarize_codec_stability_gate()
    assert rejected["passed"] is False
    assert rejected["checks"][
        "minimum_late_actual_to_isotropic_gradient_error_ratio"
    ] is False

    # The codec plan replaces the legacy functional/polar threshold schemas.
    # The terminal combined path must therefore evaluate it directly.
    oracle.codec_stability_enabled = True
    oracle.polar_amplification_enabled = True
    oracle.regularized_polar_enabled = False
    oracle.records = [record(180), record(238)]
    combined = oracle._combined_gate()
    assert combined["passed"] is True
    assert combined["classification"] == (
        "CODEC_NEIGHBOR_PATH_INSTABILITY_CONFIRMED"
    )


def test_regularized_polar_gate_selects_deepest_passing_prefix() -> None:
    oracle = PairVQFunctionalGradientOracle.__new__(
        PairVQFunctionalGradientOracle
    )
    oracle.primary_late_steps = {180, 238}
    oracle.plan = {
        "regularized_polar_frontier": {"candidate_ns_steps": [1, 2, 3, 4]},
        "regularized_polar_gate": {
            "minimum_late_relative_error_closure_vs_native": 0.20,
            "maximum_late_relative_error_amplification": 3.0,
            "maximum_late_candidate_polar_relative_error": 0.02,
            "minimum_late_dense_native_cosine": 0.99,
            "minimum_late_dense_native_norm_ratio": 0.90,
            "maximum_late_dense_native_norm_ratio": 1.05,
            "minimum_late_matrix_dense_task_alignment_retention": 0.98,
            "maximum_late_matrix_regression_fraction": 0.25,
        },
    }

    def candidate(*, passing: bool) -> dict[str, object]:
        return {
            "relative_error_closure_vs_native": 0.30 if passing else 0.10,
            "relative_error_amplification": 2.5 if passing else 3.5,
            "candidate_polar": {"relative_error": 0.018 if passing else 0.024},
            "dense_native": {"cosine": 0.995 if passing else 0.98},
            "dense_native_norm_ratio": 0.97,
            "minimum_matrix_dense_task_alignment_retention": 0.99,
            "matrix_regression_fraction": 0.0,
        }

    oracle.records = [
        {
            "step": step,
            "same_momentum_polar": {
                "regularized_polar": {
                    "1": {"all": candidate(passing=True)},
                    "2": {"all": candidate(passing=True)},
                    "3": {"all": candidate(passing=True)},
                    "4": {"all": candidate(passing=False)},
                }
            },
        }
        for step in (180, 238)
    ]
    gate = oracle._summarize_regularized_polar_gate()
    assert gate["passed"] is True
    assert gate["selected_ns_steps"] == 3
    assert gate["classification"] == "EARLY_STOPPED_MUON_NS3_STABILIZES_PAIR_VQ"
