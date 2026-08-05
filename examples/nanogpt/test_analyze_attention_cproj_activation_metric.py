from __future__ import annotations

import copy

import pytest
import torch

from examples.nanogpt.analyze_attention_cproj_activation_metric import (
    TangentChart,
    aggregate,
    conjugate_gradient,
    output_metrics,
    validate_activation,
)


def test_conjugate_gradient_solves_spd_system() -> None:
    matrix = torch.tensor([[4.0, 1.0], [1.0, 3.0]])
    rhs = torch.tensor([1.0, 2.0])
    solution, diagnostics = conjugate_gradient(
        lambda value: matrix @ value,
        rhs,
        tolerance=1e-10,
        max_iterations=8,
    )
    assert torch.allclose(solution, torch.linalg.solve(matrix, rhs), atol=1e-6)
    assert diagnostics["converged"] is True


def test_tangent_chart_jvp_and_vjp_are_adjoint() -> None:
    chart = TangentChart(
        size=16,
        out_features=4,
        in_features=4,
        latent_dim=4,
        layers=2,
        seed=17,
        weight_scale=0.25,
        output_gain=torch.tensor([0.8, 1.0, 1.2, 1.4]),
    )
    coordinates = torch.randn(4)
    cotangent = torch.randn(4, 4)
    left = (chart.jvp(coordinates) * cotangent).sum()
    right = (coordinates * chart.vjp(cotangent)).sum()
    assert float((left - right).abs()) < 1e-5


def test_output_metrics_exact_target_has_unit_recovery() -> None:
    activations = torch.randn(8, 4)
    target = torch.randn(3, 4)
    gradient = torch.randn(3, 4)
    metrics = output_metrics(activations, gradient, target, target)
    assert metrics["activation_output_recovery"] == pytest.approx(1.0)
    assert metrics["activation_output_cosine"] == pytest.approx(1.0)
    assert metrics["activation_projection_identity_residual"] == pytest.approx(0.0)


def _plan() -> dict:
    return {
        "offline_diagnostic": {
            "numerical_guards": {
                "maximum_relative_normal_residual": 1e-6,
                "minimum_valid_cells": 20,
            }
        },
        "preregistered_gate": {
            "minimum_ratio10_activation_output_recovery": 0.20,
            "minimum_multiplier_over_ratio10_euclidean_projection": 1.25,
            "minimum_multiplier_over_equal_coordinate_random_control": 1.50,
            "pass_action": "pass",
            "fail_action": "fail",
        },
    }


def _rows(candidate: float = 0.30, random: float = 0.10) -> list[dict]:
    rows = []
    values = {
        "ratio01_euclidean": 0.01,
        "ratio01_activation": 0.02,
        "ratio10_euclidean": 0.20,
        "ratio10_activation": candidate,
        "ratio10_random_activation": random,
    }
    for batch in range(4):
        for layer in (0, 3, 6, 9, 11):
            for arm, recovery in values.items():
                rows.append(
                    {
                        "batch": batch,
                        "layer": layer,
                        "arm": arm,
                        "target_output_energy": 1.0,
                        "projected_output_energy": recovery,
                        "activation_output_recovery": recovery,
                        "activation_output_cosine": recovery**0.5,
                        "activation_projection_identity_residual": 0.0,
                        "task_gradient_cosine": 0.5,
                        "relative_normal_residual": 1e-8,
                    }
                )
    return rows


def test_aggregate_requires_every_registered_gate() -> None:
    passed = aggregate(_rows(), _plan())
    assert passed["passed"] is True
    assert passed["language_model_training_authorized"] is False
    failed = aggregate(_rows(candidate=0.15, random=0.80), _plan())
    assert failed["passed"] is False
    assert failed["decision"] == "REJECT_FIXED_RANDOM_CPROJ_CHART_CERTIFIED"
    assert failed["rejection_certificate"][
        "random_multiplier_gate_mathematically_impossible"
    ] is True


def test_endpoint_validation_is_fail_closed() -> None:
    plan = {
        "schema_version": "mai_124m_attention_cproj_activation_metric_fallback_plan_v1",
        "source_state": {
            "required_config_sha256": "a" * 64,
            "required_dataset_manifest_sha256": "b" * 64,
            "required_fixed_eval_indices_sha256": "c" * 64,
        },
    }
    decision = {
        "schema_version": "mai_124m_attention_cproj_ratio10_promotion_result_v1",
        "decision": {"classification": "REJECT_CPROJ_RATIO10_TRANSFER"},
    }
    result = {
        "run": {
            "config_sha256": "a" * 64,
            "dataset_manifest_sha256": "b" * 64,
            "fixed_eval_indices_sha256": "c" * 64,
            "checkpoint_sha256": "d" * 64,
            "exit_code": 0,
            "classification": "clean",
        }
    }
    validate_activation(
        plan,
        decision,
        result,
        config_sha256="a" * 64,
        dataset_manifest_sha256="b" * 64,
        checkpoint_sha256="d" * 64,
    )
    broken = copy.deepcopy(result)
    broken["run"]["checkpoint_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="checkpoint"):
        validate_activation(
            plan,
            decision,
            broken,
            config_sha256="a" * 64,
            dataset_manifest_sha256="b" * 64,
            checkpoint_sha256="d" * 64,
        )
