from __future__ import annotations

import torch

from examples.nanogpt.analyze_attention_paper_activation_oracle import (
    AttentionFunctionalMetric,
    classify_target,
)


def _adjoint_error(metric: AttentionFunctionalMetric, weight: torch.Tensor) -> float:
    output = torch.randn_like(metric.apply(weight))
    left = (metric.apply(weight).double() * output.double()).sum()
    right = (weight.double() * metric.adjoint(output).double()).sum()
    return float((left - right).abs() / left.abs().clamp_min(1e-12))


def test_cproj_metric_adjoint() -> None:
    torch.manual_seed(1)
    metric = AttentionFunctionalMetric(
        target="cproj",
        cproj_inputs=torch.randn(2, 5, 8, dtype=torch.float64),
        value_sources=torch.randn(2, 2, 5, 8, dtype=torch.float64),
        output_weight=torch.randn(8, 8, dtype=torch.float64),
    )
    assert _adjoint_error(metric, torch.randn(8, 8, dtype=torch.float64)) < 1e-10


def test_v_metric_adjoint() -> None:
    torch.manual_seed(2)
    metric = AttentionFunctionalMetric(
        target="v",
        cproj_inputs=torch.randn(2, 5, 8, dtype=torch.float64),
        value_sources=torch.randn(2, 2, 5, 8, dtype=torch.float64),
        output_weight=torch.randn(8, 8, dtype=torch.float64),
    )
    assert _adjoint_error(metric, torch.randn(8, 8, dtype=torch.float64)) < 1e-10


def test_strict_classification_requires_activation_gain() -> None:
    thresholds = {
        "functional_image_recovery_minimum": 0.8,
        "activated_tangent_recovery_minimum": 0.8,
        "activation_gain_over_identity_minimum": 0.05,
    }
    classification, checks = classify_target(
        {
            "range_valid": True,
            "functional_image_recovery": 0.9,
            "activated_tangent_recovery": 0.9,
            "activation_gain_over_identity": 0.049,
        },
        thresholds,
    )
    assert classification == "ATTENTION_PAPER_ACTIVATION_ORACLE_REJECT"
    assert checks == {
        "range_valid": True,
        "image": True,
        "tangent": True,
        "activation_gain": False,
    }
