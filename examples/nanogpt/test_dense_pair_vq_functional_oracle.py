from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from examples.nanogpt.dense_pair_vq_functional_oracle import (
    _aggregate_directions,
    _direction_metrics,
    antithetic_average,
    gradient_comparison,
    gradient_cross_cosine,
)


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
