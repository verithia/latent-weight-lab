from __future__ import annotations

import torch

from examples.nanogpt.analyze_attention_affine_delta_path_oracle import (
    classify_summary,
    cumulative_lr_clock,
    solve_span_coefficients,
)


def test_span_solver_recovers_in_span_target() -> None:
    torch.manual_seed(7)
    basis = torch.randn(80, 5, dtype=torch.float64)
    expected = torch.randn(5, dtype=torch.float64)
    observed, rank = solve_span_coefficients(basis, basis @ expected, 1e-10)
    assert rank == 5
    assert torch.allclose(observed, expected, atol=1e-9, rtol=1e-9)


def test_lr_clock_is_monotone_and_normalized() -> None:
    steps = [0, 10, 20]
    clock = cumulative_lr_clock(
        steps,
        {
            "warmup_iters": 2,
            "lr_decay_iters": 20,
            "learning_rate": 1.0,
            "min_lr": 0.1,
        },
    )
    assert clock[0] == 0.0
    assert 0.0 < clock[10] < 1.0
    assert clock[20] == 1.0


def test_classification_requires_every_gate() -> None:
    fields = {
        "heldout_state_eval_recovery": 0.9,
        "minimum_layer_state_eval_recovery": 0.7,
        "heldout_chord_eval_recovery": 0.9,
        "minimum_layer_chord_eval_recovery": 0.7,
        "heldout_discovery_span_eval_recovery": 0.9,
        "minimum_layer_discovery_span_eval_recovery": 0.7,
        "heldout_muon_eval_recovery": 0.9,
        "minimum_layer_muon_eval_recovery": 0.59,
    }
    passed, checks = classify_summary(
        fields,
        {
            "aggregate_recovery_minimum": 0.8,
            "minimum_layer_recovery_minimum": 0.6,
        },
    )
    assert not passed
    assert not checks["exact_muon_every_layer"]
    fields["minimum_layer_muon_eval_recovery"] = 0.6
    assert classify_summary(
        fields,
        {
            "aggregate_recovery_minimum": 0.8,
            "minimum_layer_recovery_minimum": 0.6,
        },
    )[0]
