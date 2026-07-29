from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_optimizer_state_direction import (
    aggregate_rows,
    reconstruct_directions,
)


def test_reconstructs_exact_muon_state_directions() -> None:
    weight = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
    gradient = torch.tensor([[0.2, 0.3], [-0.4, 0.1]])
    buffer = torch.tensor([[0.1, -0.2], [0.3, 0.4]])
    momentum = 0.9
    combined = gradient + momentum * (momentum * buffer + gradient)
    polar = torch.tensor([[0.7, -0.1], [0.2, 0.6]])
    applied = -0.1 * weight - 1.5 * polar
    directions = reconstruct_directions(
        {
            "weight_before_step": weight,
            "gradient_after_clip": gradient,
            "momentum_buffer_before_step": buffer,
            "combined_momentum_update": combined,
            "polar_update": polar,
            "applied_direction_per_lr": applied,
        },
        {
            "momentum": momentum,
            "weight_decay": 0.1,
            "polar_scale": 1.5,
        },
    )
    torch.testing.assert_close(
        directions["raw_gradient_descent"], -gradient
    )
    torch.testing.assert_close(
        directions["momentum_buffer_descent"], -buffer
    )
    torch.testing.assert_close(
        directions["combined_momentum_descent"], -combined
    )
    torch.testing.assert_close(
        directions["muon_polar_descent"], -polar
    )
    torch.testing.assert_close(
        directions["exact_applied_direction"], applied
    )


def test_aggregate_promotes_first_registered_direction() -> None:
    rows = []
    for cell in range(2):
        for direction, recovery in (
            ("exact_applied_direction", 0.12),
            ("exact_applied_right_tangent", 0.20),
            ("muon_polar_descent", 0.05),
            ("muon_polar_right_tangent", 0.04),
            ("combined_momentum_descent", 0.03),
            ("raw_gradient_descent", 0.02),
        ):
            rows.append(
                {
                    "direction": direction,
                    "target_chord_fro": 1.0 + cell,
                    "cosine": recovery**0.5,
                    "positive_step_line_recovery": recovery,
                }
            )
    aggregate = aggregate_rows(rows, expected_cells=2)
    assert aggregate["promoted_direction"] == "exact_applied_direction"
    assert aggregate["decision"].endswith("EXACT_APPLIED_DIRECTION")
